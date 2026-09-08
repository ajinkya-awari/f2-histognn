import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from data.gdc import (
    GDCManifestError,
    build_gdc_query_payload,
    extract_gdc_response_hits,
    manifest_summary,
    parse_gdc_hits,
    select_case_disjoint_pilot,
)
from data.evidence import EvidenceError, build_stage_evidence, write_evidence
from data.hovernet import HoverNetOutputError, load_hovernet_instances
from data.tiles import TileSelectionError, select_tissue_tile_origins
from scripts.kaggle_real_data_pilot import _dependency_lock_hash


def gdc_hit(*, project="TCGA-LUAD", case="TCGA-AA-0001", file_id="file-1"):
    return {
        "file_id": file_id,
        "file_name": f"{case}-01Z-00-DX1.svs",
        "data_type": "Slide Image",
        "access": "open",
        "md5sum": "a" * 32,
        "file_size": 1234,
        "cases": [
            {
                "case_id": f"uuid-{case}",
                "submitter_id": case,
                "project": {"project_id": project},
                "samples": [
                    {
                        "portions": [
                            {"slides": [{"submitter_id": f"{case}-01Z-00-DX1"}]}
                        ]
                    }
                ],
            }
        ],
    }


def test_gdc_parser_accepts_only_open_luad_lusc_slide_records():
    records = parse_gdc_hits(
        [
            gdc_hit(),
            gdc_hit(project="TCGA-LUSC", case="TCGA-BB-0002", file_id="file-2"),
        ]
    )

    assert [record.label for record in records] == ["LUAD", "LUSC"]
    assert records[0].case_submitter_id == "TCGA-AA-0001"
    assert records[0].slide_submitter_id == "TCGA-AA-0001-01Z-00-DX1"


def test_gdc_parser_resolves_the_slide_barcode_from_a_compound_file_name():
    hit = gdc_hit()
    hit["file_name"] = "TCGA-AA-0001-01Z-00-DX1.00000000-0000-0000-0000-000000000000.svs"
    hit["cases"][0]["samples"][0]["portions"][0]["slides"].append(
        {"submitter_id": "TCGA-AA-0001-01Z-00-DX2"}
    )

    record = parse_gdc_hits([hit])[0]

    assert record.slide_submitter_id == "TCGA-AA-0001-01Z-00-DX1"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda hit: hit.update(access="controlled"), "open access"),
        (lambda hit: hit.update(data_type="Pathology Report"), "Slide Image"),
        (lambda hit: hit["cases"][0]["project"].update(project_id="TCGA-COAD"), "project"),
        (lambda hit: hit.update(md5sum="bad"), "md5"),
        (lambda hit: hit.update(file_size=0), "file_size"),
        (lambda hit: hit.update(cases=[]), "exactly one case"),
        (lambda hit: hit["cases"][0].update(submitter_id=""), "case"),
    ],
)
def test_gdc_parser_rejects_ineligible_or_incomplete_records(mutation, message):
    hit = gdc_hit()
    mutation(hit)

    with pytest.raises(GDCManifestError, match=message):
        parse_gdc_hits([hit])


def test_gdc_parser_rejects_duplicate_file_ids_and_conflicting_case_labels():
    duplicate = gdc_hit(file_id="same")
    with pytest.raises(GDCManifestError, match="duplicate file UUID"):
        parse_gdc_hits([duplicate, copy.deepcopy(duplicate)])

    with pytest.raises(GDCManifestError, match="conflicting project labels"):
        parse_gdc_hits(
            [
                gdc_hit(case="TCGA-AA-0001", project="TCGA-LUAD", file_id="one"),
                gdc_hit(case="TCGA-AA-0001", project="TCGA-LUSC", file_id="two"),
            ]
        )


def test_pilot_selection_is_seeded_bounded_balanced_and_one_slide_per_case():
    hits = []
    for label in ("LUAD", "LUSC"):
        for index in range(5):
            case = f"TCGA-{label[-2:]}-{index:04d}"
            hits.append(gdc_hit(project=f"TCGA-{label}", case=case, file_id=f"{label}-{index}-b"))
            hits.append(gdc_hit(project=f"TCGA-{label}", case=case, file_id=f"{label}-{index}-a"))
    records = parse_gdc_hits(hits)

    first = select_case_disjoint_pilot(records, per_class=3, seed=17)
    second = select_case_disjoint_pilot(tuple(reversed(records)), per_class=3, seed=17)

    assert first == second
    assert len(first) == 6
    assert {label: sum(record.label == label for record in first) for label in ("LUAD", "LUSC")} == {
        "LUAD": 3,
        "LUSC": 3,
    }
    assert len({record.case_submitter_id for record in first}) == len(first)


def test_pilot_selection_rejects_invalid_limit_and_insufficient_class():
    records = parse_gdc_hits([gdc_hit()])

    with pytest.raises(GDCManifestError, match="positive integer"):
        select_case_disjoint_pilot(records, per_class=0)
    with pytest.raises(GDCManifestError, match="LUSC"):
        select_case_disjoint_pilot(records, per_class=1)


def test_manifest_summary_exposes_counts_and_hash_but_no_case_identifiers():
    records = parse_gdc_hits(
        [
            gdc_hit(),
            gdc_hit(project="TCGA-LUSC", case="TCGA-BB-0002", file_id="file-2"),
        ]
    )

    summary = manifest_summary(records)

    assert summary["record_count"] == 2
    assert summary["case_count"] == 2
    assert summary["class_counts"] == {"LUAD": 1, "LUSC": 1}
    assert len(summary["manifest_sha256"]) == 64
    assert "TCGA-AA-0001" not in repr(summary)


def write_hovernet(tmp_path, instances):
    path = tmp_path / "hovernet.json"
    path.write_text(json.dumps(instances), encoding="utf-8")
    return path


def test_hovernet_adapter_drops_background_and_preserves_five_probabilities(tmp_path):
    path = write_hovernet(
        tmp_path,
        {
            "nucleus-1": {
                "centroid": [10.0, 20.0],
                "type": 1,
                "probs": [0.05, 0.60, 0.10, 0.10, 0.05, 0.10],
            },
            "nucleus-2": {
                "centroid": [30.0, 40.0],
                "type": 5,
                "probs": [0.10, 0.10, 0.10, 0.10, 0.10, 0.50],
            },
        },
    )

    record = load_hovernet_instances(path)

    assert record.identifiers == ("nucleus-1", "nucleus-2")
    assert record.features.shape == (2, 7)
    assert record.type_prob.tolist() == [
        [0.60, 0.10, 0.10, 0.05, 0.10],
        [0.10, 0.10, 0.10, 0.10, 0.50],
    ]


def test_hovernet_adapter_accepts_the_native_nuc_wrapper(tmp_path):
    path = write_hovernet(
        tmp_path,
        {
            "mag": None,
            "nuc": {
                "1": {
                    "centroid": [10.0, 20.0],
                    "type": 1,
                    "probs": [0.0, 0.6, 0.1, 0.1, 0.1, 0.1],
                }
            },
        },
    )

    record = load_hovernet_instances(path)

    assert record.identifiers == ("1",)
    assert record.features.shape == (1, 7)


@pytest.mark.parametrize(
    ("instance", "message"),
    [
        ({"centroid": [1, 2], "type": 0, "probs": [1, 0, 0, 0, 0, 0]}, "background"),
        ({"centroid": [1, 2], "type": 1}, "probs"),
        ({"centroid": [1, 2], "type": 1, "probs": [0, 1, 0, 0, 0]}, "six"),
        ({"centroid": [1, 2], "type": 1, "probs": [0, 0.5, 0, 0, 0, 0]}, "sum"),
        ({"centroid": [1], "type": 1, "probs": [0, 1, 0, 0, 0, 0]}, "centroid"),
        ({"centroid": [1, 2], "type": 6, "probs": [0, 0, 0, 0, 0, 1]}, "type"),
    ],
)
def test_hovernet_adapter_rejects_incompatible_instances(tmp_path, instance, message):
    path = write_hovernet(tmp_path, {"nucleus-1": instance})

    with pytest.raises(HoverNetOutputError, match=message):
        load_hovernet_instances(path)


def test_hovernet_adapter_rejects_invalid_json_empty_data_and_unbounded_read(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    with pytest.raises(HoverNetOutputError, match="JSON"):
        load_hovernet_instances(malformed)

    empty = write_hovernet(tmp_path, {})
    with pytest.raises(HoverNetOutputError, match="non-empty"):
        load_hovernet_instances(empty)

    two = write_hovernet(
        tmp_path,
        {
            "one": {"centroid": [1, 2], "type": 1, "probs": [0, 1, 0, 0, 0, 0]},
            "two": {"centroid": [2, 3], "type": 2, "probs": [0, 0, 1, 0, 0, 0]},
        },
    )
    with pytest.raises(HoverNetOutputError, match="max_nuclei"):
        load_hovernet_instances(two, max_nuclei=1)


def test_gdc_query_payload_is_open_slide_only_and_requests_grouping_fields():
    payload = build_gdc_query_payload(page_size=123)

    assert payload["size"] == 123
    serialized = json.dumps(payload, sort_keys=True)
    for required in (
        "TCGA-LUAD",
        "TCGA-LUSC",
        "Slide Image",
        "open",
        "cases.submitter_id",
        "cases.samples.portions.slides.submitter_id",
    ):
        assert required in serialized

    with pytest.raises(GDCManifestError, match="positive integer"):
        build_gdc_query_payload(page_size=0)


def test_gdc_response_parser_rejects_truncation_and_malformed_payloads():
    hit = gdc_hit()
    assert extract_gdc_response_hits(
        {"data": {"hits": [hit], "pagination": {"total": 1, "count": 1}}}
    ) == (hit,)

    with pytest.raises(GDCManifestError, match="truncated"):
        extract_gdc_response_hits(
            {"data": {"hits": [hit], "pagination": {"total": 2, "count": 1}}}
        )
    with pytest.raises(GDCManifestError, match="response"):
        extract_gdc_response_hits({"message": "error"})


def test_gdc_manifest_script_supports_direct_cli_invocation():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/query_gdc_manifest.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--private-manifest" in result.stdout


def complete_provenance():
    return {
        "dataset_release": "GDC API queried 2026-09-08T00:00:00Z",
        "manifest_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
        "code_revision": "c" * 40,
        "dependency_lock_sha256": "d" * 64,
        "seed": 17,
        "device": {"type": "cuda", "name": "synthetic GPU"},
        "artifact_count": 4,
        "class_counts": {"LUAD": 2, "LUSC": 2},
        "artifact_hashes": {"graphs": "e" * 64},
        "positive_class": "LUSC",
        "uncertainty_method": "three prespecified seeds",
    }


def test_metric_evidence_requires_complete_provenance_and_accepted_stage():
    metrics = {"accuracy": 0.5, "macro_f1": 0.4, "auroc": 0.6, "auprc": 0.55}

    with pytest.raises(EvidenceError, match="provenance"):
        build_stage_evidence(
            stage="pilot_evaluation", status="passed", provenance={}, metrics=metrics
        )
    with pytest.raises(EvidenceError, match="pilot_evaluation or benchmark"):
        build_stage_evidence(
            stage="metadata_preflight",
            status="passed",
            provenance=complete_provenance(),
            metrics=metrics,
        )

    evidence = build_stage_evidence(
        stage="pilot_evaluation",
        status="passed",
        provenance=complete_provenance(),
        metrics=metrics,
    )
    assert evidence["metrics"] == metrics


def test_evidence_rejects_identifiers_and_is_append_only(tmp_path):
    with pytest.raises(EvidenceError, match="identifier"):
        build_stage_evidence(
            stage="metadata_preflight",
            status="passed",
            provenance={"case_id": "TCGA-AA-0001"},
        )

    evidence = build_stage_evidence(
        stage="metadata_preflight",
        status="passed",
        provenance={"manifest_sha256": "a" * 64, "record_count": 2},
    )
    path = tmp_path / "evidence.json"
    write_evidence(path, evidence)
    assert json.loads(path.read_text(encoding="utf-8")) == evidence
    with pytest.raises(EvidenceError, match="already exists"):
        write_evidence(path, evidence)


def test_tissue_tile_selection_is_deterministic_bounded_and_in_slide_coordinates():
    thumbnail = np.full((4, 6, 3), 255, dtype=np.uint8)
    thumbnail[1:3, 1:5] = [120, 40, 80]

    first = select_tissue_tile_origins(
        thumbnail,
        slide_size=(6000, 4000),
        tile_size=256,
        count=3,
        seed=17,
    )
    second = select_tissue_tile_origins(
        thumbnail,
        slide_size=(6000, 4000),
        tile_size=256,
        count=3,
        seed=17,
    )

    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert all(0 <= x <= 6000 - 256 and 0 <= y <= 4000 - 256 for x, y in first)


def test_tissue_tile_selection_rejects_blank_thumbnail_and_excess_count():
    blank = np.full((4, 4, 3), 255, dtype=np.uint8)

    with pytest.raises(TileSelectionError, match="tissue"):
        select_tissue_tile_origins(blank, slide_size=(4000, 4000), tile_size=256, count=1)

    tissue = np.full((2, 2, 3), [120, 40, 80], dtype=np.uint8)
    with pytest.raises(TileSelectionError, match="eligible"):
        select_tissue_tile_origins(tissue, slide_size=(512, 512), tile_size=256, count=5)


def test_real_data_dependency_hash_covers_both_lock_files(tmp_path):
    (tmp_path / "requirements.txt").write_bytes(b"a\n")
    (tmp_path / "requirements-kaggle-real-data.txt").write_bytes(b"b\n")

    assert _dependency_lock_hash(tmp_path) == (
        "9e0cc975f5ed68a1126c908b6144b3b06ad87479a7063acc880a44930cf9a9c7"
    )

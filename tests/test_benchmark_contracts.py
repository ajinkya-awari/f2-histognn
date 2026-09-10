from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from data.benchmark import (
    BenchmarkContractError,
    sanitized_split_summary,
    select_balanced_cases,
    stratified_case_split,
)
from data.gdc import GDCManifestError, GDCSlideRecord
from data.benchmark_graphs import (
    BenchmarkGraphError,
    PrivateGraph,
    load_private_graphs,
    stream_slide_tiles,
)
from data.hovernet import HoverNetOutputError
from training.evaluation import (
    BenchmarkEvaluationError,
    aggregate_case_probabilities,
    binary_case_metrics,
    stratified_bootstrap_intervals,
)
from training.benchmark import (
    FROZEN_POLICY,
    FrozenBenchmarkPolicy,
    frozen_run_matrix,
    partition_private_graphs,
    run_frozen_benchmark,
)
from scripts.kaggle_case_disjoint_benchmark import (
    CASES_PER_CLASS,
    MAX_INDIVIDUAL_SLIDE_BYTES,
    MAX_TOTAL_SLIDE_BYTES,
    TEST_PER_CLASS,
    TRAIN_PER_CLASS,
    VALIDATION_PER_CLASS,
    _graph_valid_records,
    _split_counts_for_graph_valid_cohort,
    _query_eligible_records,
)


def benchmark_records(per_class: int = 60) -> tuple[GDCSlideRecord, ...]:
    records = []
    for label in ("LUAD", "LUSC"):
        project = f"TCGA-{label}"
        for index in range(per_class):
            case = f"TCGA-{label[-2:]}-{index:04d}"
            for slide in ("A", "B"):
                records.append(
                    GDCSlideRecord(
                        file_id=f"{label}-{index}-{slide}",
                        file_name=f"{case}-01Z-00-DX{slide}.svs",
                        file_size=1000 + index,
                        md5sum=f"{index:032x}",
                        project_id=project,
                        label=label,
                        case_id=f"uuid-{label}-{index}",
                        case_submitter_id=case,
                        slide_submitter_id=f"{case}-01Z-00-DX{slide}",
                    )
                )
    return tuple(records)


def test_balanced_case_selection_is_order_independent_and_one_slide_per_case():
    records = benchmark_records()

    first = select_balanced_cases(records, per_class=50, seed=17)
    second = select_balanced_cases(tuple(reversed(records)), per_class=50, seed=17)

    assert first == second
    assert len(first) == 100
    assert len({record.case_id for record in first}) == 100
    assert {label: sum(record.label == label for record in first) for label in ("LUAD", "LUSC")} == {
        "LUAD": 50,
        "LUSC": 50,
    }


def test_stratified_case_split_has_exact_counts_no_overlap_and_stable_hash():
    selected = select_balanced_cases(benchmark_records(), per_class=50, seed=17)

    first = stratified_case_split(
        selected,
        seed=17,
        train_per_class=30,
        validation_per_class=10,
        test_per_class=10,
    )
    second = stratified_case_split(
        tuple(reversed(selected)),
        seed=17,
        train_per_class=30,
        validation_per_class=10,
        test_per_class=10,
    )

    assert first == second
    assert [len(first.train), len(first.validation), len(first.test)] == [60, 20, 20]
    partitions = [
        {record.case_id for record in first.train},
        {record.case_id for record in first.validation},
        {record.case_id for record in first.test},
    ]
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert len(first.split_sha256) == 64

    summary = sanitized_split_summary(first)
    assert summary == {
        "split_sha256": first.split_sha256,
        "case_counts": {"train": 60, "validation": 20, "test": 20},
        "class_counts": {
            "train": {"LUAD": 30, "LUSC": 30},
            "validation": {"LUAD": 10, "LUSC": 10},
            "test": {"LUAD": 10, "LUSC": 10},
        },
    }
    assert "TCGA-" not in repr(summary)


def test_benchmark_selection_and_split_fail_closed_on_invalid_records():
    records = benchmark_records(per_class=2)
    with pytest.raises(BenchmarkContractError, match="positive integer"):
        select_balanced_cases(records, per_class=0, seed=17)
    with pytest.raises(BenchmarkContractError, match="insufficient"):
        select_balanced_cases(records, per_class=3, seed=17)

    duplicate_file = replace(records[1], file_id=records[0].file_id)
    with pytest.raises(BenchmarkContractError, match="duplicate file"):
        select_balanced_cases((records[0], duplicate_file), per_class=1, seed=17)

    invalid_label = replace(records[0], label="OTHER")
    with pytest.raises(BenchmarkContractError, match="label"):
        select_balanced_cases((invalid_label, records[-1]), per_class=1, seed=17)

    selected = select_balanced_cases(records, per_class=2, seed=17)
    with pytest.raises(BenchmarkContractError, match="requested split"):
        stratified_case_split(
            selected,
            seed=17,
            train_per_class=1,
            validation_per_class=1,
            test_per_class=1,
        )


def test_case_aggregation_averages_exactly_four_tiles_and_preserves_labels():
    probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
         [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]],
        dtype=np.float64,
    )
    result = aggregate_case_probabilities(
        np.log(probabilities),
        case_keys=["case-a"] * 4 + ["case-b"] * 4,
        labels=[0] * 4 + [1] * 4,
        tiles_per_case=4,
    )

    assert result.case_keys == ("case-a", "case-b")
    assert result.labels.tolist() == [0, 1]
    np.testing.assert_allclose(result.probabilities, [[0.75, 0.25], [0.25, 0.75]])


def test_case_metrics_have_frozen_binary_definitions_and_lusc_positive_class():
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    metrics = binary_case_metrics(probabilities, np.array([0, 0, 1, 1]))

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["auprc"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics["positive_class"] == "LUSC"


def test_case_metrics_are_invariant_to_input_order_when_scores_tie():
    probabilities = np.array([[0.2, 0.8], [0.5, 0.5], [0.5, 0.5], [0.8, 0.2]])
    labels = np.array([1, 1, 0, 0])

    first = binary_case_metrics(probabilities, labels)
    order = np.array([0, 2, 1, 3])
    second = binary_case_metrics(probabilities[order], labels[order])

    assert first == second


def test_stratified_bootstrap_is_deterministic_and_bounded():
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    labels = np.array([0, 0, 1, 1])

    first = stratified_bootstrap_intervals(probabilities, labels, resamples=200, seed=17)
    second = stratified_bootstrap_intervals(probabilities, labels, resamples=200, seed=17)

    assert first == second
    assert set(first) == {"accuracy", "macro_f1", "auroc", "auprc"}
    assert all(0.0 <= interval[0] <= interval[1] <= 1.0 for interval in first.values())


@pytest.mark.parametrize(
    ("logits", "case_keys", "labels", "message"),
    [
        (np.zeros((3, 2)), ["a"] * 3, [0] * 3, "exactly 4"),
        (np.zeros((4, 2)), ["a"] * 4, [0, 0, 0, 1], "conflicting"),
        (np.array([[np.nan, 0.0]] * 4), ["a"] * 4, [0] * 4, "finite"),
    ],
)
def test_case_aggregation_rejects_invalid_tile_evidence(logits, case_keys, labels, message):
    with pytest.raises(BenchmarkEvaluationError, match=message):
        aggregate_case_probabilities(
            logits, case_keys=case_keys, labels=labels, tiles_per_case=4
        )


def test_case_metrics_reject_single_class_and_invalid_probability_rows():
    with pytest.raises(BenchmarkEvaluationError, match="both classes"):
        binary_case_metrics(np.array([[0.8, 0.2], [0.7, 0.3]]), np.array([0, 0]))
    with pytest.raises(BenchmarkEvaluationError, match="sum to 1"):
        binary_case_metrics(np.array([[0.8, 0.3], [0.2, 0.8]]), np.array([0, 1]))


def test_stream_slide_tiles_removes_each_slide_and_maps_exactly_four_tiles(tmp_path):
    payload = b"data"
    records = tuple(
        replace(record, file_size=len(payload), md5sum=hashlib.md5(payload).hexdigest())
        for record in select_balanced_cases(benchmark_records(2), per_class=1, seed=17)
    )

    def fetch(record, destination):
        destination.write_bytes(payload)

    def extract(record, slide_path, tile_dir):
        assert slide_path.is_file()
        paths = []
        for index in range(4):
            path = tile_dir / f"{record.file_id}-{index}.png"
            path.write_bytes(b"tile")
            paths.append(path)
        return tuple(paths)

    result = stream_slide_tiles(records, tmp_path / "private", fetch, extract)

    assert result.total_slide_bytes == 8
    assert len(result.tile_to_case) == 8
    assert not list((tmp_path / "private").rglob("*.svs"))
    assert {value[1] for value in result.tile_to_case.values()} == {0, 1}


def test_stream_slide_tiles_rejects_paths_outside_private_tile_root(tmp_path):
    payload = b"data"
    record = replace(
        benchmark_records(1)[0],
        file_size=len(payload),
        md5sum=hashlib.md5(payload).hexdigest(),
    )
    outside = tuple(tmp_path / f"outside-{index}.png" for index in range(4))

    def fetch(_record, destination):
        destination.write_bytes(payload)

    def extract(_record, _slide_path, _tile_dir):
        for path in outside:
            path.write_bytes(b"tile")
        return outside

    with pytest.raises(BenchmarkGraphError, match="private tile root"):
        stream_slide_tiles((record,), tmp_path / "private", fetch, extract)
    assert all(path.is_file() for path in outside)
    assert not list((tmp_path / "private").rglob("*.svs"))


def test_private_graph_loading_is_deterministic_and_enforces_four_graphs_per_case(tmp_path):
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    mapping = {}
    for case_index, label in enumerate((0, 1)):
        for tile_index in range(4):
            stem = f"case-{case_index}-tile-{tile_index}"
            mapping[stem] = (f"private-{case_index}", label)
            nuclei = {
                str(index): {
                    "centroid": [float(index), float(index % 2)],
                    "type": 1,
                    "probs": [0.0, 0.6, 0.1, 0.1, 0.1, 0.1],
                }
                for index in range(3)
            }
            (json_dir / f"{stem}.json").write_text(json.dumps({"nuc": nuclei}), encoding="utf-8")

    first = load_private_graphs(json_dir, mapping, tiles_per_case=4, max_nodes=512)
    second = load_private_graphs(json_dir, dict(reversed(tuple(mapping.items()))), tiles_per_case=4, max_nodes=512)

    assert first.artifact_sha256 == second.artifact_sha256
    assert len(first.graphs) == 8
    assert {graph.label for graph in first.graphs} == {0, 1}
    assert all(graph.features.shape == (3, 7) for graph in first.graphs)

    missing_key = list(mapping)[-1]
    (json_dir / f"{missing_key}.json").unlink()
    with pytest.raises(BenchmarkGraphError, match="exactly 4"):
        load_private_graphs(
            json_dir,
            {key: value for key, value in mapping.items() if key != missing_key},
        )


def test_private_graph_loading_rejects_empty_candidates_but_keeps_four_valid_graphs(tmp_path):
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    mapping = {}
    case_key = "private-case"
    valid_stems = []
    for tile_index in range(6):
        stem = f"case-0-tile-{tile_index}"
        mapping[stem] = (case_key, 0)
        if tile_index in (1, 4):
            (json_dir / f"{stem}.json").write_text(json.dumps({"nuc": {}}), encoding="utf-8")
            continue
        valid_stems.append(stem)
        nuclei = {
            str(index): {
                "centroid": [float(index), float(tile_index)],
                "type": 1,
                "probs": [0.0, 0.6, 0.1, 0.1, 0.1, 0.1],
            }
            for index in range(3)
        }
        (json_dir / f"{stem}.json").write_text(json.dumps({"nuc": nuclei}), encoding="utf-8")

    graph_set = load_private_graphs(json_dir, mapping, tiles_per_case=4, max_nodes=512)

    assert [graph.tile_key for graph in graph_set.graphs] == valid_stems
    assert graph_set.rejected_tile_counts == {"empty_nuclei": 2}


def test_private_graph_loading_rejects_missing_candidate_json_but_keeps_four_valid_graphs(tmp_path):
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    mapping = {}
    valid_stems = []
    for tile_index in range(6):
        stem = f"case-0-tile-{tile_index}"
        mapping[stem] = ("private-case", 0)
        if tile_index in (1, 4):
            continue
        valid_stems.append(stem)
        nuclei = {
            str(index): {
                "centroid": [float(index), float(tile_index)],
                "type": 1,
                "probs": [0.0, 0.6, 0.1, 0.1, 0.1, 0.1],
            }
            for index in range(3)
        }
        (json_dir / f"{stem}.json").write_text(json.dumps({"nuc": nuclei}), encoding="utf-8")

    graph_set = load_private_graphs(json_dir, mapping, tiles_per_case=4, max_nodes=512)

    assert [graph.tile_key for graph in graph_set.graphs] == valid_stems
    assert graph_set.rejected_tile_counts == {"missing_hovernet_json": 2}


def test_private_graph_loading_fails_when_empty_candidates_leave_too_few_graphs(tmp_path):
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    mapping = {}
    case_key = "private-case"
    for tile_index in range(4):
        stem = f"case-0-tile-{tile_index}"
        mapping[stem] = (case_key, 0)
        payload = {"nuc": {}} if tile_index else {
            "nuc": {
                str(index): {
                    "centroid": [float(index), 0.0],
                    "type": 1,
                    "probs": [0.0, 0.6, 0.1, 0.1, 0.1, 0.1],
                }
                for index in range(3)
            }
        }
        (json_dir / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkGraphError, match="exactly 4 valid graphs"):
        load_private_graphs(json_dir, mapping, tiles_per_case=4, max_nodes=512)


def test_private_graph_loading_can_exclude_graph_qc_ineligible_cases_without_identifiers(tmp_path):
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    mapping = {}
    for case_index, label in enumerate((0, 1)):
        for tile_index in range(6):
            stem = f"case-{case_index}-tile-{tile_index}"
            mapping[stem] = (f"private-{case_index}", label)
            if case_index == 1 and tile_index > 1:
                (json_dir / f"{stem}.json").write_text(json.dumps({"nuc": {}}), encoding="utf-8")
                continue
            nuclei = {
                str(index): {
                    "centroid": [float(index), float(tile_index)],
                    "type": 1,
                    "probs": [0.0, 0.6, 0.1, 0.1, 0.1, 0.1],
                }
                for index in range(3)
            }
            (json_dir / f"{stem}.json").write_text(json.dumps({"nuc": nuclei}), encoding="utf-8")

    graph_set = load_private_graphs(
        json_dir,
        mapping,
        tiles_per_case=4,
        max_nodes=512,
        allow_incomplete_cases=True,
    )

    assert len(graph_set.graphs) == 4
    assert {graph.case_key for graph in graph_set.graphs} == {"private-0"}
    assert graph_set.complete_case_keys == ("private-0",)
    assert graph_set.incomplete_case_counts_by_label == {"LUSC": 1}
    assert graph_set.rejected_tile_counts == {"empty_nuclei": 4}
    assert "private-1" not in repr(graph_set.incomplete_case_counts_by_label)


def test_private_graph_hash_ignores_excluded_incomplete_cases(tmp_path):
    def write_case(root, mapping, case_index, label, valid_tiles):
        for tile_index in range(6):
            stem = f"case-{case_index}-tile-{tile_index}"
            mapping[stem] = (f"private-{case_index}", label)
            if tile_index >= valid_tiles:
                (root / f"{stem}.json").write_text(json.dumps({"nuc": {}}), encoding="utf-8")
                continue
            nuclei = {
                str(index): {
                    "centroid": [float(index), float(tile_index)],
                    "type": 1,
                    "probs": [0.0, 0.6, 0.1, 0.1, 0.1, 0.1],
                }
                for index in range(3)
            }
            (root / f"{stem}.json").write_text(json.dumps({"nuc": nuclei}), encoding="utf-8")

    valid_dir = tmp_path / "valid"
    mixed_dir = tmp_path / "mixed"
    valid_dir.mkdir()
    mixed_dir.mkdir()
    valid_mapping = {}
    mixed_mapping = {}
    write_case(valid_dir, valid_mapping, 0, 0, valid_tiles=6)
    write_case(mixed_dir, mixed_mapping, 0, 0, valid_tiles=6)
    write_case(mixed_dir, mixed_mapping, 1, 1, valid_tiles=2)

    valid = load_private_graphs(
        valid_dir,
        valid_mapping,
        tiles_per_case=4,
        max_nodes=512,
        allow_incomplete_cases=True,
    )
    mixed = load_private_graphs(
        mixed_dir,
        mixed_mapping,
        tiles_per_case=4,
        max_nodes=512,
        allow_incomplete_cases=True,
    )

    assert mixed.complete_case_keys == valid.complete_case_keys
    assert mixed.artifact_sha256 == valid.artifact_sha256


def test_private_graph_loading_still_fails_on_malformed_hovernet_json(tmp_path):
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    mapping = {}
    for tile_index in range(4):
        stem = f"case-0-tile-{tile_index}"
        mapping[stem] = ("private-case", 0)
        payload = {"bad": "shape"} if tile_index == 0 else {
            "nuc": {
                str(index): {
                    "centroid": [float(index), 0.0],
                    "type": 1,
                    "probs": [0.0, 0.6, 0.1, 0.1, 0.1, 0.1],
                }
                for index in range(3)
            }
        }
        (json_dir / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HoverNetOutputError, match="instance identifiers|must be a mapping"):
        load_private_graphs(json_dir, mapping, tiles_per_case=4, max_nodes=512)


def test_frozen_benchmark_policy_defines_exactly_twelve_runs():
    assert FROZEN_POLICY.hidden_dim == 32
    assert FROZEN_POLICY.max_epochs == 100
    assert FROZEN_POLICY.patience == 10
    assert FROZEN_POLICY.learning_rate == 0.001
    assert FROZEN_POLICY.weight_decay == 0.0001
    assert FROZEN_POLICY.seeds == (17, 29, 43)
    assert frozen_run_matrix() == tuple(
        (model, seed)
        for model in ("gcn", "graphsage", "gat", "graphgps")
        for seed in (17, 29, 43)
    )


def test_private_graph_partition_follows_case_split_and_requires_four_graphs():
    selected = select_balanced_cases(benchmark_records(3), per_class=3, seed=17)
    split = stratified_case_split(
        selected,
        seed=17,
        train_per_class=1,
        validation_per_class=1,
        test_per_class=1,
    )
    graphs = []
    for record in selected:
        case_key = hashlib.sha256(record.case_id.encode()).hexdigest()
        label = 0 if record.label == "LUAD" else 1
        for tile in range(4):
            graphs.append(
                PrivateGraph(
                    tile_key=f"{case_key}-{tile}",
                    case_key=case_key,
                    label=label,
                    features=np.ones((2, 7)),
                    edge_index=np.array([[0, 1], [1, 0]]),
                    edge_attr=np.ones((2, 1)),
                )
            )

    partitions = partition_private_graphs(tuple(graphs), split, tiles_per_case=4)

    assert {name: len(values) for name, values in partitions.items()} == {
        "train": 8,
        "validation": 8,
        "test": 8,
    }
    with pytest.raises(ValueError, match="exactly 4"):
        partition_private_graphs(tuple(graphs[:-1]), split, tiles_per_case=4)


def test_frozen_runner_trains_then_emits_case_level_results_only_after_success(monkeypatch):
    selected = select_balanced_cases(benchmark_records(3), per_class=3, seed=17)
    split = stratified_case_split(
        selected,
        seed=17,
        train_per_class=1,
        validation_per_class=1,
        test_per_class=1,
    )
    graphs = []
    for record in selected:
        case_key = hashlib.sha256(record.case_id.encode()).hexdigest()
        label = 0 if record.label == "LUAD" else 1
        for tile in range(4):
            graphs.append(
                PrivateGraph(
                    tile_key=f"{case_key}-{tile}",
                    case_key=case_key,
                    label=label,
                    features=np.array([[0, 0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]], dtype=float),
                    edge_index=np.array([[0, 1], [1, 0]]),
                    edge_attr=np.ones((2, 1)),
                )
            )
    partitions = partition_private_graphs(graphs, split)
    policy = FrozenBenchmarkPolicy(
        models=("gcn",),
        seeds=(17,),
        hidden_dim=4,
        max_epochs=1,
        patience=1,
        bootstrap_resamples=20,
    )

    result = run_frozen_benchmark(
        partitions,
        split_sha256=split.split_sha256,
        graph_sha256="a" * 64,
        provenance={"scope": "synthetic unit test"},
        device="cpu",
        policy=policy,
    )

    assert result["run_count"] == 1
    assert result["models"]["gcn"]["seeds"][0]["seed"] == 17
    assert result["models"]["gcn"]["seeds"][0]["metrics"]["positive_class"] == "LUSC"
    assert "case_key" not in repr(result)


def test_kaggle_benchmark_entrypoint_freezes_approved_cohort_and_safety_caps():
    assert CASES_PER_CLASS == 50
    assert (TRAIN_PER_CLASS, VALIDATION_PER_CLASS, TEST_PER_CLASS) == (30, 10, 10)
    assert MAX_INDIVIDUAL_SLIDE_BYTES == 3 * 1024**3
    assert MAX_TOTAL_SLIDE_BYTES == 70 * 1024**3


def test_benchmark_gdc_query_passes_integer_page_size_to_cli_query_adapter():
    seen = []

    def query(page_size):
        seen.append(page_size)
        return {
            "data": {
                "hits": [],
                "pagination": {"count": 0, "total": 0},
            }
        }

    with pytest.raises(GDCManifestError, match="non-empty"):
        _query_eligible_records(query)

    assert seen == [10000]


def test_benchmark_resplit_uses_only_graph_qc_valid_cases_before_training():
    records = select_balanced_cases(benchmark_records(6), per_class=6, seed=17)
    complete_keys = {
        hashlib.sha256(record.case_id.encode()).hexdigest()
        for record in records
        if not (record.label == "LUAD" and record.case_submitter_id.endswith("0000"))
        and not (record.label == "LUSC" and record.case_submitter_id.endswith(("0000", "0001")))
    }

    valid_records = _graph_valid_records(records, complete_keys)
    counts = _split_counts_for_graph_valid_cohort(valid_records)

    assert {label: sum(record.label == label for record in valid_records) for label in ("LUAD", "LUSC")} == {
        "LUAD": 5,
        "LUSC": 4,
    }
    assert counts == {"per_class": 4, "train": 2, "validation": 1, "test": 1}


def test_benchmark_notebook_is_unexecuted_and_calls_only_the_frozen_runner():
    path = Path("notebooks/kaggle_case_disjoint_benchmark.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert len(notebook["cells"]) == 3
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code)
    source = "\n".join("".join(cell["source"]) for cell in code)
    assert "scripts/kaggle_case_disjoint_benchmark.py" in source
    assert "kaggle_real_data_pilot.py" not in source
    assert "KAGGLE_KEY" not in source

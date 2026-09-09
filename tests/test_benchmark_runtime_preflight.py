"""Offline regressions for checks that must precede real-data acquisition."""

from types import SimpleNamespace
import hashlib
import json

import pytest

from scripts import kaggle_case_disjoint_benchmark as runner


def test_streaming_budget_accepts_cohort_larger_than_available_disk():
    # Only one slide is resident; total transfer is not a disk-space requirement.
    records = [SimpleNamespace(file_size=2232047646)] + [
        SimpleNamespace(file_size=650000000) for _ in range(99)
    ]
    assert runner._validate_streaming_budget(records, free_bytes=8 * 1024**3) == 66582047646


@pytest.mark.parametrize("sizes,free,message", [
    ([3 * 1024**3 + 1], 20 * 1024**3, "individual"),
    ([1024**3] * 71, 20 * 1024**3, "total"),
    ([2232047646], 3 * 1024**3, "free disk"),
])
def test_streaming_budget_fails_before_download(sizes, free, message):
    with pytest.raises(RuntimeError, match=message):
        runner._validate_streaming_budget(
            [SimpleNamespace(file_size=size) for size in sizes], free_bytes=free
        )


def test_all_model_training_preflight_runs_on_tiny_synthetic_cpu_graphs():
    evidence = runner._model_training_preflight(device="cpu", nodes_per_graph=3, graph_count=2)
    assert set(evidence) == {"gcn", "graphsage", "gat", "graphgps"}
    assert all(value["output_shape"] == [2, 2] for value in evidence.values())
    assert all(value["backward_finite"] is True for value in evidence.values())


def test_private_manifests_reconstruct_the_published_hashes(tmp_path):
    from tests.test_benchmark_contracts import benchmark_records
    from data.benchmark import select_balanced_cases, stratified_case_split
    from data.gdc import manifest_summary

    cohort = select_balanced_cases(benchmark_records(3), per_class=3, seed=17)
    split = stratified_case_split(cohort, seed=17, train_per_class=1,
                                  validation_per_class=1, test_per_class=1)
    runner._write_private_manifests(tmp_path, cohort, split)
    assert hashlib.sha256((tmp_path / "cohort.json").read_bytes()).hexdigest() == manifest_summary(cohort)["manifest_sha256"]
    assert hashlib.sha256((tmp_path / "split.json").read_bytes()).hexdigest() == split.split_sha256
    assert len(json.loads((tmp_path / "cohort.json").read_text())) == 6
    with pytest.raises(FileExistsError):
        runner._write_private_manifests(tmp_path, cohort, split)


def test_seed_evidence_retains_means_curves_and_classifier_identity():
    from data.evidence import build_stage_evidence
    common = {
        "dataset_release": "synthetic integration fixture", "manifest_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64, "code_revision": "c" * 40,
        "dependency_lock_sha256": "d" * 64, "device": {"type": "cpu", "name": "test"},
        "artifact_count": 2, "class_counts": {"LUAD": 1, "LUSC": 1},
        "artifact_hashes": {"graph": "e" * 64}, "positive_class": "LUSC",
        "uncertainty_method": "synthetic test only",
    }
    metrics = dict(accuracy=0.5, macro_f1=0.5, auroc=0.5, auprc=0.5)
    seed = dict(seed=17, best_epoch=0, epochs_run=1, best_validation_loss=0.7,
                selected_state_sha256="f" * 64, train_loss=[0.8], validation_loss=[0.7],
                confidence_intervals_95={}, metrics=metrics)
    provenance = runner._seed_provenance(common, "gcn", {"mean_metrics": metrics}, seed)
    evidence = build_stage_evidence(stage="benchmark", status="passed",
                                    provenance=provenance, metrics=metrics)
    assert evidence["provenance"]["artifact_hashes"]["selected_classifier_state"] == "f" * 64
    assert evidence["provenance"]["mean_metrics_across_seeds"] == metrics
    assert evidence["provenance"]["train_loss"] == [0.8]
    assert evidence["provenance"]["validation_loss"] == [0.7]
    assert "selected_classifier_state" not in common["artifact_hashes"]

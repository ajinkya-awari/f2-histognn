"""Run the real-data graph QC gate on private Kaggle compute only."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data.benchmark import sanitized_split_summary, stratified_case_split
from data.calibrated_cohort import select_calibrated_cases
from data.benchmark_graphs import load_private_graphs, stream_slide_tiles
from data.evidence import build_stage_evidence, write_evidence
from data.gdc import extract_gdc_response_hits, filter_diagnostic_hits, manifest_summary, parse_gdc_hits
from data.tiff_metadata import TIFFMetadataError, probe_aperio_calibration
from data.transfer import download_slide_ranges, read_header_range
from scripts.kaggle_case_disjoint_benchmark import (
    CANDIDATE_TILES_PER_CASE,
    CASES_PER_CLASS,
    DISK_RESERVE_BYTES,
    MAX_TOTAL_SLIDE_BYTES,
    TEST_PER_CLASS,
    TRAIN_PER_CLASS,
    VALIDATION_PER_CLASS,
    _digest,
    _extract_one_slide,
    _query_eligible_records,
    _require_cuda_execution,
    _require_kaggle_private_runtime,
    _validate_streaming_budget,
    _write_private_manifests,
)
from scripts.kaggle_real_data_discovery import PILOT_SEED
from scripts.kaggle_real_data_pilot import (
    CHECKPOINT_SHA256,
    HOVERNET_REVISION,
    _dependency_lock_hash,
    _prepare_hovernet,
    _run_hovernet,
)
from training.benchmark import FROZEN_POLICY, partition_private_graphs


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fetch_slide(record, destination: Path, *, max_transfer_bytes: int | None = None):
    if shutil.disk_usage(destination.parent).free < record.file_size + DISK_RESERVE_BYTES:
        raise RuntimeError("insufficient free disk for the next streamed slide")
    print(json.dumps({"stage": "slide_download", "expected_bytes": record.file_size}), flush=True)
    started = time.monotonic()
    result = download_slide_ranges(record, destination, max_transfer_bytes=max_transfer_bytes)
    print(
        json.dumps(
            {
                "stage": "slide_download_verified",
                **result,
                "seconds": round(time.monotonic() - started, 3),
            }
        ),
        flush=True,
    )
    return result


def _select_calibrated_inputs(eligible):
    def probe(record):
        try:
            return probe_aperio_calibration(
                lambda start, length: read_header_range(record, start, length), record.file_size
            )
        except TIFFMetadataError:
            return {"objective_power": None}

    return select_calibrated_cases(eligible, probe, per_class=CASES_PER_CLASS, seed=PILOT_SEED)


def main() -> int:
    started = time.monotonic()
    timestamp = _timestamp()
    working = Path("/kaggle/working")
    private = Path("/kaggle/temp/project07-private-graph-qc")
    root = Path(__file__).resolve().parents[1]
    source_revision = _require_kaggle_private_runtime(
        os.environ,
        working,
        Path("/kaggle/temp"),
        Path("/kaggle/input"),
        Path("/kaggle/lib/kaggle/gcp.py"),
        _digest(Path("/kaggle/temp/project07-source.zip"), "sha256"),
    )
    if private.exists():
        raise RuntimeError("private graph-QC directory already exists; refusing to overwrite")
    private.mkdir(mode=0o700)
    evidence_dir = working / "project07-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    device_name = _require_cuda_execution()

    eligible = _query_eligible_records()
    calibrated = _select_calibrated_inputs(eligible)
    cohort = calibrated.records
    expected_bytes = _validate_streaming_budget(cohort, free_bytes=shutil.disk_usage(private).free)
    split = stratified_case_split(
        cohort,
        seed=PILOT_SEED,
        train_per_class=TRAIN_PER_CLASS,
        validation_per_class=VALIDATION_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
    )
    split_summary = sanitized_split_summary(split)
    cohort_summary = manifest_summary(cohort)
    _write_private_manifests(private, cohort, split)
    with (private / "calibration.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(calibrated.objective_by_file), handle, sort_keys=True, separators=(",", ":"))
    print(
        json.dumps(
            {
                "stage": "graph_qc_cohort_preflight",
                "expected_slide_bytes": expected_bytes,
                "manifest_sha256": cohort_summary["manifest_sha256"],
                "split": split_summary,
            }
        ),
        flush=True,
    )

    checkout, checkpoint, patch_hash = _prepare_hovernet(private)
    transferred_bytes = 0

    def fetch_bounded(record, destination):
        nonlocal transferred_bytes
        budget = min(2 * record.file_size, MAX_TOTAL_SLIDE_BYTES - transferred_bytes)
        if budget < record.file_size:
            raise RuntimeError("remaining total transfer budget cannot cover the next slide")
        transport = _fetch_slide(record, destination, max_transfer_bytes=budget)
        transferred_bytes += transport["transferred_bytes"]

    streamed = stream_slide_tiles(
        cohort,
        private,
        fetch_bounded,
        _extract_one_slide,
        tiles_per_case=CANDIDATE_TILES_PER_CASE,
    )
    json_dir = _run_hovernet(checkout, checkpoint, streamed.tile_dir, private)
    graph_set = load_private_graphs(
        json_dir,
        streamed.tile_to_case,
        tiles_per_case=FROZEN_POLICY.tiles_per_case,
        max_nodes=512,
    )
    partitions = partition_private_graphs(graph_set.graphs, split)
    graph_counts = {name: len(value) for name, value in partitions.items()}
    if graph_counts != {"train": 240, "validation": 80, "test": 80}:
        raise RuntimeError("graph partition counts do not match the frozen case split")
    torch.cuda.synchronize()
    evidence = build_stage_evidence(
        stage="real_data_graph_qc",
        status="passed",
        timestamp_utc=timestamp,
        provenance={
            "dataset_release": f"GDC API query {timestamp}",
            "manifest_sha256": cohort_summary["manifest_sha256"],
            "split_manifest_sha256": split.split_sha256,
            "code_revision": source_revision,
            "source_archive_sha256": os.environ["PROJECT07_SOURCE_ARCHIVE_SHA256"],
            "benchmark_spec_sha256": _digest(root / "docs/CASE_DISJOINT_BENCHMARK_SPEC.md", "sha256"),
            "dependency_lock_sha256": _dependency_lock_hash(root),
            "device": {"type": "cuda", "name": device_name},
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_geometric": importlib.metadata.version("torch-geometric"),
            "cuda_runtime": torch.version.cuda,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "runtime_seconds": round(time.monotonic() - started, 3),
            "cohort_case_count": len(cohort),
            "downloaded_slide_bytes": streamed.total_slide_bytes,
            "actual_slide_transfer_bytes_including_retries": transferred_bytes,
            "split_counts": split_summary,
            "graph_counts": graph_counts,
            "tile_candidates_per_case": CANDIDATE_TILES_PER_CASE,
            "valid_graphs_per_case": FROZEN_POLICY.tiles_per_case,
            "rejected_tile_counts": dict(graph_set.rejected_tile_counts),
            "calibration_eligibility": {
                "policy": "seed-ranked cases; first Aperio slide with declared objective power 10-80; no inferred calibration",
                "excluded_case_counts": dict(calibrated.excluded_cases),
                "probed_slide_count": calibrated.probed_slides,
                "objective_counts": dict(Counter(str(value) for value in calibrated.objective_by_file.values())),
            },
            "artifact_hashes": {
                "graph": graph_set.artifact_sha256,
                "segmentation_checkpoint": CHECKPOINT_SHA256,
                "hovernet_patch": patch_hash,
                "calibration_manifest": _digest(private / "calibration.json", "sha256"),
            },
            "hovernet_revision": HOVERNET_REVISION,
            "benchmark_metrics_emitted": False,
            "classifier_training_executed": False,
            "raw_artifacts_published": False,
        },
    )
    write_evidence(evidence_dir / f"real_data_graph_qc_{timestamp.replace(':', '')}.json", evidence)
    print(json.dumps({"status": "passed", "stage": "real_data_graph_qc"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

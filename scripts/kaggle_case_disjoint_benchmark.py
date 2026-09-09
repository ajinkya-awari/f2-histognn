"""Execute the frozen 100-case benchmark only on attested private Kaggle GPU."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.benchmark import sanitized_split_summary, select_balanced_cases, stratified_case_split
from data.benchmark_graphs import load_private_graphs, stream_slide_tiles
from data.evidence import build_stage_evidence, write_evidence
from data.gdc import build_gdc_query_payload, extract_gdc_response_hits, filter_diagnostic_hits, manifest_summary, parse_gdc_hits
from data.tiles import select_tissue_tile_origins
from scripts.kaggle_real_data_discovery import PILOT_SEED, _query
from scripts.kaggle_real_data_pilot import (
    CHECKPOINT_SHA256,
    HOVERNET_REVISION,
    _bounded_copy,
    _dependency_lock_hash,
    _digest,
    _prepare_hovernet,
    _require_cuda_execution,
    _require_kaggle_private_runtime,
    _run_hovernet,
)
from training.benchmark import FROZEN_POLICY, partition_private_graphs, run_frozen_benchmark


CASES_PER_CLASS = 50
TRAIN_PER_CLASS = 30
VALIDATION_PER_CLASS = 10
TEST_PER_CLASS = 10
MAX_INDIVIDUAL_SLIDE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_SLIDE_BYTES = 20 * 1024 * 1024 * 1024
TILE_PIXELS_AT_40X = 256


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fetch_slide(record, destination: Path) -> None:
    request = Request(
        f"https://api.gdc.cancer.gov/data/{record.file_id}",
        headers={"User-Agent": "f2-histognn/0.1"},
    )
    with urlopen(request, timeout=120) as response, destination.open("xb") as handle:
        _bounded_copy(
            response,
            handle,
            expected_bytes=record.file_size,
            remaining_bytes=record.file_size,
        )


def _extract_one_slide(record, slide_path: Path, tile_dir: Path) -> tuple[Path, ...]:
    import openslide
    from PIL import Image

    paths = []
    with openslide.OpenSlide(str(slide_path)) as slide:
        try:
            objective = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("slide objective power is missing or invalid") from exc
        if not math.isfinite(objective) or not 10.0 <= objective <= 80.0:
            raise RuntimeError("slide objective power is outside the accepted range")
        source_pixels = max(1, int(round(TILE_PIXELS_AT_40X * objective / 40.0)))
        thumbnail = np.asarray(slide.get_thumbnail((512, 512)).convert("RGB"))
        origins = select_tissue_tile_origins(
            thumbnail,
            slide_size=slide.dimensions,
            tile_size=source_pixels,
            count=FROZEN_POLICY.tiles_per_case,
            seed=PILOT_SEED,
        )
        for tile_index, origin in enumerate(origins):
            tile = slide.read_region(origin, 0, (source_pixels, source_pixels)).convert("RGB")
            if source_pixels != TILE_PIXELS_AT_40X:
                tile = tile.resize(
                    (TILE_PIXELS_AT_40X, TILE_PIXELS_AT_40X),
                    resample=Image.Resampling.LANCZOS,
                )
            tile_key = hashlib.sha256(
                f"{record.file_id}:{tile_index}:{origin[0]}:{origin[1]}".encode()
            ).hexdigest()[:24]
            path = tile_dir / f"graph-{tile_key}.png"
            tile.save(path)
            paths.append(path)
    return tuple(paths)


def main() -> int:
    started = time.monotonic()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    timestamp = _timestamp()
    working = Path("/kaggle/working")
    private = Path("/kaggle/temp/project07-private-benchmark")
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
        raise RuntimeError("private benchmark directory already exists; refusing to overwrite")
    private.mkdir(mode=0o700)
    evidence_dir = working / "project07-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    device_name = _require_cuda_execution()
    response = _query(build_gdc_query_payload())
    eligible = parse_gdc_hits(filter_diagnostic_hits(extract_gdc_response_hits(response)))
    cohort = select_balanced_cases(eligible, per_class=CASES_PER_CLASS, seed=PILOT_SEED)
    if any(record.file_size > MAX_INDIVIDUAL_SLIDE_BYTES for record in cohort):
        raise RuntimeError("selected slide exceeds the 2 GiB individual safety boundary")
    expected_bytes = sum(record.file_size for record in cohort)
    if expected_bytes > MAX_TOTAL_SLIDE_BYTES:
        raise RuntimeError("selected cohort exceeds the 20 GiB total safety boundary")
    split = stratified_case_split(
        cohort,
        seed=PILOT_SEED,
        train_per_class=TRAIN_PER_CLASS,
        validation_per_class=VALIDATION_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
    )
    split_summary = sanitized_split_summary(split)
    cohort_summary = manifest_summary(cohort)
    checkout, checkpoint, patch_hash = _prepare_hovernet(private)
    streamed = stream_slide_tiles(cohort, private, _fetch_slide, _extract_one_slide)
    json_dir = _run_hovernet(checkout, checkpoint, streamed.tile_dir, private)
    graph_set = load_private_graphs(
        json_dir,
        streamed.tile_to_case,
        tiles_per_case=FROZEN_POLICY.tiles_per_case,
        max_nodes=512,
    )
    partitions = partition_private_graphs(graph_set.graphs, split)
    results = run_frozen_benchmark(
        partitions,
        split_sha256=split.split_sha256,
        graph_sha256=graph_set.artifact_sha256,
        provenance={
            "source_revision": source_revision,
            "manifest_sha256": cohort_summary["manifest_sha256"],
            "split_manifest_sha256": split.split_sha256,
        },
        device="cuda",
    )
    torch.cuda.synchronize()
    runtime_seconds = round(time.monotonic() - started, 3)
    dependency_hash = _dependency_lock_hash(root)
    common = {
        "dataset_release": f"GDC API query {timestamp}",
        "manifest_sha256": cohort_summary["manifest_sha256"],
        "split_manifest_sha256": split.split_sha256,
        "code_revision": source_revision,
        "dependency_lock_sha256": dependency_hash,
        "device": {"type": "cuda", "name": device_name},
        "artifact_count": len(split.test),
        "class_counts": dict(Counter(record.label for record in split.test)),
        "artifact_hashes": {
            "graph": graph_set.artifact_sha256,
            "checkpoint": CHECKPOINT_SHA256,
            "hovernet_patch": patch_hash,
        },
        "positive_class": "LUSC",
        "uncertainty_method": "2000 deterministic class-stratified case bootstrap resamples; percentile 95% interval",
        "cohort_case_count": len(cohort),
        "downloaded_slide_bytes": streamed.total_slide_bytes,
        "split_counts": split_summary,
        "graph_counts": results["graph_counts"],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_geometric": importlib.metadata.version("torch-geometric"),
        "cuda_runtime": torch.version.cuda,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "runtime_seconds": runtime_seconds,
        "raw_artifacts_published": False,
    }
    pending = []
    for model_name, model_result in results["models"].items():
        for seed_result in model_result["seeds"]:
            provenance = {
                **common,
                "seed": seed_result["seed"],
                "model": model_name,
                "best_epoch": seed_result["best_epoch"],
                "epochs_run": seed_result["epochs_run"],
                "best_validation_loss": seed_result["best_validation_loss"],
                "confidence_intervals_95": seed_result["confidence_intervals_95"],
            }
            metrics = {
                name: seed_result["metrics"][name]
                for name in ("accuracy", "macro_f1", "auroc", "auprc")
            }
            pending.append(
                (
                    evidence_dir / f"benchmark_{model_name}_seed{seed_result['seed']}_{timestamp.replace(':', '')}.json",
                    build_stage_evidence(
                        stage="benchmark",
                        status="passed",
                        timestamp_utc=timestamp,
                        provenance=provenance,
                        metrics=metrics,
                    ),
                )
            )
    if len(pending) != 12:
        raise RuntimeError("refusing to publish incomplete benchmark evidence")
    for path, evidence in pending:
        write_evidence(path, evidence)
    print(json.dumps({"status": "passed", "evidence_count": len(pending)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

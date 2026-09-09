"""Execute the frozen 100-case benchmark only on attested private Kaggle GPU."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.benchmark import sanitized_split_summary, select_balanced_cases, stratified_case_split
from data.benchmark_graphs import load_private_graphs, stream_slide_tiles
from data.evidence import build_stage_evidence, write_evidence
from data.gdc import extract_gdc_response_hits, filter_diagnostic_hits, manifest_summary, parse_gdc_hits
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
from models import MODEL_REGISTRY


CASES_PER_CLASS = 50
TRAIN_PER_CLASS = 30
VALIDATION_PER_CLASS = 10
TEST_PER_CLASS = 10
MAX_INDIVIDUAL_SLIDE_BYTES = 3 * 1024**3
MAX_TOTAL_SLIDE_BYTES = 70 * 1024**3
DISK_RESERVE_BYTES = 2 * 1024**3
TILE_PIXELS_AT_40X = 256


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fetch_slide(record, destination: Path) -> None:
    if shutil.disk_usage(destination.parent).free < record.file_size + DISK_RESERVE_BYTES:
        raise RuntimeError("insufficient free disk for the next streamed slide")
    print(json.dumps({"stage": "slide_download", "expected_bytes": record.file_size}), flush=True)
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


def _query_eligible_records(query=_query):
    """Call the shared GDC query with its required integer page-size contract."""

    response = query(10000)
    return parse_gdc_hits(filter_diagnostic_hits(extract_gdc_response_hits(response)))


def _validate_streaming_budget(records, *, free_bytes: int) -> int:
    largest = max(record.file_size for record in records)
    total = sum(record.file_size for record in records)
    if largest > MAX_INDIVIDUAL_SLIDE_BYTES:
        raise RuntimeError("selected slide exceeds the 3 GiB individual safety boundary")
    if total > MAX_TOTAL_SLIDE_BYTES:
        raise RuntimeError("selected cohort exceeds the 70 GiB total safety boundary")
    if free_bytes < largest + DISK_RESERVE_BYTES:
        raise RuntimeError("insufficient free disk for streamed slides plus reserve")
    return total


def _write_private_manifests(private: Path, cohort, split) -> None:
    payloads = {
        "cohort.json": [asdict(record) for record in sorted(cohort)],
        "split.json": {
            name: [
                {key: getattr(record, key) for key in ("case_id", "file_id", "label", "md5sum")}
                for record in getattr(split, name)
            ]
            for name in ("train", "validation", "test")
        },
    }
    for name, payload in payloads.items():
        with (private / name).open("xb") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _model_training_preflight(*, device="cuda", nodes_per_graph=512, graph_count=16):
    """One synthetic optimizer step per architecture, before real acquisition."""
    n = nodes_per_graph * graph_count
    indices = torch.arange(n, device=device)
    batch = indices // nodes_per_graph
    x = torch.zeros((n, 7), device=device)
    x[:, 0] = (indices % nodes_per_graph).float() / max(1, nodes_per_graph - 1)
    x[:, 2:] = 0.2
    # Eight local ring neighbours exercise the production edge-density bound.
    source = indices.repeat(8)
    destination = ((source % nodes_per_graph + torch.arange(1, 9, device=device)
                    .repeat_interleave(n)) % nodes_per_graph
                   + (source // nodes_per_graph) * nodes_per_graph)
    edges = torch.stack((source, destination))
    edge_attr = torch.ones((edges.shape[1], 1), device=device)
    labels = torch.arange(graph_count, device=device) % 2
    results = {}
    for name in FROZEN_POLICY.models:
        model = MODEL_REGISTRY[name](hidden_dim=32, seed=17).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        output = model(x, edges, edge_attr, batch)
        if output.shape != (graph_count, 2) or not torch.isfinite(output).all():
            raise RuntimeError(f"synthetic training preflight output failed: {name}")
        loss = torch.nn.functional.cross_entropy(output, labels, reduction="sum")
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise RuntimeError(f"synthetic training preflight backward failed: {name}")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(f"synthetic training preflight update failed: {name}")
        results[name] = {"output_shape": list(output.shape), "backward_finite": True}
        del model, optimizer, output, loss
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return results


def _seed_provenance(common, model_name, model_result, seed_result):
    return {
        **common,
        **{key: seed_result[key] for key in (
            "seed", "best_epoch", "epochs_run", "best_validation_loss",
            "confidence_intervals_95", "train_loss", "validation_loss",
        )},
        "model": model_name,
        "artifact_hashes": {
            **common["artifact_hashes"],
            "selected_classifier_state": seed_result["selected_state_sha256"],
        },
        "mean_metrics_across_seeds": model_result["mean_metrics"],
    }


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
    preflight = _model_training_preflight()
    print(json.dumps({"stage": "synthetic_training_preflight", "models": preflight}), flush=True)
    eligible = _query_eligible_records()
    cohort = select_balanced_cases(eligible, per_class=CASES_PER_CLASS, seed=PILOT_SEED)
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
    print(json.dumps({"stage": "cohort_preflight", "expected_slide_bytes": expected_bytes,
                      "manifest_sha256": cohort_summary["manifest_sha256"],
                      "split": split_summary}), flush=True)
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
        "source_archive_sha256": os.environ["PROJECT07_SOURCE_ARCHIVE_SHA256"],
        "benchmark_spec_sha256": _digest(root / "docs/CASE_DISJOINT_BENCHMARK_SPEC.md", "sha256"),
        "model_training_policy": results["policy"],
        "batch_size": 16,
        "batch_order": "one seed-shuffled order materialized and reused across epochs",
        "training_loss_reduction": "sum per batch; mean per graph for reporting",
        "synthetic_training_preflight": preflight,
        "hovernet_revision": HOVERNET_REVISION,
        "determinism_scope": "strict classifier process; HoVer-Net subprocess uses its pinned inference implementation",
        "dependency_lock_sha256": dependency_hash,
        "device": {"type": "cuda", "name": device_name},
        "artifact_count": len(split.test),
        "class_counts": dict(Counter(record.label for record in split.test)),
        "artifact_hashes": {
            "graph": graph_set.artifact_sha256,
            "segmentation_checkpoint": CHECKPOINT_SHA256,
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
            provenance = _seed_provenance(common, model_name, model_result, seed_result)
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

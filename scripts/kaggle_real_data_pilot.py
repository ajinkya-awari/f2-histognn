"""Run the bounded real-data nuclei/graph smoke on private Kaggle compute."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.evidence import build_stage_evidence, write_evidence
from data.gdc import (
    extract_gdc_response_hits,
    filter_diagnostic_hits,
    manifest_summary,
    PILOT_MAX_TOTAL_BYTES,
    parse_gdc_hits,
    select_case_disjoint_pilot,
)
from data.graph import build_knn_graph
from data.hovernet import load_hovernet_instances
from data.hovernet_patch import patch_hovernet_checkout
from data.sampling import farthest_point_sampling
from data.tiles import select_tissue_tile_origins
from models import GAT, GCN, GraphGPS, GraphSAGE
from scripts.kaggle_real_data_discovery import (
    HOVERNET_REPOSITORY,
    HOVERNET_REVISION,
    PILOT_PER_CLASS,
    PILOT_SEED,
    PANNUKE_WEIGHT_DRIVE_ID,
    _query,
)


CHECKPOINT_SHA256 = "4a1463467737f81203a0513f794276cbcbbd6bb470969584f314167c6acef081"
MAX_TOTAL_SLIDE_BYTES = PILOT_MAX_TOTAL_BYTES
TILES_PER_SLIDE = 4
TILE_PIXELS_AT_40X = 256
MAX_NODES_PER_GRAPH = 512


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _dependency_lock_hash(root: Path) -> str:
    hasher = hashlib.sha256()
    for name in ("requirements.txt", "requirements-kaggle-real-data.txt"):
        hasher.update(name.encode("utf-8") + b"\0")
        hasher.update((root / name).read_bytes())
    return hasher.hexdigest()


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _require_kaggle_private_runtime(
    environment,
    working: Path,
    temporary: Path,
    input_directory: Path,
    kaggle_marker: Path,
    observed_archive_hash: str,
) -> str:
    revision = environment.get("PROJECT07_SOURCE_REVISION", "")
    expected_archive_hash = environment.get("PROJECT07_SOURCE_ARCHIVE_SHA256", "")
    if (
        not working.is_dir()
        or not temporary.is_dir()
        or not input_directory.is_dir()
        or not kaggle_marker.is_file()
        or not environment.get("KAGGLE_KERNEL_RUN_TYPE")
        or environment.get("PROJECT07_PRIVATE_KERNEL") != "true"
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or re.fullmatch(r"[0-9a-f]{64}", expected_archive_hash) is None
        or expected_archive_hash != observed_archive_hash
    ):
        raise RuntimeError("refusing real-data execution without Kaggle private-runtime attestation")
    return revision


def _bounded_copy(source, destination, *, expected_bytes: int, remaining_bytes: int) -> int:
    limit = min(expected_bytes, remaining_bytes)
    copied = 0
    while True:
        block = source.read(min(1024 * 1024, limit - copied + 1))
        if not block:
            break
        if copied + len(block) > limit:
            raise RuntimeError("slide download exceeded byte boundary")
        destination.write(block)
        copied += len(block)
    if copied != expected_bytes:
        raise RuntimeError("downloaded slide byte size mismatch")
    return copied


def _prepare_hovernet(private: Path) -> tuple[Path, Path, str]:
    checkout = private / "hover_net"
    if checkout.exists():
        raise RuntimeError("HoVer-Net checkout already exists; refusing to overwrite")
    _run(["git", "clone", "--filter=blob:none", "--no-checkout", HOVERNET_REPOSITORY, str(checkout)])
    _run(["git", "checkout", "--detach", HOVERNET_REVISION], cwd=checkout)
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if observed != HOVERNET_REVISION:
        raise RuntimeError("HoVer-Net revision mismatch")

    import gdown

    checkpoint = private / "hovernet_fast_pannuke_type_tf2pytorch.tar"
    result = gdown.download(id=PANNUKE_WEIGHT_DRIVE_ID, output=str(checkpoint), quiet=True)
    if result is None or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("HoVer-Net checkpoint download failed")
    observed_hash = _digest(checkpoint, "sha256")
    if observed_hash != CHECKPOINT_SHA256:
        raise RuntimeError("HoVer-Net checkpoint SHA-256 mismatch")

    patch_hash = patch_hovernet_checkout(checkout)
    _run(
        [sys.executable, "-c", "from infer.tile import InferManager"],
        cwd=checkout,
    )
    return checkout, checkpoint, patch_hash


def _download_slides(records, private: Path) -> tuple[list[tuple[object, Path]], int]:
    total_expected = sum(record.file_size for record in records)
    if total_expected > MAX_TOTAL_SLIDE_BYTES:
        raise RuntimeError("pilot slide bytes exceed the configured 2 GiB boundary")
    slide_dir = private / "slides"
    slide_dir.mkdir()
    downloaded: list[tuple[object, Path]] = []
    actual_total = 0
    for record in records:
        safe_name = hashlib.sha256(record.file_id.encode("utf-8")).hexdigest()[:20] + ".svs"
        destination = slide_dir / safe_name
        request = Request(
            f"https://api.gdc.cancer.gov/data/{record.file_id}",
            headers={"User-Agent": "f2-histognn/0.1"},
        )
        with urlopen(request, timeout=120) as response, destination.open("xb") as handle:
            copied = _bounded_copy(
                response,
                handle,
                expected_bytes=record.file_size,
                remaining_bytes=MAX_TOTAL_SLIDE_BYTES - actual_total,
            )
        if _digest(destination, "md5") != record.md5sum:
            raise RuntimeError("downloaded slide MD5 mismatch")
        actual_total += copied
        downloaded.append((record, destination))
    return downloaded, actual_total


def _extract_tiles(downloaded, private: Path) -> tuple[Path, int]:
    import openslide
    from PIL import Image

    tile_dir = private / "tiles"
    tile_dir.mkdir()
    count = 0
    for record, path in downloaded:
        with openslide.OpenSlide(str(path)) as slide:
            value = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
            try:
                objective = float(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("slide objective power is missing or invalid") from exc
            if not math.isfinite(objective) or objective < 10.0 or objective > 80.0:
                raise RuntimeError("slide objective power is outside the accepted range")
            source_pixels = max(1, int(round(TILE_PIXELS_AT_40X * objective / 40.0)))
            thumbnail = np.asarray(slide.get_thumbnail((512, 512)).convert("RGB"))
            origins = select_tissue_tile_origins(
                thumbnail,
                slide_size=slide.dimensions,
                tile_size=source_pixels,
                count=TILES_PER_SLIDE,
                seed=PILOT_SEED,
            )
            for tile_index, origin in enumerate(origins):
                tile = slide.read_region(origin, 0, (source_pixels, source_pixels)).convert("RGB")
                if source_pixels != TILE_PIXELS_AT_40X:
                    tile = tile.resize(
                        (TILE_PIXELS_AT_40X, TILE_PIXELS_AT_40X),
                        resample=Image.Resampling.LANCZOS,
                    )
                private_id = hashlib.sha256(
                    f"{record.file_id}:{tile_index}:{origin[0]}:{origin[1]}".encode("utf-8")
                ).hexdigest()[:24]
                tile.save(tile_dir / f"graph-{private_id}.png")
                count += 1
    return tile_dir, count


def _run_hovernet(checkout: Path, checkpoint: Path, tile_dir: Path, private: Path) -> Path:
    type_info = private / "pannuke-types.json"
    type_info.write_text(
        json.dumps(
            {
                "0": ["background", [255, 255, 255]],
                "1": ["neoplastic", [255, 0, 0]],
                "2": ["inflammatory", [0, 255, 0]],
                "3": ["connective", [0, 0, 255]],
                "4": ["dead", [255, 255, 0]],
                "5": ["non-neoplastic epithelial", [0, 255, 255]],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output = private / "hovernet-output"
    _run(
        [
            sys.executable,
            "run_infer.py",
            "--gpu=0",
            "--nr_types=6",
            f"--type_info_path={type_info}",
            "--model_mode=fast",
            f"--model_path={checkpoint}",
            "--nr_inference_workers=2",
            "--nr_post_proc_workers=2",
            "--batch_size=16",
            "tile",
            f"--input_dir={tile_dir}",
            f"--output_dir={output}",
            "--mem_usage=0.1",
        ],
        cwd=checkout,
    )
    return output / "json"


def _graph_smoke(json_dir: Path) -> dict[str, object]:
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError("HoVer-Net produced no instance JSON files")
    graph_hasher = hashlib.sha256()
    graph_count = 0
    total_nodes = 0
    smoke_graph = None
    for path in json_files:
        record = load_hovernet_instances(path, max_nuclei=100000)
        selected = farthest_point_sampling(
            record.centroid, cap=MAX_NODES_PER_GRAPH, random_start=False
        )
        coordinates = record.centroid[selected]
        features = record.features[selected]
        edges, edge_attr = build_knn_graph(coordinates, k=8)
        graph_hasher.update(features.astype("<f8").tobytes())
        graph_hasher.update(edges.astype("<i8").tobytes())
        graph_hasher.update(edge_attr.astype("<f8").tobytes())
        graph_count += 1
        total_nodes += len(selected)
        if smoke_graph is None and len(selected) >= 2:
            smoke_graph = (features, edges, edge_attr)
    if smoke_graph is None:
        raise RuntimeError("no graph contained at least two nuclei")

    features, edges, edge_attr = smoke_graph
    x = torch.tensor(features, dtype=torch.float32, device="cuda")
    edge_index = torch.tensor(edges, dtype=torch.long, device="cuda")
    edge_features = torch.tensor(edge_attr, dtype=torch.float32, device="cuda")
    batch = torch.zeros(x.shape[0], dtype=torch.long, device="cuda")
    outputs = {}
    for name, model_type in (
        ("gcn", GCN),
        ("graphsage", GraphSAGE),
        ("gat", GAT),
        ("graphgps", GraphGPS),
    ):
        model = model_type(hidden_dim=16, seed=PILOT_SEED).to("cuda").eval()
        with torch.no_grad():
            logits = model(x, edge_index, edge_features, batch)
        if logits.shape != (1, 2) or not torch.isfinite(logits).all():
            raise RuntimeError(f"{name} forward contract failed")
        outputs[name] = list(logits.shape)
    return {
        "graph_count": graph_count,
        "total_sampled_nodes": total_nodes,
        "graph_artifact_sha256": graph_hasher.hexdigest(),
        "model_output_shapes": outputs,
    }


def main() -> int:
    started = time.monotonic()
    working = Path("/kaggle/working")
    private = Path("/kaggle/temp/project07-private")
    root = Path(__file__).resolve().parents[1]
    try:
        source_revision = _require_kaggle_private_runtime(
            environment=os.environ,
            working=working,
            temporary=Path("/kaggle/temp"),
            input_directory=Path("/kaggle/input"),
            kaggle_marker=Path("/kaggle/lib/kaggle/gcp.py"),
            observed_archive_hash=_digest(
                Path("/kaggle/temp/project07-source.zip"), "sha256"
            ),
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if private.exists():
        print("private pilot directory already exists; refusing to overwrite", file=sys.stderr)
        return 2
    evidence_dir = working / "project07-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    private.mkdir(mode=0o700)
    timestamp = _timestamp()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        evidence = build_stage_evidence(
            stage="real_data_graph_smoke",
            status="blocked",
            timestamp_utc=timestamp,
            provenance={"reason": "CUDA GPU unavailable", "downloaded_slide_bytes": 0},
        )
        write_evidence(evidence_dir / f"real_data_graph_smoke_{timestamp.replace(':', '')}.json", evidence)
        print(json.dumps(evidence, sort_keys=True))
        return 3

    response = _query(10000)
    records = parse_gdc_hits(filter_diagnostic_hits(extract_gdc_response_hits(response)))
    pilot = select_case_disjoint_pilot(
        records,
        per_class=PILOT_PER_CLASS,
        seed=PILOT_SEED,
        max_total_bytes=MAX_TOTAL_SLIDE_BYTES,
    )
    summary = manifest_summary(pilot)
    checkout, checkpoint, patch_hash = _prepare_hovernet(private)
    downloaded, downloaded_bytes = _download_slides(pilot, private)
    tile_dir, tile_count = _extract_tiles(downloaded, private)
    json_dir = _run_hovernet(checkout, checkpoint, tile_dir, private)
    smoke = _graph_smoke(json_dir)

    evidence = build_stage_evidence(
        stage="real_data_graph_smoke",
        status="passed",
        timestamp_utc=timestamp,
        provenance={
            "dataset_release": f"GDC API query {timestamp}",
            "gdc_source": "https://api.gdc.cancer.gov",
            "manifest_sha256": summary["manifest_sha256"],
            "record_count": summary["record_count"],
            "case_count": summary["case_count"],
            "class_counts": summary["class_counts"],
            "one_slide_per_case": True,
            "selection_seed": PILOT_SEED,
            "downloaded_slide_bytes": downloaded_bytes,
            "download_checksums_verified": True,
            "tiles_per_slide": TILES_PER_SLIDE,
            "tile_count": tile_count,
            "hovernet_revision": HOVERNET_REVISION,
            "hovernet_compatibility_patch_sha256": patch_hash,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "dependency_lock_sha256": _dependency_lock_hash(root),
            "source_revision": source_revision,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_geometric": importlib.metadata.version("torch-geometric"),
            "cuda_runtime": torch.version.cuda,
            "device": {"type": "cuda", "name": torch.cuda.get_device_name(0)},
            "runtime_seconds": round(time.monotonic() - started, 3),
            **smoke,
            "benchmark_metrics_emitted": False,
            "raw_artifacts_published": False,
        },
    )
    write_evidence(evidence_dir / f"real_data_graph_smoke_{timestamp.replace(':', '')}.json", evidence)
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

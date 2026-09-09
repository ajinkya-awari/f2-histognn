"""Private streamed slide and graph preparation for the frozen benchmark."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from data.gdc import GDCSlideRecord
from data.graph import build_knn_graph
from data.hovernet import load_hovernet_instances
from data.sampling import farthest_point_sampling


class BenchmarkGraphError(ValueError):
    """Raised when private streamed graph preparation violates the protocol."""


@dataclass(frozen=True)
class StreamedTiles:
    tile_dir: Path
    tile_to_case: Mapping[str, tuple[str, int]]
    total_slide_bytes: int


@dataclass(frozen=True)
class PrivateGraph:
    tile_key: str
    case_key: str
    label: int
    features: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray


@dataclass(frozen=True)
class PrivateGraphSet:
    graphs: tuple[PrivateGraph, ...]
    artifact_sha256: str


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stream_slide_tiles(
    records: Iterable[GDCSlideRecord],
    private_root: str | Path,
    fetch_slide: Callable[[GDCSlideRecord, Path], Any],
    extract_tiles: Callable[[GDCSlideRecord, Path, Path], Iterable[Path]],
    *,
    tiles_per_case: int = 4,
) -> StreamedTiles:
    """Fetch, verify, tile, and immediately remove one private slide at a time."""

    values = tuple(records)
    if not values or any(not isinstance(record, GDCSlideRecord) for record in values):
        raise BenchmarkGraphError("records must contain GDCSlideRecord values")
    if type(tiles_per_case) is not int or tiles_per_case <= 0:
        raise BenchmarkGraphError("tiles_per_case must be a positive integer")
    root = Path(private_root)
    root.mkdir(parents=True, exist_ok=True)
    slide_dir = root / "streamed-slides"
    tile_dir = root / "benchmark-tiles"
    if slide_dir.exists() or tile_dir.exists():
        raise BenchmarkGraphError("private streaming directories already exist")
    slide_dir.mkdir()
    tile_dir.mkdir()
    mapping: dict[str, tuple[str, int]] = {}
    total_bytes = 0
    for record in values:
        destination = slide_dir / (hashlib.sha256(record.file_id.encode()).hexdigest() + ".svs")
        try:
            fetch_slide(record, destination)
            if not destination.is_file() or destination.stat().st_size != record.file_size:
                raise BenchmarkGraphError("downloaded slide byte size mismatch")
            digest = _digest(destination, "md5")
            if digest != record.md5sum:
                raise BenchmarkGraphError("downloaded slide MD5 mismatch")
            total_bytes += destination.stat().st_size
            paths = tuple(Path(path) for path in extract_tiles(record, destination, tile_dir))
            if len(paths) != tiles_per_case or len(set(paths)) != tiles_per_case:
                raise BenchmarkGraphError(f"each case must produce exactly {tiles_per_case} tiles")
            case_key = hashlib.sha256(record.case_id.encode()).hexdigest()
            label = 0 if record.label == "LUAD" else 1 if record.label == "LUSC" else -1
            if label < 0:
                raise BenchmarkGraphError("record label must be LUAD or LUSC")
            for path in paths:
                if not _within(path, tile_dir):
                    raise BenchmarkGraphError("tile path must remain within the private tile root")
                if not path.is_file() or path.suffix.lower() != ".png":
                    raise BenchmarkGraphError("extractor must produce private PNG tile files")
                if path.stem in mapping:
                    raise BenchmarkGraphError("duplicate private tile key")
                mapping[path.stem] = (case_key, label)
        finally:
            if destination.is_file():
                destination.unlink()
    slide_dir.rmdir()
    return StreamedTiles(tile_dir, mapping, total_bytes)


def load_private_graphs(
    json_dir: str | Path,
    tile_to_case: Mapping[str, tuple[str, int]],
    *,
    tiles_per_case: int = 4,
    max_nodes: int = 512,
) -> PrivateGraphSet:
    """Convert private HoVer-Net JSON files into deterministic bounded graphs."""

    if type(tiles_per_case) is not int or tiles_per_case <= 0:
        raise BenchmarkGraphError("tiles_per_case must be a positive integer")
    if type(max_nodes) is not int or max_nodes <= 0:
        raise BenchmarkGraphError("max_nodes must be a positive integer")
    paths = sorted(Path(json_dir).glob("*.json"))
    if not paths or {path.stem for path in paths} != set(tile_to_case):
        raise BenchmarkGraphError("HoVer-Net JSON and private tile mapping must match exactly")
    counts = Counter(value[0] for value in tile_to_case.values())
    if any(count != tiles_per_case for count in counts.values()):
        raise BenchmarkGraphError(f"each case must contain exactly {tiles_per_case} graphs")
    graphs: list[PrivateGraph] = []
    hasher = hashlib.sha256()
    for path in paths:
        mapping = tile_to_case[path.stem]
        if (
            not isinstance(mapping, tuple)
            or len(mapping) != 2
            or not isinstance(mapping[0], str)
            or not mapping[0]
            or mapping[1] not in (0, 1)
        ):
            raise BenchmarkGraphError("private tile mapping is invalid")
        record = load_hovernet_instances(path, max_nuclei=100000)
        selected = farthest_point_sampling(record.centroid, cap=max_nodes, random_start=False)
        if len(selected) < 2:
            raise BenchmarkGraphError("each graph must contain at least two nuclei")
        coordinates = record.centroid[selected]
        features = record.features[selected]
        edge_index, edge_attr = build_knn_graph(coordinates, k=8)
        for array, dtype in (
            (features, "<f8"),
            (edge_index, "<i8"),
            (edge_attr, "<f8"),
        ):
            hasher.update(np.asarray(array).astype(dtype).tobytes())
        hasher.update(mapping[0].encode())
        hasher.update(str(mapping[1]).encode())
        graphs.append(
            PrivateGraph(path.stem, mapping[0], mapping[1], features, edge_index, edge_attr)
        )
    return PrivateGraphSet(tuple(graphs), hasher.hexdigest())

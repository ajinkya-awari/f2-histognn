"""Adapter from reviewed HoVer-Net instance JSON to the HistoGNN contract."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np

from data.schema import GraphRecord, SchemaValidationError, validate_graph_record


class HoverNetOutputError(ValueError):
    """Raised when HoVer-Net output cannot prove the five-class contract."""


_PROBABILITY_WIDTH_WITH_BACKGROUND = 6
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024


def _load_json(path: Path, max_bytes: int) -> Any:
    if not path.is_file():
        raise HoverNetOutputError("HoVer-Net JSON path must point to a file")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise HoverNetOutputError("max_bytes must be a positive integer")
    if path.stat().st_size > max_bytes:
        raise HoverNetOutputError("HoVer-Net JSON exceeds max_bytes")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HoverNetOutputError("HoVer-Net output must be readable UTF-8 JSON") from exc


def _instance_values(identifier: str, instance: Any) -> tuple[list[float], list[float]]:
    if not isinstance(instance, Mapping):
        raise HoverNetOutputError(f"instance {identifier} must be a mapping")
    nucleus_type = instance.get("type")
    if type(nucleus_type) is not int or not 0 <= nucleus_type <= 5:
        raise HoverNetOutputError(f"instance {identifier} type must be an integer from 0 to 5")

    centroid = instance.get("centroid")
    if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
        raise HoverNetOutputError(f"instance {identifier} centroid must contain two values")
    probabilities = instance.get("probs")
    if not isinstance(probabilities, (list, tuple)):
        raise HoverNetOutputError(f"instance {identifier} probs must be present")
    if len(probabilities) != _PROBABILITY_WIDTH_WITH_BACKGROUND:
        raise HoverNetOutputError(
            f"instance {identifier} probs must contain six values including background"
        )
    try:
        array = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HoverNetOutputError(f"instance {identifier} probs must be numeric") from exc
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise HoverNetOutputError(f"instance {identifier} probs must be finite values in [0, 1]")
    if not np.isclose(float(array.sum()), 1.0, atol=1e-6):
        raise HoverNetOutputError(f"instance {identifier} probs must sum to 1")
    non_background = array[1:]
    non_background_total = float(non_background.sum())
    if np.isclose(non_background_total, 0.0, atol=1e-12):
        raise HoverNetOutputError(
            f"instance {identifier} probs must assign positive non-background probability"
        )
    return list(centroid), (non_background / non_background_total).tolist()


def load_hovernet_instances(
    artifact_path: str | Path,
    *,
    max_nuclei: int | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> GraphRecord:
    """Read bounded instance JSON and preserve five non-background probabilities."""

    if max_nuclei is not None and (type(max_nuclei) is not int or max_nuclei <= 0):
        raise HoverNetOutputError("max_nuclei must be a positive integer")
    payload = _load_json(Path(artifact_path), max_bytes)
    if not isinstance(payload, Mapping) or not payload:
        raise HoverNetOutputError("HoVer-Net JSON must be a non-empty instance mapping")
    if "nuc" in payload:
        if not set(payload).issubset({"mag", "nuc"}) or not isinstance(payload["nuc"], Mapping):
            raise HoverNetOutputError("HoVer-Net nuc wrapper is malformed")
        payload = payload["nuc"]
        if not payload:
            raise HoverNetOutputError("HoVer-Net JSON must contain non-empty nuclei")
    if max_nuclei is not None and len(payload) > max_nuclei:
        raise HoverNetOutputError("HoVer-Net JSON exceeds max_nuclei")

    identifiers = tuple(sorted(payload))
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers):
        raise HoverNetOutputError("instance identifiers must be non-blank strings")
    centroids: list[list[float]] = []
    type_probabilities: list[list[float]] = []
    for identifier in identifiers:
        centroid, probabilities = _instance_values(identifier, payload[identifier])
        centroids.append(centroid)
        type_probabilities.append(probabilities)
    try:
        return validate_graph_record(centroids, type_probabilities, identifiers)
    except SchemaValidationError as exc:
        raise HoverNetOutputError(f"HoVer-Net graph contract failed: {exc}") from exc

"""Validation and feature construction for synthetic nuclei graph records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


NODE_FEATURE_COUNT = 7
TYPE_PROBABILITY_COUNT = 5


class SchemaValidationError(ValueError):
    """Raised when a graph record violates the approved data contract."""


@dataclass(frozen=True)
class GraphRecord:
    """Validated graph arrays and the exact seven-feature node matrix."""

    centroid: np.ndarray
    type_prob: np.ndarray
    features: np.ndarray
    identifiers: tuple[str, ...]


def _float_matrix(value: Any, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{field} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise SchemaValidationError(f"{field} must contain finite values")
    return array


def _validate_arrays(centroid: Any, type_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    coordinates = _float_matrix(centroid, "centroid")
    probabilities = _float_matrix(type_prob, "type_prob")

    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise SchemaValidationError("centroid must have shape [N, 2]")
    if probabilities.ndim != 2 or probabilities.shape[1:] != (TYPE_PROBABILITY_COUNT,):
        raise SchemaValidationError("type_prob must have shape [N, 5]")
    if coordinates.shape[0] == 0:
        raise SchemaValidationError("graph must be non-empty")
    if coordinates.shape[0] != probabilities.shape[0]:
        raise SchemaValidationError("centroid and type_prob must have the same N")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise SchemaValidationError("type_prob values must be between 0 and 1")
    return coordinates, probabilities


def build_node_features(centroid: Any, type_prob: Any) -> np.ndarray:
    """Return normalized x/y coordinates followed by all five probabilities."""

    coordinates, probabilities = _validate_arrays(centroid, type_prob)
    minimum = coordinates.min(axis=0)
    span = coordinates.max(axis=0) - minimum
    normalized = np.zeros_like(coordinates)
    np.divide(coordinates - minimum, span, out=normalized, where=span != 0)
    features = np.concatenate((normalized, probabilities), axis=1)
    if features.shape[1] != NODE_FEATURE_COUNT:
        raise SchemaValidationError("node feature width must be exactly 7")
    return features


def _validate_identifiers(identifiers: Sequence[str] | None, count: int) -> tuple[str, ...]:
    if identifiers is None:
        return tuple(str(index) for index in range(count))
    try:
        values = tuple(identifiers)
    except TypeError as exc:
        raise SchemaValidationError("identifiers must be a sequence") from exc
    if len(values) != count:
        raise SchemaValidationError("identifiers must match the graph node count")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise SchemaValidationError("identifier values must be non-blank strings")
    if len(set(values)) != len(values):
        raise SchemaValidationError("duplicate identifier values are not allowed")
    return values


def validate_graph_record(
    centroid: Any,
    type_prob: Any,
    identifiers: Sequence[str] | None = None,
) -> GraphRecord:
    """Validate arrays and identifiers, returning a seven-feature graph record."""

    coordinates, probabilities = _validate_arrays(centroid, type_prob)
    names = _validate_identifiers(identifiers, coordinates.shape[0])
    features = build_node_features(coordinates, probabilities)
    return GraphRecord(coordinates, probabilities, features, names)


validate_schema = validate_graph_record

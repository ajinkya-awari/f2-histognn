"""Deterministic synthetic-compatible nuclei graph construction utilities."""

from __future__ import annotations

import numpy as np

__all__ = ["build_knn_graph"]


def _validated_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Return coordinates as finite float64 ``[N, 2]`` data."""
    try:
        values = np.asarray(coordinates, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinates must be a numeric [N, 2] array") from exc

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("coordinates must have shape [N, 2]")
    if values.shape[0] == 0:
        raise ValueError("coordinates must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("coordinates must contain only finite values")
    if np.unique(values, axis=0).shape[0] != values.shape[0]:
        raise ValueError("coordinates must not contain duplicate rows")
    return values


def _validated_k(k: int) -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be a non-negative integer")
    if k < 0:
        raise ValueError("k must be a non-negative integer")
    return int(k)


def build_knn_graph(
    coordinates: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic directed k-nearest-neighbor graph.

    Neighbors are ranked by ``(Euclidean distance, node index)``.  This gives
    the lowest node index a stable preference when distances tie.  Edges are
    emitted by increasing source index, then by that ranked-neighbor order;
    self-edges are never emitted.  The effective k is ``min(k, N - 1)``.

    The returned edge feature is one graph-local normalized Euclidean distance
    with shape ``[E, 1]``.  Its denominator is the maximum selected-edge
    distance in this graph, so a non-empty edge set whose maximum is zero is
    rejected.  A graph with no edges returns an empty ``[0, 1]`` feature array.
    """
    values = _validated_coordinates(coordinates)
    requested_k = _validated_k(k)
    node_count = values.shape[0]
    effective_k = min(requested_k, node_count - 1)

    sources: list[int] = []
    destinations: list[int] = []
    for source in range(node_count):
        with np.errstate(over="ignore", invalid="ignore"):
            distances = np.linalg.norm(values - values[source], axis=1)
        if not np.all(np.isfinite(distances)):
            raise ValueError("pairwise distances must be finite")

        # np.lexsort uses the last key as primary: distance, then node index.
        node_indices = np.arange(node_count, dtype=np.int64)
        ranked = np.lexsort((node_indices, distances))
        ranked = ranked[ranked != source][:effective_k]
        sources.extend([source] * effective_k)
        destinations.extend(int(destination) for destination in ranked)

    edge_index = np.asarray([sources, destinations], dtype=np.int64)
    if effective_k == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        return edge_index, np.empty((0, 1), dtype=np.float64)

    source_indices = edge_index[0]
    destination_indices = edge_index[1]
    with np.errstate(over="ignore", invalid="ignore"):
        edge_distances = np.linalg.norm(
            values[source_indices] - values[destination_indices],
            axis=1,
        )
    if not np.all(np.isfinite(edge_distances)):
        raise ValueError("edge distances must be finite")
    maximum_distance = float(np.max(edge_distances))
    if maximum_distance <= 0.0:
        raise ValueError("maximum distance must be positive for non-empty edges")

    edge_attr = (edge_distances / maximum_distance).reshape(-1, 1)
    return edge_index, edge_attr

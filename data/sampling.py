"""Deterministic synthetic-compatible node sampling utilities."""

from __future__ import annotations

import numpy as np

__all__ = ["farthest_point_sampling"]


def _validated_coordinates(coordinates: np.ndarray) -> np.ndarray:
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


def _validated_cap(cap: int) -> int:
    if isinstance(cap, (bool, np.bool_)) or not isinstance(cap, (int, np.integer)):
        raise ValueError("cap must be a positive integer")
    if cap <= 0:
        raise ValueError("cap must be a positive integer")
    return int(cap)


def farthest_point_sampling(
    coordinates: np.ndarray,
    cap: int,
    *,
    random_start: bool = False,
) -> np.ndarray:
    """Select deterministic FPS indices, preserving source-index tie order.

    Inputs must be finite, non-empty, and contain no duplicate coordinate
    rows.  With ``random_start=False`` the first point is always index 0.  If
    the input is at or below ``cap``, all indices are returned in input order.
    Otherwise each next point maximizes its minimum Euclidean distance to the
    selected set; ``np.argmax`` provides the documented lowest-index tie-break.
    """
    values = _validated_coordinates(coordinates)
    maximum_count = _validated_cap(cap)
    if random_start is not False:
        raise ValueError("only random_start=False is supported")

    node_count = values.shape[0]
    if node_count <= maximum_count:
        return np.arange(node_count, dtype=np.int64)

    selected = np.empty(maximum_count, dtype=np.int64)
    selected[0] = 0
    selected_mask = np.zeros(node_count, dtype=bool)
    selected_mask[0] = True
    minimum_distances = np.full(node_count, np.inf, dtype=np.float64)

    for position in range(1, maximum_count):
        previous = selected[position - 1]
        with np.errstate(over="ignore", invalid="ignore"):
            distances = np.linalg.norm(values - values[previous], axis=1)
        if not np.all(np.isfinite(distances)):
            raise ValueError("pairwise distances must be finite")
        minimum_distances = np.minimum(minimum_distances, distances)
        minimum_distances[selected_mask] = -np.inf
        next_index = int(np.argmax(minimum_distances))
        selected[position] = next_index
        selected_mask[next_index] = True

    return selected

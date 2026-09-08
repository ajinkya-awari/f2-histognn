"""Deterministic tissue-tile selection from a low-resolution slide thumbnail."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


class TileSelectionError(ValueError):
    """Raised when a bounded set of tissue tiles cannot be selected."""


def select_tissue_tile_origins(
    thumbnail: Any,
    *,
    slide_size: tuple[int, int],
    tile_size: int,
    count: int,
    seed: int = 0,
) -> tuple[tuple[int, int], ...]:
    """Select stable level-zero tile origins from visibly stained thumbnail pixels."""

    image = np.asarray(thumbnail)
    if image.ndim != 3 or image.shape[2] < 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise TileSelectionError("thumbnail must have shape [H, W, >=3]")
    try:
        rgb = image[:, :, :3].astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TileSelectionError("thumbnail must be numeric") from exc
    if not np.all(np.isfinite(rgb)):
        raise TileSelectionError("thumbnail must contain finite values")

    if (
        not isinstance(slide_size, tuple)
        or len(slide_size) != 2
        or any(type(value) is not int or value <= 0 for value in slide_size)
    ):
        raise TileSelectionError("slide_size must contain two positive integers")
    if type(tile_size) is not int or tile_size <= 0:
        raise TileSelectionError("tile_size must be a positive integer")
    if tile_size > min(slide_size):
        raise TileSelectionError("tile_size must fit inside the slide")
    if type(count) is not int or count <= 0:
        raise TileSelectionError("count must be a positive integer")
    if type(seed) is not int:
        raise TileSelectionError("seed must be an integer")

    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    brightness = rgb.mean(axis=2)
    tissue_rows, tissue_columns = np.where((saturation >= 20.0) & (brightness <= 235.0))
    if len(tissue_rows) == 0:
        raise TileSelectionError("thumbnail contains no eligible tissue pixels")

    slide_width, slide_height = slide_size
    thumbnail_height, thumbnail_width = rgb.shape[:2]
    origins: set[tuple[int, int]] = set()
    for row, column in zip(tissue_rows.tolist(), tissue_columns.tolist(), strict=True):
        center_x = int(round((column + 0.5) * slide_width / thumbnail_width))
        center_y = int(round((row + 0.5) * slide_height / thumbnail_height))
        x = min(max(center_x - tile_size // 2, 0), slide_width - tile_size)
        y = min(max(center_y - tile_size // 2, 0), slide_height - tile_size)
        origins.add((x, y))
    if len(origins) < count:
        raise TileSelectionError(
            f"requested {count} tiles but only {len(origins)} eligible origins are available"
        )

    def rank(origin: tuple[int, int]) -> tuple[str, int, int]:
        x, y = origin
        digest = hashlib.sha256(f"{seed}:{x}:{y}".encode("utf-8")).hexdigest()
        return digest, x, y

    return tuple(sorted(origins, key=rank)[:count])


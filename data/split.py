"""Deterministic patient/case-grouped partitioning."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


class GroupOverlapError(AssertionError):
    """Raised when one group occurs in more than one partition."""


@dataclass(frozen=True)
class GroupSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def _validate_fraction(value: float, field: str) -> float:
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if value < 0 or value > 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return float(value)


def partition_groups(
    groups: Iterable[str],
    *,
    seed: int = 0,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float | None = None,
) -> GroupSplit:
    """Partition unique groups with a seeded permutation and stable outputs."""

    train = _validate_fraction(train_fraction, "train_fraction")
    validation = _validate_fraction(validation_fraction, "validation_fraction")
    if test_fraction is None:
        test = 1.0 - train - validation
        if test < 0.0:
            raise ValueError("train_fraction and validation_fraction sum exceeds 1.0")
    else:
        test = _validate_fraction(test_fraction, "test_fraction")
    if not np.isclose(train + validation + test, 1.0):
        raise ValueError("split fractions must sum to 1")

    try:
        values = tuple(groups)
    except TypeError as exc:
        raise ValueError("groups must be iterable") from exc
    if any(not isinstance(group, str) or not group.strip() for group in values):
        raise ValueError("groups must contain non-blank strings")
    ordered = tuple(sorted(set(values)))
    if not ordered:
        raise ValueError("groups must be non-empty")

    shuffled = np.asarray(ordered, dtype=object)[np.random.default_rng(seed).permutation(len(ordered))]
    train_end = int(np.floor(len(ordered) * train))
    validation_end = int(np.floor(len(ordered) * (train + validation)))
    result = GroupSplit(
        tuple(sorted(shuffled[:train_end].tolist())),
        tuple(sorted(shuffled[train_end:validation_end].tolist())),
        tuple(sorted(shuffled[validation_end:].tolist())),
    )
    assert_group_disjoint(result)
    return result


def assert_group_disjoint(partitions: GroupSplit | Mapping[str, Iterable[str]]) -> None:
    """Fail closed if a group appears in multiple train/validation/test sets."""

    values = partitions.as_dict() if isinstance(partitions, GroupSplit) else partitions
    required = ("train", "validation", "test")
    if any(name not in values for name in required):
        raise GroupOverlapError("partitions must define train, validation, and test")
    sets = {name: set(values[name]) for name in required}
    if any(
        not isinstance(group, str) or not group.strip()
        for partition in sets.values()
        for group in partition
    ):
        raise GroupOverlapError("partition groups must be non-blank strings")
    overlaps = (sets["train"] & sets["validation"]) | (sets["train"] & sets["test"]) | (sets["validation"] & sets["test"])
    if overlaps:
        raise GroupOverlapError("group overlap detected: " + ", ".join(sorted(overlaps)))


split_by_group = partition_groups

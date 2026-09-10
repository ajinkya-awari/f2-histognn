"""Deterministic cohort and case-split contracts for the real-data benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from data.gdc import GDCSlideRecord


class BenchmarkContractError(ValueError):
    """Raised when the frozen benchmark cohort or split is invalid."""


_LABELS = ("LUAD", "LUSC")


@dataclass(frozen=True)
class BenchmarkSplit:
    """Private case records and their canonical identifier-free hash."""

    train: tuple[GDCSlideRecord, ...]
    validation: tuple[GDCSlideRecord, ...]
    test: tuple[GDCSlideRecord, ...]
    split_sha256: str


def _validated_records(records: Iterable[GDCSlideRecord]) -> tuple[GDCSlideRecord, ...]:
    try:
        values = tuple(records)
    except TypeError as exc:
        raise BenchmarkContractError("records must be iterable") from exc
    if not values or any(not isinstance(record, GDCSlideRecord) for record in values):
        raise BenchmarkContractError("records must contain GDCSlideRecord values")
    file_ids = [record.file_id for record in values]
    if len(file_ids) != len(set(file_ids)):
        raise BenchmarkContractError("duplicate file UUIDs are not allowed")
    if any(record.label not in _LABELS for record in values):
        raise BenchmarkContractError("record label must be LUAD or LUSC")
    labels_by_case: dict[str, set[str]] = defaultdict(set)
    for record in values:
        labels_by_case[record.case_id].add(record.label)
    if any(len(labels) != 1 for labels in labels_by_case.values()):
        raise BenchmarkContractError("case labels must be unambiguous")
    return values


def _positive_integer(value: int, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise BenchmarkContractError(f"{field} must be a positive integer")
    return value


def _seed(value: int) -> int:
    if type(value) is not int:
        raise BenchmarkContractError("seed must be an integer")
    return value


def select_balanced_cases(
    records: Iterable[GDCSlideRecord], *, per_class: int, seed: int
) -> tuple[GDCSlideRecord, ...]:
    """Select one deterministic slide for each seeded, balanced case."""

    values = _validated_records(records)
    count = _positive_integer(per_class, "per_class")
    rng = np.random.default_rng(_seed(seed))
    selected: list[GDCSlideRecord] = []
    for label in _LABELS:
        by_case: dict[str, list[GDCSlideRecord]] = defaultdict(list)
        for record in values:
            if record.label == label:
                by_case[record.case_id].append(record)
        case_ids = sorted(by_case)
        if len(case_ids) < count:
            raise BenchmarkContractError(f"insufficient {label} cases for balanced selection")
        indices = rng.permutation(len(case_ids))[:count]
        for index in indices:
            selected.append(sorted(by_case[case_ids[int(index)]])[0])
    return tuple(sorted(selected))


def _split_hash(partitions: dict[str, tuple[GDCSlideRecord, ...]]) -> str:
    private_manifest = {
        name: [
            {
                "case_id": record.case_id,
                "file_id": record.file_id,
                "label": record.label,
                "md5sum": record.md5sum,
            }
            for record in records
        ]
        for name, records in partitions.items()
    }
    encoded = json.dumps(private_manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def stratified_case_split(
    records: Iterable[GDCSlideRecord],
    *,
    seed: int,
    train_per_class: int,
    validation_per_class: int,
    test_per_class: int,
) -> BenchmarkSplit:
    """Create the exact frozen, class-stratified, case-disjoint split."""

    values = _validated_records(records)
    requested = tuple(
        _positive_integer(value, name)
        for value, name in (
            (train_per_class, "train_per_class"),
            (validation_per_class, "validation_per_class"),
            (test_per_class, "test_per_class"),
        )
    )
    rng = np.random.default_rng(_seed(seed))
    parts: dict[str, list[GDCSlideRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    total = sum(requested)
    for label in _LABELS:
        class_records = sorted(record for record in values if record.label == label)
        if len(class_records) != total:
            raise BenchmarkContractError(
                f"requested split requires exactly {total} {label} cases"
            )
        shuffled = [class_records[int(index)] for index in rng.permutation(total)]
        train_end = requested[0]
        validation_end = train_end + requested[1]
        parts["train"].extend(shuffled[:train_end])
        parts["validation"].extend(shuffled[train_end:validation_end])
        parts["test"].extend(shuffled[validation_end:])
    canonical = {name: tuple(sorted(part)) for name, part in parts.items()}
    case_sets = [{record.case_id for record in part} for part in canonical.values()]
    if case_sets[0] & case_sets[1] or case_sets[0] & case_sets[2] or case_sets[1] & case_sets[2]:
        raise BenchmarkContractError("case overlap detected across benchmark partitions")
    return BenchmarkSplit(
        train=canonical["train"],
        validation=canonical["validation"],
        test=canonical["test"],
        split_sha256=_split_hash(canonical),
    )


def sanitized_split_summary(split: BenchmarkSplit) -> dict[str, object]:
    """Return aggregate split evidence without case, slide, or file identifiers."""

    if not isinstance(split, BenchmarkSplit):
        raise BenchmarkContractError("split must be a BenchmarkSplit")
    partitions = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    return {
        "split_sha256": split.split_sha256,
        "case_counts": {name: len(records) for name, records in partitions.items()},
        "class_counts": {
            name: dict(sorted(Counter(record.label for record in records).items()))
            for name, records in partitions.items()
        },
    }

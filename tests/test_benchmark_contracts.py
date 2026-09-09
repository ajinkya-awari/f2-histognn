from __future__ import annotations

from dataclasses import replace

import pytest

from data.benchmark import (
    BenchmarkContractError,
    sanitized_split_summary,
    select_balanced_cases,
    stratified_case_split,
)
from data.gdc import GDCSlideRecord


def benchmark_records(per_class: int = 60) -> tuple[GDCSlideRecord, ...]:
    records = []
    for label in ("LUAD", "LUSC"):
        project = f"TCGA-{label}"
        for index in range(per_class):
            case = f"TCGA-{label[-2:]}-{index:04d}"
            for slide in ("A", "B"):
                records.append(
                    GDCSlideRecord(
                        file_id=f"{label}-{index}-{slide}",
                        file_name=f"{case}-01Z-00-DX{slide}.svs",
                        file_size=1000 + index,
                        md5sum=f"{index:032x}",
                        project_id=project,
                        label=label,
                        case_id=f"uuid-{label}-{index}",
                        case_submitter_id=case,
                        slide_submitter_id=f"{case}-01Z-00-DX{slide}",
                    )
                )
    return tuple(records)


def test_balanced_case_selection_is_order_independent_and_one_slide_per_case():
    records = benchmark_records()

    first = select_balanced_cases(records, per_class=50, seed=17)
    second = select_balanced_cases(tuple(reversed(records)), per_class=50, seed=17)

    assert first == second
    assert len(first) == 100
    assert len({record.case_id for record in first}) == 100
    assert {label: sum(record.label == label for record in first) for label in ("LUAD", "LUSC")} == {
        "LUAD": 50,
        "LUSC": 50,
    }


def test_stratified_case_split_has_exact_counts_no_overlap_and_stable_hash():
    selected = select_balanced_cases(benchmark_records(), per_class=50, seed=17)

    first = stratified_case_split(
        selected,
        seed=17,
        train_per_class=30,
        validation_per_class=10,
        test_per_class=10,
    )
    second = stratified_case_split(
        tuple(reversed(selected)),
        seed=17,
        train_per_class=30,
        validation_per_class=10,
        test_per_class=10,
    )

    assert first == second
    assert [len(first.train), len(first.validation), len(first.test)] == [60, 20, 20]
    partitions = [
        {record.case_id for record in first.train},
        {record.case_id for record in first.validation},
        {record.case_id for record in first.test},
    ]
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert len(first.split_sha256) == 64

    summary = sanitized_split_summary(first)
    assert summary == {
        "split_sha256": first.split_sha256,
        "case_counts": {"train": 60, "validation": 20, "test": 20},
        "class_counts": {
            "train": {"LUAD": 30, "LUSC": 30},
            "validation": {"LUAD": 10, "LUSC": 10},
            "test": {"LUAD": 10, "LUSC": 10},
        },
    }
    assert "TCGA-" not in repr(summary)


def test_benchmark_selection_and_split_fail_closed_on_invalid_records():
    records = benchmark_records(per_class=2)
    with pytest.raises(BenchmarkContractError, match="positive integer"):
        select_balanced_cases(records, per_class=0, seed=17)
    with pytest.raises(BenchmarkContractError, match="insufficient"):
        select_balanced_cases(records, per_class=3, seed=17)

    duplicate_file = replace(records[1], file_id=records[0].file_id)
    with pytest.raises(BenchmarkContractError, match="duplicate file"):
        select_balanced_cases((records[0], duplicate_file), per_class=1, seed=17)

    invalid_label = replace(records[0], label="OTHER")
    with pytest.raises(BenchmarkContractError, match="label"):
        select_balanced_cases((invalid_label, records[-1]), per_class=1, seed=17)

    selected = select_balanced_cases(records, per_class=2, seed=17)
    with pytest.raises(BenchmarkContractError, match="requested split"):
        stratified_case_split(
            selected,
            seed=17,
            train_per_class=1,
            validation_per_class=1,
            test_per_class=1,
        )

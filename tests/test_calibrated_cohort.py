"""Synthetic, metadata-only eligibility; no images, nuclei or outcomes involved."""
import importlib
import importlib.util

import pytest

from data.benchmark import select_balanced_cases
from tests.test_benchmark_contracts import benchmark_records


def selector():
    assert importlib.util.find_spec('data.calibrated_cohort') is not None, 'calibrated metadata selection is missing'
    return importlib.import_module('data.calibrated_cohort').select_calibrated_cases


def test_all_calibrated_metadata_preserves_original_seeded_cohort():
    records = benchmark_records(6)
    result = selector()(records, lambda record: {'objective_power': 20.0}, per_class=3, seed=17)
    assert result.records == select_balanced_cases(records, per_class=3, seed=17)
    assert result.excluded_cases == {'LUAD': 0, 'LUSC': 0}
    assert len(result.objective_by_file) == 6


def test_unknown_calibration_excludes_case_without_guessing_or_changing_class_balance():
    records = benchmark_records(6)
    original = select_balanced_cases(records, per_class=3, seed=17)
    rejected_case = original[0].case_id
    probe = lambda record: {'objective_power': None if record.case_id == rejected_case else 40.0}
    first = selector()(records, probe, per_class=3, seed=17)
    reverse = selector()(tuple(reversed(records)), probe, per_class=3, seed=17)
    assert first.records == reverse.records
    assert rejected_case not in {record.case_id for record in first.records}
    assert sum(record.label == 'LUAD' for record in first.records) == 3
    assert sum(record.label == 'LUSC' for record in first.records) == 3
    assert sum(first.excluded_cases.values()) == 1


def test_other_calibrated_slide_of_same_case_is_preferred_over_case_replacement():
    records = benchmark_records(3)
    first_file = select_balanced_cases(records, per_class=3, seed=17)[0]
    result = selector()(records, lambda record: {'objective_power': None if record.file_id == first_file.file_id else 20.0}, per_class=3, seed=17)
    assert first_file.file_id not in {record.file_id for record in result.records}
    assert first_file.case_id in {record.case_id for record in result.records}
    assert sum(result.excluded_cases.values()) == 0


def test_insufficient_calibrated_cases_fail_closed():
    with pytest.raises(ValueError, match='insufficient calibrated'):
        selector()(benchmark_records(3), lambda record: {'objective_power': None}, per_class=3, seed=17)


@pytest.mark.parametrize('value', [float('nan'), 0, 100, '40', True])
def test_invalid_calibration_probe_cannot_make_a_case_eligible(value):
    with pytest.raises(ValueError, match='objective'):
        selector()(benchmark_records(3), lambda record: {'objective_power': value}, per_class=3, seed=17)

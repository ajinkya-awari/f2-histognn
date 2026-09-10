"""Seeded selection using calibration metadata, never image/model outcomes."""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
import math

import numpy as np

from data.benchmark import _validated_records
from data.gdc import GDCSlideRecord


@dataclass(frozen=True)
class CalibratedCohort:
    records: tuple[GDCSlideRecord, ...]
    objective_by_file: Mapping[str, float] = field(repr=False)
    excluded_cases: Mapping[str, int]
    probed_slides: int


def select_calibrated_cases(records, probe, *, per_class: int, seed: int) -> CalibratedCohort:
    """Visit seed-ranked cases and accept the first calibrated slide per case.

    Calibration is a prespecified source eligibility gate. Probe errors propagate;
    a transient network failure cannot be treated as missing calibration. Within
    a case, sorted file UUID order is retained; no outcome or nuclei input exists.
    """
    values = _validated_records(records)
    if type(per_class) is not int or per_class <= 0 or type(seed) is not int:
        raise ValueError('positive per_class and integer seed are required')
    rng = np.random.default_rng(seed)
    selected = []
    objectives = {}
    excluded = {'LUAD': 0, 'LUSC': 0}
    probed = 0
    for label in ('LUAD', 'LUSC'):
        by_case = defaultdict(list)
        for record in values:
            if record.label == label:
                by_case[record.case_id].append(record)
        cases = sorted(by_case)
        accepted = 0
        for index in rng.permutation(len(cases)):
            chosen = None
            for record in sorted(by_case[cases[int(index)]]):
                metadata = probe(record)
                probed += 1
                if not isinstance(metadata, Mapping) or 'objective_power' not in metadata:
                    raise ValueError('objective calibration probe returned invalid metadata')
                objective = metadata['objective_power']
                if objective is None:
                    continue
                if type(objective) not in (int, float) or not math.isfinite(objective) or not 10 <= objective <= 80:
                    raise ValueError('objective calibration is invalid')
                chosen = record
                objectives[record.file_id] = float(objective)
                break
            if chosen is None:
                excluded[label] += 1
                continue
            selected.append(chosen)
            accepted += 1
            if accepted == per_class:
                break
        if accepted != per_class:
            raise ValueError(f'insufficient calibrated {label} cases')
    return CalibratedCohort(tuple(sorted(selected)), objectives, excluded, probed)

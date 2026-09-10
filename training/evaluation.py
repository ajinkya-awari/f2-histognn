"""Case-level aggregation and frozen binary benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class BenchmarkEvaluationError(ValueError):
    """Raised when predictions cannot support the frozen metric contract."""


@dataclass(frozen=True)
class CasePredictions:
    """Private case keys with one probability row and label per case."""

    case_keys: tuple[str, ...]
    probabilities: np.ndarray
    labels: np.ndarray


def _labels(values: Any, count: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.shape[0] != count:
        raise BenchmarkEvaluationError("labels must contain one value per row")
    if labels.dtype.kind not in "iub" or np.any((labels != 0) & (labels != 1)):
        raise BenchmarkEvaluationError("labels must contain only LUAD=0 or LUSC=1")
    return labels.astype(np.int64, copy=False)


def _probabilities(values: Any, count: int | None = None) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1:] != (2,):
        raise BenchmarkEvaluationError("probabilities must have shape [N, 2]")
    if count is not None and probabilities.shape[0] != count:
        raise BenchmarkEvaluationError("probabilities and labels must have the same N")
    if not np.all(np.isfinite(probabilities)):
        raise BenchmarkEvaluationError("probabilities must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise BenchmarkEvaluationError("probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7):
        raise BenchmarkEvaluationError("probability rows must sum to 1")
    return probabilities


def aggregate_case_probabilities(
    tile_logits: Any,
    *,
    case_keys: Any,
    labels: Any,
    tiles_per_case: int = 4,
) -> CasePredictions:
    """Softmax tile logits and average exactly four predictions per private case."""

    logits = np.asarray(tile_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1:] != (2,):
        raise BenchmarkEvaluationError("tile logits must have shape [N, 2]")
    if not np.all(np.isfinite(logits)):
        raise BenchmarkEvaluationError("tile logits must be finite")
    if type(tiles_per_case) is not int or tiles_per_case <= 0:
        raise BenchmarkEvaluationError("tiles_per_case must be a positive integer")
    try:
        keys = tuple(case_keys)
    except TypeError as exc:
        raise BenchmarkEvaluationError("case_keys must be iterable") from exc
    if len(keys) != logits.shape[0] or any(not isinstance(key, str) or not key for key in keys):
        raise BenchmarkEvaluationError("case_keys must contain one non-blank key per tile")
    tile_labels = _labels(labels, logits.shape[0])
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    tile_probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    case_probabilities: list[np.ndarray] = []
    case_labels: list[int] = []
    ordered_keys = tuple(sorted(set(keys)))
    for key in ordered_keys:
        indices = np.flatnonzero(np.asarray(keys, dtype=object) == key)
        if len(indices) != tiles_per_case:
            raise BenchmarkEvaluationError(f"each case must contain exactly {tiles_per_case} tiles")
        unique_labels = np.unique(tile_labels[indices])
        if len(unique_labels) != 1:
            raise BenchmarkEvaluationError("case tiles contain conflicting labels")
        case_probabilities.append(tile_probabilities[indices].mean(axis=0))
        case_labels.append(int(unique_labels[0]))
    return CasePredictions(
        ordered_keys,
        np.asarray(case_probabilities, dtype=np.float64),
        np.asarray(case_labels, dtype=np.int64),
    )


def binary_case_metrics(probabilities: Any, labels: Any) -> dict[str, float | str]:
    """Compute the prespecified case-level metrics with LUSC as positive class."""

    values = _probabilities(probabilities)
    truth = _labels(labels, values.shape[0])
    if set(truth.tolist()) != {0, 1}:
        raise BenchmarkEvaluationError("metrics require both classes")
    predictions = np.argmax(values, axis=1)
    accuracy = float(np.mean(predictions == truth))
    f1_values = []
    for label in (0, 1):
        true_positive = int(np.sum((predictions == label) & (truth == label)))
        false_positive = int(np.sum((predictions == label) & (truth != label)))
        false_negative = int(np.sum((predictions != label) & (truth == label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    positive = values[:, 1]
    positive_scores = positive[truth == 1]
    negative_scores = positive[truth == 0]
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    auroc = float(np.mean((comparisons > 0).astype(float) + 0.5 * (comparisons == 0)))
    auprc = 0.0
    previous_recall = 0.0
    positive_count = int(np.sum(truth == 1))
    for threshold in np.unique(positive)[::-1]:
        predicted_positive = positive >= threshold
        true_positive = int(np.sum(predicted_positive & (truth == 1)))
        false_positive = int(np.sum(predicted_positive & (truth == 0)))
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        auprc += (recall - previous_recall) * precision
        previous_recall = recall
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "auroc": auroc,
        "auprc": auprc,
        "positive_class": "LUSC",
    }


def stratified_bootstrap_intervals(
    probabilities: Any,
    labels: Any,
    *,
    resamples: int = 2000,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Return deterministic 95% percentile intervals from stratified case resamples."""

    values = _probabilities(probabilities)
    truth = _labels(labels, values.shape[0])
    if set(truth.tolist()) != {0, 1}:
        raise BenchmarkEvaluationError("bootstrap requires both classes")
    if type(resamples) is not int or resamples <= 0:
        raise BenchmarkEvaluationError("resamples must be a positive integer")
    if type(seed) is not int:
        raise BenchmarkEvaluationError("seed must be an integer")
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(truth == label) for label in (0, 1)]
    samples = {name: [] for name in ("accuracy", "macro_f1", "auroc", "auprc")}
    for _ in range(resamples):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in class_indices]
        )
        metrics = binary_case_metrics(values[indices], truth[indices])
        for name in samples:
            samples[name].append(float(metrics[name]))
    return {
        name: tuple(float(value) for value in np.percentile(results, [2.5, 97.5]))
        for name, results in samples.items()
    }

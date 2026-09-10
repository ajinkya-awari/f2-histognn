"""Deterministic training contracts; execution remains data/compute gated."""

from .evaluation import (
    BenchmarkEvaluationError,
    CasePredictions,
    aggregate_case_probabilities,
    binary_case_metrics,
    stratified_bootstrap_intervals,
)
from .benchmark import (
    FROZEN_POLICY,
    FrozenBenchmarkPolicy,
    frozen_run_matrix,
    partition_private_graphs,
    run_frozen_benchmark,
)

from .loop import (
    TrainingConfig,
    TrainingResult,
    build_provenance,
    evaluate_loss,
    fit_model,
    seed_everything,
)

__all__ = [
    "BenchmarkEvaluationError",
    "CasePredictions",
    "FROZEN_POLICY",
    "FrozenBenchmarkPolicy",
    "TrainingConfig",
    "TrainingResult",
    "build_provenance",
    "aggregate_case_probabilities",
    "binary_case_metrics",
    "evaluate_loss",
    "fit_model",
    "frozen_run_matrix",
    "partition_private_graphs",
    "run_frozen_benchmark",
    "seed_everything",
    "stratified_bootstrap_intervals",
]

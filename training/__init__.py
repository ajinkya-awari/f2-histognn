"""Deterministic training contracts; execution remains data/compute gated."""

from .loop import (
    TrainingConfig,
    TrainingResult,
    build_provenance,
    evaluate_loss,
    fit_model,
    seed_everything,
)

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "build_provenance",
    "evaluate_loss",
    "fit_model",
    "seed_everything",
]

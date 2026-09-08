"""Test-only graph explanation APIs and diagnostics."""

from .diagnostics import (
    compute_faithfulness,
    compute_stability,
    faithfulness_score,
    stability_score,
)
from .wrapper import (
    ExplanationResult,
    GraphExplainerWrapper,
    GraphExplanationWrapper,
)

__all__ = [
    "ExplanationResult",
    "GraphExplanationWrapper",
    "GraphExplainerWrapper",
    "faithfulness_score",
    "stability_score",
    "compute_faithfulness",
    "compute_stability",
]

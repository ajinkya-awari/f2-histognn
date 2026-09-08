"""Small perturbation-based explanation diagnostics for synthetic graphs."""

from itertools import combinations
import math
from typing import Iterable

import torch
from torch.nn import functional as F

from models.base import EDGE_FEATURES, validate_graph_inputs

from .wrapper import ExplanationResult, GraphExplanationWrapper


def _edge_tensor(x, edge_index, edge_attr):
    if edge_attr is None:
        return x.new_zeros((edge_index.size(1), EDGE_FEATURES))
    return edge_attr


def _log_prob(model, x, edge_index, edge_attr):
    edge_tensor = _edge_tensor(x, edge_index, edge_attr)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    validate_graph_inputs(x, edge_index, edge_tensor, batch)
    logits = model(x, edge_index, edge_tensor, batch)
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise ValueError("diagnostic model must return finite logits shaped [1, 2]")
    return F.log_softmax(logits, dim=-1), edge_tensor


def faithfulness_score(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
    explanation: ExplanationResult | None = None,
    fraction: float = 0.25,
) -> float:
    """Measure target log-probability change after perturbing high-mask inputs."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if explanation is None:
        explanation = GraphExplanationWrapper(model, seed=0).explain(
            x, edge_index, edge_attr
    )
    node_count = max(1, int(x.size(0) * fraction))
    edge_count = max(1, int(edge_index.size(1) * fraction)) if edge_index.size(1) else 0
    node_indices = explanation.node_mask.topk(node_count).indices
    perturbed_x = x.detach().clone()
    perturbed_x[node_indices] = 0
    base_log_prob, base_edge_tensor = _log_prob(model, x, edge_index, edge_attr)
    edge_tensor = _edge_tensor(x, edge_index, edge_attr).detach().clone()
    if edge_count:
        edge_indices = explanation.edge_mask.topk(edge_count).indices
        edge_tensor[edge_indices] = 0

    if edge_attr is None:
        edge_tensor = base_edge_tensor
    perturbed_log_prob, _ = _log_prob(model, perturbed_x, edge_index, edge_tensor)
    target = explanation.target_class
    score = base_log_prob[0, target] - perturbed_log_prob[0, target]
    return float(score.detach().item())


def _normalized_mask(result: ExplanationResult) -> torch.Tensor:
    mask = torch.cat((result.node_mask.flatten(), result.edge_mask.flatten()))
    return mask / mask.norm().clamp_min(torch.finfo(mask.dtype).eps)


def stability_score(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
    seeds: Iterable[int] = (0, 1),
    perturbation_scale: float = 1e-3,
) -> float:
    """Compare fixed-target masks under bounded, seeded input perturbations."""

    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("seeds must contain at least one value")
    if not math.isfinite(perturbation_scale) or perturbation_scale < 0.0:
        raise ValueError("perturbation_scale must be finite and non-negative")
    baseline = GraphExplanationWrapper(model, seed=seed_list[0]).explain(
        x, edge_index, edge_attr
    )
    results = []
    for seed in seed_list:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        perturbed_x = x + noise * perturbation_scale
        results.append(
            GraphExplanationWrapper(model, seed=seed).explain(
                perturbed_x,
                edge_index,
                edge_attr,
                target_class=baseline.target_class,
            )
        )
    if len(results) == 1:
        return 1.0
    similarities = []
    for left, right in combinations(results, 2):
        distance = (_normalized_mask(left) - _normalized_mask(right)).abs().mean()
        similarities.append((1.0 - distance).clamp(0.0, 1.0))
    return float(torch.stack(similarities).mean().item())


compute_faithfulness = faithfulness_score
compute_stability = stability_score

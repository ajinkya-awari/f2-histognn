"""Test-only, graph-level explanation wrapper."""

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from models.base import EDGE_FEATURES, validate_graph_inputs


@dataclass(frozen=True)
class ExplanationResult:
    """Masks and metadata produced for one graph."""

    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    log_prob: torch.Tensor
    target_class: int
    metadata: dict[str, Any]

    def __getitem__(self, key: str):
        return getattr(self, key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_mask": self.node_mask,
            "edge_mask": self.edge_mask,
            "log_prob": self.log_prob,
            "target_class": self.target_class,
            "metadata": self.metadata,
        }


class GraphExplanationWrapper:
    """Apply input-gradient explanations with single-graph model semantics."""

    def __init__(self, model, seed: int = 0, config: dict[str, Any] | None = None) -> None:
        self.model = model
        self.seed = int(seed)
        self.config = dict(config or {"method": "input-gradient", "task": "graph"})

    def _edge_tensor(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr):
        if (
            not isinstance(edge_index, torch.Tensor)
            or edge_index.ndim != 2
            or edge_index.size(0) != 2
        ):
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_attr is None:
            return x.new_zeros((edge_index.size(1), EDGE_FEATURES))
        if not isinstance(edge_attr, torch.Tensor):
            raise ValueError("edge_attr must be a tensor or None")
        return edge_attr

    def explain(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        target_class: int | None = None,
    ) -> ExplanationResult:
        """Return node/edge masks for one graph; no graph index is accepted."""

        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.size(0) == 0:
            raise ValueError("x must contain a non-empty graph")
        edge_tensor = self._edge_tensor(x, edge_index, edge_attr)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        validate_graph_inputs(x, edge_index, edge_tensor, batch)

        x_var = x.detach().clone().requires_grad_(True)
        edge_var = edge_tensor.detach().clone().requires_grad_(True)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                try:
                    logits = self.model(x_var, edge_index, edge_var, batch)
                except TypeError as exc:
                    raise ValueError(
                        "model must accept (x, edge_index, edge_attr, batch)"
                    ) from exc
                if logits.ndim != 2 or logits.shape != (1, 2):
                    raise ValueError("graph explanation requires model logits shaped [1, 2]")
                if not torch.isfinite(logits).all():
                    raise ValueError("model logits must contain only finite values")
                log_prob = F.log_softmax(logits, dim=-1)
                chosen_class = (
                    int(log_prob.argmax(dim=-1).item())
                    if target_class is None
                    else int(target_class)
                )
                if chosen_class not in (0, 1):
                    raise ValueError("target_class must be 0 or 1")
                (target_log_prob,) = torch.autograd.grad(
                    log_prob[0, chosen_class],
                    (x_var,),
                    retain_graph=True,
                    allow_unused=True,
                )
                edge_grad = torch.autograd.grad(
                    log_prob[0, chosen_class],
                    (edge_var,),
                    allow_unused=True,
                )[0]
        finally:
            self.model.train(was_training)

        if target_log_prob is None:
            node_mask = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        else:
            node_mask = target_log_prob.abs().sum(dim=-1).detach()
        if edge_grad is None:
            edge_mask = torch.zeros(edge_tensor.size(0), device=x.device, dtype=x.dtype)
        else:
            edge_mask = edge_grad.abs().sum(dim=-1).detach()
        if not torch.isfinite(node_mask).all() or not torch.isfinite(edge_mask).all():
            raise ValueError("explanation masks must contain only finite values")

        metadata = {
            "seed": self.seed,
            "config": dict(self.config),
            "task": "graph",
            "semantics": "log-probability",
        }
        return ExplanationResult(
            node_mask=node_mask,
            edge_mask=edge_mask,
            log_prob=log_prob.detach(),
            target_class=chosen_class,
            metadata=metadata,
        )

    __call__ = explain


GraphExplainerWrapper = GraphExplanationWrapper

"""Shared deterministic training/evaluation harness for authorized runs."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TrainingConfig:
    """The training policy shared by every model in one benchmark run."""

    seed: int = 0
    split_hash: str = ""
    preprocessing_hash: str = ""
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        for name in ("split_hash", "preprocessing_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")

    @property
    def stopping_policy(self) -> dict[str, int | float]:
        return {
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": float(self.min_delta),
        }


@dataclass(frozen=True)
class TrainingResult:
    """Recorded training state; no scientific metric is inferred here."""

    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    best_epoch: int
    best_validation_loss: float
    provenance: Mapping[str, Any]
    best_state_dict: Mapping[str, torch.Tensor] = field(repr=False)


def seed_everything(seed: int) -> int:
    """Seed all local RNGs used by the harness and return the seed."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def build_provenance(
    config: TrainingConfig,
    *,
    source_id: str,
    source_checksum: str,
    schema_version: str,
    code_revision: str,
    environment: Mapping[str, str],
    compute_location: str,
) -> dict[str, Any]:
    """Build required provenance metadata without manufacturing metrics."""

    fields = {
        "source_id": source_id,
        "source_checksum": source_checksum,
        "schema_version": schema_version,
        "code_revision": code_revision,
        "compute_location": compute_location,
    }
    if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
        raise ValueError("provenance identity fields must be non-blank strings")
    if not isinstance(environment, Mapping) or not environment:
        raise ValueError("environment must be a non-empty mapping")
    return {
        **fields,
        "environment": dict(environment),
        "split_hash": config.split_hash,
        "preprocessing_hash": config.preprocessing_hash,
        "seed": config.seed,
        "stopping_policy": config.stopping_policy,
    }


def _batch_inputs(batch: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    required = ("x", "edge_index", "edge_attr", "y")
    if any(not hasattr(batch, name) for name in required):
        raise ValueError("each batch must provide x, edge_index, edge_attr, and y")
    x = batch.x.to(device)
    edge_index = batch.edge_index.to(device)
    edge_attr = batch.edge_attr.to(device)
    labels = batch.y.to(device).reshape(-1).long()
    graph_batch = getattr(batch, "batch", None)
    if graph_batch is None:
        graph_batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    else:
        graph_batch = graph_batch.to(device)
    graph_count = int(graph_batch.max().item()) + 1 if graph_batch.numel() else 0
    if labels.numel() != graph_count:
        raise ValueError("y must contain one label per graph")
    if labels.numel() and torch.any((labels < 0) | (labels > 1)):
        raise ValueError("y must contain only LUAD/LUSC class ids 0 or 1")
    return x, edge_index, edge_attr, graph_batch, labels


def _resolve_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this runtime")
    return resolved


def evaluate_loss(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    device: torch.device | str = "cpu",
) -> float:
    """Evaluate mean cross-entropy loss on an iterable of graph batches."""

    device = _resolve_device(device)
    model.to(device)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    graph_count = 0
    with torch.no_grad():
        for batch in batches:
            x, edge_index, edge_attr, graph_batch, labels = _batch_inputs(batch, device)
            logits = model(x, edge_index, edge_attr, graph_batch)
            if logits.shape != (labels.numel(), 2):
                raise ValueError("model logits must have shape [num_graphs, 2]")
            total_loss += float(
                nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            )
            graph_count += labels.numel()
    model.train(was_training)
    if graph_count == 0:
        raise ValueError("batches must be non-empty")
    return total_loss / graph_count


def fit_model(
    model: nn.Module,
    train_batches: Iterable[Any],
    validation_batches: Iterable[Any],
    config: TrainingConfig,
    provenance: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> TrainingResult:
    """Fit one model under the shared seed, preprocessing, and stop policy."""

    train_data = list(train_batches)
    validation_data = list(validation_batches)
    if not train_data or not validation_data:
        raise ValueError("train_batches and validation_batches must be non-empty")
    seed_everything(config.seed)
    device = _resolve_device(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(config.max_epochs):
        model.train()
        total_train_loss = 0.0
        train_graph_count = 0
        for batch in train_data:
            inputs = _batch_inputs(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs[:4])
            if logits.shape != (inputs[4].numel(), 2):
                raise ValueError("model logits must have shape [num_graphs, 2]")
            loss = nn.functional.cross_entropy(logits, inputs[4], reduction="sum")
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.detach().item())
            train_graph_count += inputs[4].numel()
        train_loss = total_train_loss / train_graph_count
        validation_loss = evaluate_loss(model, validation_data, device=device)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    return TrainingResult(
        train_loss=tuple(train_losses),
        validation_loss=tuple(validation_losses),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        provenance=dict(provenance),
        best_state_dict=best_state,
    )

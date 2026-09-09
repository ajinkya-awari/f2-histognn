"""Frozen run matrix and private graph partitioning for the TCGA benchmark."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data.benchmark import BenchmarkSplit
from data.benchmark_graphs import PrivateGraph
from models import MODEL_REGISTRY
from training.evaluation import (
    aggregate_case_probabilities,
    binary_case_metrics,
    stratified_bootstrap_intervals,
)
from training.loop import TrainingConfig, fit_model


@dataclass(frozen=True)
class FrozenBenchmarkPolicy:
    models: tuple[str, ...] = ("gcn", "graphsage", "gat", "graphgps")
    seeds: tuple[int, ...] = (17, 29, 43)
    hidden_dim: int = 32
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    tiles_per_case: int = 4
    bootstrap_resamples: int = 2000


FROZEN_POLICY = FrozenBenchmarkPolicy()


def frozen_run_matrix() -> tuple[tuple[str, int], ...]:
    """Return the prespecified four-architecture, three-seed execution order."""

    return tuple(
        (model, seed) for model in FROZEN_POLICY.models for seed in FROZEN_POLICY.seeds
    )


def _private_case_key(case_id: str) -> str:
    return hashlib.sha256(case_id.encode()).hexdigest()


def partition_private_graphs(
    graphs: Iterable[PrivateGraph],
    split: BenchmarkSplit,
    *,
    tiles_per_case: int = 4,
) -> Mapping[str, tuple[PrivateGraph, ...]]:
    """Assign private graphs to their frozen case partition and fail on drift."""

    if not isinstance(split, BenchmarkSplit):
        raise ValueError("split must be a BenchmarkSplit")
    if type(tiles_per_case) is not int or tiles_per_case <= 0:
        raise ValueError("tiles_per_case must be a positive integer")
    values = tuple(graphs)
    if not values or any(not isinstance(graph, PrivateGraph) for graph in values):
        raise ValueError("graphs must contain PrivateGraph values")
    partitions = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    case_lookup: dict[str, tuple[str, int]] = {}
    for partition, records in partitions.items():
        for record in records:
            key = _private_case_key(record.case_id)
            if key in case_lookup:
                raise ValueError("case overlap detected across benchmark partitions")
            case_lookup[key] = (partition, 0 if record.label == "LUAD" else 1)
    output: dict[str, list[PrivateGraph]] = {name: [] for name in partitions}
    for graph in values:
        if graph.case_key not in case_lookup:
            raise ValueError("graph case is absent from the frozen split")
        partition, expected_label = case_lookup[graph.case_key]
        if graph.label != expected_label:
            raise ValueError("graph label conflicts with the frozen split")
        output[partition].append(graph)
    counts = Counter(graph.case_key for graph in values)
    if set(counts) != set(case_lookup) or any(
        count != tiles_per_case for count in counts.values()
    ):
        raise ValueError(f"each split case must contain exactly {tiles_per_case} graphs")
    return {
        name: tuple(sorted(partition_graphs, key=lambda graph: graph.tile_key))
        for name, partition_graphs in output.items()
    }


def _pyg_data(graph: PrivateGraph) -> Data:
    return Data(
        x=torch.tensor(graph.features, dtype=torch.float32),
        edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(graph.edge_attr, dtype=torch.float32),
        y=torch.tensor([graph.label], dtype=torch.long),
    )


def _predict_graphs(
    model: torch.nn.Module,
    graphs: tuple[PrivateGraph, ...],
    device: torch.device,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    model.eval()
    logits = []
    with torch.no_grad():
        for graph in graphs:
            data = _pyg_data(graph).to(device)
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
            output = model(data.x, data.edge_index, data.edge_attr, batch)
            if output.shape != (1, 2) or not torch.isfinite(output).all():
                raise RuntimeError("benchmark model output contract failed")
            logits.append(output.cpu().numpy()[0])
    return (
        np.asarray(logits, dtype=np.float64),
        tuple(graph.case_key for graph in graphs),
        np.asarray([graph.label for graph in graphs], dtype=np.int64),
    )


def run_frozen_benchmark(
    partitions: Mapping[str, tuple[PrivateGraph, ...]],
    *,
    split_sha256: str,
    graph_sha256: str,
    provenance: Mapping[str, object],
    device: str = "cuda",
    policy: FrozenBenchmarkPolicy = FROZEN_POLICY,
) -> dict[str, object]:
    """Train every frozen model/seed and return only complete aggregate results."""

    if set(partitions) != {"train", "validation", "test"}:
        raise ValueError("partitions must contain train, validation, and test")
    if not all(partitions.values()):
        raise ValueError("benchmark partitions must be non-empty")
    if not isinstance(split_sha256, str) or len(split_sha256) != 64:
        raise ValueError("split_sha256 must contain 64 characters")
    if not isinstance(graph_sha256, str) or len(graph_sha256) != 64:
        raise ValueError("graph_sha256 must contain 64 characters")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("provenance must be a non-empty mapping")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    train_data = [_pyg_data(graph) for graph in partitions["train"]]
    validation_data = [_pyg_data(graph) for graph in partitions["validation"]]
    model_results: dict[str, dict[str, object]] = {}
    completed_runs = 0
    for model_name in policy.models:
        if model_name not in MODEL_REGISTRY:
            raise RuntimeError(f"required model is unavailable: {model_name}")
        seed_results = []
        for seed in policy.seeds:
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=generator)
            validation_loader = DataLoader(validation_data, batch_size=16, shuffle=False)
            model = MODEL_REGISTRY[model_name](hidden_dim=policy.hidden_dim, seed=seed)
            config = TrainingConfig(
                seed=seed,
                split_hash=split_sha256,
                preprocessing_hash=graph_sha256,
                max_epochs=policy.max_epochs,
                patience=policy.patience,
                min_delta=0.0,
                learning_rate=policy.learning_rate,
                weight_decay=policy.weight_decay,
            )
            training_result = fit_model(
                model,
                train_loader,
                validation_loader,
                config,
                provenance,
                device=resolved_device,
            )
            tile_logits, case_keys, labels = _predict_graphs(
                model, partitions["test"], resolved_device
            )
            cases = aggregate_case_probabilities(
                tile_logits,
                case_keys=case_keys,
                labels=labels,
                tiles_per_case=policy.tiles_per_case,
            )
            metrics = binary_case_metrics(cases.probabilities, cases.labels)
            intervals = stratified_bootstrap_intervals(
                cases.probabilities,
                cases.labels,
                resamples=policy.bootstrap_resamples,
                seed=seed,
            )
            seed_results.append(
                {
                    "seed": seed,
                    "best_epoch": training_result.best_epoch,
                    "epochs_run": len(training_result.train_loss),
                    "best_validation_loss": training_result.best_validation_loss,
                    "metrics": metrics,
                    "confidence_intervals_95": intervals,
                }
            )
            completed_runs += 1
        metric_names = ("accuracy", "macro_f1", "auroc", "auprc")
        model_results[model_name] = {
            "seeds": seed_results,
            "mean_metrics": {
                name: float(np.mean([result["metrics"][name] for result in seed_results]))
                for name in metric_names
            },
        }
    if completed_runs != len(policy.models) * len(policy.seeds):
        raise RuntimeError("benchmark did not complete every frozen model/seed run")
    return {
        "run_count": completed_runs,
        "case_counts": {
            name: len({graph.case_key for graph in graphs})
            for name, graphs in partitions.items()
        },
        "graph_counts": {name: len(graphs) for name, graphs in partitions.items()},
        "models": model_results,
    }

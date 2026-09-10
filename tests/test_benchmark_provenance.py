"""Synthetic-only regressions for retained benchmark provenance."""

from dataclasses import asdict
import re

import numpy as np
import pytest
import torch

from data.benchmark_graphs import PrivateGraph
import training.benchmark as benchmark


def test_selected_state_digest_is_independent_of_mapping_order_and_storage():
    digest = getattr(benchmark, "selected_state_sha256", None)
    assert callable(digest), "selected model state must have a canonical digest"
    first = {"weight": torch.arange(6.0).reshape(2, 3), "count": torch.tensor(1)}
    second = {"count": torch.tensor(1), "weight": first["weight"].t().contiguous().t()}
    assert re.fullmatch(r"[0-9a-f]{64}", digest(first))
    assert digest(first) == digest(second)


@pytest.mark.parametrize("changed", [
    {"other_name": torch.tensor([1.0, 2.0])},
    {"weight": torch.tensor([1.0, 3.0])},
    {"weight": torch.tensor([[1.0, 2.0]])},
    {"weight": torch.tensor([1.0, 2.0], dtype=torch.float64)},
])
def test_selected_state_digest_binds_names_values_shapes_and_dtypes(changed):
    digest = getattr(benchmark, "selected_state_sha256", None)
    assert callable(digest), "selected model state must have a canonical digest"
    assert digest({"weight": torch.tensor([1.0, 2.0])}) != digest(changed)


def test_synthetic_runner_retains_selected_state_hash_loss_curves_and_policy():
    partitions = {}
    for partition in ("train", "validation", "test"):
        partitions[partition] = tuple(
            PrivateGraph(
                tile_key=f"{partition}-{label}-{tile}",
                case_key=f"synthetic-{partition}-{label}",
                label=label,
                features=np.array([[0, 0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]], dtype=float),
                edge_index=np.array([[0, 1], [1, 0]]),
                edge_attr=np.ones((2, 1)),
            )
            for label in (0, 1) for tile in range(4)
        )
    policy = benchmark.FrozenBenchmarkPolicy(
        models=("gcn",), seeds=(17,), hidden_dim=4,
        max_epochs=1, patience=1, bootstrap_resamples=20,
    )
    result = benchmark.run_frozen_benchmark(
        partitions, split_sha256="a" * 64, graph_sha256="b" * 64,
        provenance={"scope": "synthetic-only provenance regression"},
        device="cpu", policy=policy,
    )
    seed = result["models"]["gcn"]["seeds"][0]
    assert "selected_state_sha256" in seed
    assert re.fullmatch(r"[0-9a-f]{64}", seed["selected_state_sha256"])
    assert len(seed["train_loss"]) == len(seed["validation_loss"]) == seed["epochs_run"] == 1
    assert np.isfinite(seed["train_loss"]).all()
    assert seed["validation_loss"][seed["best_epoch"]] == seed["best_validation_loss"]
    assert result["policy"] == asdict(policy)
    assert "synthetic-test-" not in repr(result)

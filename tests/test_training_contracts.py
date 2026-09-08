import pytest
import torch
from torch_geometric.data import Data

from models import GCN
from training import (
    TrainingConfig,
    build_provenance,
    evaluate_loss,
    fit_model,
    seed_everything,
)


def test_training_config_freezes_shared_seed_split_preprocessing_and_stopping_policy():
    config = TrainingConfig(
        seed=17,
        split_hash="split-sha256",
        preprocessing_hash="preprocess-sha256",
        max_epochs=12,
        patience=3,
        min_delta=0.001,
    )

    assert config.seed == 17
    assert config.split_hash == "split-sha256"
    assert config.preprocessing_hash == "preprocess-sha256"
    assert config.stopping_policy == {
        "max_epochs": 12,
        "patience": 3,
        "min_delta": 0.001,
    }


def test_training_config_rejects_invalid_stopping_values():
    kwargs = {"split_hash": "split", "preprocessing_hash": "prep"}
    with pytest.raises(ValueError, match="patience"):
        TrainingConfig(patience=0, **kwargs)
    with pytest.raises(ValueError, match="max_epochs"):
        TrainingConfig(max_epochs=0, **kwargs)


def test_provenance_records_actual_run_inputs_without_inventing_results():
    config = TrainingConfig(seed=5, split_hash="split", preprocessing_hash="prep")

    record = build_provenance(
        config,
        source_id="synthetic-fixture",
        source_checksum="sha256:" + "b" * 64,
        schema_version="1.0",
        code_revision="untracked-local",
        environment={"device": "cpu"},
        compute_location="local-synthetic",
    )

    assert record["source_id"] == "synthetic-fixture"
    assert record["source_checksum"].startswith("sha256:")
    assert record["split_hash"] == "split"
    assert record["preprocessing_hash"] == "prep"
    assert record["seed"] == 5
    assert record["compute_location"] == "local-synthetic"
    assert "metrics" not in record


def test_seed_everything_is_explicitly_callable_for_authorized_runs():
    assert seed_everything(23) == 23


def test_fit_model_records_only_observed_synthetic_losses_under_shared_policy():
    graph = Data(
        x=torch.ones(3, 7),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.ones(3, 1),
        y=torch.tensor([0]),
    )
    config = TrainingConfig(
        seed=3,
        split_hash="split",
        preprocessing_hash="prep",
        max_epochs=2,
        patience=1,
    )
    result = fit_model(
        GCN(hidden_dim=4, seed=3),
        [graph],
        [graph],
        config,
        {"seed": config.seed, "split_hash": config.split_hash},
    )

    assert 1 <= len(result.train_loss) <= config.max_epochs
    assert len(result.train_loss) == len(result.validation_loss)
    assert result.best_epoch < len(result.validation_loss)
    assert torch.isfinite(torch.tensor(result.best_validation_loss))


class _FixedLossModel(torch.nn.Module):
    def forward(self, x, edge_index, edge_attr, batch):
        graph_count = int(batch.max().item()) + 1
        if graph_count == 1:
            return x.new_tensor([[0.0, 0.0]])
        return x.new_tensor([[-2.0, 2.0]]).repeat(graph_count, 1)


def _edgeless_batch(graph_count):
    return Data(
        x=torch.ones(graph_count, 7),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1)),
        y=torch.zeros(graph_count, dtype=torch.long),
        batch=torch.arange(graph_count, dtype=torch.long),
    )


def test_evaluate_loss_weights_each_graph_equally_across_unequal_batches():
    model = _FixedLossModel()
    batches = [_edgeless_batch(1), _edgeless_batch(3)]
    logits = torch.cat((torch.tensor([[0.0, 0.0]]), torch.tensor([[-2.0, 2.0]]).repeat(3, 1)))
    expected = torch.nn.functional.cross_entropy(logits, torch.zeros(4, dtype=torch.long))

    observed = evaluate_loss(model, batches)

    assert observed == pytest.approx(float(expected.item()))


def test_requested_cuda_fails_explicitly_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        evaluate_loss(_FixedLossModel(), [_edgeless_batch(1)], device="cuda")

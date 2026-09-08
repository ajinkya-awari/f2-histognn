import inspect

import pytest
import torch

import models.architectures as architectures

from explanations import (
    ExplanationResult,
    GraphExplanationWrapper,
    faithfulness_score,
    stability_score,
)
from models import GAT, GCN, GraphGPS, GraphSAGE


MODEL_TYPES = (GCN, GraphSAGE, GAT, GraphGPS)


def _batched_graph():
    x = torch.arange(56, dtype=torch.float32).reshape(8, 7) / 10.0
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4],
            [1, 0, 2, 1, 3, 2, 0, 3, 5, 4, 6, 5, 7, 6, 4, 7],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.linspace(0.1, 1.0, edge_index.size(1)).unsqueeze(-1)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    return x, edge_index, edge_attr, batch


def _single_graph():
    x, edge_index, edge_attr, _ = _batched_graph()
    return x[:4], edge_index[:, :8], edge_attr[:8]


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_all_models_return_finite_two_class_graph_logits(model_type):
    x, edge_index, edge_attr, batch = _batched_graph()
    model = model_type(hidden_dim=12, seed=23)

    logits = model(x, edge_index, edge_attr, batch)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_edge_attributes_are_projected_to_hidden_width(model_type):
    model = model_type(hidden_dim=10, seed=7)

    assert model.edge_projection.in_features == 1
    assert model.edge_projection.out_features == 10


def test_same_seed_gives_same_initial_parameters():
    first = GCN(hidden_dim=11, seed=101)
    second = GCN(hidden_dim=11, seed=101)

    assert all(
        torch.equal(left, right)
        for left, right in zip(first.state_dict().values(), second.state_dict().values())
    )


def test_models_reject_wrong_width_and_non_finite_inputs():
    x, edge_index, edge_attr, batch = _batched_graph()
    model = GCN(hidden_dim=8, seed=3)

    with pytest.raises(ValueError, match="seven node features"):
        model(x[:, :6], edge_index, edge_attr, batch)

    with pytest.raises(ValueError, match="one edge feature"):
        model(x, edge_index, edge_attr.repeat(1, 2), batch)

    non_finite = x.clone()
    non_finite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(non_finite, edge_index, edge_attr, batch)


def test_models_reject_unnormalized_edge_features():
    x, edge_index, edge_attr, batch = _batched_graph()
    model = GCN(hidden_dim=8, seed=3)
    unnormalized = edge_attr.clone()
    unnormalized[0, 0] = 1.01

    with pytest.raises(ValueError, match="normalized"):
        model(x, edge_index, unnormalized, batch)


def test_models_reject_non_float_feature_tensors():
    x, edge_index, edge_attr, batch = _batched_graph()
    model = GCN(hidden_dim=8, seed=3)

    with pytest.raises(ValueError, match="floating point"):
        model(x.long(), edge_index, edge_attr, batch)

    with pytest.raises(ValueError, match="floating point"):
        model(x, edge_index, edge_attr.long(), batch)


def test_models_reject_cross_graph_edges_and_non_contiguous_batch_ids():
    x, edge_index, edge_attr, batch = _batched_graph()
    model = GCN(hidden_dim=8, seed=3)
    cross_graph_edges = edge_index.clone()
    cross_graph_edges[:, 0] = torch.tensor([3, 4])

    with pytest.raises(ValueError, match="cross graph"):
        model(x, cross_graph_edges, edge_attr, batch)

    gapped_batch = batch.clone()
    gapped_batch[gapped_batch == 1] = 2
    with pytest.raises(ValueError, match="contiguous"):
        model(x, edge_index, edge_attr, gapped_batch)


def test_graphgps_fails_instead_of_substituting_another_architecture(monkeypatch):
    monkeypatch.setattr(architectures, "GINEConv", None)

    with pytest.raises(RuntimeError, match="GINEConv"):
        architectures.GraphGPS(hidden_dim=8, seed=3)


def test_explanation_wrapper_supports_optional_edge_attributes_and_log_probs():
    x, edge_index, edge_attr = _single_graph()
    wrapper = GraphExplanationWrapper(
        GCN(hidden_dim=8, seed=5), seed=29, config={"method": "input-gradient"}
    )

    with_edge_attributes = wrapper.explain(x, edge_index, edge_attr)
    without_edge_attributes = wrapper.explain(x, edge_index)

    for result in (with_edge_attributes, without_edge_attributes):
        assert result.node_mask.shape == (x.size(0),)
        assert result.edge_mask.shape == (edge_index.size(1),)
        assert torch.isfinite(result.node_mask).all()
        assert torch.isfinite(result.edge_mask).all()
        assert result.log_prob.shape == (1, 2)
        assert torch.allclose(result.log_prob.exp().sum(dim=-1), torch.ones(1))
        assert result.metadata["seed"] == 29
        assert result.metadata["config"]["method"] == "input-gradient"

    assert "index" not in inspect.signature(wrapper.explain).parameters


def test_explanation_wrapper_rejects_nan_inputs():
    x, edge_index, edge_attr = _single_graph()
    x = x.clone()
    x[1, 2] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        GraphExplanationWrapper(GCN(hidden_dim=8, seed=5)).explain(
            x, edge_index, edge_attr
        )


def test_explanation_wrapper_rejects_non_tensor_edge_index_cleanly():
    x, _, edge_attr = _single_graph()

    with pytest.raises(ValueError, match="edge_index"):
        GraphExplanationWrapper(GCN(hidden_dim=8, seed=5)).explain(
            x,
            [[0], [1]],
            edge_attr,
        )


class _EdgeOnlyModel(torch.nn.Module):
    def forward(self, x, edge_index, edge_attr, batch):
        score = edge_attr.sum().reshape(1, 1)
        return torch.cat((-score, score), dim=1)


def test_faithfulness_uses_unperturbed_baseline_before_edge_perturbation():
    x = torch.ones(2, 7)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor([[0.25], [1.0]], dtype=torch.float32)
    explanation = ExplanationResult(
        node_mask=torch.zeros(2),
        edge_mask=torch.tensor([0.0, 10.0]),
        log_prob=torch.log_softmax(torch.tensor([[-1.25, 1.25]]), dim=-1),
        target_class=1,
        metadata={"method": "synthetic-regression"},
    )

    score = faithfulness_score(
        _EdgeOnlyModel(),
        x,
        edge_index,
        edge_attr,
        explanation=explanation,
        fraction=0.5,
    )

    assert score > 0.1


def test_faithfulness_is_perturbation_based_and_finite():
    x, edge_index, edge_attr = _single_graph()
    model = GAT(hidden_dim=8, seed=13)
    explanation = GraphExplanationWrapper(model, seed=2).explain(
        x, edge_index, edge_attr
    )

    score = faithfulness_score(
        model,
        x,
        edge_index,
        edge_attr,
        explanation=explanation,
        fraction=0.5,
    )

    assert isinstance(score, float)
    assert torch.isfinite(torch.tensor(score))


def test_repeat_seed_stability_is_finite_and_bounded():
    x, edge_index, edge_attr = _single_graph()
    score = stability_score(
        GCN(hidden_dim=8, seed=13),
        x,
        edge_index,
        edge_attr,
        seeds=(0, 1, 2),
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_stability_uses_seeded_input_perturbations():
    x, edge_index, edge_attr = _single_graph()
    model = GCN(hidden_dim=8, seed=13)

    first = stability_score(
        model,
        x,
        edge_index,
        edge_attr,
        seeds=(2, 5, 11),
        perturbation_scale=0.05,
    )
    second = stability_score(
        model,
        x,
        edge_index,
        edge_attr,
        seeds=(2, 5, 11),
        perturbation_scale=0.05,
    )

    assert first == pytest.approx(second)
    assert 0.0 <= first < 1.0

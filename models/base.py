"""Shared contracts and validation for the four graph classifiers."""

from contextlib import contextmanager

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool


NODE_FEATURES = 7
EDGE_FEATURES = 1
NUM_CLASSES = 2


@contextmanager
def deterministic_initialization(seed: int | None):
    """Initialize a module deterministically without changing global RNG state."""

    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


def validate_graph_inputs(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
) -> None:
    """Validate the shared seven-node-feature/one-edge-feature graph contract."""

    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.size(1) != NODE_FEATURES:
        raise ValueError("x must contain seven node features")
    if not torch.is_floating_point(x):
        raise ValueError("x must be a floating point tensor")
    if x.size(0) == 0:
        raise ValueError("x must contain at least one node")
    if not torch.isfinite(x).all():
        raise ValueError("x must contain only finite values")

    if (
        not isinstance(edge_index, torch.Tensor)
        or edge_index.ndim != 2
        or edge_index.size(0) != 2
        or edge_index.dtype not in (torch.int64, torch.int32)
    ):
        raise ValueError("edge_index must have shape [2, num_edges] and integer dtype")
    if edge_index.numel() and (
        edge_index.min().item() < 0 or edge_index.max().item() >= x.size(0)
    ):
        raise ValueError("edge_index contains an invalid node index")

    if (
        not isinstance(edge_attr, torch.Tensor)
        or edge_attr.ndim != 2
        or edge_attr.size(1) != EDGE_FEATURES
    ):
        raise ValueError("edge_attr must contain one edge feature")
    if not torch.is_floating_point(edge_attr):
        raise ValueError("edge_attr must be a floating point tensor")
    if edge_attr.device != x.device or edge_index.device != x.device:
        raise ValueError("x, edge_index, and edge_attr must be on the same device")
    if edge_attr.size(0) != edge_index.size(1):
        raise ValueError("edge_attr must have one row per edge")
    if not torch.isfinite(edge_attr).all():
        raise ValueError("edge_attr must contain only finite values")
    if edge_attr.numel() and ((edge_attr < 0.0).any() or (edge_attr > 1.0).any()):
        raise ValueError("edge_attr must be normalized to [0, 1]")

    if (
        not isinstance(batch, torch.Tensor)
        or batch.ndim != 1
        or batch.numel() != x.size(0)
        or batch.dtype not in (torch.int64, torch.int32)
    ):
        raise ValueError("batch must be an integer vector with one value per node")
    if batch.device != x.device:
        raise ValueError("x and batch must be on the same device")
    if batch.numel() == 0 or batch.min().item() < 0:
        raise ValueError("batch must contain at least one non-negative graph id")
    graph_ids = torch.unique(batch, sorted=True)
    expected_ids = torch.arange(
        int(graph_ids[-1].item()) + 1,
        dtype=graph_ids.dtype,
        device=graph_ids.device,
    )
    if not torch.equal(graph_ids, expected_ids):
        raise ValueError("batch graph ids must be contiguous and start at zero")
    if edge_index.numel() and torch.any(batch[edge_index[0]] != batch[edge_index[1]]):
        raise ValueError("edge_index must not contain cross graph edges")


def edge_context(edge_hidden: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Aggregate already-projected edge features at destination nodes."""

    context = edge_hidden.new_zeros((num_nodes, edge_hidden.size(-1)))
    if edge_hidden.size(0) == 0:
        return context
    context.index_add_(0, edge_index[1], edge_hidden)
    degree = edge_hidden.new_zeros(num_nodes)
    degree.index_add_(0, edge_index[1], edge_hidden.new_ones(edge_hidden.size(0)))
    return context / degree.clamp_min(1).unsqueeze(-1)


class GraphModelBase(nn.Module):
    """Common input validation, edge projection, pooling, and classification head."""

    def __init__(
        self,
        hidden_dim: int = 32,
        num_classes: int = NUM_CLASSES,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_classes != NUM_CLASSES:
            raise ValueError("the shared model contract requires two output classes")
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.seed = seed
        self.edge_projection = nn.Linear(EDGE_FEATURES, self.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def _validate_and_project(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        validate_graph_inputs(x, edge_index, edge_attr, batch)
        return self.edge_projection(edge_attr)

    def _pool_and_classify(self, node_states: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        pooled = global_mean_pool(node_states, batch)
        logits = self.head(pooled)
        if logits.ndim != 2 or logits.size(1) != NUM_CLASSES:
            raise RuntimeError("model output must have shape [num_graphs, 2]")
        if not torch.isfinite(logits).all():
            raise RuntimeError("model output must contain only finite values")
        return logits

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

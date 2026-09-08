"""Small, shared-interface PyG graph-classification architectures."""

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from .base import (
    EDGE_FEATURES,
    GraphModelBase,
    deterministic_initialization,
    edge_context,
)

try:
    from torch_geometric.nn import GINEConv
except ImportError:  # pragma: no cover - exercised only with a minimal PyG install.
    GINEConv = None


def _mlp(width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))


class GCN(GraphModelBase):
    """GCN with a scalar edge gate derived from the projected edge feature."""

    def __init__(self, hidden_dim: int = 32, seed: int | None = 0) -> None:
        with deterministic_initialization(seed):
            super().__init__(hidden_dim=hidden_dim, seed=seed)
            self.input_projection = nn.Linear(7, hidden_dim)
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.edge_gate = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_hidden = self._validate_and_project(x, edge_index, edge_attr, batch)
        node_states = self.input_projection(x)
        edge_weight = torch.sigmoid(self.edge_gate(edge_hidden)).squeeze(-1)
        node_states = F.relu(self.conv1(node_states, edge_index, edge_weight))
        node_states = F.relu(self.conv2(node_states, edge_index, edge_weight))
        return self._pool_and_classify(node_states, batch)


class GraphSAGE(GraphModelBase):
    """GraphSAGE with destination-node context from projected edge features."""

    def __init__(self, hidden_dim: int = 32, seed: int | None = 0) -> None:
        with deterministic_initialization(seed):
            super().__init__(hidden_dim=hidden_dim, seed=seed)
            self.input_projection = nn.Linear(7, hidden_dim)
            self.conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_hidden = self._validate_and_project(x, edge_index, edge_attr, batch)
        node_states = self.input_projection(x)
        node_states = node_states + edge_context(edge_hidden, edge_index, x.size(0))
        node_states = F.relu(self.conv1(node_states, edge_index))
        node_states = F.relu(self.conv2(node_states, edge_index))
        return self._pool_and_classify(node_states, batch)


class GAT(GraphModelBase):
    """Two-layer multi-head attention using projected edge features."""

    def __init__(self, hidden_dim: int = 32, seed: int | None = 0) -> None:
        with deterministic_initialization(seed):
            super().__init__(hidden_dim=hidden_dim, seed=seed)
            self.input_projection = nn.Linear(7, hidden_dim)
            self.conv1 = GATConv(
                hidden_dim,
                hidden_dim,
                heads=2,
                concat=False,
                edge_dim=hidden_dim,
                dropout=0.0,
            )
            self.conv2 = GATConv(
                hidden_dim,
                hidden_dim,
                heads=2,
                concat=False,
                edge_dim=hidden_dim,
                dropout=0.0,
            )

    def forward(self, x, edge_index, edge_attr, batch):
        edge_hidden = self._validate_and_project(x, edge_index, edge_attr, batch)
        node_states = self.input_projection(x)
        node_states = F.relu(self.conv1(node_states, edge_index, edge_hidden))
        node_states = F.relu(self.conv2(node_states, edge_index, edge_hidden))
        return self._pool_and_classify(node_states, batch)


class _GraphTransformerBlock(nn.Module):
    """A batch-isolated transformer block over nodes in each graph."""

    def __init__(self, hidden_dim: int, heads: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, node_states: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        blocked = batch.unsqueeze(0) != batch.unsqueeze(1)
        attended, _ = self.attention(
            node_states.unsqueeze(0),
            node_states.unsqueeze(0),
            node_states.unsqueeze(0),
            attn_mask=blocked,
            need_weights=False,
        )
        node_states = self.norm1(node_states + attended.squeeze(0))
        return self.norm2(node_states + self.feed_forward(node_states))


class GraphGPS(GraphModelBase):
    """GPS-like local edge-aware message passing followed by graph-isolated attention."""

    def __init__(self, hidden_dim: int = 32, seed: int | None = 0) -> None:
        if GINEConv is None:
            raise RuntimeError(
                "GraphGPS requires torch_geometric.nn.GINEConv; refusing to substitute another architecture"
            )
        with deterministic_initialization(seed):
            super().__init__(hidden_dim=hidden_dim, seed=seed)
            self.input_projection = nn.Linear(7, hidden_dim)
            self.local1 = GINEConv(_mlp(hidden_dim), edge_dim=hidden_dim)
            self.local2 = GINEConv(_mlp(hidden_dim), edge_dim=hidden_dim)
            heads = 2 if hidden_dim % 2 == 0 else 1
            self.global_block = _GraphTransformerBlock(hidden_dim, heads=heads)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_hidden = self._validate_and_project(x, edge_index, edge_attr, batch)
        node_states = self.input_projection(x)
        node_states = F.relu(self.local1(node_states, edge_index, edge_hidden))
        node_states = self.global_block(node_states, batch)
        node_states = F.relu(self.local2(node_states, edge_index, edge_hidden))
        return self._pool_and_classify(node_states, batch)


GCNModel = GCN
GraphSAGEModel = GraphSAGE
GATModel = GAT
GraphGPSModel = GraphGPS


MODEL_REGISTRY: dict[str, Callable[..., GraphModelBase]] = {
    "gcn": GCN,
    "graphsage": GraphSAGE,
    "gat": GAT,
    "graphgps": GraphGPS,
}

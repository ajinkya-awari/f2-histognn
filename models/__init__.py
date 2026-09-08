"""Shared model exports for the HistoGNN benchmark."""

from .architectures import (
    GAT,
    GATModel,
    GCN,
    GCNModel,
    GraphGPS,
    GraphGPSModel,
    GraphSAGE,
    GraphSAGEModel,
    MODEL_REGISTRY,
)
from .base import EDGE_FEATURES, NODE_FEATURES, NUM_CLASSES, GraphModelBase

__all__ = [
    "EDGE_FEATURES",
    "NODE_FEATURES",
    "NUM_CLASSES",
    "GraphModelBase",
    "GCN",
    "GCNModel",
    "GraphSAGE",
    "GraphSAGEModel",
    "GAT",
    "GATModel",
    "GraphGPS",
    "GraphGPSModel",
    "MODEL_REGISTRY",
]

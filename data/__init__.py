"""Dependency-light data contracts for the synthetic HistoGNN pipeline."""

from .labels import extract_group_id, parse_label
from .manifest import (
    ArtifactByteVerification,
    ManifestMetadata,
    validate_manifest,
    verify_artifact_bytes,
)
from .schema import GraphRecord, build_node_features, validate_graph_record
from .split import GroupSplit, assert_group_disjoint, partition_groups

__all__ = [
    "GraphRecord",
    "GroupSplit",
    "ArtifactByteVerification",
    "ManifestMetadata",
    "assert_group_disjoint",
    "build_node_features",
    "extract_group_id",
    "parse_label",
    "partition_groups",
    "validate_graph_record",
    "validate_manifest",
    "verify_artifact_bytes",
]

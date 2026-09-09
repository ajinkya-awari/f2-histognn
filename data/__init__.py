"""Dependency-light data contracts for the synthetic HistoGNN pipeline."""

from .benchmark import (
    BenchmarkContractError,
    BenchmarkSplit,
    sanitized_split_summary,
    select_balanced_cases,
    stratified_case_split,
)

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
    "BenchmarkContractError",
    "BenchmarkSplit",
    "ManifestMetadata",
    "assert_group_disjoint",
    "build_node_features",
    "extract_group_id",
    "parse_label",
    "partition_groups",
    "sanitized_split_summary",
    "select_balanced_cases",
    "stratified_case_split",
    "validate_graph_record",
    "validate_manifest",
    "verify_artifact_bytes",
]

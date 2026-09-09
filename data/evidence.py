"""Sanitized, append-only evidence records for real-data execution stages."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any


class EvidenceError(ValueError):
    """Raised when evidence could leak identifiers or support an invalid claim."""


_ALLOWED_STATUSES = {"passed", "failed", "blocked", "skipped"}
_METRIC_STAGES = {"pilot_evaluation", "benchmark"}
_REQUIRED_METRIC_PROVENANCE = {
    "dataset_release",
    "manifest_sha256",
    "split_manifest_sha256",
    "code_revision",
    "dependency_lock_sha256",
    "seed",
    "device",
    "artifact_count",
    "class_counts",
    "artifact_hashes",
    "positive_class",
    "uncertainty_method",
}
_FORBIDDEN_KEYS = {
    "case_id",
    "case_submitter_id",
    "file_id",
    "file_name",
    "md5sum",
    "patient_id",
    "slide_id",
    "slide_submitter_id",
}
_IDENTIFIER_PATTERNS = (
    re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Z0-9-]+)?", re.IGNORECASE),
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE),
)
_METRIC_NAMES = {"accuracy", "macro_f1", "auroc", "auprc"}


def _reject_identifiers(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise EvidenceError(f"identifier field is forbidden in public evidence: {key}")
            _reject_identifiers(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_identifiers(nested)
    elif isinstance(value, str) and any(pattern.search(value) for pattern in _IDENTIFIER_PATTERNS):
        raise EvidenceError("identifier-like value is forbidden in public evidence")


def _validate_metric_provenance(provenance: Mapping[str, Any]) -> None:
    missing = sorted(_REQUIRED_METRIC_PROVENANCE - set(provenance))
    if missing or any(provenance.get(key) is None for key in _REQUIRED_METRIC_PROVENANCE):
        raise EvidenceError("metrics require complete non-null provenance")
    for key in ("manifest_sha256", "split_manifest_sha256", "dependency_lock_sha256"):
        if not isinstance(provenance[key], str) or not re.fullmatch(r"[0-9a-f]{64}", provenance[key]):
            raise EvidenceError("metric provenance contains an invalid SHA-256")
    if not isinstance(provenance["code_revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", provenance["code_revision"]):
        raise EvidenceError("metric provenance requires a full code revision")
    if not isinstance(provenance["dataset_release"], str) or not provenance["dataset_release"].strip():
        raise EvidenceError("metric provenance requires a dataset release")
    if type(provenance["seed"]) is not int:
        raise EvidenceError("metric provenance seed must be an integer")
    device = provenance["device"]
    if not isinstance(device, Mapping) or any(
        not isinstance(device.get(key), str) or not device[key].strip() for key in ("type", "name")
    ):
        raise EvidenceError("metric provenance requires a named device")
    if type(provenance["artifact_count"]) is not int or provenance["artifact_count"] <= 0:
        raise EvidenceError("metric provenance artifact_count must be positive")
    counts = provenance["class_counts"]
    if not isinstance(counts, Mapping) or set(counts) != {"LUAD", "LUSC"} or any(
        type(value) is not int or value <= 0 for value in counts.values()
    ):
        raise EvidenceError("metric provenance requires positive LUAD/LUSC class counts")
    hashes = provenance["artifact_hashes"]
    if not isinstance(hashes, Mapping) or not hashes or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in hashes.values()
    ):
        raise EvidenceError("metric provenance requires artifact SHA-256 hashes")
    if provenance["positive_class"] not in {"LUAD", "LUSC"}:
        raise EvidenceError("metric provenance positive_class must be LUAD or LUSC")
    if not isinstance(provenance["uncertainty_method"], str) or not provenance[
        "uncertainty_method"
    ].strip():
        raise EvidenceError("metric provenance requires an uncertainty method")


def _validate_metrics(metrics: Mapping[str, Any]) -> None:
    if set(metrics) != _METRIC_NAMES:
        raise EvidenceError("metric evidence requires exactly accuracy, macro_f1, auroc, and auprc")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()):
        raise EvidenceError("metric values must be finite numbers in [0, 1]")


def build_stage_evidence(
    *,
    stage: str,
    status: str,
    provenance: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe stage record, rejecting unsupported metrics."""

    if not isinstance(stage, str) or not stage.strip():
        raise EvidenceError("stage must be a non-blank string")
    if status not in _ALLOWED_STATUSES:
        raise EvidenceError("status must be passed, failed, blocked, or skipped")
    if not isinstance(provenance, Mapping):
        raise EvidenceError("provenance must be a mapping")
    _reject_identifiers(provenance)

    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "timestamp_utc": timestamp_utc
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "stage": stage.strip(),
        "status": status,
        "provenance": dict(provenance),
    }
    if metrics is not None:
        if stage not in _METRIC_STAGES:
            raise EvidenceError("metrics are allowed only for pilot_evaluation or benchmark")
        if status != "passed":
            raise EvidenceError("metrics require a passed stage")
        if not isinstance(metrics, Mapping) or not metrics:
            raise EvidenceError("metrics must be a non-empty mapping")
        _validate_metric_provenance(provenance)
        _validate_metrics(metrics)
        _reject_identifiers(metrics)
        evidence["metrics"] = dict(metrics)

    try:
        json.dumps(evidence, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise EvidenceError("evidence must contain finite JSON-compatible values") from exc
    return evidence


def write_evidence(path: str | Path, evidence: Mapping[str, Any]) -> Path:
    """Write a new evidence file without overwriting an existing record."""

    target = Path(path)
    if target.exists():
        raise EvidenceError(f"evidence path already exists: {target}")
    _reject_identifiers(evidence)
    try:
        encoded = json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as exc:
        raise EvidenceError("evidence must contain finite JSON-compatible values") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise EvidenceError(f"evidence path already exists: {target}") from exc
    return target

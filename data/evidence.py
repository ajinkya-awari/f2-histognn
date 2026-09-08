"""Sanitized, append-only evidence records for real-data execution stages."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
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


def _reject_identifiers(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise EvidenceError(f"identifier field is forbidden in public evidence: {key}")
            _reject_identifiers(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_identifiers(nested)


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
        missing = sorted(_REQUIRED_METRIC_PROVENANCE - set(provenance))
        if missing:
            raise EvidenceError("metrics require complete provenance: " + ", ".join(missing))
        if not isinstance(metrics, Mapping) or not metrics:
            raise EvidenceError("metrics must be a non-empty mapping")
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

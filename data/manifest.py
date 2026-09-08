"""Validation for manifest metadata; this module never acquires artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


class ManifestValidationError(ValueError):
    """Raised when artifact provenance or contract metadata is incomplete."""


_REQUIRED_FIELDS = (
    "source",
    "license",
    "checksum",
    "schema_version",
    "label_mapping",
    "grouping_field",
    "rights",
    "provenance",
)

_REQUIRED_RIGHTS_FIELDS = ("authorized", "usage", "license_url")
_REQUIRED_PROVENANCE_FIELDS = ("artifact_id", "source_uri")
_SHA256_PREFIX = "sha256:"
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class ManifestMetadata:
    """The minimum metadata required before an artifact may be considered."""

    source: str
    license: str
    checksum: str
    schema_version: str
    label_mapping: Mapping[str, Any]
    grouping_field: str
    rights: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_mapping", dict(self.label_mapping))
        object.__setattr__(self, "rights", dict(self.rights))
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class ArtifactByteVerification:
    """Byte-level checksum evidence for one local synthetic or approved artifact."""

    artifact_path: Path
    checksum: str
    byte_size: int


def _normalize_sha256(checksum: str) -> str:
    if not isinstance(checksum, str):
        raise ManifestValidationError("checksum must be a sha256:<64 hex> string")
    value = checksum.strip().lower()
    if not value.startswith(_SHA256_PREFIX):
        raise ManifestValidationError("checksum must be a sha256:<64 hex> string")
    digest = value[len(_SHA256_PREFIX) :]
    if len(digest) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ManifestValidationError("checksum must be a sha256:<64 hex> string")
    return _SHA256_PREFIX + digest


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ManifestValidationError(f"{field} must be a non-empty mapping")
    return value


def _require_non_blank(mapping: Mapping[str, Any], field: str, key: str) -> None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field}.{key} must be a non-blank string")


def _validate_rights(rights: Any) -> Mapping[str, Any]:
    values = _require_mapping(rights, "rights")
    missing = [field for field in _REQUIRED_RIGHTS_FIELDS if field not in values]
    if missing:
        raise ManifestValidationError(
            "missing required rights field(s): " + ", ".join(missing)
        )
    if values["authorized"] is not True:
        raise ManifestValidationError("rights.authorized must be true")
    for field in ("usage", "license_url"):
        _require_non_blank(values, "rights", field)
    return values


def _validate_provenance(provenance: Any) -> Mapping[str, Any]:
    values = _require_mapping(provenance, "provenance")
    missing = [field for field in _REQUIRED_PROVENANCE_FIELDS if field not in values]
    if missing:
        raise ManifestValidationError(
            "missing required provenance field(s): " + ", ".join(missing)
        )
    for field in _REQUIRED_PROVENANCE_FIELDS:
        _require_non_blank(values, "provenance", field)
    return values


def validate_manifest(metadata: ManifestMetadata | Mapping[str, Any]) -> ManifestMetadata:
    """Validate and normalize manifest metadata without reading or acquiring data."""

    if isinstance(metadata, ManifestMetadata):
        candidate = metadata
    elif isinstance(metadata, Mapping):
        missing = [field for field in _REQUIRED_FIELDS if field not in metadata]
        if missing:
            raise ManifestValidationError(
                "missing required manifest field(s): " + ", ".join(missing)
            )
        candidate = ManifestMetadata(
            source=metadata["source"],
            license=metadata["license"],
            checksum=metadata["checksum"],
            schema_version=metadata["schema_version"],
            label_mapping=metadata["label_mapping"],
            grouping_field=metadata["grouping_field"],
            rights=metadata["rights"],
            provenance=metadata["provenance"],
        )
    else:
        raise ManifestValidationError("manifest metadata must be a mapping")

    for field in ("source", "license", "schema_version", "grouping_field"):
        value = getattr(candidate, field)
        if not isinstance(value, str) or not value.strip():
            raise ManifestValidationError(f"{field} must be a non-blank string")
    normalized_checksum = _normalize_sha256(candidate.checksum)

    if not isinstance(candidate.label_mapping, Mapping):
        raise ManifestValidationError("label_mapping must map LUAD and LUSC to classes 0 and 1")
    label_mapping = dict(candidate.label_mapping)
    if set(label_mapping) != {"LUAD", "LUSC"}:
        raise ManifestValidationError("label_mapping must contain exactly LUAD and LUSC")
    class_values = tuple(label_mapping.values())
    if any(type(value) is not int for value in class_values) or set(class_values) != {0, 1}:
        raise ManifestValidationError(
            "label_mapping must map LUAD and LUSC bijectively to integer classes 0 and 1"
        )

    rights = _validate_rights(candidate.rights)
    provenance = _validate_provenance(candidate.provenance)
    return ManifestMetadata(
        source=candidate.source.strip(),
        license=candidate.license.strip(),
        checksum=normalized_checksum,
        schema_version=candidate.schema_version.strip(),
        label_mapping=label_mapping,
        grouping_field=candidate.grouping_field.strip(),
        rights=rights,
        provenance=provenance,
    )


def verify_artifact_bytes(
    artifact_path: str | Path,
    expected_checksum: str,
) -> ArtifactByteVerification:
    """Verify SHA-256 bytes for a synthetic fixture or explicitly approved artifact."""

    expected = _normalize_sha256(expected_checksum)
    path = Path(artifact_path)
    if not path.is_file():
        raise ManifestValidationError("artifact path must point to a file")

    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            byte_size += len(block)
            digest.update(block)
    if byte_size == 0:
        raise ManifestValidationError("artifact file must be non-empty")

    actual = _SHA256_PREFIX + digest.hexdigest()
    if actual != expected:
        raise ManifestValidationError(
            f"checksum mismatch: expected {expected}, observed {actual}"
        )
    return ArtifactByteVerification(
        artifact_path=path,
        checksum=actual,
        byte_size=byte_size,
    )


validate_artifact_manifest = validate_manifest

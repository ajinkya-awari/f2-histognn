"""Explicit label and authorized synthetic group parsing."""

from __future__ import annotations

import re


class LabelParseError(ValueError):
    """Raised for unknown or ambiguous histology labels."""


class GroupExtractionError(ValueError):
    """Raised when a group cannot be safely extracted from an authorized record."""


LABEL_MAPPING = {
    "LUAD": "LUAD",
    "LUSC": "LUSC",
    "LUNG ADENOCARCINOMA": "LUAD",
    "LUNG SQUAMOUS CELL CARCINOMA": "LUSC",
}

_GROUP_PATTERN = re.compile(
    r"(?P<kind>patient|case)(?:[-_=])(?P<identifier>[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)",
    re.IGNORECASE,
)


def parse_label(value: str) -> str:
    """Parse only the small, explicit LUAD/LUSC mapping."""

    if not isinstance(value, str) or not value.strip():
        raise LabelParseError("unknown label")
    normalized = " ".join(value.strip().upper().replace("_", " ").split())
    if "LUAD" in normalized and "LUSC" in normalized:
        raise LabelParseError("ambiguous label")
    try:
        return LABEL_MAPPING[normalized]
    except KeyError as exc:
        raise LabelParseError(f"unknown label: {value!r}") from exc


def extract_group_id(record: str, *, authorized: bool = False) -> str:
    """Extract a patient/case group only from an explicitly authorized string."""

    if not authorized:
        raise GroupExtractionError("record string is not authorized")
    if not isinstance(record, str) or not record.strip():
        raise GroupExtractionError("record string must be non-blank")

    matches = list(_GROUP_PATTERN.finditer(record))
    groups = {
        f"{match.group('kind').lower()}-{match.group('identifier')}" for match in matches
    }
    if not groups:
        raise GroupExtractionError("record string has no patient/case group")
    if len(groups) != 1:
        raise GroupExtractionError("record string has ambiguous patient/case groups")
    return next(iter(groups))


extract_group = extract_group_id

"""Pure validation for GDC slide metadata; no network access occurs here."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any


class GDCManifestError(ValueError):
    """Raised when GDC metadata cannot satisfy the approved cohort contract."""


_PROJECT_TO_LABEL = {"TCGA-LUAD": "LUAD", "TCGA-LUSC": "LUSC"}
_MD5_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")
_DIAGNOSTIC_SLIDE_PATTERN = re.compile(r"-DX[0-9A-Z]+$")
_DIAGNOSTIC_FILE_PATTERN = re.compile(r"-DX[0-9A-Z]+(?:\.[^.]+)?\.svs$", re.IGNORECASE)


def build_gdc_query_payload(*, page_size: int = 10000) -> dict[str, Any]:
    """Build the reviewed metadata-only query for eligible lung slide files."""

    if type(page_size) is not int or page_size <= 0:
        raise GDCManifestError("page_size must be a positive integer")
    return {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": ["TCGA-LUAD", "TCGA-LUSC"],
                    },
                },
                {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
                {"op": "=", "content": {"field": "access", "value": "open"}},
                {
                    "op": "=",
                    "content": {"field": "cases.samples.sample_type", "value": "Primary Tumor"},
                },
            ],
        },
        "format": "JSON",
        "fields": ",".join(
            (
                "file_id",
                "file_name",
                "file_size",
                "md5sum",
                "data_type",
                "access",
                "cases.case_id",
                "cases.submitter_id",
                "cases.project.project_id",
                "cases.samples.sample_type",
                "cases.samples.portions.slides.submitter_id",
            )
        ),
        "size": page_size,
    }


def extract_gdc_response_hits(response: Any) -> tuple[Mapping[str, Any], ...]:
    """Extract a complete page of hits and reject silently truncated responses."""

    if not isinstance(response, Mapping) or not isinstance(response.get("data"), Mapping):
        raise GDCManifestError("GDC response must contain a data mapping")
    data = response["data"]
    hits = data.get("hits")
    pagination = data.get("pagination")
    if not isinstance(hits, list) or not isinstance(pagination, Mapping):
        raise GDCManifestError("GDC response must contain hits and pagination")
    if any(not isinstance(hit, Mapping) for hit in hits):
        raise GDCManifestError("GDC response hits must be mappings")
    total = pagination.get("total")
    count = pagination.get("count")
    if type(total) is not int or type(count) is not int or count != len(hits):
        raise GDCManifestError("GDC response pagination is inconsistent")
    if count != total:
        raise GDCManifestError(f"GDC response is truncated: received {count} of {total}")
    return tuple(hits)


def filter_diagnostic_hits(hits: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """Restrict a GDC metadata page to filenames encoding diagnostic DX slides."""

    values = tuple(hits)
    return tuple(
        hit
        for hit in values
        if isinstance(hit, Mapping)
        and isinstance(hit.get("file_name"), str)
        and _DIAGNOSTIC_FILE_PATTERN.search(hit["file_name"])
    )


@dataclass(frozen=True, order=True)
class GDCSlideRecord:
    """Validated metadata for one open-access TCGA diagnostic slide file."""

    file_id: str
    file_name: str
    file_size: int
    md5sum: str
    project_id: str
    label: str
    case_id: str
    case_submitter_id: str
    slide_submitter_id: str


def _non_blank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GDCManifestError(f"{field} must be a non-blank string")
    return value.strip()


def _single_case(hit: Mapping[str, Any]) -> Mapping[str, Any]:
    cases = hit.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], Mapping):
        raise GDCManifestError("each slide must resolve to exactly one case")
    return cases[0]


def _slide_id(case: Mapping[str, Any], file_name: str) -> str:
    found: set[str] = set()
    samples = case.get("samples", [])
    if not isinstance(samples, list) or not any(
        isinstance(sample, Mapping) and sample.get("sample_type") == "Primary Tumor"
        for sample in samples
    ):
        raise GDCManifestError("slide must be linked to a Primary Tumor sample")
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            if sample.get("sample_type") != "Primary Tumor":
                continue
            portions = sample.get("portions", [])
            if not isinstance(portions, list):
                continue
            for portion in portions:
                if not isinstance(portion, Mapping):
                    continue
                slides = portion.get("slides", [])
                if not isinstance(slides, list):
                    continue
                for slide in slides:
                    if isinstance(slide, Mapping):
                        value = slide.get("submitter_id")
                        if isinstance(value, str) and value.strip():
                            found.add(value.strip())
    matches = {value for value in found if file_name.startswith(value + ".")}
    if len(matches) != 1:
        raise GDCManifestError("each file must resolve to exactly one slide submitter ID")
    slide_id = next(iter(matches))
    if not _DIAGNOSTIC_SLIDE_PATTERN.search(slide_id):
        raise GDCManifestError("slide must be a primary diagnostic DX slide")
    return slide_id


def _parse_hit(hit: Mapping[str, Any]) -> GDCSlideRecord:
    if hit.get("access") != "open":
        raise GDCManifestError("slide record must be open access")
    if hit.get("data_type") != "Slide Image":
        raise GDCManifestError("data_type must be Slide Image")

    case = _single_case(hit)
    project = case.get("project")
    if not isinstance(project, Mapping):
        raise GDCManifestError("case project metadata is required")
    project_id = _non_blank(project.get("project_id"), "project ID")
    if project_id not in _PROJECT_TO_LABEL:
        raise GDCManifestError("project must be TCGA-LUAD or TCGA-LUSC")

    md5sum = _non_blank(hit.get("md5sum"), "md5sum").lower()
    if not _MD5_PATTERN.fullmatch(md5sum):
        raise GDCManifestError("md5sum must contain exactly 32 hexadecimal characters")
    file_size = hit.get("file_size")
    if type(file_size) is not int or file_size <= 0:
        raise GDCManifestError("file_size must be a positive integer")

    file_name = _non_blank(hit.get("file_name"), "file name")
    return GDCSlideRecord(
        file_id=_non_blank(hit.get("file_id"), "file UUID"),
        file_name=file_name,
        file_size=file_size,
        md5sum=md5sum,
        project_id=project_id,
        label=_PROJECT_TO_LABEL[project_id],
        case_id=_non_blank(case.get("case_id"), "case UUID"),
        case_submitter_id=_non_blank(case.get("submitter_id"), "case submitter ID"),
        slide_submitter_id=_slide_id(case, file_name),
    )


def parse_gdc_hits(hits: Iterable[Mapping[str, Any]]) -> tuple[GDCSlideRecord, ...]:
    """Validate raw GDC file hits and return a stable metadata sequence."""

    try:
        values = tuple(hits)
    except TypeError as exc:
        raise GDCManifestError("GDC hits must be iterable") from exc
    if not values:
        raise GDCManifestError("GDC hits must be non-empty")
    if any(not isinstance(hit, Mapping) for hit in values):
        raise GDCManifestError("each GDC hit must be a mapping")

    records = tuple(sorted(_parse_hit(hit) for hit in values))
    file_ids = [record.file_id for record in records]
    duplicates = sorted(file_id for file_id, count in Counter(file_ids).items() if count > 1)
    if duplicates:
        raise GDCManifestError("duplicate file UUID detected")

    labels_by_case: dict[str, set[str]] = defaultdict(set)
    for record in records:
        labels_by_case[record.case_submitter_id].add(record.label)
    if any(len(labels) != 1 for labels in labels_by_case.values()):
        raise GDCManifestError("conflicting project labels detected for one case")
    return records


def _seed_rank(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def select_case_disjoint_pilot(
    records: Iterable[GDCSlideRecord],
    *,
    per_class: int,
    seed: int = 0,
) -> tuple[GDCSlideRecord, ...]:
    """Select a balanced, stable pilot with at most one slide per TCGA case."""

    if type(per_class) is not int or per_class <= 0:
        raise GDCManifestError("per_class must be a positive integer")
    if type(seed) is not int:
        raise GDCManifestError("seed must be an integer")

    by_case: dict[str, list[GDCSlideRecord]] = defaultdict(list)
    for record in records:
        if not isinstance(record, GDCSlideRecord):
            raise GDCManifestError("pilot input must contain validated GDC slide records")
        by_case[record.case_submitter_id].append(record)

    one_per_case = [sorted(case_records, key=lambda item: item.file_id)[0] for case_records in by_case.values()]
    selected: list[GDCSlideRecord] = []
    for label in ("LUAD", "LUSC"):
        candidates = sorted(
            (record for record in one_per_case if record.label == label),
            key=lambda item: (_seed_rank(seed, item.case_submitter_id), item.file_id),
        )
        if len(candidates) < per_class:
            raise GDCManifestError(
                f"insufficient {label} cases: need {per_class}, found {len(candidates)}"
            )
        selected.extend(candidates[:per_class])
    return tuple(sorted(selected))


def manifest_summary(records: Iterable[GDCSlideRecord]) -> dict[str, Any]:
    """Return publishable aggregate counts and a canonical private-manifest hash."""

    values = tuple(sorted(records))
    if not values:
        raise GDCManifestError("manifest summary requires at least one record")
    if any(not isinstance(record, GDCSlideRecord) for record in values):
        raise GDCManifestError("manifest summary requires validated GDC slide records")
    canonical = json.dumps(
        [asdict(record) for record in values],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "record_count": len(values),
        "case_count": len({record.case_submitter_id for record in values}),
        "class_counts": {
            label: sum(record.label == label for record in values)
            for label in ("LUAD", "LUSC")
        },
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }

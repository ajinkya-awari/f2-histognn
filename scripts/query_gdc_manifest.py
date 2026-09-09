"""Query GDC metadata only and write private manifest plus sanitized evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.evidence import build_stage_evidence, write_evidence
from data.gdc import (
    build_gdc_query_payload,
    extract_gdc_response_hits,
    filter_diagnostic_hits,
    manifest_summary,
    parse_gdc_hits,
    select_case_disjoint_pilot,
)


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query open TCGA LUAD/LUSC slide metadata without downloading slides."
    )
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--page-size", type=int, default=10000)
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _query(page_size: int) -> dict:
    payload = json.dumps(build_gdc_query_payload(page_size=page_size)).encode("utf-8")
    request = Request(
        GDC_FILES_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "f2-histognn/0.1"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> int:
    args = _arguments()
    project_root = Path(__file__).resolve().parents[1]
    try:
        private_root = args.private_root.resolve()
        private_root.relative_to(project_root)
    except ValueError:
        pass
    else:
        print("private root must be outside the public project checkout", file=sys.stderr)
        return 2
    try:
        args.private_manifest.resolve().relative_to(private_root)
    except ValueError:
        print("private manifest must be inside the approved private root", file=sys.stderr)
        return 2
    if args.private_manifest.exists():
        print("private manifest path already exists", file=sys.stderr)
        return 2
    try:
        response = _query(args.page_size)
        records = parse_gdc_hits(filter_diagnostic_hits(extract_gdc_response_hits(response)))
        pilot = select_case_disjoint_pilot(records, per_class=args.per_class, seed=args.seed)
        queried_at = _timestamp()
        private_payload = {
            "schema_version": "1.0",
            "queried_at_utc": queried_at,
            "source": GDC_FILES_ENDPOINT,
            "seed": args.seed,
            "per_class": args.per_class,
            "records": [asdict(record) for record in pilot],
        }
        args.private_manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.private_manifest.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(private_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

        eligible_summary = manifest_summary(records)
        pilot_summary = manifest_summary(pilot)
        evidence = build_stage_evidence(
            stage="metadata_preflight",
            status="passed",
            timestamp_utc=queried_at,
            provenance={
                "source": GDC_FILES_ENDPOINT,
                "query_scope": "open primary-tumor TCGA-LUAD/TCGA-LUSC diagnostic DX slide metadata",
                "eligible_record_count": eligible_summary["record_count"],
                "eligible_case_count": eligible_summary["case_count"],
                "eligible_class_record_counts": eligible_summary["class_counts"],
                "eligible_manifest_sha256": eligible_summary["manifest_sha256"],
                "record_count": pilot_summary["record_count"],
                "case_count": pilot_summary["case_count"],
                "class_counts": pilot_summary["class_counts"],
                "manifest_sha256": pilot_summary["manifest_sha256"],
                "selection_seed": args.seed,
                "one_slide_per_case": True,
                "downloaded_slide_bytes": 0,
            },
        )
        write_evidence(args.evidence, evidence)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"metadata preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

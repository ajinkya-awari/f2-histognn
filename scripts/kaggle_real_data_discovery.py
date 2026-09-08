"""Discover and hash external real-data dependencies on private Kaggle compute."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.evidence import build_stage_evidence, write_evidence
from data.gdc import (
    build_gdc_query_payload,
    extract_gdc_response_hits,
    manifest_summary,
    parse_gdc_hits,
    select_case_disjoint_pilot,
)
from scripts.query_gdc_manifest import _query


HOVERNET_REVISION = "67e2ce5e3f1a64a2ece77ad1c24233653a9e0901"
HOVERNET_REPOSITORY = "https://github.com/vqdang/hover_net.git"
PANNUKE_WEIGHT_DRIVE_ID = "1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR"
PILOT_SEED = 17
PILOT_PER_CLASS = 3


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    working = Path("/kaggle/working")
    if not working.is_dir() or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        print("refusing to run real-data discovery outside a Kaggle kernel", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    private = Path("/kaggle/temp/project07-private")
    evidence_dir = working / "project07-evidence"
    private.mkdir(mode=0o700, parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp()

    response = _query(build_gdc_query_payload()["size"])
    records = parse_gdc_hits(extract_gdc_response_hits(response))
    pilot = select_case_disjoint_pilot(
        records, per_class=PILOT_PER_CLASS, seed=PILOT_SEED
    )
    pilot_summary = manifest_summary(pilot)

    checkout = private / "hover_net"
    if checkout.exists():
        raise RuntimeError("private HoVer-Net checkout already exists; refusing to overwrite")
    _run(["git", "clone", "--filter=blob:none", "--no-checkout", HOVERNET_REPOSITORY, str(checkout)])
    _run(["git", "checkout", "--detach", HOVERNET_REVISION], cwd=checkout)
    observed_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if observed_revision != HOVERNET_REVISION:
        raise RuntimeError("HoVer-Net revision mismatch")

    import gdown

    checkpoint = private / "hovernet_fast_pannuke_type_tf2pytorch.tar"
    if checkpoint.exists():
        checkpoint.unlink()
    result = gdown.download(id=PANNUKE_WEIGHT_DRIVE_ID, output=str(checkpoint), quiet=True)
    if result is None or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("HoVer-Net checkpoint download failed")
    checkpoint_sha256 = _sha256(checkpoint)

    evidence = build_stage_evidence(
        stage="dependency_discovery",
        status="blocked",
        timestamp_utc=timestamp,
        provenance={
            "next_action": "pin observed checkpoint SHA-256 before slide acquisition",
            "gdc_source": "https://api.gdc.cancer.gov/files",
            "manifest_sha256": pilot_summary["manifest_sha256"],
            "record_count": pilot_summary["record_count"],
            "case_count": pilot_summary["case_count"],
            "class_counts": pilot_summary["class_counts"],
            "selection_seed": PILOT_SEED,
            "downloaded_slide_bytes": 0,
            "hovernet_repository": HOVERNET_REPOSITORY,
            "hovernet_revision": HOVERNET_REVISION,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_byte_size": checkpoint.stat().st_size,
            "dependency_lock_sha256": _sha256(root / "requirements-kaggle-real-data.txt"),
        },
    )
    evidence_path = evidence_dir / f"dependency_discovery_{timestamp.replace(':', '')}.json"
    write_evidence(evidence_path, evidence)
    print(json.dumps(evidence, sort_keys=True))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

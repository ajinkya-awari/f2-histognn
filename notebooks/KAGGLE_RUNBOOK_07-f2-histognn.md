# Kaggle Runbook - 07-f2-histognn

This runbook accompanies `kaggle_run_07-f2-histognn.ipynb`. It is a bounded source-only execution plan. Any execution claim must cite downloaded Kaggle evidence.

## Current project state

The synthetic HistoGNN core and historical synthetic evidence exist. Historical evidence is `42 passed, 3 warnings`; it must not be treated as a fresh run for the notebook. No CellViT/TCGA-compatible artifact is verified, and no real metrics or explanations are claimed.

The source-only `requirements.txt` now pins the four imported runtime/test packages. Real-slide readers, a live artifact preflight entry point, and a real benchmark entry point remain absent by design.

## Supply the source tree

For CLI execution, generate a private staging notebook that embeds only the reviewed source allowlist and expands it to `/kaggle/working/07-f2-histognn` before the visible notebook cells run. This avoids inventing or publishing a Kaggle Dataset slug. The source bundle must exclude controls, caches, credentials, data, models, and evidence.

For manual UI execution, a private attached source dataset is also supported, but this project does not prescribe or assume its slug. The notebook selects exactly one tree containing:

```text
data/
models/
explanations/
training/
tests/
```

The staged source must also contain `README.md`, `DESIGN.md`, `requirements.txt`, `pyproject.toml`, and `notebooks/`. Do not attach patient-linked data or a CellViT/TCGA artifact.

## Cell order

Run one cell at a time and read its status and next-action message before continuing.

1. Safety and project configuration. Safe; no secrets or data access.
2. Inspect the staged working tree, then Kaggle inputs only as a manual fallback. Stops on ambiguity.
3. Use the already staged tree or copy the single manual input tree. Existing work is preserved under a timestamped backup.
4. Validate the working copy. Safe; stops before installation if the contract is incomplete.
5. Install dependencies from the exact source-only lock. Do not add ad hoc packages.
6. Kernel restart instruction. Informational; does not restart automatically.
7. Provider-free synthetic mode. Safe; removes known provider variable names from the process and does not print values.
8. Synthetic validation and bounded graph smoke. Writes dated text plus sanitized JSON containing versions, CUDA visibility, CPU test device, peak process memory, source hash, and explicit no-real-data/no-benchmark flags.
9. Evidence inspection. Safe; reads only the latest sanitized JSON and stops unless status is `passed`.
10. Approval gate. Safe; prints the live-access boundary.
11. Secret loading. Separate and not automatic. It reports only `secret loaded: True/False`; it remains blocked because no approved secret name is documented.
12. Bounded live preflight. Requires written approval naming the artifact and the allowed checks. It currently stops because no live preflight entry point is documented.
13. Final gated command. Requires written approval and an approved command. It currently stops because no training, evaluation, mini-gate, probe, or benchmark entry point is documented.

## Evidence

Evidence is written below the working tree:

```text
/kaggle/working/07-f2-histognn/evidence/
```

Cell 8 writes `synthetic_pytest_<UTC timestamp>.txt` and `synthetic_validation_<UTC timestamp>.json`. Cell 12 writes a sanitized blocked preflight record if run. Evidence is never deleted or overwritten by the notebook. If a cell fails, preserve the working directory and show the printed evidence path before stopping.

The JSON record contains status, timestamp, parsed test/warning counts when available, failure category, and an evidence path. It never stores raw provider responses, prompts, secrets, patient data, or exception text.

## Approval rules

Cells 1-10 are synthetic/configuration cells. Cell 11 is optional and must not be run without an approved secret name. Cell 12 requires explicit written approval naming a real artifact. Cell 13 requires separate approval for an exact benchmark command after the artifact gate passes.

Approval must name the artifact and authorize only the bounded rights, checksum-to-bytes, schema, provenance, label, and patient/case-group isolation checks. Approval for preflight is not approval for training or benchmarking.

Never automatically run GPU training, full evaluation, benchmark sweeps, W&B, Hugging Face, deployment, publication, email, Git actions, overnight work, or downloads.

## Exact next approved action

Run the private source-only kernel through Cell 10 and download its dated evidence. Do not proceed to real data until a named artifact and bounded preflight are independently approved and documented.

No real-data, provider, model-download, benchmark, training, or release result may be claimed without its own dated evidence.

## Verified private run — version 2, 2026-09-08

- Kernel: `ajinkya1225/07-f2-histognn`
- Terminal state: `KernelWorkerStatus.COMPLETE`
- Scope: source-only synthetic contracts and graph smoke
- Result: 62 passed, 3 warnings, 15.01 seconds
- Runtime: Python 3.12.13; NumPy 2.4.6; PyTorch 2.12.1; PyTorch Geometric 2.7.0; pytest 9.0.3
- Device: CPU; CUDA unavailable; zero visible GPUs; no GPU training
- Source-tree SHA-256: `272953acc4866fc39448fc7807f8565a0fbf12635e484726fe3809112d078344`
- Real-data gate: blocked record emitted; no artifact accessed
- Benchmark metrics: not emitted

Version 1 is retained as failed staging evidence. Its Windows ZIP separators were corrected by requiring POSIX archive member paths in version 2.

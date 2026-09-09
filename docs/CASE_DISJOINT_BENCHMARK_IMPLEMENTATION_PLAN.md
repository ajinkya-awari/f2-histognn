# Case-disjoint Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and privately execute the frozen 100-case TCGA LUAD/LUSC benchmark without changing the accepted scientific protocol after results are observed.

**Architecture:** Add independently tested cohort/split, case-level evaluation, and bootstrap modules, then compose them in a Kaggle-only runner that reuses the verified GDC, HoVer-Net, graph, model, and training contracts. Stream each slide through private temporary storage and publish only identifier-free aggregate evidence.

**Tech Stack:** Python 3.12, NumPy 1.26.4, PyTorch 2.7.1+cu126, PyTorch Geometric 2.7.0, OpenSlide, HoVer-Net, private Kaggle GPU.

## Global Constraints

- Use 100 distinct cases: 50 LUAD and 50 LUSC; one slide and four graphs per case.
- Use a stratified 60/20/20 case split with seed 17 and prove never expose case IDs publicly.
- Train all four existing architectures with seeds 17, 29, and 43 using only the frozen hyperparameters.
- Select checkpoints by validation loss; access test labels/predictions only once after selection.
- Emit no metric without complete source, split, dependency, device, seed, and artifact provenance.
- Keep slides, tiles, nuclei JSON, manifests, predictions, and checkpoints under `/kaggle/temp`.
- Do not modify `main`, publish raw artifacts, or present results as clinical evidence.

---

### Task 1: Deterministic stratified cohort and split

**Files:**
- Create: `data/benchmark.py`
- Modify: `data/__init__.py`
- Test: `tests/test_benchmark_contracts.py`

**Interfaces:**
- Consumes: `GDCSlideRecord` records already validated by `data.gdc`.
- Produces: `BenchmarkSplit`, `select_balanced_cases(records, per_class, seed)`, `stratified_case_split(records, seed, train_per_class, validation_per_class, test_per_class)`, and `sanitized_split_summary(split)`.

- [ ] Write focused tests proving order-independent balanced selection, exact 30/10/10-per-class allocation, one-slide-per-case, zero overlap, deterministic split hashing, and failure on insufficient/duplicate/ambiguous records.
- [ ] Run `python -m pytest -q tests/test_benchmark_contracts.py -k 'selection or split'` and confirm failures are caused by the missing interfaces.
- [ ] Implement immutable split records, canonical private hashing, and identifier-free summaries using only validated record fields.
- [ ] Run the focused tests and the full offline suite.
- [ ] Commit with `git commit -m "Add deterministic benchmark cohort split"`.

### Task 2: Case-level predictions and metrics

**Files:**
- Create: `training/evaluation.py`
- Modify: `training/__init__.py`
- Test: `tests/test_benchmark_contracts.py`

**Interfaces:**
- Consumes: tile-level two-class logits, private case keys, and binary labels.
- Produces: `aggregate_case_probabilities(...)`, `binary_case_metrics(...)`, and `stratified_bootstrap_intervals(..., resamples=2000, seed=...)`.

- [ ] Write tests proving four-tile case averaging, `LUAD=0`/`LUSC=1`, argmax accuracy/macro-F1, rank-based AUROC, average precision, deterministic percentile intervals, and rejection of unequal tile counts, conflicting labels, non-finite values, and single-class inputs.
- [ ] Run the focused evaluation tests and observe the expected missing-interface failures.
- [ ] Implement metrics directly with NumPy so the dependency lock does not grow; define ties and zero-denominator behavior explicitly and fail on undefined metrics.
- [ ] Run focused tests, then the full suite.
- [ ] Commit with `git commit -m "Add case-level benchmark evaluation"`.

### Task 3: Streamed private graph preparation

**Files:**
- Create: `data/benchmark_graphs.py`
- Create: `scripts/real_data_runtime.py`
- Modify: `scripts/kaggle_real_data_pilot.py` to import the existing download, tile, and HoVer-Net helpers from `scripts.real_data_runtime`
- Test: `tests/test_benchmark_contracts.py`

**Interfaces:**
- Consumes: one validated slide record at a time plus the pinned HoVer-Net runtime.
- Produces: four private PyG graph records with hashed case keys, labels, graph hashes, and aggregate preparation counters.

- [ ] Write tests using synthetic slide/HoVer-Net fixtures that prove per-slide cleanup, exactly four graphs per case, 512-node cap, deterministic hashes, label inheritance, and no cleanup outside the approved private root.
- [ ] Run focused preparation tests and observe expected failures.
- [ ] Implement streaming download → checksum → tile → HoVer-Net → graph conversion → raw-intermediate cleanup with fail-closed boundaries.
- [ ] Run focused and full offline suites.
- [ ] Commit with `git commit -m "Add streamed private graph preparation"`.

### Task 4: Frozen four-model benchmark runner

**Files:**
- Create: `scripts/kaggle_case_disjoint_benchmark.py`
- Modify: `training/loop.py`
- Test: `tests/test_benchmark_contracts.py`, `tests/test_training_contracts.py`

**Interfaces:**
- Consumes: private split graphs, `MODEL_REGISTRY`, `TrainingConfig`, and case-level evaluation functions.
- Produces: one private checkpoint/prediction set per model/seed and one sanitized aggregate evidence object after every run succeeds.

- [ ] Write tests proving frozen hyperparameters, validation-only checkpoint selection, restoration of the best state, exactly 12 model/seed runs, graph-balanced batching, test evaluation after training, CUDA-required behavior, and suppression of partial metrics after any failure.
- [ ] Run focused tests and observe failures for the missing runner/best-state restoration.
- [ ] Implement best-state restoration and the runner with Adam `0.001`, weight decay `0.0001`, 100 epochs, patience 10, hidden width 32, seeds 17/29/43, and deterministic loaders.
- [ ] Build identifier-free evidence containing separate seed results, means, bootstrap intervals, class/case/graph counts, hashes, device/runtime, and explicit non-claims.
- [ ] Run focused and full offline suites.
- [ ] Commit with `git commit -m "Add frozen case-disjoint benchmark runner"`.

### Task 5: Private Kaggle package and execution

**Files:**
- Create: `notebooks/kaggle_case_disjoint_benchmark.ipynb`
- Modify: `notebooks/KAGGLE_REAL_DATA_RUNBOOK.md`, `README.md`, `KAGGLE_UPLOAD_MANIFEST.md`
- Test: `tests/test_benchmark_contracts.py`

**Interfaces:**
- Consumes: an exact Git archive of the reviewed branch and the existing two Kaggle dependency locks.
- Produces: a private Kaggle kernel version and downloaded sanitized evidence only.

- [ ] Add static tests for zero notebook outputs, private/GPU/internet metadata, exact source attestation, approved dependency locks, output boundary, and absence of embedded credentials or identifiers.
- [ ] Run notebook/config tests, full pytest, compileall, JSON parsing, diff checks, and public residue/secret/data/model scans.
- [ ] Commit and normally push the reviewed draft branch; keep PR #2 draft and `main` unchanged.
- [ ] Build staging outside the repository, verify source revision/archive SHA-256, push one kernel version, and poll to a bounded terminal state.
- [ ] On `COMPLETE`, download outputs once, validate sanitized JSON/hash/logs, add only accepted evidence, rerun release scans/tests, commit, push, and update PR #2, issue #1, and internal maintenance records.
- [ ] On any failure, record the exact gate without accepting partial metrics; fix test-first and create a new reviewed version only after offline verification.

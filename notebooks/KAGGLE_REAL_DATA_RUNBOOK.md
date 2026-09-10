# Kaggle real-data runbook

Status: **graph-smoke passed on private kernel version 9**. This runbook is separate
from the source-only synthetic notebook.

## Fixed inputs

- GDC projects: `TCGA-LUAD`, `TCGA-LUSC`
- File contract: open-access `Slide Image`
- Group: TCGA case
- Pilot selection: seed 17, one slide per case, three cases per class
- HoVer-Net revision: `67e2ce5e3f1a64a2ece77ad1c24233653a9e0901`
- PanNuke checkpoint SHA-256:
  `4a1463467737f81203a0513f794276cbcbbd6bb470969584f314167c6acef081`
- HoVer-Net compatibility uses `imgaug==0.4.0` with `numpy==1.26.4` in the
  private Kaggle lock; its complete inference imports are checked before slide
  acquisition.
- Tiles: four deterministic tissue tiles per slide, normalized to 256 px at
  an effective 40x objective
- Graph cap: 512 nuclei per tile; deterministic farthest-point sampling

The PanNuke-derived checkpoint is CC BY-NC-SA 4.0. It is downloaded only into
private ephemeral Kaggle storage and is not redistributed.

## Execution

1. Stage the reviewed repository allowlist into `/kaggle/working/07-f2-histognn`.
2. Configure a private notebook with internet and GPU enabled, no attached
   datasets, models, competitions, or kernels.
3. Install `requirements-kaggle-torch-p100.txt`, then the additive
   `requirements-kaggle-real-data.txt`. The P100 lock uses the official PyTorch
   2.7.1 CUDA 12.6 wheel index, whose x86-64 build supports `sm_60`. Do not
   install the source-only CPU lock over this CUDA stack.
4. Run `scripts/kaggle_real_data_pilot.py` once.
5. Poll to `COMPLETE`, `ERROR`, `CANCELLED`, missing status, or the bounded
   timeout. The pre-push check must confirm `is_private: true`; the generated
   bootstrap attests that check to the runtime. Before acquisition, the script
   also requires Kaggle's `KAGGLE_KERNEL_RUN_TYPE` and
   `/kaggle/lib/kaggle/gcp.py` runtime markers, a full reviewed Git revision,
   and a byte-exact staged source-archive SHA-256 match.
6. Download outputs once. Inspect only `project07-evidence/*.json` as public
   evidence.

The script checks CUDA before acquiring slides. It then re-queries GDC,
validates the deterministic manifest, verifies each downloaded byte size and
MD5, extracts bounded tiles, runs nuclei inference, validates five
non-background probability features, builds deterministic graphs, and checks a
forward pass for GCN, GraphSAGE, GAT, and GraphGPS.

Raw slides, tiles, weights, nuclei JSON, and the external checkout live under
`/kaggle/temp/project07-private`; Kaggle does not publish that directory as
kernel output. Only sanitized evidence is written under `/kaggle/working`.

## Interpretation

A passed run establishes real-data acquisition, integrity, nuclei-schema,
graph-construction, GPU, and model-forward compatibility for a six-case smoke.
It does **not** establish classification performance, statistical validity,
external validity, or clinical usefulness. No accuracy, F1, AUROC, or AUPRC is
computed by this notebook.

A larger patient-disjoint benchmark requires a separate, prespecified cohort
size, split, training schedule, class-imbalance policy, seed set, and uncertainty
method after this smoke is accepted.

## Accepted smoke evidence

Kernel version 9 completed on 2026-09-09 from source revision
`2cbdb6f103de559aa6706d05a42527dbcff8feaa`. The sanitized record is
`evidence/real_data_graph_smoke_2026-09-09T115728Z.json` with SHA-256
`ae186e34aadfda523b6625dddefdbcef52309fae470f832a3a891982b9db1bc6`.
It records six cases, 24 tiles/graphs, 677 sampled nuclei, verified four-model
forward shapes, checksum-verified inputs, Tesla P100 execution, and zero
benchmark metrics or published raw artifacts.

## Frozen 100-case benchmark

The owner accepted `docs/CASE_DISJOINT_BENCHMARK_SPEC.md`. Execute
`notebooks/kaggle_case_disjoint_benchmark.ipynb` only from an exact reviewed
source archive after offline tests and release scans pass. The runner streams
100 checksum-verified slides through private temporary storage, creates four
graphs per case, enforces the 60/20/20 case-disjoint split, restores the lowest
validation-loss checkpoint, and evaluates the test cases once for each of four
models and three seeds. It writes no metric evidence unless all 12 runs finish.

Benchmark kernel version 10 stopped before metadata acquisition because the new
entry point passed a payload mapping to the shared GDC query helper instead of
its required integer page size. No slide bytes or metrics were produced. A
focused regression fixes this boundary before any subsequent version.

Version 11 stopped before slide acquisition at the initial individual-size cap.
The unchanged 100-case cohort needs 66,921,766,880 transferred bytes, not that
much resident disk. The reviewed streaming limits are now 3 GiB per slide,
70 GiB total, and a 2 GiB free-disk reserve. Staging must run the benchmark
subprocess with `timeout=14400`. No retry may change cohort membership,
splits, model policy, or select favourable metrics.

The runner now performs bounded synthetic CUDA forward/backward checks before
downloads, persists private manifests, rejects non-finite training, and records
selected classifier-state hashes, loss curves, policy, source archive hash, and
three-seed means in sanitized evidence. These implementation changes are not
evidence that a real benchmark has succeeded.

Version 12 passed the 16-graph x 512-node synthetic CUDA forward/backward
preflight for all four architectures, then failed on missing slide calibration
after 35 downloads (19,441,989,052 expected bytes through the failed slide;
approximately 7,824 seconds from kernel start). No classifier training started.
Version 14's short CPU diagnostic confirmed calibration was absent, not merely
stored under the usual alternative MPP fields. Version 15 verified GDC's bare
numeric Content-Range response syntax using four 64-byte probes.

The specification now records a pre-results calibrated-source eligibility
amendment. New cohort/split hashes must be frozen before training; old hashes
must not be reused. The next validation checks header-based selection and one
bounded full-file parallel transfer before another full benchmark. Version 13
was a diagnostic launcher serialization failure, corrected with an exercised
launcher test before version 14; it was not a benchmark/data-processing failure.

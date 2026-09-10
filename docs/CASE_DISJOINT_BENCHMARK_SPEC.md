# Case-disjoint benchmark specification

Status: **approved design; implemented; benchmark execution not yet verified**.

This document freezes the first exploratory F2 HistoGNN training benchmark
before outcome data are produced. The benchmark is not a clinical study and
must not be used for diagnostic claims.

## Cohort

- Source: open-access primary diagnostic slide images from `TCGA-LUAD` and
  `TCGA-LUSC` through the NCI Genomic Data Commons API.
- Cohort: 100 distinct TCGA cases, 50 per class, with one slide per case.
- Eligibility and labels reuse the validated GDC contract. Missing case IDs,
  ambiguous labels, controlled access, duplicates, and checksum failures fail
  closed.
- Cohort selection seed: 17. Eligible cases are sorted, permuted within class,
  and the first 50 per class are selected. Selection cannot depend on model
  output, nuclei count, graph properties, or downstream metrics.
- Individual slides are streamed, checksum-verified, tiled, and removed from
  temporary storage. Raw slides, identifiers, tiles, nuclei JSON, and weights
  are never public outputs.

## Split and leakage control

- Split unit: TCGA case. A case may occur in exactly one partition.
- Stratified allocation per class: 30 train, 10 validation, 10 test, producing
  60/20/20 cases overall.
- Split seed: 17. The complete private split manifest is hashed before graph
  construction; only its hash and aggregate class counts may be published.
- The test partition remains inaccessible to training, early stopping,
  threshold selection, architecture selection, and hyperparameter decisions.
- Four deterministic tiles are extracted per slide. Every tile inherits its
  case partition. Metrics are computed after aggregating tile probabilities to
  one case prediction, so tiles are not treated as independent patients.

## Graph and model contract

- HoVer-Net revision: `67e2ce5e3f1a64a2ece77ad1c24233653a9e0901`.
- PanNuke checkpoint SHA-256:
  `4a1463467737f81203a0513f794276cbcbbd6bb470969584f314167c6acef081`.
- Each node contains normalized `x`, normalized `y`, and five conditional
  non-background type probabilities. Each graph is capped at 512 nuclei using
  deterministic farthest-point sampling and uses `k=8` graph-local neighbors.
- Models: GCN, GraphSAGE, GAT, and GraphGPS with hidden width 32 and the existing
  two-layer/shared graph-level interface. No architecture substitution is
  allowed.

## Training policy

- Training seeds: 17, 29, and 43 for every architecture.
- Optimizer: Adam, learning rate `0.001`, weight decay `0.0001`.
- Maximum 100 epochs; early stopping patience 10; minimum improvement 0.
- Selection criterion: lowest validation cross-entropy loss. Test data are
  evaluated once using the selected epoch state.
- Each case contributes four graphs, so graph-level cross-entropy weights cases
  equally. If a case produces fewer than four valid graphs, the run fails
  rather than silently reweighting it.
- Class weights are not used because every partition is prespecified as class
  balanced. No resampling, augmentation, or hyperparameter search is allowed in
  this first benchmark.

## Prediction and metrics

- Tile logits are converted to softmax probabilities and averaged within case.
  The case label mapping is `LUAD=0`, `LUSC=1`; LUSC is the positive class.
- Report case-level accuracy, macro-F1, AUROC, and average precision (AUPRC).
  The decision rule for accuracy/F1 is argmax of the mean case probabilities.
- Report each seed separately and the mean across three seeds. No architecture
  is declared a winner from overlapping uncertainty intervals.
- For each seed and architecture, compute 95% percentile confidence intervals
  from 2,000 deterministic stratified bootstrap resamples of the 20 test cases,
  sampling within each class. Invalid single-class resamples fail closed.

## Execution and evidence

- Run only in the existing private Kaggle kernel with GPU and internet enabled.
- Freeze source revision, source archive hash, dependency-lock hash, manifest
  hash, split hash, preprocessing/graph hash, model configuration, seeds,
  runtime, CUDA/GPU, package versions, case/graph counts, and warnings.
- Raw artifacts and checkpoints remain under `/kaggle/temp`; only sanitized,
  identifier-free JSON is written under `/kaggle/working/project07-evidence`.
- Stop on insufficient eligible cases, disk/cap violation, checksum failure,
  case overlap, missing graphs, non-finite values, shape drift, CUDA loss, or
  incomplete provenance. Partial metrics from a failed run are not accepted.

## Operational preflight clarification (2026-09-09, before training)

The unchanged seed-17 cohort metadata totals 66,921,766,880 bytes; its largest
slide is 2,232,047,646 bytes. Kernel version 11 stopped before downloading slides
because the initial 2 GiB-per-slide/20 GiB-total limits were insufficient.
Execution now allows at most 3 GiB per slide and 70 GiB total transferred bytes,
with one slide resident at a time and 2 GiB free-disk reserve checked before
each download. No case is replaced to fit a size threshold. The staged runner
has a four-hour wall-clock timeout; this is a limit, not a completion estimate.

A synthetic 16-graph, 512-node-per-graph forward/backward optimizer step for each
architecture must pass on CUDA before real slide acquisition. Local equivalents
use tiny CPU fixtures only. Training uses batch size 16, one seed-shuffled batch
order reused across epochs, summed batch cross-entropy for backpropagation, and
mean per-graph losses for reporting. Non-finite logits, losses, gradients, or
updated parameters fail closed. Selected classifier states are hashed canonically;
weights and case predictions are not retained as public artifacts. Private cohort
and split manifests exist only in ephemeral Kaggle storage, so reproducing a run
after metadata changes requires recovering the same manifest; a hash alone is
not a substitute for its membership records.

## Pre-results calibration eligibility amendment (2026-09-10)

Version 12 proved all-model CUDA forward/backward compatibility but stopped
after 35 slide transfers, before nuclei inference or classifier training. One
source slide had no objective power, MPP, or other usable resolution metadata;
private diagnostic version 14 confirmed this. No eligible alternative slide
exists for that same case. Magnification must not be fabricated.

Under the owner's renewed authorization to resolve Project 07 execution issues,
the initial cohort is rejected before any model outcome is observed. The
following metadata-only eligibility refinement supersedes the original cohort
membership and its hashes, but not the 100-case, split, feature, model, seed,
training, or evaluation contracts:

- Rank GDC-eligible cases in each class using the existing sorted-ID, seed-17
  permutation. Within each case inspect diagnostic files in sorted UUID order.
- Select the first slide with bounded, readable Aperio TIFF calibration metadata
  explicitly declaring finite objective power between 10x and 80x. Do not infer
  objective power from MPP or from image appearance. Try another eligible slide
  of the same case before excluding the case. Select the first 50 calibrated
  cases per class from this ranking.
- Header transport errors abort rather than excluding a case. Missing, malformed,
  ambiguous or unsupported calibration is an explicit source-eligibility
  exclusion. Neither nuclei counts nor model/metric outcomes are consulted.
- Freeze new private cohort, calibration and case-disjoint split manifests before
  acquisition/training; report their new hashes and exclusion counts. Preserve
  the original manifest hash in failed-run evidence, not as the new cohort hash.
- Only small byte-range header probes precede whole-slide acquisition. Each used
  slide must still pass full byte-size/MD5 and OpenSlide validation.

The transport now uses at most four connections for disjoint ranges of one
private slide at a time. Exact numeric response ranges and full MD5 remain
mandatory; GDC's observed header form omits the optional `bytes ` prefix. Retry
traffic is bounded by twice each file's size and the remaining 70 GiB total
slide-transfer budget. Each file has a 600-second deadline. These are transport
changes, not permission to omit integrity checks. A short CPU-only header and
full-file transfer validation precedes the next GPU benchmark.

This eligibility amendment introduces a calibration/vendor availability
restriction. Results must not be generalized to uncalibrated slides or other
scanner formats without separate validation. No benchmark metric exists for
the rejected initial cohort; no test-set outcome informed this amendment.

## Interpretation boundary

This is a small, internally evaluated TCGA benchmark using sparse deterministic
slide sampling. It cannot establish external validity, calibration, robustness
to site/scanner shift, causal explanations, clinical utility, or deployment
readiness. A successful result remains exploratory and requires independent
external validation before any stronger claim.

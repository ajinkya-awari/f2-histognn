# Real-data benchmark protocol

Status: **approved; awaiting real-data evidence**.

This protocol defines the first real-data route for F2 HistoGNN. It does not
claim that a pilot or benchmark has succeeded. The released synthetic contracts
remain the only accepted execution evidence until the gates below pass.

## Scientific question

Evaluate whether spatial graphs of automatically detected nuclei can support an
exploratory distinction between TCGA lung adenocarcinoma (`LUAD`) and lung
squamous cell carcinoma (`LUSC`). Explanations remain exploratory sensitivity
diagnostics, not clinical or causal findings.

## Approved sources

- Open-access diagnostic slide images from `TCGA-LUAD` and `TCGA-LUSC`, queried
  through the NCI Genomic Data Commons API.
- HoVer-Net inference code from `vqdang/hover_net`, pinned to an exact revision.
- PanNuke-trained HoVer-Net weights, used under their
  CC BY-NC-SA 4.0 terms and pinned by SHA-256. Weights are never committed or
  redistributed by this repository.

The run must record the GDC data release/API response time, file UUID, file
name, project, case identifier, slide identifier when available, byte size,
open-access state, and supplied MD5. Downloaded bytes must be verified before
use. TCGA acknowledgements and HoVer-Net/PanNuke citations are mandatory.

## Cohort and leakage control

Only primary diagnostic slide images with an unambiguous `TCGA-LUAD` or
`TCGA-LUSC` project label are eligible. The TCGA case identifier is the grouping
unit. A deterministic ordering selects at most one diagnostic slide per case for
the first pilot; all slides from a case remain assigned to one partition in any
later expansion.

The pipeline must reject missing case identifiers, duplicate file UUIDs,
conflicting project labels, controlled-access records, checksum failures, empty
classes, or any case overlap across train, validation, and test.

The pilot is a pipeline validation, not a benchmark. Its exact bounded size is
selected from the live eligible manifest according to configured per-class
limits and recorded in evidence. Scaling beyond the pilot is allowed only after
the pilot evidence passes review.

## Nuclei and graph contract

HoVer-Net must run with PanNuke type prediction and probability output enabled.
Each nucleus probability vector is the mean of the model's per-pixel softmax
outputs over that nucleus mask; hard argmax label fractions are not accepted.
For every accepted nucleus, the adapter emits:

1. centroid `x` and `y` in slide coordinates;
2. probabilities for exactly five non-background PanNuke classes:
   neoplastic, inflammatory, connective, dead, and non-neoplastic epithelial.

Background is excluded rather than encoded as a sixth feature. The remaining
five values are renormalized to their conditional distribution given a
non-background class. A segmented nucleus whose type-head argmax is background
is retained only when its non-background probability mass is positive; zero
non-background mass fails closed. Records with missing/non-finite coordinates,
missing probabilities, values outside `[0, 1]`, or an incompatible probability
width fail closed. Coordinates are normalized within each graph by the existing
project contract. Spatial graph construction, sampling, and batching reuse the
deterministic tested implementation.

## Execution stages

1. **Metadata preflight:** query and validate the live GDC manifest without
   downloading slide bytes.
2. **Bounded acquisition:** download only the deterministic pilot files inside
   private Kaggle working storage and verify their bytes.
3. **Nuclei smoke:** run pinned HoVer-Net on a deterministic bounded tissue
   sample, then validate the seven-feature schema.
4. **Graph smoke:** construct graphs, validate batching and model forward passes,
   and emit sanitized provenance evidence.
5. **Pilot evaluation:** only after stages 1–4 pass, train/evaluate the smallest
   configured case-disjoint pilot and label all outputs `pilot`.
6. **Benchmark:** scale only after explicit pilot acceptance. Report uncertainty
   across prespecified seeds; never select or tune on the test partition.

Every stage is resumable and writes append-only sanitized JSON. Raw slides,
nuclei outputs, weights, checkpoints, and case-level records remain private and
must not enter Git or public Kaggle outputs.

## Evidence required for any metric

- dataset source, GDC release/query, eligibility filters, and manifest hash;
- license/usage terms and citations;
- included/excluded case and slide counts by class;
- group-disjoint split manifest hash and fixed seeds;
- HoVer-Net revision, weight URI identifier, and weight hash;
- source revision and dependency lock hash;
- Python, PyTorch, PyG, CUDA, GPU, peak memory, and runtime;
- graph counts, failure counts, and artifact hashes;
- accuracy, macro-F1, AUROC, and AUPRC definitions, positive class, and
  uncertainty method.

Absent evidence makes the corresponding result `not verified`, never zero and
never implicitly successful.

## GitHub state

Development occurs in a feature branch and draft pull request. `main` continues
to describe the verified synthetic release. The tracking issue and draft pull
request stay marked **awaiting real-data evidence** until the Kaggle artifacts
are downloaded, sanitized, reviewed, and linked. Merge is not automatic.

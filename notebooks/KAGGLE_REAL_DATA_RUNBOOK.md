# Kaggle real-data runbook

Status: **approved; graph-smoke evidence pending**. This runbook is separate
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
3. Install only the additive `requirements-kaggle-real-data.txt`. Do not install
   the source-only CPU lock over Kaggle's CUDA-enabled PyTorch stack.
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

# Kaggle Upload Manifest - Project 07 HistoGNN

This is a source-only manifest. A temporary private staging bundle may be created outside the project/public export for `kernels push`; no real artifact is included.

## Destination

Expand the reviewed source bundle into `/kaggle/working/07-f2-histognn`.

## Include

- `data/`, `models/`, `explanations/`, `training/`, and `tests/`
- `README.md`, `DESIGN.md`, `requirements.txt`, and `pyproject.toml`
- `notebooks/kaggle_run_07-f2-histognn.ipynb`
- For the separately approved real-data stages only: `requirements-kaggle-torch-p100.txt`, `requirements-kaggle-real-data.txt`, the reviewed real-data modules, `scripts/query_gdc_manifest.py`, `scripts/kaggle_real_data_discovery.py`, `scripts/kaggle_real_data_pilot.py`, `scripts/kaggle_case_disjoint_benchmark.py`, and the corresponding private-execution notebooks.

## Exclude

- `.git/`, `.pytest_cache/`, `__pycache__/`, `.venv/`, and local environments
- `data/raw/`, `data/processed/`, `results/`, `checkpoints/`, `wandb/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.h5`, `*.hdf5`, `*.svs`, `*.tif`, `*.tiff`, and archives
- Any patient-linked or restricted artifact unless separately approved
- Agent-control directories, prompts, handovers, internal plans, and tool logs

## Evidence paths

- `/kaggle/working/07-f2-histognn/evidence/synthetic_pytest_<UTC>.txt`
- `/kaggle/working/07-f2-histognn/evidence/synthetic_validation_<UTC>.json`

## Stop gates

Run the offline synthetic suite first and stop to show its evidence path. Ask for explicit approval before reading or handling any real CellViT/TCGA or patient-linked artifact. After approval, run only the bounded rights, checksum, schema, provenance, and split-isolation preflight. Stop again before GPU training, real benchmarking, W&B, Hugging Face, deployment, publication, email, or Git actions.

## Execution boundary

The private kernel metadata must set `is_private: true`, `enable_gpu: false`, and empty dataset/model/kernel/competition sources. Internet may be enabled only for the pinned dependency-install cell; project code must not make network requests. The generated bootstrap cell must be a byte-derived copy of the reviewed allowlist, not separately edited source. Its ZIP member names must use POSIX `/` separators so Python on Kaggle creates the intended directories. Stop after the synthetic evidence cell. GPU training and real benchmarking remain blocked.

Version 1 on 2026-09-07 failed before dependency installation because a Windows-built ZIP used backslash member names. Kaggle extracted those as literal filenames, so the source-tree directory check correctly failed. The committed notebook now has stable cell IDs and accepts an already staged working tree; a future staging build must normalize archive paths before any second push.

Version 2 on 2026-09-08 used POSIX-normalized ZIP members and completed the source-only synthetic pipeline: 62 passed, 3 warnings, 15.01 seconds on CPU. Archive SHA-256: `a692c091382d12be101ac2b2762821778009df804f7afc3c7a3f3b9cbcb8d71c`. Verified source-tree SHA-256: `272953acc4866fc39448fc7807f8565a0fbf12635e484726fe3809112d078344`.

The real-data protocol is a separate private execution boundary. It may enable GPU and internet only for the reviewed GDC/HoVer-Net workflow. All raw slides, weights, tiles, and nuclei outputs must stay under `/kaggle/temp/project07-private`; only sanitized aggregate JSON may be written under `/kaggle/working/project07-evidence`.

The accepted 100-case benchmark specification is implemented only by `notebooks/kaggle_case_disjoint_benchmark.ipynb`. Its staging metadata must remain private with GPU and internet enabled and with empty dataset, model, competition, and kernel sources. The runner must complete all 12 frozen model/seed runs before writing any of the 12 per-run metric evidence records.

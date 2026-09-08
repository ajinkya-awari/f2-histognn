# F2 HistoGNN — Approved Design

## 1. Scope

This document defines the approved implementation boundary for Project 07. The local synthetic core is implemented; verification evidence is historical unless a fresh command output is recorded. Real-data acceptance, GPU/Kaggle execution, and external release remain gated.

## 2. Pipeline

`verified manifest → schema validator → patient-grouped partition → deterministic graph builder → model benchmark → test-only explanations → evidence report`

The approved design requires each stage to emit a typed, inspectable artifact and failure evidence. The local synthetic core currently exposes typed values and named validation errors; a real-data report layer is not present. A later stage must not silently repair an earlier contract violation.

## 3. Data model

Required source fields:

- `centroid`: finite numeric array `[N, 2]`.
- `type_prob`: finite numeric array `[N, 5]` with every probability in `[0, 1]`.
- A manifest-resolved class label and patient/case identifier.

Derived node fields:

- `x_norm`, `y_norm`: coordinates normalized within the graph’s documented frame.
- `type_prob_0` through `type_prob_4`: the five source probabilities.

Therefore `x` is `[N, 7]`. The feature count must be defined once in configuration and asserted by tests.

Derived edge fields:

- k-NN connectivity with a documented tie policy.
- `edge_attr[:, 0]`: Euclidean distance normalized by the maximum selected-edge distance in the current graph, with values in `[0, 1]`. This graph-local rule is implemented for synthetic fixtures; a real run must freeze and record it before processing. Model interfaces reject non-floating, cross-device, or unnormalized edge tensors.

No raw patient identifier belongs in model features or public artifacts.

## 4. Graph construction

Use deterministic k-NN construction with a fixed k and deterministic tie handling. Use PyG FPS with `random_start=False` when the graph exceeds the configured node cap. Test both the under-cap path and the capped path. Report dropped, empty, duplicate, and non-finite nodes rather than silently masking them.

## 5. Partitioning

Derive the grouping key before any graph-level split. Group identifiers are mutually exclusive across train, validation, and test. The test set is untouched until final evaluation and explanation. Add an assertion that fails if a group appears in more than one partition.

## 6. Model interface

All four implemented model surfaces expose the same contract:

`forward(x, edge_index, edge_attr, batch) -> logits [num_graphs, 2]`

The edge attribute is one-dimensional at input and must be projected to the hidden dimension before any GINE/GPS operation that requires hidden-width edge features. The model registry, seed policy, optimizer policy, and checkpoint metadata must be shared.

## 7. Explanation interface

The implemented graph-level explainer wrapper accepts `x`, `edge_index`, and optional `edge_attr`; it creates a zero batch for a single graph, uses graph-level task semantics, returns log probabilities, and does not pass an index for graph-level explanation. It rejects malformed/non-tensor inputs and NaN values. Faithfulness diagnostics establish the unperturbed baseline before edge perturbation. The method is input-gradient based, not GNNExplainer; explanations are exploratory evidence, not causal attribution.

## 8. Evaluation

Primary metrics: ROC-AUC and a class-balanced metric selected before seeing results. Secondary metrics may include PR-AUC, balanced accuracy, calibration, parameter count, and runtime. Report per-seed values and aggregate uncertainty where feasible. Do not encode a target score or presumed GraphGPS win.

Explanation diagnostics must include a faithfulness test and a stability comparison across repeated explainer seeds or equivalent perturbations. A visually attractive explanation is not sufficient evidence.

## 9. Evidence and provenance

Every result records source identifier, checksum, schema version, split manifest hash, configuration hash, code revision, environment summary, seed, and whether computation occurred locally or on Kaggle. Results without provenance are incomplete.

## Implementation status

Implemented locally: manifest/schema validation with declared rights/provenance and byte-level SHA-256 verification, explicit label and group parsing, grouped splits, deterministic k-NN/FPS graphs, the four model interfaces, test-only input-gradient explanations, diagnostics, and a provenance-aware training harness. Historical verification is synthetic-only; current hardening tests are written but not locally executed. The CellViT/TCGA artifact gate, real-data processing, benchmark metrics, Kaggle/GPU execution, and release remain unresolved.

## Audit addendum - 2026-08-29

Static reconciliation found the implementation surfaces above still present. It did not run Python, pytest, Kaggle, GPU work, providers, APIs, deployment, commit, or push. The current project label is `PARTIAL` / `LOCAL-IMPLEMENTATION` / `IMPLEMENTED-UNVERIFIED` because the full current test tree lacks fresh execution evidence. The next design-preserving task is an approved Kaggle `requirements.txt` dependency lock for source-only synthetic validation; no feature, split, model, explanation, privacy, or benchmark-methodology contract change is approved by this audit.

## 10. Non-goals

No clinical decision support, causal interpretation, patient-data publication, external deployment, automatic W&B logging, Hugging Face upload, email, or cross-project dependency is part of this project’s baseline.

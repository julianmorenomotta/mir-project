# Implementation Plan — Query-Bandit Zero-Shot Evaluation on Unseen Music Data
> **How to use this document**
> - This file is the single source of truth for backlog, task assignment, and progress tracking.
> - Mark tasks with `[x]` when done. Add your initials and date in parentheses, e.g. `[x] (julian, 2026-03-10)`.
> - Keep sub-tasks granular enough that a teammate can pick one up independently.
> - Never delete a completed item — it is the audit trail of what has already been done.
> - If a task is blocked, prepend `[BLOCKED]` to its line and add a note below it explaining why.

---

## Table of Contents
1. [Current Main Focus (April 2026)](#0-current-main-focus-april-5-2026)
2. [Project Overview](#1-project-overview)
3. [Milestone Map](#2-milestone-map)
4. [Phase A — Scope & Environment Freeze](#phase-a--scope--environment-freeze)
5. [Phase B — BabySlakh Eval Data Preparation](#phase-b--babyslakh-eval-data-preparation)
6. [Phase C — Config & Smoke Validation](#phase-c--config--smoke-validation)
7. [Phase D — Researcher Pipeline Execution](#phase-d--researcher-pipeline-execution)
8. [Phase E — Reporting & Interpretation](#phase-e--reporting--interpretation)
9. [Phase F — Reproducibility Closeout](#phase-f--reproducibility-closeout)
10. [Open Questions & Decisions Log](#open-questions--decisions-log)
11. [References](#references)

---

## 0. Current Main Focus (April 5 2026)

**Priority override:** The active objective right now is to complete a same-afternoon, evaluation-only **zero-shot generalization test** of Query-Bandit on an unseen musical dataset, starting with BabySlakh. Until this is complete, this workstream is the sole active plan.

**Target outcome (first pass):**
- End-to-end pretrained evaluation on BabySlakh (all available tracks)
- Built-in query-bandit test metrics from `query_test`
- Notebook-style metric summary from saved inference audio
- Clear notes on what can and cannot be concluded from this first-pass OOD test

**Included in scope now:**
- Evaluation-only path (no training)
- BabySlakh-only first pass using all available tracks
- Reuse of researcher testing entrypoints in `third_party/query-bandit/train.py`
- Minimal data preparation required to make BabySlakh compatible with query-bandit evaluation flow

**Out of scope now:**
- Any model training/fine-tuning or architecture changes
- Multi-dataset benchmarking beyond BabySlakh
- Full paper replication claims

### 0.1 Immediate Execution Checklist (This Afternoon)

#### Phase A: Freeze run scope and reproducibility envelope
- [ ] Set run identifiers and constants:
  - [ ] `EVAL_TAG=<short run id, e.g. babyslakh-zs-v1>`
  - [ ] `EVAL_CONFIG=third_party/query-bandit/config/bandit-babyslakh.yml`
  - [ ] `EVAL_CKPT=models/ev-pre-aug.ckpt` (or exact selected checkpoint path)
- [ ] Standardize env vars for this run:
  - [ ] `CONFIG_ROOT=third_party/query-bandit/config`
  - [ ] `DATA_ROOT=<parent of babyslakh working root>`
  - [ ] `LOG_ROOT=<writable local output root>`

#### Phase B: Prepare BabySlakh in query-bandit-compatible eval layout
- [ ] Use all available BabySlakh tracks under `data/datasets/babyslakh_16k/`
- [ ] Materialize/verify evaluation-ready structure at `${DATA_ROOT}/babyslakh_16k`:
  - [ ] `test/mixture_xxxxx/{mixture.wav, individual_*.wav, metadata.json}`
  - [ ] `queries/<individual_or_target_id>/query_*.wav`
- [ ] Verify each test sample has a valid target-query pairing and metadata lineage (track + stem origin)
- [ ] Validate sample-rate assumptions used by the chosen config (explicitly document any resampling)

#### Phase C: Make pretrained eval config runnable for BabySlakh
- [ ] Create `third_party/query-bandit/config/bandit-babyslakh.yml` from `third_party/query-bandit/config/bandit-macaque.yml`
- [ ] Point `data.data_root` to BabySlakh working root and set evaluation-safe data/test kwargs
- [ ] Confirm stem/target naming used in config matches generated metadata/query directories
- [ ] Run smoke load test: instantiate model + load `${EVAL_CKPT}` successfully

#### Phase D: Execute researcher test pipeline
- [ ] Run objective metrics pass (built-in):
  ```bash
  python third_party/query-bandit/train.py query_test \
    --config_path=${EVAL_CONFIG} \
    --ckpt_path=${EVAL_CKPT}
  ```
- [ ] Run audio export pass for notebook-based analysis:
  ```bash
  python third_party/query-bandit/train.py query_inference \
    --config_path=${EVAL_CONFIG} \
    --ckpt_path=${EVAL_CKPT}
  ```
- [ ] Confirm outputs are written under logger tree and audio artifacts are present

#### Phase E: Report first-pass usefulness on unseen music data
- [ ] Run `third_party/query-bandit/notebooks/evaluate_ev_query.ipynb` on local inference outputs
- [ ] Produce an afternoon summary table with:
  - [ ] run id, config path, checkpoint path
  - [ ] key query_test metrics (from logger CSV)
  - [ ] notebook summary metrics from saved audio
  - [ ] 3-5 qualitative notes on observed behavior/failure modes
- [ ] Add a short interpretation note explicitly labeling conclusions as first-pass directional evidence

#### Phase F: Reproducibility and closeout notes
- [ ] Record exact command lines, env vars, and output directories used
- [ ] Record known caveats (query construction assumptions, stem taxonomy mismatch, domain shift effects)
- [ ] Mark completed afternoon tasks with initials/date in this plan

### 0.2 Success Criteria for this focus
- [ ] End-to-end run succeeds on BabySlakh using pretrained checkpoint and researcher evaluation entrypoints
- [ ] Built-in query_test metrics and notebook-compatible inference outputs are generated
- [ ] First-pass results table is produced with reproducibility metadata and caveats

## 1. Project Overview

**Goal:** Assess the practical usefulness of pretrained Query-Bandit on out-of-distribution music data by running the original researcher evaluation workflow on BabySlakh without retraining.

**Scope (current pass):**
- Dataset: BabySlakh (all available tracks in local copy)
- Method: evaluation-only using pretrained checkpoint(s)
- Pipeline: researcher-provided `query_test` and `query_inference` entrypoints
- Outputs: built-in metrics, saved inference audio, and notebook-style summary table

**Core deliverable:** A reproducible first-pass zero-shot report that captures objective metrics and qualitative behavior of Query-Bandit on unseen music data.

**Non-goals for this run:**
- Any training/fine-tuning
- Architecture or loss modifications
- Multi-dataset benchmarking beyond BabySlakh
- Definitive benchmark claims beyond directional first-pass evidence

**Banquet source:** `github.com/kwatcharasupat/query-bandit` — cloned into `third_party/query-bandit/` (read-only reference; we import from it, not modify it)

---

## 2. Milestone Map

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| M-A | Scope/env freeze completed and run identifiers set | This afternoon | ⬜ |
| M-B | BabySlakh evaluation dataset layout validated | This afternoon | ⬜ |
| M-C | BabySlakh eval config created and smoke-tested | This afternoon | ⬜ |
| M-D | query_test and query_inference completed | This afternoon | ⬜ |
| M-E | Notebook summary and results table produced | This afternoon | ⬜ |
| M-F | Reproducibility notes and caveats logged | This afternoon | ⬜ |

> Status key: ⬜ not started · 🔄 in progress · ✅ done · 🚫 blocked

---

## Phase A — Scope & Environment Freeze

- [ ] Set run identifiers and constants:
  - [ ] `EVAL_TAG=<short run id, e.g. babyslakh-zs-v1>`
  - [ ] `EVAL_CONFIG=third_party/query-bandit/config/bandit-babyslakh.yml`
  - [ ] `EVAL_CKPT=models/ev-pre-aug.ckpt` (or exact selected checkpoint path)
- [ ] Standardize env vars:
  - [ ] `CONFIG_ROOT=third_party/query-bandit/config`
  - [ ] `DATA_ROOT=<parent of babyslakh working root>`
  - [ ] `LOG_ROOT=<writable local output root>`

## Phase B — BabySlakh Eval Data Preparation

- [ ] Use all available BabySlakh tracks under `data/datasets/babyslakh_16k/`
- [ ] Materialize/verify evaluation-ready structure at `${DATA_ROOT}/babyslakh_16k`:
  - [ ] `test/mixture_xxxxx/{mixture.wav, individual_*.wav, metadata.json}`
  - [ ] `queries/<individual_or_target_id>/query_*.wav`
- [ ] Verify each test sample has a valid target-query pairing and metadata lineage (track + stem origin)
- [ ] Validate sample-rate assumptions used by the chosen config (document any resampling)

## Phase C — Config & Smoke Validation

- [ ] Create `third_party/query-bandit/config/bandit-babyslakh.yml` from `third_party/query-bandit/config/bandit-macaque.yml`
- [ ] Point `data.data_root` to BabySlakh working root and set evaluation-safe data/test kwargs
- [ ] Confirm stem/target naming in config matches generated metadata/query directories
- [ ] Run smoke load test: instantiate model + load `${EVAL_CKPT}` successfully

## Phase D — Researcher Pipeline Execution

- [ ] Run objective metrics pass:
  ```bash
  python third_party/query-bandit/train.py query_test \
    --config_path=${EVAL_CONFIG} \
    --ckpt_path=${EVAL_CKPT}
  ```
- [ ] Run audio export pass:
  ```bash
  python third_party/query-bandit/train.py query_inference \
    --config_path=${EVAL_CONFIG} \
    --ckpt_path=${EVAL_CKPT}
  ```
- [ ] Confirm logger outputs and exported audio artifacts are present

## Phase E — Reporting & Interpretation

- [ ] Run `third_party/query-bandit/notebooks/evaluate_ev_query.ipynb` on local inference outputs
- [ ] Produce first-pass summary table with:
  - [ ] run id, config path, checkpoint path
  - [ ] key `query_test` metrics (from logger CSV)
  - [ ] notebook summary metrics from saved audio
  - [ ] 3-5 qualitative notes on observed behavior/failure modes
- [ ] Add interpretation note: first-pass directional evidence only (not a definitive benchmark)

## Phase F — Reproducibility Closeout

- [ ] Record exact command lines, env vars, output directories, and config/checkpoint identifiers
- [ ] Record known caveats (query construction assumptions, stem taxonomy mismatch, domain shift effects)
- [ ] Mark completed items with initials/date in this plan

---

## Open Questions & Decisions Log

> Add new entries as questions arise. Mark resolved ones with ✅.

| # | Question | Decision | Resolved By | Date |
|---|----------|----------|-------------|------|
| 1 | Which checkpoint should be treated as primary for this afternoon run? | `models/ev-pre-aug.ckpt` unless smoke test fails | — | — |
| 2 | What BabySlakh stem mapping policy should be used for target/query ids? | TBD | — | — |
| 3 | Should we report only built-in metrics or include notebook summary from saved audio? | Include both for this first pass | — | — |

---

## References

- Watcharasupat, K. N., & Lerch, A. (2024). A stem-agnostic single-decoder system for music source separation beyond four stems. *ISMIR 2024*. arXiv:2406.18747.
- BabySlakh dataset: https://zenodo.org/records/4603879
- Query-Bandit codebase: https://github.com/kwatcharasupat/query-bandit

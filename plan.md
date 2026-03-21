# Implementation Plan — Query-Conditioned Bioacoustic Source Separation
> **How to use this document**
> - This file is the single source of truth for backlog, task assignment, and progress tracking.
> - Mark tasks with `[x]` when done. Add your initials and date in parentheses, e.g. `[x] (julia, 2026-03-10)`.
> - Keep sub-tasks granular enough that a teammate can pick one up independently.
> - Never delete a completed item.
> - If a task is blocked, prepend `[BLOCKED]` to its line and add a note below it explaining why.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Milestone Map](#2-milestone-map)
3. [Phase 0 — Environment & Repository Setup](#phase-0--environment--repository-setup)
4. [Phase 1 — Data Pipeline](#phase-1--data-pipeline)
5. [Phase 2 — Baselines](#phase-2--baselines)
6. [Phase 3 — Minimal Banquet Adaptation to Bioacoustics](#phase-3--minimal-banquet-adaptation-to-bioacoustics)
7. [Phase 4 — Training & Evaluation](#phase-4--training--evaluation)
8. [Phase 5 — Analysis & Deliverables](#phase-5--analysis--deliverables)
9. [Open Questions & Decisions Log](#open-questions--decisions-log)
10. [References](#references)



---

## 1. Project Overview

**Goal:** Adapt the Banquet query-conditioned source separation framework (Watcharasupat & Lerch, 2024) to bioacoustic recordings, using a bio-acustics dataset (eg. bottlenose dolphin signature whistles or macaque vocalizations TBD) as the primary testbed. The system accepts a short enrollment clip of the target individual, encodes it via a frozen PaSST model, and uses the resulting embedding to condition a bandsplit decoder via FiLM — steering the network to extract the matching individual and suppress others.

**Scope (POC):**
- Adaptation of the original query bandit to a new dataset (Bioacustics)
- Task: two-speaker mixture separation
- Primary metric: SI-SDR / SI-SDRi; secondary: individual identity classification accuracy on separated outputs

**Core deliverable:** A working query-conditioned separation model for dolphin vocalizations, benchmarked against BioCPPNet and a Conv-TasNet query baseline, built via minimal adaptation of Banquet interfaces.

**Banquet source:** `github.com/kwatcharasupat/query-bandit` — cloned into `third_party/query-bandit/`

**Implementation strategy (locked for P0):** Minimal adaptation. Reuse Banquet model/loss/inference components as-is, and add only local glue code (dataset, datamodule, training entrypoint shim, configs) needed for non-music bioacoustic data.

### Resources
- [Datasets](https://github.com/earthspecies/library?tab=readme-ov-file)

---

## Phase 0 — Environment & Repository Setup

### 0.1 Repository
- [x] Initialise this repo (`mir-project`) as the single working repo for the team
- [ ] Add all team members as collaborators
- [x] Create branch naming convention: `feature/<phase>-<short-desc>`, `fix/<desc>`
- [x] Clone Banquet into `third_party/` as a **read-only reference** (do not modify files inside it):
  ```bash
  git clone https://github.com/kwatcharasupat/query-bandit third_party/query-bandit
  ```
- [x] Add `third_party/` to `.gitignore` so it is not tracked by our repo
- [x] Add `.gitignore` entries for data directories, checkpoints, and generated artifacts

### 0.2 Python Environment
- [x] Pin Python version (recommended: 3.10)
- [x] Create `environment.yml` (conda) or `requirements.txt` + `pyproject.toml`
- [x] Verify Banquet core package imports cleanly from `third_party/` (run from project root):
  ```bash
  python -c "import sys; sys.path.insert(0, 'third_party/query-bandit'); import core; print('OK', core.__file__)"
  ```
  > Note: the package name is `core`, not `bandit`. It is not installed — it must be on `sys.path` first.
- [x] Add `torchaudio`, `numpy`, `scikit-learn`, `librosa`, `soundfile` to dependencies
- [ ] Document GPU requirements and tested CUDA version in README

### 0.3 Project Structure
- [x] Create directory layout:
  ```
  mir-project/
  ├── third_party/
  │   └── query-bandit/   # Banquet clone — read-only, not tracked by git
  ├── data/
  │   ├── raw/            # original recordings, never modified
  │   ├── processed/      # resampled, segmented clips
  │   └── mixtures/       # synthesized two-speaker mixtures + metadata
  ├── embeddings/         # precomputed PaSST embeddings
  ├── checkpoints/        # model checkpoints
  ├── results/            # evaluation outputs, tables, figures
  ├── src/                # all source code modules (our code only)
  ├── notebooks/          # exploratory analysis notebooks
  ├── configs/            # YAML config files per experiment
  ├── plan.md             # this file
  └── proposal.md
  ```
- [x] Add `src/__init__.py` and stub modules matching the planned architecture
- [ ] Set up a basic logging config (use Python `logging` or `wandb`)

### 0.4 Continuous Integration (optional)
- [ ] Add a GitHub Actions workflow that runs `pytest` on push
- [ ] Add a `tests/` directory with at least one smoke test for data loading
- [ ] Add a smoke test that verifies `core` imports correctly from `third_party/` — catches broken clones for new team members

---

## Phase 1 — Data Pipeline

### 1.1 Data Acquisition
 Data can be obtained from [here](https://seamap.env.duke.edu/dataset/563)
- [ ] Obtain bottlenose dolphin signature whistle dataset (Sarasota Dolphin Research Program)
  - [ ] Document expected file format (wav/flac, sample rate, channel count)
  - [ ] Place raw files in `data/raw/dolphins/`
- [ ] (Optional) Obtain macaque coo-call dataset — only if time permits for generalization analysis
  - [ ] Place raw files in `data/raw/macaques/`
- [ ] (Optional) Obtain Egyptian fruit bat vocalizations — only if time permits
  - [ ] Place raw files in `data/raw/bats/`
- [ ] Write `src/data/verify_raw.py` — checksums / file-count sanity checks

### 1.2 Preprocessing
- [ ] Write `src/data/preprocess.py` with the following steps:
  - [ ] Resample all audio to **32 kHz** (PaSST compatibility requirement)
  - [ ] Convert to mono if multi-channel
  - [ ] Segment recordings into fixed-length clips (e.g., 4 s) with 50% overlap
  - [ ] Assign individual identity labels from metadata files
  - [ ] Filter out clips below an energy threshold (silence removal)
  - [ ] Save processed clips to `data/processed/<species>/<individual_id>/clip_NNNN.wav`
- [ ] Write unit test: verify at least N clips per individual, all at 32 kHz
- [ ] Log summary statistics (clips per individual, total duration) as `data/processed/stats.json`

### 1.3 Train / Validation / Test Splits
- [ ] Define split strategy: **individual-level** (no individual appears in more than one split)
- [ ] Write `src/data/split.py` to generate reproducible splits (fix random seed)
- [ ] Save split manifests as `data/processed/splits_<species>.json`
  - Structure: `{"train": [...], "val": [...], "test": [...]}`

### 1.4 Mixture Generation
- [ ] Write `src/data/mixture.py`:
  - [ ] For each mixture: randomly draw 2 individuals (one target, one interferer) from the same split
  - [ ] Randomly select non-overlapping source clips
  - [ ] Mix at randomized SNR offset ∈ [0, 5] dB (uniform), following BioCPPNet protocol
  - [ ] Save mixture as `data/mixtures/<split>/<mixture_id>/mix.wav`
  - [ ] Save clean sources as `s1.wav`, `s2.wav` alongside mixture
  - [ ] Save metadata as `meta.json` (individual IDs, SNR offset, source clip paths)
- [ ] Generate N_train, N_val, N_test mixtures (decide counts — suggested: 10k/1k/1k for dolphins)
- [ ] Write validation notebook: `notebooks/01_data_inspection.ipynb` — visualize spectrograms and confirm mixture quality

### 1.5 Query Clip Extraction
- [ ] Write `src/data/query.py`:
  - [ ] For each mixture, select a **held-out enrollment clip** from the target individual (not used in the mixture)
  - [ ] Enrollment clips must come from the **same split but different segments**
  - [ ] Save as `data/mixtures/<split>/<mixture_id>/query.wav`
- [ ] Verify that no query clip is acoustically identical to any source segment in its associated mixture

---

## Phase 2 — Baselines

### 2.1 BioCPPNet Baseline
- [ ] Clone or obtain BioCPPNet code (Bermant 2021) — check if public repo exists, otherwise re-implement from paper
- [ ] Adapt BioCPPNet data loader to accept our mixture format
- [ ] Train BioCPPNet on dolphin two-speaker mixtures
- [ ] Evaluate on test set: compute SI-SDR, SI-SDRi
- [ ] Save results to `results/biocppnet_dolphins.json`

### 2.2 Conv-TasNet + Dot-Product Attention Baseline
- [ ] Implement query-conditioned Conv-TasNet adapter in `src/baselines/convtasnet_query.py`
  - [ ] Use standard Conv-TasNet encoder/decoder
  - [ ] Inject query via dot-product attention on the encoder feature map
- [ ] Train on dolphin mixtures with the same query clips
- [ ] Evaluate: SI-SDR, SI-SDRi
- [ ] Save results to `results/convtasnet_query_dolphins.json`

---

## Phase 3 — Minimal Banquet Adaptation to Bioacoustics

> **Goal:** Keep `third_party/query-bandit/` unchanged for P0. Adapt to bioacoustics by implementing only minimal local glue code around Banquet's existing interfaces.

### 3.1 Interface Audit (Contract First)
- [ ] Read through `third_party/query-bandit/` and lock the required batch contract for training/inference:
  - [ ] Required fields: `mixture`, `sources`, `query`, `metadata`, `estimates`
  - [ ] Target convention: `sources["target"]` and `estimates["target"]`
  - [ ] Required tensor shape conventions for `(B, C, T)` audio
- [ ] Document exactly which Banquet components are reused unchanged vs. wrapped in local code
- [ ] Add `third_party/query-bandit` to `sys.path` in `src/utils/paths.py` for clean imports

### 3.2 Local Data Adapter (New Code Only)
- [ ] Write `src/data/query_bio/dataset.py` implementing a non-music dataset that emits Banquet-compatible samples
- [ ] Write `src/data/query_bio/datamodule.py` with the same constructor pattern used by Banquet datamodules (`data_root`, `batch_size`, `num_workers`, `train_kwargs`, `val_kwargs`, `test_kwargs`, `datamodule_kwargs`)
- [ ] Confirm train/val/test loaders work with dolphin mixtures and held-out query clips
- [ ] Add smoke test: one real batch through the model forward pass

### 3.3 Local Training Entrypoint Shim
- [ ] Create `src/train_query_bandit.py` as a thin local runner that mirrors Banquet training setup but registers our bio datamodule
- [ ] Keep `third_party/query-bandit/train.py` unmodified in P0
- [ ] Support checkpoint resume/fine-tune via `ckpt_path`
- [ ] Support model config options `pretrain_encoder` and `freeze_encoder`
- [ ] Add minimal CLI examples to README/notes for train/validate/inference

### 3.4 Config Layer for Bioacoustics
- [ ] Create `configs/query_bandit/models/bandit-query-pre-bio.yml`
- [ ] Create `configs/query_bandit/data/bio-query-d.yml`
- [ ] Create `configs/query_bandit/expt/bio-query-pre-d.yml` linking model/data/loss/optim/trainer
- [ ] Create one tiny overfit/smoke config for fast end-to-end validation

### 3.5 Query Encoder and Assumptions (P0 Scope)
- [ ] Keep frozen PaSST query encoder for P0 baseline
- [ ] Evaluate embedding quality on dolphins (`notebooks/02_embedding_exploration.ipynb`)
- [ ] Document known assumptions and mitigations:
  - [ ] Audio sample rate expectation (44.1 kHz in current Banquet configs)
  - [ ] Query duration expectation (10 s in current setup)
  - [ ] Local preprocessing/resampling policy for dolphin data
- [ ] Defer query-encoder fine-tuning/replacement until after first stable baseline run

---

## Phase 4 — Training & Evaluation

### 4.1 Loss Function
- [ ] Reuse Banquet L1SNR-based losses from `third_party/query-bandit` via config (no local reimplementation in P0)
- [ ] Add one integration test confirming loss computes on a dolphin batch without shape/key mismatches

### 4.2 Training
- [ ] Implement and use `src/train_query_bandit.py` (minimal shim)
  - [ ] Config-driven via YAML (`configs/query_bandit/expt/bio-query-pre-d.yml`)
  - [ ] Log loss curves to `wandb` or tensorboard
  - [ ] Save checkpoints to `checkpoints/`
  - [ ] Implement early stopping on validation SI-SDRi
- [ ] Run at least one training run from pretrained weights and one from scratch for comparison
- [ ] Train on dolphin mixtures
- [ ] Save best checkpoint and training log

### 4.3 Evaluation
- [ ] Write `src/evaluate.py`:
  - [ ] Load checkpoint and test mixtures
  - [ ] Compute per-mixture SI-SDR, SI-SDRi, and aggregate mean ± std
  - [ ] Save results to `results/banquet_bio_dolphins.json`
- [ ] Evaluate all models on the dolphin test set:
  - [ ] BioCPPNet
  - [ ] Conv-TasNet + dot-product query
  - [ ] Banquet point-query (ours)
- [ ] Report: mean SI-SDR, SI-SDRi, std across test set
- [ ] Compile into `results/evaluation_table.csv`

### 4.4 Identity Classification (Secondary Metric)
- [ ] Train a simple individual identity classifier (e.g., SVM on PaSST embeddings of clean clips)
- [ ] Run classifier on separation outputs for each model
- [ ] Compute classification accuracy per model
- [ ] Add column to `results/evaluation_table.csv`

---

## Phase 5 — Analysis & Deliverables

### 5.1 Qualitative Analysis
- [ ] Create `notebooks/03_qualitative_analysis.ipynb`:
  - [ ] Select 5–10 representative examples per model: plot mixture, reference source, and estimated source spectrograms
  - [ ] Identify recurring failure patterns (e.g., spectrally overlapping calls, low-SNR mixtures)
  - [ ] Assess whether the model degrades gracefully or collapses on hard cases

### 5.2 Ablation Studies (if time permits)
- [ ] Ablation A: Query enrollment clip length (0.5 s / 1 s / 2 s) vs. SI-SDRi
- [ ] Ablation B: Averaging multiple enrollment clips into one query embedding vs. single clip
- [ ] (Optional) Secondary species: evaluate on macaque / bat mixtures without retraining

### 5.3 Final Report
- [ ] Write report sections:
  - [ ] Abstract
  - [ ] Introduction & motivation
  - [ ] Related work (BioCPPNet, Banquet 2024)
  - [ ] Method: data pipeline, PaSST query encoding, FiLM conditioning, training setup
  - [ ] Experiments & results (evaluation table)
  - [ ] Discussion: where it works, where it fails, domain transfer challenges
  - [ ] Conclusion & future work (mention hyperellipsoidal extension as natural next step)
- [ ] Include evaluation table and representative spectrograms as figures

### 5.4 Stretch Goal — Hyperellipsoidal Query Extension
> Only pursue if the core system is stable, performing well, and time remains.
- [ ] Review Watcharasupat & Lerch (2025) design spec carefully
- [ ] Implement hyperellipsoidal query representation (flatten `(c, K)` to `R^{D(D+3)/2}`)
- [ ] Widen FiLM input layer to accept the higher-dimensional query
- [ ] Precompute enclosing/excluding ellipsoid bounds per training clip
- [ ] Train and evaluate; compare against point-query baseline

### 5.5 Code Release
- [ ] Clean up and document all `src/` modules (docstrings, type hints)
- [ ] Write `README.md` with: setup instructions, dataset download, training commands, evaluation commands
- [ ] Tag release `v1.0.0` on the forked repository

---

## Open Questions & Decisions Log

> Add new entries as questions arise. Mark resolved ones with ✅.

| # | Question | Decision | Resolved By | Date |
|---|----------|----------|-------------|------|
| 1 | Is the Sarasota dolphin dataset publicly downloadable or does it require a data agreement? | TBD | — | — |
| 2 | Should we use the original BioCPPNet codebase or reimplement from scratch? | TBD — depends on code availability | — | — |
| 3 | How long should enrollment clips be (PaSST expects ~10 s input; real recordings may be shorter)? | TBD | — | — |
| 4 | Number of mixtures per split (N_train / N_val / N_test)? | Suggested 10k/1k/1k — confirm with team | — | — |
| 5 | Does PaSST generalize well to dolphin whistles out of the box, or will we need fine-tuning? | P0 decision: keep frozen PaSST baseline first; revisit after initial results | Team | 2026-03-16 |
| 6 | Should we modify Banquet internals now or use a minimal local adapter? | Use minimal local adapter for P0; keep `third_party/query-bandit/` unchanged | Team | 2026-03-16 |

---

## References

- Bermant, P. C. (2021). BioCPPNet: automatic bioacoustic source separation with deep neural networks. *Scientific Reports*, 11, 23502.
- Luo, Y., & Mesgarani, N. (2019). Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation. *IEEE/ACM TASLP*, 27(8), 1256–1266.
- Watcharasupat, K. N., & Lerch, A. (2024). A stem-agnostic single-decoder system for music source separation beyond four stems. *ISMIR 2024*. arXiv:2406.18747.
- Watcharasupat, K. N., & Lerch, A. (2025). Separate This, and All of These Things Around It: Music Source Separation via Hyperellipsoidal Queries. arXiv:2501.16171. *(stretch goal reference)*
- Banquet codebase: https://github.com/kwatcharasupat/query-bandit

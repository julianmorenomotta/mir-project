# Implementation Plan — Query-Conditioned Bioacoustic Source Separation
> **How to use this document**
> - This file is the single source of truth for backlog, task assignment, and progress tracking.
> - Mark tasks with `[x]` when done. Add your initials and date in parentheses, e.g. `[x] (julian, 2026-03-10)`.
> - Keep sub-tasks granular enough that a teammate can pick one up independently.
> - Never delete a completed item — it is the audit trail of what has already been done.
> - If a task is blocked, prepend `[BLOCKED]` to its line and add a note below it explaining why.

---

## Table of Contents
1. [Current Main Focus (March 2026)](#0-current-main-focus-march-2026)
2. [Project Overview](#1-project-overview)
3. [Milestone Map](#2-milestone-map)
4. [Phase 0 — Environment & Repository Setup](#phase-0--environment--repository-setup)
5. [Phase 1 — Data Pipeline](#phase-1--data-pipeline)
6. [Phase 2 — Baselines](#phase-2--baselines)
7. [Phase 3 — Point-Query Banquet Adaptation](#phase-3--point-query-banquet-adaptation)
8. [Phase 4 — Training & Evaluation](#phase-4--training--evaluation)
9. [Phase 5 — Analysis & Deliverables](#phase-5--analysis--deliverables)
10. [Open Questions & Decisions Log](#open-questions--decisions-log)
11. [References](#references)

---

## 0. Current Main Focus (March 21 2026)

**Priority override:** The active objective right now is to reproduce **Banquet Q:ALL evaluation for fold 5** using pretrained checkpoints in `third_party/query-bandit`. Until this is complete, this workstream takes priority over the longer bioacoustics adaptation phases below.

**Target outcome (first pass):**
- Directional reproduction of fold-5 Q:ALL results using `models/ev-pre-aug.ckpt`
- Paper-style SNR reporting generated from saved inference audio
- Clear documentation of any reproducibility deltas

**Included in scope now:**
- Evaluation-only reproduction path
- Minimal required MoisesDB preprocessing to match query-bandit expected layout
- Fold-5 only

**Out of scope now:**
- Any model training/fine-tuning
- Multi-fold averaging
- Exact-number replication requirement (directional match is acceptable for this pass)

### 0.1 Immediate Execution Checklist

#### Phase A: Prepare data root expected by query-bandit
- [x] Create local working root `${DATA_ROOT}/moisesdb` with query-bandit-compatible structure
- [ ] Generate required artifacts from raw/canonical MoisesDB using `third_party/query-bandit/core/data/moisesdb/npyify.py`:
  - [x] `metadata.csv` via `consolidate_metadata()`
  - [x] `npy2/` via `convert_to_npy()`
  - [x] `stems.csv` via `consolidate_stems()`
  - [x] `durations.csv` via `get_durations()`
  - [x] `npyq/` onset queries via `get_query_from_onset()` with `query-10s`
- [x] Copy published split/query pairing files into the working root:
  - [x] `third_party/query-bandit/reproducibility/splits.csv`
  - [x] `third_party/query-bandit/reproducibility/test_indices.csv`

#### Phase B: Make pretrained eval config runnable locally
- [x] Start from `third_party/query-bandit/expt/setup-c/bandit-everything-query-pre-d-aug.yml`
- [x] Add an evaluation-safe override for `third_party/query-bandit/config/models/bandit-query-pre.yml` so model init does not require the original stage-1 encoder path
- [x] Standardize env vars for reproducibility:
  - [x] `CONFIG_ROOT=third_party/query-bandit/config`
  - [x] `DATA_ROOT=<parent-of-local-moisesdb-working-root>`
  - [x] `LOG_ROOT=<writable-local-output-root>`
- [ ] Run smoke load test: instantiate model + load `models/ev-pre-aug.ckpt` successfully

#### Phase C: Run fold-5 inference and compute paper-style metrics
- [ ] Run `query_inference` (not only `query_test`) to produce audio outputs for notebook-based reporting
- [ ] Confirm outputs are saved under Lightning logger tree (`audio/<song>/<stem>.wav`)
- [ ] Run `third_party/query-bandit/notebooks/evaluate_ev_query.ipynb` on local outputs
- [ ] (Optional) Run `third_party/query-bandit/notebooks/evaluate_ev_query_merged.ipynb` for coarse merged reporting
- [ ] Produce fold-5 summary table and compare directionally to paper claims

#### Phase D: Reproducibility validation and notes
- [ ] Verify final working root contains: `metadata.csv`, `stems.csv`, `durations.csv`, `splits.csv`, `test_indices.csv`, `npy2/`, `npyq/`
- [ ] Record exact command/config/checkpoint identifiers used
- [ ] Document gaps vs paper values and suspected causes (query extraction variance, path/layout mismatches, checkpoint/config pairing)

### 0.2 Success Criteria for this focus
- [ ] End-to-end run succeeds for fold-5 Q:ALL using pretrained checkpoint
- [ ] Notebook-compatible outputs and summary metrics are generated
- [ ] Results are directionally aligned with paper trends, especially weaker long-tail stems

## 1. Project Overview

**Goal:** Adapt the Banquet query-conditioned source separation framework (Watcharasupat & Lerch, 2024) to bioacoustic recordings, using bottlenose dolphin signature whistles as the primary testbed. The system accepts a short enrollment clip of the target individual, encodes it via a frozen PaSST model, and uses the resulting embedding to condition a bandsplit decoder via FiLM — steering the network to extract the matching individual and suppress others.

**Scope (POC):**
- Species: **bottlenose dolphins** (primary); macaques and bats are secondary, pursued only if time permits
- Task: two-speaker mixture separation
- Primary metric: SI-SDR / SI-SDRi; secondary: individual identity classification accuracy on separated outputs

**Core deliverable:** A working query-conditioned separation model for dolphin vocalizations, benchmarked against BioCPPNet and a Conv-TasNet query baseline.

**Stretch goal:** Hyperellipsoidal region-based query extension (Watcharasupat & Lerch, 2025) — only if the core system is stable and time allows.

**Banquet source:** `github.com/kwatcharasupat/query-bandit` — cloned into `third_party/query-bandit/` (read-only reference; we import from it, not modify it)

---

## 2. Milestone Map

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| M0 | Environment set up, Banquet cloned into third_party/, smoke test passes | Week 1 | ⬜ |
| M1 | Raw dolphin data acquired and verified | Week 1 | ⬜ |
| M2 | Mixture generation pipeline validated | Week 2 | ⬜ |
| M3 | BioCPPNet baseline trained and evaluated | Week 3 | ⬜ |
| M4 | Point-query Banquet model trained and evaluated | Week 5 | ⬜ |
| M5 | Full evaluation table + qualitative analysis | Week 6 | ⬜ |
| M6 | Final report and code freeze | Week 7 | ⬜ |

> Status key: ⬜ not started · 🔄 in progress · ✅ done · 🚫 blocked

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

## Phase 3 — Banquet Adaptation to Bioacoustics

> **Goal:** Fork and adapt the Banquet codebase to work on bioacoustic data. No architectural changes — only the data pipeline and query extraction routine are replaced.

### 3.1 Codebase Audit
- [ ] Read through `third_party/query-bandit/` — identify the key files:
  - [ ] Bandsplit encoder/decoder model
  - [ ] FiLM conditioning module
  - [ ] Training loop and loss
  - [ ] Data loading and query extraction logic
- [ ] Document which components we reuse as-is vs. which we need to adapt in `src/`
- [ ] Add `third_party/query-bandit` to `sys.path` in a shared `src/utils/paths.py` so we can import from it cleanly

### 3.2 Model Wrapper
- [ ] Create `src/models/banquet_bio.py` — thin wrapper that imports the Banquet model from `third_party/` and exposes a clean interface for our training loop
- [ ] Confirm the wrapper is importable and the Banquet model instantiates correctly

### 3.3 Data Loader Replacement
- [ ] Write `src/data/dataset.py` as a drop-in replacement for Banquet's music dataloader
- [ ] Confirm the dataloader returns: `(mixture, clean_source, query_clip)` triplets with the shapes the Banquet model expects
- [ ] Smoke test: one forward pass through the full model with a real dolphin batch

### 3.3 PaSST Query Encoder
- [ ] Verify PaSST model loads correctly (`hear-eval-kit` or direct weights)
- [ ] Write `src/embeddings/passt_encoder.py`:
  - [ ] Load frozen PaSST (no gradient updates)
  - [ ] Encode a query enrollment clip → 768-dim embedding vector
  - [ ] Confirm output shape and dtype
- [ ] Evaluate whether PaSST embeddings separate dolphin individuals: plot t-SNE/UMAP in `notebooks/02_embedding_exploration.ipynb`
- [ ] Write unit test: same clip → same embedding (determinism check)

### 3.4 FiLM Conditioning
- [ ] Confirm existing Banquet FiLM layer accepts the 768-dim PaSST embedding
- [ ] Implement `src/models/film.py` — standalone FiLM module:
  - [ ] `gamma, beta = FC(query_vec)` where FC is a small MLP
  - [ ] `output = gamma * features + beta`
- [ ] Unit test: FiLM layer output shape matches bandsplit bottleneck feature shape

---

## Phase 4 — Training & Evaluation

### 4.1 Loss Function
- [ ] Implement `src/losses/l1snr.py` — Multichannel L1SNR loss (carry over from Banquet)
- [ ] Unit test: loss decreases for a perfect prediction

### 4.2 Training
- [ ] Write/adapt `src/train.py`:
  - [ ] Config-driven via YAML (`configs/banquet_bio_dolphins.yaml`)
  - [ ] Log loss curves to `wandb` or tensorboard
  - [ ] Save checkpoints to `checkpoints/`
  - [ ] Implement early stopping on validation SI-SDRi
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
| 5 | Does PaSST generalize well to dolphin whistles out of the box, or will we need fine-tuning? | TBD — assess via t-SNE of embeddings before training (Phase 3.3) | — | — |

---

## References

- Bermant, P. C. (2021). BioCPPNet: automatic bioacoustic source separation with deep neural networks. *Scientific Reports*, 11, 23502.
- Luo, Y., & Mesgarani, N. (2019). Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation. *IEEE/ACM TASLP*, 27(8), 1256–1266.
- Watcharasupat, K. N., & Lerch, A. (2024). A stem-agnostic single-decoder system for music source separation beyond four stems. *ISMIR 2024*. arXiv:2406.18747.
- Watcharasupat, K. N., & Lerch, A. (2025). Separate This, and All of These Things Around It: Music Source Separation via Hyperellipsoidal Queries. arXiv:2501.16171. *(stretch goal reference)*
- Banquet codebase: https://github.com/kwatcharasupat/query-bandit

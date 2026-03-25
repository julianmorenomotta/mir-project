# Query-Bandit Adaptation to Bioacoustics

This repository adapts Query-Bandit to non-music bioacoustic data.

The current supported preprocessing flow is the macaque pipeline under `src/tools/`:
- `macaque_audit.py`
- `build_macaque_sessions.py`
- `build_macaque_queries.py`

Important naming convention: use `train` and `val` splits (not `valid`).

## 1) Environment Setup

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`pip install -e .` is required so `third_party/query-bandit/train.py` can import `data.macaque.*` from `src/`.

Set environment variables (example):

```bash
export CONFIG_ROOT=./third_party/query-bandit/config
export DATA_ROOT=./data/datasets
export LOG_ROOT=./data/logs
```

If you keep these in `.env`, load them before running commands:

```bash
source .env
```

## 2) Raw Data Prerequisites

Place source audio under:

```text
data/datasets/macaque_raw/
  train/
  val/
  test/   # required by MacaqueDataModule; can be a held-out subset or copied from val for smoke checks
```

Filenames should begin with a 2-letter individual ID so the audit script can parse speakers.

## 3) Audit Raw Data and Build Query Pool

Run:

```bash
python src/tools/macaque_audit.py \
  --raw-root data/datasets/macaque_raw \
  --splits train val test \
  --seed 1337
```

This generates inventory and query-pool artifacts under `data/datasets/macaque_raw/`.

## 4) Build Session Mixtures

Run:

```bash
python src/tools/build_macaque_sessions.py \
  --raw-root data/datasets/macaque_raw \
  --query-pool-csv data/datasets/macaque_raw/query_pool.csv \
  --output-root data/datasets/macaque_dataset \
  --splits train val test \
  --sessions-per-split "train=500" "val=100" "test=100" \
  --target-sample-rate 44100
```

Outputs:

```text
data/datasets/macaque_dataset/
  train/mixture_00000/
    mixture.wav
    <speaker>.wav
    metadata.json
  val/mixture_00000/
    ...
```

## 5) Build Query Clips

Run:

```bash
python src/tools/build_macaque_queries.py \
  --raw-root data/datasets/macaque_raw \
  --query-pool-csv data/datasets/macaque_raw/query_pool.csv \
  --output-root data/datasets/macaque_dataset/queries \
  --splits train val test \
  --clips-per-individual 4 \
  --target-seconds 10 \
  --target-sample-rate 44100
```

Outputs:

```text
data/datasets/macaque_dataset/queries/
  train/<speaker>/query_clip_000.wav
  val/<speaker>/query_clip_000.wav
  test/<speaker>/query_clip_000.wav
```

## 6) Validate DataModule Wiring (No Model Needed)

You can validate dataloaders without checkpoints using:

```bash
python third_party/query-bandit/train.py train \
  --config_path=third_party/query-bandit/config/bandit-macaque.yml \
  --test_datamodule=true
```

This command iterates train/val/test dataloaders only.

## 7) Optional Query-Bandit Evaluation Command

When you have a valid checkpoint:

```bash
python third_party/query-bandit/train.py query_test \
  --config_path third_party/query-bandit/expt/setup-bio/bandit-everything-query-pre-d-aug.yml \
  --ckpt_path models/ev-pre-aug.ckpt
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'data'`
  - Ensure editable install was done from repo root: `python -m pip install -e .`.

- `FileNotFoundError ... /valid/...`
  - Old artifacts still reference `valid`. Rebuild with `--splits train val test`.
  - Ensure directory names are `train/`, `val/`, and `test/` everywhere.

- `Split directory not found: .../test`
  - `MacaqueDataModule` currently builds train/val/test datasets.
  - Provide a `test` split in raw data and rebuild, or create a temporary smoke-test copy:
    `cp -r data/datasets/macaque_dataset/val data/datasets/macaque_dataset/test`
    and
    `cp -r data/datasets/macaque_dataset/queries/val data/datasets/macaque_dataset/queries/test`

- Fire flag mismatch
  - Use `--config_path` (singular), not `--config_paths`.


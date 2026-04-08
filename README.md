# Query-Bandit Zero-Shot Evaluation on Unseen Music Data
 
This repository evaluates the pretrained [Banquet (Query-Bandit)](https://github.com/kwatcharasupat/query-bandit)
model on BabySlakh, a synthetic MIDI-rendered music dataset unseen during training.
The goal is to measure how well the model generalizes to out-of-distribution music
without any retraining or fine-tuning.
 
Audio examples are available in `examples/`.
 
## Project structure
 
```
mir-project/
  notebooks/
    babyslakh_vs_moises_comparison.ipynb   # main analysis notebook
  src/
    tools/
      build_babyslakh_metadata.py          # generates splits.csv, stems.csv, etc.
      babyslakh_npyify.py                  # converts BabySlakh WAVs to npy2/ and npyq/
  third_party/
    query-bandit/                          # original Banquet codebase (read-only)
  data/
    datasets/
      babyslakh_16k/                       # raw BabySlakh audio (not tracked by git)
    logs/                                  # inference outputs (not tracked by git)
  models/
    ev-pre-aug.ckpt                        # pretrained checkpoint (not tracked by git)
```
 
## 1. Environment setup
 
From the repository root:
 
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```
 
The editable install is required so `third_party/query-bandit/train.py`
can import modules from `src/`.
 
Set environment variables:
 
```bash
export CONFIG_ROOT=./third_party/query-bandit/config
export DATA_ROOT=./data/datasets
export LOG_ROOT=./data/logs
```
 
## 2. Data prerequisites
 
Download BabySlakh (20-track subset of Slakh2100, 16 kHz):
 
```bash
# BabySlakh is available at https://zenodo.org/records/4603879
# Place the extracted folder at:
data/datasets/babyslakh_16k/
```
 
The pretrained Banquet checkpoint (`ev-pre-aug.ckpt`) should be placed at `models/`.
 
## 3. Build BabySlakh artifacts
 
BabySlakh must be converted into the Moises-compatible format expected by the
Banquet evaluation pipeline. This involves resampling from 16 kHz to 44.1 kHz,
saving stems as `.npy` arrays, extracting query clips, and generating index CSVs.
 
**Step 1: Generate metadata and index files:**
 
```bash
python src/tools/build_babyslakh_metadata.py \
  --data-root data/datasets/babyslakh_16k \
  --output-root data/datasets/babyslakh_16k
```
 
This produces:
- `splits.csv`: all tracks assigned to test fold 5
- `stems.csv`: binary instrument-presence matrix
- `durations.csv`: track durations
- `test_indices.csv`: (song_id, query_id, stem) evaluation pairs
 
**Step 2:  Convert audio to npy arrays:**
 
```bash
python src/tools/babyslakh_npyify.py \
  --data-root data/datasets/babyslakh_16k \
  --output-root data/datasets/babyslakh_16k \
  --target-sr 44100
```
 
This produces:
- `npy2/<song_id>/<stem>.npy`: full-track stems and mixture
- `npyq/<song_id>/<stem>.query-10s.npy`: 10-second onset-based query clips
 
All arrays are saved as float32 at 44.1 kHz. Note that resampling from 16 kHz
means frequency content above 8 kHz is absent in the upsampled signal.
 
## 4. Run zero-shot evaluation
 
**Objective metrics:**
 
```bash
python third_party/query-bandit/train.py query_test \
  --config_path=third_party/query-bandit/config/bandit-babyslakh-moises.yml \
  --ckpt_path=models/ev-pre-aug.ckpt
```
 
**Audio export for notebook analysis:**
 
```bash
python third_party/query-bandit/train.py query_inference \
  --config_path=third_party/query-bandit/config/bandit-babyslakh-moises.yml \
  --ckpt_path=models/ev-pre-aug.ckpt
```
 
Outputs are written under `data/logs/`.
 
## 5. Analyze results
 
Open and run the analysis notebook:
 
```bash
jupyter notebook notebooks/babyslakh_vs_moises_comparison.ipynb
```
 
This notebook loads the BabySlakh test metrics and the MoisesDB researcher
baseline, computes per-stem SNR deltas, and produces the comparison figures
used in the paper.
 
## 6. MoisesDB baseline reproduction (optional)
 
To verify the pipeline against the original Banquet results on MoisesDB:
 
```bash
python third_party/query-bandit/train.py query_test \
  --config_path=third_party/query-bandit/config/bandit-moisesdb.yml \
  --ckpt_path=models/ev-pre-aug.ckpt
```
 
Reference results are available at
`third_party/query-bandit/reproducibility/results/query/bandit_ev.csv`.
 
## Troubleshooting
 
- `ModuleNotFoundError: No module named 'data'` — run `pip install -e .` from
  the repo root before any other command.
- `FileNotFoundError` on npy files — run Steps 3a and 3b before evaluation.
- `Split directory not found` — confirm `splits.csv` has `split=5` for all
  BabySlakh tracks and that `test_indices.csv` exists.
- `allowed_stems` mismatch error — check that the stem names in
  `third_party/query-bandit/config/data/babyslakh-test.yml` match the column
  names in `stems.csv` and the filenames in `npy2/`.
 
## Reference
 
Watcharasupat, K. N., & Lerch, A. (2024). A stem-agnostic single-decoder system
for music source separation beyond four stems. *ISMIR 2024*. 
[github.com/kwatcharasupat/query-bandit](https://github.com/kwatcharasupat/query-bandit)
 
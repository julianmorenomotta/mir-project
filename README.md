# Query-bandit model adaptation to bioacustics dataset

## Testing
```bash
source .env
python third_party/query-bandit/train.py query_test --config_path third_party/query-bandit/expt/setup-bio/bandit-everything-query-pre-d-aug.yml --ckpt_path models/ev-pre-aug.ckpt
```
* Runs in a NVIDIA 3090 GPU with a batch size of 1

**Testing note — compute profile used during local debugging**
- We are currently using a temporary low-memory profile to avoid CUDA OOM during `query_test` on limited hardware.
- Active low-memory settings:
  - `third_party/query-bandit/config/data/moisesdb-test.yml`: `num_workers: 1`, `inference_kwargs.batch_size: 1`
  - `third_party/query-bandit/core/data/base.py`: dataloaders with `pin_memory=False`, `prefetch_factor=2` only when workers are enabled, and `persistent_workers=(num_workers > 0)`
- Reproducibility intent:
  - These changes are expected to affect runtime and memory behavior, not model architecture or checkpoint contents.
  - Keep this profile for stability while validating the end-to-end pipeline.
- Scale-up plan when more compute is available:
  - Increase `inference_kwargs.batch_size` gradually (e.g., `1 -> 2 -> 4 -> 8 -> 12`) while monitoring GPU memory.
  - Increase `num_workers` in `moisesdb-test.yml` to improve throughput.
  - Optionally re-enable `pin_memory` in `core/data/base.py` if memory is no longer a bottleneck.

## Building Bioacoustics dataset

```bash
python src/tools/build_macaque_sessions.py --sessions-per-split "train=500" "valid=100"
```


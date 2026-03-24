import pandas as pd
import numpy as np

# Paths
BASELINE_CSV = "third_party/query-bandit/reproducibility/results/query/bandit_ev.csv"
LIGHTNING_CSV = "data/logs/e2e/test/7P21AG/lightning_logs/version_0/metrics.csv"

# Load data
baseline = pd.read_csv(BASELINE_CSV)
lightning_df = pd.read_csv(LIGHTNING_CSV)

# Parse Lightning columns: test/<stem>/snr/<MetricType>
parsed_data = []
for col in lightning_df.columns:
    if col == "step":
        continue
    parts = col.split("/")
    if len(parts) >= 4 and parts[0] == "test" and parts[2] == "snr":
        stem = parts[1]
        metric_type = parts[3]
        values = lightning_df[col].dropna()
        for value in values:
            parsed_data.append({"stem": stem, "metric": metric_type, "value": value})

parsed_df = pd.DataFrame(parsed_data)

if parsed_df.empty:
    raise RuntimeError("No parsed test metrics found. Check LIGHTNING_CSV path/format.")

# Build your SNR rows
your_snr_raw = (
    parsed_df[parsed_df["metric"] == "SafeSignalNoiseRatio"]
    .rename(columns={"value": "snr"})[["stem", "snr"]]
    .dropna()
)

if your_snr_raw.empty:
    raise RuntimeError("No SafeSignalNoiseRatio values found in parsed metrics.")

# Keep only stems you actually evaluated
stems_eval = sorted(your_snr_raw["stem"].unique())
baseline_sub = baseline[baseline["stem"].isin(stems_eval)].copy()

if baseline_sub.empty:
    raise RuntimeError("No overlapping stems found between baseline and your results.")

# Match per-stem counts to your run
your_counts = your_snr_raw.groupby("stem").size().to_dict()


def draw_matched_baseline(df: pd.DataFrame, counts: dict, seed: int) -> pd.DataFrame:
    rs = np.random.default_rng(seed)
    chunks = []
    for stem, k in counts.items():
        s = df[df["stem"] == stem]
        if len(s) == 0:
            continue
        replace = len(s) < k
        idx = rs.choice(s.index.to_numpy(), size=k, replace=replace)
        chunks.append(s.loc[idx, ["stem", "snr"]])
    if not chunks:
        raise RuntimeError("Sampling failed: no chunks created.")
    return pd.concat(chunks, ignore_index=True)


# Your observed micro-mean
your_micro_mean = your_snr_raw["snr"].mean()

# Bootstrap expected baseline distribution at same sample budget/composition
B = 2000
boot_means = np.empty(B, dtype=float)
for b in range(B):
    sample_b = draw_matched_baseline(baseline_sub, your_counts, seed=1000 + b)
    boot_means[b] = sample_b["snr"].mean()

lo, hi = np.percentile(boot_means, [2.5, 97.5])
baseline_expected = boot_means.mean()

print("\n=== Matched-Subset Comparison (fair for partial run) ===")
print(f"Evaluated stems: {len(stems_eval)}")
print(f"Your sample count: {len(your_snr_raw)}")
print(f"Your SNR micro-mean: {your_micro_mean:.4f} dB")
print(f"Baseline expected (matched): {baseline_expected:.4f} dB")
print(f"95% expected range: [{lo:.4f}, {hi:.4f}] dB")
print(f"Delta vs expected: {your_micro_mean - baseline_expected:.4f} dB")
print(f"Inside expected range? {lo <= your_micro_mean <= hi}")

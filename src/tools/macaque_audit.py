from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import soundfile as sf
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


DEFAULT_RAW_ROOT = Path("data/macaque_raw")
DEFAULT_SPLITS = ("train", "valid")
DEFAULT_QUERY_FRACTION = 0.20
DEFAULT_MIN_MIXTURE_CALLS = 1
DEFAULT_EXPECTED_SAMPLE_RATE = 0  # 0 disables enforcement
DEFAULT_TARGET_SAMPLE_RATE = 44_100
DEFAULT_EXPECTED_CHANNELS = 1


@dataclass
class FileRecord:
    split: str
    individual_id: str
    path: Path
    relative_path: str
    duration_sec: float
    sample_rate: int
    channels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Path to data/macaque_raw containing train/ and valid/ folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to inspect (default: train valid).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used when selecting the held-out query pool.",
    )
    parser.add_argument(
        "--query-fraction",
        type=float,
        default=DEFAULT_QUERY_FRACTION,
        help="Fraction of calls per individual reserved for queries (default 0.20).",
    )
    parser.add_argument(
        "--min-mixture-calls",
        type=int,
        default=DEFAULT_MIN_MIXTURE_CALLS,
        help="Ensure at least this many calls remain outside the query pool.",
    )
    parser.add_argument(
        "--expected-sample-rate",
        type=int,
        default=DEFAULT_EXPECTED_SAMPLE_RATE,
        help="Strict WAV sample rate to enforce (Hz). Set to 0 to skip this check (default).",
    )
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=DEFAULT_EXPECTED_CHANNELS,
        help="Expected channel count (default 1 for mono).",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=DEFAULT_TARGET_SAMPLE_RATE,
        help="Desired post-processing sample rate in Hz (used to flag files for resampling).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".wav"],
        help="File extensions (lowercase) to audit. Defaults to .wav.",
    )
    return parser.parse_args()


def extract_individual_id(path: Path) -> str | None:
    match = re.match(r"([A-Za-z]{2})", path.stem)
    if not match:
        return None
    return match.group(1).upper()


def read_audio_metadata(path: Path) -> Tuple[int, int, float]:
    with sf.SoundFile(path) as audio:
        frames = audio.frames
        sample_rate = audio.samplerate
        channels = audio.channels
    duration = frames / sample_rate if sample_rate else 0.0
    return sample_rate, channels, duration


def collect_records(
    raw_root: Path,
    splits: Sequence[str],
    extensions: Sequence[str],
) -> Tuple[Dict[str, Dict[str, List[FileRecord]]], List[str]]:
    records: Dict[str, Dict[str, List[FileRecord]]] = {}
    issues: List[str] = []
    normalized_exts = tuple(ext.lower() for ext in extensions)

    for split in splits:
        split_dir = raw_root / split
        if not split_dir.exists():
            issues.append(f"Missing split directory: {split_dir}")
            continue
        if not split_dir.is_dir():
            issues.append(f"Split path is not a directory: {split_dir}")
            continue

        split_records: Dict[str, List[FileRecord]] = {}
        for wav_path in sorted(split_dir.rglob("*")):
            if not wav_path.is_file():
                continue
            if wav_path.suffix.lower() not in normalized_exts:
                continue

            individual_id = extract_individual_id(wav_path)
            if individual_id is None:
                issues.append(f"Unable to parse individual ID from {wav_path.name}")
                continue

            try:
                sample_rate, channels, duration = read_audio_metadata(wav_path)
            except RuntimeError as exc:
                issues.append(f"Failed reading {wav_path}: {exc}")
                continue

            try:
                relative_path = wav_path.relative_to(raw_root)
            except ValueError:
                relative_path = wav_path.name

            file_record = FileRecord(
                split=split,
                individual_id=individual_id,
                path=wav_path,
                relative_path=str(relative_path).replace("\\", "/"),
                duration_sec=duration,
                sample_rate=sample_rate,
                channels=channels,
            )
            split_records.setdefault(individual_id, []).append(file_record)

        records[split] = split_records

    return records, issues


def validate_audio(
    records: Mapping[str, Dict[str, List[FileRecord]]],
    expected_sr: int,
    expected_ch: int,
) -> List[str]:
    issues: List[str] = []
    enforce_sr = expected_sr > 0
    for split_records in records.values():
        for file_list in split_records.values():
            for record in file_list:
                if enforce_sr and record.sample_rate != expected_sr:
                    issues.append(
                        f"Sample rate mismatch ({record.sample_rate} Hz) in {record.relative_path}"
                    )
                if record.channels != expected_ch:
                    issues.append(
                        f"Channel mismatch ({record.channels}) in {record.relative_path}"
                    )
    return issues


def choose_query_pool(
    records: Mapping[str, Dict[str, List[FileRecord]]],
    seed: int,
    query_fraction: float,
    min_mixture_calls: int,
) -> Dict[str, Dict[str, List[FileRecord]]]:
    rng = random.Random(seed)
    assignments: Dict[str, Dict[str, List[FileRecord]]] = {}

    for split, split_records in records.items():
        assignments[split] = {}
        for individual_id, file_list in split_records.items():
            file_count = len(file_list)
            if file_count == 0:
                assignments[split][individual_id] = []
                continue

            desired = math.ceil(file_count * query_fraction)
            if desired == 0 and file_count > 0:
                desired = 1

            max_assignable = max(0, file_count - min_mixture_calls)
            query_count = min(desired, max_assignable)
            if query_count == 0:
                assignments[split][individual_id] = []
                continue

            sampled = rng.sample(file_list, query_count)
            sampled.sort(key=lambda rec: rec.relative_path)
            assignments[split][individual_id] = sampled

    return assignments


def write_inventory_csv(
    inventory_path: Path,
    records: Mapping[str, Dict[str, List[FileRecord]]],
    query_assignments: Mapping[str, Dict[str, List[FileRecord]]],
    target_sample_rate: int,
) -> None:
    fieldnames = [
        "split",
        "individual_id",
        "call_count",
        "query_pool_count",
        "mixture_pool_count",
        "needs_resample_count",
        "total_duration_sec",
        "mean_duration_sec",
        "min_duration_sec",
        "max_duration_sec",
    ]
    rows: List[Dict[str, float | int | str]] = []

    for split, split_records in sorted(records.items()):
        for individual_id, file_list in sorted(split_records.items()):
            durations = [record.duration_sec for record in file_list]
            call_count = len(file_list)
            total_duration = sum(durations)
            mean_duration = total_duration / call_count if call_count else 0.0
            min_duration = min(durations) if durations else 0.0
            max_duration = max(durations) if durations else 0.0
            query_count = len(query_assignments.get(split, {}).get(individual_id, []))
            mixture_pool = call_count - query_count
            needs_resample = 0
            if target_sample_rate > 0:
                needs_resample = sum(
                    1
                    for record in file_list
                    if record.sample_rate != target_sample_rate
                )

            rows.append(
                {
                    "split": split,
                    "individual_id": individual_id,
                    "call_count": call_count,
                    "query_pool_count": query_count,
                    "mixture_pool_count": mixture_pool,
                    "needs_resample_count": needs_resample,
                    "total_duration_sec": round(total_duration, 6),
                    "mean_duration_sec": round(mean_duration, 6),
                    "min_duration_sec": round(min_duration, 6),
                    "max_duration_sec": round(max_duration, 6),
                }
            )

    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_query_csv(
    query_path: Path,
    query_assignments: Mapping[str, Dict[str, List[FileRecord]]],
) -> None:
    fieldnames = ["split", "individual_id", "relative_path", "duration_sec"]
    rows: List[Dict[str, str | float]] = []

    for split, individuals in sorted(query_assignments.items()):
        for individual_id, file_list in sorted(individuals.items()):
            for record in file_list:
                rows.append(
                    {
                        "split": split,
                        "individual_id": individual_id,
                        "relative_path": record.relative_path,
                        "duration_sec": round(record.duration_sec, 6),
                    }
                )

    with query_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_resample_plan(
    resample_path: Path,
    records: Mapping[str, Dict[str, List[FileRecord]]],
    target_sample_rate: int,
) -> int:
    fieldnames = [
        "split",
        "individual_id",
        "relative_path",
        "sample_rate",
        "target_sample_rate",
        "needs_resample",
    ]
    rows: List[Dict[str, str | int | bool]] = []
    needs_resample_total = 0
    target = target_sample_rate if target_sample_rate > 0 else None

    for split, split_records in sorted(records.items()):
        for individual_id, file_list in sorted(split_records.items()):
            for record in file_list:
                needs_resample = bool(target and record.sample_rate != target)
                if needs_resample:
                    needs_resample_total += 1
                rows.append(
                    {
                        "split": split,
                        "individual_id": individual_id,
                        "relative_path": record.relative_path,
                        "sample_rate": record.sample_rate,
                        "target_sample_rate": target or "n/a",
                        "needs_resample": needs_resample,
                    }
                )

    with resample_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return needs_resample_total


def write_metadata(
    metadata_path: Path,
    seed: int,
    query_fraction: float,
    min_mixture_calls: int,
    splits: Sequence[str],
    inventory_path: Path,
    query_path: Path,
    resample_plan_path: Path,
    target_sample_rate: int,
) -> None:
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "query_fraction": query_fraction,
        "min_mixture_calls": min_mixture_calls,
        "splits": list(splits),
        "inventory_csv": inventory_path.name,
        "query_pool_csv": query_path.name,
        "resample_plan_csv": resample_plan_path.name,
        "target_sample_rate": target_sample_rate,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def summarize_counts(
    records: Mapping[str, Dict[str, List[FileRecord]]],
) -> Tuple[int, int]:
    individuals = 0
    files = 0
    for split_records in records.values():
        individuals += len(split_records)
        for file_list in split_records.values():
            files += len(file_list)
    return individuals, files


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()

    records, structural_issues = collect_records(raw_root, args.splits, args.extensions)
    validation_issues = validate_audio(
        records, args.expected_sample_rate, args.expected_channels
    )
    issues = structural_issues + validation_issues

    individuals, files = summarize_counts(records)
    if files == 0:
        print(f"No audio files found under {raw_root}. Nothing to do.", file=sys.stderr)
        if issues:
            for msg in issues:
                print(f" - {msg}", file=sys.stderr)
        raise SystemExit(1)

    if issues:
        print("Audit detected format issues:", file=sys.stderr)
        for msg in issues:
            print(f" - {msg}", file=sys.stderr)
        raise SystemExit(1)

    query_assignments = choose_query_pool(
        records,
        seed=args.seed,
        query_fraction=args.query_fraction,
        min_mixture_calls=args.min_mixture_calls,
    )

    inventory_path = raw_root / "inventory.csv"
    query_path = raw_root / "query_pool.csv"
    metadata_path = raw_root / "audit_metadata.json"
    resample_plan_path = raw_root / "resample_plan.csv"

    write_inventory_csv(
        inventory_path,
        records,
        query_assignments,
        target_sample_rate=args.target_sample_rate,
    )
    write_query_csv(query_path, query_assignments)
    needs_resample_total = write_resample_plan(
        resample_plan_path,
        records,
        target_sample_rate=args.target_sample_rate,
    )
    write_metadata(
        metadata_path,
        seed=args.seed,
        query_fraction=args.query_fraction,
        min_mixture_calls=args.min_mixture_calls,
        splits=args.splits,
        inventory_path=inventory_path,
        query_path=query_path,
        resample_plan_path=resample_plan_path,
        target_sample_rate=args.target_sample_rate,
    )

    print(f"Audited {files} files across {individuals} individual entries.")
    print(f"Inventory written to {inventory_path}")
    print(f"Query assignments written to {query_path}")
    print(f"Resample plan written to {resample_plan_path}")
    if args.target_sample_rate > 0:
        print(
            f"{needs_resample_total} files need resampling to {args.target_sample_rate} Hz."
        )
    else:
        print("Target sample rate disabled; resample plan recorded for reference only.")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()

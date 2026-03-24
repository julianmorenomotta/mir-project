"""Materialize macaque query enrollment clips from the held-out pool."""

from __future__ import annotations

import argparse
import json
import random
import sys
import torch  # noqa: F401 (used for typing and ensuring dependency presence)
import torchaudio
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.macaque.query_builder import QueryBuilder


def save_audio(path: Path, waveform, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.detach().cpu().unsqueeze(0), sample_rate)


def write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def build_queries(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    builder = QueryBuilder(
        raw_root=args.raw_root,
        query_pool_csv=args.query_pool_csv,
        splits=args.splits,
        clip_duration_sec=args.target_seconds,
        gap_range=(args.gap_min_sec, args.gap_max_sec),
        target_sample_rate=args.target_sample_rate,
        normalize_output=not args.disable_normalize,
    )

    rng = random.Random(args.seed)
    total_written = 0

    for split in args.splits:
        individuals = builder.available_individuals(split)
        if not individuals:
            print(f"[WARN] No individuals available for split '{split}'.")
            continue

        if args.individuals:
            filtered = set(ind.upper() for ind in args.individuals)
            individuals = [ind for ind in individuals if ind in filtered]
            if not individuals:
                print(f"[WARN] Filter removed all individuals for split '{split}'.")
                continue

        for individual_id in individuals:
            clips_requested = args.clips_per_individual
            individual_dir = output_root / split / individual_id
            individual_dir.mkdir(parents=True, exist_ok=True)

            for clip_offset in range(clips_requested):
                clip_index = args.start_index + clip_offset
                clip_stem = f"{args.clip_prefix}_{clip_index:03d}"
                clip_path = individual_dir / f"{clip_stem}.wav"
                metadata_path = individual_dir / f"{clip_stem}.json"

                if clip_path.exists() and not args.overwrite:
                    print(f"[SKIP] {clip_path} exists (use --overwrite to regenerate).")
                    continue

                if args.dry_run:
                    print(f"[DRY RUN] Would create {clip_path}")
                    total_written += 1
                    continue

                clip_seed = rng.randrange(0, 2**32)
                result = builder.build_clip(
                    split=split,
                    individual_id=individual_id,
                    seed=clip_seed,
                )

                save_audio(clip_path, result.waveform, result.sample_rate)

                metadata = dict(result.metadata)
                metadata.update(
                    {
                        "split": split,
                        "individual_id": individual_id,
                        "clip_name": clip_stem,
                        "clip_path": clip_path.relative_to(output_root).as_posix(),
                        "raw_root": str(Path(args.raw_root).resolve()),
                        "query_pool_csv": str(Path(args.query_pool_csv).resolve()),
                    }
                )
                write_metadata(metadata_path, metadata)

                total_written += 1
                print(f"[OK] {clip_path} (seed={clip_seed})")

    print(f"Finished. Query clips created: {total_written}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/macaque_raw"))
    parser.add_argument(
        "--query-pool-csv",
        type=Path,
        default=Path("data/macaque_raw/query_pool.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/macaque_dataset/queries"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Split names to process.",
    )
    parser.add_argument(
        "--individuals",
        nargs="+",
        default=None,
        help="Optional subset of individual IDs to build (case-insensitive).",
    )
    parser.add_argument(
        "--clips-per-individual",
        type=int,
        default=4,
        help="How many query clips to build per individual per split.",
    )
    parser.add_argument(
        "--clip-prefix",
        default="query_clip",
        help="Filename prefix for per-individual clips.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index used when numbering clips.",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=10.0,
        help="Target duration per clip in seconds (used for zero-padding/truncation).",
    )
    parser.add_argument("--gap-min-sec", type=float, default=0.2)
    parser.add_argument("--gap-max-sec", type=float, default=1.0)
    parser.add_argument("--target-sample-rate", type=int, default=44_100)
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Master RNG seed controlling per-clip seeds.",
    )
    parser.add_argument(
        "--disable-normalize",
        action="store_true",
        help="Skip peak normalization before saving.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate clips even if files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended actions without writing files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    build_queries(args)


if __name__ == "__main__":
    main()

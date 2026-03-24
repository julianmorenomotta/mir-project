#!/usr/bin/env python3
"""Materialize macaque session mixtures using :class:`SessionMixer`."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import torch
import torchaudio
from pathlib import Path
from typing import Dict, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.macaque.session_mixer import SessionMixer


def parse_sessions_per_split(
    values: Sequence[str], splits: Sequence[str]
) -> Dict[str, int]:
    """Parse ``split=count`` pairs for session generation."""

    mapping: Dict[str, int] = {split: 0 for split in splits}
    if not values:
        return mapping

    for raw in values:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --sessions-per-split entry: '{raw}'. Use format split=count."
            )
        split, count_str = raw.split("=", 1)
        split = split.strip()
        if split not in mapping:
            raise ValueError(f"Split '{split}' not in configured splits {splits}.")
        try:
            count = int(count_str)
        except ValueError as exc:  # pragma: no cover - CLI validation
            raise ValueError(
                f"Count for split '{split}' must be an integer (got '{count_str}')."
            ) from exc
        if count < 0:
            raise ValueError(f"Count for split '{split}' must be non-negative.")
        mapping[split] = count
    return mapping


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.detach().cpu().unsqueeze(0), sample_rate)


def write_metadata(
    path: Path,
    result_metadata: Dict[str, object],
    mixture_file: Path,
    stem_files: Dict[str, Path],
) -> None:
    payload = dict(result_metadata)
    payload["mixture_file"] = mixture_file.as_posix()
    payload["stem_files"] = {
        speaker: stem_path.as_posix() for speaker, stem_path in stem_files.items()
    }
    path.write_text(json.dumps(payload, indent=2))


def build_sessions(args: argparse.Namespace) -> None:
    dataset_root = Path(args.output_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    session_counts = parse_sessions_per_split(args.sessions_per_split, args.splits)
    total_requested = sum(session_counts.values())
    if total_requested == 0:
        raise SystemExit(
            "No sessions requested. Provide --sessions-per-split entries like 'train=500'."
        )

    mixer = SessionMixer(
        raw_root=args.raw_root,
        query_pool_csv=args.query_pool_csv,
        splits=args.splits,
        session_duration_range=(args.session_min_sec, args.session_max_sec),
        gap_range=(args.gap_min_sec, args.gap_max_sec),
        target_sample_rate=args.target_sample_rate,
        normalize_output=not args.disable_normalize,
    )

    rng = random.Random(args.seed)
    total_written = 0
    for split in args.splits:
        requested = session_counts.get(split, 0)
        if requested == 0:
            continue

        split_dir = dataset_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(requested):
            session_index = args.start_index + idx
            session_name = f"{args.session_prefix}_{session_index:05d}"
            session_dir = split_dir / session_name

            if session_dir.exists():
                if not args.overwrite:
                    print(
                        f"[SKIP] {session_dir} exists (use --overwrite to regenerate)."
                    )
                    continue
                shutil.rmtree(session_dir)

            if args.dry_run:
                print(f"[DRY RUN] Would create {session_dir}")
                total_written += 1
                continue

            session_dir.mkdir(parents=True, exist_ok=False)
            session_seed = rng.randrange(0, 2**32)
            result = mixer.generate_session(split=split, seed=session_seed)

            mixture_path = session_dir / args.mixture_filename
            save_audio(mixture_path, result.mixture, result.sample_rate)

            stem_paths: Dict[str, Path] = {}
            for speaker_id, waveform in result.stems.items():
                stem_path = session_dir / f"{speaker_id}.wav"
                save_audio(stem_path, waveform, result.sample_rate)
                stem_paths[speaker_id] = stem_path

            metadata = dict(result.metadata)
            metadata.update(
                {
                    "session_dir": session_dir.relative_to(dataset_root).as_posix(),
                    "mixture_filename": args.mixture_filename,
                    "speakers": list(stem_paths.keys()),
                    "raw_root": str(Path(args.raw_root).resolve()),
                    "query_pool_csv": (
                        str(Path(args.query_pool_csv).resolve())
                        if args.query_pool_csv
                        else None
                    ),
                }
            )
            metadata_path = session_dir / "metadata.json"
            write_metadata(
                metadata_path,
                metadata,
                mixture_path.relative_to(dataset_root),
                {
                    speaker: path.relative_to(dataset_root)
                    for speaker, path in stem_paths.items()
                },
            )

            total_written += 1
            print(f"[OK] {session_dir} (seed={session_seed})")

    print(f"Finished. Sessions created: {total_written}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/macaque_raw"))
    parser.add_argument(
        "--query-pool-csv", type=Path, default=Path("data/macaque_raw/query_pool.csv")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/macaque_dataset")
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        help="Split names to process.",
    )
    parser.add_argument(
        "--sessions-per-split",
        nargs="+",
        default=[],
        help="Pairs like train=500 val=100 describing how many sessions to create per split.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for session numbering (per split).",
    )
    parser.add_argument(
        "--session-prefix",
        default="mixture",
        help="Folder prefix for each session directory.",
    )
    parser.add_argument(
        "--mixture-filename",
        default="mixture.wav",
        help="Filename used for the summed mixture audio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260324,
        help="Master RNG seed controlling per-session seeds.",
    )
    parser.add_argument("--session-min-sec", type=float, default=8.0)
    parser.add_argument("--session-max-sec", type=float, default=10.0)
    parser.add_argument("--gap-min-sec", type=float, default=0.2)
    parser.add_argument("--gap-max-sec", type=float, default=1.0)
    parser.add_argument("--target-sample-rate", type=int, default=44_100)
    parser.add_argument(
        "--disable-normalize",
        action="store_true",
        help="Skip peak normalization before saving.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing session directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work without generating audio.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    build_sessions(args)


if __name__ == "__main__":
    main()

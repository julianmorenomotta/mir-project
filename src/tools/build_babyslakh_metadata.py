"""Build Moises-style metadata/index CSVs from BabySlakh tracks.

Outputs under --output-root:
- splits.csv
- stems.csv
- durations.csv
- test_indices.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import soundfile as sf


# Maps BabySlakh class names to Moises-style stem labels used by Query-Bandit.
DEFAULT_STEM_MAP: Dict[str, str] = {
    "drums": "drums",
    "bass": "bass_guitar",
    "guitar": "acoustic_guitar",
    "piano": "grand_piano",
    "organ": "organ_electric_organ",
    "synth lead": "synth_lead",
    "synth pad": "synth_pad",
    "strings (continued)": "string_section",
    "brass": "brass",
    "sound effects": "fx",
    "percussive": "pitched_percussion",
    "chromatic percussion": "pitched_percussion",
    "pitched percussion": "pitched_percussion",
}


@dataclass(frozen=True)
class StemSpec:
    stem_id: str
    inst_class: str
    mapped_stem: Optional[str]


@dataclass(frozen=True)
class TrackSpec:
    song_id: str
    track_dir: Path
    mix_path: Path
    stems: List[StemSpec]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _parse_metadata_yaml(path: Path, stem_map: Dict[str, str]) -> List[StemSpec]:
    stems: List[StemSpec] = []
    current_stem_id: Optional[str] = None
    current_inst_class: Optional[str] = None

    stem_header = re.compile(r"^\s{2}(S\d+):\s*$")
    inst_class_line = re.compile(r"^\s{4}inst_class:\s*(.+?)\s*$")

    for raw in path.read_text().splitlines():
        m_header = stem_header.match(raw)
        if m_header:
            if current_stem_id is not None:
                mapped = stem_map.get(_norm(current_inst_class or ""))
                stems.append(
                    StemSpec(
                        stem_id=current_stem_id,
                        inst_class=(current_inst_class or ""),
                        mapped_stem=mapped,
                    )
                )
            current_stem_id = m_header.group(1)
            current_inst_class = None
            continue

        m_inst = inst_class_line.match(raw)
        if m_inst and current_stem_id is not None:
            current_inst_class = m_inst.group(1).strip()

    if current_stem_id is not None:
        mapped = stem_map.get(_norm(current_inst_class or ""))
        stems.append(
            StemSpec(
                stem_id=current_stem_id,
                inst_class=(current_inst_class or ""),
                mapped_stem=mapped,
            )
        )

    return stems


def _discover_tracks(
    input_root: Path, stem_map: Dict[str, str], selected_tracks: Optional[Set[str]]
) -> List[TrackSpec]:
    tracks: List[TrackSpec] = []
    for track_dir in sorted(input_root.glob("Track*")):
        if not track_dir.is_dir():
            continue
        song_id = track_dir.name
        if selected_tracks is not None and song_id not in selected_tracks:
            continue

        metadata_path = track_dir / "metadata.yaml"
        mix_path = track_dir / "mix.wav"
        if not metadata_path.exists() or not mix_path.exists():
            continue

        stems = _parse_metadata_yaml(metadata_path, stem_map)
        tracks.append(
            TrackSpec(
                song_id=song_id,
                track_dir=track_dir,
                mix_path=mix_path,
                stems=stems,
            )
        )

    if not tracks:
        raise RuntimeError(f"No tracks found under {input_root}")

    return tracks


def _write_splits(path: Path, tracks: Sequence[TrackSpec], split_id: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["song_id", "split"])
        for track in tracks:
            writer.writerow([track.song_id, split_id])


def _write_durations(path: Path, tracks: Sequence[TrackSpec]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["song_id", "duration"])
        for track in tracks:
            info = sf.info(str(track.mix_path))
            duration = float(info.frames) / float(info.samplerate)
            writer.writerow([track.song_id, f"{duration:.6f}"])


def _write_stems(
    path: Path, tracks: Sequence[TrackSpec], all_stems: Sequence[str]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["song_id", *all_stems])

        for track in tracks:
            present = {s.mapped_stem for s in track.stems if s.mapped_stem}
            row = [track.song_id]
            for stem in all_stems:
                row.append("1" if stem in present else "0")
            writer.writerow(row)


def _write_test_indices(
    path: Path,
    tracks: Sequence[TrackSpec],
    allowed_stems: Sequence[str],
    self_query_only: bool,
) -> None:
    stem_to_songs: Dict[str, List[str]] = defaultdict(list)
    song_to_stems: Dict[str, List[str]] = {}

    for track in tracks:
        present = sorted({s.mapped_stem for s in track.stems if s.mapped_stem})
        present = [stem for stem in present if stem in allowed_stems]
        song_to_stems[track.song_id] = present
        for stem in present:
            stem_to_songs[stem].append(track.song_id)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["song_id", "query_id", "stem", "same_genre", "different_artist"]
        )

        for song_id in sorted(song_to_stems):
            for stem in song_to_stems[song_id]:
                # Smoke-safe default path.
                writer.writerow([song_id, song_id, stem, True, True])

                if self_query_only:
                    continue

                # Cross-track pairing fallback: first other song with same stem.
                candidates = [sid for sid in stem_to_songs[stem] if sid != song_id]
                if candidates:
                    writer.writerow([song_id, candidates[0], stem, False, False])


def _summarize_unmapped(tracks: Sequence[TrackSpec]) -> None:
    unmapped_counts: Dict[str, int] = defaultdict(int)
    for track in tracks:
        for stem in track.stems:
            if stem.mapped_stem is None:
                key = stem.inst_class or "<missing>"
                unmapped_counts[key] += 1

    if not unmapped_counts:
        print("[OK] All BabySlakh inst_class values were mapped.")
        return

    print(
        "[WARN] Unmapped inst_class values detected (excluded from stems.csv/test_indices):"
    )
    for inst_class, count in sorted(
        unmapped_counts.items(), key=lambda x: (-x[1], x[0])
    ):
        print(f"  - {inst_class}: {count}")


def build_metadata(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stem_map = {_norm(k): v for k, v in DEFAULT_STEM_MAP.items()}

    selected_tracks = set(args.tracks) if args.tracks else None
    tracks = _discover_tracks(input_root, stem_map, selected_tracks)

    discovered_stems = sorted(
        {
            stem.mapped_stem
            for track in tracks
            for stem in track.stems
            if stem.mapped_stem is not None
        }
    )

    allowed_stems = (
        sorted(set(args.allowed_stems)) if args.allowed_stems else discovered_stems
    )

    if not allowed_stems:
        raise RuntimeError(
            "No mapped stems available to write stems.csv/test_indices.csv"
        )

    _write_splits(output_root / "splits.csv", tracks, split_id=args.test_split)
    _write_stems(output_root / "stems.csv", tracks, all_stems=allowed_stems)
    _write_durations(output_root / "durations.csv", tracks)
    _write_test_indices(
        output_root / "test_indices.csv",
        tracks,
        allowed_stems=allowed_stems,
        self_query_only=args.self_query_only,
    )

    _summarize_unmapped(tracks)

    print(f"[OK] Wrote {output_root / 'splits.csv'}")
    print(f"[OK] Wrote {output_root / 'stems.csv'} with {len(allowed_stems)} stems")
    print(f"[OK] Wrote {output_root / 'durations.csv'}")
    print(f"[OK] Wrote {output_root / 'test_indices.csv'}")
    print(f"[OK] Tracks processed: {len(tracks)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/datasets/babyslakh_16k/babyslakh_16k"),
        help="Root containing Track*/ folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/babyslakh_16k"),
        help="Directory where CSV metadata files are written.",
    )
    parser.add_argument(
        "--test-split",
        type=int,
        default=5,
        help="Fold id used in splits.csv for test-only Moises workflow.",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=None,
        help="Optional subset of song IDs (e.g., Track00001 Track00002).",
    )
    parser.add_argument(
        "--allowed-stems",
        nargs="+",
        default=None,
        help="Optional explicit stem allow-list for stems.csv and test_indices.csv.",
    )
    parser.add_argument(
        "--self-query-only",
        action="store_true",
        help="Write only query_id=song_id rows in test_indices.csv (recommended smoke default).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    build_metadata(args)


if __name__ == "__main__":
    main()

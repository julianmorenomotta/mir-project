#!/usr/bin/env python3
"""Convert BabySlakh WAV stems into Moises-style npy2/npyq artifacts.

Outputs under --output-root:
- npy2/<song_id>/<stem>.npy + mixture.npy
- npyq/<song_id>/<stem>.query-10s.npy
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import librosa
import numpy as np
import soundfile as sf


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
    input_root: Path,
    stem_map: Dict[str, str],
    selected_tracks: Optional[Set[str]],
) -> List[TrackSpec]:
    tracks: List[TrackSpec] = []
    for track_dir in sorted(input_root.glob("Track*")):
        if not track_dir.is_dir():
            continue

        song_id = track_dir.name
        if selected_tracks is not None and song_id not in selected_tracks:
            continue

        metadata_path = track_dir / "metadata.yaml"
        if not metadata_path.exists():
            continue

        stems = _parse_metadata_yaml(metadata_path, stem_map)
        tracks.append(TrackSpec(song_id=song_id, track_dir=track_dir, stems=stems))

    if not tracks:
        raise RuntimeError(f"No tracks found under {input_root}")

    return tracks


def _load_allowed_stems(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None or not path.exists():
        return None

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)

    if not header or header[0] != "song_id":
        raise RuntimeError(f"Unexpected stems.csv header in {path}")

    return set(header[1:])


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    # audio shape expected as (channels, samples)
    if audio.ndim != 2:
        raise RuntimeError(f"Expected 2D audio array, got shape {audio.shape}")

    channels = audio.shape[0]
    if channels == 1:
        return np.repeat(audio, repeats=2, axis=0)
    if channels >= 2:
        return audio[:2, :]

    raise RuntimeError("Audio has zero channels")


def _read_audio_resample(path: Path, target_sr: int) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=True)
    audio = audio.T.astype(np.float32, copy=False)
    audio = _ensure_stereo(audio)

    if sample_rate == target_sr:
        return audio

    resampled = []
    for ch in range(audio.shape[0]):
        resampled_ch = librosa.resample(
            audio[ch],
            orig_sr=sample_rate,
            target_sr=target_sr,
        )
        resampled.append(resampled_ch)

    out = np.stack(resampled, axis=0).astype(np.float32)
    return out


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32, copy=False))


def _extract_onset_query(audio: np.ndarray, sample_rate: int, seconds: float, hop: int) -> np.ndarray:
    target_samples = int(round(seconds * sample_rate))
    if target_samples <= 0:
        raise RuntimeError("query duration must be positive")

    n_samples = audio.shape[-1]
    if n_samples <= target_samples:
        out = np.zeros((audio.shape[0], target_samples), dtype=np.float32)
        out[:, :n_samples] = audio
        return out

    mono = np.mean(audio, axis=0)
    onset_strength = librosa.onset.onset_strength(y=mono, sr=sample_rate, hop_length=hop)

    if onset_strength.size == 0:
        start_sample = 0
    else:
        n_frames_per_chunk = max(1, target_samples // hop)
        if onset_strength.size < n_frames_per_chunk:
            start_frame = int(np.argmax(onset_strength))
        else:
            kernel = np.ones(n_frames_per_chunk, dtype=np.float32) / float(n_frames_per_chunk)
            score = np.convolve(onset_strength.astype(np.float32), kernel, mode="valid")
            start_frame = int(np.argmax(score))

        start_sample = int(librosa.frames_to_samples(start_frame, hop_length=hop))
        start_sample = max(0, min(start_sample, n_samples - target_samples))

    end_sample = start_sample + target_samples
    return audio[:, start_sample:end_sample]


def _process_track(
    track: TrackSpec,
    output_root: Path,
    target_sr: int,
    query_seconds: float,
    query_file: str,
    allowed_stems: Optional[Set[str]],
) -> Dict[str, int]:
    stem_to_segments: Dict[str, List[np.ndarray]] = defaultdict(list)

    for stem in track.stems:
        if stem.mapped_stem is None:
            continue
        if allowed_stems is not None and stem.mapped_stem not in allowed_stems:
            continue

        wav_path = track.track_dir / "stems" / f"{stem.stem_id}.wav"
        if not wav_path.exists():
            continue

        audio = _read_audio_resample(wav_path, target_sr)
        stem_to_segments[stem.mapped_stem].append(audio)

    if not stem_to_segments:
        print(f"[WARN] {track.song_id}: no mapped stems found, skipping")
        return {"stems": 0, "queries": 0}

    min_length = min(
        segment.shape[-1]
        for segments in stem_to_segments.values()
        for segment in segments
    )

    npy2_song_dir = output_root / "npy2" / track.song_id
    npyq_song_dir = output_root / "npyq" / track.song_id
    npy2_song_dir.mkdir(parents=True, exist_ok=True)
    npyq_song_dir.mkdir(parents=True, exist_ok=True)

    mixture = np.zeros((2, min_length), dtype=np.float32)
    num_stems = 0
    num_queries = 0

    for stem_name, segments in sorted(stem_to_segments.items()):
        stack = np.stack([segment[:, :min_length] for segment in segments], axis=0)
        stem_audio = np.sum(stack, axis=0).astype(np.float32)

        _save_npy(npy2_song_dir / f"{stem_name}.npy", stem_audio)
        mixture += stem_audio
        num_stems += 1

        query = _extract_onset_query(
            stem_audio,
            sample_rate=target_sr,
            seconds=query_seconds,
            hop=512,
        )
        _save_npy(npyq_song_dir / f"{stem_name}.{query_file}.npy", query)
        num_queries += 1

    _save_npy(npy2_song_dir / "mixture.npy", mixture)

    print(
        f"[OK] {track.song_id}: stems={num_stems}, queries={num_queries}, samples={min_length}"
    )
    return {"stems": num_stems, "queries": num_queries}


def build_npy_artifacts(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stem_map = {_norm(k): v for k, v in DEFAULT_STEM_MAP.items()}
    selected_tracks = set(args.tracks) if args.tracks else None
    tracks = _discover_tracks(input_root, stem_map, selected_tracks)

    allowed_stems = _load_allowed_stems(args.stems_csv)
    if args.allowed_stems:
        requested = set(args.allowed_stems)
        allowed_stems = requested if allowed_stems is None else allowed_stems & requested

    total_stems = 0
    total_queries = 0

    for track in tracks:
        stats = _process_track(
            track,
            output_root=output_root,
            target_sr=args.target_sample_rate,
            query_seconds=args.query_seconds,
            query_file=args.query_file,
            allowed_stems=allowed_stems,
        )
        total_stems += stats["stems"]
        total_queries += stats["queries"]

    print(f"[OK] Tracks processed: {len(tracks)}")
    print(f"[OK] Stem files written: {total_stems}")
    print(f"[OK] Query files written: {total_queries}")


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
        help="Root where npy2/npyq directories are written.",
    )
    parser.add_argument(
        "--stems-csv",
        type=Path,
        default=Path("data/datasets/babyslakh_16k/stems.csv"),
        help="Optional stems.csv used to filter allowed stems by column names.",
    )
    parser.add_argument(
        "--allowed-stems",
        nargs="+",
        default=None,
        help="Optional explicit stem allow-list.",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=None,
        help="Optional subset of song IDs (e.g., Track00001 Track00002).",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=44_100,
        help="Target sample rate used by Moises data pipeline.",
    )
    parser.add_argument(
        "--query-seconds",
        type=float,
        default=10.0,
        help="Duration in seconds for query extraction.",
    )
    parser.add_argument(
        "--query-file",
        default="query-10s",
        help="Query suffix used in filename <stem>.<query-file>.npy",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    build_npy_artifacts(args)


if __name__ == "__main__":
    main()

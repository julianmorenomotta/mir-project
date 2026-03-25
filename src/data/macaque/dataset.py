from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

try:  # pragma: no cover - only triggered outside Banquet repo root
    from core.types import input_dict
except ModuleNotFoundError:  # pragma: no cover
    THIRD_PARTY_ROOT = (
        Path(__file__).resolve().parents[3] / "third_party" / "query-bandit"
    )
    if THIRD_PARTY_ROOT.exists() and str(THIRD_PARTY_ROOT) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY_ROOT))
    from core.types import input_dict  # type: ignore


@dataclass(frozen=True)
class SessionRecord:
    """Lightweight container describing a synthesized mixture directory."""

    index: int
    session_dir: Path
    mixture_path: Path
    stem_paths: Dict[str, Path]
    metadata: Dict[str, object]

    @property
    def session_id(self) -> str:
        return self.session_dir.name

    @property
    def speakers(self) -> List[str]:
        return sorted(self.stem_paths.keys())


class MacaqueDataset(Dataset):
    """PyTorch dataset that feeds Banquet-style batches from macaque mixtures.

    Each item contains a mixture waveform, the ground-truth target stem for one
    of the speakers, a 10 s enrollment clip sampled from the held-out query pool,
    and lineage metadata.
    """

    def __init__(
        self,
        data_root: Path | str = Path("data/macaque_dataset"),
        split: str = "train",
        query_root: Optional[Path | str] = None,
        target_sample_rate: int = 44_100,
        query_duration_sec: float = 10.0,
        speaker_selection: str = "random",
        seed: int = 1337,
    ) -> None:
        if query_duration_sec <= 0:
            raise ValueError("query_duration_sec must be positive.")
        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive.")
        if speaker_selection not in {"random", "cycle", "first"}:
            raise ValueError(
                "speaker_selection must be one of {'random', 'cycle', 'first'}."
            )

        self.data_root = Path(data_root)
        self.split = split
        self.split_root = self.data_root / split
        self._split_candidates = self._build_split_candidates(split)
        self.query_root = (
            Path(query_root) if query_root is not None else self.data_root / "queries"
        )
        self.target_sample_rate = int(target_sample_rate)
        self.query_duration_sec = float(query_duration_sec)
        self.query_num_samples = int(
            round(self.query_duration_sec * self.target_sample_rate)
        )
        self.speaker_selection = speaker_selection
        self.seed = seed

        if not self.split_root.exists():
            fallback = self._resolve_existing_split_root()
            if fallback is None:
                raise FileNotFoundError(f"Split directory not found: {self.split_root}")
            self.split_root = fallback

        self.sessions = self._discover_sessions()
        if not self.sessions:
            raise RuntimeError(f"No sessions discovered under {self.split_root}.")

        self.query_index = self._discover_queries()
        self._validate_query_coverage()

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, index: int):
        session = self.sessions[index]
        speaker_id = self._select_speaker(session, index)
        target_path = session.stem_paths[speaker_id]
        query_path = self._select_query_clip(speaker_id, index)

        mixture = self._load_wave(session.mixture_path)
        target = self._load_wave(target_path)
        mixture, target = self._align_lengths(mixture, target)

        query = self._load_wave(query_path)
        query = self._pad_or_trim(query, self.query_num_samples)

        metadata = {
            "split": self.split,
            "session_id": session.session_id,
            "session_dir": session.session_dir.relative_to(self.data_root).as_posix(),
            "mixture_file": session.mixture_path.relative_to(self.data_root).as_posix(),
            "target_file": target_path.relative_to(self.data_root).as_posix(),
            "query_file": query_path.relative_to(self.query_root).as_posix(),
            "speakers": session.speakers,
            "target_speaker": speaker_id,
            "stem": speaker_id,
            "session_metadata": session.metadata,
        }

        return input_dict(
            mixture=mixture,
            sources={"target": target},
            query=query,
            metadata=metadata,
            modality="audio",
        )

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------
    def _build_split_candidates(self, split: str) -> List[str]:
        if split == "val":
            return ["val", "valid"]
        if split == "valid":
            return ["valid", "val"]
        return [split]

    def _resolve_existing_split_root(self) -> Optional[Path]:
        for candidate in self._split_candidates[1:]:
            candidate_root = self.data_root / candidate
            if candidate_root.exists():
                return candidate_root
        return None

    def _discover_sessions(self) -> List[SessionRecord]:
        records: List[SessionRecord] = []
        session_dirs = sorted(
            path
            for path in self.split_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        for idx, session_dir in enumerate(session_dirs):
            metadata_path = session_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text())

            mixture_path = self._resolve_media_path(
                metadata.get("mixture_file"),
                fallback=session_dir / metadata.get("mixture_filename", "mixture.wav"),
            )
            if not mixture_path.exists():
                raise FileNotFoundError(f"Mixture file missing: {mixture_path}")

            stem_files = metadata.get("stem_files", {})
            stem_paths: Dict[str, Path] = {}
            for speaker, rel_path in stem_files.items():
                speaker_path = self._resolve_media_path(
                    rel_path, fallback=session_dir / f"{speaker}.wav"
                )
                if not speaker_path.exists():
                    raise FileNotFoundError(
                        f"Stem file missing for {speaker}: {speaker_path}"
                    )
                stem_paths[speaker] = speaker_path

            if not stem_paths:
                raise RuntimeError(f"Session {session_dir} is missing speaker stems.")

            records.append(
                SessionRecord(
                    index=idx,
                    session_dir=session_dir,
                    mixture_path=mixture_path,
                    stem_paths=stem_paths,
                    metadata=metadata,
                )
            )
        return records

    def _discover_queries(self) -> Dict[str, List[Path]]:
        index: Dict[str, List[Path]] = {}
        seen_query_dir = False
        for split_name in self._split_candidates:
            query_split_dir = self.query_root / split_name
            if not query_split_dir.exists():
                continue
            seen_query_dir = True
            for individual_dir in sorted(query_split_dir.iterdir()):
                if not individual_dir.is_dir() or individual_dir.name.startswith("."):
                    continue
                wavs = sorted(individual_dir.glob("*.wav"))
                if not wavs:
                    continue
                key = individual_dir.name.upper()
                index.setdefault(key, []).extend(wavs)

        if not seen_query_dir:
            requested = self.query_root / self.split
            raise FileNotFoundError(f"Query split directory not found: {requested}")

        return index

    def _validate_query_coverage(self) -> None:
        missing = {
            speaker
            for record in self.sessions
            for speaker in record.speakers
            if speaker.upper() not in self.query_index
        }
        if missing:
            raise RuntimeError(
                "Missing query clips for speakers: " + ", ".join(sorted(missing))
            )

    def _resolve_media_path(self, rel: Optional[str], fallback: Path) -> Path:
        if rel:
            rel_path = Path(rel)
            if rel_path.is_absolute():
                return rel_path
            primary = (self.data_root / rel_path).resolve()
            if primary.exists():
                return primary

            rel_str = rel_path.as_posix()
            if "/valid/" in rel_str:
                alt = (self.data_root / Path(rel_str.replace("/valid/", "/val/"))).resolve()
                if alt.exists():
                    return alt
            if "/val/" in rel_str:
                alt = (self.data_root / Path(rel_str.replace("/val/", "/valid/"))).resolve()
                if alt.exists():
                    return alt
            return primary
        return fallback.resolve()

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------
    def _select_speaker(self, session: SessionRecord, sample_index: int) -> str:
        speakers = session.speakers
        if len(speakers) == 1 or self.speaker_selection == "first":
            return speakers[0]
        if self.speaker_selection == "cycle":
            return speakers[sample_index % len(speakers)]
        salt = self._salt(session.index, sample_index, 0)
        rng = random.Random(salt)
        return speakers[rng.randrange(len(speakers))]

    def _select_query_clip(self, speaker_id: str, sample_index: int) -> Path:
        speaker_key = speaker_id.upper()
        clips = self.query_index.get(speaker_key)
        if not clips:
            raise RuntimeError(f"No query clips registered for speaker '{speaker_id}'.")
        speaker_tag = abs(hash(speaker_key)) % 1_000_000_007
        salt = self._salt(speaker_tag, sample_index, 1)
        rng = random.Random(salt)
        return clips[rng.randrange(len(clips))]

    def _salt(self, primary: int, secondary: int, offset: int) -> int:
        return (
            (self.seed + 1) * 1_000_003
            + primary * 10_000_019
            + secondary * 1_000_033
            + offset * 97
        )

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------
    def _load_wave(self, path: Path) -> np.ndarray:
        audio, sample_rate = sf.read(path, dtype="float32")
        if sample_rate != self.target_sample_rate:
            raise ValueError(
                f"Expected sample rate {self.target_sample_rate}, got {sample_rate} for {path}."
            )
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        else:
            audio = audio.T
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _align_lengths(
        self, mixture: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        length = min(mixture.shape[-1], target.shape[-1])
        if length == 0:
            raise RuntimeError("Encountered zero-length audio segment.")
        return mixture[..., :length], target[..., :length]

    def _pad_or_trim(self, audio: np.ndarray, num_samples: int) -> np.ndarray:
        current = audio.shape[-1]
        if current == num_samples:
            return audio
        if current > num_samples:
            return audio[..., :num_samples]
        pad = np.zeros((audio.shape[0], num_samples - current), dtype=audio.dtype)
        return np.concatenate([audio, pad], axis=-1)

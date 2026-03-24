from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchaudio
from torchaudio import functional as AF

__all__ = ["SessionMixer", "SessionMixResult", "CallPlacement"]


def _extract_individual_id(path: Path) -> Optional[str]:
    stem = path.stem
    if len(stem) < 2:
        return None
    prefix = stem[:2]
    if not prefix.isalpha():
        return None
    return prefix.upper()


@dataclass
class CallPlacement:
    """Metadata describing where an individual call was placed on the session timeline."""

    relative_path: str
    original_sample_rate: int
    original_duration_sec: float
    resampled_duration_sec: float
    start_sample: int
    end_sample: int
    gap_before_sec: float


@dataclass
class SessionMixResult:
    """Container returned by :class:`SessionMixer.generate_session`."""

    split: str
    sample_rate: int
    duration_sec: float
    mixture: torch.Tensor
    stems: Dict[str, torch.Tensor]
    placements: Dict[str, List[CallPlacement]]
    metadata: Dict[str, object]


class SessionMixer:
    """Generate multi-call macaque sessions ready for mixture export.

    The mixer constructs 8–10 second (configurable) timelines for two randomly
    chosen individuals per split. Each individual's timeline is assembled by
    sequentially laying down calls separated by random gaps, all resampled to a
    common sample rate (44.1 kHz by default). The resulting two timelines serve
    as stems; their sum becomes the mixture waveform.
    """

    def __init__(
        self,
        raw_root: Path | str = Path("data/macaque_raw"),
        query_pool_csv: Path | str | None = Path("data/macaque_raw/query_pool.csv"),
        splits: Sequence[str] = ("train", "val"),
        session_duration_range: Tuple[float, float] = (8.0, 10.0),
        gap_range: Tuple[float, float] = (0.2, 1.0),
        target_sample_rate: int = 44_100,
        extensions: Sequence[str] = (".wav",),
        normalize_output: bool = True,
        dtype: torch.dtype = torch.float32,
        max_segments_per_track: int = 128,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.query_pool_csv = Path(query_pool_csv) if query_pool_csv else None
        self.splits = tuple(splits)
        self.session_duration_range = session_duration_range
        self.gap_range = gap_range
        self.target_sample_rate = int(target_sample_rate)
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.normalize_output = normalize_output
        self.dtype = dtype
        self.max_segments_per_track = max_segments_per_track

        if self.session_duration_range[0] <= 0 or self.session_duration_range[1] <= 0:
            raise ValueError("Session duration range must be positive.")
        if self.session_duration_range[0] > self.session_duration_range[1]:
            raise ValueError("Session duration range must be ordered (min <= max).")
        if self.gap_range[0] < 0 or self.gap_range[1] < 0:
            raise ValueError("Gap range must be non-negative.")
        if self.gap_range[0] > self.gap_range[1]:
            raise ValueError("Gap range must be ordered (min <= max).")

        self._query_pool = self._load_query_pool()
        self._mixture_pool = self._build_mixture_pool()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def available_individuals(self, split: str) -> List[str]:
        """Return the list of individuals with mixture-pool calls for *split*."""

        pool = self._mixture_pool.get(split)
        if not pool:
            return []
        return sorted(individual for individual, files in pool.items() if files)

    def generate_session(
        self,
        split: str,
        seed: Optional[int] = None,
        duration_seconds: Optional[float] = None,
    ) -> SessionMixResult:
        """Create a new two-speaker session for the given *split*.

        Parameters
        ----------
        split:
            Dataset split name (usually ``"train"`` or ``"val"``).
        seed:
            Optional random seed for deterministic call/gap selection.
        duration_seconds:
            Optional fixed session length in seconds. Defaults to a uniform draw
            inside ``session_duration_range`` when ``None``.
        """

        if split not in self._mixture_pool:
            raise ValueError(f"Unknown split '{split}'. Available: {list(self._mixture_pool)}")

        if seed is None:
            actual_seed = random.randrange(0, 2**32)
        else:
            actual_seed = seed
        rng = random.Random(actual_seed)
        available = self.available_individuals(split)
        if len(available) < 2:
            raise RuntimeError(f"Need at least two individuals with mixture data in split '{split}'.")

        chosen_ids = rng.sample(available, 2)
        if duration_seconds is None:
            duration_seconds = rng.uniform(*self.session_duration_range)
        target_samples = int(self.target_sample_rate * duration_seconds)
        if target_samples <= 0:
            raise ValueError("Target session duration must be positive once converted to samples.")

        stems: Dict[str, torch.Tensor] = {}
        placements: Dict[str, List[CallPlacement]] = {}
        for individual_id in chosen_ids:
            stem_waveform, stem_placements = self._build_stem(
                self._mixture_pool[split][individual_id],
                rng=rng,
                target_samples=target_samples,
            )
            stems[individual_id] = stem_waveform
            placements[individual_id] = stem_placements

        mixture = torch.zeros(target_samples, dtype=self.dtype)
        for stem_wave in stems.values():
            mixture += stem_wave

        if self.normalize_output:
            peak = max(mixture.abs().max().item(), *(wave.abs().max().item() for wave in stems.values()))
            if peak > 1.0:
                scale = 1.0 / peak
                mixture = mixture * scale
                stems = {k: v * scale for k, v in stems.items()}

        metadata = {
            "split": split,
            "seed": actual_seed,
            "session_duration_sec": duration_seconds,
            "target_sample_rate": self.target_sample_rate,
            "gap_range_sec": self.gap_range,
            "selected_individuals": chosen_ids,
            "placements": {
                individual_id: [asdict(call) for call in placement_list]
                for individual_id, placement_list in placements.items()
            },
        }

        return SessionMixResult(
            split=split,
            sample_rate=self.target_sample_rate,
            duration_sec=duration_seconds,
            mixture=mixture,
            stems=stems,
            placements=placements,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_query_pool(self) -> set[str]:
        if not self.query_pool_csv or not self.query_pool_csv.exists():
            return set()
        pool: set[str] = set()
        with self.query_pool_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rel_path = row.get("relative_path") or row.get("path")
                if rel_path:
                    pool.add(rel_path.strip())
        return pool

    def _build_mixture_pool(self) -> Dict[str, Dict[str, List[Path]]]:
        pool: Dict[str, Dict[str, List[Path]]] = {}
        for split in self.splits:
            split_dir = self.raw_root / split
            if not split_dir.exists():
                continue
            individual_map: Dict[str, List[Path]] = {}
            for wav_path in sorted(split_dir.rglob("*")):
                if not wav_path.is_file() or wav_path.suffix.lower() not in self.extensions:
                    continue
                rel_path = wav_path.relative_to(self.raw_root).as_posix()
                if rel_path in self._query_pool:
                    continue
                individual_id = _extract_individual_id(wav_path)
                if individual_id is None:
                    continue
                individual_map.setdefault(individual_id, []).append(wav_path)
            pool[split] = individual_map
        return pool

    def _load_waveform(self, path: Path) -> Tuple[torch.Tensor, int, int]:
        waveform, sample_rate = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        original_num_samples = waveform.shape[-1]
        if sample_rate != self.target_sample_rate:
            waveform = AF.resample(waveform, sample_rate, self.target_sample_rate)
        waveform = waveform.to(self.dtype).squeeze(0)
        return waveform, sample_rate, original_num_samples

    def _build_stem(
        self,
        file_paths: Sequence[Path],
        rng: random.Random,
        target_samples: int,
    ) -> Tuple[torch.Tensor, List[CallPlacement]]:
        if not file_paths:
            raise RuntimeError("Cannot build stem without source calls.")

        stem = torch.zeros(target_samples, dtype=self.dtype)
        placements: List[CallPlacement] = []
        cursor = 0
        first_call = True
        iterations = 0

        while cursor < target_samples and iterations < self.max_segments_per_track:
            gap_before = 0.0
            if not first_call:
                gap_before = rng.uniform(*self.gap_range)
                gap_samples = min(int(round(gap_before * self.target_sample_rate)), target_samples - cursor)
                cursor += gap_samples
            else:
                first_call = False

            if cursor >= target_samples:
                break

            call_path = rng.choice(file_paths)
            waveform, original_sr, original_num_samples = self._load_waveform(call_path)
            if waveform.numel() == 0:
                continue

            remaining = target_samples - cursor
            if waveform.shape[-1] > remaining:
                waveform = waveform[:remaining]
            length = waveform.shape[-1]
            if length == 0:
                break

            stem[cursor : cursor + length] += waveform

            placement = CallPlacement(
                relative_path=call_path.relative_to(self.raw_root).as_posix(),
                original_sample_rate=original_sr,
                original_duration_sec=original_num_samples / original_sr if original_sr else 0.0,
                resampled_duration_sec=length / self.target_sample_rate,
                start_sample=cursor,
                end_sample=cursor + length,
                gap_before_sec=gap_before,
            )
            placements.append(placement)

            cursor += length
            iterations += 1

        if not placements:
            raise RuntimeError("Failed to place any calls when building stem.")

        return stem, placements

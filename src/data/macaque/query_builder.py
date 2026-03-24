from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchaudio
from torchaudio import functional as AF

__all__ = ["QueryBuilder", "QueryClipResult", "QueryClipPlacement"]


@dataclass
class QueryClipPlacement:
    """Metadata describing a single call inside a query clip."""

    relative_path: str
    original_sample_rate: int
    original_duration_sec: float
    resampled_duration_sec: float
    start_sample: int
    end_sample: int
    gap_before_sec: float


@dataclass
class QueryClipResult:
    """Container returned by :class:`QueryBuilder.build_clip`."""

    split: str
    individual_id: str
    sample_rate: int
    target_duration_sec: float
    waveform: torch.Tensor
    placements: List[QueryClipPlacement]
    metadata: Dict[str, object]


class QueryBuilder:
    """Construct deterministic query enrollment clips from held-out calls."""

    def __init__(
        self,
        raw_root: Path | str = Path("data/macaque_raw"),
        query_pool_csv: Path | str = Path("data/macaque_raw/query_pool.csv"),
        splits: Sequence[str] = ("train", "val"),
        clip_duration_sec: float = 10.0,
        gap_range: Tuple[float, float] = (0.2, 1.0),
        target_sample_rate: int = 44_100,
        extensions: Sequence[str] = (".wav",),
        normalize_output: bool = True,
        dtype: torch.dtype = torch.float32,
        max_segments_per_clip: int = 256,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.query_pool_csv = Path(query_pool_csv)
        self.splits = tuple(splits)
        self.clip_duration_sec = float(clip_duration_sec)
        self.gap_range = gap_range
        self.target_sample_rate = int(target_sample_rate)
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.normalize_output = normalize_output
        self.dtype = dtype
        self.max_segments_per_clip = max_segments_per_clip

        if not self.query_pool_csv.exists():
            raise FileNotFoundError(f"Query pool CSV not found: {self.query_pool_csv}")
        if self.clip_duration_sec <= 0:
            raise ValueError("clip_duration_sec must be positive.")
        if self.gap_range[0] < 0 or self.gap_range[1] < 0:
            raise ValueError("gap_range values must be non-negative.")
        if self.gap_range[0] > self.gap_range[1]:
            raise ValueError("gap_range must satisfy min <= max.")

        self._query_map = self._load_query_pool()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def available_individuals(self, split: str) -> List[str]:
        pool = self._query_map.get(split)
        if not pool:
            return []
        return sorted(individual for individual, files in pool.items() if files)

    def build_clip(
        self,
        split: str,
        individual_id: str,
        seed: Optional[int] = None,
        duration_seconds: Optional[float] = None,
    ) -> QueryClipResult:
        """Assemble a single query clip for ``individual_id`` within ``split``."""

        if split not in self._query_map:
            raise ValueError(
                f"Unknown split '{split}'. Available: {list(self._query_map)}"
            )
        individual_pool = self._query_map[split]
        if individual_id not in individual_pool:
            raise ValueError(
                f"Individual '{individual_id}' not present in split '{split}'."
            )
        file_paths = individual_pool[individual_id]
        if not file_paths:
            raise RuntimeError(
                f"No query calls available for individual '{individual_id}' in split '{split}'."
            )

        actual_seed = seed if seed is not None else random.randrange(0, 2**32)
        rng = random.Random(actual_seed)
        target_duration = (
            duration_seconds if duration_seconds is not None else self.clip_duration_sec
        )
        target_samples = int(round(target_duration * self.target_sample_rate))
        if target_samples <= 0:
            raise ValueError("Target duration must yield at least one sample.")

        waveform = torch.zeros(target_samples, dtype=self.dtype)
        placements: List[QueryClipPlacement] = []
        cursor = 0
        first_segment = True
        iterations = 0

        call_indices = list(range(len(file_paths)))
        rng.shuffle(call_indices)
        index_pointer = 0

        while cursor < target_samples and iterations < self.max_segments_per_clip:
            gap_before = 0.0
            if not first_segment:
                gap_before = rng.uniform(*self.gap_range)
                gap_samples = min(
                    int(round(gap_before * self.target_sample_rate)),
                    target_samples - cursor,
                )
                cursor += gap_samples
            else:
                first_segment = False

            if cursor >= target_samples:
                break

            if index_pointer >= len(call_indices):
                rng.shuffle(call_indices)
                index_pointer = 0
            call_path = file_paths[call_indices[index_pointer]]
            index_pointer += 1

            waveform_src, original_sr, original_num_samples = self._load_waveform(
                call_path
            )
            if waveform_src.numel() == 0:
                continue

            remaining = target_samples - cursor
            if waveform_src.shape[-1] > remaining:
                waveform_src = waveform_src[:remaining]
            segment_length = waveform_src.shape[-1]
            if segment_length == 0:
                break

            waveform[cursor : cursor + segment_length] += waveform_src

            placement = QueryClipPlacement(
                relative_path=call_path.relative_to(self.raw_root).as_posix(),
                original_sample_rate=original_sr,
                original_duration_sec=(
                    original_num_samples / original_sr if original_sr else 0.0
                ),
                resampled_duration_sec=segment_length / self.target_sample_rate,
                start_sample=cursor,
                end_sample=cursor + segment_length,
                gap_before_sec=gap_before,
            )
            placements.append(placement)

            cursor += segment_length
            iterations += 1

        if not placements:
            raise RuntimeError(
                f"Failed to place any calls for individual '{individual_id}' in split '{split}'."
            )

        if cursor < target_samples:
            # Remaining samples are already zero (silence), nothing to do.
            pass

        if self.normalize_output:
            peak = waveform.abs().max().item()
            if peak > 1.0:
                waveform = waveform / peak

        metadata = {
            "split": split,
            "individual_id": individual_id,
            "seed": actual_seed,
            "target_duration_sec": target_duration,
            "target_sample_rate": self.target_sample_rate,
            "gap_range_sec": self.gap_range,
            "num_segments": len(placements),
            "placements": [asdict(call) for call in placements],
        }

        return QueryClipResult(
            split=split,
            individual_id=individual_id,
            sample_rate=self.target_sample_rate,
            target_duration_sec=target_duration,
            waveform=waveform,
            placements=placements,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_query_pool(self) -> Dict[str, Dict[str, List[Path]]]:
        pool: Dict[str, Dict[str, List[Path]]] = {}
        for split in self.splits:
            pool[split] = {}

        with self.query_pool_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                split = (row.get("split") or "").strip()
                individual_id = (row.get("individual_id") or "").strip().upper()
                relative_path = (row.get("relative_path") or "").strip()
                if not split or split not in pool:
                    continue
                if not individual_id or not relative_path:
                    continue
                absolute_path = self.raw_root / relative_path
                if absolute_path.suffix.lower() not in self.extensions:
                    continue
                if not absolute_path.exists():
                    raise FileNotFoundError(
                        f"Listed query file does not exist: {absolute_path}"
                    )
                pool[split].setdefault(individual_id, []).append(absolute_path)
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

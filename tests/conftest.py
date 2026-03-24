from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture()
def macaque_dataset_root(tmp_path: Path) -> Path:
    sample_rate = 44_100
    total_samples = sample_rate
    dataset_root = tmp_path / "macaque_dataset"

    def write_audio(path: Path, data: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, data, sample_rate)

    def build_split(split: str, speakers: tuple[str, str]) -> None:
        session_dir = dataset_root / split / "mixture_00000"
        session_dir.mkdir(parents=True, exist_ok=True)

        speaker_a, speaker_b = speakers
        wave_a = np.full(total_samples, 0.05, dtype=np.float32)
        wave_b = np.full(total_samples, 0.02, dtype=np.float32)
        mixture = wave_a + wave_b

        stem_a_path = session_dir / f"{speaker_a}.wav"
        stem_b_path = session_dir / f"{speaker_b}.wav"
        mixture_path = session_dir / "mixture.wav"

        write_audio(stem_a_path, wave_a)
        write_audio(stem_b_path, wave_b)
        write_audio(mixture_path, mixture)

        metadata = {
            "split": split,
            "seed": 123,
            "session_duration_sec": len(mixture) / sample_rate,
            "mixture_filename": "mixture.wav",
            "mixture_file": f"{split}/mixture_00000/mixture.wav",
            "selected_individuals": [speaker_a, speaker_b],
            "stem_files": {
                speaker_a: f"{split}/mixture_00000/{speaker_a}.wav",
                speaker_b: f"{split}/mixture_00000/{speaker_b}.wav",
            },
        }
        (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        query_root = dataset_root / "queries" / split
        for speaker in speakers:
            clip_dir = query_root / speaker
            clip_dir.mkdir(parents=True, exist_ok=True)
            clip = np.linspace(0, 1, sample_rate, dtype=np.float32)
            write_audio(clip_dir / "query_clip_000.wav", clip)

    for split in ("train", "val", "test"):
        build_split(split, ("AA", "BB"))

    return dataset_root

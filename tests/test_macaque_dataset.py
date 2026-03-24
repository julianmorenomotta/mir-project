from __future__ import annotations

from pathlib import Path

import pytest

from data.macaque.dataset import MacaqueDataset


def test_macaque_dataset_item_shapes(macaque_dataset_root: Path) -> None:
    dataset = MacaqueDataset(
        data_root=macaque_dataset_root,
        split="train",
        target_sample_rate=44_100,
        query_duration_sec=1.0,
        speaker_selection="cycle",
    )

    sample = dataset[0]

    mixture = sample["mixture"]["audio"]
    target = sample["sources"]["target"]["audio"]
    query = sample["query"]["audio"]

    assert mixture.ndim == 2 and mixture.shape[0] == 1
    assert target.shape == mixture.shape
    assert query.shape == (1, dataset.query_num_samples)
    assert sample["metadata"]["session_id"] == "mixture_00000"
    assert sample["metadata"]["target_speaker"] == "AA"

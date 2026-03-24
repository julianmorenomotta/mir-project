from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pytorch_lightning as pl

from core.data.base import from_datasets
from data.macaque.dataset import MacaqueDataset


def _normalize_kwargs(mapping: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(mapping) if mapping is not None else {}


def MacaqueDataModule(
    data_root: str | Path,
    batch_size: int,
    num_workers: int = 8,
    train_kwargs: Optional[Mapping[str, Any]] = None,
    val_kwargs: Optional[Mapping[str, Any]] = None,
    test_kwargs: Optional[Mapping[str, Any]] = None,
    datamodule_kwargs: Optional[Mapping[str, Any]] = None,
) -> pl.LightningDataModule:
    """Factory that mirrors Banquet's Moises data modules for macaque data."""

    train_dataset = MacaqueDataset(
        data_root=data_root,
        split="train",
        **_normalize_kwargs(train_kwargs),
    )

    val_dataset = MacaqueDataset(
        data_root=data_root,
        split="val",
        **_normalize_kwargs(val_kwargs),
    )

    test_dataset = MacaqueDataset(
        data_root=data_root,
        split="test",
        **_normalize_kwargs(test_kwargs),
    )

    datamodule = from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **_normalize_kwargs(datamodule_kwargs),
    )

    datamodule.predict_dataloader = datamodule.test_dataloader  # type: ignore[method-assign]
    return datamodule

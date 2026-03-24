from __future__ import annotations

from data.macaque.datamodule import MacaqueDataModule
from data.macaque.dataset import MacaqueDataset


def test_macaque_datamodule_builds_split_loaders(macaque_dataset_root) -> None:
    dm = MacaqueDataModule(
        data_root=macaque_dataset_root,
        batch_size=2,
        num_workers=0,
        train_kwargs={"speaker_selection": "cycle"},
        val_kwargs={"speaker_selection": "first"},
        test_kwargs={"speaker_selection": "first"},
    )

    loaders = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
        "test": dm.test_dataloader(),
    }

    for split, loader in loaders.items():
        ds = loader.dataset
        assert isinstance(ds, MacaqueDataset)
        assert ds.split == split
        batch = next(iter(loader))
        audio = batch["mixture"]["audio"]
        expected_batch = min(loader.batch_size or 0, len(ds)) or len(ds)
        assert audio.shape[0] == expected_batch and audio.shape[1] == 1
        assert batch["query"]["audio"].shape[-1] == ds.query_num_samples

    predict_loader = dm.predict_dataloader()
    test_loader = dm.test_dataloader()
    assert isinstance(predict_loader, type(test_loader))

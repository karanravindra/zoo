"""DigitNet data: any `zoo.data` image dataset, with a validation split held out of the train set."""

from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from config import Config

from zoo import data
from zoo.paths import DATA_ROOT


class DigitData(L.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ds_cls = getattr(data, cfg.dataset)

    def prepare_data(self) -> None:  # downloads and caches, once
        self.ds_cls(root=str(DATA_ROOT), train=True)
        self.ds_cls(root=str(DATA_ROOT), train=False)

    def setup(self, stage: str | None = None) -> None:
        full = self.ds_cls(root=str(DATA_ROOT), train=True)
        self.train_ds, self.val_ds = random_split(
            full, [len(full) - self.cfg.val_size, self.cfg.val_size], generator=torch.Generator().manual_seed(self.cfg.seed)
        )
        self.test_ds = self.ds_cls(root=str(DATA_ROOT), train=False)

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        n = self.cfg.num_workers
        return DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=n, persistent_workers=n > 0 and shuffle, pin_memory=True
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, False)

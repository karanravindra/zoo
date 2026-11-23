"""DigitDreamer data: padded MNIST for the autoencoder, and the autoencoder's cached latents for the flow."""

from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset, random_split
from torchvision.transforms import v2

from ae import Autoencoder
from config import AEConfig, FlowConfig

from zoo import data
from zoo.paths import DATA_ROOT
from zoo.utils.device import get_device

# MNIST is 28x28: pad 2 px of background on each side to 32x32, then uint8 [0, 255] -> float [-1, 1]
TO_PM1 = v2.Compose([v2.Pad(2, fill=0), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5,), (0.5,))])


def prepare_data(dataset: str) -> None:
    """Download and cache the dataset's splits (nothing is kept in memory)."""
    ds_cls = getattr(data, dataset)
    for train in (True, False):
        ds_cls(root=str(DATA_ROOT), train=train)


def make_splits(dataset: str, val_size: int, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    """(train, val, test), with val carved deterministically out of the train split."""
    ds_cls = getattr(data, dataset)
    full_train = ds_cls(root=str(DATA_ROOT), train=True, transform=TO_PM1)
    test = ds_cls(root=str(DATA_ROOT), train=False, transform=TO_PM1)
    sizes = [len(full_train) - val_size, val_size]
    train, val = random_split(full_train, sizes, generator=torch.Generator().manual_seed(seed))
    return train, val, test


class ImageData(L.LightningDataModule):
    """Stage 1: padded images in [-1, 1]."""

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self) -> None:  # downloads and caches, once
        prepare_data(self.cfg.dataset)

    def setup(self, stage: str | None = None) -> None:
        self.train_ds, self.val_ds, self.test_ds = make_splits(self.cfg.dataset, self.cfg.val_size, self.cfg.seed)

    def _loader(self, ds: Dataset, train: bool) -> DataLoader:
        n = self.cfg.num_workers
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=train, drop_last=train, num_workers=n,
                          persistent_workers=n > 0 and train, pin_memory=True)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, False)


@torch.no_grad()
def encode(ae: Autoencoder, ds: Dataset, device: torch.device, num_workers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A whole dataset as (latents, labels) on the CPU."""
    zs, ys = [], []
    for x, y in DataLoader(ds, batch_size=1024, num_workers=num_workers):
        zs.append(ae.encode(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(zs), torch.cat(ys)


class LatentData(L.LightningDataModule):
    """Stage 2: the frozen AE's latents, encoded once and standardized per channel with the train split's statistics.

    The splits are the AE's own (its dataset, `val_size` and `seed`), so the flow never trains on an image the AE
    validated on. `mean` and `std` are (1, C, 1, 1); the flow module copies them into its buffers."""

    def __init__(self, cfg: FlowConfig, ae: Autoencoder, ae_cfg: AEConfig):
        super().__init__()
        self.cfg, self.ae, self.ae_cfg = cfg, ae, ae_cfg
        self.mean = self.std = None

    def setup(self, stage: str | None = None) -> None:
        if self.mean is not None:  # already encoded (fit, then test)
            return
        device = get_device()
        splits = make_splits(self.ae_cfg.dataset, self.ae_cfg.val_size, self.ae_cfg.seed)
        encoded = [encode(self.ae.to(device), ds, device, self.cfg.num_workers) for ds in splits]
        self.ae.cpu()
        z_train = encoded[0][0]
        self.mean = z_train.mean(dim=(0, 2, 3), keepdim=True)
        self.std = z_train.std(dim=(0, 2, 3), keepdim=True)
        self.train_ds, self.val_ds, self.test_ds = (TensorDataset((z - self.mean) / self.std, y) for z, y in encoded)
        print(f"Latents: train {tuple(z_train.shape)}, val {tuple(encoded[1][0].shape)}, test {tuple(encoded[2][0].shape)}")

    def _loader(self, ds: TensorDataset, train: bool) -> DataLoader:
        # The tensors are in memory: index whole batches at once instead of collating one sample at a time.
        sampler = BatchSampler(RandomSampler(ds) if train else SequentialSampler(ds), self.cfg.batch_size, drop_last=train)
        return DataLoader(ds, batch_size=None, sampler=sampler)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, False)

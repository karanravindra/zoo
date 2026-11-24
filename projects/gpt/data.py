"""GPT data: token-shard corpora from `prepare`, served as fixed-length windows."""

from __future__ import annotations

import json
from pathlib import Path

import gigatoken as gt
import lightning as L
import numpy as np
import torch
from config import VAL_ALIGN, Config
from torch.utils.data import DataLoader, Dataset, Subset

from zoo.data.text import Shards
from zoo.paths import DATA_ROOT

EOT = "<|endoftext|>"  # a document ends with it as a target, and it starts the next one as an input
IM_START = "<|im_start|>"  # chat markers, reserved now so the base model's vocabulary already has them for SFT
IM_END = "<|im_end|>"
SPECIALS = [EOT, IM_START, IM_END]  # ids 256, 257, 258


def corpus_dir(dataset: str) -> Path:
    return DATA_ROOT / "gpt" / dataset


def load_meta(dataset: str, split: str) -> dict:
    path = corpus_dir(dataset) / split / "meta.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: run `uv run projects/gpt/cli.py prepare --dataset {dataset}` first")
    return json.loads(path.read_text())


def load_tokenizer(dataset: str) -> gt.Tokenizer:
    return gt.Tokenizer(corpus_dir(dataset) / "tokenizer.json")


class Windows(Dataset):
    """Consecutive windows of `block_size + 1` tokens (adjacent windows share one token, so every token is a target once).

    Item `i` is the int64 tensor `tokens[i * block_size : (i + 1) * block_size + 1]`; the module splits it into inputs
    and targets. Shuffling is the sampler's job. The shards are opened lazily, in each DataLoader worker."""

    def __init__(self, split_dir: Path, block_size: int) -> None:
        self.dir, self.block_size = split_dir, block_size
        self._tokens: Shards | None = None
        self.n = (len(self.tokens) - 1) // block_size
        if self.n < 1:
            raise ValueError(f"{split_dir}: too few tokens for block_size {block_size}")

    @property
    def tokens(self) -> Shards:
        if self._tokens is None:
            self._tokens = Shards(self.dir)
        return self._tokens

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        start = i * self.block_size
        return torch.from_numpy(self.tokens[start : start + self.block_size + 1].astype(np.int64))


def val_window_ids(n_windows: int, block_size: int, val_tokens: int) -> list[int]:
    """The windows of the validation split that `val/loss` is measured on: evenly spaced, the same in every run.

    The count depends on `val_tokens` only, and is a multiple of VAL_ALIGN so that no micro-batch size leaves a short
    last batch (a new shape to compile)."""
    n = min(n_windows, max(VAL_ALIGN, val_tokens // block_size // VAL_ALIGN * VAL_ALIGN))
    return np.linspace(0, n_windows - 1, n).astype(int).tolist()


class GPTData(L.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self) -> None:
        load_meta(self.cfg.dataset, "train")  # fails with the command to run if `prepare` has not been
        load_meta(self.cfg.dataset, "validation")

    def setup(self, stage: str | None = None) -> None:
        root, bs = corpus_dir(self.cfg.dataset), self.cfg.block_size
        self.train_ds = Windows(root / "train", bs)
        val = Windows(root / "validation", bs)
        self.val_ds = Subset(val, val_window_ids(len(val), bs, self.cfg.val_tokens))

    def train_dataloader(self) -> DataLoader:
        n = self.cfg.num_workers
        return DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True, num_workers=n, persistent_workers=n > 0, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        n = min(2, self.cfg.num_workers)
        return DataLoader(self.val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=n, persistent_workers=n > 0, pin_memory=True)

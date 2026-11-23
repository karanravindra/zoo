"""Token corpora cached on disk as shards, mirroring what `ImageDataset` does for images (tokenizer-agnostic).

`write_token_corpus` streams token-id sequences into `<cache_dir>/<split>/shard_0000.npy, shard_0001.npy, ...` (uint16
if the vocab fits, else uint32) plus `meta.json`, holding at most about `shard_tokens` tokens in RAM. A shard closes
at a sequence boundary once it has `shard_tokens` tokens, so keep sequences small (a document, or a few thousand of
them) for even shards. `Shards` is the read side: the shards as one logical, memory-mapped token array. Caches
written before sharding (a single `tokens.npy`) still load. `TokenDataset` returns random next-token windows.
Sequences are simply concatenated, so put any end-of-text id in the sequences yourself. If every document ends with
such a delimiter id, `TokenDataset.doc_batch` returns windows that start at a document.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from itertools import accumulate
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SHARD_TOKENS = 1 << 28  # about 268M tokens: 512 MB as uint16


def write_token_corpus(
    sequences: Iterable[Sequence[int]],
    cache_dir: str | Path,
    split: str,
    vocab_size: int,
    extra_meta: dict | Callable[[], dict] | None = None,
    shard_tokens: int = SHARD_TOKENS,
) -> Path:
    """Write `sequences` (iterable of token-id arrays/lists) to `<cache_dir>/<split>/shard_NNNN.npy` and `meta.json`.

    `extra_meta` is merged into meta.json (e.g. the UTF-8 byte count of the raw text). Pass a callable to compute
    it after the stream is consumed, since the iterable is only read inside this function. meta.json also lists
    the shards (`shards`: file and length of each), the total `length` and `num_items`. Whatever the split held
    before (shards or a legacy tokens.npy) is removed first."""
    out = Path(cache_dir) / split
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("shard_*.npy"), out / "tokens.npy", *out.glob("delim_*.npy")]:
        old.unlink(missing_ok=True)
    dtype = np.dtype(np.uint16 if vocab_size <= 2**16 else np.uint32)
    shards: list[dict] = []
    buf: list[np.ndarray] = []
    buffered, n_items = 0, 0

    def flush() -> None:
        nonlocal buffered
        if not buf:
            return
        name = f"shard_{len(shards):04d}.npy"
        tmp = out / (name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, np.concatenate(buf).astype(dtype))
        os.replace(tmp, out / name)
        shards.append({"file": name, "length": buffered})
        buf.clear()
        buffered = 0

    with tqdm(desc=f"{split}: write", unit="tok", unit_scale=True) as bar:
        for seq in sequences:
            a = np.asarray(seq, dtype=np.int64)
            if len(a) and (a.min() < 0 or a.max() >= vocab_size):
                raise ValueError(f"token id outside [0, {vocab_size})")
            buf.append(a)
            buffered += len(a)
            n_items += 1
            bar.update(len(a))
            if buffered >= shard_tokens:
                flush()
        flush()
    extra = extra_meta() if callable(extra_meta) else (extra_meta or {})
    meta = {
        "length": sum(s["length"] for s in shards),
        "num_items": n_items,
        "vocab_size": vocab_size,
        "dtype": dtype.name,
        "shards": shards,
        **extra,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


class Shards:
    """The shards of one split as a single read-only token array: `len`, and `[i]` / `[i:j]` by global position.

    A slice that crosses a shard boundary is stitched together, so shards behave as if concatenated. Each shard is
    memory-mapped, so nothing is read until it is indexed. Opens a legacy single `tokens.npy` as one shard."""

    def __init__(self, split_dir: str | Path) -> None:
        self.dir = Path(split_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        files = [s["file"] for s in meta["shards"]] if "shards" in meta else ["tokens.npy"]
        self.arrays = [np.load(self.dir / f, mmap_mode="r") for f in files]
        self.starts = [0, *accumulate(len(a) for a in self.arrays)]  # global start of each shard, then the total
        self.dtype = self.arrays[0].dtype if self.arrays else np.dtype(np.uint16)

    def __getstate__(self) -> dict:  # DataLoader workers re-open the memmaps instead of pickling their contents
        return {"dir": self.dir}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["dir"])

    def __len__(self) -> int:
        return self.starts[-1]

    def __getitem__(self, key: int | slice) -> np.ndarray | np.integer:
        n = len(self)
        if isinstance(key, int):
            if not -n <= key < n:
                raise IndexError(key)
            key %= n
            s = bisect_right(self.starts, key, 0, len(self.arrays)) - 1
            return self.arrays[s][key - self.starts[s]]
        start, stop, step = key.indices(n)
        if step != 1:
            raise ValueError("Shards only supports contiguous slices")
        if stop <= start:
            return np.empty(0, self.dtype)
        first = bisect_right(self.starts, start, 0, len(self.arrays)) - 1
        parts = []
        for s in range(first, len(self.arrays)):
            lo, hi = self.starts[s], self.starts[s + 1]
            if lo >= stop:
                break
            parts.append(self.arrays[s][max(start, lo) - lo : min(stop, hi) - lo])
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


class TokenDataset:
    """Random `(x, y)` next-token windows from a corpus written by `write_token_corpus` (sharded or not).

    `batch(batch_size)` returns int64 tensors of shape (batch_size, block_size) on `device`, with y = x shifted by
    one. Window starts come from a generator seeded with `seed` (`reset` restarts it), so val batches repeat."""

    def __init__(self, cache_dir: str | Path, split: str, block_size: int, device="cpu", seed: int = 0) -> None:
        d = Path(cache_dir) / split
        self.dir, self._delims = d, {}
        self.meta = json.loads((d / "meta.json").read_text())
        self.tokens = Shards(d)
        if len(self.tokens) <= block_size:
            raise ValueError(f"{split}: {len(self.tokens)} tokens is too few for block_size {block_size}")
        self.block_size, self.device, self.seed = block_size, device, seed
        self.reset()

    def __len__(self) -> int:
        return len(self.tokens)

    def reset(self, seed: int | None = None) -> None:
        self.gen = torch.Generator().manual_seed(self.seed if seed is None else seed)

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(len(self.tokens) - self.block_size, (batch_size,), generator=self.gen).tolist()
        w = np.stack([self.tokens[i : i + self.block_size + 1] for i in starts]).astype(np.int64)
        return self._to_device(torch.from_numpy(w[:, :-1]), torch.from_numpy(w[:, 1:]))

    def delimiters(self, delim_id: int) -> np.ndarray:
        """Sorted positions of `delim_id` in the corpus (found once, then cached next to the shards)."""
        if delim_id not in self._delims:
            path = self.dir / f"delim_{delim_id}.npy"
            if not path.exists():
                chunk = 1 << 24
                found = [np.flatnonzero(self.tokens[i : i + chunk] == delim_id) + i for i in range(0, len(self.tokens), chunk)]
                tmp = path.with_suffix(".tmp.npy")
                np.save(tmp, np.concatenate(found).astype(np.int64))
                os.replace(tmp, path)
            self._delims[delim_id] = np.load(path)
        return self._delims[delim_id]

    def doc_batch(self, batch_size: int, delim_id: int, pad: bool, ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
        """`(x, y)` windows that start at a document, for a corpus where every document ends with `delim_id`.

        Each row starts on a delimiter, which then acts as the next document's start token: x[0] = delimiter,
        y[0] = the document's first token. Documents are drawn uniformly (the corpus' first one, which has no
        delimiter before it, is never drawn).

        - `pad=False` (packed): the window simply continues through the following documents, so there is no
          padding and only the last document of a row can be cut short.
        - `pad=True`: one document per row, cut at `block_size`. After the document's closing delimiter, x is
          filled with `delim_id` and y with `ignore_index`, so the loss skips those positions."""
        d = self.delimiters(delim_id)
        n = len(d) - 1 if pad else int(np.searchsorted(d, len(self.tokens) - self.block_size - 1, side="right"))
        if n < 1:
            raise ValueError("corpus has too few delimiters for document-aligned batches")
        j = torch.randint(n, (batch_size,), generator=self.gen).numpy()
        starts = d[j]
        w = np.full((batch_size, self.block_size + 1), delim_id, dtype=np.int64)
        for r, i in enumerate(starts.tolist()):  # a padded row near the end of the corpus can be shorter than the window
            row = self.tokens[i : i + self.block_size + 1]
            w[r, : len(row)] = row
        x, y = w[:, :-1].copy(), w[:, 1:].copy()
        if pad:
            valid = np.arange(self.block_size)[None, :] < (d[j + 1] - starts)[:, None]  # up to the closing delimiter
            x[~valid], y[~valid] = delim_id, ignore_index
        return self._to_device(torch.from_numpy(x), torch.from_numpy(y))

    def _to_device(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.device(self.device).type == "cuda":
            x, y = x.pin_memory(), y.pin_memory()
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

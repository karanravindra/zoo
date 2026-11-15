"""Data for Angais/pixel-art-atlas-1.4m: a one-off cache builder, a word-level tokenizer, and loaders.

Build the cache once (downloads ~435 MB of parquet; pyarrow is not a zoo dependency, so pull it in):
    uv run --with pyarrow projects/pixel_art/pixel_data.py [--max-len 24] [--min-count 5] [--force]

Writes <repo>/data/cache/PixelArtAtlas/:
  vocab.json                 tokenizer (vocab built from the train captions only)
  <split>/images.npy         (N, 4, 16, 16) uint8 RGBA; RGB is zeroed wherever alpha == 0
  <split>/tokens.npy         (N, max_len) int32 token ids, 0-padded
  validation/captions.json   raw captions of the official validation split (used for previews)
`<split>` is `train` and `validation`. Everything is read back into memory by `load_data`
(1.4M sprites are only ~1.4 GB as uint8).
"""

from __future__ import annotations

import io
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import tyro

from zoo.paths import DATA_ROOT

REPO_ID = "Angais/pixel-art-atlas-1.4m"
CACHE_ROOT = DATA_ROOT / "cache" / "PixelArtAtlas"

IMAGE_SIZE = 16
CHANNELS = 4  # RGBA

PAD_ID, NULL_ID, UNK_ID = 0, 1, 2  # NULL is the "empty caption" used for classifier-free guidance
SPECIALS = ["<pad>", "<null>", "<unk>"]
WORD = re.compile(r"[a-z0-9']+")

# --- tokenizer -------------------------------------------------------------------


class Tokenizer:
    """Lowercase word-level tokenizer; unknown words become <unk>, captions are cut to `max_len`."""

    def __init__(self, vocab: list[str], max_len: int):
        self.vocab, self.max_len = vocab, max_len
        self.index = {w: i for i, w in enumerate(vocab)}

    def __len__(self) -> int:
        return len(self.vocab)

    @classmethod
    def build(cls, captions: list[str], max_len: int, min_count: int) -> Tokenizer:
        counts = Counter(w for c in captions for w in WORD.findall(c.lower()))
        words = sorted((w for w, n in counts.items() if n >= min_count), key=lambda w: (-counts[w], w))
        return cls(SPECIALS + words, max_len)

    def encode(self, captions: list[str]) -> np.ndarray:
        """(N, max_len) int32, 0-padded. An empty caption becomes a single <unk> so no row is all padding."""
        out = np.zeros((len(captions), self.max_len), np.int32)
        for i, c in enumerate(captions):
            ids = [self.index.get(w, UNK_ID) for w in WORD.findall(c.lower())][: self.max_len] or [UNK_ID]
            out[i, : len(ids)] = ids
        return out

    def to_dict(self) -> dict:
        return {"vocab": self.vocab, "max_len": self.max_len}

    @classmethod
    def from_dict(cls, d: dict) -> Tokenizer:
        return cls(d["vocab"], d["max_len"])


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_dict(json.loads(_cache_file("vocab.json").read_text()))


# --- images ----------------------------------------------------------------------


def to_pm1(x: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [-1, 1]."""
    return x.float() / 127.5 - 1


def _cache_file(*parts: str) -> Path:
    p = CACHE_ROOT.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; build the cache first: uv run --with pyarrow projects/pixel_art/pixel_data.py")
    return p


@dataclass
class PixelData:
    """Train images/tokens with a deterministic train/val index split, plus the official validation split as test."""

    images: torch.Tensor  # (N, 4, 16, 16) uint8, the whole train split
    tokens: torch.Tensor  # (N, L) int32
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_images: torch.Tensor
    test_tokens: torch.Tensor
    test_captions: list[str]

    def to(self, device) -> PixelData:
        return PixelData(
            self.images.to(device), self.tokens.to(device), self.train_idx.to(device), self.val_idx.to(device),
            self.test_images.to(device), self.test_tokens.to(device), self.test_captions,
        )


def load_test_captions() -> list[str]:
    return json.loads(_cache_file("validation", "captions.json").read_text())


def load_data(val_size: int, seed: int) -> PixelData:
    """`val_size` train sprites are held out as val (same split for a given seed, so the denoiser never trains on AE-val)."""
    images = torch.from_numpy(np.load(_cache_file("train", "images.npy")))
    tokens = torch.from_numpy(np.load(_cache_file("train", "tokens.npy")))
    perm = torch.randperm(len(images), generator=torch.Generator().manual_seed(seed))
    return PixelData(
        images, tokens, perm[val_size:], perm[:val_size],
        torch.from_numpy(np.load(_cache_file("validation", "images.npy"))),
        torch.from_numpy(np.load(_cache_file("validation", "tokens.npy"))),
        load_test_captions(),
    )


# --- cache builder ----------------------------------------------------------------


def _decode_shard(path: str) -> tuple[np.ndarray, list[str]]:
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path, columns=["image", "caption"])
    imgs = np.zeros((table.num_rows, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), np.uint8)
    for i, d in enumerate(table.column("image").to_pylist()):
        a = np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGBA"))
        if a.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"{path} row {i}: expected 16x16, got {a.shape}")
        imgs[i] = a
    imgs[imgs[..., 3] == 0] = 0  # hidden RGB under fully transparent pixels is noise
    return imgs.transpose(0, 3, 1, 2), table.column("caption").to_pylist()


def prepare(max_len: int = 24, min_count: int = 5, workers: int = 4, force: bool = False) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from numpy.lib.format import open_memmap

    if (CACHE_ROOT / "vocab.json").exists() and not force:
        print(f"{CACHE_ROOT} already built (use --force to rebuild)")
        return
    root = Path(snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=["*.parquet"]))
    files = {s: sorted(map(str, root.rglob(f"{s}-*.parquet"))) for s in ("train", "validation")}
    for s, f in files.items():
        if not f:
            raise FileNotFoundError(f"no {s}-*.parquet under {root}")
        print(f"{s}: {len(f)} shards")

    captions: dict[str, list[str]] = {}
    for split, fs in files.items():
        out = CACHE_ROOT / split
        out.mkdir(parents=True, exist_ok=True)
        n = sum(pq.ParquetFile(f).metadata.num_rows for f in fs)
        tmp = out / "images.tmp.npy"
        images = open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        caps: list[str] = []
        i = 0
        with Pool(min(workers, len(fs))) as pool:
            for imgs, c in pool.imap(_decode_shard, fs):  # ordered, so rows keep the dataset order
                images[i : i + len(imgs)] = imgs
                caps += c
                i += len(imgs)
                print(f"  {split}: {i}/{n}")
        assert i == n
        images.flush()
        del images
        os.replace(tmp, out / "images.npy")
        captions[split] = caps

    tok = Tokenizer.build(captions["train"], max_len, min_count)
    for split, caps in captions.items():
        np.save(CACHE_ROOT / split / "tokens.npy", tok.encode(caps))
    (CACHE_ROOT / "validation" / "captions.json").write_text(json.dumps(captions["validation"]))
    lens = np.array([len(WORD.findall(c.lower())) for c in captions["train"]])
    print(f"vocab: {len(tok)} tokens (min_count={min_count}); {np.mean(lens > max_len):.4%} of train captions exceed max_len={max_len}")
    (CACHE_ROOT / "vocab.json").write_text(json.dumps(tok.to_dict()))  # written last: marks the cache complete
    print(f"Wrote {CACHE_ROOT}")


@dataclass
class Args:
    max_len: int = 24  # tokens per caption (longer ones are cut)
    min_count: int = 5  # words rarer than this in train become <unk>
    workers: int = 4
    force: bool = False


if __name__ == "__main__":
    a = tyro.cli(Args, description=__doc__)
    prepare(a.max_len, a.min_count, a.workers, a.force)

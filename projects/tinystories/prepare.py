"""One-off data prep for TinyStories: train a BPE tokenizer and write token streams.

Run once, from anywhere: `uv run projects/tinystories/prepare.py`

Downloads `roneneldan/TinyStories` (text column `text`), then writes to `<repo>/data/cache/tinystories/`:
  tokenizer.json      byte-level BPE (GPT-2 style) trained with gigatoken on the first `vocab_docs` train stories;
                      id 256 is the end-of-text token, loaded by train.py / sample.py with `gigatoken.Tokenizer`
  <split>/tokens.npy  every story's token ids followed by one end-of-text id, concatenated (uint16)
  <split>/meta.json   token count, UTF-8 byte count of the raw story text, story count, vocab size
`<split>` is `train` and `validation`.

Bits per byte needs tokens per byte. The token count includes the end-of-text tokens (the loss is averaged over them
too, so tokens and loss cover the same stream); the byte count is the story text only, so the delimiters' cost is
charged to the text, as if the boundaries had to be encoded.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import gigatoken as gt
import numpy as np
import tyro
from datasets import load_dataset

from zoo.data.text import write_token_corpus
from zoo.paths import DATA_ROOT

CACHE_ROOT = DATA_ROOT / "cache" / "tinystories"
EOT = "<|endoftext|>"


@dataclass
class PrepareConfig:
    dataset: str = "roneneldan/TinyStories"  # Hub dataset with a `text` column and train / validation splits
    vocab_size: int = 4096  # BPE vocabulary size, end-of-text token included
    vocab_docs: int = 300_000  # train stories used to learn the BPE merges (all stories if larger)
    shard_docs: int = 200_000  # stories per temporary text shard (bounds RAM while encoding)
    force: bool = False  # rebuild even if the cache exists


def byte_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable unicode char map, as used by byte-level BPE tokenizer.json files."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, extra = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, map(chr, chars)))


def stories(split) -> Iterator[str]:
    for batch in split.iter(batch_size=10_000):
        yield from batch["text"]


def to_tokenizer_json(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> dict:
    """tokenizer.json (byte-level BPE, GPT-2 pre-tokenization) from `gigatoken.train_bpe`'s vocab and merges."""
    m = byte_unicode()
    enc = lambda b: "".join(m[x] for x in b) if b != EOT.encode() else EOT  # noqa: E731
    eot_id = next(i for i, b in vocab.items() if b == EOT.encode())
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": eot_id, "content": EOT, "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False, "special": True}
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
            "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": {enc(b): i for i, b in vocab.items()},
            "merges": [f"{enc(a)} {enc(b)}" for a, b in merges],
        },
    }


def write_shards(texts, out_dir: Path, shard_docs: int, limit: int | None = None) -> tuple[list[Path], int, int]:
    """Stream stories into `<out_dir>/shard_NNNN.txt` files joined by the end-of-text string.

    Returns (paths, number of stories, UTF-8 bytes of story text). Empty stories are dropped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, n_docs, n_bytes, buf = [], 0, 0, []

    def flush():
        if buf:
            paths.append(out_dir / f"shard_{len(paths):04d}.txt")
            paths[-1].write_text(EOT.join(buf), encoding="utf-8")
            buf.clear()

    for text in texts:
        if not text.strip():
            continue
        buf.append(text)
        n_docs, n_bytes = n_docs + 1, n_bytes + len(text.encode("utf-8"))
        if len(buf) == shard_docs:
            flush()
        if limit is not None and n_docs >= limit:
            break
    flush()
    return paths, n_docs, n_bytes


def token_chunks(tok: gt.Tokenizer, shards: list[Path], eot_id: int) -> Iterator[np.ndarray]:
    """Per shard: gigatoken reads the file in Rust, splits on the separator and encodes every story in parallel."""
    for path in shards:
        docs = tok.encode_files(gt.TextFileSource([path], separator=EOT.encode()))
        flat = ak.to_numpy(ak.flatten(docs)).astype(np.int64)
        ends = np.cumsum(ak.to_numpy(ak.num(docs)))
        yield np.insert(flat, ends, eot_id)  # one end-of-text after every story


def main(cfg: PrepareConfig):
    if (CACHE_ROOT / "tokenizer.json").exists() and not cfg.force:
        print(f"{CACHE_ROOT} already built (use --force to rebuild)")
        return
    ds = load_dataset(cfg.dataset)
    tmp = CACHE_ROOT / "tmp_shards"

    # 1) tokenizer: gigatoken trains the BPE merges from a text file of stories
    vocab_shards, n, _ = write_shards(stories(ds["train"]), tmp / "vocab", cfg.vocab_docs, limit=cfg.vocab_docs)
    print(f"training a {cfg.vocab_size}-token BPE on {n:,} stories")
    vocab, merges = gt.train_bpe(gt.TextFileSource(vocab_shards, separator=EOT.encode()), cfg.vocab_size, [EOT])
    (CACHE_ROOT / "tokenizer.json").write_text(json.dumps(to_tokenizer_json(vocab, merges), ensure_ascii=False))
    tok = gt.Tokenizer(CACHE_ROOT / "tokenizer.json")
    eot_id = int(tok.encode(EOT)[0])
    assert tok.vocab_size == cfg.vocab_size and eot_id == next(i for i, b in vocab.items() if b == EOT.encode())
    shutil.rmtree(tmp / "vocab")

    # 2) token streams, shard by shard
    for split in ("train", "validation"):
        shards, n_docs, n_bytes = write_shards(stories(ds[split]), tmp / split, cfg.shard_docs)
        write_token_corpus(
            token_chunks(tok, shards, eot_id), CACHE_ROOT, split, cfg.vocab_size,
            extra_meta=lambda: {"num_docs": n_docs, "num_bytes": n_bytes, "eot_id": eot_id, "tokens_include_eot": True},
        )
        meta = json.loads((CACHE_ROOT / split / "meta.json").read_text())
        print(f"{split}: {n_docs:,} stories, {meta['length']:,} tokens, {n_bytes:,} bytes "
              f"({meta['length'] / n_bytes:.3f} tokens/byte)")
        shutil.rmtree(tmp / split)
    tmp.rmdir()
    print(f"Wrote {CACHE_ROOT}")


if __name__ == "__main__":
    main(tyro.cli(PrepareConfig, description=__doc__))

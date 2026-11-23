"""Corpus preparation: text shards -> BPE tokenizer (gigatoken) -> token shards (`zoo.data.text`).

Everything is streamed, so a corpus as large as FineWeb-Edu never sits in RAM: documents go to temporary text shards of
`shard_mb` megabytes, the tokenizer is trained on the first `vocab_docs` of them, and every text shard is then
encoded (gigatoken reads the file and tokenizes the documents in parallel, in Rust) and appended to the token
shards of the split (`write_token_corpus`, `shard_tokens` tokens each). The corpus lives in `<DATA_ROOT>/gpt/<dataset>/`:
  tokenizer.json         byte-level BPE; ids 256, 257, 258 are <|endoftext|>, <|im_start|>, <|im_end|>
  <split>/shard_NNNN.npy token ids, every document followed by one end-of-text id (uint16, or uint32 for a vocab > 65536)
  <split>/meta.json      shards, token count, UTF-8 byte count of the raw text, document count, vocab, special-token ids
`<split>` is `train` and `validation`.

Bits per byte needs tokens per byte. The token count includes the end-of-text tokens (the loss is averaged over them
too, so tokens and loss cover the same stream); the byte count is the document text only, so the delimiters' cost is
charged to the text, as if the boundaries had to be encoded.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import awkward as ak
import gigatoken as gt
import numpy as np
from config import PrepareConfig
from data import EOT, SPECIALS, corpus_dir
from datasets import load_dataset

from zoo.data.text import write_token_corpus


@dataclass(frozen=True)
class Source:
    hub: str  # Hub dataset id
    name: str | None = None  # its config name
    text: str = "text"  # column with the document text
    val_split: str | None = "validation"  # None: hold the first `val_docs` of the train split out instead
    val_docs: int = 0
    streaming: bool = False  # stream from the Hub instead of caching the whole dataset first
    vocab_size: int = 4096  # default tokenizer size


SOURCES: dict[str, Source] = {
    "tinystories": Source("roneneldan/TinyStories", vocab_size=4096),
    "fineweb-edu": Source("HuggingFaceFW/fineweb-edu", name="sample-10BT", val_split=None, val_docs=10_000, streaming=True, vocab_size=32_768),
}


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


def to_tokenizer_json(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> dict:
    """tokenizer.json (byte-level BPE, GPT-2 pre-tokenization) from `gigatoken.train_bpe`'s vocab and merges."""
    m = byte_unicode()
    specials = {s.encode(): s for s in SPECIALS}
    enc = lambda b: specials[b] if b in specials else "".join(m[x] for x in b)  # noqa: E731
    added = [
        {"id": i, "content": specials[b], "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        for i, b in sorted(vocab.items())
        if b in specials
    ]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": {enc(b): i for i, b in vocab.items()},
            "merges": [f"{enc(a)} {enc(b)}" for a, b in merges],
        },
    }  # fmt: skip


def documents(source: Source, split: str) -> Iterator[str]:
    ds = load_dataset(source.hub, source.name, split=split, streaming=source.streaming)
    for batch in ds.iter(batch_size=10_000):
        yield from batch[source.text]


class TextShards:
    """Temporary text files `shard_NNNN.txt` in `dir`, documents joined by the end-of-text string."""

    def __init__(self, dir: Path, shard_bytes: int) -> None:
        self.dir, self.shard_bytes = dir, shard_bytes
        self.paths: list[Path] = []
        self.docs = self.bytes = 0  # documents kept and their UTF-8 bytes
        dir.mkdir(parents=True, exist_ok=True)

    def add(self, texts: Iterable[str]) -> None:
        """Append documents (empty ones dropped, special-token strings removed from the text), starting a new file
        whenever the current one reaches `shard_bytes`. A partial shard is closed at the end of the iterable."""
        buf: list[str] = []
        size = 0

        def flush() -> None:
            nonlocal size
            if buf:
                self.paths.append(self.dir / f"shard_{len(self.paths):04d}.txt")
                self.paths[-1].write_text(EOT.join(buf), encoding="utf-8")
                buf.clear()
                size = 0

        for text in texts:
            for s in SPECIALS:
                text = text.replace(s, "")
            if not text.strip():
                continue
            n = len(text.encode("utf-8"))
            buf.append(text)
            size += n
            self.docs, self.bytes = self.docs + 1, self.bytes + n
            if size >= self.shard_bytes:
                flush()
        flush()


def token_chunks(tok: gt.Tokenizer, shards: list[Path], eot_id: int) -> Iterator[np.ndarray]:
    """Per text shard: gigatoken reads the file, splits on the separator and encodes every document in parallel."""
    for path in shards:
        docs = tok.encode_files(gt.TextFileSource([path], separator=EOT.encode()))
        flat = ak.to_numpy(ak.flatten(docs)).astype(np.int64)
        ends = np.cumsum(ak.to_numpy(ak.num(docs)))
        yield np.insert(flat, ends, eot_id)  # one end-of-text after every document


def build_corpus(
    train: Iterable[str],
    val: Iterable[str] | None,
    out: Path,
    *,
    vocab_size: int,
    vocab_docs: int,
    val_docs: int = 0,
    max_docs: int = 0,
    shard_bytes: int = 256 << 20,
    shard_tokens: int = 1 << 28,
) -> None:
    """The whole pipeline over document streams. `val=None` holds the first `val_docs` documents of `train` out as the
    validation split. `train` is read once: its first `vocab_docs` documents also train the tokenizer."""
    tmp = out / "tmp_shards"
    shutil.rmtree(tmp, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    train = iter(train)
    splits = {"train": TextShards(tmp / "train", shard_bytes), "validation": TextShards(tmp / "validation", shard_bytes)}
    if val is None:
        splits["validation"].add(islice(train, val_docs))
    else:
        splits["validation"].add(val)
    if max_docs:
        train = islice(train, max_docs)

    # 1) the tokenizer, from the first training documents (those shards are real training data, kept)
    splits["train"].add(islice(train, vocab_docs))
    print(f"training a {vocab_size}-token BPE on {splits['train'].docs:,} documents")
    vocab, merges = gt.train_bpe(gt.TextFileSource(splits["train"].paths, separator=EOT.encode()), vocab_size, SPECIALS)
    (out / "tokenizer.json").write_text(json.dumps(to_tokenizer_json(vocab, merges), ensure_ascii=False))
    tok = gt.Tokenizer(out / "tokenizer.json")
    special_ids = {s: next(i for i, b in vocab.items() if b == s.encode()) for s in SPECIALS}
    assert tok.vocab_size == vocab_size and all(int(tok.encode(s)[0]) == i for s, i in special_ids.items())
    eot_id = special_ids[EOT]

    # 2) the rest of the training stream, then every split's token shards
    splits["train"].add(train)
    for split, text in splits.items():
        write_token_corpus(
            token_chunks(tok, text.paths, eot_id), out, split, vocab_size, shard_tokens=shard_tokens,
            extra_meta=lambda text=text: {"num_docs": text.docs, "num_bytes": text.bytes, "eot_id": eot_id, "special_tokens": special_ids, "tokens_include_eot": True},
        )  # fmt: skip
        meta = json.loads((out / split / "meta.json").read_text())
        print(f"{split}: {text.docs:,} documents, {meta['length']:,} tokens in {len(meta['shards'])} shards, {text.bytes:,} bytes ({meta['length'] / max(1, text.bytes):.3f} tokens/byte)")
    shutil.rmtree(tmp)
    print(f"Wrote {out}")


def prepare_corpus(cfg: PrepareConfig) -> None:
    source = SOURCES[cfg.dataset]
    out = corpus_dir(cfg.dataset)
    if (out / "tokenizer.json").exists() and not cfg.force:
        print(f"{out} already built (use --force to rebuild)")
        return
    build_corpus(
        documents(source, "train"),
        documents(source, source.val_split) if source.val_split else None,
        out,
        vocab_size=cfg.vocab_size or source.vocab_size,
        vocab_docs=cfg.vocab_docs,
        val_docs=source.val_docs,
        max_docs=cfg.max_docs,
        shard_bytes=cfg.shard_mb << 20,
        shard_tokens=cfg.shard_tokens,
    )

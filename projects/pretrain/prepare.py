"""One-off data prep: mix several story corpora, train one BPE tokenizer on them and write token streams.

Run once, from anywhere: `uv run projects/pretrain/prepare.py` (`--dry-run` first to see how big each source is).

Sources (`SOURCES`; pick with `--sources a b c`): TinyStories, SimpleStories, Cosmopedia's `stories` config,
WritingPrompts and Project Gutenberg. Each is one Hub dataset with a text column; a new source is one more entry.
Sources without a validation split (Cosmopedia, Gutenberg) hold out `val_docs` random rows; books are cut into
paragraph-aligned chunks of about 8,000 characters (`book_chunks`) so a "story" is never a whole book.

Everything goes to `<repo>/data/cache/pretrain/`:
  tokenizer.json         byte-level BPE (GPT-2 style) trained with gigatoken on `--vocab-mb` of text from every source
                         (equal bytes each, so the biggest source doesn't set the vocabulary); id 256 is the
                         end-of-text token, loaded by train.py / sample.py with `gigatoken.Tokenizer`
  parts/<source>/<split>/  one source's tokens, written once and reused: a crashed run resumes here, and adding a
                         source later only encodes that source (delete `parts/` to free the space once `train/` exists)
  train/                 every source's train stories (uint16 `tokens.npy`, `meta.json` with a per-source breakdown)
  validation/            held-out stories of `--val-source` only, so val bpb stays comparable with the TinyStories runs
  validation_<source>/   the other sources' held-out stories (train.py scores each at every eval)

Each split's meta.json has the token count, the UTF-8 byte count of the raw text, the story count and the vocab size.

Bits per byte needs tokens per byte. The token count includes the end-of-text tokens (the loss is averaged over them
too, so tokens and loss cover the same stream); the byte count is the story text only, so the delimiters' cost is
charged to the text, as if the boundaries had to be encoded.
"""

from __future__ import annotations

import json
import re
import shutil
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import gigatoken as gt
import numpy as np
import tyro
from datasets import load_dataset

from zoo.data.text import write_token_corpus
from zoo.paths import DATA_ROOT

CACHE_ROOT = DATA_ROOT / "cache" / "pretrain"
EOT = "<|endoftext|>"


# ---- cleaning: turn one row's text into one or more documents ------------------------------------------------------

_DETOKENIZE = [  # WritingPrompts is Penn-Treebank tokenized in places ("did n't", "`` Hi ''", "don ’ t", "word .")
    (re.compile(r"<newline>"), "\n"),
    (re.compile(r"`` *"), '"'),
    (re.compile(r" *''"), '"'),
    (re.compile(r"(\w) ’ (\w)"), r"\1’\2"),
    (re.compile(r" +(n't|'s|'re|'ve|'ll|'d|'m)\b"), r"\1"),
    (re.compile(r"([“‘„]) +"), r"\1"),
    (re.compile(r" +([”’.,;:!?])"), r"\1"),
]


def detokenize(text: str) -> Iterator[str]:
    for pattern, repl in _DETOKENIZE:
        text = pattern.sub(repl, text)
    yield text


def cut(paragraph: str, size: int) -> Iterator[str]:
    """Pieces of at most `size` characters, split after a sentence end (else at a space) when possible."""
    while len(paragraph) > size:
        i = paragraph.rfind(". ", 0, size)
        i = i + 1 if i > size // 2 else max(paragraph.rfind(" ", 0, size), size // 2)
        yield paragraph[:i].strip()
        paragraph = paragraph[i:].strip()
    yield paragraph


def book_chunks(text: str, size: int = 8_000, min_chars: int = 400) -> Iterator[str]:
    """A book as chunks of about `size` characters made of whole paragraphs (blank-line separated).

    Gutenberg text is hard-wrapped at ~70 columns, so lines inside a paragraph are joined with a space."""
    chunk, n = [], 0
    for par in re.split(r"\n\s*\n", text):
        par = " ".join(par.split())
        for piece in (cut(par, size) if par else ()):
            if chunk and n + len(piece) > size and n >= min_chars:  # a tiny chunk (a title) rides along with the next
                yield "\n\n".join(chunk)
                chunk, n = [], 0
            chunk.append(piece)
            n += len(piece) + 2
    if n >= min_chars:
        yield "\n\n".join(chunk)


def whole(text: str) -> Iterator[str]:
    yield text


@dataclass(frozen=True)
class Source:
    dataset: str  # Hub dataset id
    column: str  # text column
    config: str | None = None  # dataset config
    val_split: str | None = None  # validation split; None holds `val_docs` random rows out of the train split
    val_docs: int = 10_000  # rows held out when there is no validation split
    documents: Callable[[str], Iterable[str]] = whole  # row text -> documents (cleaning, chunking)


SOURCES: dict[str, Source] = {
    "tinystories": Source("roneneldan/TinyStories", "text", val_split="validation"),
    "simplestories": Source("SimpleStories/SimpleStories", "story", val_split="test"),
    "cosmopedia": Source("HuggingFaceTB/cosmopedia", "text", config="stories", val_docs=10_000),
    "writingprompts": Source("euclaise/writingprompts", "story", val_split="validation", documents=detokenize),
    "gutenberg": Source("common-pile/project_gutenberg", "text", val_docs=100, documents=book_chunks),
}


@dataclass
class PrepareConfig:
    sources: tuple[str, ...] = tuple(SOURCES)  # which sources to mix, from: tinystories simplestories cosmopedia writingprompts gutenberg
    val_source: str = "tinystories"  # source whose held-out stories become `validation` (the others get validation_<name>)
    vocab_size: int = 4096  # BPE vocabulary size, end-of-text token included (train.py's --vocab-size must match)
    vocab_mb: int = 64  # MB of text per source used to learn the BPE merges
    shard_mb: int = 256  # MB of text per temporary shard (bounds RAM while encoding)
    max_docs: int | None = None  # rows per source and split, for a quick trial of the whole pipeline (e.g. 2000)
    seed: int = 0  # picks held-out rows and the tokenizer's text sample
    dry_run: bool = False  # only count stories and bytes per source and estimate tokens; writes nothing
    tokens_per_byte: float = 0.25  # the dry run's token estimate (TinyStories measures 0.255 at vocab 4096)
    force: bool = False  # delete the cache and rebuild everything


# ---- reading a source ---------------------------------------------------------------------------------------------


class Corpus:
    """One source's train and validation documents. Loading downloads the dataset (and reads it from disk after)."""

    def __init__(self, name: str, src: Source, max_docs: int | None, seed: int):
        self.name, self.src = name, src
        ds = load_dataset(src.dataset, src.config)
        self.train = ds["train"].select_columns([src.column])
        self.n_train = len(self.train) if max_docs is None else min(len(self.train), max_docs)
        self.rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        if src.val_split:
            self.val = ds[src.val_split].select_columns([src.column])
            self.val_rows = np.arange(len(self.val) if max_docs is None else min(len(self.val), max_docs))
        else:  # hold out random rows (random, since a dataset can be sorted by topic or date)
            self.val = self.train
            k = min(src.val_docs, max(1, self.n_train // 20))
            self.val_rows = np.sort(self.rng.choice(self.n_train, k, replace=False))
        self.held_out = frozenset(self.val_rows.tolist()) if self.val is self.train else frozenset()

    def _docs(self, ds, skip: frozenset = frozenset(), limit: int | None = None) -> Iterator[str]:
        i = 0
        for batch in ds.iter(batch_size=1_000):
            for text in batch[self.src.column]:
                if limit is not None and i >= limit:
                    return
                if i not in skip and text.strip():
                    yield from (d for d in self.src.documents(text) if d.strip())
                i += 1

    def train_docs(self) -> Iterator[str]:
        return self._docs(self.train, self.held_out, self.n_train)

    def val_docs(self) -> Iterator[str]:
        return self._docs(self.val.select(self.val_rows))

    def sample(self, budget_bytes: int) -> Iterator[str]:
        """Documents of random train rows (no held-out ones) until `budget_bytes` of text, for the tokenizer."""
        pool = np.setdiff1d(np.arange(self.n_train), self.val_rows if self.held_out else [])
        probe = self.rng.choice(pool, min(200, len(pool)), replace=False)
        sizes = [len(t.encode("utf-8")) for t in self.train.select(np.sort(probe))[self.src.column]]
        k = min(len(pool), int(budget_bytes / max(1, np.mean(sizes))) + 1)
        used = 0
        for doc in self._docs(self.train.select(np.sort(self.rng.choice(pool, k, replace=False)))):
            yield doc
            used += len(doc.encode("utf-8"))
            if used >= budget_bytes:
                return


# ---- tokenizer ----------------------------------------------------------------------------------------------------


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


def train_tokenizer(corpora: Callable[[str], Corpus], cfg: PrepareConfig, tmp: Path) -> None:
    """Train the BPE on equal amounts of text from every source and write `tokenizer.json`."""
    files = []
    for name in cfg.sources:
        texts = list(corpora(name).sample(cfg.vocab_mb * 10**6))
        files.append(tmp / f"vocab_{name}.txt")
        files[-1].parent.mkdir(parents=True, exist_ok=True)
        files[-1].write_text(EOT.join(texts), encoding="utf-8")
        print(f"tokenizer sample from {name}: {len(texts):,} documents, {files[-1].stat().st_size / 1e6:.0f} MB")
    print(f"training a {cfg.vocab_size}-token BPE")
    vocab, merges = gt.train_bpe(gt.TextFileSource(files, separator=EOT.encode()), cfg.vocab_size, [EOT])
    (CACHE_ROOT / "tokenizer.json").write_text(json.dumps(to_tokenizer_json(vocab, merges), ensure_ascii=False))
    for f in files:
        f.unlink()


# ---- encoding -----------------------------------------------------------------------------------------------------


@dataclass
class Stats:
    docs: int = 0
    bytes: int = 0
    tokens: int = 0


def encode_docs(tok: gt.Tokenizer, docs: Iterable[str], tmp: Path, eot_id: int, shard_bytes: int, stats: Stats) -> Iterator[np.ndarray]:
    """Stream documents through temporary text shards of ~`shard_bytes`: gigatoken reads each file in Rust, splits on
    the separator and encodes every document in parallel. Yields token ids with an end-of-text after each document."""
    tmp.mkdir(parents=True, exist_ok=True)
    shard, size = [], 0

    def encode(texts: list[str]) -> np.ndarray:
        path = tmp / "shard.txt"
        path.write_text(EOT.join(texts), encoding="utf-8")
        encoded = tok.encode_files(gt.TextFileSource([path], separator=EOT.encode()))
        path.unlink()
        flat = ak.to_numpy(ak.flatten(encoded)).astype(np.int64)
        ends = np.cumsum(ak.to_numpy(ak.num(encoded)))
        out = np.insert(flat, ends, eot_id)  # one end-of-text after every document
        stats.tokens += len(out)
        return out

    for doc in docs:
        n = len(doc.encode("utf-8"))
        shard.append(doc)
        size += n
        stats.docs, stats.bytes = stats.docs + 1, stats.bytes + n
        if size >= shard_bytes:
            yield encode(shard)
            shard, size = [], 0
    if shard:
        yield encode(shard)


def write_part(tok: gt.Tokenizer, corpus: Corpus, split: str, cfg: PrepareConfig, eot_id: int) -> None:
    out = CACHE_ROOT / "parts" / corpus.name
    stats = Stats()
    docs = corpus.train_docs() if split == "train" else corpus.val_docs()
    chunks = encode_docs(tok, docs, CACHE_ROOT / "tmp", eot_id, cfg.shard_mb * 10**6, stats)
    write_token_corpus(
        chunks, out, split, cfg.vocab_size,
        extra_meta=lambda: {"num_docs": stats.docs, "num_bytes": stats.bytes, "eot_id": eot_id, "tokens_include_eot": True},
    )
    print(f"{corpus.name} {split}: {stats.docs:,} documents, {stats.tokens:,} tokens, {stats.bytes:,} bytes "
          f"({stats.tokens / max(1, stats.bytes):.3f} tokens/byte)")


def assemble(parts: dict[str, Path], split: str, out: str, vocab_size: int) -> None:
    """Concatenate the sources' `split` tokens into `<cache>/<out>/tokens.npy`, with one meta.json for the lot."""
    metas = {name: json.loads((p / split / "meta.json").read_text()) for name, p in parts.items()}

    def chunks() -> Iterator[np.ndarray]:
        for p in parts.values():
            tokens = np.load(p / split / "tokens.npy", mmap_mode="r")
            for i in range(0, len(tokens), 1 << 24):
                yield tokens[i : i + (1 << 24)]

    first = next(iter(metas.values()))
    extra = {
        "num_docs": sum(m["num_docs"] for m in metas.values()), "num_bytes": sum(m["num_bytes"] for m in metas.values()),
        "eot_id": first["eot_id"], "tokens_include_eot": True,
        "sources": {n: {"tokens": m["length"], "num_docs": m["num_docs"], "num_bytes": m["num_bytes"]} for n, m in metas.items()},
    }
    shutil.rmtree(CACHE_ROOT / out, ignore_errors=True)  # also drops a stale delim_<id>.npy cache
    write_token_corpus(chunks(), CACHE_ROOT, out, vocab_size, extra_meta=extra)
    meta = json.loads((CACHE_ROOT / out / "meta.json").read_text())
    print(f"{out}: {meta['num_docs']:,} documents, {meta['length']:,} tokens, {meta['num_bytes']:,} bytes "
          f"({meta['length'] / meta['num_bytes']:.3f} tokens/byte)")


# ---- main ---------------------------------------------------------------------------------------------------------


def main(cfg: PrepareConfig):
    unknown = [s for s in (*cfg.sources, cfg.val_source) if s not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s) {unknown}; choose from {list(SOURCES)}")
    if cfg.val_source not in cfg.sources:
        raise SystemExit(f"--val-source {cfg.val_source} is not in --sources {list(cfg.sources)}")

    loaded: dict[str, Corpus] = {}

    def corpus(name: str) -> Corpus:  # loaded on first use, so a resumed run doesn't touch finished sources
        if name not in loaded:
            loaded[name] = Corpus(name, SOURCES[name], cfg.max_docs, cfg.seed)
        return loaded[name]

    if cfg.dry_run:
        total = 0
        for name in cfg.sources:
            c = corpus(name)
            docs = nbytes = 0
            for doc in c.train_docs():
                docs, nbytes = docs + 1, nbytes + len(doc.encode("utf-8"))
            total += int(nbytes * cfg.tokens_per_byte)
            print(f"{name}: {docs:,} train documents, {nbytes / 1e9:.2f} GB of text, "
                  f"~{nbytes * cfg.tokens_per_byte / 1e9:.2f}B tokens at {cfg.tokens_per_byte} tokens/byte "
                  f"({len(c.val_rows):,} validation rows)")
        print(f"total: ~{total / 1e9:.2f}B tokens (a guess: tokens per byte depends on the tokenizer and the text)")
        return

    if cfg.force:
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not (CACHE_ROOT / "tokenizer.json").exists():
        train_tokenizer(corpus, cfg, CACHE_ROOT / "tmp")
    tok = gt.Tokenizer(CACHE_ROOT / "tokenizer.json")
    if tok.vocab_size != cfg.vocab_size:
        raise SystemExit(f"{CACHE_ROOT} has a {tok.vocab_size}-token tokenizer, not {cfg.vocab_size}; use --force to rebuild")
    eot_id = int(tok.encode(EOT)[0])

    parts = {name: CACHE_ROOT / "parts" / name for name in cfg.sources}
    for name, part in parts.items():
        for split in ("train", "validation"):
            if (part / split / "meta.json").exists():
                print(f"{name} {split}: already encoded")
            else:
                shutil.rmtree(part / split, ignore_errors=True)  # a crashed run can leave a half-written split
                write_part(tok, corpus(name), split, cfg, eot_id)
    shutil.rmtree(CACHE_ROOT / "tmp", ignore_errors=True)

    keep = {f"validation_{n}" for n in cfg.sources if n != cfg.val_source}
    for stale in CACHE_ROOT.glob("validation_*"):  # left over from an earlier mix of sources; train.py would score them
        if stale.name not in keep:
            shutil.rmtree(stale)
    assemble(parts, "train", "train", cfg.vocab_size)
    assemble({cfg.val_source: parts[cfg.val_source]}, "validation", "validation", cfg.vocab_size)
    for name in cfg.sources:
        if name != cfg.val_source:
            assemble({name: parts[name]}, "validation", f"validation_{name}", cfg.vocab_size)
    print(f"Wrote {CACHE_ROOT}")


if __name__ == "__main__":
    main(tyro.cli(PrepareConfig, description=__doc__))

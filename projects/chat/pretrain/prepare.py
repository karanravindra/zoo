import argparse
import json
import re
from pathlib import Path

import awkward as ak
import gigatoken as gt
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from tqdm import tqdm

from data import DATA_DIR, END, EOS, ROLES, SPECIAL_TOKENS


def render_chat(turns):
    return "".join(f"{ROLES[role]}\n{text.strip()}{END}\n" for role, text in turns)


def format_text(row, field="text"):
    return row[field]


def format_smoltalk(row):
    return render_chat((m["role"], m["content"]) for m in row["messages"])


GLAIVE_ROLES = {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}
GLAIVE_TURN = re.compile(r"\n\n\n(?=(?:USER|ASSISTANT|FUNCTION RESPONSE): )")


def format_glaive(row):
    """Function-calling chats: tool schemas, <functioncall> JSON and JSON tool responses."""
    turns = [("system", row["system"].removeprefix("SYSTEM: "))]
    for part in GLAIVE_TURN.split(row["chat"].strip()):
        label, _, body = part.partition(": ")
        if label not in GLAIVE_ROLES:
            return None
        turns.append((GLAIVE_ROLES[label], body.replace(EOS, "")))
    return render_chat(turns)


FORMATS = {
    "text": format_text,
    "content": lambda row: format_text(row, "content"),
    "smoltalk": format_smoltalk,
    "glaive": format_glaive,
}

# weight is the share of characters (a proxy for tokens) drawn from each source.
# Sources without a validation split hold out their first `holdout` rows from train.
SOURCES = [
    {
        "name": "fineweb-edu",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "train_split": "train",
        "val_split": None,
        "holdout": 10_000,
        "format": "text",
        "weight": 0.55,
    },
    {
        "name": "smoltalk",
        "dataset": "HuggingFaceTB/smoltalk",
        "config": "all",
        "train_split": "train",
        "val_split": "test",
        "format": "smoltalk",
        "weight": 0.15,
    },
    {
        "name": "function-calling-json",
        "dataset": "glaiveai/glaive-function-calling-v2",
        "config": None,
        "train_split": "train",
        "val_split": None,
        "holdout": 2_000,
        "format": "glaive",
        "weight": 0.10,
    },
    {
        "name": "python-code",
        "dataset": "codeparrot/codeparrot-clean",
        "config": None,
        "train_split": "train",
        "val_split": None,
        "holdout": 2_000,
        "format": "content",
        "weight": 0.08,
    },
    {
        "name": "finemath",
        "dataset": "HuggingFaceTB/finemath",
        "config": "finemath-4plus",
        "train_split": "train",
        "val_split": None,
        "holdout": 2_000,
        "format": "text",
        "weight": 0.07,
    },
    {
        "name": "wikipedia",
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "train_split": "train",
        "val_split": None,
        "holdout": 2_000,
        "format": "text",
        "weight": 0.05,
    },
]
BATCH_DOCS = 5_000


def source_documents(source, split, seed):
    from datasets import load_dataset

    validation = split == "validation"
    name = source["val_split"] if validation and source["val_split"] else None
    dataset = load_dataset(
        source["dataset"],
        source["config"],
        split=name or source["train_split"],
        streaming=True,
    )
    if name is None:
        holdout = source.get("holdout", 0)
        if validation:
            dataset = dataset.take(holdout)
        else:
            dataset = dataset.skip(holdout).shuffle(seed=seed, buffer_size=10_000)
    render = FORMATS[source["format"]]
    for row in dataset:
        text = render(row)
        if text and text.strip():
            yield text


def mixed_documents(split, seed, max_chars=None):
    """Yield (source index, document), always drawing from the source furthest below its share."""
    streams = [source_documents(s, split, seed) for s in SOURCES]
    indices = list(range(len(SOURCES)))
    weights = [s["weight"] for s in SOURCES]
    used = [0] * len(SOURCES)
    total = 0
    while streams and (max_chars is None or total < max_chars):
        slot = min(range(len(streams)), key=lambda i: used[i] / weights[i])
        try:
            doc = next(streams[slot])
        except StopIteration:
            # An exhausted source drops out and the rest are renormalised.
            for items in (streams, indices, weights, used):
                del items[slot]
            continue
        used[slot] += len(doc)
        total += len(doc)
        yield indices[slot], doc


def byte_level_tokenizer(vocab, merges):
    # Gigatoken trains on raw bytes; map them to the GPT-2 byte-level alphabet.
    printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
    chars = dict(zip(printable, map(chr, printable)))
    for byte in range(256):
        chars.setdefault(byte, chr(256 + len(chars) - len(printable)))

    def text(token):
        return "".join(chars[byte] for byte in token)

    tokenizer = Tokenizer(
        models.BPE(
            {text(token): id for id, token in vocab.items()},
            [(text(left), text(right)) for left, right in merges],
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


def pack_split(path, tokenizer, eos_id, split, budget, seed):
    """Write mixed documents to `path` until about `budget` tokens; count per source."""
    tokens = np.zeros(len(SOURCES), dtype=np.int64)
    docs = np.zeros(len(SOURCES), dtype=np.int64)
    batch, batch_sources = [], []

    def flush(file):
        lengths = write_batch(file, tokenizer, batch, eos_id)
        np.add.at(tokens, batch_sources, lengths)
        np.add.at(docs, batch_sources, 1)
        batch.clear()
        batch_sources.clear()

    with path.open("wb") as file, tqdm(desc=split, unit=" tok", total=budget) as bar:
        for source, doc in mixed_documents(split, seed):
            batch.append(doc)
            batch_sources.append(source)
            if len(batch) == BATCH_DOCS:
                before = tokens.sum()
                flush(file)
                bar.update(int(tokens.sum() - before))
                if tokens.sum() >= budget:
                    break
        else:
            if batch:
                flush(file)
    return tokens, docs


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer and pack the pretraining mix"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--tokenizer-chars", type=int, default=100_000_000)
    parser.add_argument("--train-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not len(SPECIAL_TOKENS) < args.vocab_size <= 65536:
        parser.error("vocab-size must fit the special tokens and uint16 storage")
    for name in ("tokenizer_chars", "train_tokens", "val_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.data_dir / "metadata.json"
    metadata_path.unlink(missing_ok=True)
    corpus = EOS.join(
        doc for _, doc in mixed_documents("train", args.seed, args.tokenizer_chars)
    )
    vocab, merges = gt.train_bpe(corpus.encode(), args.vocab_size, SPECIAL_TOKENS)
    del corpus
    hf_tokenizer = byte_level_tokenizer(vocab, merges)
    hf_tokenizer.save(str(args.data_dir / "tokenizer.json"))
    special_ids = {token: hf_tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    eos_id = special_ids[EOS]
    vocab_size = hf_tokenizer.get_vocab_size()
    tokenizer = gt.Tokenizer(hf_tokenizer)
    names = [source["name"] for source in SOURCES]
    counts, doc_counts, by_source = {}, {}, {}
    for split, budget in (
        ("train", args.train_tokens),
        ("validation", args.val_tokens),
    ):
        tokens, docs = pack_split(
            args.data_dir / f"{split}.bin", tokenizer, eos_id, split, budget, args.seed
        )
        if not tokens.sum():
            raise ValueError(f"{split} has no documents")
        counts[split], doc_counts[split] = int(tokens.sum()), int(docs.sum())
        by_source[split] = dict(zip(names, map(int, tokens)))
    metadata = {
        "sources": SOURCES,
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "special_tokens": special_ids,
        "dtype": "uint16",
        "tokens": counts,
        "documents": doc_counts,
        "tokens_by_source": by_source,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def write_batch(file, tokenizer, documents, eos_id):
    """Pack one batch of documents and return each document's token count."""
    # Gigatoken returns awkward arrays; flatten in bulk, never iterate per token.
    encoded = tokenizer.encode_batch(documents)
    eos = ak.Array(np.full((len(documents), 1), eos_id))
    packed = ak.concatenate([encoded, eos], axis=1)
    ak.to_numpy(ak.flatten(packed)).astype(np.uint16).tofile(file)
    return ak.to_numpy(ak.num(packed))


if __name__ == "__main__":
    main()

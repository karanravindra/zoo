import argparse
import json
from itertools import islice
from pathlib import Path

import gigatoken as gt
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

from data import DATA_DIR, EOS


def stories(split, local_file=None, limit=None):
    if local_file is not None:
        # Local fixtures use one story per nonempty line.
        with local_file.open() as file:
            yield from islice((line.strip() for line in file if line.strip()), limit)
    else:
        from datasets import load_dataset

        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        yield from islice(
            (row["text"] for row in dataset if row["text"].strip()), limit
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer and pack TinyStories tokens"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer-stories", type=int, default=100_000)
    parser.add_argument("--train-stories", type=int, default=None)
    parser.add_argument("--val-stories", type=int, default=None)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--val-file", type=Path)
    args = parser.parse_args()
    if not 257 <= args.vocab_size <= 65536:
        parser.error("vocab-size must be between 257 and 65536 for uint16 storage")
    for name in ("tokenizer_stories", "train_stories", "val_stories"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if bool(args.train_file) != bool(args.val_file):
        parser.error("Provide both train-file and val-file")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.data_dir / "metadata.json"
    metadata_path.unlink(missing_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer_limit = min(
        args.tokenizer_stories, args.train_stories or args.tokenizer_stories
    )
    tokenizer.train_from_iterator(
        stories("train", args.train_file, tokenizer_limit), trainer
    )
    tokenizer.save(str(args.data_dir / "tokenizer.json"))
    eos_id = tokenizer.token_to_id(EOS)
    vocab_size = tokenizer.get_vocab_size()
    tokenizer = gt.Tokenizer(tokenizer)
    counts, story_counts = {}, {}
    for split, local_file, limit in (
        ("train", args.train_file, args.train_stories),
        ("validation", args.val_file, args.val_stories),
    ):
        count, story_count = 0, 0
        buffer = []
        with (args.data_dir / f"{split}.bin").open("wb") as file:
            for story in tqdm(
                stories(split, local_file, limit), desc=split, unit=" stories"
            ):
                buffer.append(story)
                if len(buffer) == 1000:
                    count += write_batch(file, tokenizer, buffer, eos_id)
                    story_count += len(buffer)
                    buffer.clear()
            count += write_batch(file, tokenizer, buffer, eos_id)
            story_count += len(buffer)
        if not count:
            raise ValueError(f"{split} has no stories")
        counts[split], story_counts[split] = count, story_count
    metadata = {
        "dataset": "local" if args.train_file else "roneneldan/TinyStories",
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "dtype": "uint16",
        "tokens": counts,
        "stories": story_counts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def write_batch(file, tokenizer, stories_batch, eos_id):
    tokens = []
    for encoding in tokenizer.encode_batch(stories_batch):
        tokens.extend(encoding)
        tokens.append(eos_id)
    np.asarray(tokens, dtype=np.uint16).tofile(file)
    return len(tokens)


if __name__ == "__main__":
    main()

"""Generate a story continuation from a trained TinyStories checkpoint.

`uv run projects/tinystories/sample.py [ckpt.pt] [--prompt "Once upon a time"] [--max-tokens 200] [--temperature 0.8] [--top-k 40]`

Defaults to best.pt of the latest run. Needs the tokenizer written by prepare.py. Prints the samples, and with
`--out` also writes them to a file. An empty `--prompt ""` starts from the end-of-text token, i.e. a whole new story.
Generation uses a KV cache (see `Llama.generate`).
`--prompts FILE` instead continues every prompt in a JSONL file (the shared set that compares models) and writes one
`{id, source, prompt, text, n_tokens, ended}` row per story as JSONL.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import gigatoken as gt
import torch
import tyro
from llama import load_checkpoint
from tyro.conf import Positional

from zoo.paths import DATA_ROOT
from zoo.runs import latest_checkpoint
from zoo.utils.device import get_device

CACHE_ROOT = DATA_ROOT / "cache" / "tinystories"


@dataclass
class Args:
    ckpt: Positional[str | None] = None  # checkpoint (default: best.pt of the latest run)
    prompt: str = "Once upon a time"  # "" starts a new story from the end-of-text token
    max_tokens: int = 200  # new tokens per sample (stops early at end-of-text)
    temperature: float = 0.8  # 0 = greedy
    top_k: int | None = 40
    num_samples: int = 1  # generated as one batch
    seed: int = 0
    out: Path | None = None  # also write the samples to this file
    prompts: Path | None = None  # JSONL of {id, source, prompt} rows (see projects/pretrain/make_prompts.py): one story per prompt, written as JSONL to --out (default: eval_stories.jsonl next to the checkpoint); ignores --prompt and --num-samples
    compile: bool = False  # torch.compile the model with mode="reduce-overhead"; turns the KV cache off, and every step has a new length, so it recompiles often


def main(args: Args):
    device = get_device()
    ckpt = Path(args.ckpt) if args.ckpt else latest_checkpoint("tinystories", "best.pt")
    model, cfg, _ = load_checkpoint(ckpt, map_location=device)
    model = model.to(device).eval()
    if args.compile:
        model.compile(mode="reduce-overhead", dynamic=False)  # in place, so state_dict keys are unchanged
    tok = gt.Tokenizer(CACHE_ROOT / "tokenizer.json")
    eot_id = int(tok.encode("<|endoftext|>")[0])

    hidden_fn = (lambda i: model(i, return_hidden=True)) if args.compile else None  # the compiled path

    def encode(prompt: str) -> list[int]:
        ids = [int(t) for t in tok.encode(prompt)] if prompt else [eot_id]
        if prompt and cfg.doc_align != "off":
            ids = [eot_id] + ids  # a story-aligned model has only ever seen stories that begin with the end-of-text token
        return ids

    def generate(batch: list[list[int]], seeds: list[int] | None = None) -> list[tuple[str, int, bool]]:
        """Continue equal-length prompts as one batch: (story text, tokens, ended) per row. `seeds` gives each row its own
        generator; without it the batch shares one seeded by `--seed`."""
        prompt = torch.tensor(batch, device=device)
        g = [torch.Generator(device=device).manual_seed(x) for x in seeds] if seeds else torch.Generator(device=device).manual_seed(args.seed)
        out = model.generate(prompt, args.max_tokens, args.temperature, args.top_k, eos_id=eot_id, generator=g, hidden_fn=hidden_fn)
        rows = []
        n = prompt.shape[1]
        for row in out.tolist():
            finished = eot_id in row[n:]
            if finished:  # drop everything from the first generated end-of-text on
                row = row[: n + row[n:].index(eot_id)]
            if row and row[0] == eot_id:
                row = row[1:]  # the end-of-text start token is not part of the story
            rows.append((tok.decode(row).decode("utf-8", errors="replace"), len(row), finished))
        return rows

    if args.prompts:  # one continuation per prompt; prompts of equal token length share a batch
        items = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [encode(it["prompt"]) for it in items]
        by_len: dict[int, list[int]] = {}
        for i, x in enumerate(ids):
            by_len.setdefault(len(x), []).append(i)
        results: list[tuple[str, int, bool] | None] = [None] * len(items)
        for group in by_len.values():
            for k in range(0, len(group), 64):
                chunk = group[k : k + 64]
                # seeded per prompt, so a story does not change when other prompts are added to or removed from the file
                seeds = [int(hashlib.sha1(f"{args.seed}:{items[i]['id']}".encode()).hexdigest()[:8], 16) for i in chunk]
                for i, res in zip(chunk, generate([ids[i] for i in chunk], seeds)):
                    results[i] = res
        out_path = args.out or ckpt.parent / "eval_stories.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for it, (text, n_tokens, ended) in zip(items, results):
                f.write(json.dumps({"id": it["id"], "source": it["source"], "prompt": it["prompt"], "text": text, "n_tokens": n_tokens, "ended": ended}) + "\n")
        print(f"Wrote {len(items)} stories from {ckpt} to {out_path}")
        return

    header = f"Checkpoint: {ckpt}\nprompt={args.prompt!r} max_tokens={args.max_tokens} temperature={args.temperature} top_k={args.top_k} seed={args.seed}"
    blocks = []
    for i, (text, n_tokens, finished) in enumerate(generate([encode(args.prompt)] * args.num_samples)):
        blocks.append(f"--- sample {i} ({n_tokens} tokens, {'ended' if finished else 'cut off at max_tokens'}) ---\n{text}\n")
    text = header + "\n\n" + "\n".join(blocks)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))

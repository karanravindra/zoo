"""Generate a story continuation from a trained TinyStories checkpoint.

`uv run projects/pretrain/sample.py [ckpt.pt] [--prompt "Once upon a time"] [--max-tokens 200] [--temperature 0.8] [--top-k 40]`

Defaults to best.pt of the latest run. Needs the tokenizer written by prepare.py. Prints the samples, and with
`--out` also writes them to a file. An empty `--prompt ""` starts from the end-of-text token, i.e. a whole new story.
Generation uses a KV cache (see `Llama.generate`).
"""

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

CACHE_ROOT = DATA_ROOT / "cache" / "pretrain"


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
    compile: bool = False  # torch.compile the model with mode="reduce-overhead"; turns the KV cache off, and every step has a new length, so it recompiles often


def main(args: Args):
    device = get_device()
    ckpt = Path(args.ckpt) if args.ckpt else latest_checkpoint("pretrain", "best.pt")
    model, cfg, _ = load_checkpoint(ckpt, map_location=device)
    model = model.to(device).eval()
    if args.compile:
        model.compile(mode="reduce-overhead", dynamic=False)  # in place, so state_dict keys are unchanged
    tok = gt.Tokenizer(CACHE_ROOT / "tokenizer.json")
    eot_id = int(tok.encode("<|endoftext|>")[0])

    ids = [int(t) for t in tok.encode(args.prompt)] if args.prompt else [eot_id]
    if args.prompt and cfg.doc_align != "off":
        ids = [eot_id] + ids  # a story-aligned model has only ever seen stories that begin with the end-of-text token
    prompt = torch.tensor([ids], device=device).repeat(args.num_samples, 1)
    g = torch.Generator(device=device).manual_seed(args.seed)
    hidden_fn = (lambda i: model(i, return_hidden=True)) if args.compile else None  # the compiled path
    out = model.generate(prompt, args.max_tokens, args.temperature, args.top_k, eos_id=eot_id, generator=g, hidden_fn=hidden_fn)
    header = f"Checkpoint: {ckpt}\nprompt={args.prompt!r} max_tokens={args.max_tokens} temperature={args.temperature} top_k={args.top_k} seed={args.seed}"
    blocks = []
    for i, row in enumerate(out.tolist()):
        n = prompt.shape[1]
        finished = eot_id in row[n:]
        if finished:  # drop everything from the first generated end-of-text on
            row = row[: n + row[n:].index(eot_id)]
        if row and row[0] == eot_id:
            row = row[1:]  # the end-of-text start token is not part of the story
        text = tok.decode(row).decode("utf-8", errors="replace")
        blocks.append(f"--- sample {i} ({len(row)} tokens, {'ended' if finished else 'cut off at max_tokens'}) ---\n{text}\n")
    text = header + "\n\n" + "\n".join(blocks)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))

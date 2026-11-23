"""Write the shared story prompts that `sample.py --prompts` and `judge.py` use to compare models.

`uv run projects/pretrain/make_prompts.py [--in-domain 100] [--per-source 10] [--seed 0]`

Each prompt is the first sentence of a held-out story, so every model continues the same openings:
- in-domain: stories from `validation/` (the held-out TinyStories split), the style every model is compared on;
- out-of-domain: `--per-source` stories from each `validation_<source>/` (SimpleStories, Cosmopedia, WritingPrompts,
  Gutenberg), which tests whether the pretrain mix generalises beyond TinyStories.

Writes `eval_prompts.jsonl` next to this script (small, so it is committed): one `{id, source, prompt, reference}`
row per prompt, where `reference` is the rest of the held-out story, for a human spot check only (never judged).
Needs the cache written by `prepare.py`; it only decodes the few sampled stories and runs no model.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gigatoken as gt
import numpy as np
import tyro

from zoo.paths import DATA_ROOT

CACHE_ROOT = DATA_ROOT / "cache" / "pretrain"
SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")  # a full stop (and closing quotes) followed by whitespace


@dataclass
class Config:
    in_domain: int = 100  # prompts from the held-out TinyStories split
    per_source: int = 10  # prompts from each other source's held-out split
    min_chars: int = 25  # skip openings shorter than this
    max_chars: int = 200  # skip openings longer than this (a long first sentence is usually not a story opening)
    reference_chars: int = 1000  # characters of the rest of the story kept as `reference`
    seed: int = 0
    out: Path = Path(__file__).parent / "eval_prompts.jsonl"


def split_story(text: str, cfg: Config) -> tuple[str, str] | None:
    """(first sentence, rest of the story), or None if the opening has an unsuitable length."""
    text = text.strip()
    m = SENTENCE_END.search(text)
    if not m:
        return None
    first, rest = text[: m.end()].strip(), text[m.end() :]
    return (first, rest) if cfg.min_chars <= len(first) <= cfg.max_chars else None


def sample_prompts(tok, split: str, n: int, source: str, cfg: Config, rng: np.random.Generator) -> list[dict]:
    tokens = np.load(CACHE_ROOT / split / "tokens.npy", mmap_mode="r")
    eot_id = json.loads((CACHE_ROOT / split / "meta.json").read_text())["eot_id"]
    chunk = 1 << 24  # scan the memmap in pieces: a large split need not fit in RAM as a mask
    eots = np.concatenate([np.flatnonzero(tokens[o : o + chunk] == eot_id) + o for o in range(0, len(tokens), chunk)])  # story boundaries
    starts = np.concatenate([[0], eots + 1])
    ends = np.concatenate([eots, [len(tokens)]])
    keep = np.flatnonzero(ends > starts)  # drops empty stories (a leading end-of-text token)
    rows = []
    for i in rng.permutation(keep):  # decode only as many stories as it takes to find n usable openings
        text = tok.decode([int(t) for t in tokens[starts[i] : ends[i]]]).decode("utf-8", errors="replace")
        parts = split_story(text, cfg)
        if parts:
            rows.append({"id": f"{source}-{len(rows):03d}", "source": source, "prompt": parts[0], "reference": parts[1][: cfg.reference_chars]})
        if len(rows) == n:
            return rows
    raise SystemExit(f"{split}: only {len(rows)} usable openings, wanted {n}")


def main(cfg: Config) -> None:
    tok = gt.Tokenizer(CACHE_ROOT / "tokenizer.json")
    rng = np.random.default_rng(cfg.seed)
    rows = sample_prompts(tok, "validation", cfg.in_domain, "tinystories", cfg, rng)
    for d in sorted(CACHE_ROOT.glob("validation_*")):
        source = d.name.removeprefix("validation_")
        rows += sample_prompts(tok, d.name, cfg.per_source, source, cfg, rng)
    cfg.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    counts = dict(Counter(r["source"] for r in rows))
    print(f"Wrote {len(rows)} prompts to {cfg.out}: {counts}")


if __name__ == "__main__":
    main(tyro.cli(Config))

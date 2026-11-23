"""Zero-shot scores of TinyStories checkpoints on the usual small-model benchmarks.

`uv run projects/pretrain/bench.py [run_dir ...] [--tasks blimp hellaswag ...] [--limit N] [--ckpt best.pt]`

Defaults to the checkpoint of every run under `<repo>/runs/pretrain`. Writes `bench.json` into each run dir and one
`benchmarks.csv` next to the runs (not with `--limit`, which is for a quick look), and prints a table.

Tasks and their conventions follow lm-evaluation-harness, which is not needed: each candidate text is scored by the
sum of its token log-probabilities after a context, and the highest wins (`acc`; `acc_norm` divides by the candidate's
length in characters). `blimp`: the grammatical sentence of a minimal pair must score higher than the ungrammatical
one (67 subsets, also reported per phenomenon). `lambada`: greedy accuracy on the last word, and its perplexity.
`winogrande` scores the text after the blank given each option. The datasets come from the Hugging Face Hub.

Every sequence starts with the end-of-text token, which is how the models saw stories in training; a context that
does not fit in the block size is cut from the left. A story mask changes nothing for these single-story rows, so
the model runs the way it trains without it. Models trained on TinyStories only are expected to sit near chance on
most of these; the standard error and the chance level are in the output to show how much a gap can mean.
"""

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import gigatoken as gt
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from compare import IGNORE_KEYS, load, short_labels
from datasets import get_dataset_config_names, load_dataset
from llama import load_checkpoint
from tyro.conf import Positional

from zoo.paths import DATA_ROOT, runs_root
from zoo.runs import sort_runs
from zoo.utils.device import get_device

RUNS_ROOT = runs_root("pretrain")
CACHE_ROOT = DATA_ROOT / "cache" / "pretrain"

TaskName = Literal["blimp", "hellaswag", "lambada", "piqa", "arc_easy", "arc_challenge", "winogrande", "sciq", "openbookqa", "copa"]
TASKS: list[TaskName] = ["blimp", "hellaswag", "lambada", "piqa", "arc_easy", "arc_challenge", "winogrande", "sciq", "openbookqa", "copa"]
CHANCE = {"blimp": 0.5, "hellaswag": 0.25, "lambada": 0.0, "piqa": 0.5, "arc_easy": 0.25, "arc_challenge": 0.25, "winogrande": 0.5, "sciq": 0.25, "openbookqa": 0.25, "copa": 0.5}
NORM_FREE = {"blimp", "winogrande", "copa", "lambada"}  # no acc_norm: one candidate length (or the same continuation for every option)
HEADLINE = {"hellaswag": "acc_norm", "arc_challenge": "acc_norm", "openbookqa": "acc_norm"}  # the metric of the table; every other task: acc

# An example is its (context, continuation) candidates, the index of the right one, and an optional tag to group by.
Example = tuple[list[tuple[str, str]], int, object]


@dataclass
class Args:
    runs: Positional[list[Path]] = field(default_factory=list)  # run dirs (default: all under runs/pretrain that have the checkpoint)
    ckpt: str = "best.pt"  # checkpoint file inside each run dir
    tasks: list[TaskName] = field(default_factory=lambda: list(TASKS))  # which benchmarks to run
    limit: int = 0  # examples per task (per subset for blimp); 0 = all. Nothing is written when set
    batch_size: int = 64  # sequences per forward pass; they are sorted by length first, so padding is small
    bf16: bool = False  # bfloat16 autocast (cuda); fp32 by default so that near-tied candidates are not decided by rounding
    compile: bool = False  # model.compile(mode="reduce-overhead", dynamic=False); every new padded shape recompiles, so lengths are padded to multiples of 64
    out: Path = RUNS_ROOT / "benchmarks.csv"


# ---------------------------------------------------------------- tasks


def _take(ds, limit: int):
    return ds.select(range(min(limit, len(ds)))) if limit else ds


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def build(task: str, limit: int) -> list[Example]:
    ex: list[Example] = []
    if task == "blimp":
        for name in get_dataset_config_names("nyu-mll/blimp"):
            for d in _take(load_dataset("nyu-mll/blimp", name, split="train"), limit):
                ex.append(([("", d["sentence_good"]), ("", d["sentence_bad"])], 0, (d["linguistics_term"], d["UID"])))
    elif task == "hellaswag":
        for d in _take(load_dataset("Rowan/hellaswag", split="validation"), limit):
            ctx = _hellaswag_clean(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize())
            ex.append(([(ctx, " " + _hellaswag_clean(e)) for e in d["endings"]], int(d["label"]), None))
    elif task == "lambada":
        for d in _take(load_dataset("EleutherAI/lambada_openai", "en", split="test"), limit):
            ctx, last = d["text"].rsplit(" ", 1)
            ex.append(([(ctx, " " + last)], 0, None))
    elif task == "piqa":
        for d in _take(load_dataset("baber/piqa", split="validation"), limit):
            ctx = f"Question: {d['goal']}\nAnswer:"
            ex.append(([(ctx, " " + d["sol1"]), (ctx, " " + d["sol2"])], int(d["label"]), None))
    elif task in ("arc_easy", "arc_challenge"):
        for d in _take(load_dataset("allenai/ai2_arc", "ARC-Easy" if task == "arc_easy" else "ARC-Challenge", split="test"), limit):
            ctx = f"Question: {d['question']}\nAnswer:"
            ex.append(([(ctx, " " + t) for t in d["choices"]["text"]], d["choices"]["label"].index(d["answerKey"]), None))
    elif task == "winogrande":
        for d in _take(load_dataset("allenai/winogrande", "winogrande_xl", split="validation"), limit):
            i = d["sentence"].index("_")
            cont = " " + d["sentence"][i + 1 :].strip()  # the text after the blank is scored given each option filled in
            ex.append(([(d["sentence"][:i] + d["option1"], cont), (d["sentence"][:i] + d["option2"], cont)], int(d["answer"]) - 1, None))
    elif task == "sciq":
        for d in _take(load_dataset("allenai/sciq", split="test"), limit):
            ctx = f"{d['support'].lstrip()}\nQuestion: {d['question']}\nAnswer:"
            choices = [d["distractor1"], d["distractor2"], d["distractor3"], d["correct_answer"]]
            ex.append(([(ctx, " " + c) for c in choices], 3, None))
    elif task == "openbookqa":
        for d in _take(load_dataset("allenai/openbookqa", "main", split="test"), limit):
            ex.append(([(d["question_stem"], " " + t) for t in d["choices"]["text"]], d["choices"]["label"].index(d["answerKey"]), None))
    elif task == "copa":
        for d in _take(load_dataset("aps/super_glue", "copa", split="validation"), limit):
            ctx = d["premise"].strip()[:-1] + " " + {"cause": "because", "effect": "therefore"}[d["question"]]
            ex.append(([(ctx, " " + c[0].lower() + c[1:]) for c in (d["choice1"], d["choice2"])], int(d["label"]), None))
    else:
        raise ValueError(task)
    return ex


# ---------------------------------------------------------------- scoring


def encode_pair(ctx: str, cont: str) -> tuple[str, str]:
    """(context, context + continuation) as strings to tokenize; trailing spaces of the context move to the continuation."""
    n = len(ctx) - len(ctx.rstrip())
    if n:
        ctx, cont = ctx[:-n], ctx[-n:] + cont
    return ctx, ctx + cont


def tokenize(tok: gt.Tokenizer, examples: list[Example]) -> list[tuple[np.ndarray, np.ndarray]]:
    """(context ids, continuation ids) of every candidate; the continuation is the tail of the joint encoding."""
    pairs = [encode_pair(c, t) for e in examples for c, t in e[0]]
    uniq = list({s for p in pairs for s in p})
    enc = dict(zip(uniq, (np.asarray(x, dtype=np.int32) for x in tok.encode_batch_list(uniq)), strict=True))
    return [(enc[c], enc[w][len(enc[c]) :]) for c, w in pairs]


def batches(reqs: list[tuple[np.ndarray, np.ndarray]], block_size: int, window: int, eot: int, rows: int, mult: int):
    """Length-sorted padded batches of `[eot] + context + continuation`, yielding (request indices, ids, context
    lengths including eot, continuation lengths). Padding goes on the right, where causal attention cannot see it."""
    seqs = []
    for c, t in reqs:
        room = block_size - 1 - len(t)
        assert room >= 0, "a continuation is longer than the context window"
        seqs.append(np.concatenate([[eot], c[max(0, len(c) - room) :], t]).astype(np.int64))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for s in range(0, len(order), rows):
        idx = order[s : s + rows]
        longest = max(len(seqs[i]) for i in idx)
        m = 128 if window and longest > window else mult  # FlexAttention (a window past its length) wants 128-multiples
        t_len = min(block_size, -(-longest // m) * m)
        x = np.zeros((len(idx), t_len), dtype=np.int64)
        for j, i in enumerate(idx):
            x[j, : len(seqs[i])] = seqs[i]
        cont = np.array([len(reqs[i][1]) for i in idx])
        ctx = np.array([len(seqs[i]) for i in idx]) - cont
        yield idx, torch.from_numpy(x), torch.from_numpy(ctx), torch.from_numpy(cont)


@torch.no_grad()
def score(model, cfg, reqs, device, eot: int, rows: int, bf16: bool, mult: int, via_cache: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per candidate: sum of continuation token log-probs, and whether every continuation token was the argmax.

    `via_cache` runs the batch as a KV-cache priming pass, which applies an attention window with a plain SDPA band
    mask: the same attention as the training forward for a single story, but without FlexAttention, whose uncompiled
    fallback materializes the whole (batch, heads, T, T) score matrix and runs out of memory on long rows."""
    ll, greedy = np.zeros(len(reqs)), np.zeros(len(reqs), dtype=bool)
    w = model.classifier_weight.float()
    for idx, x, c, m in batches(reqs, cfg.block_size, cfg.attn_window, eot, rows, mult):
        x, c, m = x.to(device), c.to(device), m.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            h = model(x, return_hidden=True, cache=[{} for _ in model.blocks] if via_cache else None)
        pos = torch.arange(x.shape[1], device=device)[None]
        sel = (pos >= (c - 1)[:, None]) & (pos < (c + m - 1)[:, None])  # positions whose next token is a continuation token
        tgt = torch.cat([x[:, 1:], x[:, :1]], 1)[sel]
        row = torch.arange(len(x), device=device)[:, None].expand_as(sel)[sel]
        logits = F.linear(h[sel].float(), w)
        tok_lp = logits.log_softmax(-1).gather(1, tgt[:, None])[:, 0]
        wrong = (logits.argmax(-1) != tgt).float()
        z = torch.zeros(len(x), device=device)
        ll[idx] = z.index_add(0, row, tok_lp).double().cpu().numpy()
        greedy[idx] = z.index_add(0, row, wrong).cpu().numpy() == 0
    return ll, greedy


def stderr(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else float("nan")


def summarize(task: str, examples: list[Example], reqs, ll: np.ndarray, greedy: np.ndarray) -> dict:
    """Metrics of one task from the candidates' scores (in the order of `examples`, candidates flattened)."""
    if task == "lambada":  # one candidate per example
        acc = float(greedy.mean())
        return {"n": len(examples), "acc": acc, "acc_stderr": stderr(acc, len(examples)), "ppl": float(math.exp(-ll.mean()))}
    correct, correct_norm, tags, o = [], [], [], 0
    for cands, gold, tag in examples:
        k = len(cands)
        lens = np.array([max(1, len(t.strip())) for _, t in cands])
        correct.append(int(np.argmax(ll[o : o + k]) == gold))
        correct_norm.append(int(np.argmax(ll[o : o + k] / lens) == gold))
        tags.append(tag)
        o += k
    n, acc = len(examples), float(np.mean(correct))
    out = {"n": n, "acc": acc, "acc_stderr": stderr(acc, n)}
    if task not in NORM_FREE:
        an = float(np.mean(correct_norm))
        out |= {"acc_norm": an, "acc_norm_stderr": stderr(an, n)}
    if task == "blimp":
        for name, i in (("by_phenomenon", 0), ("by_subset", 1)):
            groups: dict[str, list[int]] = {}
            for t, c in zip(tags, correct, strict=True):
                groups.setdefault(t[i], []).append(c)
            out[name] = {g: float(np.mean(v)) for g, v in sorted(groups.items())}
    return out


def headline(task: str, res: dict) -> float:
    return res[HEADLINE.get(task, "acc")]


# ---------------------------------------------------------------- main


def main(args: Args):
    device = get_device()
    dirs = args.runs or [d for d in sorted(RUNS_ROOT.iterdir()) if d.is_dir() and (d / args.ckpt).exists()]
    if not dirs:
        raise SystemExit(f"no run with {args.ckpt} under {RUNS_ROOT}")
    runs = sort_runs([load(d) for d in dirs], IGNORE_KEYS)
    labels = short_labels(runs)
    tok = gt.Tokenizer(CACHE_ROOT / "tokenizer.json")
    eot = int(tok.encode("<|endoftext|>")[0])

    prepared = {}
    for task in args.tasks:
        t0 = time.time()
        examples = build(task, args.limit)
        reqs = tokenize(tok, examples)
        prepared[task] = (examples, reqs)
        print(f"{task}: {len(examples)} examples, {len(reqs)} candidates ({time.time() - t0:.0f}s to load and tokenize)", flush=True)

    results = {}
    for run, label in zip(runs, labels, strict=True):
        model, cfg, ckpt = load_checkpoint(run["path"] / args.ckpt, map_location=device)
        model = model.to(device).eval()
        if args.compile:
            model.compile(mode="reduce-overhead", dynamic=False)  # in place, so state_dict keys are unchanged
        t0, res = time.time(), {}
        for task, (examples, reqs) in prepared.items():
            ll, greedy = score(model, cfg, reqs, device, eot, args.batch_size, args.bf16, 64 if args.compile else 16, via_cache=cfg.attn_window > 0 and not args.compile)
            res[task] = summarize(task, examples, reqs, ll, greedy)
        res["avg"] = float(np.mean([headline(t, r) for t, r in res.items() if t in prepared]))
        results[run["name"]] = res
        print(f"{label:34} {time.time() - t0:5.0f}s  " + "  ".join(f"{t} {headline(t, r):.3f}" for t, r in res.items() if t in prepared), flush=True)
        if not args.limit:
            meta = {"checkpoint": args.ckpt, "step": ckpt["step"], "params": run["params"], "precision": "bf16" if args.bf16 else "fp32"}
            (run["path"] / "bench.json").write_text(json.dumps({**meta, "results": res}, indent=2))
        del model

    # table: the headline metric of each task (acc_norm for hellaswag, arc_challenge, openbookqa; acc otherwise)
    cols = list(args.tasks)
    head = f"{'run':34} {'params':>7} " + " ".join(f"{c[:11]:>11}" for c in cols) + ("  lambada_ppl" if "lambada" in cols else "") + "    avg"
    print("\n" + head)
    for run, label in zip(runs, labels, strict=True):
        r = results[run["name"]]
        row = f"{label:34} {run['params'] / 1e6:6.1f}M " + " ".join(f"{100 * headline(c, r[c]):11.1f}" for c in cols)
        print(row + (f"  {r['lambada']['ppl']:11.0f}" if "lambada" in cols else "") + f"  {100 * r['avg']:5.1f}")
    print(f"{'chance':34} {'':>7} " + " ".join(f"{100 * CHANCE[c]:11.1f}" for c in cols))
    first = results[runs[0]["name"]]
    print(f"{'+- 1 stderr (first run)':34} {'':>7} " + " ".join(f"{100 * first[c]['acc_norm_stderr' if HEADLINE.get(c) else 'acc_stderr']:11.1f}" for c in cols))

    if not args.limit:
        def flat(r: dict) -> dict[str, float]:  # one column per task metric (blimp's breakdowns stay in bench.json)
            return {f"{c}_{m}": v for c in cols for m, v in r[c].items() if m not in ("n", "by_phenomenon", "by_subset")}

        lines = ["run,label,params," + ",".join(flat(results[runs[0]["name"]])) + ",avg"]
        for run, label in zip(runs, labels, strict=True):
            r = results[run["name"]]
            lines.append(f"{run['name']},{label},{run['params']}," + ",".join(f"{v:.6g}" for v in flat(r).values()) + f",{r['avg']:.6g}")
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nWrote {args.out} and bench.json in each run dir")


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))

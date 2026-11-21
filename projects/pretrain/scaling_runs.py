"""Launch the scaling-law runs: a table of (width, depth, batch, lr, compute) cells, trained one after another.

`uv run projects/pretrain/scaling_runs.py PHASE [PHASE ...] [--dry-run]`

One protocol for every cell, so that runs can be compared and fitted together (`scaling.py` reads them back):
- Constant LR after a warmup of the same number of tokens (13M) in every run, and a weight EMA whose window is a
  fraction of the steps taken so far. Neither depends on how long the run is meant to be, so the first k steps of a
  long run are exactly a shorter run of k steps: every eval row of a run is the result for a smaller compute budget too.
- Head size 64 (`--n-heads` = width / 64), context 2048, document mask, window attention, as in scales.md.
- The same 800 validation windows in every run (micro-batch x `--eval-batches` = 800, drawn from one seeded stream), and
  no per-source validation or text samples, so the evals are cheap; about 25 eval rows per run, evenly spaced in steps.
- The micro-batch is 16 sequences, or 8 for the larger shapes so they fit in 16 GB; the global batch is made up with
  `--grad-accum`.

A cell is skipped when its run directory already has results.json, so a phase can be rerun after a crash or extended by
adding cells to the table. `PHASES` is that table; the `--dry-run` listing shows parameters, tokens per parameter,
steps and a time estimate for each cell (from the throughput of the 1e16 d320 L5 runs; small models run slower).
"""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import tyro
from llama import Config, build_model
from tyro.conf import Positional

from zoo.paths import REPO_ROOT, runs_root

RUNS_ROOT = runs_root("pretrain")
BLOCK = 2048
HEAD_DIM = 64
WARMUP_TOKENS = 200 * 32 * BLOCK  # the warmup of the batch-32 runs in the first sweep, kept as a number of tokens
EVAL_SEQUENCES = 800  # validation windows per eval: micro-batch x eval batches
N_EVALS = 25
FLOPS_PER_S = 5.4e13  # measured on the d320 L5 runs: 1e16 FLOPs in 186 s


@dataclass(frozen=True)
class Cell:
    dim: int
    layers: int
    batch: int  # global batch, sequences per optimizer step
    lr: float
    budget: float = 1e16  # training FLOPs
    seed: int = 0


def short(x: float) -> str:
    """4e-3 -> '4e-3', 0.0028 -> '2.8e-3', 1e16 -> '1e16'."""
    m, e = f"{x:.1e}".split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def name_of(c: Cell) -> str:
    return f"sc-d{c.dim}-L{c.layers}-bs{c.batch}-lr{short(c.lr)}-c{short(c.budget)}" + (f"-s{c.seed}" if c.seed else "")


def model_config(c: Cell) -> Config:
    return Config(dim=c.dim, n_layers=c.layers, n_heads=c.dim // HEAD_DIM, block_size=BLOCK, doc_mask=True, attn_window=512, attn_global_every=5)


def micro_batch(c: Cell) -> int:
    return 16 if c.dim * c.layers <= 2600 else 8  # activation memory grows with width x depth x micro-batch


def plan(c: Cell) -> dict:
    """Everything derived from a cell: steps, eval cadence, warmup, micro-batch split, parameters."""
    cfg = model_config(c)
    tokens_per_step = c.batch * BLOCK
    steps = max(1, round(c.budget / (tokens_per_step * cfg.train_flops_per_token)))
    micro = micro_batch(c)
    params = sum(p.numel() for p in build_model(cfg).parameters())
    return dict(steps=steps, micro=micro, accum=c.batch // micro, eval_every=max(1, steps // N_EVALS), eval_batches=EVAL_SEQUENCES // micro,
                warmup=max(10, round(WARMUP_TOKENS / tokens_per_step)), params=params, tokens=steps * tokens_per_step,
                minutes=(c.budget / FLOPS_PER_S + 60) / 60)


def command(c: Cell, p: dict) -> list[str]:
    return ["uv", "run", "projects/pretrain/train.py",
            "--dim", str(c.dim), "--n-heads", str(c.dim // HEAD_DIM), "--n-layers", str(c.layers), "--block-size", str(BLOCK),
            "--doc-mask", "--attn-window", "512", "--attn-global-every", "5",
            "--flop-budget", f"{c.budget:g}", "--batch-size", str(p["micro"]), "--grad-accum", str(p["accum"]), "--lr", f"{c.lr:g}",
            "--warmup-steps", str(p["warmup"]), "--eval-every", str(p["eval_every"]), "--eval-batches", str(p["eval_batches"]),
            "--ema", "--ema-frac", "0.1", "--min-lr-ratio", "1.0", "--val-source-batches", "0", "--sample-every", "0",
            "--seed", str(c.seed), "--run-name", name_of(c)]


# The phases, in the order they are meant to run. Later phases are written after reading the earlier ones (see README).
PHASES: dict[str, list[Cell]] = {
    # Batch size at the reference shape (d320 L5): the first sweep covered 16-64; 128 and 256 find where the loss turns up.
    # The 3e16 run checks that a long run's early evals match a short run (compare its 1e16 point with sweep-bs32-lr4e-3).
    "batch": [Cell(320, 5, 32, 4e-3, 3e16)] + [Cell(320, 5, b, lr) for b in (128, 256) for lr in (2.8e-3, 4e-3, 5.6e-3)],
    # The width x depth grid at batch 32, one run per shape at the LR read off lrcal (best LR ~ 8e-3 at d128, ~ 4e-3 from d320 to d448 at
    # 2 to 12 layers: flat in depth, flat in width above d320, kept a little low for the wide and deep shapes, where the cliff above the
    # best LR is closest). Shapes up to 5M parameters get 1e16 FLOPs; the larger ones 3e16, where 6M to 16M parameters are near compute-optimal
    # and 640x5 (27M, ~6 tokens per parameter) anchors the undertrained side. Width/depth pairs at about equal N (5-7M: 192x12, 256x5, 320x3,
    # 448x2) separate the effect of shape from that of size.
    "shapes": [Cell(d, l, 32, lr, c) for d, l, lr, c in [
        (128, 5, 8e-3, 1e16), (192, 5, 6e-3, 1e16), (256, 5, 4.5e-3, 1e16), (320, 2, 4e-3, 1e16), (320, 3, 4e-3, 1e16),
        (192, 12, 5e-3, 3e16), (448, 2, 3.6e-3, 3e16), (256, 12, 4e-3, 3e16), (320, 8, 3.8e-3, 3e16), (448, 5, 3.6e-3, 3e16),
        (320, 12, 3.5e-3, 3e16), (640, 3, 3.2e-3, 3e16), (640, 5, 3.2e-3, 3e16)]],
    # The test of scaling.py's parametric fit, which was fitted on runs of at most 3e16 FLOPs: three shapes at 3.3x that compute, spread
    # over the range of N that the fits with different beta call compute-optimal (11M to 37M; the split of the budget between N and D is
    # what the data do not pin down, while the loss at 1e17 is nearly the same for all of them). `scaling.py holdout` prints the
    # prediction for each before they are run. d320 x 8 is the 3e16 run continued; the LRs are those of the `shapes` phase.
    "holdout": [Cell(d, l, 32, lr, 1e17) for d, l, lr in [(640, 7, 3.2e-3), (320, 8, 3.8e-3), (448, 8, 3.6e-3)]],
    # bs128 and bs256 had their best LR at the lowest value tried (2.8e-3): lower ones, so that the critical batch size is read at a tuned LR.
    "batchlr": [Cell(320, 5, 128, lr) for lr in (1.4e-3, 2e-3)] + [Cell(320, 5, 256, lr) for lr in (1e-3, 1.4e-3, 2e-3)],
    # The best LR of four shapes at the ends of the width and depth ranges: five LRs spaced by sqrt(2) around a guess (LR ~ 1/width,
    # ~ depth^-0.5), each shape at a budget where it is actually trained (the small ones 3e15, the big ones 1e16: a 27M model gets
    # 234 steps out of 3e15, most of them warmup). With the reference shape's sweeps these are the groups of scaling.py's LR law.
    "lrcal": [Cell(d, l, 32, lr, c) for d, l, c, lrs in [
        (128, 5, 3e15, (4e-3, 5.6e-3, 8e-3, 1.1e-2, 1.6e-2)),
        (320, 2, 3e15, (3e-3, 4.2e-3, 6e-3, 8.5e-3, 1.2e-2)),
        (448, 5, 1e16, (1.4e-3, 2e-3, 2.8e-3, 4e-3, 5.6e-3)),
        (320, 12, 1e16, (1.3e-3, 1.8e-3, 2.5e-3, 3.5e-3, 5e-3))] for lr in lrs],
}


@dataclass
class Args:
    phases: Positional[list[str]]  # phase names from PHASES (see the table at the bottom of the file)
    dry_run: bool = False  # list the cells with parameters, steps and a time estimate, and stop


def main(args: Args):
    unknown = [p for p in args.phases if p not in PHASES]
    if unknown:
        raise SystemExit(f"unknown phase {', '.join(unknown)}; the table has: {', '.join(PHASES)}")
    cells = [c for ph in args.phases for c in PHASES[ph]]
    todo = [(c, plan(c)) for c in cells if not (RUNS_ROOT / name_of(c) / "results.json").exists()]
    total = sum(p["minutes"] for _, p in todo)
    print(f"{len(cells)} cells, {len(cells) - len(todo)} already done, {len(todo)} to run, about {total:.0f} min ({total / 60:.1f} h)")
    w = max((len(name_of(c)) for c, _ in todo), default=10) + 2
    print(f"{'run':<{w}}{'params':>8}{'tok/param':>11}{'steps':>8}{'micro':>6}{'accum':>6}{'min':>6}")
    for c, p in todo:
        print(f"{name_of(c):<{w}}{p['params'] / 1e6:>7.2f}M{p['tokens'] / p['params']:>11.0f}{p['steps']:>8}{p['micro']:>6}{p['accum']:>6}{p['minutes']:>6.1f}")
    if args.dry_run:
        return
    failed = []
    for i, (c, p) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {name_of(c)}  ({time.strftime('%H:%M:%S')})", flush=True)
        if subprocess.run(command(c, p), cwd=REPO_ROOT).returncode != 0:  # a crash must not end the phase; divergence exits 0 anyway
            failed.append(name_of(c))
            print(f"FAILED: {name_of(c)}", flush=True)
    print(f"\ndone; failed: {', '.join(failed) or 'none'}")


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))

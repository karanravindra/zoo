"""Scaling laws from the pretraining runs: noise, learning rate, batch size, then (width, depth) and compute.

`uv run projects/pretrain/scaling.py [noise lr batch ...] [--out-dir DIR]`

Reads every finished run under `<repo>/runs/pretrain` that follows the scaling protocol of scaling_runs.py (constant LR
after warmup, weight EMA, head size 64, context 2048, window attention with a document mask). With a constant LR the
first k steps of a run do not depend on its planned length, so each run is a whole loss-vs-compute curve: `Run.at(c)` is
its val loss (val_doc_loss, whole stories from their start) after `c` training FLOPs, interpolated between eval rows. The
analyses take every number from there:

- `noise`: seeds of the same cell, and a long run against a shorter one (the eval-row-is-a-short-run claim).
- `lr`: the best LR of each (width, depth, batch) group at a compute checkpoint, from a parabola in log LR through the
  three runs around the lowest loss (an optimum at the edge of the LR range is flagged, not interpolated), and how it
  scales with batch, width, depth and compute.
- `batch`: loss at the best LR against batch size, and the critical batch size: optimizer steps S needed to reach a target
  loss at batch B follow S = S_min (1 + B_crit / B), a line in 1/B whose intercept is S_min and slope S_min * B_crit.

- `lrlaw`: one regression for the best LR of every group at once. Around its best LR a group's loss is a parabola in log LR,
  and with a power law for the best LR, LR* = LR0 (width/320)^a (depth/5)^b (batch/32)^c, the loss is linear in x = log2 LR,
  x^2 and x times (1, log width, log depth, log batch), with one free offset per group. The run-to-run noise (about 0.03
  between seeds) is much bigger than the dip of a parabola through three runs, so this pools every run instead of
  fitting each group alone; error bars are a jackknife over groups.
- `iso`: IsoFLOP profiles. At each compute c the loss of the tuned runs against parameter count N (a parabola in log N
  through the five shapes around the lowest) gives the compute-optimal N*(c) and loss L*(c), and the power laws of both.
- `fit`: the parametric law L(N, D) = E + A N^-alpha + B D^-beta fitted to every eval row of the tuned runs, error bars
  from a bootstrap over runs, the compute-optimal N* and D* it implies, and each shape's residual (a shape effect
  beyond N shows up there).

- `holdout`: the test of `fit`. The law is fitted on runs of at most `--holdout-from` FLOPs only, and its prediction for
  each longer run (the `holdout` phase of scaling_runs.py: three shapes at 1e17) is printed next to what the run reached; with
  the exponent beta fixed at other values too, since the data pin beta down only loosely (see the `fit` printout).

Figures and a results.json go to `--out-dir` (default runs/pretrain/scaling), the fits are printed.
"""

import json
import math
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from llama import Config, build_model
from tyro.conf import Positional

from zoo.paths import runs_root
from zoo.plot import plt
from zoo.runs import load_run

RUNS_ROOT = runs_root("pretrain")
BLOCK = 2048
REF = dict(dim=320, layers=5, batch=32)  # the shape of the first sweep: the other axes are varied around it


@dataclass
class Args:
    analyses: Positional[list[Literal["noise", "lr", "batch", "lrlaw", "iso", "fit", "holdout"]]] = field(default_factory=lambda: ["noise", "lr", "batch", "lrlaw", "iso", "fit", "holdout"])  # which analyses to run
    metric: Literal["val_doc_loss", "val_loss"] = "val_doc_loss"  # loss to fit; val_doc_loss is comparable across document options
    iso_batch: int = 32  # global batch of the runs used for the IsoFLOP profiles and the parametric fit
    min_budget: float = 1e16  # shortest budget of a run that counts as a tuned run for iso and fit
    fit_max_budget: float = 1e17  # longest budget of a run used by iso and fit (the default is every run)
    holdout_from: float = 3e16  # holdout: fit on runs of at most this budget and predict the longer ones
    fit_min_tokens: float = 6e7  # eval rows before this many training tokens (the warmup and its aftermath) are left out of the parametric fit
    bootstraps: int = 100
    out_dir: Path = RUNS_ROOT / "scaling"


@dataclass
class Run:
    name: str
    dim: int
    layers: int
    batch: int  # global, sequences per optimizer step
    lr: float
    seed: int
    budget: float
    params: int
    fpt: float  # training FLOPs per token
    flops: np.ndarray  # at each eval row
    loss: np.ndarray

    @property
    def group(self) -> tuple[int, int, int]:
        return (self.dim, self.layers, self.batch)

    def at(self, c: float) -> float:
        """Val loss after `c` training FLOPs: interpolated between eval rows in log FLOPs, nan outside the run."""
        if c > self.flops[-1] * 1.01 or c < self.flops[0] / 1.01:  # a budget is rounded to whole steps
            return math.nan
        return float(np.interp(math.log(c), np.log(self.flops), self.loss))


def in_protocol(c: Config) -> bool:
    return (c.min_lr_ratio == 1.0 and c.ema and c.ema_frac == 0.1 and c.block_size == BLOCK and c.attn_window == 512 and c.attn_global_every == 5
            and c.doc_mask and c.doc_align == "off" and c.flop_budget > 0 and c.time_budget_s <= 0 and c.n_heads * 64 == c.dim)


@cache
def count_params(dim: int, layers: int) -> int:
    cfg = Config(dim=dim, n_layers=layers, n_heads=dim // 64, block_size=BLOCK, doc_mask=True, attn_window=512, attn_global_every=5)
    return sum(p.numel() for p in build_model(cfg).parameters())


def load_runs(metric: str) -> list[Run]:
    runs = []
    for d in sorted(RUNS_ROOT.iterdir()):
        if not (d / "results.json").exists() or not (d / "config.json").exists():
            continue
        r = load_run(d)
        cfg = Config.from_dict(r["config"])
        m = r["metrics"]
        if not in_protocol(cfg) or metric not in m:
            continue
        rows = [i for i, v in enumerate(m[metric]) if math.isfinite(v)]
        if len(rows) < 2:  # diverged, or a single eval row
            continue
        flops = np.array([m["flops"][i] for i in rows]) if "flops" in m else np.array([m["tokens"][i] for i in rows]) * cfg.train_flops_per_token
        runs.append(Run(d.name, cfg.dim, cfg.n_layers, cfg.batch_size * cfg.grad_accum, cfg.lr, cfg.seed, cfg.flop_budget,
                        count_params(cfg.dim, cfg.n_layers), cfg.train_flops_per_token, flops, np.array([m[metric][i] for i in rows])))
    return runs


def one_per_cell(runs: list[Run]) -> dict[tuple, Run]:
    """Seed 0 only, and of runs with the same (shape, batch, lr) the one with the shortest budget (the same trajectory, so
    its last eval row is a real one at that budget)."""
    cells: dict[tuple, Run] = {}
    for r in sorted(runs, key=lambda r: r.budget):
        if r.seed == 0:
            cells.setdefault((*r.group, r.lr), r)
    return cells


def parabola_min(x: np.ndarray, y: np.ndarray) -> tuple[float, float, bool]:
    """Minimum of the parabola through the lowest point of (x, y) and its two neighbours: (x*, y*, at_edge). The lowest
    point at either end of the range (or a parabola that opens down) is returned as it is, flagged."""
    i = int(np.argmin(y))
    if i in (0, len(y) - 1):
        return float(x[i]), float(y[i]), True
    a, b, c = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
    if a <= 0:
        return float(x[i]), float(y[i]), True
    xs = float(np.clip(-b / (2 * a), x[i - 1], x[i + 1]))
    return xs, float(c - b * b / (4 * a)), False


def lr_optimum(cells: dict[tuple, Run], group: tuple, c: float) -> tuple[float, float, bool, int] | None:
    """(best LR, loss there, at_edge, runs used) of a group at compute `c`, or None with fewer than three LRs."""
    pts = sorted((r.lr, r.at(c)) for (*g, _), r in cells.items() if tuple(g) == group)
    pts = [(lr, v) for lr, v in pts if math.isfinite(v)]
    if len(pts) < 3:
        return None
    x, y = np.log2([p[0] for p in pts]), np.array([p[1] for p in pts])
    xs, ys, edge = parabola_min(x, y)
    return 2 ** xs, ys, edge, len(pts)


def slope(x: list[float], y: list[float]) -> tuple[float, float] | None:
    """Slope of log2 y against log2 x with its standard error, or None with fewer than three points."""
    if len(x) < 3:
        return None
    lx, ly = np.log2(x), np.log2(y)
    (m, b), cov = np.polyfit(lx, ly, 1, cov=True) if len(x) > 2 else (np.polyfit(lx, ly, 1), None)
    return float(m), float(math.sqrt(cov[0, 0]))


def horizons(cells: dict[tuple, Run]) -> dict[tuple, float]:
    """The compute every run of a group has reached: its shortest budget (a model needs a budget where it is trained)."""
    h: dict[tuple, float] = {}
    for key, r in cells.items():
        h[key[:3]] = min(h.get(key[:3], math.inf), r.flops[-1])
    return h


def fmt_lr(x: float) -> str:
    return f"{x:.2e}"


def run_noise(runs: list[Run], out: dict) -> None:
    print("\n== noise: the same cell with different seeds, and a long run against a short one ==")
    cells: dict[tuple, list[Run]] = {}
    for r in runs:
        cells.setdefault((*r.group, r.lr), []).append(r)
    for key, rs in cells.items():
        seeds = {r.seed: r for r in sorted(rs, key=lambda r: r.budget)}  # shortest budget per seed
        if len(seeds) < 2:
            continue
        print(f"d{key[0]} L{key[1]} bs{key[2]} lr{key[3]:g}, seeds {sorted(seeds)}: val loss at")
        for c in (2e15, 4e15, 1e16):
            v = np.array([r.at(c) for _, r in sorted(seeds.items())])
            if np.isfinite(v).all():
                print(f"  {c:.0e} FLOPs  mean {v.mean():.4f}  std {v.std(ddof=1):.4f}  range {v.max() - v.min():.4f}  by seed: " + "  ".join(f"{x:.4f}" for x in v))
                out.setdefault("noise_std", {})[f"{c:.0e}"] = float(v.std(ddof=1))
    for key, rs in cells.items():
        for seed in {r.seed for r in rs}:
            same = sorted((r for r in rs if r.seed == seed), key=lambda r: r.budget)
            if len(same) >= 2:
                short, long = same[0], same[-1]
                if long.budget > short.budget:
                    c = short.budget
                    print(f"seed {seed}, d{key[0]} L{key[1]} bs{key[2]} lr{key[3]:g}: {short.name} gives {short.at(c):.4f} at {c:.0e} FLOPs, the same point in {long.name} (budget {long.budget:.0e}) {long.at(c):.4f}")


def run_lr(runs: list[Run], args: Args, out: dict) -> None:
    print("\n== lr: best LR of each (width, depth, batch) group at its own budget (loss against LR, in the order of the LRs) ==")
    cells = one_per_cell(runs)
    horizon = horizons(cells)
    groups = sorted({k[:3] for k in cells})
    rows = []
    for g in groups:
        o = lr_optimum(cells, g, horizon[g])
        lrs = sorted(k[3] for k in cells if k[:3] == g)
        if o is None:
            continue
        rows.append((g, *o))
        curve = " ".join(f"{cells[(*g, lr)].at(horizon[g]):.3f}" for lr in lrs)
        print(f"  d{g[0]:<4} L{g[1]:<3} bs{g[2]:<4} at {horizon[g]:.0e}: best lr {fmt_lr(o[0])}  loss {o[1]:.4f}  {'AT THE EDGE of ' if o[2] else 'between '}{len(lrs)} runs {lrs[0]:g}..{lrs[-1]:g}:  {curve}")
    out["lr_best"] = {f"d{g[0]}-L{g[1]}-bs{g[2]}": dict(lr=lr, loss=loss, edge=edge) for g, lr, loss, edge, _ in rows}
    good = [r for r in rows if not r[3]]  # an edge optimum is only a bound on the real one
    print("  power laws, best LR against one factor with the others at the reference (d320 L5 bs32); a slope of -1 is LR ~ 1/factor")
    fams = {"batch": (lambda g: (g[0], g[1]) == (REF["dim"], REF["layers"]), 2), "width": (lambda g: (g[1], g[2]) == (REF["layers"], REF["batch"]), 0),
            "depth": (lambda g: (g[0], g[2]) == (REF["dim"], REF["batch"]), 1)}
    for name, (sel, idx) in fams.items():
        fam = [r for r in good if sel(r[0])]
        s = slope([r[0][idx] for r in fam], [r[1] for r in fam])
        print(f"    LR* vs {name}: " + (f"slope {s[0]:+.2f} +- {s[1]:.2f}  (n={len(fam)})" if s else f"needs 3 groups with an interior optimum (has {len(fam)})"))
        if s:
            out.setdefault("lr_slopes", {})[name] = dict(slope=s[0], se=s[1], n=len(fam))

    ref = (REF["dim"], REF["layers"], REF["batch"])
    print("  best LR of the reference group against compute:")
    cs = [c for c in (2e15, 4e15, 1e16, 2e16) if c <= max(r.flops[-1] for r in cells.values() if r.group == ref)]
    ref_pts = [(c, lr_optimum(cells, ref, c)) for c in cs]
    for c, o in ref_pts:
        if o:
            print(f"    {c:.0e}  lr* {fmt_lr(o[0])}  loss {o[1]:.4f}{'  (edge)' if o[2] else ''}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for g in groups:
        pts = sorted((k[3], r.at(horizon[g])) for k, r in cells.items() if k[:3] == g)
        pts = [p for p in pts if math.isfinite(p[1])]
        if len(pts) >= 2:
            ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"d{g[0]} L{g[1]} bs{g[2]}")
    for g, lr, loss, edge, _ in rows:
        ax[0].plot([lr], [loss], marker="*", ms=14, mfc="none", mec="k", ls="none")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("learning rate")
    ax[0].set_ylabel(f"{args.metric} at the end of each group's runs")
    ymin = min(r[2] for r in rows) if rows else 0
    ax[0].set_ylim(ymin * 0.99, max(r[2] for r in rows) * 1.12)
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3)
    for a, (name, (sel, idx)) in zip(ax[1:], [("batch", fams["batch"]), ("width", fams["width"])]):
        fam = sorted([r for r in rows if sel(r[0])], key=lambda r: r[0][idx])
        a.plot([r[0][idx] for r in fam], [r[1] for r in fam], "o-")
        for r in fam:
            if r[3]:
                a.plot([r[0][idx]], [r[1]], "o", mfc="white", mec="C0", ms=9)  # open marker: only a bound
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel({"batch": "global batch (sequences)", "width": "width"}[name])
        a.set_ylabel("best learning rate")
        a.grid(alpha=0.3, which="both")
    ax[1].set_title("best LR vs batch (open: at the edge of the sweep)")
    ax[2].set_title("best LR vs width")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_dir / "lr.png", dpi=200)
    plt.close(fig)


def run_batch(runs: list[Run], args: Args, out: dict) -> None:
    ref = (REF["dim"], REF["layers"])
    print(f"\n== batch: loss at the best LR against batch size, and the critical batch size (d{ref[0]} L{ref[1]}) ==")
    cells = one_per_cell([r for r in runs if (r.dim, r.layers) == ref])
    batches = sorted({k[2] for k in cells})
    grid = [2e15, 4e15, 6e15, 8e15, 1e16]
    best = {}  # (batch, c) -> loss at the best LR
    for b in batches:
        for c in grid:
            o = lr_optimum(cells, (*ref, b), c)
            if o:
                best[b, c] = o[1]
    bs = [b for b in batches if all((b, c) in best for c in grid)]
    print("  best-LR loss    " + "".join(f"{c:>9.0e}" for c in grid))
    for b in bs:
        print(f"  bs{b:<4}          " + "".join(f"{best[b, c]:>9.4f}" for c in grid))
    if len(bs) < 3:
        print("  needs three batch sizes with three LRs each")
        return
    curves = {b: np.array([best[b, c] for c in grid]) for b in bs}  # decreasing with compute
    # Target losses from just above what the small batches reach to just below where the early part of the curves is; each
    # target uses only the batch sizes whose curve passes through it (a large batch that never gets there has no step count).
    lo = min(v.min() for v in curves.values()) + 0.02
    hi = float(np.median([v.max() for v in curves.values()])) - 0.02
    targets = np.linspace(lo, hi, 6) if hi > lo else []
    print("  critical batch size: steps S to reach a target loss, S = S_min (1 + B_crit/B), a line in 1/B (the batch sizes that reach the target)")
    crit = []
    for t in targets:
        used = [b for b in bs if curves[b].min() <= t <= curves[b].max()]
        if len(used) < 3:
            continue
        steps = []
        for b in used:
            c_t = math.exp(np.interp(t, curves[b][::-1], np.log(grid)[::-1]))
            steps.append(c_t / cells[(*ref, b, sorted(k[3] for k in cells if k[:3] == (*ref, b))[0])].fpt / (b * BLOCK))
        m, s0 = np.polyfit(1 / np.array(used), steps, 1)  # S = s0 + m / B
        crit.append((t, s0, m / s0 if s0 > 0 else math.nan, steps, used))
        print(f"    target {t:.3f}: S_min {s0:7.0f} steps, B_crit {m / s0:6.1f} sequences ({m / s0 * BLOCK / 1e3:.0f}k tokens);  steps at B=" + " ".join(f"{b}:{s:.0f}" for b, s in zip(used, steps)))
    if not crit:
        print("  no target loss is reached by three batch sizes")
    out["critical_batch"] = [dict(target=t, s_min=s0, b_crit=bc, batches=used) for t, s0, bc, _, used in crit]
    if len(crit) >= 3 and all(math.isfinite(c[2]) and c[2] > 0 for c in crit):
        m = np.polyfit(np.log([c[0] for c in crit]), np.log([c[2] for c in crit]), 1)[0]
        print(f"  B_crit ~ loss^{m:.1f} over this range (B_crit grows as the loss falls when the exponent is negative)")
        out["critical_batch_exponent_in_loss"] = float(m)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for c in grid:
        ax[0].plot(bs, [best[b, c] for b in bs], "o-", label=f"{c:.0e} FLOPs")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("global batch (sequences)")
    ax[0].set_ylabel(f"{args.metric} at the best LR")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for t, s0, bc, steps, used in crit:
        ax[1].plot(used, steps, "o-", label=f"target {t:.3f}")
        xs = np.geomspace(min(used), max(used), 50)
        ax[1].plot(xs, s0 * (1 + bc / xs), "k:", lw=0.8)
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("global batch (sequences)")
    ax[1].set_ylabel("optimizer steps to the target loss")
    if crit:
        ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    ax[2].plot([c[0] for c in crit], [c[2] for c in crit], "o-")
    ax[2].set_xlabel("target loss")
    ax[2].set_ylabel("critical batch size (sequences)")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "batch.png", dpi=200)
    plt.close(fig)


def tuned_runs(runs: list[Run], args: Args) -> list[Run]:
    """One run per (width, depth) at the IsoFLOP batch size: seed 0, budget from `min_budget` to `fit_max_budget`, and where several
    LRs were run (the reference shape) the one with the lowest loss at the shortest such budget; of runs with the same LR
    (the same trajectory) the longest."""
    groups: dict[tuple, list[Run]] = {}
    for r in runs:
        if r.seed == 0 and r.batch == args.iso_batch and args.min_budget <= r.budget <= args.fit_max_budget * 1.01:
            groups.setdefault((r.dim, r.layers), []).append(r)
    tuned = []
    for rs in groups.values():
        horizon = min(r.budget for r in rs)
        best = min(rs, key=lambda r: (r.at(horizon) if math.isfinite(r.at(horizon)) else math.inf, -r.budget))
        tuned.append(max((r for r in rs if r.lr == best.lr), key=lambda r: r.budget))
    return sorted(tuned, key=lambda r: r.params)


def power_fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """y = k x^m by least squares in logs: (m, standard error of m, k)."""
    (m, b), cov = np.polyfit(np.log(x), np.log(y), 1, cov=True)
    return float(m), float(math.sqrt(cov[0, 0])), float(math.exp(b))


def lstsq_nonneg(cols: list[np.ndarray], y: np.ndarray) -> tuple[float, np.ndarray]:
    """Least squares with every coefficient >= 0 for a few columns: the best feasible fit over all subsets of the columns,
    from the normal equations. Returns (residual sum of squares, coefficients)."""
    k = len(cols)
    G = np.array([[a @ b for b in cols] for a in cols])
    h = np.array([a @ y for a in cols])
    best = (math.inf, np.zeros(k))
    for mask in range(1, 2 ** k):
        idx = [i for i in range(k) if mask >> i & 1]
        try:
            c = np.linalg.solve(G[np.ix_(idx, idx)], h[idx])
        except np.linalg.LinAlgError:
            continue
        if (c < 0).any():
            continue
        rss = float(y @ y - 2 * c @ h[idx] + c @ G[np.ix_(idx, idx)] @ c)
        if rss < best[0]:
            full = np.zeros(k)
            full[idx] = c
            best = (rss, full)
    return best


def fit_chinchilla(n_m: np.ndarray, d_b: np.ndarray, loss: np.ndarray, beta: float | None = None) -> dict:
    """L = E + A N^-alpha + B D^-beta, with N in millions of parameters and D in billions of tokens (so A and B are the loss
    terms at 1M parameters and 1B tokens). For fixed exponents the model is linear in (E, A, B), so a grid over (alpha, beta)
    with a nonnegative least squares at each point, refined once around the best, finds the fit. `beta` fixes that exponent."""
    one = np.ones_like(loss)

    def at(alpha, beta):
        rss, c = lstsq_nonneg([one, n_m ** -alpha, d_b ** -beta], loss)
        return rss, c

    def grid(centre, lo, hi, k, fixed):
        return [fixed] if fixed is not None else centre * np.geomspace(lo, hi, k)

    best = (math.inf, 0.0, 0.0, None)
    for alpha in np.geomspace(0.03, 2.0, 40):
        for b in grid(1.0, 0.03, 2.0, 40, beta):
            rss, c = at(alpha, b)
            if rss < best[0]:
                best = (rss, alpha, b, c)
    _, a0, b0, _ = best
    for alpha in a0 * np.geomspace(0.8, 1.25, 17):
        for b in grid(b0, 0.8, 1.25, 17, beta):
            rss, c = at(alpha, b)
            if rss < best[0]:
                best = (rss, alpha, b, c)
    rss, alpha, beta, c = best
    return dict(E=float(c[0]), A=float(c[1]), B=float(c[2]), alpha=float(alpha), beta=float(beta), rms=math.sqrt(rss / len(loss)))


def optimal_split(fit: dict, k_m_b: float) -> tuple[float, float, float]:
    """Parameters N* (millions), tokens D* (billions) and loss L* that minimise A N^-alpha + B D^-beta with N * D = k (in
    millions x billions), the closed form of the usual constrained optimum."""
    a, b, A, B = fit["alpha"], fit["beta"], fit["A"], fit["B"]
    n = (a * A / (b * B)) ** (1 / (a + b)) * k_m_b ** (b / (a + b))
    d = k_m_b / n
    return n, d, fit["E"] + A * n ** -a + B * d ** -b


def predict(fit: dict, n_m: float, d_b: float) -> float:
    return fit["E"] + fit["A"] * n_m ** -fit["alpha"] + fit["B"] * d_b ** -fit["beta"]


def run_iso(runs: list[Run], args: Args, out: dict) -> list[Run]:
    tuned = tuned_runs(runs, args)
    print(f"\n== iso: compute-optimal model size from {len(tuned)} tuned runs (batch {args.iso_batch}, seed 0) ==")
    if len(tuned) < 5:
        print("  needs five shapes")
        return tuned
    print("  shapes (width x depth, parameters, LR, budget): " + "; ".join(f"{r.dim}x{r.layers} {r.params / 1e6:.1f}M lr{r.lr:g} {r.budget:.0e}" for r in tuned))
    top = max(r.flops[-1] for r in tuned)
    cs = [c for c in np.geomspace(3e14, top, 14) if sum(math.isfinite(r.at(c)) for r in tuned) >= 5]
    rows = []
    for c in cs:
        pts = sorted((r.params, r.at(c), r) for r in tuned if math.isfinite(r.at(c)))
        x, y = np.log([p[0] for p in pts]), np.array([p[1] for p in pts])
        j = int(np.argmin(y))
        lo = min(max(j - 2, 0), len(y) - 5)  # five points around the lowest, shifted at the ends
        sl = slice(lo, lo + 5)
        a, b, c0 = np.polyfit(x[sl], y[sl], 2)
        vertex = -b / (2 * a) if a > 0 else math.nan
        edge = j in (0, len(y) - 1) or not (x[sl][0] <= vertex <= x[sl][-1])  # a vertex outside the five points is an extrapolation
        xs = float(np.clip(vertex, x[sl][0], x[sl][-1])) if math.isfinite(vertex) else float(x[j])
        rows.append((c, math.exp(xs), float(np.polyval([a, b, c0], xs)) if math.isfinite(vertex) else float(y[j]), edge, len(pts), pts))
        print(f"  {c:.1e} FLOPs: N* {math.exp(xs) / 1e6:6.2f}M  loss {rows[-1][2]:.4f}  tokens/param {c / (6 * math.exp(xs)) / math.exp(xs):.0f}{'  (at the edge of the shapes: only a bound)' if edge else ''}  [{len(pts)} shapes]")
    good = [r for r in rows if not r[3]]
    if len(good) >= 4:
        m, se, k = power_fit([r[0] for r in good], [r[1] for r in good])
        print(f"  N*(C) ~ C^{m:.2f} +- {se:.2f}  (the rows are not independent: they share the same runs, so the error bar is optimistic)")
        out["iso"] = dict(n_exponent=m, n_exponent_se=se, n_star=[dict(c=r[0], n=r[1], loss=r[2]) for r in good])
        cc, ll = np.array([r[0] for r in good]), np.array([r[2] for r in good])
        best = None
        for al in np.geomspace(0.02, 1.0, 120):  # L*(C) = E + A C^-al, linear in (E, A) for each al
            rss, coef = lstsq_nonneg([np.ones_like(cc), (cc / 1e15) ** -al], ll)
            if best is None or rss < best[0]:
                best = (rss, al, coef)
        print(f"  L*(C) = {best[2][0]:.3f} + {best[2][1]:.3f} (C/1e15)^-{best[1]:.3f}")
        out["iso"]["loss_curve"] = dict(E=float(best[2][0]), A=float(best[2][1]), alpha=float(best[1]))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    cmap = plt.get_cmap("viridis")
    for i, (c, ns, ls, edge, _, pts) in enumerate(rows):
        col = cmap(i / max(len(rows) - 1, 1))
        ax[0].plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], "o-", ms=3, color=col, label=f"{c:.0e}")
        ax[0].plot([ns / 1e6], [ls], marker="v", color=col, ms=8, mec="k")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("parameters (millions)")
    ax[0].set_ylabel(f"{args.metric}")
    lo_y = min(r[2] for r in rows)
    ax[0].set_ylim(lo_y * 0.97, min(lo_y * 1.5, max(p[1] for r in rows for p in r[5])))
    ax[0].legend(title="FLOPs", fontsize=6, ncol=2)
    ax[0].grid(alpha=0.3)
    ax[1].plot([r[0] for r in rows if not r[3]], [r[1] / 1e6 for r in rows if not r[3]], "o-")
    ax[1].plot([r[0] for r in rows if r[3]], [r[1] / 1e6 for r in rows if r[3]], "o", mfc="white", mec="C0")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("training FLOPs")
    ax[1].set_ylabel("compute-optimal parameters (millions)")
    ax[1].grid(alpha=0.3, which="both")
    ax[2].plot([r[0] for r in rows], [r[2] for r in rows], "o-")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("training FLOPs")
    ax[2].set_ylabel("loss of the best shape")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "iso.png", dpi=200)
    plt.close(fig)
    return tuned


def eval_points(tuned: list[Run], min_tokens: float) -> dict[str, np.ndarray]:
    cols: dict[str, list] = {k: [] for k in ("n", "d", "loss", "run", "flops")}
    for i, r in enumerate(tuned):
        tokens = r.flops / r.fpt
        keep = tokens >= min_tokens
        cols["n"] += [r.params / 1e6] * int(keep.sum())
        cols["d"] += list(tokens[keep] / 1e9)
        cols["loss"] += list(r.loss[keep])
        cols["run"] += [i] * int(keep.sum())
        cols["flops"] += list(r.flops[keep])
    return {k: np.array(v) for k, v in cols.items()}


def run_fit(runs: list[Run], args: Args, out: dict, tuned: list[Run] | None = None) -> None:
    tuned = tuned if tuned is not None else tuned_runs(runs, args)
    print(f"\n== fit: L(N, D) = E + A N^-alpha + B D^-beta on the eval rows of {len(tuned)} tuned runs ==")
    if len(tuned) < 5:
        print("  needs five shapes")
        return
    pts = eval_points(tuned, args.fit_min_tokens)
    fit = fit_chinchilla(pts["n"], pts["d"], pts["loss"])
    rho = float(np.median([r.fpt / (6 * r.params) for r in tuned]))  # training FLOPs per token over 6 N: attention and the output projection on top of 6 N
    boots = []
    rng = np.random.default_rng(0)
    for _ in range(args.bootstraps):
        pick = rng.integers(0, len(tuned), len(tuned))
        sel = np.concatenate([np.flatnonzero(pts["run"] == i) for i in pick])
        f = fit_chinchilla(pts["n"][sel], pts["d"][sel], pts["loss"][sel])
        boots.append(f)

    def ci(key):
        v = np.array([b[key] for b in boots])
        return float(np.percentile(v, 16)), float(np.percentile(v, 84))

    print(f"  E {fit['E']:.3f}  A {fit['A']:.3f}  alpha {fit['alpha']:.3f}  B {fit['B']:.3f}  beta {fit['beta']:.3f}   (rms residual {fit['rms']:.4f}, {len(pts['loss'])} rows)")
    print(f"  68% bootstrap over runs ({args.bootstraps}): alpha {ci('alpha')[0]:.3f}..{ci('alpha')[1]:.3f}  beta {ci('beta')[0]:.3f}..{ci('beta')[1]:.3f}  E {ci('E')[0]:.3f}..{ci('E')[1]:.3f}")
    ex = lambda f: f["beta"] / (f["alpha"] + f["beta"])  # noqa: E731
    exs = np.array([ex(b) for b in boots])
    print(f"  compute-optimal split: N* ~ C^{ex(fit):.2f} (68%: {np.percentile(exs, 16):.2f}..{np.percentile(exs, 84):.2f}), D* ~ C^{1 - ex(fit):.2f}   (FLOPs = {6 * rho:.2f} N D here)")
    print("  compute  N* (M)  D* (B)  tokens/param  predicted loss")
    frontier = []
    for c in (1e15, 3e15, 1e16, 3e16, 1e17, 1e18):
        n, d, l = optimal_split(fit, c / (6 * rho) / 1e6 / 1e9)
        frontier.append(dict(c=c, n_m=n, d_b=d, loss=l))
        print(f"  {c:7.0e}  {n:6.2f}  {d:6.2f}  {d * 1e3 / n:11.0f}  {l:.4f}{'   (extrapolated beyond the runs)' if c > 1.5 * max(r.flops[-1] for r in tuned) else ''}")
    out["fit"] = dict(**fit, rho=rho, n_exponent=ex(fit), n_exponent_p16=float(np.percentile(exs, 16)), n_exponent_p84=float(np.percentile(exs, 84)), frontier=frontier, max_budget=args.fit_max_budget,
                      ci={k: ci(k) for k in ("alpha", "beta", "E")})

    # How well do the data pin the exponents down? Refit with beta fixed: if the rms hardly moves, beta is not identified, and
    # the compute-optimal split (the exponent of N*, and N* at 1e17) moves with it while the predicted loss hardly does.
    print("  identifiability: beta fixed, everything else refitted")
    print("    beta    rms  alpha     E   N* exponent  N* at 1e17 (M)  tokens/param  loss at 1e17")
    prof = []
    for b in (0.3, 0.5, 0.7, 1.0, 1.5):
        f = fit_chinchilla(pts["n"], pts["d"], pts["loss"], beta=b)
        n, d, l = optimal_split(f, 1e17 / (6 * rho) / 1e15)
        prof.append(dict(f, n_exponent=ex(f), n_star_m=n, loss_1e17=l))
        print(f"    {b:4.2f}  {f['rms']:.4f}  {f['alpha']:.3f}  {f['E']:.3f}  {ex(f):11.2f}  {n:14.1f}  {d * 1e3 / n:12.0f}  {l:12.3f}")
    late = eval_points(tuned, 3 * args.fit_min_tokens)
    f = fit_chinchilla(late["n"], late["d"], late["loss"])
    print(f"    rows from {3 * args.fit_min_tokens:.0e} tokens on only: alpha {f['alpha']:.3f} beta {f['beta']:.3f} E {f['E']:.3f}  N* exponent {ex(f):.2f}  ({len(late['loss'])} rows)")
    out["fit"]["beta_profile"] = prof

    c_ref = min(1e16, min(r.flops[-1] for r in tuned))
    print(f"  residuals at {c_ref:.0e} FLOPs (observed - fitted); a pattern with width/depth is a shape effect beyond N:")
    print("    shape     N (M)  width/depth  observed  fitted  residual")
    res = []
    for r in tuned:
        obs = r.at(c_ref)
        if math.isfinite(obs):
            pred = fit["E"] + fit["A"] * (r.params / 1e6) ** -fit["alpha"] + fit["B"] * (c_ref / r.fpt / 1e9) ** -fit["beta"]
            res.append((r, obs, pred))
            print(f"    {r.dim:>4}x{r.layers:<3} {r.params / 1e6:6.2f}  {r.dim / r.layers:11.0f}  {obs:8.4f}  {pred:6.4f}  {obs - pred:+.4f}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    pred_all = fit["E"] + fit["A"] * pts["n"] ** -fit["alpha"] + fit["B"] * pts["d"] ** -fit["beta"]
    ax[0].plot(pts["loss"], pred_all, ".", ms=2, alpha=0.4)
    lim = [pts["loss"].min(), np.percentile(pts["loss"], 97)]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)
    ax[0].set_xlabel("observed loss")
    ax[0].set_ylabel("fitted loss")
    ax[1].scatter([r.dim / r.layers for r, *_ in res], [o - p for _, o, p in res], c=[math.log10(r.params) for r, *_ in res])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("width / depth")
    ax[1].set_ylabel(f"residual at {c_ref:.0e} FLOPs")
    ax[1].grid(alpha=0.3)
    cs = np.geomspace(1e14, 1e19, 60)
    ns = [optimal_split(fit, c / (6 * rho) / 1e15)[2] for c in cs]
    ax[2].plot(cs, ns, "k-", label="fitted frontier")
    for r in tuned:
        ax[2].plot(r.flops, r.loss, lw=0.8, alpha=0.7)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("training FLOPs")
    ax[2].set_ylabel(args.metric)
    ax[2].set_ylim(lim[0] * 0.98, lim[1])
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "fit.png", dpi=200)
    plt.close(fig)


def run_holdout(runs: list[Run], args: Args, out: dict) -> None:
    """Predict the runs beyond `holdout_from` from a fit on the shorter ones. The planned cells of the `holdout` phase are
    listed before they are run, so that the prediction is on record first."""
    try:
        from scaling_runs import PHASES, model_config
        planned = [(c.dim, c.layers, c.lr, c.budget) for c in PHASES.get("holdout", [])]
    except ImportError:
        planned, model_config = [], None
    seen = {(r.dim, r.layers, r.budget): r for r in runs if r.seed == 0 and r.batch == args.iso_batch and r.budget > args.holdout_from * 1.01}
    shapes = sorted({(d, l, lr, c) for d, l, lr, c in planned} | {(r.dim, r.layers, r.lr, r.budget) for r in seen.values()}, key=lambda s: (s[3], count_params(s[0], s[1])))
    if not shapes:
        return
    tuned = tuned_runs(runs, replace(args, fit_max_budget=args.holdout_from))
    print(f"\n== holdout: fit on the {len(tuned)} tuned runs of at most {args.holdout_from:.0e} FLOPs, prediction for longer runs ==")
    pts = eval_points(tuned, args.fit_min_tokens)
    free = fit_chinchilla(pts["n"], pts["d"], pts["loss"])
    fits = {"free": free} | {f"beta={b}": fit_chinchilla(pts["n"], pts["d"], pts["loss"], beta=b) for b in (0.3, 0.5, 1.0)}
    print("  " + "; ".join(f"{k}: alpha {f['alpha']:.2f} beta {f['beta']:.2f} E {f['E']:.2f} rms {f['rms']:.4f}" for k, f in fits.items()))
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(args.bootstraps):
        pick = rng.integers(0, len(tuned), len(tuned))
        sel = np.concatenate([np.flatnonzero(pts["run"] == i) for i in pick])
        boots.append(fit_chinchilla(pts["n"][sel], pts["d"][sel], pts["loss"][sel]))
    print(f"  {'shape':<9}{'budget':>7}{'N (M)':>7}{'tok/par':>8}   " + "".join(f"{k:>12}" for k in fits) + f"  {'free 68%':>13}{'observed':>10}{'obs-free':>9}")
    rows = []
    for dim, layers, lr, c in shapes:
        n = count_params(dim, layers) / 1e6
        fpt = seen[(dim, layers, c)].fpt if (dim, layers, c) in seen else model_config(next(x for x in PHASES["holdout"] if (x.dim, x.layers, x.budget) == (dim, layers, c))).train_flops_per_token
        d = c / fpt / 1e9
        pred = {k: predict(f, n, d) for k, f in fits.items()}
        band = np.percentile([predict(b, n, d) for b in boots], [16, 84])
        run = seen.get((dim, layers, c))
        obs = run.at(c) if run is not None else math.nan
        rows.append(dict(dim=dim, layers=layers, lr=lr, budget=c, n_m=n, d_b=d, pred=pred, band=band.tolist(), observed=obs))
        print(f"  {dim:>4}x{layers:<4}{c:>7.0e}{n:>7.1f}{d * 1e3 / n:>8.0f}   " + "".join(f"{v:>12.3f}" for v in pred.values()) + f"  {band[0]:>6.3f}..{band[1]:.3f}"
              + (f"{obs:>10.4f}{obs - pred['free']:>+9.3f}" if math.isfinite(obs) else f"{'(not run)':>10}"))
    out["holdout"] = dict(fit_budget=args.holdout_from, rows=rows)


def run_lrlaw(runs: list[Run], args: Args, out: dict) -> None:
    print(f"\n== lrlaw: LR* = LR0 (width/320)^a (depth/5)^b (batch/32)^c, fitted jointly, each group read at its own budget ==")
    cells = one_per_cell(runs)
    horizon = horizons(cells)
    groups: dict[tuple, list[tuple[float, float]]] = {}
    for key, r in cells.items():
        v = r.at(horizon[key[:3]])
        if math.isfinite(v):
            groups.setdefault(key[:3], []).append((math.log2(r.lr), v))
    # Keep the points near each group's best (the parabola is only a local model) and groups with enough of them.
    local = {}
    for g, pts in groups.items():
        lo = min(v for _, v in pts)
        near = [(x, v) for x, v in pts if v <= lo + 0.12]
        if len(near) >= 3:
            local[g] = near
    if len(local) < 3:
        print(f"  needs three groups with three LRs near their best (has {len(local)})")
        return
    gs = sorted(local)
    feats = {"width": [math.log2(g[0] / 320) for g in gs], "depth": [math.log2(g[1] / 5) for g in gs], "batch": [math.log2(g[2] / 32) for g in gs]}
    used = [k for k, v in feats.items() if np.ptp(v) > 0]  # a factor that does not vary has no exponent

    def solve(groups_used: list[tuple]) -> np.ndarray | None:
        rows, y = [], []
        idx = {g: i for i, g in enumerate(groups_used)}
        ng = len(groups_used)
        f = {k: {g: feats[k][gs.index(g)] for g in groups_used} for k in used}
        for g in groups_used:
            for x, v in local[g]:
                row = np.zeros(ng + 2 + len(used))
                row[idx[g]] = 1
                row[ng] = x * x
                row[ng + 1] = -2 * x
                for j, k in enumerate(used):
                    row[ng + 2 + j] = -2 * x * f[k][g]
                rows.append(row)
                y.append(v)
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(y), rcond=None)
        kappa = sol[ng]
        if kappa <= 0:
            return None
        return np.concatenate([[kappa], sol[ng + 1:] / kappa])  # kappa, log2 LR0, one exponent per used factor

    full = solve(gs)
    if full is None:
        print("  the loss is not convex in log LR in these groups")
        return
    jack = [s for s in (solve([g for g in gs if g != h]) for h in gs) if s is not None]
    jm = np.array(jack)
    se = np.sqrt((len(jm) - 1) / len(jm) * ((jm - jm.mean(0)) ** 2).sum(0)) if len(jm) > 2 else np.full(len(full), math.nan)
    print(f"  {len(gs)} groups, {sum(len(v) for v in local.values())} runs; curvature {full[0]:.4f} loss per octave^2")
    print(f"  LR0 (d320 L5 bs32) = {2 ** full[1]:.2e}  (jackknife over groups: times or divided by {2 ** se[1]:.2f})")
    law = {"lr0": float(2 ** full[1]), "curvature": float(full[0])}
    for j, k in enumerate(used):
        print(f"  exponent of {k:<5}: {full[2 + j]:+.2f} +- {se[2 + j]:.2f}   (LR ~ {k}^{full[2 + j]:+.2f})")
        law[f"{k}_exponent"] = float(full[2 + j])
        law[f"{k}_exponent_se"] = float(se[2 + j])
    out["lrlaw"] = law
    fig, ax = plt.subplots(1, max(len(used), 1), figsize=(5.2 * max(len(used), 1), 4.6), squeeze=False)
    for a, k in zip(ax[0], used):
        # each group's own best LR (parabola, 3 points) against the law
        xs = []
        for g in gs:
            pts = sorted(local[g])
            xv, yv = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            xo, _, edge = parabola_min(xv, yv)
            xs.append((feats[k][gs.index(g)], xo, edge))
        a.plot([2 ** p[0] * {"width": 320, "depth": 5, "batch": 32}[k] for p in xs], [2 ** p[1] for p in xs], "o")
        grid = np.linspace(min(p[0] for p in xs), max(p[0] for p in xs), 20)  # the other factors at the reference
        a.plot(2 ** grid * {"width": 320, "depth": 5, "batch": 32}[k], 2 ** (full[1] + full[2 + used.index(k)] * grid), "k--", label=f"exponent {full[2 + used.index(k)]:+.2f}")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(k)
        a.set_ylabel("best learning rate")
        a.legend()
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "lrlaw.png", dpi=200)
    plt.close(fig)


def main(args: Args):
    runs = load_runs(args.metric)
    if not runs:
        raise SystemExit("no finished runs that follow the scaling protocol")
    print(f"{len(runs)} runs in the protocol: " + ", ".join(sorted({f"d{r.dim}L{r.layers}" for r in runs})))
    out: dict = {}
    if "noise" in args.analyses:
        run_noise(runs, out)
    if "lr" in args.analyses:
        run_lr(runs, args, out)
    if "batch" in args.analyses:
        run_batch(runs, args, out)
    if "lrlaw" in args.analyses:
        run_lrlaw(runs, args, out)
    tuned = None
    if "iso" in args.analyses:
        tuned = run_iso(runs, args, out)
    if "fit" in args.analyses:
        run_fit(runs, args, out, tuned)
    if "holdout" in args.analyses:
        run_holdout(runs, args, out)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "results.json"
    if path.exists():  # a run of some analyses only must not erase the others' results
        out = {**json.loads(path.read_text()), **out}
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out_dir}/")


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))

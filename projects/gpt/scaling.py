"""Scaling-law analysis of the sweeps in sweeps/: the learning-rate rule, the compute-optimal fit, the annealing check.

Reads the finished runs of a W&B group from their local files (`<runs>/gpt/<run id>/config.json` and `csv/metrics.csv`), so
it needs no network. With a constant LR after warmup and an EMA whose window is a share of the steps so far, the first k
steps of a run do not depend on how long the run was planned to be: every validation row is the result for a smaller budget
too, and a run is a whole loss-vs-compute curve. `Run.at(c)` reads it at `c` FLOPs.

  fit-lr         the best peak LR at each width (a parabola in log LR through the runs around the lowest), and one rule
                 LR* = LR0 (width / 256)^a for all of them with the flags that set it
  fit-scaling    on a ladder sweep: noise between seeds, IsoFLOP profiles (the compute-optimal size N*(C)), the law
                 L(N, D) = E + A N^-alpha + B D^-beta with error bars and the compute-optimal split it implies, and how
                 well a fit on part of the data predicts the rest
  check-anneal   constant LR with EMA against a run that anneals its LR to zero over the same steps
  fit-shape      at equal compute: depth x width (a fitted surface and the best aspect ratio), MLP size, head size, context
                 length, and whether the width-only LR rule is fair to every depth
  fit-optim      at equal compute: weight decay, AdamW beta2, warmup, tokens per step with its LR, against the shape grid's run

Figures and a results.json go to `--out` (default `<runs>/gpt/scaling/<group>`); the fits are printed.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from config import LR_REF_DIM, AnnealConfig, Config, LRFitConfig, OptimConfig, ScalingConfig, ShapeConfig
from matplotlib.ticker import FuncFormatter

from zoo.paths import runs_root
from zoo.plot import plt

PROJECT = "gpt"


@dataclass
class Run:
    id: str
    cfg: Config
    steps: np.ndarray  # optimizer steps taken at each validation
    tokens: np.ndarray
    flops: np.ndarray
    loss: np.ndarray  # val/loss, nan after a run diverged
    pos: dict[str, np.ndarray] = field(default_factory=dict)  # val/loss_pos_<from>_<to> of a run with `val_by_pos`

    def n(self, which: str) -> float:
        return float(self.cfg.n_params if which == "total" else self.cfg.n_params_body)

    @property
    def label(self) -> str:
        return f"L{self.cfg.n_layers}x{self.cfg.dim}"

    def at(self, c: float, col: str | None = None) -> float:
        """Val loss after `c` FLOPs (of the position range `col` of a `val_by_pos` run): interpolated between validations in
        log FLOPs, nan outside the run."""
        values = self.loss if col is None else self.pos[col]
        ok = np.isfinite(self.loss) & np.isfinite(values)
        if not ok.any() or c > self.flops[ok][-1] * 1.01 or c < self.flops[ok][0] / 1.01:  # a budget is rounded to whole steps
            return math.nan
        return float(np.interp(math.log(c), np.log(self.flops[ok]), values[ok]))

    def trim(self, min_tokens: float) -> Run | None:
        """The run's validation rows from `min_tokens` tokens on (None when it has none)."""
        keep = self.tokens >= min_tokens
        if not keep.any():
            return None
        return Run(self.id, self.cfg, self.steps[keep], self.tokens[keep], self.flops[keep], self.loss[keep], {k: v[keep] for k, v in self.pos.items()})

    def at_steps(self, s: int) -> float:
        hit = self.loss[self.steps == s]
        return float(hit[0]) if len(hit) else math.nan


def load_runs(group: str, min_tokens: float = 0.0) -> list[Run]:
    """The finished runs of a W&B group, with the validation rows from `min_tokens` tokens on."""
    runs, unfinished = [], []
    for d in sorted(runs_root(PROJECT).iterdir()):
        cfg_file, csv_file = d / "config.json", d / "csv" / "metrics.csv"
        if not (cfg_file.exists() and csv_file.exists()):
            continue
        cfg = Config.from_dict(json.loads(cfg_file.read_text())["config"])
        if cfg.group != group:
            continue
        with csv_file.open() as f:
            rows = [r for r in csv.DictReader(f) if r.get("val/loss") not in (None, "")]
        steps = np.array([int(float(r["step"])) + 1 for r in rows])
        if not len(steps) or steps[-1] < cfg.max_steps:  # still running, or crashed
            unfinished.append(d.name)
            continue
        tokens = np.array([float(r["val/tokens_seen"]) if r.get("val/tokens_seen") else s * cfg.global_batch for r, s in zip(rows, steps)])
        flops = np.array([float(r["val/flops"]) if r.get("val/flops") else t * cfg.train_flops_per_token for r, t in zip(rows, tokens)])
        loss = np.array([float(r["val/loss"]) for r in rows])
        keep = tokens >= min_tokens
        pos_keys = sorted((k for k in rows[0] if k.startswith("val/loss_pos_")), key=lambda k: tuple(int(x) for x in k[len("val/loss_pos_"):].split("_")))  # the CSV's columns are alphabetical
        pos = {k: np.array([float(r[k]) if r.get(k) not in (None, "") else math.nan for r in rows])[keep] for k in pos_keys}
        if keep.any():
            runs.append(Run(d.name, cfg, steps[keep], tokens[keep], flops[keep], loss[keep], pos))
    if unfinished:
        print(f"(skipped {len(unfinished)} unfinished runs of {group}: {', '.join(unfinished)})")
    return runs


def out_dir(path: Path | None, group: str) -> Path:
    d = path or runs_root(PROJECT) / "scaling" / group
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_results(d: Path, out: dict) -> None:
    path = d / "results.json"
    if path.exists():  # a run of one analysis must not erase another's results
        out = {**json.loads(path.read_text()), **out}
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {d}/")


def _log_label(v: float, _pos=None) -> str:
    """A tick label for a log axis: plain numbers at 1, 2 and 5 of each decade, nothing in between (crowded minor labels overlap)."""
    if v <= 0:
        return ""
    mantissa = v / 10 ** math.floor(math.log10(v))
    return f"{v:g}" if round(mantissa, 6) in (1, 2, 5) else ""


def save_fig(fig, path: Path) -> None:
    for ax in fig.axes:
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            lo, hi = axis.get_view_interval()
            if scale == "log" and hi / lo >= 8:  # a narrower range keeps matplotlib's labels: there are too few ticks to thin out
                axis.set_major_formatter(FuncFormatter(_log_label))
                axis.set_minor_formatter(FuncFormatter(_log_label))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- numerics

def parabola_min(x: np.ndarray, y: np.ndarray) -> tuple[float, float, bool]:
    """Minimum of the parabola through the lowest point of (x, y) and its two neighbours: (x*, y*, at_edge). The lowest
    point at either end of the range (or a parabola that opens down) is returned as it is, flagged."""
    i = int(np.argmin(y))
    if i in (0, len(y) - 1):
        return float(x[i]), float(y[i]), True
    a, b, c = np.polyfit(x[i - 1 : i + 2], y[i - 1 : i + 2], 2)
    if a <= 0:
        return float(x[i]), float(y[i]), True
    xs = float(np.clip(-b / (2 * a), x[i - 1], x[i + 1]))
    return xs, float(c - b * b / (4 * a)), False


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
    for mask in range(1, 2**k):
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

    def at(alpha, b):
        return lstsq_nonneg([one, n_m**-alpha, d_b**-b], loss)

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


def optimal_split(fit: dict, k: float) -> tuple[float, float, float]:
    """Parameters N* (millions), tokens D* (billions) and loss L* that minimise A N^-alpha + B D^-beta with N * D = k (in
    millions x billions): the closed form of the usual constrained optimum."""
    a, b, A, B = fit["alpha"], fit["beta"], fit["A"], fit["B"]
    n = (a * A / (b * B)) ** (1 / (a + b)) * k ** (b / (a + b))
    d = k / n
    return n, d, fit["E"] + A * n**-a + B * d**-b


def predict(fit: dict, n_m, d_b):
    return fit["E"] + fit["A"] * np.asarray(n_m) ** -fit["alpha"] + fit["B"] * np.asarray(d_b) ** -fit["beta"]


# ---------------------------------------------------------------- fit-lr

def lr_optimum(pts: list[tuple[float, float]]) -> tuple[float, float, bool]:
    """(best LR, loss there, at_edge) of a width from its (lr, loss) runs, by `parabola_min` in log2 LR."""
    pts = sorted(pts)
    xs, ys, edge = parabola_min(np.log2([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return 2**xs, ys, edge


def lr_law(groups: dict[int, list[tuple[float, float]]], near: float) -> dict | None:
    """One power law LR*(width) = LR0 (width / 256)^a fitted to every run at once. Around its best LR a width's loss is a
    parabola in x = log2 LR, so with log2 LR* = x0 + a log2(width / 256) the loss is linear in x^2, x and x times log2(width / 256),
    with one free offset per width. Pooling the runs beats a parabola through three runs per width when the loss is noisy;
    the error bars are a jackknife over widths (rough with only a few of them)."""
    local = {}
    for dim, pts in groups.items():
        lo = min(v for _, v in pts)
        near_pts = [(math.log2(lr), v) for lr, v in pts if v <= lo + near]  # the parabola is only a local model
        if len(near_pts) >= 3:
            local[dim] = near_pts
    dims = sorted(local)
    if len(dims) < 2:
        return None

    def solve(ds: list[int]) -> np.ndarray | None:
        rows, y = [], []
        for i, d in enumerate(ds):
            f = math.log2(d / LR_REF_DIM)
            for x, v in local[d]:
                row = np.zeros(len(ds) + 3)
                row[i], row[len(ds)], row[len(ds) + 1], row[len(ds) + 2] = 1, x * x, -2 * x, -2 * x * f
                rows.append(row)
                y.append(v)
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(y), rcond=None)
        kappa = sol[len(ds)]
        return None if kappa <= 0 else np.array([kappa, sol[len(ds) + 1] / kappa, sol[len(ds) + 2] / kappa])  # kappa, log2 LR0, a

    full = solve(dims)
    if full is None:
        return None
    jack = [s for s in (solve([d for d in dims if d != h]) for h in dims) if s is not None]
    jm = np.array(jack)
    se = np.sqrt((len(jm) - 1) / len(jm) * ((jm - jm.mean(0)) ** 2).sum(0)) if len(jm) > 2 else np.full(3, math.nan)
    return dict(lr0=float(2 ** full[1]), a=float(full[2]), kappa=float(full[0]), lr0_factor_se=float(2 ** se[1]), a_se=float(se[2]), widths=dims)


def run_fit_lr(args: LRFitConfig) -> None:
    runs = load_runs(args.group)
    if not runs:
        raise SystemExit(f"no finished runs in group {args.group!r}")
    diverged = [r for r in runs if not math.isfinite(r.loss[-1])]
    groups: dict[int, list[tuple[float, float]]] = {}
    layers: dict[int, int] = {}
    for r in runs:
        if math.isfinite(r.loss[-1]):
            groups.setdefault(r.cfg.dim, []).append((r.cfg.peak_lr, float(r.loss[-1])))
            layers[r.cfg.dim] = r.cfg.n_layers
    steps = runs[0].steps[-1]
    print(f"== lr: {len(runs)} runs of {args.group}, loss after {steps:,} steps ({runs[0].tokens[-1] / 1e6:.0f}M tokens)"
          + (f"; diverged: {', '.join(f'{r.label} lr {r.cfg.peak_lr:.1e}' for r in diverged)}" if diverged else "") + " ==")
    best, out = {}, {}
    for dim in sorted(groups):
        pts = sorted(groups[dim])
        lr, loss, edge = lr_optimum(pts)
        best[dim] = (lr, loss, edge)
        print(f"  width {dim:>4} ({layers[dim]} layers): best lr {lr:.2e}  loss {loss:.4f}  {'AT THE EDGE of the grid (a bound only)' if edge else 'between the runs'}"
              f"   [{', '.join(f'{p[0]:.1e}: {p[1]:.3f}' for p in pts)}]")
    out["best_lr"] = {str(d): dict(lr=v[0], loss=v[1], edge=v[2]) for d, v in best.items()}
    law = lr_law(groups, args.near)
    if law is None:
        print("  no rule: it needs two widths with three runs each near their best")
    else:
        print(f"\n  rule (all runs pooled): LR* = {law['lr0']:.2e} * (width / {LR_REF_DIM}) ^ {law['a']:+.2f}"
              f"   (jackknife over {len(law['widths'])} widths: LR0 x or / {law['lr0_factor_se']:.2f}, exponent +- {law['a_se']:.2f}; curvature {law['kappa']:.3f} loss per octave^2)")
        lr0, p = law["lr0"], -law["a"]
        print(f"  flags for the ladder: --lr {lr0:.2e} --lr-width-exp {p:.2f}     bracket (x0.7, x1.4): lr {0.7 * lr0:.2e} / {lr0:.2e} / {1.4 * lr0:.2e}")
        out["rule"] = dict(law, lr=lr0, lr_width_exp=p)
    d = out_dir(args.out, args.group)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, dim in enumerate(sorted(groups)):
        pts = sorted(groups[dim])
        floor = min(p[1] for p in pts)  # each width relative to its own best run, so the curves overlay
        ax[0].plot([p[0] for p in pts], [p[1] - floor for p in pts], "o-", color=f"C{i}", label=f"width {dim} (best {floor:.3f})")
        ax[0].plot([best[dim][0]], [best[dim][1] - floor], marker="*", ms=14, mfc="none", mec=f"C{i}", ls="none")
    ax[0].set(xscale="log", xlabel="peak learning rate", ylabel=f"val loss after {steps:,} steps, above the width's best run", ylim=(-0.02, args.near * 2))
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    dims = np.array(sorted(best))
    ax[1].plot(dims, [best[x][0] for x in dims], "o", label="parabola minimum per width")
    if law:
        xs = np.geomspace(dims.min() / 1.3, dims.max() * 1.3, 20)
        ax[1].plot(xs, law["lr0"] * (xs / LR_REF_DIM) ** law["a"], "k--", label=f"rule, exponent {law['a']:+.2f}")
    ax[1].set(xscale="log", yscale="log", xlabel="width", ylabel="best learning rate")
    ax[1].legend()
    ax[1].grid(alpha=0.3, which="both")
    save_fig(fig, d / "lr.png")
    save_results(d, out)


# ---------------------------------------------------------------- fit-scaling

def tuned_runs(runs: list[Run], rule_lr: float | None = None) -> list[Run]:
    """One run per model size: seed 0, and of the learning rates run at that size the one with the lowest final loss (with `rule_lr`: only the runs of that base LR, the sweep's rule)."""
    by_size: dict[tuple[int, int], list[Run]] = {}
    for r in runs:
        if rule_lr is not None and not math.isclose(r.cfg.lr, rule_lr, rel_tol=1e-6):
            continue
        if r.cfg.seed == 0 and np.isfinite(r.loss).any():
            by_size.setdefault((r.cfg.dim, r.cfg.n_layers), []).append(r)
    tuned = [min(rs, key=lambda r: r.loss[-1] if math.isfinite(r.loss[-1]) else math.inf) for rs in by_size.values()]
    return sorted(tuned, key=lambda r: r.cfg.n_params)


def eval_points(tuned: list[Run], which: str) -> dict[str, np.ndarray]:
    cols: dict[str, list] = {k: [] for k in ("n", "d", "loss", "run", "flops")}
    for i, r in enumerate(tuned):
        ok = np.isfinite(r.loss)
        cols["n"] += [r.n(which) / 1e6] * int(ok.sum())
        cols["d"] += list(r.tokens[ok] / 1e9)
        cols["loss"] += list(r.loss[ok])
        cols["run"] += [i] * int(ok.sum())
        cols["flops"] += list(r.flops[ok])
    return {k: np.array(v) for k, v in cols.items()}


def run_noise(runs: list[Run], out: dict) -> None:
    cells: dict[tuple, dict[int, Run]] = {}
    for r in runs:
        cells.setdefault((r.cfg.dim, r.cfg.n_layers, round(r.cfg.peak_lr, 9)), {})[r.cfg.seed] = r
    print("\n== noise: the same size and learning rate with different seeds ==")
    found = False
    for (dim, layers, lr), seeds in sorted(cells.items()):
        if len(seeds) < 2:
            continue
        found = True
        last = min(r.steps[-1] for r in seeds.values())
        print(f"  L{layers}x{dim} lr {lr:.1e}, seeds {sorted(seeds)}: val loss at")
        for s in sorted({2000, 5000, 10000, int(last)}):
            v = np.array([r.at_steps(s) for r in seeds.values()])
            if np.isfinite(v).all():
                print(f"    step {s:>6}  mean {v.mean():.4f}  std {v.std(ddof=1):.4f}  range {v.max() - v.min():.4f}")
                out.setdefault("noise_std", {})[f"L{layers}x{dim}@{s}"] = float(v.std(ddof=1))
    if not found:
        print("  no size has two seeds")


def run_bracket(runs: list[Run], out: dict, d: Path, rule_lr: float) -> None:
    """The sizes that ran at more than one peak LR (the bracket sweep around the rule's): the loss at each rate, so a rule that is off at
    the ends of the sizes shows (x is the LR over the rule's at that width). `tuned_runs` then takes each size's best, which is what the fit uses."""
    cells: dict[tuple[int, int], dict[float, Run]] = {}
    for r in runs:
        if r.cfg.seed == 0 and np.isfinite(r.loss).any():
            by_lr = cells.setdefault((r.cfg.n_layers, r.cfg.dim), {})
            if r.cfg.peak_lr not in by_lr or len(r.steps) > len(by_lr[r.cfg.peak_lr].steps):  # the longest run at an LR: a shorter one is its prefix
                by_lr[r.cfg.peak_lr] = r
    print("\n== learning-rate bracket: the sizes that ran at more than one peak LR (seed 0), val loss at two steps (x = the LR over the rule's at that width) ==")
    multi = {k: v for k, v in cells.items() if len(v) >= 2}
    if not multi:
        print("  no size has two learning rates")
        return
    panels: dict[str, list] = {"10,000 steps": [], "the full run": []}  # (label, LR over the rule's, loss above the best cell)
    for (n_l, dim), rs in sorted(multi.items()):
        lrs = sorted(rs)
        ends = sorted((int(r.steps[-1]) for r in rs.values()), reverse=True)
        last = ends[1]  # the longest length that at least two of the LRs reached (a half-length cell does not hide the full-length ones)
        rule_peak = rule_lr * LR_REF_DIM / dim
        for s in sorted({10000, last}):
            if s > last:
                continue
            pts = [(lr, rs[lr].at_steps(s)) for lr in lrs if math.isfinite(rs[lr].at_steps(s))]
            if len(pts) < 2:
                continue
            txt = ", ".join(f"{lr:.1e} (x{lr / rule_peak:.2g}): {v:.4f}" for lr, v in pts)
            if len(pts) >= 3:
                best, _, edge = lr_optimum(pts)
                txt += f"   best LR {best:.1e} (x{best / rule_peak:.2g})" + (" (at the edge of the bracket: only a bound)" if edge else "")
            print(f"  L{n_l}x{dim}, step {s:>6,}: {txt}")
            out.setdefault("bracket", {})[f"L{n_l}x{dim}@{s}"] = {f"{lr:.3e}": v for lr, v in pts}
            top = min(v for _, v in pts)
            tokens_per_param = s * rs[lrs[0]].cfg.global_batch / rs[lrs[0]].n("total")
            panels["10,000 steps" if s == 10000 else "the full run"].append((f"L{n_l}x{dim} ({s:,} steps, {tokens_per_param:.0f} tokens/param)", [lr / rule_peak for lr, _ in pts], [v - top for _, v in pts]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, series) in zip(axes, panels.items()):
        for i, (label, xs, ys) in enumerate(series):
            ax.plot(xs, ys, "o-", color=f"C{i}", label=label)
        ax.axvline(1.0, color="0.5", ls=":", lw=1)
        ax.set(xscale="log", xlabel="peak LR over the rule's (dotted: the rule)", title=title)
        ax.grid(alpha=0.3)
        if series:
            ax.legend(fontsize=8)
    axes[0].set_ylabel("val loss above the best LR of that size")
    save_fig(fig, d / "bracket.png")


def run_iso(tuned: list[Run], which: str, rho: float, d: Path, out: dict) -> None:
    print(f"\n== iso: IsoFLOP profiles of {len(tuned)} sizes (N = {which} parameters) ==")
    if len(tuned) < 5:
        print("  needs five sizes")
        return
    top = max(r.flops[-1] for r in tuned)
    cs = [c for c in np.geomspace(1e14, top, 40) if sum(math.isfinite(r.at(c)) for r in tuned) >= 5]
    if not cs:
        print("  no compute budget has a validation row for five sizes at once (the small sizes stop at one epoch before the large ones start)")
        return
    rows = []
    for c in cs:
        pts = sorted((r.n(which), r.at(c)) for r in tuned if math.isfinite(r.at(c)))
        x, y = np.log([p[0] for p in pts]), np.array([p[1] for p in pts])
        j = int(np.argmin(y))
        lo = min(max(j - 2, 0), len(y) - 5)  # five points around the lowest, shifted at the ends
        sl = slice(lo, lo + 5)
        a, b, c0 = np.polyfit(x[sl], y[sl], 2)
        vertex = -b / (2 * a) if a > 0 else math.nan
        edge = j in (0, len(y) - 1) or not (math.isfinite(vertex) and x[sl][0] <= vertex <= x[sl][-1])  # a vertex outside the five points is an extrapolation
        xs = float(np.clip(vertex, x[sl][0], x[sl][-1])) if math.isfinite(vertex) else float(x[j])
        rows.append((c, math.exp(xs), float(np.polyval([a, b, c0], xs)) if math.isfinite(vertex) else float(y[j]), edge, pts))
    for c, n, loss, edge, pts in rows:
        print(f"  {c:.1e} FLOPs: N* {n / 1e6:6.2f}M  loss {loss:.4f}  tokens/param {c / (6 * rho * n) / n:.0f}{'  (the minimum is at the edge of the sizes: only a bound)' if edge else ''}  [{len(pts)} sizes]")
    good = [r for r in rows if not r[3]]
    if len(good) >= 4:
        m, se, _ = power_fit([r[0] for r in good], [r[1] for r in good])
        print(f"  N*(C) ~ C^{m:.2f} +- {se:.2f}  (the rows share the same runs, so the error bar is optimistic)")
        out["iso"] = dict(n_exponent=m, n_exponent_se=se, n_star=[dict(c=r[0], n=r[1], loss=r[2]) for r in good])
    else:
        print(f"  only {len(good)} contours have their minimum inside the sizes: no power law for N*(C)")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, (c, n, loss, edge, pts) in enumerate(rows):
        col = cmap(i / max(len(rows) - 1, 1))
        ax[0].plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], "o-", ms=3, color=col, label=f"{c:.1e}")
        ax[0].plot([n / 1e6], [loss], marker="v", color=col, ms=8, mec="k", mfc="none" if edge else col)
    ax[0].set(xscale="log", xlabel=f"{which} parameters (millions)", ylabel="val loss")
    lo = min(r[2] for r in rows)
    ax[0].set_ylim(lo * 0.98, lo * 1.4)
    ax[0].legend(title="FLOPs", fontsize=6, ncol=2)
    ax[0].grid(alpha=0.3)
    ax[1].plot([r[0] for r in good], [r[1] / 1e6 for r in good], "o-")
    ax[1].plot([r[0] for r in rows if r[3]], [r[1] / 1e6 for r in rows if r[3]], "o", mfc="white", mec="C0")
    ax[1].set(xscale="log", yscale="log", xlabel="training FLOPs", ylabel="compute-optimal parameters (millions)")
    ax[1].grid(alpha=0.3, which="both")
    save_fig(fig, d / "iso.png")


def run_fit(tuned: list[Run], which: str, rho: float, args: ScalingConfig, d: Path, out: dict, full: list[Run] | None = None) -> dict:
    print(f"\n== fit: L(N, D) = E + A N^-alpha + B D^-beta on the validation rows of {len(tuned)} sizes (N = {which} parameters) ==")
    pts = eval_points(tuned, which)
    fit = fit_chinchilla(pts["n"], pts["d"], pts["loss"])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(args.bootstraps):
        pick = rng.integers(0, len(tuned), len(tuned))
        sel = np.concatenate([np.flatnonzero(pts["run"] == i) for i in pick])
        boots.append(fit_chinchilla(pts["n"][sel], pts["d"][sel], pts["loss"][sel]))

    def ci(key):
        v = np.array([b[key] for b in boots])
        return float(np.percentile(v, 16)), float(np.percentile(v, 84))

    ex = lambda f: f["beta"] / (f["alpha"] + f["beta"])  # noqa: E731
    exs = np.array([ex(b) for b in boots])
    print(f"  E {fit['E']:.3f}  A {fit['A']:.3f}  alpha {fit['alpha']:.3f}  B {fit['B']:.3f}  beta {fit['beta']:.3f}   (rms residual {fit['rms']:.4f}, {len(pts['loss'])} rows)")
    print(f"  68% bootstrap over sizes ({args.bootstraps}): alpha {ci('alpha')[0]:.3f}..{ci('alpha')[1]:.3f}  beta {ci('beta')[0]:.3f}..{ci('beta')[1]:.3f}  E {ci('E')[0]:.3f}..{ci('E')[1]:.3f}")
    print(f"  compute-optimal split: N* ~ C^{ex(fit):.2f} (68%: {np.percentile(exs, 16):.2f}..{np.percentile(exs, 84):.2f}), D* ~ C^{1 - ex(fit):.2f}   (FLOPs = {6 * rho:.2f} N D here)")
    top = max(r.flops[-1] for r in tuned)
    print("  compute   N* (M)  D* (B)  tokens/param  predicted loss")
    frontier = []
    for c in (1e15, 3e15, 1e16, 3e16, 1e17, 3e17, 1e18):
        n, db, loss = optimal_split(fit, c / (6 * rho) / 1e15)
        frontier.append(dict(c=c, n_m=n, d_b=db, loss=loss))
        print(f"  {c:7.0e}  {n:6.2f}  {db:6.2f}  {db * 1e3 / n:11.0f}  {loss:.4f}{'   (extrapolated beyond the runs)' if c > 1.5 * top else ''}")
    out[f"fit_{which}"] = dict(**fit, rho=rho, n_exponent=ex(fit), n_exponent_p16=float(np.percentile(exs, 16)), n_exponent_p84=float(np.percentile(exs, 84)),
                               frontier=frontier, ci={k: ci(k) for k in ("alpha", "beta", "E")})
    # How well do the data pin the exponents down? Refit with beta fixed: if the rms hardly moves, beta is not identified and the
    # compute-optimal split moves with it while the predicted loss hardly does.
    print("  identifiability: beta fixed, everything else refitted")
    print("    beta    rms  alpha     E   N* exponent  N* at 1e17 (M)  tokens/param  loss at 1e17")
    prof = []
    for b in (0.3, 0.5, 0.7, 1.0, 1.5):
        f = fit_chinchilla(pts["n"], pts["d"], pts["loss"], beta=b)
        n, db, loss = optimal_split(f, 1e17 / (6 * rho) / 1e15)
        prof.append(dict(f, n_exponent=ex(f), n_star_m=n, loss_1e17=loss))
        print(f"    {b:4.2f}  {f['rms']:.4f}  {f['alpha']:.3f}  {f['E']:.3f}  {ex(f):11.2f}  {n:14.1f}  {db * 1e3 / n:12.0f}  {loss:12.3f}")
    out[f"fit_{which}"]["beta_profile"] = prof

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    pred = predict(fit, pts["n"], pts["d"])
    ax[0].plot(pts["loss"], pred, ".", ms=3, alpha=0.5)
    lim = [pts["loss"].min(), np.percentile(pts["loss"], 97)]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set(xlim=lim, ylim=lim, xlabel="observed val loss", ylabel="fitted val loss")
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(tuned):
        col = cmap(i / max(len(tuned) - 1, 1))
        m = pts["run"] == i
        ax[1].plot(pts["d"][m] * 1e3, pts["loss"][m] - pred[m], color=col, label=f"{r.label} ({r.n(which) / 1e6:.1f}M)")
        ax[2].plot(r.flops, r.loss, color=col, lw=1)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xscale="log", xlabel="tokens (millions)", ylabel="observed - fitted (a pattern with size or tokens is a misfit)")
    ax[1].legend(fontsize=6)
    ax[1].grid(alpha=0.3)
    cs = np.geomspace(max(1e14, min(r.flops[0] for r in tuned)), 1e18, 60)
    ax[2].plot(cs, [optimal_split(fit, c / (6 * rho) / 1e15)[2] for c in cs], "k--", label="fitted frontier")
    ax[2].set(xscale="log", yscale="log", xlabel="training FLOPs", ylabel="val loss", ylim=(lim[0] * 0.97, lim[1]))
    ax[2].legend()
    ax[2].grid(alpha=0.3, which="both")
    save_fig(fig, d / "fit.png")
    # the plot of the runs against compute, with the envelope they make and the frontier the law implies
    fig, ax = plt.subplots(figsize=(7.5, 5))
    whole = {r.id: r for r in (full or tuned)}  # every row of each run: the fit leaves out the early ones, the plot does not
    for i, r in enumerate(tuned):
        col = cmap(i / max(len(tuned) - 1, 1))
        ax.plot(whole[r.id].flops, whole[r.id].loss, color=col, lw=0.7, alpha=0.5)
        ax.plot(r.flops, r.loss, color=col, lw=1.6, label=f"{r.label} ({r.n(which) / 1e6:.1f}M)")  # thick: the rows the fit used
    grid = np.geomspace(min(r.flops[0] for r in whole.values()), top, 80)
    env = [min((r.at(c) for r in whole.values() if math.isfinite(r.at(c))), default=math.nan) for c in grid]
    ax.plot(grid, env, "k-", lw=2.2, label="best of the runs")
    ax.plot(grid, [optimal_split(fit, c / (6 * rho) / 1e15)[2] for c in grid], "r--", lw=1.5, label="fitted frontier")
    ax.set(xscale="log", yscale="log", xlabel="training FLOPs", ylabel="val loss", ylim=(lim[0] * 0.97, lim[1]))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    save_fig(fig, d / "curves.png")
    return fit


def run_cuts(tuned_all: list[Run], which: str, rho: float, cuts: tuple[float, ...], out: dict) -> None:
    """The same fit with the early rows cut at different token counts: how much the exponents and the compute-optimal split depend on
    where the early transient is taken to end. A size drops out of a cut it never reaches (the largest sizes stop first)."""
    print("\n== early-training cut: the same fit on the validation rows from this many tokens on ==")
    print("   from (tokens)  sizes  rows     rms  alpha   beta      E  N* exponent  N* at 1e17 (M)  loss at 1e17")
    rows = []
    for cut in cuts:
        runs = [x for x in (r.trim(cut) for r in tuned_all) if x and len(x.loss) >= 3]
        if len(runs) < 5:
            print(f"   {cut:13.3g}  fewer than five sizes have three rows past it")
            continue
        p = eval_points(runs, which)
        f = fit_chinchilla(p["n"], p["d"], p["loss"])
        n, db, loss = optimal_split(f, 1e17 / (6 * rho) / 1e15)
        ex = f["beta"] / (f["alpha"] + f["beta"])
        print(f"   {cut:13.3g}  {len(runs):>5}  {len(p['loss']):>4}  {f['rms']:.4f}  {f['alpha']:.3f}  {f['beta']:.3f}  {f['E']:.3f}  {ex:11.2f}  {n:14.1f}  {loss:12.3f}")
        rows.append(dict(cut=cut, sizes=len(runs), rows=len(p["loss"]), alpha=f["alpha"], beta=f["beta"], E=f["E"], rms=f["rms"], n_exponent=ex, n_star_1e17_m=n, loss_1e17=loss))
    out["cut_sensitivity"] = rows


def run_holdout(tuned: list[Run], which: str, holdout_tokens: float, out: dict) -> None:
    """Predict the part of the data a fit has not seen: the two largest sizes (extrapolating in N), and the rows past `holdout_tokens`
    tokens of every run that goes beyond them (extrapolating in D). A mean error near zero and a worst case within ~0.01 mean the law
    can be trusted that far."""
    print("\n== holdout: a fit on part of the data predicting the rest (prediction - observed, in loss) ==")
    res = {}
    if len(tuned) >= 6:
        rest, big = tuned[:-2], tuned[-2:]
        p = eval_points(rest, which)
        fit = fit_chinchilla(p["n"], p["d"], p["loss"])
        for r in big:
            q = eval_points([r], which)
            err = predict(fit, q["n"], q["d"]) - q["loss"]
            print(f"  without the two largest sizes: {r.label} ({r.n(which) / 1e6:.1f}M): mean {err.mean():+.4f}  worst {np.abs(err).max():.4f}  at the last row {err[-1]:+.4f}")
            res[f"largest_{r.label}"] = dict(mean=float(err.mean()), worst=float(np.abs(err).max()), last=float(err[-1]))
    else:
        print("  the size holdout needs six sizes")
    def part(r: Run, keep: np.ndarray) -> Run:
        return Run(r.id, r.cfg, r.steps[keep], r.tokens[keep], r.flops[keep], r.loss[keep])

    early = [part(r, r.tokens <= holdout_tokens) for r in tuned]
    beyond = [r for r in tuned if (r.tokens > holdout_tokens).sum() >= 3]
    if beyond and all(len(r.loss) >= 3 for r in early):
        p = eval_points(early, which)
        fit = fit_chinchilla(p["n"], p["d"], p["loss"])
        errs = []
        for r in beyond:
            qp = eval_points([part(r, r.tokens > holdout_tokens)], which)
            errs.append(predict(fit, qp["n"], qp["d"]) - qp["loss"])
        allerr = np.concatenate(errs)
        print(f"  fit on the rows up to {holdout_tokens:.0e} tokens (alpha {fit['alpha']:.2f}, beta {fit['beta']:.2f}, E {fit['E']:.2f}), predicting those beyond (up to {max(r.tokens[-1] for r in beyond) / holdout_tokens:.1f}x further):"
              f" mean {allerr.mean():+.4f}  worst {np.abs(allerr).max():.4f}  [per size: {', '.join(f'{r.label} {e.mean():+.3f}' for r, e in zip(beyond, errs))}]")
        res["beyond_tokens"] = dict(mean=float(allerr.mean()), worst=float(np.abs(allerr).max()))
    else:
        print(f"  the token holdout needs runs with three validation rows past {holdout_tokens:.0e} tokens and three before")
    out["holdout"] = res


def run_fit_scaling(args: ScalingConfig) -> None:
    runs = load_runs(args.group)  # every validation row; each analysis cuts its own early rows
    if not runs:
        raise SystemExit(f"no finished runs in group {args.group!r}")
    d, out = out_dir(args.out, args.group), {}
    print(f"{len(runs)} finished runs in {args.group}; sizes: {', '.join(sorted({f'{r.label} ({r.cfg.n_params / 1e6:.1f}M)' for r in runs}, key=lambda s: float(s.split('(')[1][:-2])))}")
    print(f"the fit uses validation rows from {args.min_tokens:.0e} tokens on, the IsoFLOP profiles from {args.iso_min_tokens:.0e}")
    run_noise(runs, out)
    run_bracket(runs, out, d, args.rule_lr)
    tuned_all = tuned_runs(runs, args.rule_lr if args.lr == "rule" else None)
    tuned = [x for x in (r.trim(args.min_tokens) for r in tuned_all) if x]
    chosen = ", ".join(f"{r.label} lr {r.cfg.peak_lr:.1e}" for r in tuned)
    print(f"\n{'tuned' if args.lr == 'tuned' else 'rule-LR'} runs (seed 0, " + ("the best learning rate of each size" if args.lr == "tuned" else f"base LR {args.rule_lr:g}, the bracket's runs left out") + f"): {chosen}")
    rho = float(np.median([r.cfg.train_flops_per_token / (6 * r.n(args.params)) for r in tuned]))  # training FLOPs per token over 6 N
    spread = [r.cfg.train_flops_per_token / (6 * r.n(args.params)) for r in tuned]
    print(f"FLOPs per token / (6 N): median {rho:.2f}, range {min(spread):.2f}..{max(spread):.2f}" + ("  (not constant: the compute-optimal conversions below are rough)" if max(spread) > 1.25 * min(spread) else ""))
    run_iso([x for x in (r.trim(args.iso_min_tokens) for r in tuned_all) if x], args.params, rho, d, out)
    if len(tuned) < 5:
        print("\nthe fit needs five sizes")
        save_results(d, out)
        return
    run_fit(tuned, args.params, rho, args, d, out, full=tuned_all)
    run_cuts(tuned_all, args.params, rho, args.cuts, out)
    other = "body" if args.params == "total" else "total"
    p = eval_points(tuned, other)
    f2 = fit_chinchilla(p["n"], p["d"], p["loss"])
    rho2 = float(np.median([r.cfg.train_flops_per_token / (6 * r.n(other)) for r in tuned]))
    print(f"\nrobustness, N = {other} parameters instead: alpha {f2['alpha']:.3f}  beta {f2['beta']:.3f}  E {f2['E']:.3f}  rms {f2['rms']:.4f}  (FLOPs per token / 6N spans {min(r.cfg.train_flops_per_token / (6 * r.n(other)) for r in tuned):.2f}..{max(r.cfg.train_flops_per_token / (6 * r.n(other)) for r in tuned):.2f}, median {rho2:.2f})")
    out[f"fit_{other}_summary"] = f2
    run_holdout(tuned, args.params, args.holdout_tokens, out)
    save_results(d, out)


# ---------------------------------------------------------------- check-anneal

def run_check_anneal(args: AnnealConfig) -> None:
    anneal, ladder = load_runs(args.group), load_runs(args.ladder)
    if not anneal or not ladder:
        raise SystemExit(f"needs finished runs in both {args.group!r} ({len(anneal)}) and {args.ladder!r} ({len(ladder)})")
    print(f"== anneal: a constant LR with EMA against an LR cut to zero over the same steps ({args.group} vs {args.ladder}) ==")
    rows = []
    for a in sorted(anneal, key=lambda r: (r.cfg.n_layers, r.cfg.max_steps)):
        match = [r for r in ladder if r.cfg.dim == a.cfg.dim and r.cfg.seed == a.cfg.seed and math.isclose(r.cfg.peak_lr, a.cfg.peak_lr, rel_tol=1e-6)]
        s = a.cfg.max_steps
        const = match[0].at_steps(s) if match else math.nan
        if not math.isfinite(const):
            print(f"  {a.label} {s} steps: no constant-LR run of the same size and learning rate with a validation row there")
            continue
        ann = float(a.loss[-1])
        rows.append((a, s, const, ann))
        print(f"  {a.label:<9}{s:>7} steps ({a.tokens[-1] / 1e6:>4.0f}M tokens)  constant LR + EMA {const:.4f}   annealed {ann:.4f}   annealed - EMA {ann - const:+.4f}")
    out: dict = {"rows": [dict(label=a.label, steps=s, ema=c, annealed=n, delta=n - c) for a, s, c, n in rows]}
    if rows:
        deltas = np.array([n - c for _, _, c, n in rows])
        print(f"\n  annealed - EMA: mean {deltas.mean():+.4f}, range {deltas.min():+.4f}..{deltas.max():+.4f}. If it is small and about the same at every size and length, the constant-LR "
              "curves have the same shape as annealed runs, shifted by that much, and the exponents carry over; if it grows or shrinks with size or tokens, they do not.")
        out["delta_mean"], out["delta_range"] = float(deltas.mean()), [float(deltas.min()), float(deltas.max())]
        d = out_dir(args.out, args.group)
        fig, ax = plt.subplots(figsize=(6, 4))
        for layers in sorted({a.cfg.n_layers for a, *_ in rows}):
            pts = [(s, n - c) for a, s, c, n in rows if a.cfg.n_layers == layers]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"{layers} layers")
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xscale="log", xlabel="steps", ylabel="annealed - constant LR with EMA (val loss)")
        ax.legend()
        ax.grid(alpha=0.3)
        save_fig(fig, d / "anneal.png")
        save_results(d, out)


# ---------------------------------------------------------------- fit-shape

BASE_VARIANT = (0.0, 64, 256)  # (ff_mult, head_dim, block_size) of the ladder's recipe


def _variant(cfg: Config) -> tuple:
    return (cfg.ff_mult, cfg.head_dim, cfg.block_size)


def _loss_at(r: Run, c: float, min_tokens: float, col: str | None = None) -> float:
    """The run's loss at `c` FLOPs, nan if it has not got there or got there with fewer than `min_tokens` tokens."""
    return math.nan if c / r.cfg.train_flops_per_token < min_tokens else r.at(c, col)


def fit_surface(u: np.ndarray, v: np.ndarray, y: np.ndarray) -> dict | None:
    """Loss at one budget as a smooth function of u = ln(N / 1M) and v = ln(aspect / 64): a cubic in u (the compute-optimal size
    makes the profile in N lopsided), a quadratic in v, and their product. Coefficients in the order 1, u, u^2, u^3, v, v^2, uv.
    With fewer than 14 cells the cubic term is dropped (its coefficient is 0)."""
    x = np.stack([np.ones_like(u), u, u**2, u**3, v, v**2, u * v], axis=1)
    if len(y) < x.shape[1] + 2:
        return None
    if len(y) < 14:  # a small grid cannot carry the cubic: leave the u^3 term out (its coefficient is 0), which keeps three degrees of freedom at nine cells
        coef, *_ = np.linalg.lstsq(np.delete(x, 3, axis=1), y, rcond=None)
        coef = np.insert(coef, 3, 0.0)
    else:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return dict(coef=coef, rms=float(np.sqrt(np.mean((x @ coef - y) ** 2))), n=len(y))


def best_log_aspect(coef: np.ndarray, u: float) -> float:
    """ln(aspect / 64) with the lowest loss at size u: nan unless the loss is convex in the aspect."""
    return -(coef[4] + coef[6] * u) / (2 * coef[5]) if coef[5] > 0 else math.nan


def surface_at(coef: np.ndarray, u: float, v: float) -> float:
    """The fitted loss at size u = ln(N / 1M) and aspect v = ln(aspect / 64)."""
    return float(coef[0] + coef[1] * u + coef[2] * u**2 + coef[3] * u**3 + coef[4] * v + coef[5] * v**2 + coef[6] * u * v)


def run_fit_shape(args: ShapeConfig) -> None:
    runs = load_runs(args.group)
    if not runs:
        raise SystemExit(f"no finished runs in group {args.group!r}")
    d, out = out_dir(args.out, args.group), {}
    budgets, ln2 = list(args.flops), math.log(2)
    plain = [r for r in runs if not r.cfg.val_by_pos]
    base = {(r.cfg.n_layers, r.cfg.dim): r for r in plain if r.cfg.seed == 0 and _variant(r.cfg) == BASE_VARIANT}
    val = lambda r, c, col=None: _loss_at(r, c, args.min_tokens, col)  # noqa: E731
    print(f"{len(runs)} finished runs in {args.group}: {len(base)} depth x width cells at the ladder's recipe (MLP 8/3, head size 64, context 256), "
          f"{sum(1 for r in plain if r.cfg.seed and _variant(r.cfg) == BASE_VARIANT)} extra seeds, {sum(1 for r in plain if _variant(r.cfg) != BASE_VARIANT)} MLP / head variants, "
          f"{sum(1 for r in runs if r.cfg.val_by_pos)} context-length runs")
    print(f"compared at {', '.join(f'{c:.1e}' for c in budgets)} FLOPs; a run counts at a budget when it got there with at least {args.min_tokens:.0e} tokens")
    layers, dims = sorted({k[0] for k in base}), sorted({k[1] for k in base})
    surf: dict[float, dict | None] = {}
    rng = np.random.default_rng(0)

    # ---- depth x width
    print("\n== parameters (millions, embedding included) and aspect ratio (width / layers) ==")
    print("  layers \\ width " + "".join(f"{w:>14}" for w in dims))
    for n_l in layers:
        print(f"  {n_l:>14} " + "".join((f"{base[(n_l, w)].cfg.n_params / 1e6:>7.1f}M a{w / n_l:<4.0f}" if (n_l, w) in base else f"{'':>14}") for w in dims))
    for c in budgets:
        print(f"\n== val loss at {c:.1e} FLOPs, depth x width (* = the lowest; - = not reached, or with too few tokens) ==")
        cells = {k: val(r, c) for k, r in base.items()}
        best = min((v for v in cells.values() if math.isfinite(v)), default=math.nan)
        print("  layers \\ width " + "".join(f"{w:>9}" for w in dims))
        for n_l in layers:
            print(f"  {n_l:>14} " + "".join(
                (f"{cells[(n_l, w)]:>8.4f}{'*' if cells[(n_l, w)] == best else ' '}" if math.isfinite(cells.get((n_l, w), math.nan)) else f"{'-':>9}") for w in dims))
        pts = [(math.log(base[k].n("total") / 1e6), math.log(base[k].cfg.dim / base[k].cfg.n_layers / 64), v) for k, v in cells.items() if math.isfinite(v)]
        s = fit_surface(*(np.array(p) for p in zip(*pts))) if pts else None
        surf[c] = s
        if s is None:
            print("  too few cells for a surface")
            continue
        coef, (u_lo, u_hi) = s["coef"], (min(p[0] for p in pts), max(p[0] for p in pts))
        s["u_range"] = (u_lo, u_hi)
        print(f"  surface over {s['n']} cells ({'cubic' if s['n'] >= 14 else 'quadratic'} in ln N, quadratic in ln aspect): rms residual {s['rms']:.4f}; the loss rises {coef[5] * ln2**2:.4f} per octave^2 of aspect away from the best")
        boots = []
        for _ in range(200):
            idx = rng.integers(0, len(pts), len(pts))
            b = fit_surface(*(np.array([pts[i][j] for i in idx]) for j in range(3)))
            if b is not None:
                boots.append(b["coef"])
        out.setdefault("surface", {})[f"{c:.3g}"] = dict(coef=coef.tolist(), rms=s["rms"], n=s["n"], octave2_curvature=float(coef[5] * ln2**2), best_aspect={}, regret_at_64={})
        for n_m in (1, 2, 3, 4, 6, 8, 12, 16):
            u = math.log(n_m)
            if not u_lo - 0.1 <= u <= u_hi + 0.1:
                continue
            v_star = best_log_aspect(coef, u)
            if not math.isfinite(v_star):
                print(f"  N = {n_m}M: the loss is not convex in the aspect there (no interior best)")
                continue
            reg = float(coef[5] * v_star**2)  # the loss at aspect 64 (v = 0) above the loss at the best aspect, at this N
            bv = np.array([best_log_aspect(cf, u) for cf in boots])
            bv = bv[np.isfinite(bv)]
            br = np.array([cf[5] * best_log_aspect(cf, u) ** 2 for cf in boots if cf[5] > 0])
            ci = f" (68%: {64 * math.exp(np.percentile(bv, 16)):.0f}..{64 * math.exp(np.percentile(bv, 84)):.0f}; {len(bv)}/{len(boots)} resamples convex)" if len(bv) > 20 else ""
            print(f"  N = {n_m:>2}M: best aspect {64 * math.exp(v_star):5.0f}{ci}; aspect 64 costs {reg:+.4f}" + (f" (68%: up to {np.percentile(br, 84):.4f})" if len(br) > 20 else ""))
            out["surface"][f"{c:.3g}"]["best_aspect"][f"{n_m}M"] = 64 * math.exp(v_star)
            out["surface"][f"{c:.3g}"]["regret_at_64"][f"{n_m}M"] = reg
        grid_u = np.linspace(u_lo, u_hi, 300)  # the lowest loss at this budget: over the sizes the cells cover, at aspect 64 and at the best aspect of each size
        v_best = np.array([best_log_aspect(coef, u) for u in grid_u])
        if np.isfinite(v_best).all():
            g64 = np.array([surface_at(coef, u, 0.0) for u in grid_u])
            gbest = np.array([surface_at(coef, u, v) for u, v in zip(grid_u, v_best)])
            i64, ib = int(g64.argmin()), int(gbest.argmin())
            edge = " (at the edge of the sizes run: only a bound)" if {i64, ib} & {0, len(grid_u) - 1} else ""
            print(f"  lowest loss: {g64[i64]:.4f} at aspect 64 (N = {math.exp(grid_u[i64]):.1f}M); {gbest[ib]:.4f} at the best shape (N = {math.exp(grid_u[ib]):.1f}M, aspect {64 * math.exp(v_best[ib]):.0f}), "
                  f"so the shape buys {g64[i64] - gbest[ib]:.4f}{edge}")
            out["surface"][f"{c:.3g}"]["lowest"] = dict(aspect64=dict(n_m=float(math.exp(grid_u[i64])), loss=float(g64[i64])),
                                                       best=dict(n_m=float(math.exp(grid_u[ib])), aspect=float(64 * math.exp(v_best[ib])), loss=float(gbest[ib])))

    # ---- seed noise and the ladder
    print("\n== seed noise: the same cell with more seeds ==")
    noisy = False
    for k, r0 in sorted(base.items()):
        seeds = [r for r in plain if (r.cfg.n_layers, r.cfg.dim) == k and _variant(r.cfg) == BASE_VARIANT and r.cfg.peak_lr == r0.cfg.peak_lr]
        if len(seeds) < 2:
            continue
        noisy = True
        parts = []
        for c in budgets:
            v = np.array([val(r, c) for r in seeds])
            if np.isfinite(v).all():
                parts.append(f"{c:.1e}: std {v.std(ddof=1):.4f} (range {v.max() - v.min():.4f})")
                out.setdefault("seed_std", {})[f"{r0.label}@{c:.3g}"] = float(v.std(ddof=1))
        print(f"  {r0.label}, {len(seeds)} seeds: " + "; ".join(parts))
    if not noisy:
        print("  no cell has two seeds")
    if args.ladder:
        lad = {(r.cfg.n_layers, r.cfg.dim, round(r.cfg.peak_lr, 9)): r for r in load_runs(args.ladder) if r.cfg.seed == 0}  # a size can have several LRs (the bracket's)
        print(f"\n== ladder check: the grid's aspect-64 cells against the ladder runs of the same size and LR ({args.ladder}), grid - ladder ==")
        for k, r0 in sorted(base.items()):
            lk = (*k, round(r0.cfg.peak_lr, 9))
            if lk in lad:
                diffs = [(c, val(r0, c) - val(lad[lk], c)) for c in budgets]
                print(f"  {r0.label}: " + "  ".join(f"{c:.1e}: {x:+.4f}" for c, x in diffs if math.isfinite(x)))
        print("  (the grid validates on 524,288 tokens every 200 steps, the ladder on 2M every 500: expect a small constant offset, the same at every size)")

    # ---- the learning rate by depth
    if args.lr_group:
        lr_runs = [r for r in load_runs(args.lr_group) if r.cfg.seed == 0]
        widths = sorted({r.cfg.dim for r in lr_runs})
        for w in widths:
            print(f"\n== learning rate by depth at width {w}: val loss at the budgets, the grid's rule-LR run in the middle ({args.lr_group}) ==")
            for n_l in sorted({r.cfg.n_layers for r in lr_runs if r.cfg.dim == w}):
                cell = [r for r in lr_runs if r.cfg.dim == w and r.cfg.n_layers == n_l]
                if (n_l, w) in base:
                    cell.append(base[(n_l, w)])
                cell.sort(key=lambda r: r.cfg.peak_lr)
                parts = []
                for c in budgets:
                    pts = [(r.cfg.peak_lr, val(r, c)) for r in cell if math.isfinite(val(r, c))]
                    if len(pts) < 3:
                        continue
                    lr_best, loss_best, edge = lr_optimum(pts)
                    rule = [v for lr_, v in pts if (n_l, w) in base and math.isclose(lr_, base[(n_l, w)].cfg.peak_lr, rel_tol=1e-6)]
                    parts.append(f"{c:.1e}: [" + ", ".join(f"{lr_:.1e}: {v:.4f}" for lr_, v in pts) + f"] best LR {lr_best:.1e}{' (at the edge: a bound)' if edge else ''}"
                                 + (f", the rule's LR costs {rule[0] - loss_best:+.4f}" if rule and not edge else ""))
                    if not edge:
                        out.setdefault("lr_by_depth", {})[f"L{n_l}x{w}@{c:.3g}"] = dict(best_lr=lr_best, rule_cost=(rule[0] - loss_best) if rule else None)
                print(f"  {n_l:>2} layers: " + ("\n             ".join(parts) if parts else "too few runs"))

    # ---- one-factor slices
    mid = budgets[len(budgets) // 2]
    deltas: list[tuple[str, float]] = []

    def slice_table(title: str, items: list[Run], what, correct_size: bool) -> None:
        print(f"\n== {title}: val loss minus the ladder's recipe at the same layers x width and compute (negative = better) ==")
        print("  " + f"{'run':<10}{what[0]:<10}{'N (M)':>8}{'N vs ladder':>13}" + "".join(f"{c:>16.1e}" if correct_size else f"{c:>13.1e}" for c in budgets) + ("   (raw / size-corrected)" if correct_size else ""))
        for r in sorted(items, key=lambda r: (r.cfg.n_layers, what[1](r))):
            b = base.get((r.cfg.n_layers, r.cfg.dim))
            if b is None:
                continue
            cells = []
            for c in budgets:
                raw = val(r, c) - val(b, c)
                if not math.isfinite(raw):
                    cells.append(f"{'-':>16}" if correct_size else f"{'-':>13}")
                    continue
                adj = raw
                if correct_size:
                    s, u_b, u_r = surf.get(c), math.log(b.n("total") / 1e6), math.log(r.n("total") / 1e6)
                    if s and s["u_range"][0] - 0.05 <= u_r <= s["u_range"][1] + 0.05:  # no correction by extrapolating the surface
                        v_b = math.log(b.cfg.dim / b.cfg.n_layers / 64)
                        adj = raw - (surface_at(s["coef"], u_r, v_b) - surface_at(s["coef"], u_b, v_b))
                    else:
                        adj = math.nan
                cells.append((f"{raw:+.4f}/" + (f"{adj:+.4f}" if math.isfinite(adj) else "n/a")).rjust(16) if correct_size else f"{raw:>+13.4f}")
                if c == mid and math.isfinite(adj):
                    deltas.append((f"{r.label} {what[0]} {what[1](r)}", adj))
                    out.setdefault(title.split(" (")[0], {})[f"{r.label} {what[1](r)}@{c:.3g}"] = dict(raw=raw, size_corrected=adj)
            print("  " + f"{r.label:<10}{str(what[1](r)):<10}{r.n('total') / 1e6:>8.2f}{r.n('total') / b.n('total'):>13.2f}" + "".join(cells))

    ff = [r for r in plain if r.cfg.seed == 0 and r.cfg.ff_mult != 0 and r.cfg.head_dim == 64 and r.cfg.block_size == 256]
    if ff:
        slice_table("MLP size (ff_mult: hidden size / width; the ladder's is 8/3 = 2.67)", ff, ("ff_mult", lambda r: r.cfg.ff_mult), True)
        print("  size-corrected: the raw gap minus the change the fitted surface above gives for moving to the variant's size at the same aspect ratio and compute,"
              " so it is the gap at equal parameters (about equal tokens): what the MLP size does besides changing the size. n/a: the variant is outside the sizes the surface was fitted on.")
    hd = [r for r in plain if r.cfg.seed == 0 and r.cfg.head_dim != 64 and r.cfg.ff_mult == 0 and r.cfg.block_size == 256]
    if hd:
        slice_table("head size (head_dim; 64 heads of the ladder's width / 64)", hd, ("head_dim", lambda r: r.cfg.head_dim), False)

    # ---- MLP and head size at equal parameters, from one joint fit of the surface and the variants
    kinds: dict[tuple, str] = {}
    for r in plain:
        if r.cfg.block_size != 256 or _variant(r.cfg) == BASE_VARIANT:
            continue
        if r.cfg.ff_mult != 0 and r.cfg.head_dim == 64:
            kinds[("ff_mult", r.cfg.ff_mult)] = f"MLP x{r.cfg.ff_mult:g}"
        elif r.cfg.head_dim != 64 and r.cfg.ff_mult == 0:
            kinds[("head_dim", r.cfg.head_dim)] = f"head size {r.cfg.head_dim}"
    joint_bars: list[tuple[str, float, float]] = []
    if kinds:
        cols = sorted(kinds, key=lambda k: (k[0] != "ff_mult", k[1]))
        print("\n== MLP and head size at equal parameters, aspect ratio and compute: one fit of the surface and the variants together ==")
        print("   (every run and seed of the grid; loss = surface(ln N, ln aspect) + one offset per variant; positive = worse than the ladder's 8/3 MLP and 64-wide heads.")
        print("    The error is the fit's standard error, which takes the surface to be right. At equal compute a bigger MLP also sees a few percent more tokens (a smaller share of its FLOPs goes to attention")
        print("    and the output layer), which is part of the offset, as it is of any use of the variant; residuals of the variants at the middle budget follow, a size trend in them would say the offset is no constant.)")
        print("  " + f"{'variant':<16}" + "".join(f"{c:>18.1e}" for c in budgets))
        table: dict[tuple, list[str]] = {k: [] for k in cols}
        residuals: dict[tuple, list[str]] = {k: [] for k in cols}
        slopes: list[tuple[str, float, float]] = []
        for c in budgets:
            pts = []
            for r in plain:
                if r.cfg.block_size != 256 or not math.isfinite(val(r, c)):
                    continue
                if _variant(r.cfg) == BASE_VARIANT:
                    key = None
                elif r.cfg.ff_mult != 0 and r.cfg.head_dim == 64:
                    key = ("ff_mult", r.cfg.ff_mult)
                elif r.cfg.head_dim != 64 and r.cfg.ff_mult == 0:
                    key = ("head_dim", r.cfg.head_dim)
                else:
                    continue
                pts.append((math.log(r.n("total") / 1e6), math.log(r.cfg.dim / r.cfg.n_layers / 64), val(r, c), key, r))
            used = [k for k in cols if any(p[3] == k for p in pts)]
            u, v, y = (np.array([p[i] for p in pts]) for i in range(3))
            x = np.stack([np.ones_like(u), u, u**2, u**3, v, v**2, u * v] + [np.array([1.0 if p[3] == k else 0.0 for p in pts]) for k in used], axis=1)
            if len(y) < x.shape[1] + 3 or np.linalg.matrix_rank(x) < x.shape[1]:
                for k in cols:
                    table[k].append(f"{'-':>18}")
                continue
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            resid = y - x @ coef
            se = np.sqrt(np.diag(resid @ resid / (len(y) - x.shape[1]) * np.linalg.inv(x.T @ x)))
            for k in cols:
                if k not in used:
                    table[k].append(f"{'-':>18}")
                    continue
                j = 7 + used.index(k)
                table[k].append(f"{coef[j]:+.4f} +-{se[j]:.4f}".rjust(18))
                out.setdefault("joint_offsets", {})[f"{kinds[k]}@{c:.3g}"] = dict(offset=float(coef[j]), se=float(se[j]), rms=float(np.sqrt(resid @ resid / len(y))))
                if c == mid:
                    joint_bars.append((kinds[k], float(coef[j]), float(se[j])))
                    residuals[k] = [f"{p[4].label} (a{p[4].cfg.dim / p[4].cfg.n_layers:.0f}) {resid[i]:+.4f}" for i, p in enumerate(pts) if p[3] == k]
            if c == mid:  # does an MLP offset depend on the aspect ratio? one more column per MLP size with enough cells: the offset's slope in ln aspect
                inter = [k for k in used if k[0] == "ff_mult" and sum(p[3] == k for p in pts) >= 5 and len({round(p[1], 2) for p in pts if p[3] == k}) >= 3]
                if inter:
                    x2 = np.concatenate([x, np.stack([np.array([p[1] if p[3] == k else 0.0 for p in pts]) for k in inter], axis=1)], axis=1)
                    if np.linalg.matrix_rank(x2) == x2.shape[1] and len(y) > x2.shape[1] + 2:
                        coef2, *_ = np.linalg.lstsq(x2, y, rcond=None)
                        r2 = y - x2 @ coef2
                        se2 = np.sqrt(np.diag(r2 @ r2 / (len(y) - x2.shape[1]) * np.linalg.inv(x2.T @ x2)))
                        slopes = [(kinds[k], float(coef2[x.shape[1] + i] * ln2), float(se2[x.shape[1] + i] * ln2)) for i, k in enumerate(inter)]
                        for name, sl, e in slopes:
                            out.setdefault("joint_offsets", {})[f"{name} per octave of aspect@{c:.3g}"] = dict(slope=sl, se=e)
        for k in cols:
            print("  " + f"{kinds[k]:<16}" + "".join(table[k]))
        for k in cols:
            if residuals[k]:
                print(f"   residuals at {mid:.1e}, {kinds[k]}: " + ", ".join(residuals[k]))
        if slopes:
            print(f"   how the offset changes per doubling of the aspect ratio, at {mid:.1e} (+ = the variant gets worse as the model gets wider and shallower): "
                  + ", ".join(f"{n} {sl:+.4f} +-{e:.4f}" for n, sl, e in slopes))
    blk = {(r.cfg.n_layers, r.cfg.dim, r.cfg.block_size): r for r in runs if r.cfg.val_by_pos and r.cfg.seed == 0 and r.cfg.ff_mult == 0 and r.cfg.head_dim == 64}
    if blk:
        print("\n== context length: val loss by position range minus the 256-token context at the same layers x width and compute (negative = better) ==")
        print("   (the mean over a window is not comparable across lengths, a longer window has more late, easy positions; ranges every length has are)")
        for (n_l, w) in sorted({k[:2] for k in blk}):
            ref = blk.get((n_l, w, 256))
            if ref is None:
                continue
            sizes = sorted(k[2] for k in blk if k[:2] == (n_l, w))
            for c in budgets:
                if not math.isfinite(val(ref, c)):
                    continue
                print(f"  L{n_l}x{w} at {c:.1e} FLOPs, context " + " ".join(f"{s:>7}" for s in sizes) + "   [tokens seen (M): " + ", ".join(f"{c / blk[(n_l, w, s)].cfg.train_flops_per_token / 1e6:.0f}" for s in sizes) + "]")
                names = [k for k in ref.pos if int(k.split("_")[-1]) <= 256]
                for key in names:
                    row = []
                    for s in sizes:
                        x = val(blk[(n_l, w, s)], c, key) - val(ref, c, key) if key in blk[(n_l, w, s)].pos else math.nan
                        row.append(f"{x:>+7.4f}" if math.isfinite(x) else f"{'-':>7}")
                        if s != 256 and math.isfinite(x) and c == mid:
                            out.setdefault("context", {})[f"L{n_l}x{w} {key[9:]} ctx{s}@{c:.3g}"] = x
                    print(f"    positions {key[len('val/loss_pos_'):].replace('_', '-'):>8}:  " + " ".join(row))
                allpos = " ".join(f"{val(blk[(n_l, w, s)], c):>7.4f}" for s in sizes)
                print(f"    all positions (not comparable): {allpos}")

    # ---- figures
    ok_cells = {k: val(r, mid) for k, r in base.items() if math.isfinite(val(r, mid))}
    if ok_cells:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        grid = np.full((len(layers), len(dims)), np.nan)
        for (n_l, w), v in ok_cells.items():
            grid[layers.index(n_l), dims.index(w)] = v
        im = ax[0].imshow(grid, origin="lower", cmap="viridis_r", aspect="auto")
        for (n_l, w), v in ok_cells.items():
            ax[0].text(dims.index(w), layers.index(n_l), f"{v:.3f}\n{base[(n_l, w)].n('total') / 1e6:.1f}M", ha="center", va="center", fontsize=7, color="w")
        ax[0].set(xticks=range(len(dims)), xticklabels=dims, yticks=range(len(layers)), yticklabels=layers, xlabel="width", ylabel="layers",
                  title=f"val loss at {mid:.1e} FLOPs (parameters below)")
        fig.colorbar(im, ax=ax[0])
        us = np.array([math.log(base[k].n("total") / 1e6) for k in ok_cells])
        vs = np.array([math.log(k[1] / k[0] / 64) for k in ok_cells])
        sc = ax[1].scatter(64 * np.exp(vs), list(ok_cells.values()), c=us, cmap="plasma", s=30)
        fig.colorbar(sc, ax=ax[1], label="ln(parameters / 1M)")
        if surf.get(mid):
            xs = np.linspace(vs.min(), vs.max(), 60)
            for n_m in (2, 4, 8):
                u = math.log(n_m)
                if us.min() - 0.1 <= u <= us.max() + 0.1:
                    cf = surf[mid]["coef"]
                    ax[1].plot(64 * np.exp(xs), cf[0] + cf[1] * u + cf[2] * u**2 + cf[3] * u**3 + cf[4] * xs + cf[5] * xs**2 + cf[6] * u * xs, lw=1, label=f"fit at {n_m}M")
        ax[1].axvline(64, color="k", lw=0.8, ls=":")
        ax[1].set(xscale="log", xlabel="aspect ratio (width / layers); the ladder is 64", ylabel=f"val loss at {mid:.1e} FLOPs")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3, which="both")
        save_fig(fig, d / "shape.png")
    bars = joint_bars or [(n, x, 0.0) for n, x in deltas]
    if bars:
        fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(bars) + 1.5))
        ax.barh([n for n, _, _ in bars], [x for _, x, _ in bars], xerr=[e for _, _, e in bars] if joint_bars else None, color=["C0" if x <= 0 else "C3" for _, x, _ in bars], capsize=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set(xlabel=f"val loss minus the ladder's recipe at {mid:.1e} FLOPs (negative = better)\nat equal parameters, aspect ratio and compute; bars: +- the fit's standard error" if joint_bars
               else f"val loss minus the ladder's recipe at {mid:.1e} FLOPs (negative = better; MLP variants size-corrected)")
        ax.tick_params(axis="y", labelsize=7)
        save_fig(fig, d / "variants.png")
    save_results(d, out)


# ---------------------------------------------------------------- fit-optim

OPTIM_KNOBS = (("weight_decay", "weight decay"), ("beta2", "AdamW beta2"), ("warmup", "warmup steps"), ("global_batch", "tokens per step"),
               ("grad_clip", "gradient clipping"), ("rope_theta", "RoPE theta"), ("doc_mask", "document mask"))


def _fmt(v: float) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    return f"{v:,}" if isinstance(v, int) and v >= 1000 else f"{v:g}"


def run_fit_optim(args: OptimConfig) -> None:
    """One model size run with one optimizer setting changed at a time (or the batch size with its learning rate): the loss minus the
    base recipe's at the same compute, with the seed noise of the base cell next to it."""
    runs = load_runs(args.group)
    if not runs:
        raise SystemExit(f"no finished runs in group {args.group!r}")
    every = [r for r in load_runs(args.base) if not r.cfg.val_by_pos]  # the context-length runs repeat the base cell (seed 0) and would count as a second seed
    base = {(r.cfg.n_layers, r.cfg.dim): r for r in every if r.cfg.seed == 0 and _variant(r.cfg) == BASE_VARIANT}
    base_seed = {(r.cfg.n_layers, r.cfg.dim, r.cfg.seed): r for r in every if _variant(r.cfg) == BASE_VARIANT}  # a run of another seed is compared with the base cell of that seed, when there is one
    budgets, d, out = list(args.flops), out_dir(args.out, args.group), {}
    mid = budgets[len(budgets) // 2]
    val = lambda r, c: _loss_at(r, c, args.min_tokens)  # noqa: E731
    print(f"{len(runs)} finished runs in {args.group}, compared at {', '.join(f'{c:.1e}' for c in budgets)} FLOPs with the {args.base} run of the same size (seed 0)")
    bars: list[tuple[str, float]] = []
    noise = 0.0
    for key in sorted({(r.cfg.n_layers, r.cfg.dim) for r in runs}):
        b = base.get(key)
        if b is None:
            print(f"\n  L{key[0]}x{key[1]}: no {args.base} run of this size to compare with")
            continue
        bc, mine = b.cfg, [r for r in runs if (r.cfg.n_layers, r.cfg.dim) == key]
        print(f"\n== optimizer settings at {b.label} ({b.n('total') / 1e6:.2f}M parameters): val loss minus the base recipe at equal compute (negative = better) ==")
        print(f"   base ({args.base}): weight decay {bc.weight_decay:g}, AdamW beta2 {bc.beta2:g}, warmup {bc.warmup} steps, {bc.global_batch:,} tokens per step, peak LR {bc.peak_lr:.1e}, "
              f"gradient clipping {bc.grad_clip:g}, RoPE theta {bc.rope_theta:g}, document mask {_fmt(bc.doc_mask)}")
        seeds = [r for r in every if (r.cfg.n_layers, r.cfg.dim) == key and _variant(r.cfg) == BASE_VARIANT and math.isclose(r.cfg.peak_lr, bc.peak_lr, rel_tol=1e-6)]
        if len(seeds) >= 2:
            sd = [float(np.std([val(r, c) for r in seeds], ddof=1)) for c in budgets]
            noise = max(noise, sd[len(sd) // 2])
            print(f"   seed noise of the base cell ({len(seeds)} seeds): std " + ", ".join(f"{x:.4f}" for x in sd) + " at the budgets below")
        one, by_batch = [], {}
        for r in mine:
            diff = [k for k, _ in OPTIM_KNOBS if getattr(r.cfg, k) != getattr(bc, k)]
            lr_differs = not math.isclose(r.cfg.peak_lr, bc.peak_lr, rel_tol=1e-6)
            if diff in ([], ["global_batch"]) and (lr_differs or diff):  # a batch size with its learning rate, or the learning rate alone
                if r.cfg.seed == 0:
                    by_batch.setdefault(r.cfg.global_batch, []).append(r)
            elif len(diff) == 1 and not lr_differs:
                one.append((diff[0], getattr(r.cfg, diff[0]), r))
            else:
                print(f"   (left out {r.id}: it differs from the base in {diff} and the learning rate {'too' if lr_differs else 'not'})")
        names = dict(OPTIM_KNOBS)
        print("   " + f"{'setting':<18}{'value':<18}" + "".join(f"{c:>12.1e}" for c in budgets))
        for knob, v, r in sorted(one, key=lambda t: ([k for k, _ in OPTIM_KNOBS].index(t[0]), t[1], t[2].cfg.seed)):
            bs = base_seed.get((*key, r.cfg.seed), b)  # the base cell of the run's own seed (seed 0 unless the run is a replicate)
            tag = f" (seed {r.cfg.seed})" if r.cfg.seed else ""
            cells = [val(r, c) - val(bs, c) for c in budgets]
            print("   " + f"{names[knob]:<18}{_fmt(v) + tag:<18}" + "".join(f"{x:>+12.4f}" if math.isfinite(x) else f"{'-':>12}" for x in cells))
            for c, x in zip(budgets, cells):
                if math.isfinite(x):
                    out.setdefault(b.label, {})[f"{knob}={v:g}{'/seed' + str(r.cfg.seed) if r.cfg.seed else ''}@{c:.3g}"] = x
            if math.isfinite(cells[len(cells) // 2]):
                bars.append((f"{b.label} {names[knob]} {_fmt(v)}{tag}", cells[len(cells) // 2]))
        for batch in sorted(by_batch):
            cell = sorted(by_batch[batch] + ([b] if batch == bc.global_batch else []), key=lambda r: r.cfg.peak_lr)
            for c in budgets:
                pts = [(r.cfg.peak_lr, val(r, c) - val(b, c)) for r in cell if math.isfinite(val(r, c))]
                what = f"{batch:,} tokens per step" if batch != bc.global_batch else "the base batch"
                txt = ", ".join(f"{lr:.1e}: {x:+.4f}" for lr, x in pts)
                if len(pts) >= 3:
                    lr_b, x_b, edge = lr_optimum(pts)
                    txt += f"   best LR {lr_b:.1e}" + (" (at the edge: only a bound)" if edge else f", {x_b:+.4f}")
                    out.setdefault(b.label, {})[f"batch={batch}@{c:.3g}"] = dict(best_lr=lr_b, delta=x_b, edge=edge)
                    if c == mid:
                        bars.append((f"{b.label} {what}, its best LR", x_b))
                print(f"   {what}, at {c:.1e}: [{txt}]")
    if bars:
        fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(bars) + 1.5))
        ax.barh([n for n, _ in bars], [x for _, x in bars], color=["C0" if x <= 0 else "C3" for _, x in bars])
        if noise:
            ax.axvspan(-noise, noise, color="0.8", zorder=0, label="seed noise (std)")
            ax.legend(fontsize=7)
        ax.axvline(0, color="k", lw=0.8)
        ax.set(xlabel=f"val loss minus the base recipe at {mid:.1e} FLOPs (negative = better)")
        ax.tick_params(axis="y", labelsize=7)
        save_fig(fig, d / "optim.png")
    save_results(d, out)

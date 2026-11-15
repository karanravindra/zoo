"""Compare TinyStories training runs side by side.

`uv run projects/tinystories/compare.py [run_dir ...] [--x tokens] [--out PATH]`

Defaults to every run under `<repo>/runs/tinystories` and writes `comparison.png` next to them. The top row is val
loss against the three budgets a run can be compared at: training time, training FLOPs, and `--x` (steps by default).
The loss is `val_doc_loss` (whole stories from their start, comparable across the document options) when every run
has it, else `val_loss`; `--metric` picks one. FLOPs come from the `flops` column of metrics.csv, or for runs from before it existed from tokens times the config's
`train_flops_per_token`, which is the same number. Each run is labelled
by the config values that differ between the runs (e.g. `dim=256, lr=0.001`), and the runs are ordered by those
values (numerically where they are numbers), so a sweep reads in order. Runs still training (no `results.json`
yet) are included, scored by their best eval row so far; runs with no eval row yet are skipped.
"""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from llama import Config, build_model
from tyro.conf import Positional

from zoo.paths import runs_root
from zoo.plot import plt
from zoo.runs import diff_labels, load_run, read_metrics, sort_runs

RUNS_ROOT = runs_root("tinystories")

# Config keys that do not change what was trained, so they never label a run.
IGNORE_KEYS = {"run_name", "log_every", "sample_every", "sample_prompt", "sample_tokens", "sample_temperature", "sample_top_k"}
X_LABELS = {"step": "step", "tokens": "training tokens", "elapsed_s": "training time (s)", "flops": "training FLOPs"}


@dataclass
class Args:
    runs: Positional[list[Path]] = field(default_factory=list)  # run dirs (default: all under runs/tinystories)
    x: Literal["step", "tokens", "elapsed_s", "flops"] = "step"  # x axis of the third loss panel and the bottom row: optimizer steps, training tokens, seconds or FLOPs
    metric: Literal["auto", "val_loss", "val_doc_loss"] = "auto"  # loss to plot and rank by: val_loss is measured the way each run trains; val_doc_loss scores whole stories from their start and is comparable across --doc-mask / --doc-align; auto = val_doc_loss when every run has it
    log_x: bool = False  # log-scale x axes, to see the early part of long runs
    out: Path = RUNS_ROOT / "comparison.png"


def load(run: Path) -> dict:
    """`load_run`, plus the parameter count and the best eval row (lowest val loss, as train.py picks best.pt)."""
    r = load_run(run)
    cfg = Config.from_dict(r["config"])
    r["config"] = asdict(cfg)  # fills keys that older runs lack with today's defaults
    r["params"] = sum(p.numel() for p in build_model(cfg).parameters())
    m = r["metrics"]
    if "flops" not in m:  # runs from before the column existed: the count is analytic, so it can be rebuilt
        m["flops"] = [t * cfg.train_flops_per_token for t in m["tokens"]]
    return r


def set_best(r: dict, metric: str) -> None:
    """The run's best eval row by `metric` (train.py picks best.pt by val_loss; the two rarely disagree)."""
    m = r["metrics"]
    evals = [i for i, v in enumerate(m[metric]) if not math.isnan(v)]  # newer runs also log rows between evals
    best = min(evals, key=lambda i: m[metric][i])
    r["best"] = {k: v[best] for k, v in m.items()}


def has_rows(run: Path) -> bool:
    """True once the run has logged at least one eval row."""
    path = run / "metrics.csv"
    if not path.exists():
        return False
    return any(not math.isnan(v) for v in read_metrics(run).get("val_loss", []))


def main(args: Args):
    dirs = args.runs or [d for d in RUNS_ROOT.iterdir() if d.is_dir()]
    dirs = [d for d in dirs if has_rows(d)]
    if not dirs:
        raise SystemExit("no runs with at least one eval row")
    runs = sort_runs([load(d) for d in dirs], IGNORE_KEYS)  # legend/colour/table order: by the config values that differ
    labels = diff_labels(runs, IGNORE_KEYS)
    metric = args.metric
    if metric == "auto":
        metric = "val_doc_loss" if all("val_doc_loss" in r["metrics"] for r in runs) else "val_loss"
    missing = [r["name"] for r in runs if metric not in r["metrics"]]
    if missing:
        raise SystemExit(f"{metric} is not in metrics.csv of: {', '.join(missing)} (runs from before it was logged); use --metric val_loss")
    bpb = metric.replace("loss", "bpb")
    name = {"val_loss": "Val loss", "val_doc_loss": "Val loss per story"}[metric]
    for r in runs:
        set_best(r, metric)
    colors = [f"C{i % 10}" for i in range(len(runs))]

    fig, ax = plt.subplots(3, 2, figsize=(13, 13))  # 3 rows x 2 columns on a square figure
    panels = [  # (x column, y column, title)
        ("elapsed_s", metric, f"{name} vs training time"),
        ("flops", metric, f"{name} vs training FLOPs"),
        (args.x, metric, f"{name} vs {X_LABELS[args.x]}"),
        (args.x, "train_loss", "Train loss (nats/token)"),
        (args.x, "tokens_per_s", "Training tokens/s"),
    ]
    for a, (x, key, title) in zip(ax.flat, panels):
        for run, label, c in zip(runs, labels, colors):
            pts = [(u, v) for u, v in zip(run["metrics"][x], run["metrics"][key]) if not math.isnan(v)]  # skip blank cells
            a.plot([p[0] for p in pts], [p[1] for p in pts], c=c, label=label, marker="." if len(pts) < 3 else None, lw=1.2)
        a.set_title(title)
        a.set_xlabel(X_LABELS[x])
        if args.log_x:
            a.set_xscale("log")
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)

    bars = ax[2, 1]
    xs = list(range(len(runs)))
    vals = [r["best"][bpb] for r in runs]
    bars.bar(xs, vals, color=colors)
    bars.set_title(f"Best {bpb.replace('_', ' ')} (hatched: still training)")
    bars.set_xticks(xs)
    bars.set_xticklabels([r["name"] for r in runs], fontsize=8, rotation=30, ha="right")  # short: config labels overlap
    for patch, r, x, v in zip(bars.patches, runs, xs, vals):
        if r["results"] is None:
            patch.set_hatch("//")
        bars.text(x, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    bars.set_ylim(min(vals) * 0.95, max(vals) * 1.03)

    fig.suptitle("TinyStories runs")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)

    w = max(len(r["name"]) for r in runs) + 2
    print(f"{'run':<{w}}{'params':>8}{'step':>8}{metric:>14}{bpb:>13}{'time_s':>8}{'FLOPs':>10}{'tokens/s':>11}{'mem_GiB':>9}  config")
    for r, label in zip(runs, labels):
        b, m = r["best"], r["metrics"]
        # mean training tokens/s: results.json for finished runs, else tokens seen over training time so far
        tps = r["results"]["mean_tokens_per_s"] if r["results"] else m["tokens"][-1] / m["elapsed_s"][-1]
        mem = (r["results"] or {}).get("peak_gpu_mem_gb") or max((v for v in m.get("gpu_mem_gb", []) if not math.isnan(v)), default=None)
        mem = f"{mem:>9.1f}" if mem else f"{'-':>9}"  # runs from before GPU logging have none
        state = "" if r["results"] else f"  (training: step {int(m['step'][-1])})"
        # time and FLOPs are what it took to reach the best row
        print(f"{r['name']:<{w}}{r['params'] / 1e6:>7.2f}M{int(b['step']):>8}{b[metric]:>14.4f}{b[bpb]:>13.4f}"
              f"{b['elapsed_s']:>8.0f}{b['flops']:>10.2e}{tps:>11,.0f}{mem}  {label}{state}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))

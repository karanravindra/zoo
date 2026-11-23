"""Compare pretraining runs side by side.

`uv run projects/pretrain/compare.py [run_dir ...] [--x tokens] [--out PATH]`

Defaults to every run under `<repo>/runs/pretrain` and writes `comparison.png` next to them. The top row is val
loss against the three budgets a run can be compared at: training time, training FLOPs, and `--x` (steps by default;
`--x tokens` compares runs with different batch sizes). The loss is `val_doc_loss` (whole stories from their start,
comparable across the document options) when every run has it, else `val_loss`; `--metric` picks one. The val panels
clip the steep start of the curves: their y axis ends just above the worst run's best value. FLOPs come from the `flops`
column of metrics.csv, or for runs from before it existed from tokens times the config's `train_flops_per_token`, which
is the same number. Runs are ordered by FLOP budget, then by the other config values that differ, and each is labelled
with the budget and the differing values in a short form (e.g. `3e17 d640 L8 64k tok/step lr0.0008`; batch size is
shown as tokens per step, 1k = 1024; the legend sits below the panels). Runs still training (no `results.json` yet) are
included, scored by their best eval row so far; runs with no eval row yet are skipped.

Runs trained on the story mix also log `val_<source>_bpb` (train.py's per-source validation). When any run has them, a
second figure, `comparison_sources.png` next to the first, shows every run's bits per byte on each source at its best
eval row, and the same numbers are printed as a table (`-` for runs from before per-source validation).
"""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from llama import Config, build_model
from tyro.conf import Positional

from zoo.paths import runs_root
from zoo.plot import plt, time_unit
from zoo.runs import load_run, read_metrics, sort_runs, varying_keys

RUNS_ROOT = runs_root("pretrain")

# Config keys that do not change what was trained, so they never label a run.
IGNORE_KEYS = {"grad_accum", "run_name", "log_every", "sample_every", "sample_prompt", "sample_tokens", "sample_temperature", "sample_top_k", "val_source_batches",
               "eval_every", "grad_checkpoint"}
# Short names for the config keys in a legend label; a bool shows as its name when True and not at all when False.
ALIASES = {"dim": "d", "block_size": "ctx", "n_heads": "h", "attn_window": "win", "attn_global_every": "g", "batch_size": "bs",
           "doc_mask": "mask", "doc_align": "align", "n_layers": "L", "lr": "lr"}
X_LABELS = {"step": "step", "tokens": "training tokens", "elapsed_s": "training time", "flops": "training FLOPs"}
# Reading order of config keys in a label (and the axis order of sweep.py): budget, architecture, then batch and lr.
LABEL_ORDER = ["flop_budget", "block_size", "dim", "n_layers", "n_heads", "attn_window", "attn_global_every", "doc_mask", "doc_align", "batch_size", "lr"]


@dataclass
class Args:
    runs: Positional[list[Path]] = field(default_factory=list)  # run dirs (default: all under runs/pretrain)
    x: Literal["step", "tokens", "elapsed_s", "flops"] = "step"  # x axis of the third loss panel and the bottom row: optimizer steps, training tokens, seconds or FLOPs (tokens compares runs with different batch sizes)
    metric: Literal["auto", "val_loss", "val_doc_loss"] = "auto"  # loss to plot and rank by: val_loss is measured the way each run trains; val_doc_loss scores whole stories from their start and is comparable across --doc-mask / --doc-align; auto = val_doc_loss when every run has it
    log_x: bool = False  # log-scale x axes, to see the early part of long runs
    out: Path = RUNS_ROOT / "comparison.png"


def load(run: Path) -> dict:
    """`load_run`, plus the parameter count and the best eval row (lowest val loss, as train.py picks best.pt)."""
    r = load_run(run)
    cfg = Config.from_dict(r["config"])
    r["config"] = asdict(cfg)  # fills keys that older runs lack with today's defaults
    # An OOM restart splits the batch and accumulates (same tokens per step), and a FLOP budget fixes max_steps: neither is a difference between runs
    r["config"]["batch_size"] = cfg.batch_size * cfg.grad_accum
    if cfg.flop_budget > 0 and cfg.time_budget_s <= 0:
        r["config"]["max_steps"] = None
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


def short_labels(runs: list[dict]) -> list[str]:
    """One compact label per run from the config keys that differ, e.g. `ctx2048 d512 win512 mask`."""
    keys = varying_keys(runs, IGNORE_KEYS)
    if "dim" in keys:  # n_heads follows dim in a width sweep, so it only adds noise
        keys = [k for k in keys if k != "n_heads"]
    keys = sorted(keys, key=lambda k: LABEL_ORDER.index(k) if k in LABEL_ORDER else len(LABEL_ORDER))  # the same reading order in every label
    labels = []
    for r in runs:
        parts = []
        for k in keys:
            v, a = r["config"].get(k), ALIASES.get(k, k)
            if k == "flop_budget":
                parts.append(fmt_budget(v) if v else "time budget")
            elif k == "batch_size":  # sequences per step is not comparable across context lengths: show tokens per step
                parts.append(f"{v * r['config']['block_size'] / 1024:.0f}k tok/step")  # 1k = 1024 tokens: 64k, not 66k
            elif isinstance(v, bool):
                parts += [a] if v else []
            elif v in (None, 0, "off"):  # off / unset: nothing to say
                continue
            else:
                parts.append(f"{a}{v}" if isinstance(v, (int, float)) else f"{a}={v}" if k != "doc_align" else v)
        labels.append(" ".join(parts) or "baseline")
    return labels


def fmt_budget(flops: float) -> str:
    """`3e+17` -> `3e17`."""
    return f"{flops:.0e}".replace("e+", "e")


def source_names(runs: list[dict]) -> list[str]:
    """Sources with a `val_<source>_bpb` column in at least one run, in the order first seen."""
    names: list[str] = []
    for r in runs:
        for k in r["metrics"]:
            if k.startswith("val_") and k.endswith("_bpb") and k not in ("val_bpb", "val_doc_bpb") and k[4:-4] not in names:
                names.append(k[4:-4])
    return names


def plot_sources(runs: list[dict], labels: list[str], colors: list, names: list[str], path: Path) -> None:
    """Grouped bars: each run's val bits per byte on every source, at the run's best eval row (a run without a
    source's column leaves a gap)."""
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(names) + 3), 5))
    width = 0.8 / len(runs)
    for i, (r, label, c) in enumerate(zip(runs, labels, colors)):
        vals = [r["best"].get(f"val_{n}_bpb", math.nan) for n in names]
        ax.bar([j + (i - (len(runs) - 1) / 2) * width for j in range(len(names))], vals, width, color=c, label=label)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("val bits per byte")
    ax.set_title("Val bits per byte by source (at each run's best eval row)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(len(runs), 4), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


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
    runs.sort(key=lambda r: r["config"].get("flop_budget") or 0)  # stable: the budget ladder first, the other differences within a budget
    labels = short_labels(runs)
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
    colors = [plt.cm.tab20(i % 20) for i in range(len(runs))]

    fig, ax = plt.subplots(3, 2, figsize=(13, 14))  # 3 rows x 2 columns, with room below for the legend
    panels = [  # (x column, y column, title)
        ("elapsed_s", metric, f"{name} vs training time"),
        ("flops", metric, f"{name} vs training FLOPs"),
        (args.x, metric, f"{name} vs {X_LABELS[args.x]}"),
        (args.x, "train_loss", "Train loss (nats/token)"),
        (args.x, "tokens_per_s", "Training tokens/s"),
    ]
    t_unit, t_div = time_unit(max(max(r["metrics"]["elapsed_s"]) for r in runs))  # one unit for every time axis
    x_labels = {**X_LABELS, "elapsed_s": f"training time ({t_unit})"}
    # The val panels skip the steep start of the curves: the y range ends just above the worst run's best
    best_vals = [r["best"][metric] for r in runs]
    for a, (x, key, title) in zip(ax.flat, panels):
        div = t_div if x == "elapsed_s" else 1.0
        for run, label, c in zip(runs, labels, colors):
            pts = [(u / div, v) for u, v in zip(run["metrics"][x], run["metrics"][key]) if not math.isnan(v)]  # skip blank cells
            a.plot([p[0] for p in pts], [p[1] for p in pts], c=c, label=label, marker="." if len(pts) < 3 else None, lw=1.2)
        a.set_title(title)
        a.set_xlabel(x_labels[x])
        if key == metric:
            a.set_ylim(min(best_vals) * 0.97, max(best_vals) * 1.12)
        if args.log_x:
            a.set_xscale("log")
        a.grid(alpha=0.3)

    bars = ax[2, 1]
    xs = list(range(len(runs)))
    vals = [r["best"][bpb] for r in runs]
    bars.bar(xs, vals, color=colors)
    bars.set_title(f"Best {bpb.replace('_', ' ')} (hatched: still training)")
    bars.set_xticks(xs)
    bars.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    for patch, r, x, v in zip(bars.patches, runs, xs, vals):
        if r["results"] is None:
            patch.set_hatch("//")
        bars.text(x, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    bars.set_ylim(min(vals) * 0.95, max(vals) * 1.03)

    fig.suptitle("Pretrain runs")
    handles = [plt.Line2D([], [], c=c, lw=2) for c in colors]
    ncol = min(len(runs), 3)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=9, frameon=False)  # shared, below the panels
    fig.tight_layout(rect=(0, 0.01 + 0.014 * math.ceil(len(runs) / ncol), 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=260)
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

    names = source_names(runs)
    if names:
        path = args.out.with_name(f"{args.out.stem}_sources{args.out.suffix}")
        plot_sources(runs, labels, colors, names, path)
        cw = [max(len(n) + 2, 10) for n in names]
        print(f"\nval bpb by source, at each run's best row\n{'run':<{w}}" + "".join(f"{n:>{c}}" for n, c in zip(names, cw)))
        for r in runs:
            vals = [r["best"].get(f"val_{n}_bpb", math.nan) for n in names]
            print(f"{r['name']:<{w}}" + "".join(f"{v:>{c}.4f}" if v == v else f"{'-':>{c}}" for v, c in zip(vals, cw)))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))

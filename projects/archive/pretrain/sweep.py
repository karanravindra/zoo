"""Results of a hyperparameter sweep, laid out as a grid over the config values the runs differ in.

`uv run projects/pretrain/sweep.py ['sweep-*'] [--rows batch_size --cols lr] [--x tokens] [--out PATH]`

Takes the runs under `<repo>/runs/pretrain` whose directory name matches the glob (quote it) and puts each at the
crossing of two config keys: `--rows` down, `--cols` across (a sweep over one key needs only `--cols`). With neither
flag the axes are the config keys that differ, so a sweep needs no setup beyond naming its runs: ones started with
`--run-name "sweep-bs${batch}-lr${lr}"` are read back with `sweep.py 'sweep-*'`. Keys that move together count as one axis
(a warmup length derived from the batch size), and the batch size is the global one (OOM-restart splits are folded back).

The figure has three panels: a heatmap of every run's best val loss (the best cell outlined; colours saturate 10% above
it so a diverged run does not wash out the rest), the same numbers as best val loss against `--cols` with one line per
row, and the val curves against `--x` (tokens by default: runs with different batch sizes have different step counts),
coloured by row (a darker shade is a later column). Runs still training are included, scored by their best eval row so far and marked
`*`; cells of runs that have not logged an eval row yet are blank, so a sweep can be read while it is running, and a
finished run that never logged a finite loss is a `diverged` cell. After the table, each row's best column value is
listed with the number of tested values on either side (`--margin`), which is how to see whether a sweep brackets its
optimum or needs another decade. The loss
is `val_doc_loss` when every run has it, else `val_loss` (`--metric` picks one). The numbers are also printed as a table.
Use `compare.py` on the same runs for time, FLOPs, throughput and memory.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from compare import ALIASES, LABEL_ORDER, X_LABELS, has_rows, load, set_best
from matplotlib.patches import Rectangle
from tyro.conf import Positional

from zoo.paths import runs_root
from zoo.plot import plt, time_unit
from zoo.runs import read_metrics, sweep_axes

RUNS_ROOT = runs_root("pretrain")

# Config keys that do not change what was trained (compare.py's list), so they never count as a sweep axis.
IGNORE_KEYS = {"grad_accum", "run_name", "log_every", "sample_every", "sample_prompt", "sample_tokens", "sample_temperature", "sample_top_k",
               "val_source_batches", "eval_every", "grad_checkpoint"}
ROW_CMAPS = ["Blues", "Oranges", "Greens", "Reds", "Purples"]  # the hues of C0, C1, ...: a row has the same colour in every panel


@dataclass
class Args:
    match: Positional[str] = "*"  # glob on the run directory names under runs/pretrain, e.g. 'sweep-*' (quote it so the shell leaves it alone)
    rows: str | None = None  # config key down the grid; default: the first config key that differs (batch_size before lr, see compare.py's LABEL_ORDER)
    cols: str | None = None  # config key across the grid; default: the second. A one-key sweep is given as --cols only
    metric: Literal["auto", "val_loss", "val_doc_loss"] = "auto"  # loss to plot and rank by; auto = val_doc_loss when every run has it
    x: Literal["step", "tokens", "elapsed_s", "flops"] = "tokens"  # x axis of the curves panel; tokens lines up runs with different batch sizes
    margin: int = 2  # tested values wanted either side of each row's best column value (reported, not enforced)
    out: Path | None = None  # default: runs/pretrain/<match without wildcards>_grid.png


def fmt(key: str, v, cfg: dict, short: bool = False) -> str:
    """A config value for a tick or label: the batch size also as tokens per step (it is sequences per step in the config)."""
    if key == "batch_size":
        tok = f"{v * cfg['block_size'] / 1024:.0f}k tok"  # 1k = 1024 tokens, as in compare.py
        return f"bs{v}" if short else f"{v} seq\n{tok}"
    if isinstance(v, bool):
        return (ALIASES.get(key, key) if v else f"no {ALIASES.get(key, key)}") if short else str(v)
    text = f"{v:g}" if isinstance(v, (int, float)) else str(v)
    return f"{ALIASES.get(key, key)}{text}" if short and isinstance(v, (int, float)) else text


def ordered(values) -> list:
    """Distinct values, numbers by value first, then anything else as text."""
    return sorted(set(values), key=lambda v: (0, v) if isinstance(v, (int, float)) and not isinstance(v, bool) else (1, str(v)))


def main(args: Args):
    dirs, waiting = [], []
    for d in sorted(RUNS_ROOT.glob(args.match)):
        if not d.is_dir():
            continue
        if has_rows(d) or ((d / "results.json").exists() and read_metrics(d)):  # a finished run without a finite eval row diverged
            dirs.append(d)
        else:
            waiting.append(d.name)
    if not dirs:
        raise SystemExit(f"no runs under {RUNS_ROOT} match {args.match!r} with at least one eval row")
    if waiting:
        print(f"note: no eval row yet: {', '.join(waiting)}")
    runs = [load(d) for d in dirs]

    chosen = [k for k in (args.rows, args.cols) if k]
    for k in chosen:
        if k not in runs[0]["config"]:
            raise SystemExit(f"{k!r} is not a config key (see config.json in a run dir)")
    axes = sweep_axes(runs, IGNORE_KEYS, chosen + LABEL_ORDER)  # the chosen keys win ties, so a key tied to one of them drops out
    rows, cols = args.rows, args.cols
    if rows and not cols:
        raise SystemExit("--rows needs --cols (a sweep over one key is --cols alone)")
    if not chosen:
        if not axes:
            raise SystemExit(f"no config value differs between the {len(runs)} runs matching {args.match!r}")
        if len(axes) > 2:
            raise SystemExit(f"{len(axes)} config keys differ ({', '.join(axes)}); choose the grid's axes with --rows and --cols")
        rows, cols = (axes[0], axes[1]) if len(axes) == 2 else (None, axes[0])
    extra = [k for k in axes if k not in (rows, cols)]
    if extra:
        print(f"note: {', '.join(extra)} also differ between the runs and are not on the grid; a cell with several runs shows the best one")

    metric = args.metric
    if metric == "auto":
        metric = "val_doc_loss" if all("val_doc_loss" in r["metrics"] for r in runs) else "val_loss"
    missing = [r["name"] for r in runs if metric not in r["metrics"]]
    if missing:
        raise SystemExit(f"{metric} is not in metrics.csv of: {', '.join(missing)} (runs from before it was logged); use --metric val_loss")
    name = {"val_loss": "val loss", "val_doc_loss": "val loss per story"}[metric]
    for r in runs:
        if any(math.isfinite(v) for v in r["metrics"][metric]):
            set_best(r, metric)
        else:
            r["best"] = None  # never a finite eval: the run diverged

    def score(r: dict) -> float:
        return r["best"][metric] if r["best"] else math.inf

    # One run per cell: the best, when a rerun or a third key puts several in the same one.
    row_vals = ordered(r["config"][rows] for r in runs) if rows else [None]
    col_vals = ordered(r["config"][cols] for r in runs)
    cells: dict[tuple, dict] = {}
    for r in runs:
        key = (r["config"][rows] if rows else None, r["config"][cols])
        if key not in cells or score(r) < score(cells[key]):
            cells[key] = r
    grid = np.array([[score(cells[(rv, cv)]) if (rv, cv) in cells else math.nan for cv in col_vals] for rv in row_vals])  # nan: not run, inf: diverged
    if not np.isfinite(grid).any():
        raise SystemExit("every run diverged")
    cfg = runs[0]["config"]  # block_size and the like are the same in every run of a sweep
    row_labels = [fmt(rows, v, cfg) if rows else "" for v in row_vals]
    col_labels = [fmt(cols, v, cfg) for v in col_vals]
    best_rc = np.unravel_index(np.nanargmin(grid), grid.shape)
    training = any(r["results"] is None for r in cells.values())

    def label(rv, cv) -> str:
        return " ".join(([fmt(rows, rv, cfg, short=True)] if rows else []) + [fmt(cols, cv, cfg, short=True)])

    live = {key: r for key, r in cells.items() if r["best"] is not None}  # the runs with a curve to draw
    legend_rows = max(sum((rv, cv) in live for cv in col_vals) for rv in row_vals) if rows else math.ceil(len(live) / 6)  # one legend column per row
    height = 4.8 + 0.2 * legend_rows
    fig, (a_grid, a_line, a_curve) = plt.subplots(1, 3, figsize=(17, height), gridspec_kw={"width_ratios": [1, 1, 1.3]})

    # Panel 1: heatmap. Colours saturate just above the best so one diverged run does not flatten the rest; the text keeps the real value.
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("0.92")
    lo = np.nanmin(grid)
    top = min(np.nanmax(grid), lo * 1.3)  # a diverged run runs off the line charts instead of squashing the rest
    norm = plt.Normalize(lo, min(np.nanmax(grid), lo * 1.1))
    a_grid.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto")
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            if (rv, cv) not in cells:
                a_grid.text(j, i, "-", ha="center", va="center", color="0.5")
                continue
            v, r = grid[i, j], cells[(rv, cv)]
            if not math.isfinite(v):
                a_grid.text(j, i, "diverged", ha="center", va="center", color="tab:red", fontsize=9)
                continue
            shade = cmap(norm(v))
            dark = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2] < 0.5
            a_grid.text(j, i, f"{v:.4f}" + ("*" if r["results"] is None else ""), ha="center", va="center", fontsize=10,
                        color="white" if dark else "black", fontweight="bold" if (i, j) == tuple(best_rc) else "normal")
    a_grid.add_patch(Rectangle((best_rc[1] - 0.5, best_rc[0] - 0.5), 1, 1, fill=False, ec="tab:red", lw=2.5))
    a_grid.set_xticks(range(len(col_vals)), col_labels)
    a_grid.set_yticks(range(len(row_vals)), row_labels)
    a_grid.set_xlabel(cols)
    if rows:
        a_grid.set_ylabel(rows)
    a_grid.set_title(f"Best {name}" + (" (* still training)" if training else ""))

    # Panel 2: best loss against the column key, one line per row.
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in col_vals)
    xs = col_vals if numeric else list(range(len(col_vals)))
    for i, rv in enumerate(row_vals):
        pts = [(x, grid[i, j]) for j, x in enumerate(xs) if math.isfinite(grid[i, j])]
        a_line.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=f"C{i}", label=row_labels[i].replace("\n", ", ") or None)
        if pts:  # each row's best, ringed
            a_line.plot(*min(pts, key=lambda p: p[1]), marker="*", ms=16, mfc="none", mec=f"C{i}", mew=2, ls="none")
        dead = [x for j, x in enumerate(xs) if (rv, col_vals[j]) in cells and grid[i, j] == math.inf]
        a_line.plot(dead, [top] * len(dead), marker="x", ms=9, mew=2, color=f"C{i}", ls="none")  # diverged: off the scale
    if numeric and min(col_vals) > 0 and max(col_vals) / min(col_vals) >= 4:
        a_line.set_xscale("log")
    a_line.set_xticks(xs, col_labels)  # the swept values themselves, not log-axis ticks
    a_line.minorticks_off()
    a_line.set_ylim(lo * 0.99, top * 1.02)
    a_line.set_xlabel(cols)
    a_line.set_ylabel(f"best {name}")
    a_line.set_title(f"Best {name} vs {cols}")
    a_line.grid(alpha=0.3)
    if rows:
        a_line.legend(title=rows, fontsize=9)

    # Panel 3: the val curves, a hue per row (as in panel 2) that darkens with the column; a one-key sweep has no rows, so it shades by column.
    t_unit, t_div = time_unit(max(max(r["metrics"]["elapsed_s"]) for r in cells.values()))
    div = t_div if args.x == "elapsed_s" else 1.0
    shades = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(col_vals)))
    handles, labels = [], []
    for i, rv in enumerate(row_vals):
        row_handles = []
        for j, cv in enumerate(col_vals):
            if (rv, cv) not in live:
                continue
            m = live[(rv, cv)]["metrics"]
            pts = [(u / div, v) for u, v in zip(m[args.x], m[metric]) if not math.isnan(v)]
            color = plt.get_cmap(ROW_CMAPS[i % len(ROW_CMAPS)])(0.3 + 0.65 * j / max(len(col_vals) - 1, 1)) if rows else shades[j]
            line, = a_curve.plot([p[0] for p in pts], [p[1] for p in pts], c=color, lw=1.4, marker="." if len(pts) < 3 else None)
            row_handles.append((line, label(rv, cv)))
        row_handles += [(plt.Line2D([], [], alpha=0), "")] * (legend_rows - len(row_handles)) if rows else []  # pad so each legend column is one row
        handles += [h for h, _ in row_handles]
        labels += [t for _, t in row_handles]
    a_curve.set_ylim(lo * 0.99, top * 1.02)
    a_curve.set_xlabel(f"training time ({t_unit})" if args.x == "elapsed_s" else X_LABELS[args.x])
    a_curve.set_ylabel(name)
    a_curve.set_title(f"{name.capitalize()} vs {X_LABELS[args.x]}")
    a_curve.grid(alpha=0.3)

    fig.suptitle(f"{args.match}: {len(runs)} runs")
    fig.legend(handles, labels, loc="lower center", ncol=len(row_vals) if rows else min(len(handles), 6), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, (0.25 + 0.2 * legend_rows) / height, 1, 1))
    out = args.out or RUNS_ROOT / f"{re.sub(r'[^A-Za-z0-9_.-]+', '', args.match).strip('-_.') or 'all'}_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260)
    plt.close(fig)

    # The heatmap's numbers as a table.
    w = max(len(s.replace("\n", " ")) for s in row_labels + [rows or ""]) + 2
    cw = max(8, *(len(s.replace("\n", " ")) for s in col_labels)) + 2  # a cell is up to 8 characters: diverged
    print(f"best {name} ({metric}) at each run's best eval row; * = still training; rows: {rows or '-'}, cols: {cols}")
    print(f"{'':<{w}}" + "".join(f"{s.replace(chr(10), ' '):>{cw}}" for s in col_labels))
    for i, rv in enumerate(row_vals):
        line = f"{row_labels[i].replace(chr(10), ' '):<{w}}"
        for j, cv in enumerate(col_vals):
            r = cells.get((rv, cv))
            line += f"{('diverged' if grid[i, j] == math.inf else f'{grid[i, j]:.4f}' + ('*' if r['results'] is None else '')) if r else '-':>{cw}}"
        print(line)
    print(f"best: {label(row_vals[best_rc[0]], col_vals[best_rc[1]])}  {name} {grid[best_rc]:.4f}  ({cells[(row_vals[best_rc[0]], col_vals[best_rc[1]])]['name']})")
    print(f"\nbest {cols} {'for each ' + rows if rows else 'overall'}, and the values tested either side of it (want {args.margin}; a diverged run counts, it brackets the optimum):")
    for i, rv in enumerate(row_vals):
        if not np.isfinite(grid[i]).any():
            continue
        jb = int(np.nanargmin(grid[i]))
        below = sum((rv, col_vals[k]) in cells for k in range(jb))
        above = sum((rv, col_vals[k]) in cells for k in range(jb + 1, len(col_vals)))
        short = [side for side, n in (("below", below), ("above", above)) if n < args.margin]
        print(f"  {row_labels[i].replace(chr(10), ' ') or 'all runs':<{w}}{fmt(cols, col_vals[jb], cfg, short=True):<10}{below} below, {above} above"
              + (f"  -> needs more {' and '.join(short)}" if short else ""))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))

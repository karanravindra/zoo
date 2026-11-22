"""Compare DC-AE runs side by side.

`uv run projects/celeba_gen/compare_dcae.py [run_dir ...] [--phase N] [--out PATH]`

Defaults to every run under `<repo>/runs/celeba_gen/dcae-phase<N>` (`--phase`, default 1) and writes
`dcae-phase<N>_comparison.png` next to it. Pass run dirs instead to compare any mix of phases, e.g. one chain
phase1 -> phase2 -> phase3; the plot is then `dcae_comparison.png`.

Each run is labelled by the config values that differ between the runs (e.g. `lr=0.0004, model.downsample=16`).
Validation and test metrics (rFID first, then MSE, PSNR, SSIM and LPIPS) are drawn at every evaluation resolution:
colour is the run, line style (and bar hatch) the resolution, and the thick line is the run's training resolution. Phases 2 and 3 start at epoch 0, the init
checkpoint before any training. Runs still training (no `results.json` yet) get curves but no test bars.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from matplotlib.lines import Line2D
from tyro.conf import Positional

from zoo.paths import runs_root
from zoo.plot import plt
from zoo.runs import diff_labels, sort_runs
from zoo.runs import load_run as load_raw_run

RUNS_ROOT = runs_root("celeba_gen")

IGNORE_KEYS = {"log_every", "compile", "num_workers", "data_on_device", "dev"}
LINESTYLES = ["-", "--", ":", "-."]  # one per evaluation resolution
HATCHES = ["", "//", "..", "xx"]
TABLE_METRICS = ("rfid", "psnr", "ssim", "lpips")


def load_run(run: Path) -> dict:
    r = load_raw_run(run)
    # flatten the nested model config (phase 1 only) so each architecture flag is its own label key
    r["config"] = {
        k2: v2 for k, v in r["config"].items() for k2, v2 in ({f"{k}.{a}": b for a, b in v.items()} if isinstance(v, dict) else {k: v}).items()
    }
    r["resolutions"] = sorted({int(m.group(1)) for k in r["metrics"] if (m := re.fullmatch(r"val_mse_(\d+)", k))})
    return r


def run_labels(runs: list[dict]) -> list[str]:
    """The config values that differ between the runs. Across phases nearly every key differs (and only phase 1 has
    `model.*`), so there each run is `phase N`, plus what differs among the runs of that phase."""
    phases = sorted({r["config"]["phase"] for r in runs})
    if len(phases) == 1:
        return diff_labels(runs, IGNORE_KEYS)
    labels = [""] * len(runs)
    for p in phases:
        idx = [i for i, r in enumerate(runs) if r["config"]["phase"] == p]
        for i, label in zip(idx, diff_labels([runs[i] for i in idx], IGNORE_KEYS)):
            labels[i] = f"phase {p}" if len(idx) == 1 else f"phase {p}: {label}"
    return labels


@dataclass
class Args:
    runs: Positional[list[Path]] = field(default_factory=list)  # run dirs, any phases (default: all under runs/celeba_gen/dcae-phase<N>)
    phase: Literal[1, 2, 3] = 1  # which phase's runs to compare when no run dirs are given
    out: Path | None = None  # default: runs/celeba_gen/dcae-phase<N>_comparison.png, or dcae_comparison.png with run dirs


def curves(ax, runs: list[dict], colors: list[str], key: str, all_res: list[int]) -> None:
    for run, c in zip(runs, colors):
        m = run["metrics"]
        for r in run["resolutions"]:
            if f"val_{key}_{r}" not in m:
                continue
            thick = r == run["config"]["resolution"]
            ax.plot(m["epoch"], m[f"val_{key}_{r}"], c=c, ls=LINESTYLES[all_res.index(r) % len(LINESTYLES)], lw=2.2 if thick else 1.1)


def bars(ax, done: list[tuple[dict, str]], key: str, all_res: list[int], truncate: bool = False) -> None:
    """Test metric `key` of each finished run, one bar per resolution (hatched), coloured by run."""
    n = len(all_res)
    w = 0.8 / n
    shown = []
    for j, r in enumerate(all_res):
        pos, vals, cols = [], [], []
        for i, (run, c) in enumerate(done):
            v = run["results"].get(f"test_{key}_{r}")
            if v is not None:
                pos.append(i + (j - (n - 1) / 2) * w)
                vals.append(v)
                cols.append(c)
        ax.bar(pos, vals, w, color=cols, hatch=HATCHES[j % len(HATCHES)], edgecolor="k", linewidth=0.4)
        for x, v in zip(pos, vals):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
        shown += vals
    ax.set_xticks(range(len(done)))
    ax.set_xticklabels([run["name"] for run, _ in done], fontsize=8, rotation=30, ha="right")  # short: config labels overlap
    if shown and truncate:  # bars far from zero (PSNR, SSIM): zoom in, the values are printed on the bars
        lo, hi = min(shown), max(shown)
        ax.set_ylim(lo - 0.3 * max(hi - lo, 1e-3), hi + 0.15 * max(hi - lo, 1e-3))
    ax.margins(y=0.15)


def main():
    args = tyro.cli(Args, description=__doc__)

    if args.runs:
        dirs, default_out = args.runs, RUNS_ROOT / "dcae_comparison.png"
    else:
        stage = RUNS_ROOT / f"dcae-phase{args.phase}"
        if not stage.is_dir():
            raise SystemExit(f"No runs under {stage}; train phase{args.phase} first, or pass run dirs.")
        dirs, default_out = [d for d in stage.iterdir() if d.is_dir()], RUNS_ROOT / f"dcae-phase{args.phase}_comparison.png"
    out = args.out or default_out
    dirs = [d for d in dirs if (d / "metrics.csv").exists() and (d / "metrics.csv").read_text().count("\n") > 1]  # skip runs with no epoch yet
    if not dirs:
        raise SystemExit("No runs with metrics to compare.")
    runs = sort_runs([load_run(d) for d in dirs], IGNORE_KEYS)  # legend/colour order: by the config values that differ
    runs.sort(key=lambda r: r["config"]["phase"])  # stable: a chain reads phase 1, 2, 3
    labels = run_labels(runs)
    colors = [f"C{i}" for i in range(len(runs))]
    all_res = sorted({r for run in runs for r in run["resolutions"]})

    fig, ax = plt.subplots(5, 2, figsize=(13, 21))
    for run, label, c in zip(runs, labels, colors):
        m = run["metrics"]
        ax[0, 0].plot(m["epoch"], m["train_loss"], c=c, label=label)
    curves(ax[0, 1], runs, colors, "rfid", all_res)
    curves(ax[1, 0], runs, colors, "mse", all_res)
    curves(ax[1, 1], runs, colors, "psnr", all_res)
    curves(ax[2, 0], runs, colors, "ssim", all_res)
    curves(ax[2, 1], runs, colors, "lpips", all_res)

    done = [(r, c) for r, c in zip(runs, colors) if r["results"]]  # runs still training have no test results
    bars(ax[3, 0], done, "rfid", all_res)
    bars(ax[3, 1], done, "lpips", all_res)
    bars(ax[4, 0], done, "psnr", all_res, truncate=True)
    bars(ax[4, 1], done, "ssim", all_res, truncate=True)

    titles = ["Train loss (weights differ between runs)", "Val rFID", "Val MSE", "Val PSNR (dB)", "Val SSIM", "Val LPIPS", "Test rFID", "Test LPIPS", "Test PSNR (dB)", "Test SSIM"]
    for a, title in zip(ax.flat, titles):
        a.set_title(title)
        a.grid(alpha=0.3)
        if not title.startswith("Test"):
            a.set_xlabel("epoch")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_yscale("log")
    ax[0, 0].legend(fontsize=8)
    res_handles = [Line2D([], [], c="k", ls=LINESTYLES[i % len(LINESTYLES)], label=f"{r} px") for i, r in enumerate(all_res)]
    ax[0, 1].legend(handles=res_handles + [Line2D([], [], c="k", lw=2.2, label="thick: training res")], fontsize=8)

    fig.suptitle("celeba_gen DC-AE runs")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)

    head = "".join(f"{f'{m}@{r}':>11}" for r in all_res for m in TABLE_METRICS)
    print(f"{'run':<17}{'phase':>6}{head}  config")
    for r, label in zip(runs, labels):
        res = r["results"]
        if res is None:
            print(f"{r['name']:<17}{r['config']['phase']:>6}{'(training: epoch ' + str(int(r['metrics']['epoch'][-1])) + ')':>34}  {label}")
            continue
        cells = "".join(f"{res[f'test_{m}_{x}']:>11.4f}" if f"test_{m}_{x}" in res else f"{'-':>11}" for x in all_res for m in TABLE_METRICS)
        print(f"{r['name']:<17}{r['config']['phase']:>6}{cells}  {label}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

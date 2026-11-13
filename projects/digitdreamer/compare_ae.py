"""Compare autoencoder runs side by side.

`uv run projects/digitdreamer/compare_ae.py [run_dir ...] [--out PATH]`

Defaults to every run under `<repo>/runs/digitdreamer/ae` and writes `ae_comparison.png` next to it.
Each run is labelled by the config values that differ between the runs (e.g. `latent_channels=4, mask_prob=0.5`).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS_ROOT  # noqa: E402

IGNORE_KEYS = {"log_every", "num_workers"}


def read_metrics(run: Path) -> dict[str, list[float]]:
    path = next(run.glob("metrics.*"))  # early runs wrote metrics.tsv (comma-separated); later ones metrics.csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def load_run(run: Path) -> dict:
    return {
        "name": run.name,
        "config": json.loads((run / "config.json").read_text())["config"],
        "results": json.loads((run / "results.json").read_text()),
        "metrics": read_metrics(run),
    }


def make_labels(runs: list[dict]) -> list[str]:
    """Label each run with the config keys that are not identical across runs; fall back to the run name."""
    keys = sorted({k for r in runs for k in r["config"]} - IGNORE_KEYS)
    varying = [k for k in keys if len({json.dumps(r["config"].get(k)) for r in runs}) > 1]
    labels = []
    for r in runs:
        parts = [f"{k}={r['config'].get(k, '-')}" for k in varying]
        labels.append(", ".join(parts) or r["name"])
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", type=Path, help="AE run dirs (default: all under runs/digitdreamer/ae)")
    p.add_argument("--out", type=Path, default=RUNS_ROOT / "ae_comparison.png")
    args = p.parse_args()

    dirs = args.runs or [d for d in (RUNS_ROOT / "ae").iterdir() if d.is_dir()]
    runs = [load_run(d) for d in sorted(dirs, key=lambda d: d.name)]  # legend/colour order = directory order
    labels = make_labels(runs)
    colors = [f"C{i}" for i in range(len(runs))]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for run, label, c in zip(runs, labels, colors):
        m = run["metrics"]
        ax[0, 0].plot(m["epoch"], m["train_mse"], c=c, label=label)
        ax[0, 1].plot(m["epoch"], m["val_mse"], c=c, label=label)
        if "val_mse_global" in m:  # only logged for runs trained with detail-channel masking support
            ax[0, 1].plot(m["epoch"], m["val_mse_global"], c=c, ls="--", label=f"{label} (global half only)")
        ax[0, 2].plot(m["epoch"], m["val_psnr"], c=c, label=label)
        ax[1, 0].plot(m["epoch"], m["val_ssim"], c=c, label=label)

    for a, title in zip(ax.flat, ["Train MSE", "Val MSE (dashed: detail channels zeroed)", "Val PSNR (dB)", "Val SSIM"]):
        a.set_title(title)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[0, 0].set_yscale("log")
    ax[0, 1].set_yscale("log")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].legend(fontsize=7)

    xs = list(range(len(runs)))
    for a, key, title in ((ax[1, 1], "test_psnr", "Test PSNR (dB)"), (ax[1, 2], "test_ssim", "Test SSIM")):
        vals = [r["results"][key] for r in runs]
        a.bar(xs, vals, color=colors)
        a.set_title(title)
        a.set_xticks(xs)
        a.set_xticklabels([r["name"] for r in runs], fontsize=8, rotation=30, ha="right")  # short: config labels overlap
        for x, v in zip(xs, vals):
            a.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax[1, 2].set_ylim(min(r["results"]["test_ssim"] for r in runs) - 0.02, 1)

    fig.suptitle("DigitDreamer AE runs")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)

    print(f"{'run':<17}{'test_mse':>10}{'psnr':>8}{'ssim':>8}  config")
    for r, label in zip(runs, labels):
        res = r["results"]
        print(f"{r['name']:<17}{res['test_mse']:>10.5f}{res['test_psnr']:>8.2f}{res['test_ssim']:>8.4f}  {label}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

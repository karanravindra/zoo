"""Compare autoencoder runs side by side.

`uv run projects/digitdreamer/compare_ae.py [run_dir ...] [--out PATH]`

Defaults to every run under `<repo>/runs/digitdreamer/ae` and writes `ae_comparison.png` next to it.
Each run is labelled by the config values that differ between the runs (e.g. `latent_channels=4, mask_prob=0.5`).
"""

from dataclasses import dataclass, field
from pathlib import Path

import tyro
from tyro.conf import Positional

from zoo.paths import runs_root
from zoo.plot import plt
from zoo.runs import diff_labels, load_run, sort_runs

RUNS_ROOT = runs_root("digitdreamer")

IGNORE_KEYS = {"log_every", "num_workers"}


@dataclass
class Args:
    runs: Positional[list[Path]] = field(default_factory=list)  # AE run dirs (default: all under runs/digitdreamer/ae)
    out: Path = RUNS_ROOT / "ae_comparison.png"


def main():
    args = tyro.cli(Args, description=__doc__)

    dirs = args.runs or [d for d in (RUNS_ROOT / "ae").iterdir() if d.is_dir()]
    runs = sort_runs([load_run(d) for d in dirs], IGNORE_KEYS)  # legend/colour order: by the config values that differ
    labels = diff_labels(runs, IGNORE_KEYS)
    colors = [f"C{i}" for i in range(len(runs))]

    fig, ax = plt.subplots(3, 2, figsize=(13, 13))  # 3 rows x 2 columns on a square figure
    for run, label, c in zip(runs, labels, colors):
        m = run["metrics"]
        ax[0, 0].plot(m["epoch"], m["train_mse"], c=c, label=label)
        ax[0, 1].plot(m["epoch"], m["val_mse"], c=c, label=label)
        if "val_mse_global" in m:  # only logged for runs trained with detail-channel masking support
            ax[0, 1].plot(m["epoch"], m["val_mse_global"], c=c, ls="--", label=f"{label} (global half only)")
        ax[1, 0].plot(m["epoch"], m["val_psnr"], c=c, label=label)
        ax[1, 1].plot(m["epoch"], m["val_ssim"], c=c, label=label)

    for a, title in zip(ax.flat, ["Train MSE", "Val MSE (dashed: detail channels zeroed)", "Val PSNR (dB)", "Val SSIM"]):
        a.set_title(title)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[0, 0].set_yscale("log")
    ax[0, 1].set_yscale("log")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].legend(fontsize=7)

    xs = list(range(len(runs)))
    for a, key, title in ((ax[2, 0], "test_psnr", "Test PSNR (dB)"), (ax[2, 1], "test_ssim", "Test SSIM")):
        vals = [r["results"][key] for r in runs]
        a.bar(xs, vals, color=colors)
        a.set_title(title)
        a.set_xticks(xs)
        a.set_xticklabels([r["name"] for r in runs], fontsize=8, rotation=30, ha="right")  # short: config labels overlap
        for x, v in zip(xs, vals):
            a.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax[2, 1].set_ylim(min(r["results"]["test_ssim"] for r in runs) - 0.02, 1)

    fig.suptitle("DigitDreamer AE runs")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=260)
    plt.close(fig)

    print(f"{'run':<17}{'test_mse':>10}{'psnr':>8}{'ssim':>8}  config")
    for r, label in zip(runs, labels):
        res = r["results"]
        print(f"{r['name']:<17}{res['test_mse']:>10.5f}{res['test_psnr']:>8.2f}{res['test_ssim']:>8.4f}  {label}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

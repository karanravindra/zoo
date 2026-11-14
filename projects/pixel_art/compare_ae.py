"""Compare autoencoder runs side by side.

`uv run projects/pixel_art/compare_ae.py [run_dir ...] [--out PATH]`

Defaults to every run under `<repo>/runs/pixel_art/ae` and writes `ae_comparison.png` next to it.
Each run is labelled by the config values that differ between the runs (e.g. `hflip_prob=0.5, mask_prob=0.5`).
Runs still training (no `results.json` yet) get curves but no test bars.
"""

import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tyro
from tyro.conf import Positional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS_ROOT  # noqa: E402
from ldm import AEConfig  # noqa: E402

IGNORE_KEYS = {"log_every", "compile"}


def read_metrics(run: Path) -> dict[str, list[float]]:
    with open(run / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def load_run(run: Path) -> dict:
    results = run / "results.json"
    return {
        "name": run.name,
        # from_dict fills keys older runs lack (with defaults, or the legacy value where behaviour changed)
        "config": asdict(AEConfig.from_dict(json.loads((run / "config.json").read_text())["config"])),
        "results": json.loads(results.read_text()) if results.exists() else None,
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


@dataclass
class Args:
    runs: Positional[list[Path]] = field(default_factory=list)  # AE run dirs (default: all under runs/pixel_art/ae)
    out: Path = RUNS_ROOT / "ae_comparison.png"


def main():
    args = tyro.cli(Args, description=__doc__)

    dirs = args.runs or [d for d in (RUNS_ROOT / "ae").iterdir() if d.is_dir()]
    dirs = [d for d in dirs if (d / "metrics.csv").exists() and (d / "metrics.csv").read_text().count("\n") > 1]  # skip runs with no epoch yet
    runs = [load_run(d) for d in sorted(dirs, key=lambda d: d.name)]  # legend/colour order = directory order
    labels = make_labels(runs)
    colors = [f"C{i}" for i in range(len(runs))]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for run, label, c in zip(runs, labels, colors):
        m = run["metrics"]
        ax[0, 0].plot(m["epoch"], m["train_mse"], c=c, label=label)
        ax[0, 1].plot(m["epoch"], m["val_mse"], c=c, label=label)
        if run["config"]["mask_prob"] > 0:  # without masking in training, the global-only recon is meaningless
            ax[0, 1].plot(m["epoch"], m["val_mse_global"], c=c, ls="--", label=f"{label} (global half only)")
        ax[0, 2].plot(m["epoch"], m["val_psnr"], c=c, label=label)
        ax[1, 0].plot(m["epoch"], m["val_ssim"], c=c, label=label)

    val_title = "Val MSE (dashed: detail channels zeroed)" if any(r["config"]["mask_prob"] > 0 for r in runs) else "Val MSE"
    for a, title in zip(ax.flat, ["Train MSE", val_title, "Val PSNR (dB)", "Val SSIM"]):
        a.set_title(title)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[0, 0].set_yscale("log")
    ax[0, 1].set_yscale("log")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].legend(fontsize=7)

    done = [(r, c) for r, c in zip(runs, colors) if r["results"]]  # runs still training have no test results
    xs = list(range(len(done)))
    for a, key, title in ((ax[1, 1], "test_psnr", "Test PSNR (dB)"), (ax[1, 2], "test_ssim", "Test SSIM")):
        vals = [r["results"][key] for r, _ in done]
        a.bar(xs, vals, color=[c for _, c in done])
        a.set_title(title)
        a.set_xticks(xs)
        a.set_xticklabels([r["name"] for r, _ in done], fontsize=8, rotation=30, ha="right")  # short: config labels overlap
        for x, v in zip(xs, vals):
            a.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    if done:
        ax[1, 2].set_ylim(min(r["results"]["test_ssim"] for r, _ in done) - 0.02, 1)

    fig.suptitle("pixel_art AE runs")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)

    print(f"{'run':<17}{'test_mse':>10}{'psnr':>8}{'ssim':>8}  config")
    for r, label in zip(runs, labels):
        res = r["results"]
        if res is None:
            print(f"{r['name']:<17}{'(training: epoch ' + str(int(r['metrics']['epoch'][-1])) + ')':>26}  {label}")
            continue
        print(f"{r['name']:<17}{res['test_mse']:>10.5f}{res['test_psnr']:>8.2f}{res['test_ssim']:>8.4f}  {label}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

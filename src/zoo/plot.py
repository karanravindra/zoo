"""Plot helpers. Importing this module selects the headless Agg backend; use `from zoo.plot import plt`."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt", "plot_curves", "time_unit"]


def time_unit(max_seconds: float) -> tuple[str, float]:
    """Unit name and its size in seconds for a time axis that runs up to `max_seconds`: s, min or h."""
    if max_seconds >= 3 * 3600:
        return "h", 3600.0
    if max_seconds >= 3 * 60:
        return "min", 60.0
    return "s", 1.0


def plot_curves(
    rows: list[dict], series: list[str], path: Path, ylabel: str = "loss", log: bool = False, x: str = "epoch"
) -> None:
    """One line per `series` key against `x`. Rows that lack a key (or hold None / nan for it) are skipped for
    that line, so a column logged only on some rows, like val_loss on eval rows, still plots as a line."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in series:
        pts = [(r[x], r[key]) for r in rows if r.get(key) is not None and r[key] == r[key]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label=key)
    ax.set(xlabel=x, ylabel=ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

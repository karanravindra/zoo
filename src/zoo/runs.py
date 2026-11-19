"""Run directories, checkpoint lookup, metrics.csv logging, and loading finished runs back for comparison."""

from __future__ import annotations

import atexit
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from zoo.paths import REPO_ROOT, runs_root


def git_commit() -> str | None:
    """HEAD commit of the repo, with a `-dirty` suffix if the working tree has changes."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
        return out + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def dev_run_dir(project: str) -> Path:
    """A throwaway run directory outside `runs/`, deleted when the process exits (a script's `--dev` flag).

    The script writes its usual outputs there, so every code path still runs, but nothing is kept: the run never
    shows up in `runs/`, compare scripts, the runs viewer or `latest_checkpoint`."""
    run_dir = Path(tempfile.mkdtemp(prefix=f"zoo-dev-{project}-"))
    atexit.register(shutil.rmtree, run_dir, ignore_errors=True)
    print(f"Dev run: nothing is logged to runs/ (scratch outputs in {run_dir}, deleted on exit)")
    return run_dir


class _Tee:
    """File-like that writes to the original stream and to the log file."""

    def __init__(self, stream, log) -> None:
        self._stream, self._log = stream, log

    def write(self, text: str) -> int:
        self._log.write(text)
        self._log.flush()
        return self._stream.write(text)

    def flush(self) -> None:
        self._log.flush()
        self._stream.flush()

    def __getattr__(self, name):  # isatty, fileno, encoding, ... (so tqdm and friends keep working)
        return getattr(self._stream, name)


def tee_console(run_dir: Path, name: str = "console.log") -> Path:
    """Copy everything printed to stdout and stderr from now on into `<run_dir>/console.log` (appended, so a
    resumed run keeps its earlier output). `new_run_dir` calls this; call it yourself for hand-made run dirs."""
    path = Path(run_dir) / name
    log = open(path, "a", buffering=1)
    atexit.register(log.close)
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    return path


def new_run_dir(
    project: str, config: dict, device, stage: str | None = None, dev: bool = False, name: str | None = None, **extra
) -> Path:
    """Create `runs/<project>[/<stage>]/<timestamp>/` and write config.json (config, git commit, device, extras).

    `name` replaces the timestamp as the directory name; if that directory already exists, `-2`, `-3`, ... is
    appended, so an existing run is never written into. With `dev`, the directory is a `dev_run_dir` instead.
    Console output (stdout and stderr) is also saved to `console.log` in the run directory."""
    if dev:
        run_dir = dev_run_dir(project)
    else:
        base = runs_root(project) / stage if stage else runs_root(project)
        run_dir = base / (name or datetime.now().strftime("%Y%m%d-%H%M%S"))
        n = 2
        while name and run_dir.exists():
            run_dir, n = base / f"{name}-{n}", n + 1
        run_dir.mkdir(parents=True, exist_ok=False)
    meta = {"config": config, "git_commit": git_commit(), "device": str(device), **extra}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    tee_console(run_dir)
    print(f"Run directory: {run_dir}")
    return run_dir


def latest_checkpoint(project: str, name: str, stage: str | None = None) -> Path:
    """Most recently written `<run>/<name>` under a project (or stage), by mtime so hand-renamed run dirs work."""
    base = runs_root(project) / stage if stage else runs_root(project)
    runs = sorted(base.glob(f"*/{name}"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No {name} found under {base}; train that stage first.")
    return runs[-1]


def read_metrics(run: Path) -> dict[str, list[float]]:
    """A run's metrics file as columns ({} if it has no rows yet); blank cells (a column that is only filled on
    some rows, e.g. val_loss on eval rows) become nan."""
    path = next(Path(run).glob("metrics.*"))  # early runs wrote metrics.tsv (comma-separated); later ones metrics.csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    num = lambda v: float(v) if v not in ("", None) else float("nan")  # noqa: E731
    return {k: [num(r[k]) for r in rows] for k in rows[0]} if rows else {}


def load_run(run: Path) -> dict:
    """{name, path, config, results, metrics} of a run dir; `results` is None while the run is still training."""
    run = Path(run)
    results = run / "results.json"
    return {
        "name": run.name,
        "path": run,
        "config": json.loads((run / "config.json").read_text())["config"],
        "results": json.loads(results.read_text()) if results.exists() else None,
        "metrics": read_metrics(run),
    }


def varying_keys(runs: list[dict], ignore: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Config keys (sorted by name) whose value is not the same in every run (from `load_run`)."""
    keys = sorted({k for r in runs for k in r["config"]} - set(ignore))
    return [k for k in keys if len({json.dumps(r["config"].get(k)) for r in runs}) > 1]


def diff_labels(runs: list[dict], ignore: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Label each run (from `load_run`) with the config keys that differ between the runs; fall back to its name."""
    varying = varying_keys(runs, ignore)
    return [", ".join(f"{k}={r['config'].get(k, '-')}" for k in varying) or r["name"] for r in runs]


def _order(v) -> tuple:
    """Sort key for one config value: missing first, then numbers by value, then strings, then anything else."""
    if v is None:
        return (0, 0)
    if isinstance(v, (bool, int, float)):
        return (1, v)
    return (2, v) if isinstance(v, str) else (3, json.dumps(v, sort_keys=True))


def sort_runs(runs: list[dict], ignore: set[str] | frozenset[str] = frozenset()) -> list[dict]:
    """Order runs (from `load_run`) as a sweep: by the config values that differ between them, compared as numbers
    where they are numbers (so max_steps 2000 < 5000 < 10000), key by key in the order `diff_labels` prints them.
    Runs with identical configs keep their relative order by name (timestamps, so oldest first)."""
    varying = varying_keys(runs, ignore)
    return sorted(runs, key=lambda r: ([_order(r["config"].get(k)) for k in varying], r["name"]))


class MetricsLogger:
    """metrics.csv writer: header on open, one flushed row per `log`; `rows` keeps everything logged.

    `float_fmt` (e.g. ".6g") formats float cells in the file only; `rows` keeps the raw values."""

    def __init__(self, path: Path, fields: list[str], float_fmt: str | None = None, **writer_kw) -> None:
        self.rows: list[dict] = []
        self.float_fmt = float_fmt
        self._f = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=fields, **writer_kw)
        self._writer.writeheader()

    def log(self, row: dict) -> dict:
        self.rows.append(row)
        if self.float_fmt:
            row = {k: (format(v, self.float_fmt) if isinstance(v, float) else v) for k, v in row.items()}
        self._writer.writerow(row)
        self._f.flush()
        return row

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

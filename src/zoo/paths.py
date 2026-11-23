"""Repo-level paths shared by every project.

`DATA_ROOT` and the runs root come from `ZOO_DATA_ROOT` / `ZOO_RUNS_ROOT` when set (a Lightning Studio points them at
its persistent storage), and default to `<repo>/data` and `<repo>/runs` (symlinks to the big disk on the home box).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ZOO_DATA_ROOT", REPO_ROOT / "data"))
RUNS_ROOT = Path(os.environ.get("ZOO_RUNS_ROOT", REPO_ROOT / "runs"))


def runs_root(project: str) -> Path:
    """`<runs root>/<project>` (gitignored)."""
    return RUNS_ROOT / project

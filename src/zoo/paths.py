"""Repo-level paths shared by every project."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


def runs_root(project: str) -> Path:
    """`<repo>/runs/<project>` (gitignored)."""
    return REPO_ROOT / "runs" / project

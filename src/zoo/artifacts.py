"""Publish files and versioned directories on the same filesystem."""

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def write_json(path: Path, value: dict, *, indent: int = 2) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, indent=indent)
            f.close()
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


@contextmanager
def staged_build(root: Path) -> Iterator[Path]:
    """Build at a unique path, retaining successful builds for existing readers."""
    builds = root / ".builds"
    builds.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=builds)).resolve()
    try:
        yield stage
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def publish_build(root: Path, stage: Path) -> None:
    """Switch the current build with one rename, leaving legacy files readable."""
    pointer = stage.with_suffix(".link")
    try:
        pointer.symlink_to(stage.relative_to(root.resolve()), target_is_directory=True)
        pointer.replace(root / "current")
    finally:
        pointer.unlink(missing_ok=True)


def current_build(root: Path) -> Path:
    pointer = root / "current"
    return pointer.resolve(strict=True) if pointer.is_symlink() else root

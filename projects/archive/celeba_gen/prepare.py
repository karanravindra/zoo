"""Download CelebA-HQ and check the extracted copy.

Run once, from anywhere: `uv run projects/celeba_gen/prepare.py`

Fetches the StarGAN v2 release of CelebA-HQ through `zoo.data.stargan.download` into `<root>/celeba_hq/`, laid out as
`<split>/<class>/<image>` (the `ImageFolder` layout). Then prints the image count per split and class, and the size and
mode of one sample image, so a bad extraction shows up here rather than mid-training.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro
from PIL import Image

from zoo.data.stargan import download, splits
from zoo.paths import DATA_ROOT

NAME = "celeba_hq"


@dataclass
class PrepareConfig:
    root: str = str(DATA_ROOT)  # directory that receives the celeba_hq/ folder
    force: bool = False  # delete and re-download even if the folder already exists


def report(out: Path) -> None:
    """Print image counts per split and class, and the size and mode of one sample."""
    for split in splits(NAME):
        classes = sorted(p for p in (out / split).iterdir() if p.is_dir())
        counts = {c.name: sum(1 for f in c.iterdir() if f.is_file()) for c in classes}
        print(f"{split}: {sum(counts.values())} images {counts}")

    sample = next(p for p in (out / splits(NAME)[0]).rglob("*") if p.is_file())
    with Image.open(sample) as im:
        print(f"sample {sample.relative_to(out)}: {im.size[0]}x{im.size[1]} {im.mode}")


def main(cfg: PrepareConfig) -> None:
    root = Path(cfg.root)
    if cfg.force:
        shutil.rmtree(root / NAME, ignore_errors=True)
    out = download(NAME, root)
    print(f"{NAME}: {out}")
    report(out)


if __name__ == "__main__":
    main(tyro.cli(PrepareConfig))

"""Download the StarGAN v2 image datasets (AFHQv2, CelebA-HQ).

`python -m zoo.data.stargan afhq_v2 celeba_hq`

URLs come from https://github.com/clovaai/stargan-v2/blob/master/download.sh. Each
dataset is extracted to `<root>/<name>/<split>/<class>/<image>`, which is the layout
`torchvision.datasets.ImageFolder` expects.

COMBINED names datasets made of several of these (e.g. "stargan" = AFHQv2 + CelebA-HQ),
for analysis code that treats them as one. They aren't downloaded as such: each member is.
Their splits are roles matched by position, so the combined "test" split is AFHQv2's
"test" plus CelebA-HQ's "val".
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

# name -> (URL, splits); the first split is the training split.
DATASETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "afhq_v2": ("https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1", ("train", "test")),
    "celeba_hq": ("https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=1", ("train", "val")),
}

# name -> (member datasets, split roles); split i of the combined dataset is each member's split i.
COMBINED: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "stargan": (("afhq_v2", "celeba_hq"), ("train", "test")),
}
ALL = sorted([*DATASETS, *COMBINED])  # every name analysis scripts accept


def splits(name: str) -> tuple[str, ...]:
    return COMBINED[name][1] if name in COMBINED else DATASETS[name][1]


def members(name: str) -> tuple[str, ...]:
    """The real datasets behind `name`: its members if combined, else just itself."""
    return COMBINED[name][0] if name in COMBINED else (name,)


def member_split(name: str, member: str, split: str) -> str:
    """The split of `member` that plays the role of `split` in `name`."""
    return splits(member)[splits(name).index(split)]


def download(name: str, root: str | Path = "data") -> Path:
    """Fetch and extract `name` into `root/name` unless it is already there; return that path."""
    url, split_names = DATASETS[name]
    root = Path(root)
    out = root / name
    if all((out / s).is_dir() for s in split_names):
        return out

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{name}.zip"
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".zip.part")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0) or None
            with tqdm(total=total, unit="B", unit_scale=True, desc=f"downloading {name}") as pbar:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    pbar.update(len(chunk))
        tmp.rename(zip_path)

    # Extract into a staging dir, then move the dataset folder into place, so a
    # half-finished extraction is never mistaken for a complete one.
    staging = root / f".{name}.extracting"
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.infolist(), desc=f"extracting {name}", unit="file"):
            zf.extract(member, staging)  # zipfile strips absolute paths and `..`
    # The zip holds either `<split>/...` at the top level or a single `<name>/` folder.
    src = staging / name if (staging / name).is_dir() else staging
    missing = [s for s in split_names if not (src / s).is_dir()]
    if missing:
        raise RuntimeError(f"{zip_path} has no {missing} split(s); found {sorted(p.name for p in src.iterdir())}")
    shutil.rmtree(out, ignore_errors=True)
    src.rename(out)
    shutil.rmtree(staging, ignore_errors=True)
    zip_path.unlink()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+", choices=sorted(DATASETS))
    p.add_argument("--root", default="data")
    args = p.parse_args()
    for name in args.datasets:
        print(f"{name}: {download(name, args.root)}")


if __name__ == "__main__":
    main()

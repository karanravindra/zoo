"""Generate digits from a trained flow checkpoint.

`uv run projects/digitdreamer/sample.py [flow.pt] [--digits 3 7] [--per-digit 10] [--steps 50] [--cfg 2.0]`

Defaults to the latest flow run's last.pt and every digit. Writes a grid (rows = digits) to
`<repo>/runs/digitdreamer/sample.png` unless `--out` is given.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from torchvision.utils import save_image
from tyro.conf import Positional

from dreamer import LATENT_SIZE, load_dreamer

from zoo.metrics import to_unit
from zoo.paths import runs_root
from zoo.runs import latest_checkpoint
from zoo.utils.device import get_device


@dataclass
class Args:
    ckpt: Positional[str | None] = None  # flow checkpoint (default: latest flow run's last.pt)
    digits: list[int] = field(default_factory=lambda: list(range(10)))
    per_digit: int = 10
    steps: int | None = None
    cfg: float | None = None  # classifier-free guidance scale (1 = off)
    seed: int = 0
    out: Path = runs_root("digitdreamer") / "sample.png"


def main():
    args = tyro.cli(Args, description=__doc__)

    device = get_device()
    ckpt = Path(args.ckpt) if args.ckpt else latest_checkpoint("digitdreamer", "last.pt", stage="flow")
    dreamer = load_dreamer(ckpt).to(device)

    y = torch.tensor(args.digits, device=device).repeat_interleave(args.per_digit)
    noise = torch.randn(
        len(y), dreamer.ae_cfg.latent_channels, LATENT_SIZE, LATENT_SIZE, device=device, generator=torch.Generator(device=device).manual_seed(args.seed)
    )
    imgs = dreamer.generate(y, steps=args.steps, cfg_scale=args.cfg, noise=noise)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(to_unit(imgs), args.out, nrow=args.per_digit, padding=1)
    print(f"Wrote {args.out} (from {ckpt})")


if __name__ == "__main__":
    main()

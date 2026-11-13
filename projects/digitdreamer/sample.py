"""Generate digits from a trained flow checkpoint.

`uv run projects/digitdreamer/sample.py [flow.pt] [--digits 3 7] [--per-digit 10] [--steps 50] [--cfg 2.0]`

Defaults to the latest flow run's last.pt and every digit. Writes a grid (rows = digits) to
`<repo>/runs/digitdreamer/sample.png` unless `--out` is given.
"""

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS_ROOT, latest_checkpoint, to_unit  # noqa: E402
from dreamer import LATENT_SIZE, load_dreamer  # noqa: E402
from zoo.utils.device import get_device  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", help="flow checkpoint (default: latest flow run's last.pt)")
    p.add_argument("--digits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--per-digit", type=int, default=10)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--cfg", type=float, default=None, help="classifier-free guidance scale (1 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=RUNS_ROOT / "sample.png")
    args = p.parse_args()

    device = get_device()
    ckpt = Path(args.ckpt) if args.ckpt else latest_checkpoint("flow", "last.pt")
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

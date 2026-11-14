"""Generate sprites from a text prompt with a trained latent diffusion checkpoint.

`uv run projects/pixel_art/sample.py [ldm.pt] [--prompt "gold game pawn" --prompt "eagle nest"] [--per-prompt 8] [--steps 50] [--guidance 3.0]`

Defaults to the latest ldm run's last.pt and the first 8 validation captions. Writes a grid
(one row per prompt) to `<repo>/runs/pixel_art/sample.png` unless `--out` is given.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from tyro.conf import Positional, UseAppendAction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS_ROOT, latest_checkpoint, save_grid  # noqa: E402
from ldm import load_ldm  # noqa: E402
from pixel_data import load_test_captions  # noqa: E402

from zoo.utils.device import get_device  # noqa: E402


@dataclass
class Args:
    ckpt: Positional[str | None] = None  # ldm checkpoint (default: latest ldm run's last.pt)
    prompt: UseAppendAction[list[str]] = field(default_factory=list)  # caption to draw (repeatable; default: first validation captions)
    per_prompt: int = 8
    steps: int | None = None
    guidance: float | None = None  # classifier-free guidance scale (1 = off)
    seed: int = 0
    out: Path = RUNS_ROOT / "sample.png"


def main():
    args = tyro.cli(Args, description=__doc__)

    device = get_device()
    ckpt = Path(args.ckpt) if args.ckpt else latest_checkpoint("ldm", "last.pt")
    ldm = load_ldm(ckpt).to(device)

    prompts = args.prompt or load_test_captions()[:8]
    tokens = ldm.encode_prompts(prompts).repeat_interleave(args.per_prompt, dim=0)
    s = ldm.ae_cfg.latent_size
    noise = torch.randn(
        len(tokens), ldm.ae_cfg.latent_channels, s, s, device=device, generator=torch.Generator(device=device).manual_seed(args.seed)
    )
    imgs = ldm.generate(tokens, steps=args.steps, guidance_scale=args.guidance, noise=noise)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_grid(imgs, args.out, nrow=args.per_prompt)
    print("\n".join(f"row {i}: {t}" for i, t in enumerate(prompts)))
    print(f"Wrote {args.out} (from {ckpt})")


if __name__ == "__main__":
    main()

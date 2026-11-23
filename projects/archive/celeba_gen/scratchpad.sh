#!/usr/bin/env bash
set -euo pipefail

uv run projects/celeba_gen/train_titok.py phase1 \
  --compile \
  --lpips-net alex \
  --no-lr-decay \
  --ema \
  --epochs 20 \
  --batch-size 16 \
  --lr 8e-4 \
  --optimizer muon \
  --model.patch-size 16 \
  --model.width 384 \
  --model.heads 6 \
  --model.mlp-ratio 4 \
  --model.depth 12 \
  --name "batch-size-16_lr-8e-4_depth-4" --dev

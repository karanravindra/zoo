# 1e15: d192 L3 (~2.1M params, ~29 tokens/param, 32k tokens/step) — ran in 35 s
uv run projects/pretrain/train.py \
    --dim 192 --n-heads 3 --n-layers 3 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 1e15 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 16 --grad-accum 1 --lr 1e-3 \
    --ema --min-lr-ratio 1.0

# 3e15: d192 L6 (~3.4M params, ~28 tokens/param, 32k tokens/step) — ran in 1m19s
uv run projects/pretrain/train.py \
    --dim 192 --n-heads 3 --n-layers 6 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 3e15 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 16 --grad-accum 1 --lr 1e-3 \
    --ema --min-lr-ratio 1.0

# 1e16: d320 L5 (~7.5M params, ~22 tokens/param, 65k tokens/step) — ran in 3m21s
uv run projects/pretrain/train.py \
    --dim 320 --n-heads 5 --n-layers 5 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 1e16 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 32 --grad-accum 1 --lr 1e-3 \
    --ema --min-lr-ratio 1.0

# 3e16: d384 L6 (~12.2M params, ~26 tokens/param, 65k tokens/step) — ran in 8m52s
uv run projects/pretrain/train.py \
    --dim 384 --n-heads 6 --n-layers 6 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 3e16 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 32 --grad-accum 1 --lr 1e-3 \
    --ema --min-lr-ratio 1.0

# 1e17: d576 L6 (~26M params, ~20 tokens/param, 131k tokens/step) — ran in 26m22s
uv run projects/pretrain/train.py \
    --dim 576 --n-heads 9 --n-layers 6 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 1e17 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 32 --grad-accum 2 --lr 1e-3 \
    --ema --min-lr-ratio 1.0

# 3e17: d640 L8 (~42M params, ~24 tokens/param, 65k tokens/step) — ran in 1h16m
uv run projects/pretrain/train.py \
    --dim 640 --n-heads 10 --n-layers 8 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 3e17 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 32 --grad-accum 1 --lr 8e-4 \
    --ema --min-lr-ratio 1.0

# 1e18: d768 L10 (~74M params, ~26 tokens/param, 131k tokens/step) — ran in 3h57m
uv run projects/pretrain/train.py \
    --dim 768 --n-heads 12 --n-layers 10 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 1e18 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 16 --grad-accum 4 --lr 6e-4 \
    --ema --min-lr-ratio 1.0

# 3e18: d960 L12 (~137M params, ~24 tokens/param, 131k tokens/step) — ran in 11h31m
uv run projects/pretrain/train.py \
    --dim 960 --n-heads 15 --n-layers 12 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 3e18 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 8 --grad-accum 8 --lr 5e-4 \
    --ema --min-lr-ratio 1.0

# 1e19: d1280 L14 (~281M params, ~19 tokens/param, 131k tokens/step) — not run yet
uv run projects/pretrain/train.py \
    --dim 1280 --n-heads 20 --n-layers 14 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 1e19 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 8 --grad-accum 8 --lr 4e-4 \
    --ema --min-lr-ratio 1.0

# 3e19: d1536 L16 (~459M params, ~22 tokens/param, 262k tokens/step) — not run yet
uv run projects/pretrain/train.py \
    --dim 1536 --n-heads 24 --n-layers 16 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 3e19 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 4 --grad-accum 32 --lr 3e-4 \
    --ema --min-lr-ratio 1.0

# 1e20: d1792 L20 (~780M params, ~25 tokens/param, 262k tokens/step) — not run yet
uv run projects/pretrain/train.py \
    --dim 1792 --n-heads 28 --n-layers 20 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 1e20 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 4 --grad-accum 32 --lr 2.5e-4 \
    --ema --min-lr-ratio 1.0

# 1.5e20: d1920 L22 (~981M params, ~24 tokens/param, 262k tokens/step) — not run yet
uv run projects/pretrain/train.py \
    --dim 1920 --n-heads 30 --n-layers 22 --block-size 2048 \
    --doc-mask --attn-window 512 --attn-global-every 5 \
    --flop-budget 1.5e20 \
    --eval-every 500 --sample-every 20000 \
    --batch-size 4 --grad-accum 32 --lr 2e-4 \
    --ema --min-lr-ratio 1.0

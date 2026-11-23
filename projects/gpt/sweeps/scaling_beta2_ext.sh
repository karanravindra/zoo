#!/bin/bash
# Replicates and one more point for the AdamW beta2 pilot (scaling_beta2.yaml): beta2 0.99 gave -0.007 against the base's 0.95 at 2.5e15, which is 1.4x the seed noise, so
# two more seeds of it (each compared by `fit-optim` with the base cell of the same seed) and beta2 0.999. Separate commands, not a sweep (a W&B grid cannot vary the seed with one setting only).
#   bash projects/gpt/sweeps/scaling_beta2_ext.sh
cd "$(dirname "$0")/../../.." || exit 1
COMMON="--n-layers 4 --lr 2.0e-3 --aspect 64 --max-flops 3.0e15 --lr-width-exp 1.0 --val-every 200 --val-tokens 524288 --ema 0.99999 --sample-tokens 0 --group tinystories-optim"
uv run projects/gpt/cli.py pretrain $COMMON --beta2 0.99 --seed 1 --name beta2-0.99-seed1
uv run projects/gpt/cli.py pretrain $COMMON --beta2 0.99 --seed 2 --name beta2-0.99-seed2
uv run projects/gpt/cli.py pretrain $COMMON --beta2 0.999 --name beta2-0.999

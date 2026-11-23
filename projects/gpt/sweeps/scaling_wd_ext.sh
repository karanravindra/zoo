#!/bin/bash
# Replicates for the weight-decay pilot (scaling_wd.yaml): weight decay 0 gave -0.010 against the base's 0.1 at 2.5e15 FLOPs, twice the seed noise, so two more seeds
# of it (each compared by `fit-optim` with the base cell of the same seed). Separate commands, not a sweep (a W&B grid cannot vary the seed for one setting only).
#   bash projects/gpt/sweeps/scaling_wd_ext.sh
cd "$(dirname "$0")/../../.." || exit 1
COMMON="--n-layers 4 --lr 2.0e-3 --aspect 64 --max-flops 3.0e15 --lr-width-exp 1.0 --val-every 200 --val-tokens 524288 --ema 0.99999 --sample-tokens 0 --group tinystories-optim"
uv run projects/gpt/cli.py pretrain $COMMON --weight-decay 0 --seed 1 --name wd-0-seed1
uv run projects/gpt/cli.py pretrain $COMMON --weight-decay 0 --seed 2 --name wd-0-seed2

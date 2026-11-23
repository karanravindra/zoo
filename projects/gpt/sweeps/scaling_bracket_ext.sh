#!/bin/bash
# Scaling-law study, step 2b, extension: lower learning rates for the ladder bracket (scaling_bracket.yaml). At the full 29,500 steps the lowest LR of the
# bracket (0.7x the rule) beat the rule for 2 and 4 layers, so the best was at its edge. This adds 0.5x for 2 and 4 layers (`--lr` is the peak at width 256:
# 1.0e-3 against the rule's 2.0e-3) and 0.7x for 3 layers, a size the bracket skipped. Same group as the ladder: `fit-scaling` takes, for each size, the
# run with the lowest final loss. Separate commands, not a sweep: a W&B grid would cross the sizes with every LR.
#   bash projects/gpt/sweeps/scaling_bracket_ext.sh
cd "$(dirname "$0")/../../.." || exit 1
COMMON="--aspect 64 --max-flops 1.0e17 --lr-width-exp 1.0 --val-every 500 --ema 0.99999 --sample-tokens 0 --group tinystories-ladder"
uv run projects/gpt/cli.py pretrain $COMMON --n-layers 4 --lr 1.0e-3 --name ladder-bracket-L4-lr1.0e-3
uv run projects/gpt/cli.py pretrain $COMMON --n-layers 2 --lr 1.0e-3 --name ladder-bracket-L2-lr1.0e-3
uv run projects/gpt/cli.py pretrain $COMMON --n-layers 3 --lr 1.4e-3 --name ladder-bracket-L3-lr1.4e-3

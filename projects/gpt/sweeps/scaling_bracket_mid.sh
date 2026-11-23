#!/bin/bash
# Scaling-law study, step 2b, a mid size: the LR bracket of scaling_bracket.yaml found the rule's LR too high at the full 29,500 steps for 2 and 4 layers, so this
# checks 6 layers x 384 (12.2M parameters) at half that length, 10,000 steps, at 0.5x and 0.7x of the rule's LR (`--lr` is the peak at width 256: 1.0e-3 and 1.4e-3
# against 2.0e-3, so 0.67e-3 and 0.93e-3 at this width), and at the full length (29,500 steps) at 0.5x. A constant LR with an EMA makes a run of 10,000 steps the first
# 10,000 steps of the ladder run, so `fit-scaling` compares them with it in the bracket block (same group as the ladder; a run this short is never a size's tuned run).
#   bash projects/gpt/sweeps/scaling_bracket_mid.sh
cd "$(dirname "$0")/../../.." || exit 1
COMMON="--n-layers 6 --aspect 64 --max-steps 10000 --lr-width-exp 1.0 --val-every 500 --ema 0.99999 --sample-tokens 0 --group tinystories-ladder"
uv run projects/gpt/cli.py pretrain $COMMON --lr 1.0e-3 --name bracket-mid-L6-lr1.0e-3
uv run projects/gpt/cli.py pretrain $COMMON --lr 1.4e-3 --name bracket-mid-L6-lr1.4e-3
uv run projects/gpt/cli.py pretrain --n-layers 6 --aspect 64 --max-flops 1.0e17 --lr-width-exp 1.0 --val-every 500 --ema 0.99999 --sample-tokens 0 --group tinystories-ladder --lr 1.0e-3 --name ladder-bracket-L6-lr1.0e-3

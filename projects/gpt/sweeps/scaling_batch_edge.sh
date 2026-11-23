#!/bin/bash
# Scaling-law study, optimizer pilot, extension: learning rates beyond the bracket of scaling_batch.yaml. At 8192 tokens per step the loss rose with the LR
# over the whole bracket (best 1.4e-3, the lowest), and at 32768 the best was at the top (2.8e-3), so this adds 1.0e-3 and 7.0e-4 at 8192 and 4.0e-3 at 32768
# (`--lr` is the peak at width 256, which is this size's width). Same flags as the sweep; separate commands, not a sweep: a W&B grid would cross the settings.
#   bash projects/gpt/sweeps/scaling_batch_edge.sh
cd "$(dirname "$0")/../../.." || exit 1
COMMON="--n-layers 4 --batch-size 32 --aspect 64 --max-flops 3.0e15 --lr-width-exp 1.0 --val-every 200 --val-tokens 524288 --ema 0.99999 --sample-tokens 0 --group tinystories-optim"
uv run projects/gpt/cli.py pretrain $COMMON --global-batch 8192 --lr 1.0e-3 --name batch-8192-lr1.0e-3
uv run projects/gpt/cli.py pretrain $COMMON --global-batch 8192 --lr 7.0e-4 --name batch-8192-lr7.0e-4
uv run projects/gpt/cli.py pretrain $COMMON --global-batch 32768 --lr 4.0e-3 --name batch-32768-lr4.0e-3

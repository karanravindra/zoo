#!/bin/bash
# Scaling-law study, the optimizer and data settings the sweeps leave out: gradient clipping off and tight, and a document mask. One-factor runs on the
# same 4-layer aspect-64 size (4.3M parameters) and budget as scaling_wd.yaml (group tinystories-optim), so `cli.py fit-optim` compares each with the shape
# grid's run of the same size at equal compute. Separate commands, not a sweep: a W&B grid would cross the settings. Each is independent (one that crashes
# does not stop the rest).
#   bash projects/gpt/sweeps/scaling_misc.sh
cd "$(dirname "$0")/../../.." || exit 1
COMMON="--n-layers 4 --lr 2.0e-3 --aspect 64 --max-flops 3.0e15 --lr-width-exp 1.0 --val-every 200 --val-tokens 524288 --ema 0.99999 --sample-tokens 0 --group tinystories-optim"
uv run projects/gpt/cli.py pretrain $COMMON --grad-clip 0 --name misc-clip-off
uv run projects/gpt/cli.py pretrain $COMMON --grad-clip 0.3 --name misc-clip-0.3
uv run projects/gpt/cli.py pretrain $COMMON --doc-mask --name misc-doc-mask

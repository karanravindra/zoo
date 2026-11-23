#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root so the relative script path works from anywhere.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# Sweep LR x global batch size (sequences per step). Everything else is held fixed:
# - The micro-batch stays at MICRO_BATCH and the global batch is changed through --grad-accum, because the
#   eval set is --batch-size x --eval-batches sequences: every run is scored on the same validation data.
# - Warmup and eval cadence are in steps, so they are rescaled from REF_BATCH to cover the same tokens.
# Extra CLI args override the flags below, e.g.: ./scratchpad.sh --flop-budget 1e16 --no-dev
MICRO_BATCH=16
REF_BATCH=32

run() {  # run <global batch> <lr> [extra train.py args]
    local batch=$1 lr=$2
    shift 2
    uv run projects/pretrain/train.py \
        --dim 320 --n-heads 5 --n-layers 5 --block-size 2048 \
        --doc-mask --attn-window 512 --attn-global-every 5 \
        --flop-budget 1e16 \
        --sample-every 20000 \
        --batch-size "$MICRO_BATCH" --grad-accum $((batch / MICRO_BATCH)) --lr "$lr" \
        --warmup-steps $((200 * REF_BATCH / batch)) --eval-every $((500 * REF_BATCH / batch)) \
        --ema --ema-frac 0.1 --min-lr-ratio 1.0 \
        --run-name "sweep-bs${batch}-lr${lr}" \
        "$@"
}

# LRs per global batch size. A cell whose run already has results.json is skipped (a rerun would only write a -2 copy),
# so extending the sweep means editing these lists and running the script again: only the new cells train.
# Keep at least 2 values either side of each batch size's best LR; read the optimum off
#   uv run projects/pretrain/sweep.py 'sweep-*'
declare -A LRS=(
    [16]="2.5e-4 5e-4 1e-3 2e-3 4e-3"   # best was 1e-3, the bottom of the first range
    [32]="1e-3 2e-3 4e-3 8e-3 1.6e-2"   # best was 4e-3, the top
    [64]="1e-3 2e-3 4e-3 8e-3 1.6e-2"   # best was 4e-3, the top
)

# set -e stops the whole sweep at the first failed run, and Ctrl-C ends it instead of moving on to the next run.
for BATCH_SIZE in 16 32 64; do
    for lr in ${LRS[$BATCH_SIZE]}; do
        if [[ -f runs/pretrain/sweep-bs${BATCH_SIZE}-lr${lr}/results.json ]]; then
            echo "skip sweep-bs${BATCH_SIZE}-lr${lr}: already done"
            continue
        fi
        run "$BATCH_SIZE" "$lr" "$@"
    done
done

# Replicates of a cell with other seeds ("<batch>:<lr>:<seed>"): the run-to-run noise, to judge which loss gaps in the grid
# mean anything. Named noise-* so that 'sweep-*' stays one run per cell; look at one cell's seeds with
#   uv run projects/pretrain/sweep.py '*bs32-lr4e-3*' --cols seed
REPEATS=("32:4e-3:1" "32:4e-3:2")
for rep in "${REPEATS[@]}"; do
    IFS=: read -r batch lr seed <<< "$rep"
    name="noise-bs${batch}-lr${lr}-s${seed}"
    if [[ -f runs/pretrain/$name/results.json ]]; then
        echo "skip $name: already done"
        continue
    fi
    run "$batch" "$lr" --seed "$seed" --run-name "$name" "$@"  # the later --run-name overrides the one in run()
done

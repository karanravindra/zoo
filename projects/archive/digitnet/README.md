# DigitNet

A small CNN (two conv blocks, global pooling, linear head) that classifies 28x28 digits. Trains on MNIST
by default; any dataset class in `zoo.data` with `image_shape` and `num_classes` works via `Config.dataset`.

```
uv run projects/digitnet/train.py
```

Each run writes to `<repo>/runs/digitnet/<timestamp>/`:

| File | Contents |
| --- | --- |
| `config.json` | full config plus git commit and device |
| `metrics.csv` | one row per epoch (loss, accuracy, lr, time) |
| `best.pt` | checkpoint with the best validation accuracy |
| `last.pt` | checkpoint from the final epoch |
| `results.json` | test-set score of `best.pt` |
| `curves.png` | loss and accuracy curves |

Files:
- `digitnet.py`: `Config`, the `DigitNet` model, and checkpoint save/load.
- `train.py`: the training script.

Checkpoints are self-describing (weights plus the config that built them), so
`load_checkpoint(path)` returns a ready model with no other setup. The validation set is carved
deterministically from the train split (`val_size`, `seed`) and is used only to pick `best.pt`.
Every `Config` field is a CLI flag (via tyro), e.g. `uv run projects/digitnet/train.py --epochs 20 --lr 1e-3`; see `--help`.

## Dev runs

Pass `train.py` `--dev` to try a change without logging anything: the usual outputs go to a temporary directory that is deleted
when the script exits, so the run never appears under `runs/`.

## Training diagnostics in metrics.csv

Besides the columns above, every row has gradient and GPU columns (blank where a value was not measured):

- `grad_norm`, `grad_norm_max`: mean and largest global gradient norm (before clipping) since the previous row;
  `clip_frac`: fraction of steps clipped, when the script clips.
- `weight_norm`, `update_ratio` (size of one optimizer step relative to the weights), `grad_cos` (cosine between
  consecutive steps' gradients: near 1 steady, near 0 noisy, negative oscillating), averaged over every 100th step.
- `gpu_util` (mean %), `gpu_mem_gb` (peak device memory), `gpu_alloc_gb` (peak memory torch allocated),
  `gpu_reserved_gb`, `gpu_power_w` (mean), `gpu_temp_c` (peak), sampled in the background while training.

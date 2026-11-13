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
Settings are changed by editing `Config` defaults; there are no CLI flags.

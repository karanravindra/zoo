import json
from pathlib import Path

import numpy as np
import torch


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "tinystories"
EOS = "<|endoftext|>"


class TokenData:
    """Sample packed next-token windows without loading the corpus into RAM."""

    def __init__(self, directory, context_length):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        self.context_length = context_length
        self.splits = {}
        for split in ("train", "validation"):
            tokens = np.memmap(directory / f"{split}.bin", dtype=np.uint16, mode="r")
            if len(tokens) != self.metadata["tokens"][split]:
                raise ValueError(f"{split} token count does not match metadata; prepare data again")
            if len(tokens) <= context_length:
                raise ValueError(f"{split} needs at least {context_length + 1} tokens")
            self.splits[split] = tokens

    def batch(self, split, batch_size, generator, device):
        tokens = self.splits[split]
        starts = torch.randint(len(tokens) - self.context_length, (batch_size,),
                               generator=generator).tolist()
        windows = np.stack([tokens[i:i + self.context_length + 1] for i in starts])
        windows = torch.from_numpy(windows.astype(np.int64)).to(device)
        return windows[:, :-1], windows[:, 1:]

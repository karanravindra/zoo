"""Storage and transform adapters for map-style image datasets."""

import bisect
import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TensorChunks(Dataset):
    """Read fixed-shape (image, label) tensors from lazily mapped .pt chunks.

    Each worker owns its bounded mapping cache. Transforms must not mutate the
    returned views in place. Include source identity and preprocessing versions
    in metadata when building a cache.
    """

    FORMAT_VERSION = 1

    def __init__(self, path, max_open_chunks=16):
        if max_open_chunks < 1:
            raise ValueError("max_open_chunks must be positive")
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest["format_version"] != self.FORMAT_VERSION:
            raise ValueError("Unsupported tensor cache format")
        self.max_open_chunks = max_open_chunks
        self._chunks = OrderedDict()
        self._pid = os.getpid()
        self._ends = []
        count = 0
        for chunk in self.manifest["chunks"]:
            if not (self.path / chunk["file"]).is_file():
                raise ValueError(f"Missing tensor cache chunk: {chunk['file']}")
            count += chunk["count"]
            self._ends.append(count)
        if count != self.manifest["count"]:
            raise ValueError("Tensor cache sample count mismatch")

    @classmethod
    def prepare(cls, root, dataset, metadata, chunk_size=8192, max_open_chunks=16):
        """Build once, then reuse by content identity; publish complete caches only.

        The caller supplies deterministic preprocessing through dataset. Only
        one chunk is materialized at a time. Concurrent builders can safely
        publish the same identity without replacing a completed cache.
        """
        if chunk_size < 1 or len(dataset) < 1:
            raise ValueError("chunk_size and dataset length must be positive")
        identity = dict(format_version=cls.FORMAT_VERSION, metadata=metadata,
                        count=len(dataset), chunk_size=chunk_size)
        encoded = json.dumps(identity, sort_keys=True)
        key = hashlib.sha256(encoded.encode()).hexdigest()[:24]
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / key
        if not path.exists():
            temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=root))
            try:
                chunks = []
                for start in range(0, len(dataset), chunk_size):
                    samples = [dataset[i] for i in range(start, min(start + chunk_size, len(dataset)))]
                    images, labels = zip(*samples)
                    filename = f"chunk-{len(chunks):05d}.pt"
                    torch.save({"images": torch.stack(images),
                                "labels": torch.as_tensor(labels, dtype=torch.long)},
                               temporary / filename)
                    chunks.append(dict(file=filename, count=len(samples)))
                manifest = dict(identity, chunks=chunks)
                (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                try:
                    temporary.rename(path)
                except OSError:
                    if not (path / "manifest.json").is_file():
                        raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        result = cls(path, max_open_chunks=max_open_chunks)
        if any(result.manifest[field] != value for field, value in identity.items()):
            raise ValueError("Tensor cache identity mismatch")
        return result

    def __len__(self):
        return self.manifest["count"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_chunks"] = OrderedDict()
        return state

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._pid != os.getpid():
            self._chunks.clear()
            self._pid = os.getpid()
        chunk_index = bisect.bisect_right(self._ends, index)
        if chunk_index not in self._chunks:
            chunk = self.manifest["chunks"][chunk_index]
            self._chunks[chunk_index] = torch.load(
                self.path / chunk["file"], mmap=True, weights_only=True,
                map_location="cpu",
            )
            if len(self._chunks) > self.max_open_chunks:
                self._chunks.popitem(last=False)
        self._chunks.move_to_end(chunk_index)
        offset = index - (self._ends[chunk_index - 1] if chunk_index else 0)
        tensors = self._chunks[chunk_index]
        return tensors["images"][offset], tensors["labels"][offset]

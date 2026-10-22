import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset
from torchvision.transforms.functional import to_pil_image

from zoo.data import TensorChunks, make_loader
from zoo.data.mnist import DataConfig, make_dataset


class FakeMNIST:
    def __init__(self):
        self.data = torch.randint(
            0,
            256,
            (23, 28, 28),
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(12),
        )
        self.targets = torch.arange(23) % 10

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return to_pil_image(self.data[index]), int(self.targets[index])


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_chunks_roundtrip_random_order_eviction_and_pickle(self):
        images = torch.arange(23 * 16).reshape(23, 1, 4, 4).float()
        labels = torch.arange(23)
        source = TensorDataset(images, labels)
        stored = TensorChunks.prepare(self.root, source, {"version": 1}, 5, 2)
        self.assertEqual(len(stored.manifest["chunks"]), 5)
        for i in torch.randperm(23).tolist() + [-1, -23]:
            image, label = stored[i]
            self.assertTrue(torch.equal(image, images[i]))
            self.assertEqual(label, labels[i])
            self.assertLessEqual(len(stored._chunks), 2)
        with self.assertRaises(IndexError):
            stored[23]
        restored = pickle.loads(pickle.dumps(stored))
        self.assertFalse(restored._chunks)
        self.assertTrue(torch.equal(restored[6][0], images[6]))
        with patch.object(
            TensorDataset, "__getitem__", side_effect=AssertionError("rebuilt")
        ):
            reused = TensorChunks.prepare(self.root, source, {"version": 1}, 5)
        self.assertEqual(reused.path, stored.path)
        changed = TensorChunks.prepare(self.root, source, {"version": 2}, 5)
        self.assertNotEqual(changed.path, stored.path)

    def test_failed_build_is_not_published(self):
        source = TensorDataset(torch.zeros(8, 1, 4, 4), torch.arange(8))
        with (
            patch("zoo.data.cache.torch.save", side_effect=RuntimeError("interrupted")),
            self.assertRaises(RuntimeError),
        ):
            TensorChunks.prepare(self.root, source, {}, 3)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_chunk_rejected(self):
        source = TensorDataset(torch.zeros(8, 1, 4, 4), torch.arange(8))
        stored = TensorChunks.prepare(self.root, source, {}, 3)
        (stored.path / "chunk-00000.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            TensorChunks(stored.path)

    def test_cached_pipeline_matches_pil_and_keeps_augmentation_random(self):
        with patch("zoo.data.mnist.datasets.MNIST", return_value=FakeMNIST()):
            for split in ("train", "val", "test"):
                reference = make_dataset(DataConfig(), split)
                cached = make_dataset(
                    DataConfig(
                        backend="tensor-cache", cache_dir=self.root, chunk_size=5
                    ),
                    split,
                )
                for index in range(len(reference)):
                    torch.manual_seed(index)
                    image, label = reference[index]
                    torch.manual_seed(index)
                    actual, target = cached[index]
                    self.assertTrue(torch.equal(image, actual), f"{split}: {index}")
                    self.assertEqual(label, target)
                if split == "train":
                    torch.manual_seed(1)
                    first = cached[0][0]
                    torch.manual_seed(2)
                    self.assertFalse(torch.equal(first, cached[0][0]))

    def test_multiworker_loader(self):
        source = TensorDataset(
            torch.arange(23).reshape(23, 1, 1, 1).float(), torch.arange(23)
        )
        stored = TensorChunks.prepare(self.root, source, {}, 5)
        loader = make_loader(stored, DataConfig(num_workers=2), batch_size=4)
        batches = list(loader)
        self.assertTrue(
            torch.equal(torch.cat([labels for _, labels in batches]), torch.arange(23))
        )
        self.assertTrue(
            torch.equal(
                torch.cat([images.flatten() for images, _ in batches]), torch.arange(23)
            )
        )


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_pil_image

from zoo.data.cifar100 import DataConfig, make_dataset, make_splits, stratified_indices

PROJECT = Path(__file__).resolve().parents[1] / "projects/cifar100/classify"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load_module("cifar_model", "model.py")
reports_module = load_module("cifar_reports", "reports.py")
with patch.dict(sys.modules, model=model_module, reports=reports_module):
    training = load_module("cifar_training", "train.py")


class FakeCIFAR100:
    def __init__(self, count=23):
        self.classes = [f"class_{index}" for index in range(100)]
        self.data = torch.randint(0, 256, (count, 32, 32, 3), dtype=torch.uint8,
                                  generator=torch.Generator().manual_seed(12)).numpy()
        self.targets = (torch.arange(count) % 100).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return to_pil_image(self.data[index]), self.targets[index]


class CIFAR100Tests(unittest.TestCase):
    def test_stratified_split_is_balanced_disjoint_and_seeded(self):
        targets = torch.arange(100).repeat_interleave(500)
        train, val = stratified_indices(targets, 42)
        self.assertEqual((len(train), len(val)), (45000, 5000))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(set(train) | set(val)), 50000)
        self.assertTrue(torch.equal(torch.bincount(targets[val]), torch.full((100,), 50)))
        self.assertTrue(torch.equal(torch.bincount(targets[train]), torch.full((100,), 450)))
        self.assertEqual((train, val), stratified_indices(targets, 42))
        self.assertNotEqual(val, stratified_indices(targets, 43)[1])

    def test_split_uses_independent_train_and_validation_transforms(self):
        source = FakeCIFAR100(5200)
        with patch("zoo.data.cifar100.datasets.CIFAR100", return_value=source):
            train, val, test = make_splits(DataConfig(), 42)
        self.assertIsNot(train.dataset, val.dataset)
        self.assertIsNot(train.dataset.transform, val.dataset.transform)
        first = val[0][0]
        self.assertTrue(torch.equal(first, val[0][0]))
        self.assertEqual(first.shape, (3, 32, 32))
        self.assertEqual(first.dtype, torch.float32)
        self.assertEqual(test.classes, source.classes)

    def test_cache_matches_pil_and_retains_random_augmentation(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("zoo.data.cifar100.datasets.CIFAR100", return_value=FakeCIFAR100()),
        ):
            for split in ("train", "val", "test"):
                reference = make_dataset(DataConfig(), split)
                cached = make_dataset(DataConfig(backend="tensor-cache", cache_dir=Path(temporary),
                                                 chunk_size=5, max_open_chunks=2), split)
                for index in range(len(reference)):
                    torch.manual_seed(index)
                    expected, label = reference[index]
                    torch.manual_seed(index)
                    actual, target = cached[index]
                    self.assertTrue(torch.equal(expected, actual), f"{split}: {index}")
                    self.assertEqual(label, target)
                    self.assertTrue(0 <= target < 100)
                if split == "train":
                    torch.manual_seed(1)
                    first = cached[0][0]
                    torch.manual_seed(2)
                    self.assertFalse(torch.equal(first, cached[0][0]))
            # Reopening a normalized cache must preserve metadata identity.
            make_dataset(DataConfig(backend="tensor-cache", cache_dir=Path(temporary), chunk_size=5), "val")

    def test_model_backward_and_configuration(self):
        model = model_module.ConvNeXt(widths=(8, 16, 32), depths=(1, 1, 1))
        output = model(torch.randn(3, 3, 32, 32))
        self.assertEqual(output.shape, (3, 100))
        torch.nn.functional.cross_entropy(output, torch.tensor([0, 50, 99])).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        with self.assertRaises(ValueError):
            model_module.ConvNeXt(widths=(8, 16), depths=(1,))

    def test_evaluation_weights_partial_batches_and_top5(self):
        logits = torch.tensor([[9., 8., 7., 6., 5., 4.],
                               [9., 8., 7., 6., 5., 4.],
                               [9., 8., 7., 6., 5., 4.]])
        labels = torch.tensor([0, 4, 5])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        metrics, targets, predictions = training.evaluate(torch.nn.Identity(), loader, torch.device("cpu"))
        self.assertAlmostEqual(metrics["loss"], torch.nn.functional.cross_entropy(logits, labels).item(), places=5)
        self.assertAlmostEqual(metrics["top1"], 1 / 3)
        self.assertAlmostEqual(metrics["top5"], 2 / 3)
        self.assertEqual(metrics["samples"], 3)
        self.assertTrue(torch.equal(targets, labels))
        self.assertEqual(predictions.tolist(), [0, 0, 0])

    def test_scheduler_restores_warmup_and_cosine_progress(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        scheduler = training.make_scheduler(optimizer, total_steps=8, warmup_steps=2)
        rates = []
        for _ in range(4):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        new_optimizer = torch.optim.AdamW([parameter], lr=0.1)
        new_scheduler = training.make_scheduler(new_optimizer, 8, 2)
        new_optimizer.load_state_dict(optimizer.state_dict())
        new_scheduler.load_state_dict(scheduler.state_dict())
        for _ in range(4):
            self.assertEqual(optimizer.param_groups[0]["lr"], new_optimizer.param_groups[0]["lr"])
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            new_optimizer.step()
            scheduler.step()
            new_scheduler.step()
        self.assertEqual(rates[:2], [0.05, 0.1])
        self.assertAlmostEqual(rates[-1], 0.)

    def test_epoch_checkpoint_resume_matches_uninterrupted_training_and_evaluation(self):
        from zoo.data import TransformDataset
        from zoo.data.cifar100 import EVALUATE, TRAIN

        source = FakeCIFAR100(16)
        train = TransformDataset(source, TRAIN)
        val = TransformDataset(source, EVALUATE)
        test = TransformDataset(source, EVALUATE)
        test.classes = source.classes
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = ["train.py", "--epochs", "2", "--warmup-epochs", "1", "--batch-size", "4",
                      "--widths", "4", "8", "--depths", "1", "1", "--max-train-batches", "2",
                      "--max-eval-batches", "2", "--log-every", "1"]

            def run(extra):
                with (
                    patch.object(sys, "argv", common + extra),
                    patch.object(training, "make_splits", return_value=(train, val, test)),
                    patch.object(training, "write_reports"),
                ):
                    training.main()

            run(["--output-dir", str(root / "full")])
            run(["--output-dir", str(root / "resume"), "--stop-after-epoch", "1"])
            run(["--output-dir", str(root / "resume"), "--resume", str(root / "resume/checkpoint.pt")])
            full = torch.load(root / "full/checkpoint.pt", weights_only=True)
            resumed = torch.load(root / "resume/checkpoint.pt", weights_only=True)
            for key in full["model"]:
                self.assertTrue(torch.equal(full["model"][key], resumed["model"][key]), key)
            self.assertEqual(full["scheduler"], resumed["scheduler"])
            self.assertEqual((resumed["epoch"], resumed["step"]), (2, 4))
            for first, second in zip(full["history"], resumed["history"]):
                self.assertEqual({key: value for key, value in first.items() if key != "elapsed_seconds"},
                                 {key: value for key, value in second.items() if key != "elapsed_seconds"})
            run(["--output-dir", str(root / "eval"), "--resume", str(root / "resume/best.pt"), "--eval-only"])
            first = json.loads((root / "full/summary.json").read_text())
            second = json.loads((root / "eval/summary.json").read_text())
            self.assertEqual(first["test"], second["test"])
            self.assertTrue((root / "eval/metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()

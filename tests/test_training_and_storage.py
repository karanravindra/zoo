import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from zoo.data.text import Shards, TokenDataset, write_token_corpus
from zoo.training.callbacks import ThroughputMonitor
from zoo.training.config import RunConfig
from zoo.training.schedule import make_scheduler


class ShardTests(unittest.TestCase):
    def test_interrupted_and_successful_rewrites_preserve_reader_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = write_token_corpus([[1, 0], [2, 0]], tmp, "train", 4, shard_tokens=2)
            old = Shards(split)
            dataset = TokenDataset(tmp, "train", block_size=1)
            np.testing.assert_array_equal(dataset.delimiters(0), [1, 3])

            def fail():
                yield [0, 3]
                raise RuntimeError("interrupted")

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                write_token_corpus(fail(), tmp, "train", 4, shard_tokens=2)
            np.testing.assert_array_equal(Shards(split)[:], [1, 0, 2, 0])

            write_token_corpus([[0, 3]], tmp, "train", 4, shard_tokens=2)
            np.testing.assert_array_equal(Shards(split)[:], [0, 3])
            np.testing.assert_array_equal(pickle.loads(pickle.dumps(old))[:], [1, 0, 2, 0])
            np.testing.assert_array_equal(TokenDataset(tmp, "train", 1).delimiters(0), [0])
            self.assertEqual(len(list((Path(split) / ".builds").iterdir())), 2)


class TrainingTests(unittest.TestCase):
    def test_throughput_uses_optimizer_steps_including_after_resume(self):
        monitor = ThroughputMonitor(every=2)
        monitor.tp = Mock()
        trainer = SimpleNamespace(global_step=8)
        module = Mock()
        monitor.on_train_start(trainer, module)
        for step in (8, 8, 9, 9, 10, 10, 10, 11, 12):
            trainer.global_step = step
            monitor.on_train_batch_end(trainer, module, None, None, 0)
        monitor.tp.restart.assert_called_once_with(8)
        self.assertEqual([call.args[0] for call in monitor.tp.tick.call_args_list], [10, 12])

    def test_decay_floor_is_used_on_the_final_optimizer_update(self):
        for shape in ("linear", "cosine"):
            with self.subTest(shape=shape):
                parameter = torch.nn.Parameter(torch.tensor(0.0))
                optimizer = torch.optim.SGD([parameter], lr=1)
                scheduler = make_scheduler(optimizer, RunConfig(lr_schedule=shape, warmup=2, min_lr_ratio=0.1), 10)["scheduler"]
                used = []
                for _ in range(10):
                    used.append(optimizer.param_groups[0]["lr"])
                    optimizer.step()
                    scheduler.step()
                self.assertEqual(used[:3], [0.5, 1.0, 1.0])
                self.assertGreater(used[-2], 0.1)
                self.assertAlmostEqual(used[-1], 0.1)

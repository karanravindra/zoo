import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from cli import prepare
from config import PROMPT, PrepareConfig, TrainConfig
from data import DistillData
from module import LitDrafter
from pages import distill_shards


class FastDoclingTests(unittest.TestCase):
    def test_small_rebuild_and_failed_rebuild_expose_only_committed_shards(self):
        def generate(_cmd, **_kwargs):
            cfg = json.loads(_cmd[-1])
            stage = Path(cfg["out"])
            shards = [f"shard_{i:04d}.parquet" for i in range(cfg["n_pages"])]
            for name in shards:
                (stage / name).touch()
            (stage / "manifest.json").write_text(json.dumps({
                "settings": {"target": cfg["target"], "prompt": PROMPT}, "shards": shards,
            }))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = PrepareConfig(out=root, n_pages=2)
            (root / "shard_0000.parquet").touch()
            self.assertEqual(distill_shards(str(root)), [str(root / "shard_0000.parquet")])
            with patch("cli.subprocess.run", side_effect=generate):
                prepare(cfg)
                old = distill_shards(str(root), target=cfg.target)
                self.assertEqual(len(old), 2)
                cfg.n_pages = 1
                prepare(cfg)
            new = distill_shards(str(root))
            self.assertEqual(len(new), 1)
            self.assertNotEqual(new[0], old[0])
            self.assertTrue(all(Path(path).exists() for path in old))
            with patch("cli.subprocess.run", side_effect=subprocess.CalledProcessError(1, ["uv"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    prepare(cfg)
            self.assertEqual(distill_shards(str(root)), new)
            with self.assertRaisesRegex(ValueError, "different target"):
                distill_shards(str(root), target="other")

    def test_collate_discards_unusable_rows_and_empty_batches_skip_metrics(self):
        datamodule = DistillData(TrainConfig(max_len=5))
        datamodule.prompt = "prompt"
        def process(**kwargs):
            count = len(kwargs["images"])
            return {
                "input_ids": torch.tensor([[0, 5, 6], [1, 2, 3]])[:count],
                "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]])[:count],
                "pixel_values": torch.tensor([[10], [20]])[:count],
                "pixel_attention_mask": torch.tensor([[1], [0]])[:count],
            }

        datamodule.processor = Mock(side_effect=process, tokenizer=SimpleNamespace(pad_token_id=0))
        png = io.BytesIO()
        Image.new("RGB", (1, 1)).save(png, format="PNG")
        batch = datamodule.collate([(png.getvalue(), [1]), (png.getvalue(), [7, 8, 9])])
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3, 7, 8]])
        self.assertEqual(batch["pixel_values"].tolist(), [[20]])
        self.assertIsNone(datamodule.collate([(png.getvalue(), [1])]))
        lit = SimpleNamespace(trainer=SimpleNamespace(world_size=1))
        self.assertIsNone(LitDrafter._step(lit, None, "train", True))
        self.assertIsNone(LitDrafter._step(lit, None, "val", False))
        lit.trainer.world_size = 2
        lit.drafter = torch.nn.Linear(2, 2)
        loss = LitDrafter._step(lit, None, "train", True)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        for param in lit.drafter.parameters():
            torch.testing.assert_close(param.grad, torch.zeros_like(param))

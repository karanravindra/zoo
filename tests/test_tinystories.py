import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1] / "projects/tinystories/gpt"
spec = importlib.util.spec_from_file_location("tinystories_model", PROJECT / "model.py")
model_module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, tinystories_model=model_module):
    spec.loader.exec_module(model_module)
data_spec = importlib.util.spec_from_file_location(
    "tinystories_data", PROJECT / "data.py"
)
data_module = importlib.util.module_from_spec(data_spec)
data_spec.loader.exec_module(data_module)
train_spec = importlib.util.spec_from_file_location(
    "tinystories_train", PROJECT / "train.py"
)
training = importlib.util.module_from_spec(train_spec)
with patch.dict(sys.modules, data=data_module, model=model_module):
    train_spec.loader.exec_module(training)


class TinyStoriesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_parameter_budget(self):
        model = model_module.GPT()
        count = sum(p.numel() for p in model.parameters())
        self.assertTrue(6_500_000 <= count <= 6_600_000, count)

    def test_grouped_query_attention_configuration(self):
        config = model_module.GPTConfig(
            vocab_size=16,
            context_length=8,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            d_model=32,
        )
        block = model_module.Block(config)
        self.assertEqual(block.qkv.out_features, 64)
        logits, _ = model_module.GPT(config)(torch.ones((2, 8), dtype=torch.long))
        self.assertEqual(logits.shape, (2, 8, 16))
        with self.assertRaisesRegex(ValueError, "n_heads"):
            model_module.GPTConfig(n_heads=4, n_kv_heads=3)
        with self.assertRaisesRegex(ValueError, "head dimension"):
            model_module.GPTConfig(n_heads=3, d_model=33)

    def test_muon_routes_hidden_weights_and_adamw_routes_embeddings_and_norms(self):
        model = model_module.GPT(
            model_module.GPTConfig(
                vocab_size=16,
                context_length=8,
                n_layers=1,
                n_heads=2,
                d_model=32,
            )
        )
        optimizers = training.make_optimizers(model, lr=3e-4, weight_decay=0.1)
        self.assertIsInstance(optimizers["muon"], torch.optim.Muon)
        self.assertIsInstance(optimizers["adamw"], torch.optim.AdamW)
        routed = {
            name: [p for group in optimizer.param_groups for p in group["params"]]
            for name, optimizer in optimizers.items()
        }
        muon_ids = {id(p) for p in routed["muon"]}
        adamw_ids = {id(p) for p in routed["adamw"]}
        self.assertEqual(
            muon_ids,
            {
                id(module.weight)
                for module in model.modules()
                if isinstance(module, torch.nn.Linear)
            },
        )
        self.assertFalse(muon_ids & adamw_ids)
        self.assertEqual(muon_ids | adamw_ids, {id(p) for p in model.parameters()})
        self.assertEqual(sum(map(len, routed.values())), len(list(model.parameters())))
        self.assertIn(id(model.token_embedding.weight), adamw_ids)
        for group in optimizers["adamw"].param_groups:
            for parameter in group["params"]:
                self.assertEqual(
                    group["weight_decay"], 0.0 if parameter.ndim < 2 else 0.1
                )

    def test_initialize_gradients_allocates_zeroed_stable_buffers(self):
        model = model_module.GPT(
            model_module.GPTConfig(
                vocab_size=16,
                context_length=8,
                n_layers=1,
                n_heads=2,
                d_model=32,
            )
        )
        training.initialize_gradients(model)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
        gradient_ids = [id(parameter.grad) for parameter in model.parameters()]
        for optimizer in training.make_optimizers(model, 3e-4, 0.1).values():
            optimizer.zero_grad(set_to_none=False)
        self.assertEqual(
            gradient_ids, [id(parameter.grad) for parameter in model.parameters()]
        )

    def test_attention_is_causal_and_model_learns(self):
        torch.manual_seed(7)
        config = model_module.GPTConfig(
            vocab_size=16, context_length=8, n_layers=2, n_heads=2, d_model=32
        )
        model = model_module.GPT(config)
        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        changed = x.clone()
        changed[:, 4:] = 0
        model.eval()
        with torch.no_grad():
            original, _ = model(x)
            modified, _ = model(changed)
        torch.testing.assert_close(
            original[:, :4], modified[:, :4], atol=1e-6, rtol=1e-5
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        targets = (x + 1) % config.vocab_size
        _, initial = model(x, targets)
        for _ in range(15):
            optimizer.zero_grad()
            _, loss = model(x, targets)
            loss.backward()
            optimizer.step()
        _, final = model(x, targets)
        self.assertLess(final.item(), initial.item() / 2)

    @unittest.skipUnless(
        importlib.util.find_spec("tokenizers"), "Install the tinystories extra"
    )
    def test_prepare_train_resume_and_generate(self):
        def run(script, *args):
            result = subprocess.run(
                [sys.executable, str(PROJECT / script), *map(str, args)],
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.txt"
            val = root / "val.txt"
            train.write_text(
                "A little cat found a red ball and played with her friend.\n" * 32
            )
            val.write_text(
                "A little dog found a blue ball and played in the park.\n" * 8
            )
            data, output = root / "data", root / "output"
            run(
                "prepare.py",
                "--data-dir",
                data,
                "--train-file",
                train,
                "--val-file",
                val,
                "--vocab-size",
                300,
            )
            import numpy as np
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
            eos = tokenizer.token_to_id("<|endoftext|>")
            tokens = np.fromfile(data / "train.bin", dtype=np.uint16)
            self.assertEqual(int((tokens == eos).sum()), 32)
            self.assertEqual(
                tokenizer.decode(tokens[: len(tokens) // 32].tolist()),
                train.read_text().splitlines()[0],
            )
            common = [
                "--data-dir",
                data,
                "--output-dir",
                output,
                "--device",
                "cpu",
                "--context-length",
                16,
                "--batch-size",
                2,
                "--grad-accum",
                1,
                "--eval-batches",
                1,
                "--eval-every",
                1,
                "--warmup-steps",
                0,
            ]
            run("train.py", *common, "--steps", 2)
            before = torch.load(output / "checkpoint.pt", weights_only=True)
            run("train.py", *common, "--steps", 3, "--resume", output / "checkpoint.pt")
            after = torch.load(output / "checkpoint.pt", weights_only=True)
            self.assertEqual(after["step"], 3)
            self.assertEqual(after["tokens_seen"], 96)
            self.assertEqual(set(after["optimizers"]), {"muon", "adamw"})
            self.assertTrue(after["optimizers"]["muon"]["state"])
            for state in after["optimizers"]["adamw"]["state"].values():
                self.assertEqual(state["step"].item(), 3)
            self.assertFalse(
                torch.equal(
                    before["model"]["token_embedding.weight"],
                    after["model"]["token_embedding.weight"],
                )
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(0 < summary["val_loss"] < 20)
            self.assertTrue((output / "curves.png").is_file())
            text = run(
                "generate.py",
                "--checkpoint",
                output / "checkpoint.pt",
                "--prompt",
                "A little cat",
                "--max-new-tokens",
                3,
                "--device",
                "cpu",
            )
            self.assertTrue(text.startswith("A little cat"), text)


if __name__ == "__main__":
    unittest.main()

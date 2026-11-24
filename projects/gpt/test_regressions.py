import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import data
from config import Config, EvalConfig, PrepareConfig
from evals import Scorer, eval_identity, run_eval, save_task
from model import Llama
from prepare import prepare_corpus
from scaling import Run, tuned_runs


class GPTTests(unittest.TestCase):
    def test_rolling_scores_match_a_reference_for_each_target(self):
        sc = Scorer.__new__(Scorer)
        sc.bs, sc.eot, sc.batch_tokens, sc.device = 4, 0, 9, torch.device("cpu")
        sc.hidden = lambda ids: ids.cumsum(1).unsqueeze(-1)

        def token_result(prefix, targets):
            return (prefix[:, 0] + targets).float(), prefix[:, 0].remainder(2) == targets.remainder(2)

        sc.nll_hits = token_result
        contexts = [[5], [4, 3, 2, 1, 0], [0], []]
        continuations = [[1, 1, 1, 2], [4, 5, 6, 7, 8, 9], [], [3]]
        expected_lp, expected_ok = [], []
        for context, continuation in zip(contexts, continuations):
            context = context or [sc.eot]
            seq = context + continuation
            sums = [sum(seq[max(0, i - sc.bs) : i]) for i in range(len(context), len(seq))]
            expected_lp.append(-sum(s + t for s, t in zip(sums, continuation)))
            expected_ok.append(all(s % 2 == t % 2 for s, t in zip(sums, continuation)))
        lp, ok = sc.score(contexts, continuations, "fixture")
        np.testing.assert_array_equal(lp, expected_lp)
        np.testing.assert_array_equal(ok, expected_ok)

    def test_eval_keeps_completed_tasks_and_separates_models_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "first.ckpt"
            path.write_bytes(b"first checkpoint")
            cfg = EvalConfig(ckpts=(path,), tasks=("val", "piqa"), wandb=False)
            key, _ = eval_identity(path, cfg)
            model = SimpleNamespace(cfg=Config(), tokens_seen=100)
            model.to = lambda _: model
            with patch("evals.get_device", return_value=torch.device("cpu")), patch("evals.load_module", return_value=model), \
                    patch("evals.Scorer"), patch("evals.run_task", side_effect=[{"acc": 0.7}, RuntimeError("failed task")]):
                with self.assertRaisesRegex(RuntimeError, "failed task"):
                    run_eval(cfg)
            latest = root / "eval.json"
            self.assertEqual(json.loads(latest.read_text())["tasks"], {"val": {"acc": 0.7}})

            cfg.limit = 5
            limited, info = eval_identity(path, cfg)
            save_task(root, limited, info, "val", {"n": 5}, latest=False)
            self.assertEqual(json.loads(latest.read_text())["tasks"], {"val": {"acc": 0.7}})
            self.assertNotEqual(key, limited)

            path.write_bytes(b"updated checkpoint")
            cfg.limit = 0
            other, info = eval_identity(path, cfg)
            self.assertNotEqual(key, other)
            save_task(root, other, info, "piqa", {"acc": 0.4}, latest=True)
            self.assertEqual(json.loads(latest.read_text())["tasks"], {"piqa": {"acc": 0.4}})
            self.assertTrue((root / "eval" / f"{key}.json").is_file())

    def test_failed_corpus_build_leaves_no_current_pointer_and_can_be_retried(self):
        def build(_train, _val, out, **_settings):
            (out / "tokenizer.json").write_text("{}")
            for split in ("train", "validation"):
                directory = out / split
                directory.mkdir()
                (directory / "tokens.npy").touch()
                (directory / "meta.json").write_text('{"length": 8}')

        with tempfile.TemporaryDirectory() as tmp, patch.object(data, "DATA_ROOT", Path(tmp)):
            cfg = PrepareConfig()
            root = data.corpus_root(cfg.dataset)
            root.mkdir(parents=True)
            (root / "tokenizer.json").write_text("{}")  # an incomplete legacy build
            with patch("prepare.build_corpus", side_effect=RuntimeError("failed build")):
                with self.assertRaisesRegex(RuntimeError, "failed build"):
                    prepare_corpus(cfg)
            self.assertEqual(data.corpus_dir(cfg.dataset), root)
            with patch("prepare.build_corpus", side_effect=build):
                prepare_corpus(cfg)
            snapshot = data.corpus_dir(cfg.dataset)
            self.assertNotEqual(snapshot, root)
            cfg.force = True
            with patch("prepare.build_corpus", side_effect=RuntimeError("failed rebuild")):
                with self.assertRaisesRegex(RuntimeError, "failed rebuild"):
                    prepare_corpus(cfg)
            self.assertEqual(data.corpus_dir(cfg.dataset), snapshot)
            pinned = Config(corpus_build=str(snapshot.relative_to(root)))
            self.assertEqual(data.checkpoint_corpus(pinned), snapshot)

    def test_cache_and_full_forward_agree_across_document_boundaries(self):
        torch.manual_seed(17)
        idx = torch.tensor([[1, 2, 0, 3, 4, 0, 5], [2, 0, 3, 4, 0, 5, 6]])
        for masked in (True, False):
            with self.subTest(masked=masked), torch.inference_mode():
                model = Llama(Config(vocab_size=8, dim=16, head_dim=8, n_layers=2, block_size=8, eot_id=0, doc_mask=masked)).eval()
                cache = [{} for _ in model.blocks]
                prime = model(idx[:, :3], cache)
                torch.testing.assert_close(prime, model(idx[:, :3]))
                for stop in range(4, idx.shape[1] + 1):
                    step = model(idx[:, stop - 1 : stop], cache)
                    torch.testing.assert_close(step[:, 0], model(idx[:, :stop])[:, -1])

    def test_tuning_for_token_holdout_ignores_future_losses(self):
        steps = np.array([1, 2, 3, 4])
        base = Run("early", Config(lr=0.001), steps, steps * 100, steps * 200, np.array([1.6, 1.4, 1.2, 1.1]))
        other = Run("late", Config(lr=0.002), steps, steps * 100, steps * 200, np.array([1.7, 1.5, 1.0, 0.9]))
        self.assertEqual(tuned_runs([base, other])[0].id, "late")
        self.assertEqual(tuned_runs([base, other], cutoff=200)[0].id, "early")
        other.loss[-2:] = [100, 100]
        self.assertEqual(tuned_runs([base, other], cutoff=200)[0].id, "early")

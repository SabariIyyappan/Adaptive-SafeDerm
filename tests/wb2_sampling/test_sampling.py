"""Unit tests for Member A's WB2 sampling code (implementation.md §4 test list).

Run against a random-init V1 architecture and the synthetic HAM10000-shaped
fixture, so they need no real checkpoint, images or GPU. Numbers produced
here are fixture output, never model performance.

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from src.data.split import build_split, write_split_files
from src.model.model import build_model
from src.wb2.sampling.bank import CLASS_ORDER, generate_bank, validate_bank
from src.wb2.sampling.calibrate import fit_temperature, mc_probs, nll
from src.wb2.sampling.fixed_budget import fixed_budget_table
from src.wb2.sampling.mc import (
    deterministic_logits,
    enable_head_dropout,
    load_v1,
    mc_logits,
    stochastic_modules,
)
from tests.fixtures import TempFixture


def _bn_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items() if "running_" in k or "num_batches" in k}


class MCModeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_model(pretrained=False).eval()
        self.x = torch.randn(3, 3, 224, 224)

    def test_only_head_dropout_is_stochastic(self):
        enable_head_dropout(self.model)
        self.assertEqual(stochastic_modules(self.model), ["classifier.2"])

    def test_model_train_would_enable_stochastic_depth(self):
        # Guards the reason we never call model.train(): it turns on far more than the head dropout.
        self.model.train()
        self.assertGreater(len(stochastic_modules(self.model)), 1)

    def test_batchnorm_stats_unchanged_by_mc(self):
        before = _bn_buffers(self.model)
        mc_logits(self.model, self.x, 5, feature_reuse=False)
        mc_logits(self.model, self.x, 5, feature_reuse=True)
        for k, v in _bn_buffers(self.model).items():
            self.assertTrue(torch.equal(v, before[k]), k)

    def test_mc_passes_differ(self):
        out = mc_logits(self.model, self.x, 2)
        self.assertFalse(torch.allclose(out[:, 0], out[:, 1]))

    def test_deterministic_is_reproducible(self):
        mc_logits(self.model, self.x, 3)  # MC must leave the model back in eval mode
        a = deterministic_logits(self.model, self.x)
        b = deterministic_logits(self.model, self.x)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(stochastic_modules(self.model), [])

    def test_shape_and_finite(self):
        out = mc_logits(self.model, self.x, 4)
        self.assertEqual(tuple(out.shape), (3, 4, 7))
        self.assertTrue(torch.isfinite(out).all())

    def test_feature_reuse_matches_full_network(self):
        torch.manual_seed(7)
        reuse = mc_logits(self.model, self.x, 4, feature_reuse=True)
        torch.manual_seed(7)
        naive = mc_logits(self.model, self.x, 4, feature_reuse=False)
        self.assertTrue(torch.allclose(reuse, naive, atol=1e-5))

    def test_same_seed_same_draws(self):
        torch.manual_seed(42)
        a = mc_logits(self.model, self.x, 6)
        torch.manual_seed(42)
        b = mc_logits(self.model, self.x, 6)
        self.assertTrue(torch.equal(a, b))


class CheckpointLoadTests(unittest.TestCase):
    def test_strict_roundtrip_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            model = build_model(pretrained=False)
            torch.save({"epoch": 3, "model_state_dict": model.state_dict(), "config": {"seed": 42}}, path)
            loaded, info = load_v1(path)
            self.assertEqual(info["epoch"], 3)
            self.assertEqual(len(info["checkpoint_sha256"]), 64)
            self.assertFalse(loaded.training)


class BankTests(unittest.TestCase):
    """End-to-end: synthetic split + random checkpoint -> delivered bank files."""

    @classmethod
    def setUpClass(cls):
        cls._fixture = TempFixture()
        root = cls._fixture.__enter__()
        split_dir = root / "splits"
        write_split_files(build_split(pd.read_csv(root / "HAM10000_metadata.csv")), split_dir)
        torch.manual_seed(0)
        ckpt = root / "best.pt"
        torch.save({"epoch": 1, "model_state_dict": build_model(pretrained=False).state_dict()}, ckpt)
        cls.split_dir = split_dir
        cls.cfg = {
            "run_id": "fixture-run", "checkpoint": str(ckpt), "split_dir": str(split_dir),
            "images_dir": str(root / "images"), "out_dir": str(root / "out"), "shard_dir": str(root / "shards"),
            "seed": 42, "n_passes": 12, "feature_reuse": True, "batch_size": 8, "shard_size": 10,
            "num_workers": 0, "device": "cpu",
        }
        cls.manifest = generate_bank(cls.cfg, "val")
        cls.out = Path(cls.cfg["out_dir"])

    @classmethod
    def tearDownClass(cls):
        cls._fixture.__exit__(None, None, None)

    def test_manifest_fields(self):
        m = self.manifest
        self.assertEqual(m["status"], "COMPLETE")
        self.assertEqual(m["stochastic_modules"], ["classifier.2"])
        self.assertEqual(m["class_order"], CLASS_ORDER)
        self.assertIs(m["synthetic"], False)
        for key in ("checkpoint_sha256", "seed", "transform", "split_csv_sha256", "n_images", "n_passes",
                    "feature_reuse", "started_at", "finished_at", "hardware", "torch_version"):
            self.assertIn(key, m)

    def test_validates_against_split_file(self):
        validate_bank(self.out, "val", self.split_dir / "val.csv")

    def test_ids_match_split_row_for_row(self):
        ids = pd.read_csv(self.out / "val_ids.csv")
        split = pd.read_csv(self.split_dir / "val.csv")
        self.assertEqual(ids["image_id"].tolist(), split["image_id"].tolist())
        self.assertEqual(ids["y_true"].tolist(), split["label"].tolist())

    def test_array_layout(self):
        mc = np.load(self.out / "val_mc_logits.npz")["logits"]
        det = np.load(self.out / "val_det_logits.npz")["logits"]
        n = len(pd.read_csv(self.out / "val_ids.csv"))
        self.assertEqual(mc.shape, (n, 12, 7))
        self.assertEqual(det.shape, (n, 7))
        self.assertEqual(mc.dtype, np.float32)
        self.assertTrue(np.isfinite(mc).all())
        probs = torch.softmax(torch.from_numpy(mc), dim=-1).numpy()
        np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-5)

    def test_det_differs_from_single_stochastic_draw(self):
        mc = np.load(self.out / "val_mc_logits.npz")["logits"]
        det = np.load(self.out / "val_det_logits.npz")["logits"]
        self.assertFalse(np.allclose(det, mc[:, 0]))

    def test_resume_reproduces_bank(self):
        mc_first = np.load(self.out / "val_mc_logits.npz")["logits"]
        shards = sorted((Path(self.cfg["shard_dir"]) / "val").glob("shard_*.npz"))
        shards[-1].unlink()  # simulate an interrupted run
        generate_bank(self.cfg, "val")
        np.testing.assert_array_equal(np.load(self.out / "val_mc_logits.npz")["logits"], mc_first)

    def test_corrupted_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for f in self.out.glob("val_*"):
                (tmp / f.name).write_bytes(f.read_bytes())
            ids = pd.read_csv(tmp / "val_ids.csv")
            ids.loc[0, "y_true"] = (ids.loc[0, "y_true"] + 1) % 7
            ids.to_csv(tmp / "val_ids.csv", index=False)
            with self.assertRaises(ValueError):
                validate_bank(tmp, "val", self.split_dir / "val.csv")

    def test_pilot_subset_is_flagged(self):
        cfg = {**self.cfg, "out_dir": self.cfg["out_dir"] + "_pilot", "shard_dir": self.cfg["shard_dir"] + "_pilot"}
        ids = pd.read_csv(self.split_dir / "val.csv")["image_id"].tolist()[:5]
        m = generate_bank(cfg, "val", image_ids=ids)
        self.assertTrue(m["pilot_subset"])
        self.assertEqual(m["n_images"], 5)
        validate_bank(Path(cfg["out_dir"]), "val", self.split_dir / "val.csv")

    def test_fixed_budgets_are_prefixes(self):
        table = fixed_budget_table(self.out, "val").set_index("method")
        self.assertEqual(table.index.tolist(), ["deterministic_T1", "mc_T10"])  # 12-pass bank: T=20/50 skipped
        mc = np.load(self.out / "val_mc_logits.npz")["logits"]
        y = pd.read_csv(self.out / "val_ids.csv")["y_true"].to_numpy()
        self.assertAlmostEqual(table.loc["mc_T10", "nll"], nll(y, mc_probs(mc[:, :10])))
        self.assertNotAlmostEqual(table.loc["mc_T10", "nll"], nll(y, mc_probs(mc)))


class CalibrationTests(unittest.TestCase):
    def test_recovers_known_temperature(self):
        rng = np.random.default_rng(0)
        n, t = 2000, 5
        true_logits = rng.normal(size=(n, 1, 7)) * 2.0
        y = np.array([rng.choice(7, p=p) for p in mc_probs(true_logits)])
        overconfident = np.repeat(true_logits * 3.0, t, axis=1)  # true T = 3
        temp = fit_temperature(overconfident, y)
        self.assertAlmostEqual(temp, 3.0, delta=0.3)
        self.assertLess(nll(y, mc_probs(overconfident, temp)), nll(y, mc_probs(overconfident)))

    def test_temperature_positive_and_probs_normalised(self):
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(50, 4, 7))
        y = rng.integers(0, 7, size=50)
        temp = fit_temperature(logits, y)
        self.assertGreater(temp, 0)
        np.testing.assert_allclose(mc_probs(logits, temp).sum(axis=1), 1.0, atol=1e-9)

    def test_calibrate_refuses_non_val_bank(self):
        from src.wb2.sampling import calibrate
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "val_manifest.json").write_text(json.dumps({"split": "test", "status": "COMPLETE"}))
            argv = sys.argv
            sys.argv = ["calibrate", "--bank-dir", tmp]
            try:
                with self.assertRaises(SystemExit):
                    calibrate.main()
            finally:
                sys.argv = argv


if __name__ == "__main__":
    unittest.main()

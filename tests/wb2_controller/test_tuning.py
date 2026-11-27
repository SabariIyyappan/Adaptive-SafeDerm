"""Tests for validation-only threshold tuning (synthetic banks only)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.wb2.controller.bank import LogitBank
from src.wb2.controller.tune import entropy_threshold_for_budget, tune
from tests.wb2_controller.synthetic import make_random_bank, synthetic_manifest

SMALL_RISK_GRID = {"tau_s": [0.3, 0.5], "h_lo": [0.25], "h_hi": [0.5, 0.7], "margin": [0.0, 0.05], "delta_h": [0.02, 0.05]}
SMALL_VAR_GRID = {"eps_p": [0.01, 0.05], "eps_var": [0.001, 0.005]}


def _bank(split="val", seed=4):
    mc, det, ids = make_random_bank(300, seed=seed)
    return LogitBank(split, mc, ids, synthetic_manifest(split, len(ids)), det)


class TuningTests(unittest.TestCase):
    def test_refuses_non_validation_bank(self):
        with self.assertRaisesRegex(ValueError, "validation-only"):
            tune(_bank("test"), risk_grid=SMALL_RISK_GRID, variance_grid=SMALL_VAR_GRID)

    def test_config_is_locked_and_complete(self):
        cfg, grid = tune(_bank(), risk_grid=SMALL_RISK_GRID, variance_grid=SMALL_VAR_GRID)
        self.assertTrue(cfg["locked"])
        self.assertEqual(cfg["tuned_on_split"], "val")
        self.assertTrue(cfg["val_bank"]["synthetic"])
        for key in ("tau_s", "h_lo", "h_hi", "margin", "delta_h", "min_passes", "min_passes_high"):
            self.assertIn(key, cfg["risk_aware_adaptive"])
        self.assertIn("referral_entropy", cfg["variance_adaptive"])
        self.assertIn("referral_entropy", cfg["fixed_T50_referral"])
        self.assertNotIn("test", str(cfg["val_bank"]))
        self.assertEqual(len(grid), cfg["val_selection"]["grid_size"])

    def test_deterministic(self):
        a, _ = tune(_bank(), risk_grid=SMALL_RISK_GRID, variance_grid=SMALL_VAR_GRID)
        b, _ = tune(_bank(), risk_grid=SMALL_RISK_GRID, variance_grid=SMALL_VAR_GRID)
        for key in ("fixed_T50_referral", "variance_adaptive", "risk_aware_adaptive", "val_selection"):
            self.assertEqual(a[key], b[key], key)

    def test_entropy_threshold_matches_budget(self):
        h = np.random.default_rng(0).uniform(size=1000)
        theta = entropy_threshold_for_budget(h, 0.2)
        self.assertEqual(int((h >= theta).sum()), 200)
        self.assertEqual(entropy_threshold_for_budget(h, 0.0), float("inf"))


if __name__ == "__main__":
    unittest.main()

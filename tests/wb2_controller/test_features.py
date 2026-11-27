"""Tests for WB2 per-checkpoint statistics (synthetic logits only).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.wb2.controller.features import CHECKPOINTS, prefix_stats, softmax
from tests.wb2_controller.synthetic import make_random_bank, scenario_bank


class SoftmaxTemperatureTests(unittest.TestCase):
    def setUp(self):
        self.logits = np.random.default_rng(1).normal(size=(20, 7)) * 3

    def test_rows_sum_to_one(self):
        np.testing.assert_allclose(softmax(self.logits).sum(axis=1), 1.0, atol=1e-12)

    def test_tau_one_is_plain_softmax(self):
        e = np.exp(self.logits)
        np.testing.assert_allclose(softmax(self.logits, 1.0), e / e.sum(axis=1, keepdims=True), rtol=1e-12)

    def test_argmax_invariant_to_temperature(self):
        for tau in (0.5, 1.7, 4.0):
            np.testing.assert_array_equal(softmax(self.logits, tau).argmax(1), self.logits.argmax(1))

    def test_higher_temperature_flattens(self):
        mc, _, _ = make_random_bank(50, seed=3)
        cold, hot = prefix_stats(mc, 1.0), prefix_stats(mc, 2.0)
        self.assertTrue((hot.entropy >= cold.entropy - 1e-12).all())

    def test_rejects_nonpositive_temperature(self):
        with self.assertRaises(ValueError):
            softmax(self.logits, 0.0)


class PrefixStatsTests(unittest.TestCase):
    def setUp(self):
        self.mc, _, _ = make_random_bank(60, seed=2)
        self.stats = prefix_stats(self.mc)

    def test_shapes(self):
        self.assertEqual(self.stats.mean_probs.shape, (60, len(CHECKPOINTS), 7))
        self.assertEqual(self.stats.entropy.shape, (60, len(CHECKPOINTS)))

    def test_mean_is_mean_of_first_k_passes(self):
        for j, k in enumerate(CHECKPOINTS):
            expected = softmax(self.mc[:, :k, :].astype(np.float64)).mean(axis=1)
            np.testing.assert_allclose(self.stats.mean_probs[:, j], expected, rtol=1e-6, atol=1e-9)

    def test_causal_no_lookahead(self):
        """Changing passes after k must not change any statistic at checkpoints <= k."""
        k = 20
        j_last = CHECKPOINTS.index(k)
        tampered = self.mc.copy()
        tampered[:, k:, :] = np.random.default_rng(9).normal(size=tampered[:, k:, :].shape) * 10
        tampered[0, k + 2, 0] = np.nan
        t = prefix_stats(tampered)
        for field in ("mean_probs", "pred", "entropy", "mean_var", "suspicious", "flip", "nonfinite"):
            np.testing.assert_array_equal(getattr(t, field)[:, : j_last + 1], getattr(self.stats, field)[:, : j_last + 1], err_msg=field)

    def test_first_checkpoint_has_no_cross_checkpoint_delta(self):
        self.assertTrue(np.isnan(self.stats.delta_h[:, 0]).all())
        self.assertTrue(np.isnan(self.stats.delta_p[:, 0]).all())
        self.assertFalse(self.stats.flip[:, 0].any())

    def test_entropy_normalized_and_mi_nonnegative(self):
        self.assertTrue(((self.stats.entropy >= 0) & (self.stats.entropy <= 1 + 1e-9)).all())
        self.assertTrue((self.stats.mutual_info >= 0).all())

    def test_nonfinite_flagged_from_its_checkpoint_on(self):
        logits, ids = scenario_bank()
        s = prefix_stats(logits)
        row = list(ids["scenario"]).index("nonfinite")
        self.assertTrue(s.nonfinite[row].all())  # NaN at pass 3 -> inside every prefix
        others = [i for i in range(len(ids)) if i != row]
        self.assertFalse(s.nonfinite[others].any())
        self.assertTrue(np.isfinite(s.mean_probs).all())


if __name__ == "__main__":
    unittest.main()

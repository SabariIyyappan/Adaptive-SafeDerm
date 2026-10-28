"""Tests for the WB2 stopping/review policies (synthetic logits only).

Covers the implementation.md §4 test list: pass counts on the 5-step grid,
stable stop, budget exhaustion, all four tiers reachable, review-reason
consistency, explanation check reported NOT_RUN, identical prefixes across
methods, no ground-truth leakage.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.wb2.controller.bank import LogitBank
from src.wb2.controller.features import CHECKPOINTS, prefix_stats
from src.wb2.controller.policies import (
    BUDGET_EXHAUSTED,
    CRITICAL_TIER,
    HIGH_ENTROPY,
    NONFINITE,
    PROB_COLUMNS,
    TIERS,
    UNSTABLE_CLASS,
    FixedT,
    RiskAware,
    VarianceConvergence,
)
from src.wb2.controller.replay import replay_bank
from tests.wb2_controller.synthetic import (
    TEST_RISK_PARAMS,
    make_random_bank,
    scenario_bank,
    synthetic_manifest,
)


class RiskAwareScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logits, ids = scenario_bank()
        cls.d = RiskAware(**TEST_RISK_PARAMS).decide(prefix_stats(logits)).set_index(ids["scenario"])

    def row(self, name):
        return self.d.loc[name]

    def test_low_tier_stops_at_minimum(self):
        r = self.row("benign_stable")
        self.assertEqual((r.tier, r.passes, r.decision, r.stop_reason), ("low", 10, "TRUSTED RESULT", "converged"))

    def test_medium_tier_stable_case_is_trusted(self):
        r = self.row("benign_medium")
        self.assertEqual((r.tier, r.passes, r.review), ("medium", 10, False))

    def test_stable_suspicious_is_trusted_but_flagged_for_attention(self):
        r = self.row("mel_stable")
        self.assertEqual((r.tier, r.passes, r.review, r.pred_dx), ("high", 20, False, "mel"))
        self.assertTrue(r.clinical_attention)

    def test_benign_prediction_with_suspicious_mass_is_critical_and_reviewed(self):
        r = self.row("benign_masked_mel")
        self.assertEqual((r.tier, r.passes, r.review, r.pred_dx), ("critical", 50, True, "nv"))
        self.assertIn(CRITICAL_TIER, r.review_reasons.split("|"))
        self.assertIn(BUDGET_EXHAUSTED, r.review_reasons.split("|"))

    def test_high_entropy_exhausts_budget(self):
        r = self.row("high_entropy")
        self.assertEqual((r.tier, r.passes, r.stop_reason), ("high", 50, "budget_exhausted"))
        self.assertIn(HIGH_ENTROPY, r.review_reasons.split("|"))

    def test_flipping_class_is_unstable(self):
        r = self.row("flip")
        self.assertEqual((r.passes, r.review), (50, True))
        self.assertIn(UNSTABLE_CLASS, r.review_reasons.split("|"))

    def test_nonfinite_goes_to_review_immediately(self):
        r = self.row("nonfinite")
        self.assertEqual((r.passes, r.stop_reason, r.review_reasons), (5, "nonfinite", NONFINITE))

    def test_all_four_tiers_reachable(self):
        self.assertEqual(set(self.d["tier"]), set(TIERS))

    def test_explanation_check_never_claimed(self):
        self.assertTrue((self.d["explanation_check"] == "NOT_RUN").all())


class PolicyInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mc, cls.det, cls.ids = make_random_bank(300, seed=5)
        cls.stats = prefix_stats(cls.mc)
        cls.policies = [
            FixedT(10), FixedT(50, referral_entropy=0.5),
            VarianceConvergence(eps_p=0.02, eps_var=0.001, referral_entropy=0.6),
            RiskAware(**TEST_RISK_PARAMS),
        ]
        cls.decisions = {p.method: p.decide(cls.stats) for p in cls.policies}

    def test_pass_counts_on_grid_and_minimum(self):
        for name in ("variance_adaptive", "risk_aware_adaptive"):
            passes = set(self.decisions[name]["passes"])
            self.assertTrue(passes <= set(CHECKPOINTS), name)
            self.assertGreaterEqual(min(passes), 10, name)
            self.assertLessEqual(max(passes), 50, name)

    def test_adaptive_probs_equal_fixed_prefix_probs(self):
        """A policy that stopped at k reports exactly the fixed-T=k prediction."""
        fixed = {k: FixedT(k).decide(self.stats)[PROB_COLUMNS].to_numpy() for k in CHECKPOINTS}
        for name in ("variance_adaptive", "risk_aware_adaptive"):
            d = self.decisions[name]
            for i, k in enumerate(d["passes"]):
                np.testing.assert_array_equal(d[PROB_COLUMNS].to_numpy()[i], fixed[k][i])

    def test_fixed_T_is_mean_of_first_T(self):
        d = self.decisions["fixed_T10"]
        e = np.exp(self.mc[:, :10, :].astype(np.float64))
        expected = (e / e.sum(axis=2, keepdims=True)).mean(axis=1)
        np.testing.assert_allclose(d[PROB_COLUMNS].to_numpy(), expected, rtol=1e-6)
        self.assertTrue((d["passes"] == 10).all())

    def test_review_reasons_consistent_with_flag(self):
        for name, d in self.decisions.items():
            self.assertTrue(((d["review_reasons"] != "") == d["review"]).all(), name)
            self.assertTrue(((d["decision"] == "NEEDS REVIEW") == d["review"]).all(), name)
        r = self.decisions["risk_aware_adaptive"]
        exhausted = r["stop_reason"] == "budget_exhausted"
        self.assertTrue(r.loc[exhausted, "review"].all())
        self.assertTrue(r.loc[exhausted, "review_reasons"].str.contains(BUDGET_EXHAUSTED).all())
        self.assertFalse(r.loc[~exhausted & (r["stop_reason"] != "nonfinite"), "review"].any())

    def test_referral_threshold_applies(self):
        d = self.decisions["fixed_T50+referral"]
        np.testing.assert_array_equal(d["review"].to_numpy(), d["entropy"].to_numpy() >= 0.5)

    def test_constant_logits_converge_for_variance_policy(self):
        logits, _ = scenario_bank(("benign_stable",))
        d = VarianceConvergence(eps_p=1e-6, eps_var=1e-9).decide(prefix_stats(logits))
        self.assertEqual((d["passes"].iloc[0], d["stop_reason"].iloc[0]), (10, "converged"))

    def test_no_ground_truth_leakage(self):
        """Shuffling y_true (and only y_true) leaves every decision identical."""
        manifest = synthetic_manifest("val", len(self.ids))
        bank = LogitBank("val", self.mc, self.ids.copy(), manifest, self.det)
        shuffled = self.ids.copy()
        shuffled["y_true"] = np.random.default_rng(0).permutation(shuffled["y_true"].to_numpy())
        bank2 = LogitBank("val", self.mc, shuffled, manifest, self.det)
        a, b = replay_bank(bank, self.policies), replay_bank(bank2, self.policies)
        pd.testing.assert_frame_equal(a, b)
        self.assertNotIn("y_true", a.columns)

    def test_deterministic_baseline_is_single_pass(self):
        manifest = synthetic_manifest("val", len(self.ids))
        d = replay_bank(LogitBank("val", self.mc, self.ids, manifest, self.det), [])
        self.assertEqual(set(d["method"]), {"det_T1"})
        self.assertTrue((d["passes"] == 1).all())
        np.testing.assert_array_equal(d["pred"].to_numpy(), self.det.argmax(axis=1))


if __name__ == "__main__":
    unittest.main()

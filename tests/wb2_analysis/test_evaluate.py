"""Tests for WB2 evaluation metrics on hand-built toy decisions."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.wb2.analysis.evaluate import (
    comparison_table,
    coverage_risk_curve,
    method_summary,
    nll,
    safety_counts,
    stop_distribution,
    tier_distribution,
)
from src.wb2.controller.policies import PROB_COLUMNS

MEL, NV, BCC = 4, 5, 1


def toy(y_true, pred, review, passes=None, method="m"):
    """One-hot-ish probabilities so pred == argmax."""
    n = len(y_true)
    probs = np.full((n, 7), 0.02)
    probs[np.arange(n), pred] = 0.88
    d = pd.DataFrame(probs, columns=PROB_COLUMNS)
    d["image_id"] = [f"I{i}" for i in range(n)]
    d["lesion_id"] = [f"L{i}" for i in range(n)]
    d["method"] = method
    d["y_true"] = y_true
    d["pred"] = pred
    d["review"] = review
    d["passes"] = passes if passes is not None else [50] * n
    d["clinical_attention"] = np.isin(pred, [MEL, BCC])
    d["tier"] = ""
    d["entropy"] = np.linspace(0, 1, n)
    return d


class SafetyCountTests(unittest.TestCase):
    def test_two_denominators_never_mixed(self):
        # 4 true mel: one TRUSTED+correct, one TRUSTED+missed(nv), one reviewed+missed, one reviewed+correct
        d = toy([MEL, MEL, MEL, MEL, NV], [MEL, NV, NV, MEL, NV], [False, False, True, True, False])
        s = safety_counts(d)
        self.assertEqual((s["mel_total"], s["mel_accepted"], s["mel_missed_accepted"]), (4, 2, 1))
        self.assertEqual(s["mel_miss_all_str"], "1/4")
        self.assertEqual(s["mel_miss_accepted_str"], "1/2")
        self.assertAlmostEqual(s["mel_miss_all"], 0.25)
        self.assertAlmostEqual(s["mel_miss_accepted"], 0.5)

    def test_mel_predicted_as_bcc_is_a_mel_miss_but_not_false_reassurance(self):
        d = toy([MEL], [BCC], [False])
        s = safety_counts(d)
        self.assertEqual(s["mel_missed_accepted"], 1)
        self.assertEqual(s["suspicious_false_reassurance"], 0)

    def test_no_accepted_mel_gives_nan_not_zero(self):
        s = safety_counts(toy([MEL, NV], [NV, NV], [True, False]))
        self.assertTrue(np.isnan(s["mel_miss_accepted"]))
        self.assertEqual(s["mel_miss_accepted_str"], "0/0")


class SummaryTests(unittest.TestCase):
    def setUp(self):
        y = list(range(7)) * 3
        pred = list(y)
        pred[0] = NV
        self.d = toy(y, pred, [False] * 20 + [True], passes=[10] * 10 + [50] * 11)

    def test_summary_values(self):
        s = method_summary(self.d)
        self.assertEqual(s["n"], 21)
        self.assertAlmostEqual(s["accuracy"], 20 / 21)
        self.assertAlmostEqual(s["review_rate"], 1 / 21)
        self.assertAlmostEqual(s["coverage"], 20 / 21)
        self.assertAlmostEqual(s["passes_mean"], (100 + 550) / 21)
        self.assertAlmostEqual(s["pct_at_cap"], 11 / 21)
        self.assertAlmostEqual(s["mel_recall"], 1.0)

    def test_nll_matches_hand_value(self):
        self.assertAlmostEqual(nll(np.array([0]), np.array([[0.5, 0.5, 0, 0, 0, 0, 0]])), np.log(2))

    def test_missing_class_gives_nan_auroc(self):
        s = method_summary(toy([NV, MEL], [NV, MEL], [False, False]))
        self.assertTrue(np.isnan(s["macro_auroc"]))

    def test_tables(self):
        two = pd.concat([self.d, self.d.assign(method="other")])
        t = comparison_table(two, status="PILOT ONLY")
        self.assertEqual(list(t["method"]), ["m", "other"])
        self.assertTrue((t["status"] == "PILOT ONLY").all())
        sd = stop_distribution(two)
        self.assertEqual(int(sd.loc[sd["method"] == "m", 10].iloc[0]), 10)
        self.assertTrue(tier_distribution(two).empty)

    def test_coverage_curve_full_coverage_matches_overall_error(self):
        c = coverage_risk_curve(self.d)
        self.assertAlmostEqual(c["coverage"].iloc[-1], 1.0)
        self.assertAlmostEqual(c["error_rate"].iloc[-1], 1 / 21)
        self.assertTrue(np.all(np.diff(c["coverage"]) >= 0))


if __name__ == "__main__":
    unittest.main()

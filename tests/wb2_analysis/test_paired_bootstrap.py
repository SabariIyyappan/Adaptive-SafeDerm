"""Tests for the paired lesion-grouped bootstrap."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.wb2.analysis.paired_bootstrap import lesion_resample_indices, paired_bootstrap
from src.wb2.controller.bank import LogitBank
from src.wb2.controller.policies import FixedT
from src.wb2.controller.replay import attach_labels, replay_bank
from tests.wb2_controller.synthetic import make_random_bank, synthetic_manifest


class LesionResampleTests(unittest.TestCase):
    def test_whole_lesions_are_drawn(self):
        lesions = np.array(["A", "A", "B", "C", "C", "C"])
        rng = np.random.default_rng(0)
        for _ in range(50):
            idx = lesion_resample_indices(lesions, rng)
            drawn = lesions[idx]
            for lesion, size in (("A", 2), ("B", 1), ("C", 3)):
                self.assertEqual((drawn == lesion).sum() % size, 0)


class PairedBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mc, det, ids = make_random_bank(200, seed=7)
        bank = LogitBank("val", mc, ids, synthetic_manifest("val", len(ids)), det)
        cls.labeled = attach_labels(replay_bank(bank, [FixedT(10), FixedT(50)]), bank)

    def by(self, method):
        return self.labeled[self.labeled["method"] == method]

    def test_identical_methods_have_zero_difference(self):
        t = paired_bootstrap(self.by("fixed_T50"), self.by("fixed_T50"), n_resamples=30)
        np.testing.assert_allclose(t[["diff_b_minus_a", "ci_low", "ci_high"]].to_numpy(), 0.0)
        self.assertFalse(t["ci_excludes_zero"].any())

    def test_reproducible_and_labeled(self):
        a = paired_bootstrap(self.by("fixed_T10"), self.by("fixed_T50"), n_resamples=30)
        b = paired_bootstrap(self.by("fixed_T10"), self.by("fixed_T50"), n_resamples=30)
        np.testing.assert_array_equal(a["ci_low"].to_numpy(), b["ci_low"].to_numpy())
        self.assertTrue((a["resampling_unit"] == "lesion_id").all())
        row = a.set_index("metric").loc["passes_mean"]
        self.assertAlmostEqual(row["diff_b_minus_a"], 40.0)

    def test_requires_same_images(self):
        with self.assertRaises(ValueError):
            paired_bootstrap(self.by("fixed_T10").iloc[1:], self.by("fixed_T50"), n_resamples=5)


if __name__ == "__main__":
    unittest.main()

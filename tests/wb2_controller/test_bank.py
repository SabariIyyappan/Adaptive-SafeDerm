"""Tests for WB2 logit-bank loading/validation (synthetic banks in a temp dir)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.eda import CLASS_ORDER
from src.wb2.controller.bank import load_bank, save_bank
from tests.wb2_controller.synthetic import make_random_bank, synthetic_manifest


class BankTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.mc, self.det, self.ids = make_random_bank(40, seed=1)
        split = self.ids.rename(columns={"y_true": "label"})
        split["dx"] = [CLASS_ORDER[i] for i in split["label"]]
        self.split_csv = self.dir / "val.csv"
        split[["image_id", "lesion_id", "dx", "label"]].to_csv(self.split_csv, index=False)

    def tearDown(self):
        self._tmp.cleanup()

    def _save(self, **manifest_overrides):
        m = {**synthetic_manifest("val", len(self.ids)), **manifest_overrides}
        save_bank(self.dir, "val", self.mc, self.ids, m, self.det)

    def test_round_trip(self):
        self._save()
        bank = load_bank(self.dir, "val", split_csv=self.split_csv)
        self.assertEqual(bank.logits.shape, (40, 50, 7))
        self.assertEqual(bank.det_logits.shape, (40, 7))
        np.testing.assert_array_equal(bank.y_true, self.ids["y_true"].to_numpy())
        self.assertTrue(bank.is_synthetic)
        self.assertIn("val_mc_logits.npz", bank.files_sha256)

    def test_missing_image_rejected(self):
        self._save()
        pd.read_csv(self.split_csv).iloc[1:].to_csv(self.split_csv, index=False)
        with self.assertRaisesRegex(ValueError, "differ"):
            load_bank(self.dir, "val", split_csv=self.split_csv)

    def test_subset_allowed_only_when_declared(self):
        sub = self.ids.iloc[:20].reset_index(drop=True)
        save_bank(self.dir, "val", self.mc[:20], sub, synthetic_manifest("val", 20), self.det[:20])
        with self.assertRaises(ValueError):
            load_bank(self.dir, "val", split_csv=self.split_csv)
        self.assertEqual(load_bank(self.dir, "val", split_csv=self.split_csv, allow_subset=True).n_images, 20)

    def test_label_mismatch_rejected(self):
        self.ids.loc[0, "y_true"] = (self.ids.loc[0, "y_true"] + 1) % 7
        self._save()
        with self.assertRaisesRegex(ValueError, "y_true"):
            load_bank(self.dir, "val", split_csv=self.split_csv)

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            save_bank(self.dir, "val", self.mc[:, :, :6], self.ids, synthetic_manifest("val", 40))
        with self.assertRaises(ValueError):
            save_bank(self.dir, "val", self.mc[:39], self.ids, synthetic_manifest("val", 40))

    def test_duplicate_ids_rejected(self):
        self.ids.loc[1, "image_id"] = self.ids.loc[0, "image_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            save_bank(self.dir, "val", self.mc, self.ids, synthetic_manifest("val", 40))

    def test_partial_requires_opt_in_and_failed_refused(self):
        self._save(status="PARTIAL")
        with self.assertRaisesRegex(ValueError, "PARTIAL"):
            load_bank(self.dir, "val")
        load_bank(self.dir, "val", allow_partial=True)
        self._save(status="FAILED")
        with self.assertRaisesRegex(ValueError, "FAILED"):
            load_bank(self.dir, "val", allow_partial=True)

    def test_class_order_mismatch_rejected(self):
        self._save(class_order=list(reversed(CLASS_ORDER)))
        with self.assertRaisesRegex(ValueError, "class_order"):
            load_bank(self.dir, "val")

    def test_nonfinite_rows_reported(self):
        self.mc[3, 7, 1] = np.inf
        self._save()
        np.testing.assert_array_equal(load_bank(self.dir, "val").nonfinite_rows, [3])

    def test_manifest_is_json(self):
        self._save()
        self.assertEqual(json.loads((self.dir / "val_manifest.json").read_text())["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()

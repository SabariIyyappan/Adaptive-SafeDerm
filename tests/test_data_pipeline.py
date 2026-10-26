"""Unit tests for Member A's V1 pipeline, run against the synthetic fixture
in fixtures.py (PLAN.md's own "dummy-first" rule, §4.4, applied to A's own
modules so the logic is verified before real HAM10000 data is available —
see data/README.md).

Uses Python's stdlib `unittest` rather than pytest: this environment's
network policy blocks pypi.org, so pytest cannot be installed here. Run
with:

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.eda import dataset_summary_table, run_eda
from src.data.integrity_audit import run_integrity_audit
from src.data.labels import EXPECTED_HAM10000_LABEL_MAP, build_label_map
from src.data.split import build_split, lesion_level_table
from src.data.verify_split import verify_split
from tests.fixtures import TempFixture


class LabelMapTests(unittest.TestCase):
    def test_alphabetical_and_matches_expected(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            label_map = build_label_map(metadata["dx"])
            self.assertEqual(label_map, EXPECTED_HAM10000_LABEL_MAP)
            self.assertEqual(list(label_map.keys()), sorted(label_map.keys()))

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            build_label_map([])


class LesionLevelTableTests(unittest.TestCase):
    def test_one_row_per_lesion(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            lesions = lesion_level_table(metadata)
            self.assertTrue(lesions["lesion_id"].is_unique)
            self.assertEqual(set(lesions["lesion_id"]), set(metadata["lesion_id"]))

    def test_rejects_inconsistent_dx(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            corrupted = metadata.copy()
            corrupted.loc[0, "dx"] = "vasc" if corrupted.loc[0, "dx"] != "vasc" else "mel"
            with self.assertRaises(ValueError):
                lesion_level_table(corrupted)


class SplitTests(unittest.TestCase):
    def test_no_lesion_or_image_leakage(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            split_df = build_split(metadata, seed=42)

            lesion_to_splits = split_df.groupby("lesion_id")["split"].nunique()
            self.assertTrue((lesion_to_splits == 1).all(), "every lesion must land in exactly one split")
            self.assertTrue(split_df["image_id"].is_unique, "no image should appear more than once")

    def test_union_covers_all_images(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            split_df = build_split(metadata, seed=42)
            self.assertEqual(set(split_df["image_id"]), set(metadata["image_id"]))

    def test_every_class_in_every_split(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            split_df = build_split(metadata, seed=42)
            all_classes = set(metadata["dx"].unique())
            for split_name in ("train", "val", "test"):
                present = set(split_df.loc[split_df["split"] == split_name, "dx"].unique())
                self.assertEqual(present, all_classes, f"split '{split_name}' missing {all_classes - present}")

    def test_deterministic_given_seed(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            a = build_split(metadata, seed=42)
            b = build_split(metadata, seed=42)
            pd.testing.assert_frame_equal(
                a.sort_values("image_id").reset_index(drop=True),
                b.sort_values("image_id").reset_index(drop=True),
            )

    def test_roughly_matches_target_fractions(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            split_df = build_split(metadata, seed=42)
            lesion_counts = split_df.drop_duplicates("lesion_id")["split"].value_counts(normalize=True)
            # Small fixture (120 lesions) so allow a generous tolerance around 70/15/15.
            self.assertAlmostEqual(lesion_counts["train"], 0.70, delta=0.03)
            self.assertAlmostEqual(lesion_counts["val"], 0.15, delta=0.03)
            self.assertAlmostEqual(lesion_counts["test"], 0.15, delta=0.03)


class VerifySplitTests(unittest.TestCase):
    def test_passes_on_a_correct_split(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            split_df = build_split(metadata, seed=42)
            splits = {n: split_df[split_df["split"] == n].drop(columns="split") for n in ("train", "val", "test")}
            result = verify_split(splits, metadata)
            self.assertTrue(result.passed, result.anomalies)

    def test_catches_injected_lesion_leakage(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            split_df = build_split(metadata, seed=42)
            splits = {n: split_df[split_df["split"] == n].drop(columns="split") for n in ("train", "val", "test")}
            leaked_row = splits["train"].iloc[[0]].copy()
            leaked_row["image_id"] = "ISIC_LEAK_0000000"
            splits["test"] = pd.concat([splits["test"], leaked_row], ignore_index=True)

            result = verify_split(splits, metadata)
            self.assertFalse(result.passed)
            self.assertTrue(any("lesion_id overlap" in a for a in result.anomalies))


class IntegrityAuditTests(unittest.TestCase):
    def test_passes_on_the_fixture_with_its_own_expected_counts(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            expected_class_counts = metadata["dx"].value_counts().to_dict()
            result = run_integrity_audit(
                data_root,
                expected_image_count=metadata["image_id"].nunique(),
                expected_lesion_count=metadata["lesion_id"].nunique(),
                expected_class_counts=expected_class_counts,
            )
            self.assertTrue(result.passed, result.anomalies)

    def test_catches_wrong_expected_counts(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            expected_class_counts = metadata["dx"].value_counts().to_dict()
            result = run_integrity_audit(
                data_root,
                expected_image_count=999_999,
                expected_lesion_count=1,
                expected_class_counts=expected_class_counts,
            )
            self.assertFalse(result.passed)
            self.assertTrue(any("unique image_ids" in a for a in result.anomalies))
            self.assertTrue(any("unique lesion_ids" in a for a in result.anomalies))

    def test_catches_missing_image_file(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            expected_class_counts = metadata["dx"].value_counts().to_dict()
            victim = metadata.iloc[0]["image_id"]
            (data_root / "images" / f"{victim}.jpg").unlink()

            result = run_integrity_audit(
                data_root,
                expected_image_count=metadata["image_id"].nunique(),
                expected_lesion_count=metadata["lesion_id"].nunique(),
                expected_class_counts=expected_class_counts,
            )
            self.assertFalse(result.passed)
            self.assertTrue(any("missing" in a.lower() for a in result.anomalies))


class EdaTests(unittest.TestCase):
    def test_dataset_summary_table_sums_to_total(self):
        with TempFixture() as data_root:
            metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
            summary = dataset_summary_table(metadata)
            self.assertEqual(summary["n_images"].sum(), len(metadata))
            self.assertAlmostEqual(summary["pct_images"].sum(), 100.0, delta=0.05)

    def test_run_eda_produces_expected_files(self):
        with TempFixture() as data_root:
            out_dir = data_root / "eda_out"
            run_eda(data_root, out_dir)
            self.assertTrue((out_dir / "class_distribution.png").exists())
            self.assertTrue((out_dir / "images_per_lesion_hist.png").exists())
            self.assertTrue((out_dir / "dataset_summary.csv").exists())
            self.assertTrue((out_dir / "sample_grid.png").exists())


if __name__ == "__main__":
    unittest.main()

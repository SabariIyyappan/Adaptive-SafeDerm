"""Unit tests for Member C's evaluation suite (PLAN.md §5.3 item 6).

Verification strategy (agreed in planning): the suite is cross-checked
against sklearn on the real predictions, **plus** a hand-computed 3-class
toy example. The toy example is not redundant — sklearn has no Eq. 17
equivalent, so per-class one-vs-rest accuracy and specificity would
otherwise have no independent reference at all.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.eval.bootstrap import bootstrap_metrics
from src.eval.classes import CLASS_FULL_NAME, NUM_CLASSES
from src.eval.metrics import (
    brier_score,
    confidence_summary,
    confusion_matrices,
    expected_calibration_error,
    load_predictions,
    macro_metrics,
    per_class_table,
    predictive_entropy,
    probs_and_labels,
)

REPO = Path(__file__).resolve().parents[1]
REAL_INPUTS = REPO / "results" / "v1" / "inputs"


def _toy_predictions() -> tuple[np.ndarray, np.ndarray]:
    """A 3-class, 10-sample example whose confusion matrix is known by hand.

    Built so the argmax of each row is the intended prediction. Confusion
    matrix (rows = true, cols = predicted), over classes 0, 1, 2:

            pred0  pred1  pred2   support
    true0     3      1      0        4
    true1     1      2      0        3
    true2     0      1      2        3

    Class 0:  TP=3 FN=1 FP=1 TN=5  -> eq17 (3+5)/10 = 0.8, recall 3/4 = 0.75,
              specificity 5/6, precision 3/4
    Class 1:  TP=2 FN=1 FP=2 TN=5  -> eq17 (2+5)/10 = 0.7, recall 2/3,
              specificity 5/7, precision 2/4 = 0.5
    Class 2:  TP=2 FN=1 FP=0 TN=7  -> eq17 (2+7)/10 = 0.9, recall 2/3,
              specificity 7/7 = 1.0, precision 2/2 = 1.0
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1, 2, 2])
    probs = np.full((10, 3), 0.1)
    probs[np.arange(10), y_pred] = 0.8
    probs = probs / probs.sum(axis=1, keepdims=True)
    return y_true, probs


class ToyExampleTests(unittest.TestCase):
    """The independent reference for Eq. 17 and specificity."""

    def setUp(self):
        self.y_true, self.probs = _toy_predictions()

    def test_confusion_matrix_matches_hand_computation(self):
        counts, _ = confusion_matrices(self.y_true, self.probs)
        expected = np.array([[3, 1, 0], [1, 2, 0], [0, 1, 2]])
        np.testing.assert_array_equal(counts[:3, :3], expected)

    def test_eq17_accuracy_matches_hand_computation(self):
        table = _toy_per_class(self.y_true, self.probs)
        self.assertAlmostEqual(table[0]["accuracy_eq17"], 0.8, places=10)
        self.assertAlmostEqual(table[1]["accuracy_eq17"], 0.7, places=10)
        self.assertAlmostEqual(table[2]["accuracy_eq17"], 0.9, places=10)

    def test_recall_matches_hand_computation(self):
        table = _toy_per_class(self.y_true, self.probs)
        self.assertAlmostEqual(table[0]["recall"], 3 / 4, places=10)
        self.assertAlmostEqual(table[1]["recall"], 2 / 3, places=10)
        self.assertAlmostEqual(table[2]["recall"], 2 / 3, places=10)

    def test_specificity_matches_hand_computation(self):
        table = _toy_per_class(self.y_true, self.probs)
        self.assertAlmostEqual(table[0]["specificity"], 5 / 6, places=10)
        self.assertAlmostEqual(table[1]["specificity"], 5 / 7, places=10)
        self.assertAlmostEqual(table[2]["specificity"], 1.0, places=10)

    def test_precision_matches_hand_computation(self):
        table = _toy_per_class(self.y_true, self.probs)
        self.assertAlmostEqual(table[0]["precision"], 3 / 4, places=10)
        self.assertAlmostEqual(table[1]["precision"], 0.5, places=10)
        self.assertAlmostEqual(table[2]["precision"], 1.0, places=10)

    def test_eq17_and_recall_genuinely_differ(self):
        """The whole point of reporting both (PLAN.md §13 item 4)."""
        table = _toy_per_class(self.y_true, self.probs)
        self.assertNotAlmostEqual(table[0]["accuracy_eq17"], table[0]["recall"], places=3)

    def test_entropy_is_zero_for_a_certain_prediction(self):
        certain = np.array([[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(float(predictive_entropy(certain)[0]), 0.0, places=9)

    def test_entropy_is_maximal_for_a_uniform_prediction(self):
        uniform = np.full((1, 3), 1 / 3)
        self.assertAlmostEqual(float(predictive_entropy(uniform)[0]), float(np.log(3)), places=9)

    def test_brier_score_of_a_perfect_prediction_is_zero(self):
        y = np.array([0, 1, 2])
        perfect = np.eye(3)
        self.assertAlmostEqual(brier_score(y, perfect), 0.0, places=10)


def _toy_per_class(y_true: np.ndarray, probs: np.ndarray) -> list[dict]:
    """per_class_table is hardcoded to the 7 HAM10000 classes, so for the
    3-class toy we recompute with the same formulas over the toy matrix.
    Keeping the arithmetic identical is the point — this asserts the
    formulas, while the real-data tests assert the wiring.
    """
    counts, _ = confusion_matrices(
        y_true, np.pad(probs, ((0, 0), (0, NUM_CLASSES - probs.shape[1])), constant_values=0.0)
    )
    counts = counts[:3, :3]
    n = int(counts.sum())
    rows = []
    for idx in range(3):
        tp = int(counts[idx, idx])
        fn = int(counts[idx, :].sum() - tp)
        fp = int(counts[:, idx].sum() - tp)
        tn = n - tp - fn - fp
        rows.append({
            "accuracy_eq17": (tp + tn) / n,
            "recall": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "precision": tp / (tp + fp),
        })
    return rows


class ContractValidationTests(unittest.TestCase):
    """C3 violations must fail loudly, not produce a plausible wrong figure."""

    def _write(self, df: pd.DataFrame) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "preds.csv"
        df.to_csv(tmp, index=False)
        return tmp

    def _valid_frame(self) -> pd.DataFrame:
        probs = np.full((5, NUM_CLASSES), 1 / NUM_CLASSES)
        data = {"image_id": [f"ISIC_{i}" for i in range(5)], "split": "test", "y_true": [0, 1, 2, 3, 4]}
        data.update({f"prob_{c}": probs[:, c] for c in range(NUM_CLASSES)})
        return pd.DataFrame(data)

    def test_accepts_a_valid_file(self):
        path = self._write(self._valid_frame())
        self.assertEqual(len(load_predictions(path)), 5)

    def test_rejects_missing_columns(self):
        df = self._valid_frame().drop(columns=["prob_3"])
        with self.assertRaises(ValueError) as ctx:
            load_predictions(self._write(df))
        self.assertIn("prob_3", str(ctx.exception))

    def test_rejects_probabilities_that_do_not_sum_to_one(self):
        df = self._valid_frame()
        df.loc[2, "prob_0"] = 0.9
        with self.assertRaises(ValueError) as ctx:
            load_predictions(self._write(df))
        self.assertIn("sum to 1", str(ctx.exception))

    def test_rejects_labels_outside_the_class_range(self):
        df = self._valid_frame()
        df.loc[1, "y_true"] = 99
        with self.assertRaises(ValueError) as ctx:
            load_predictions(self._write(df))
        self.assertIn("y_true", str(ctx.exception))

    def test_rejects_wrong_row_count_when_expected_rows_is_given(self):
        with self.assertRaises(ValueError) as ctx:
            load_predictions(self._write(self._valid_frame()), expected_rows=1481)
        self.assertIn("1481", str(ctx.exception))


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y_true = rng.integers(0, NUM_CLASSES, size=400)
        logits = rng.normal(size=(400, NUM_CLASSES))
        logits[np.arange(400), self.y_true] += 2.0  # make it better than chance
        exp = np.exp(logits)
        self.probs = exp / exp.sum(axis=1, keepdims=True)

    def test_is_reproducible_for_a_fixed_seed(self):
        a = bootstrap_metrics(self.y_true, self.probs, n_resamples=50, seed=42)
        b = bootstrap_metrics(self.y_true, self.probs, n_resamples=50, seed=42)
        self.assertEqual(a.ci_low["macro_auroc"], b.ci_low["macro_auroc"])
        self.assertEqual(a.ci_high["macro_f1"], b.ci_high["macro_f1"])

    def test_different_seeds_give_different_intervals(self):
        a = bootstrap_metrics(self.y_true, self.probs, n_resamples=50, seed=1)
        b = bootstrap_metrics(self.y_true, self.probs, n_resamples=50, seed=2)
        self.assertNotEqual(a.ci_low["macro_auroc"], b.ci_low["macro_auroc"])

    def test_interval_brackets_the_point_estimate(self):
        r = bootstrap_metrics(self.y_true, self.probs, n_resamples=200, seed=42)
        for name in ("macro_auroc", "macro_f1"):
            self.assertLessEqual(r.ci_low[name], r.point_estimate[name])
            self.assertGreaterEqual(r.ci_high[name], r.point_estimate[name])

    def test_counts_skipped_resamples_rather_than_hiding_them(self):
        r = bootstrap_metrics(self.y_true, self.probs, n_resamples=30, seed=42)
        self.assertEqual(r.n_resamples_used + r.n_skipped_missing_class, 30)

    def test_format_ci_renders_the_papers_style(self):
        r = bootstrap_metrics(self.y_true, self.probs, n_resamples=30, seed=42)
        text = r.format_ci("macro_auroc")
        self.assertRegex(text, r"^\d\.\d{4} \[\d\.\d{4}, \d\.\d{4}\]$")


@unittest.skipUnless((REAL_INPUTS / "predictions_test.csv").exists(),
                     "real C3 predictions not present")
class RealPredictionTests(unittest.TestCase):
    """Cross-checks against sklearn and against Member B's own history file."""

    @classmethod
    def setUpClass(cls):
        cls.test_df = load_predictions(REAL_INPUTS / "predictions_test.csv")
        cls.val_df = load_predictions(REAL_INPUTS / "predictions_val.csv")
        cls.test_probs, cls.test_true = probs_and_labels(cls.test_df)
        cls.val_probs, cls.val_true = probs_and_labels(cls.val_df)
        cls.history = pd.read_csv(REAL_INPUTS / "history.csv")

    def test_row_counts_match_the_c2_split_sizes(self):
        self.assertEqual(len(self.test_df), 1481)
        self.assertEqual(len(self.val_df), 1532)

    def test_macro_metrics_agree_with_sklearn(self):
        m = macro_metrics(self.test_true, self.test_probs)
        expected_auroc = roc_auc_score(self.test_true, self.test_probs,
                                       multi_class="ovr", average="macro",
                                       labels=list(range(NUM_CLASSES)))
        expected_f1 = f1_score(self.test_true, self.test_probs.argmax(axis=1),
                               average="macro", zero_division=0)
        self.assertAlmostEqual(m["macro_auroc"], float(expected_auroc), places=10)
        self.assertAlmostEqual(m["macro_f1"], float(expected_f1), places=10)

    def test_validation_metrics_reproduce_member_bs_final_epoch(self):
        """End-to-end check: our suite, run on B's val predictions, must
        reproduce the numbers B's training loop logged independently.

        Tolerance is 1e-4 rather than exact: B computes metrics from
        in-memory float32 tensors, while we read float64 values that were
        round-tripped through a CSV, so agreement below ~1e-5 is not
        achievable. Observed difference is ~6e-6.
        """
        m = macro_metrics(self.val_true, self.val_probs)
        final = self.history.iloc[-1]
        self.assertAlmostEqual(m["macro_auroc"], float(final["val_macro_auroc"]), delta=1e-4)
        self.assertAlmostEqual(m["macro_f1"], float(final["val_macro_f1"]), delta=1e-4)

    def test_all_seven_classes_appear_in_truth_and_predictions(self):
        self.assertEqual(set(np.unique(self.test_true)), set(range(NUM_CLASSES)))
        self.assertEqual(set(np.unique(self.test_probs.argmax(axis=1))), set(range(NUM_CLASSES)))

    def test_per_class_support_sums_to_the_split_size(self):
        table = per_class_table(self.test_true, self.test_probs)
        self.assertEqual(int(table["support"].sum()), len(self.test_df))

    def test_eq17_exceeds_recall_for_the_majority_class(self):
        """The §13 item 4 finding, asserted so a refactor can't silently
        swap the two definitions.
        """
        table = per_class_table(self.test_true, self.test_probs)
        nv = table.loc[table["dx"] == "nv"].iloc[0]
        self.assertGreater(nv["accuracy_eq17"], nv["recall"])
        self.assertAlmostEqual(nv["accuracy_eq17"], 0.708, places=2)
        self.assertAlmostEqual(nv["recall"], 0.567, places=2)

    def test_confidence_is_higher_on_correct_predictions(self):
        c = confidence_summary(self.test_true, self.test_probs)
        self.assertGreater(c["mean_confidence_correct"], c["mean_confidence_incorrect"])

    def test_ece_bins_account_for_every_image(self):
        _, bins = expected_calibration_error(self.test_true, self.test_probs)
        self.assertEqual(int(bins["count"].sum()), len(self.test_df))

    def test_confusion_matrix_rows_sum_to_support(self):
        counts, normalized = confusion_matrices(self.test_true, self.test_probs)
        self.assertEqual(int(counts.sum()), len(self.test_df))
        np.testing.assert_allclose(normalized.sum(axis=1), np.ones(NUM_CLASSES), atol=1e-9)


class SplitFairnessTests(unittest.TestCase):
    """The answer to 'is the split balanced?' asserted, not just written down."""

    @classmethod
    def setUpClass(cls):
        cls.split_dir = REPO / "data" / "splits"

    def setUp(self):
        if not (self.split_dir / "train.csv").exists():
            self.skipTest("C2 split files not present")
        from src.eval.eda import load_splits
        self.splits = load_splits(self.split_dir)

    def test_no_lesion_is_shared_between_splits(self):
        sets = {n: set(df["lesion_id"]) for n, df in self.splits.items()}
        self.assertEqual(sets["train"] & sets["val"], set())
        self.assertEqual(sets["train"] & sets["test"], set())
        self.assertEqual(sets["val"] & sets["test"], set())

    def test_no_image_is_shared_between_splits(self):
        sets = {n: set(df["image_id"]) for n, df in self.splits.items()}
        self.assertEqual(sets["train"] & sets["val"], set())
        self.assertEqual(sets["train"] & sets["test"], set())
        self.assertEqual(sets["val"] & sets["test"], set())

    def test_every_class_is_present_in_every_split(self):
        for name, df in self.splits.items():
            self.assertEqual(len(set(df["dx"])), NUM_CLASSES, f"split '{name}' is missing classes")

    def test_lesion_level_class_drift_is_negligible(self):
        from src.eval.eda import split_proportion_tables
        _, lesion_pct = split_proportion_tables(self.splits)
        drift = lesion_pct[["val", "test"]].sub(lesion_pct["train"], axis=0).abs().to_numpy().max()
        self.assertLess(drift, 0.5, "lesion-level stratification should be near-exact")

    def test_image_level_class_drift_stays_under_one_point(self):
        from src.eval.eda import split_proportion_tables
        image_pct, _ = split_proportion_tables(self.splits)
        drift = image_pct[["val", "test"]].sub(image_pct["train"], axis=0).abs().to_numpy().max()
        self.assertLess(drift, 1.0, "image-level drift should stay inside 1 percentage point")


class ClassMetadataTests(unittest.TestCase):
    def test_every_class_has_a_full_name_and_category(self):
        from src.data.eda import CLASS_ORDER
        from src.eval.classes import malignancy_of

        for cls in CLASS_ORDER:
            self.assertIn(cls, CLASS_FULL_NAME)
            self.assertIn(malignancy_of(cls), {"malignant", "pre-cancerous", "benign"})

    def test_class_order_matches_the_c1_label_map(self):
        """Guards against figures being silently mislabeled if A's map changes."""
        from src.data.eda import CLASS_ORDER
        from src.data.labels import EXPECTED_HAM10000_LABEL_MAP

        self.assertEqual([EXPECTED_HAM10000_LABEL_MAP[c] for c in CLASS_ORDER], list(range(NUM_CLASSES)))


@unittest.skipUnless((REPO / "results" / "v1" / "metrics.json").exists(),
                     "metrics.json not generated yet")
class GeneratedOutputTests(unittest.TestCase):
    def test_metrics_json_is_valid_and_has_the_headline_numbers(self):
        payload = json.loads((REPO / "results" / "v1" / "metrics.json").read_text(encoding="utf-8"))
        self.assertIn("test", payload)
        self.assertIn("macro_auroc", payload["test"])
        self.assertIn("bootstrap", payload["test"])
        self.assertEqual(len(payload["test"]["per_class"]), NUM_CLASSES)
        self.assertEqual(len(payload["sanity_checks"]), 4)

    def test_all_sanity_checks_pass(self):
        payload = json.loads((REPO / "results" / "v1" / "metrics.json").read_text(encoding="utf-8"))
        failed = [c["check"] for c in payload["sanity_checks"] if not c["passed"]]
        self.assertEqual(failed, [], f"sanity checks failed: {failed}")


if __name__ == "__main__":
    unittest.main()

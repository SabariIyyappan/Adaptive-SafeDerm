"""Unit tests for Member B's V1 model/training code (PLAN.md §5.3), run
against the same synthetic HAM10000-shaped fixture Member A's tests use
(dummy-first rule, §4.4) plus A's real split.py, since B consumes C2 exactly
as A produces it.

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.data.split import build_split, write_split_files
from src.model.dataset import HAM10000Split, build_eval_transform, build_train_transform
from src.model.imbalance import build_weighted_sampler, class_counts, class_weights_for_loss
from src.model.model import NUM_CLASSES, build_model
from src.model.train import run_epoch_eval, run_epoch_train, set_seed, train
from tests.fixtures import TempFixture


def _build_fixture_splits(data_root: Path) -> Path:
    metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
    split_df = build_split(metadata)
    split_dir = data_root / "splits"
    write_split_files(split_df, split_dir)
    return split_dir


class ImbalanceTests(unittest.TestCase):
    def test_class_counts_matches_manual_tally(self):
        labels = [0, 0, 0, 1, 1, 2]
        self.assertEqual(class_counts(labels, num_classes=3), [3, 2, 1])

    def test_class_counts_rejects_missing_class(self):
        with self.assertRaises(ValueError):
            class_counts([0, 0, 1], num_classes=3)

    def test_class_weights_formula(self):
        # n = [3, 2, 1] -> inv = [1/3, 1/2, 1] -> sum = 11/6
        weights = class_weights_for_loss([0, 0, 0, 1, 1, 2], num_classes=3)
        expected = torch.tensor([(1 / 3) / (11 / 6), (1 / 2) / (11 / 6), 1 / (11 / 6)])
        self.assertTrue(torch.allclose(weights, expected, atol=1e-6))
        # Rarer classes get larger weight.
        self.assertTrue(weights[2] > weights[1] > weights[0])

    def test_weighted_sampler_balances_rare_class_over_many_draws(self):
        # 100 majority-class samples, 5 minority-class samples.
        labels = [0] * 100 + [1] * 5
        sampler = build_weighted_sampler(labels, num_classes=2)
        drawn = list(sampler)
        self.assertEqual(len(drawn), len(labels))
        minority_fraction = sum(1 for i in drawn if labels[i] == 1) / len(drawn)
        # Not exactly 0.5 (stochastic), but nowhere near the raw 5/105 ~ 0.048.
        self.assertGreater(minority_fraction, 0.25)


class ModelTests(unittest.TestCase):
    def test_head_shape_and_forward_pass(self):
        model = build_model(pretrained=False)
        head = model.classifier
        self.assertEqual(len(head), 4)
        self.assertIsInstance(head[0], torch.nn.Linear)
        self.assertEqual((head[0].in_features, head[0].out_features), (1280, 256))
        self.assertIsInstance(head[1], torch.nn.ReLU)
        self.assertIsInstance(head[2], torch.nn.Dropout)
        self.assertAlmostEqual(head[2].p, 0.3)
        self.assertIsInstance(head[3], torch.nn.Linear)
        self.assertEqual((head[3].in_features, head[3].out_features), (256, NUM_CLASSES))

        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, NUM_CLASSES))

    def test_all_parameters_trainable(self):
        model = build_model(pretrained=False)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


class DatasetTests(unittest.TestCase):
    def test_reads_c2_columns_and_matching_image(self):
        with TempFixture() as data_root:
            split_dir = _build_fixture_splits(data_root)
            ds = HAM10000Split(split_dir / "train.csv", data_root / "images", build_eval_transform())
            self.assertGreater(len(ds), 0)
            tensor, label, image_id = ds[0]
            self.assertEqual(tuple(tensor.shape), (3, 224, 224))
            self.assertIsInstance(label, int)
            self.assertIsInstance(image_id, str)

    def test_rejects_split_csv_missing_columns(self):
        with TempFixture() as data_root:
            bad_csv = data_root / "bad.csv"
            pd.DataFrame({"image_id": ["x"], "dx": ["nv"]}).to_csv(bad_csv, index=False)
            with self.assertRaises(ValueError):
                HAM10000Split(bad_csv, data_root / "images", build_eval_transform())

    def test_train_transform_is_stochastic_eval_transform_is_not(self):
        with TempFixture() as data_root:
            split_dir = _build_fixture_splits(data_root)
            train_ds = HAM10000Split(split_dir / "train.csv", data_root / "images", build_train_transform())
            eval_ds = HAM10000Split(split_dir / "train.csv", data_root / "images", build_eval_transform())

            eval_a, _, _ = eval_ds[0]
            eval_b, _, _ = eval_ds[0]
            self.assertTrue(torch.equal(eval_a, eval_b))

            train_samples = [train_ds[0][0] for _ in range(10)]
            self.assertTrue(any(not torch.equal(train_samples[0], s) for s in train_samples[1:]))


class SmokeTrainTests(unittest.TestCase):
    """PLAN.md §5.3.8: a tiny smoke run to confirm the loop works end-to-end
    and the loss moves — not a claim about real HAM10000 performance.
    """

    def test_two_epochs_on_fixture_run_without_error_and_produce_artifacts(self):
        with TempFixture() as data_root:
            split_dir = _build_fixture_splits(data_root)
            out_dir = data_root / "run_out"
            config = {
                "run_id": "smoke-test",
                "seed": 42,
                "pretrained": False,  # no network access needed for the test
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 8,
                "epochs": 2,
            }
            train(split_dir, data_root / "images", config, out_dir, device="cpu")

            self.assertTrue((out_dir / "last.pt").exists())
            self.assertTrue((out_dir / "best.pt").exists())
            history = pd.read_csv(out_dir / "history.csv")
            self.assertEqual(len(history), 2)
            self.assertEqual(
                list(history.columns),
                ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_macro_auroc", "lr"],
            )

    def test_set_seed_is_reproducible_for_torch_rand(self):
        set_seed(42)
        a = torch.rand(5)
        set_seed(42)
        b = torch.rand(5)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()

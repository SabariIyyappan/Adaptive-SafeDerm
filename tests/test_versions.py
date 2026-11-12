"""Unit tests for the V2-V5 additions (preprocessing, multi-backbone model,
MC Dropout, uncertainty/referral, masks, Grad-CAM). Synthetic inputs only —
no dataset, GPU or checkpoint needed.

    python -m unittest tests.test_versions -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.data.masks import align_mask
from src.data.preprocess import TARGET_SIZE, clahe_lab, hair_mask, preprocess_rgb
from src.eval.uncertainty import accept_mask, decompose, referral_table, referral_threshold
from src.model.gradcam import GradCAM, binarize, iou, upsample_normalize
from src.model.mc_dropout import enable_head_dropout, sample_head, verify_feature_reuse
from src.model.model import ARCHITECTURES, NUM_CLASSES, arch_of, build_model, pooled_features


def _synthetic_hairy_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.full((450, 600, 3), 190, dtype=np.uint8) + rng.integers(0, 10, (450, 600, 3), dtype=np.uint8)
    img[200:260, 250:350] = (120, 70, 50)          # a "lesion"
    for x in range(50, 600, 80):                    # thin dark "hairs"
        img[:, x:x + 2] = (20, 20, 20)
    return img


class PreprocessTests(unittest.TestCase):
    def test_hair_mask_finds_thin_dark_lines(self):
        mask = hair_mask(_synthetic_hairy_image())
        self.assertGreater(mask[:, 50:52].mean(), 200)   # hair column flagged
        self.assertLess(mask[:, 10:40].mean(), 5)       # clean skin not flagged

    def test_output_is_224_rgb_and_hair_removed(self):
        img = _synthetic_hairy_image()
        out, frac = preprocess_rgb(img)
        self.assertEqual(out.shape, (*TARGET_SIZE[::-1], 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(0 < frac < 0.2)

    def test_clahe_preserves_shape_and_dtype(self):
        out = clahe_lab(_synthetic_hairy_image())
        self.assertEqual(out.shape, (450, 600, 3))
        self.assertEqual(out.dtype, np.uint8)


class ModelTests(unittest.TestCase):
    def test_every_arch_has_shared_head_and_feature_path(self):
        for arch in ARCHITECTURES:
            model = build_model(pretrained=False, arch=arch).eval()
            head = model.classifier
            self.assertIsInstance(head[2], torch.nn.Dropout, arch)
            self.assertAlmostEqual(head[2].p, 0.3)
            self.assertEqual(head[3].out_features, NUM_CLASSES)
            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(x), head(pooled_features(model, x)), atol=1e-5), arch)

    def test_arch_of_defaults_to_b0_for_v1_checkpoints(self):
        self.assertEqual(arch_of({"config": {"run_id": "v1"}}), "efficientnet_b0")
        self.assertEqual(arch_of({"config": {"arch": "resnet50"}}), "resnet50")

    def test_unknown_arch_raises(self):
        with self.assertRaises(ValueError):
            build_model(pretrained=False, arch="vgg")


class MCDropoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_model(pretrained=False).eval()

    def test_only_head_dropout_is_stochastic(self):
        active = enable_head_dropout(self.model)
        self.assertEqual(active, ["classifier.2"])

    def test_feature_reuse_matches_full_forward(self):
        self.assertLess(verify_feature_reuse(self.model, torch.randn(4, 3, 224, 224), T=3), 1e-4)

    def test_sample_head_shapes_and_variation(self):
        feats = torch.randn(5, 1280)
        mc, det = sample_head(self.model, feats, T=8, seed=1)
        self.assertEqual(tuple(mc.shape), (8, 5, NUM_CLASSES))
        self.assertEqual(tuple(det.shape), (5, NUM_CLASSES))
        self.assertGreater(float(mc.std(dim=0).mean()), 0)
        mc2, _ = sample_head(self.model, feats, T=8, seed=1)
        self.assertTrue(torch.equal(mc, mc2))        # seeded => reproducible
        self.assertFalse(self.model.classifier[2].training)  # left in eval mode

    def test_batchnorm_stats_untouched(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items() if "running_" in k}
        enable_head_dropout(self.model)
        with torch.no_grad():
            self.model(torch.randn(2, 3, 224, 224))
        after = self.model.state_dict()
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))


class UncertaintyTests(unittest.TestCase):
    def test_identical_passes_have_zero_epistemic(self):
        p = np.tile(np.array([[0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]]), (10, 4, 1))
        u = decompose(p)
        self.assertTrue(np.allclose(u["epistemic"], 0, atol=1e-9))
        self.assertTrue(np.allclose(u["total"], u["aleatoric"]))

    def test_disagreeing_passes_have_positive_epistemic(self):
        a = np.eye(7)[[0]] * 0.94 + 0.01
        b = np.eye(7)[[1]] * 0.94 + 0.01
        u = decompose(np.stack([a, b] * 5))
        self.assertGreater(u["epistemic"][0], 0.5)

    def test_referral_threshold_quantile(self):
        h = np.arange(100, dtype=float)
        self.assertEqual(referral_threshold(h, 0.0), 99.0)
        self.assertEqual(int((~accept_mask(h, referral_threshold(h, 0.3))).sum()), 30)

    def test_referral_table_r0_accepts_all(self):
        rng = np.random.default_rng(0)
        y = np.repeat(np.arange(7), 30)
        probs = rng.dirichlet(np.ones(7), size=len(y))
        probs[np.arange(len(y)), y] += 1
        probs /= probs.sum(1, keepdims=True)
        ent = -np.sum(probs * np.log(probs), axis=1)
        t = referral_table(y, probs, ent, rates=(0.0, 0.3), n_resamples=20)
        self.assertEqual(int(t["n_accepted"].iloc[0]), len(y))
        self.assertEqual(int(t["n_referred"].iloc[1]), round(0.3 * len(y)))


class GradCAMTests(unittest.TestCase):
    def test_iou_identical_and_disjoint(self):
        a = np.zeros((10, 10), bool)
        a[2:5, 2:5] = True
        b = np.zeros((10, 10), bool)
        b[6:9, 6:9] = True
        self.assertEqual(iou(a, a), 1.0)
        self.assertEqual(iou(a, b), 0.0)

    def test_binarize_at_half_max(self):
        cam = np.array([[0.1, 0.49], [0.5, 1.0]])
        self.assertEqual(binarize(cam).tolist(), [[False, False], [True, True]])

    def test_cam_shape_and_range(self):
        model = build_model(pretrained=False).eval()
        engine = GradCAM(model, model.features[-1])
        cams, logits, target = engine(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(cams.shape), (2, 7, 7))
        self.assertTrue(torch.equal(target, logits.argmax(1)))
        maps = upsample_normalize(cams)
        self.assertEqual(maps.shape, (2, 224, 224))
        self.assertTrue(maps.min() >= 0 and maps.max() <= 1 + 1e-6)

    def test_mask_alignment_is_binary_and_224(self):
        m = np.zeros((450, 600), np.uint8)
        m[100:300, 200:400] = 255
        out = align_mask(Image.fromarray(m))
        self.assertEqual(out.shape, (224, 224))
        self.assertEqual(set(np.unique(out).tolist()), {0, 1})
        self.assertAlmostEqual(out.mean(), m.mean() / 255, delta=0.01)


if __name__ == "__main__":
    unittest.main()

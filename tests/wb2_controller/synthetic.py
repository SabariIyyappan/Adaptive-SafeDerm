"""SYNTHETIC logit banks for unit tests only (implementation.md §0 rule 2).

Nothing here is model output. Scenario banks are hand-built so each one
forces a specific controller path; `make_random_bank` gives a label-
correlated bank with all 7 classes so the tuner and metrics can run end to
end. Numbers produced from these arrays must never appear in results/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.eda import CLASS_ORDER

C = len(CLASS_ORDER)
P = 50
MEL, NV, BKL, BCC = (CLASS_ORDER.index(c) for c in ("mel", "nv", "bkl", "bcc"))

# Thresholds the scenario expectations below were worked out against.
TEST_RISK_PARAMS = dict(tau_s=0.3, h_lo=0.25, h_hi=0.6, margin=0.05, delta_h=0.02)


def _const(**logit_by_class: float) -> np.ndarray:
    row = np.zeros(C)
    for cls, v in logit_by_class.items():
        row[CLASS_ORDER.index(cls)] = v
    return np.tile(row, (P, 1))


def scenario_logits(name: str) -> np.ndarray:
    """One image's P x C logits for a named scenario."""
    if name == "benign_stable":          # -> low tier, TRUSTED at 10
        return _const(nv=6.0)
    if name == "benign_medium":          # normalized entropy ~0.45 -> medium tier, TRUSTED at 10
        return _const(nv=4.0, bkl=3.0)
    if name == "mel_stable":             # -> high tier, TRUSTED at min_passes_high=20 + clinical attention
        return _const(mel=6.0)
    if name == "benign_masked_mel":      # pred nv but p_mel+p_bcc >= tau_s -> critical, NEEDS REVIEW at 50
        return _const(nv=2.0, mel=1.7)
    if name == "high_entropy":           # near-uniform -> high tier via entropy, NEEDS REVIEW at 50
        return _const(nv=0.5)
    if name == "flip":                   # alternating 5-pass blocks flip the argmax every checkpoint
        x = np.zeros((P, C))
        for b in range(P // 5):
            if b % 2 == 0:
                x[b * 5:(b + 1) * 5, MEL] = 6.0
            else:
                x[b * 5:(b + 1) * 5, NV] = 6.5
        return x
    if name == "nonfinite":              # NaN inside the first 5 passes -> NEEDS REVIEW at 5
        x = _const(nv=6.0)
        x[3, 2] = np.nan
        return x
    raise ValueError(name)


SCENARIOS = ("benign_stable", "benign_medium", "mel_stable", "benign_masked_mel", "high_entropy", "flip", "nonfinite")
SCENARIO_LABEL = {"benign_stable": NV, "benign_medium": NV, "mel_stable": MEL, "benign_masked_mel": MEL,
                  "high_entropy": NV, "flip": MEL, "nonfinite": NV}


def scenario_bank(names=SCENARIOS) -> tuple[np.ndarray, pd.DataFrame]:
    logits = np.stack([scenario_logits(n) for n in names]).astype(np.float32)
    ids = pd.DataFrame({
        "image_id": [f"SYN_{i:04d}" for i in range(len(names))],
        "lesion_id": [f"SYNL_{i:04d}" for i in range(len(names))],
        "y_true": [SCENARIO_LABEL[n] for n in names],
        "scenario": list(names),
    })
    return logits, ids


def make_random_bank(n: int = 400, seed: int = 0, passes: int = P) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Label-correlated random bank: returns (mc N x P x C, det N x C, ids)."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.arange(C), rng.choice(C, size=n - C, p=[.04, .06, .12, .02, .14, .58, .04])])
    signal = rng.uniform(0.0, 4.0, size=n)              # per-image difficulty
    base = rng.normal(0.0, 1.0, size=(n, C))
    base[np.arange(n), y] += signal
    noise_scale = rng.uniform(0.05, 1.0, size=(n, 1, 1))
    mc = base[:, None, :] + noise_scale * rng.normal(size=(n, passes, C))
    lesions = [f"SYNL_{i // 2:05d}" for i in range(n)]  # pairs of images share a lesion
    ids = pd.DataFrame({"image_id": [f"SYN_{i:05d}" for i in range(n)], "lesion_id": lesions, "y_true": y})
    return mc.astype(np.float32), base.astype(np.float32), ids


def synthetic_manifest(split: str, n: int) -> dict:
    return {
        "synthetic": True,
        "split": split,
        "status": "COMPLETE",
        "n_images": n,
        "n_passes": P,
        "class_order": list(CLASS_ORDER),
        "checkpoint_sha256": "SYNTHETIC-NOT-A-MODEL",
        "seed": 0,
        "stochastic_modules": ["classifier.2"],
        "feature_reuse": True,
    }

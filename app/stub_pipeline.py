"""STUB analyze() for building page 1 before the real pipeline lands.

Returns a FIXED, FAKE result in the implementation.md §2 contract shape so the
UI can be developed in parallel. None of these numbers come from the model.
Page 1 uses this only when `app/pipeline.py` (Soumya, A) cannot be imported,
and it shows a "STUB DATA" banner whenever it does.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.eval.classes import CLASS_ORDER

STUB_PROBS = {"akiec": 0.04, "bcc": 0.06, "bkl": 0.10, "df": 0.02, "mel": 0.18, "nv": 0.58, "vasc": 0.02}


def _fake_heatmap(size: int = 224) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
    cam = np.exp(-(((xx - 0.55) ** 2) + ((yy - 0.45) ** 2)) / 0.04)
    return (cam - cam.min()) / (cam.max() - cam.min())


def analyze(image: Image.Image) -> dict:
    probs = {c: STUB_PROBS[c] for c in CLASS_ORDER}
    trajectory = [
        {"passes": p, "top_prob": 0.55 + 0.03 * min(i, 1), "entropy": 1.30 - 0.02 * min(i, 1),
         "pred_class": "nv", "tier": "medium" if i == 0 else "critical"}
        for i, p in enumerate(range(10, 55, 5))
    ]
    return {
        "pred_class": "nv",
        "probs": probs,
        "suspicious_score": probs["mel"] + probs["bcc"],
        "tier": "critical",
        "decision": "NEEDS_REVIEW",
        "reasons": ["CRITICAL_TIER", "BUDGET_EXHAUSTED"],
        "clinical_attention": False,
        "passes_used": 50,
        "trajectory": trajectory,
        "heatmap": _fake_heatmap(),
        "latency_ms": {"backbone": 0.0, "mc_head": 0.0, "gradcam": 0.0, "total": 0.0},
        "checkpoint_sha256": "STUB-no-checkpoint",
        "mode": "stub",
    }

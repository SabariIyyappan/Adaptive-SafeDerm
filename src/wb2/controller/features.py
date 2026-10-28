"""Per-checkpoint statistics from a prefix of stochastic passes
(implementation.md §3 C2, §6 "stop-rule spec quality bar").

Everything is computed from cumulative sums over the pass axis, so the
statistics at checkpoint k depend **only on passes 0..k-1** — the offline
replay can never look ahead of where an online controller would be. That
causality is asserted in `tests/wb2_controller/test_features.py`.

Temperature is applied to each per-pass logit vector *before* softmax, then
the probabilities are averaged (the order recorded in WB2-I4). tau=1.0 means
uncalibrated.

Three distinct quantities, deliberately not interchangeable (§6):
  - output class stability: `flip` / `tie`;
  - uncertainty (entropy-estimate) stability: `delta_h`;
  - clinical-risk margin: `suspicious` (p_mel + p_bcc) vs a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.eda import CLASS_ORDER
from src.eval.classes import MALIGNANT, PRE_CANCEROUS

NUM_CLASSES = len(CLASS_ORDER)
MAX_PASSES = 50
STEP = 5
CHECKPOINTS = tuple(range(STEP, MAX_PASSES + 1, STEP))  # 5, 10, ..., 50

# Research screening group (implementation.md §3 C1, §6): mel + bcc are
# "suspicious"; akiec is pre-cancerous and reported separately as an
# attention score, never merged into the suspicious group.
SUSPICIOUS_IDX = tuple(CLASS_ORDER.index(c) for c in sorted(MALIGNANT))
ATTENTION_IDX = tuple(CLASS_ORDER.index(c) for c in sorted(PRE_CANCEROUS))
MEL_IDX = CLASS_ORDER.index("mel")

_LOG_K = np.log(NUM_CLASSES)
_TIE_ATOL = 1e-9


def softmax(logits: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over the last axis of logits / tau."""
    if not tau > 0:
        raise ValueError(f"temperature must be > 0, got {tau}")
    z = logits / tau
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=-1)


@dataclass(frozen=True)
class PrefixStats:
    """Statistics for N images at J checkpoints (arrays are N x J unless noted)."""

    checkpoints: tuple[int, ...]
    mean_probs: np.ndarray       # N x J x C — predictive mean over the first k passes
    pred: np.ndarray             # argmax of mean_probs (lowest index on ties)
    tie: np.ndarray              # top-2 mean probabilities equal
    entropy: np.ndarray          # total predictive entropy, normalized by ln C (0..1)
    expected_entropy: np.ndarray # mean per-pass entropy, normalized (aleatoric estimate)
    mutual_info: np.ndarray      # entropy - expected_entropy (epistemic estimate)
    mean_var: np.ndarray         # mean over classes of per-class probability variance
    suspicious: np.ndarray       # p_mel + p_bcc
    attention: np.ndarray        # p_akiec (reported, not part of the suspicious group)
    flip: np.ndarray             # pred changed since previous checkpoint (False at j=0)
    delta_p: np.ndarray          # L-inf change of mean_probs vs previous checkpoint (nan at j=0)
    delta_h: np.ndarray          # |entropy change| vs previous checkpoint (nan at j=0)
    delta_var: np.ndarray        # |mean_var change| vs previous checkpoint (nan at j=0)
    nonfinite: np.ndarray        # some pass inside the prefix had a non-finite logit

    @property
    def n_images(self) -> int:
        return self.pred.shape[0]


def prefix_stats(logits: np.ndarray, tau: float = 1.0, checkpoints: tuple[int, ...] = CHECKPOINTS) -> PrefixStats:
    """Compute PrefixStats from an N x P x C logit bank (P >= max(checkpoints))."""
    if logits.ndim != 3 or logits.shape[2] != NUM_CLASSES:
        raise ValueError(f"expected N x P x {NUM_CLASSES} logits, got shape {logits.shape}")
    n, p, _ = logits.shape
    if max(checkpoints) > p:
        raise ValueError(f"checkpoint {max(checkpoints)} exceeds the {p} available passes")

    bad_pass = ~np.isfinite(logits).all(axis=2)                 # N x P
    safe = np.where(np.isfinite(logits), logits, 0.0)
    probs = softmax(safe.astype(np.float64), tau)                # N x P x C
    pass_entropy = _entropy(probs) / _LOG_K                      # N x P

    cum_p = np.cumsum(probs, axis=1)
    cum_p2 = np.cumsum(probs ** 2, axis=1)
    cum_h = np.cumsum(pass_entropy, axis=1)
    cum_bad = np.cumsum(bad_pass, axis=1)

    idx = np.asarray(checkpoints) - 1
    k = np.asarray(checkpoints, dtype=float)[None, :, None]
    mean_probs = cum_p[:, idx, :] / k                            # N x J x C
    second = cum_p2[:, idx, :] / k
    var = np.clip(second - mean_probs ** 2, 0.0, None)

    entropy = _entropy(mean_probs) / _LOG_K
    expected_entropy = cum_h[:, idx] / k[..., 0]
    sorted_p = np.sort(mean_probs, axis=2)
    tie = np.isclose(sorted_p[..., -1], sorted_p[..., -2], atol=_TIE_ATOL, rtol=0.0)
    pred = mean_probs.argmax(axis=2)
    mean_var = var.mean(axis=2)

    flip = np.zeros_like(pred, dtype=bool)
    flip[:, 1:] = pred[:, 1:] != pred[:, :-1]
    nan_col = np.full((n, 1), np.nan)
    delta_p = np.concatenate([nan_col, np.abs(np.diff(mean_probs, axis=1)).max(axis=2)], axis=1)
    delta_h = np.concatenate([nan_col, np.abs(np.diff(entropy, axis=1))], axis=1)
    delta_var = np.concatenate([nan_col, np.abs(np.diff(mean_var, axis=1))], axis=1)

    return PrefixStats(
        checkpoints=tuple(checkpoints),
        mean_probs=mean_probs,
        pred=pred,
        tie=tie,
        entropy=entropy,
        expected_entropy=expected_entropy,
        mutual_info=np.clip(entropy - expected_entropy, 0.0, None),
        mean_var=mean_var,
        suspicious=mean_probs[..., list(SUSPICIOUS_IDX)].sum(axis=2),
        attention=mean_probs[..., list(ATTENTION_IDX)].sum(axis=2),
        flip=flip,
        delta_p=delta_p,
        delta_h=delta_h,
        delta_var=delta_var,
        nonfinite=cum_bad[:, idx] > 0,
    )

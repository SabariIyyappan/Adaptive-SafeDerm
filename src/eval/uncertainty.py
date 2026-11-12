"""Uncertainty decomposition and entropy-based referral (paper §3.4.3-3.4.4,
§3.7.3-3.7.4; PLAN.md §7.3 Members A and C).

Pure functions over C4 arrays (`probs_mc` shaped T×N×7):
  total     H[p̄]            entropy of the MC-mean prediction      (Eq. 8)
  aleatoric E_t H[p_t]       mean per-pass entropy                 (Eq. 9)
  epistemic total - aleatoric (mutual information, >= 0)           (Eq. 10)
All in nats.

Referral (§3.4.4): for rate r, tau = the (1-r)-quantile of TEST-set total
entropy and an image is accepted iff H <= tau — replicated exactly as the
paper states (PLAN.md §13 item 5 flags that tau comes from the test set).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import brier_score, expected_calibration_error, macro_metrics, predictive_entropy

REFERRAL_RATES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def decompose(probs_mc: np.ndarray) -> dict[str, np.ndarray]:
    """Per-image total / aleatoric / epistemic uncertainty from T×N×C passes."""
    mean = probs_mc.mean(axis=0)
    total = predictive_entropy(mean)
    aleatoric = np.mean([predictive_entropy(p) for p in probs_mc], axis=0)
    return {"mean_probs": mean, "total": total, "aleatoric": aleatoric, "epistemic": total - aleatoric}


def referral_threshold(entropy: np.ndarray, rate: float) -> float:
    """tau = (1 - r)-quantile of the entropy distribution."""
    return float(np.quantile(entropy, 1.0 - rate))


def accept_mask(entropy: np.ndarray, tau: float) -> np.ndarray:
    return entropy <= tau


def bootstrap_scalar(y_true: np.ndarray, probs: np.ndarray, fn, n_resamples: int = 1000, seed: int = 42,
                     require_all_classes: bool = False) -> tuple[float, float, float, int]:
    """Percentile 95% CI of any scalar fn(y, probs). Returns (point, lo, hi, n_used)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if require_all_classes and len(np.unique(y_true[idx])) < probs.shape[1]:
            continue
        vals.append(fn(y_true[idx], probs[idx]))
    vals = np.asarray(vals)
    return float(fn(y_true, probs)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


def ece_of(y, p) -> float:
    return expected_calibration_error(y, p)[0]


def auroc_of(y, p) -> float:
    return macro_metrics(y, p)["macro_auroc"]


def f1_of(y, p) -> float:
    return macro_metrics(y, p)["macro_f1"]


def referral_table(y_true: np.ndarray, probs: np.ndarray, entropy: np.ndarray,
                   rates=REFERRAL_RATES, tau_source: np.ndarray | None = None,
                   n_resamples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Paper Table 6: accepted-subset AUROC / F1 with bootstrap CIs per r.

    `tau_source` is the entropy array the quantile is taken from. Default is
    the evaluated (test) set itself, as the paper does; passing validation
    entropies gives the leakage-free variant.
    """
    tau_source = entropy if tau_source is None else tau_source
    base = macro_metrics(y_true, probs)
    rows = []
    for r in rates:
        tau = referral_threshold(tau_source, r)
        keep = accept_mask(entropy, tau)
        y, p = y_true[keep], probs[keep]
        auc, auc_lo, auc_hi, used = bootstrap_scalar(y, p, auroc_of, n_resamples, seed, require_all_classes=True)
        f1, f1_lo, f1_hi, _ = bootstrap_scalar(y, p, f1_of, n_resamples, seed, require_all_classes=True)
        rows.append({
            "referral_rate": r, "tau": tau, "n_accepted": int(keep.sum()), "n_referred": int((~keep).sum()),
            "actual_referral_rate": float((~keep).mean()),
            "macro_auroc": auc, "auroc_ci_low": auc_lo, "auroc_ci_high": auc_hi,
            "macro_f1": f1, "f1_ci_low": f1_lo, "f1_ci_high": f1_hi,
            "auroc_gain_pct": 100 * (auc - base["macro_auroc"]) / base["macro_auroc"],
            "f1_gain_pct": 100 * (f1 - base["macro_f1"]) / base["macro_f1"],
            "accepted_accuracy": float((p.argmax(1) == y).mean()),
            "bootstrap_resamples_used": used,
        })
    return pd.DataFrame(rows)


def budget_sweep(y_true: np.ndarray, probs_mc: np.ndarray, probs_det: np.ndarray,
                 budgets=(1, 5, 10, 20, 30, 50)) -> pd.DataFrame:
    """Metrics of the mean over the first T passes, plus the deterministic pass."""
    rows = [{"T": "deterministic (dropout off)", **_point_metrics(y_true, probs_det)}]
    for t in budgets:
        if t <= probs_mc.shape[0]:
            rows.append({"T": t, **_point_metrics(y_true, probs_mc[:t].mean(axis=0))})
    return pd.DataFrame(rows)


def _point_metrics(y, p) -> dict:
    m = macro_metrics(y, p)
    return {"macro_auroc": m["macro_auroc"], "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
            "ece_10bin": ece_of(y, p), "brier": brier_score(y, p),
            "mel_sensitivity": float(((p.argmax(1) == 4) & (y == 4)).sum() / max((y == 4).sum(), 1))}

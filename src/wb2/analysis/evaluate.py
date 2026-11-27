"""Per-method metrics over a labeled decisions frame (implementation.md §3 C5,
§6). Classification metrics reuse the V1 suite in `src.eval.metrics` so
WB2 numbers are computed by exactly the same code as V1's.

Safety accounting never mixes denominators (§3 C5, §5 E4):
  - mel_miss_all      = true mel, TRUSTED, predicted non-mel / all true mel
  - mel_miss_accepted = true mel, TRUSTED, predicted non-mel / true mel that were TRUSTED
Both are reported as raw numerator/denominator alongside the fraction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.eda import CLASS_ORDER
from src.eval.metrics import brier_score, expected_calibration_error, macro_metrics
from src.wb2.controller.features import MAX_PASSES, MEL_IDX, SUSPICIOUS_IDX
from src.wb2.controller.policies import PROB_COLUMNS, TIERS


def _frac(num: int, den: int) -> float:
    return num / den if den else float("nan")


def nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, None))))


def safety_counts(d: pd.DataFrame) -> dict:
    y, pred, accepted = d["y_true"].to_numpy(), d["pred"].to_numpy(), ~d["review"].to_numpy(bool)
    is_mel = y == MEL_IDX
    mel_accepted = is_mel & accepted
    mel_missed = mel_accepted & (pred != MEL_IDX)
    is_susp = np.isin(y, SUSPICIOUS_IDX)
    susp_reassured = is_susp & accepted & ~np.isin(pred, SUSPICIOUS_IDX)
    n_mel, n_mel_acc, n_missed = int(is_mel.sum()), int(mel_accepted.sum()), int(mel_missed.sum())
    return {
        "mel_total": n_mel,
        "mel_accepted": n_mel_acc,
        "mel_missed_accepted": n_missed,
        "mel_miss_all": _frac(n_missed, n_mel),
        "mel_miss_all_str": f"{n_missed}/{n_mel}",
        "mel_miss_accepted": _frac(n_missed, n_mel_acc),
        "mel_miss_accepted_str": f"{n_missed}/{n_mel_acc}",
        "suspicious_total": int(is_susp.sum()),
        "suspicious_false_reassurance": int(susp_reassured.sum()),
        "suspicious_false_reassurance_str": f"{int(susp_reassured.sum())}/{int(is_susp.sum())}",
    }


def method_summary(d: pd.DataFrame) -> dict:
    """All headline metrics for one method's decisions (one row per image)."""
    y = d["y_true"].to_numpy(int)
    probs = d[PROB_COLUMNS].to_numpy(float)
    pred = d["pred"].to_numpy(int)
    review = d["review"].to_numpy(bool)
    accepted = ~review
    passes = d["passes"].to_numpy(float)

    try:
        cls = macro_metrics(y, probs)
    except ValueError:  # a pilot subset missing a class: AUROC undefined, say so
        cls = {"macro_auroc": float("nan"), "macro_f1": float("nan"), "overall_accuracy": float((pred == y).mean()),
               "balanced_accuracy": float("nan"), "n": int(len(y))}
    ece, _ = expected_calibration_error(y, probs, n_bins=10)

    is_mel = y == MEL_IDX
    is_susp = np.isin(y, SUSPICIOUS_IDX)
    out = {
        "n": int(len(y)),
        "macro_auroc": cls["macro_auroc"],
        "macro_f1": cls["macro_f1"],
        "balanced_accuracy": cls["balanced_accuracy"],
        "accuracy": cls["overall_accuracy"],
        "ece_10bin": ece,
        "brier": brier_score(y, probs),
        "nll": nll(y, probs),
        "mel_recall": _frac(int((is_mel & (pred == MEL_IDX)).sum()), int(is_mel.sum())),
        "suspicious_recall": _frac(int((is_susp & np.isin(pred, SUSPICIOUS_IDX)).sum()), int(is_susp.sum())),
        "passes_mean": float(passes.mean()),
        "passes_median": float(np.median(passes)),
        "passes_p95": float(np.percentile(passes, 95)),
        "pct_at_cap": float((passes == MAX_PASSES).mean()),
        "review_rate": float(review.mean()),
        "coverage": float(accepted.mean()),
        "n_accepted": int(accepted.sum()),
        "accepted_accuracy": _frac(int((accepted & (pred == y)).sum()), int(accepted.sum())),
        "clinical_attention_rate": float(d["clinical_attention"].to_numpy(bool).mean()),
    }
    out.update(safety_counts(d))
    if "backbone_passes" in d:
        out["backbone_passes_mean"] = float(pd.to_numeric(d["backbone_passes"]).mean())
    return out


def comparison_table(labeled: pd.DataFrame, status: str) -> pd.DataFrame:
    """One row per method, in the order methods appear in the frame."""
    rows = []
    for method, d in labeled.groupby("method", sort=False):
        rows.append({"method": method, "status": status, **method_summary(d)})
    return pd.DataFrame(rows)


def stop_distribution(labeled: pd.DataFrame) -> pd.DataFrame:
    """Count of images stopping at each pass budget, per method."""
    tab = labeled.groupby(["method", "passes"], sort=False).size().unstack(fill_value=0)
    return tab.reset_index()


def tier_distribution(labeled: pd.DataFrame) -> pd.DataFrame:
    d = labeled[labeled["tier"] != ""]
    if d.empty:
        return pd.DataFrame(columns=["method", *TIERS])
    tab = d.groupby(["method", "tier"]).size().unstack(fill_value=0).reindex(columns=list(TIERS), fill_value=0)
    return tab.reset_index()


def class_counts(labeled: pd.DataFrame) -> dict[str, int]:
    first = labeled.drop_duplicates("image_id")
    counts = np.bincount(first["y_true"].to_numpy(int), minlength=len(CLASS_ORDER))
    return dict(zip(CLASS_ORDER, counts.tolist()))


def coverage_risk_curve(d: pd.DataFrame, score: str = "entropy", n_points: int = 41) -> pd.DataFrame:
    """Selective-prediction curve: refer the most uncertain images first.

    For coverage c, accept the (c * N) lowest-`score` images and report error
    rate and accepted-melanoma misses among them (denominator: all true mel).
    """
    y = d["y_true"].to_numpy(int)
    pred = d["pred"].to_numpy(int)
    order = np.argsort(d[score].to_numpy(float), kind="stable")
    n = len(y)
    n_mel = int((y == MEL_IDX).sum())
    rows = []
    for c in np.linspace(0.05, 1.0, n_points):
        m = max(1, int(round(c * n)))
        acc = order[:m]
        missed = int(((y[acc] == MEL_IDX) & (pred[acc] != MEL_IDX)).sum())
        rows.append({
            "coverage": m / n,
            "error_rate": float((pred[acc] != y[acc]).mean()),
            "mel_missed_accepted": missed,
            "mel_miss_all": _frac(missed, n_mel),
        })
    return pd.DataFrame(rows)


def melanoma_audit(labeled: pd.DataFrame, method: str) -> pd.DataFrame:
    """Every true-melanoma image under one method (§5 E4)."""
    d = labeled[(labeled["method"] == method) & (labeled["y_true"] == MEL_IDX)].copy()
    d["correct"] = d["pred"] == MEL_IDX
    cols = ["image_id", "lesion_id", "pred_dx", "correct", "decision", "review_reasons", "tier", "passes",
            "prob_4", "suspicious_score", "entropy"]
    return d[cols].rename(columns={"prob_4": "p_mel"}).sort_values(["correct", "decision", "p_mel"])


def confident_errors(labeled: pd.DataFrame, method: str, top: int = 20) -> pd.DataFrame:
    """Most confident wrong predictions under one method (§3 C6 failure cases)."""
    d = labeled[(labeled["method"] == method) & (labeled["pred"] != labeled["y_true"])].copy()
    d["confidence"] = d[PROB_COLUMNS].max(axis=1)
    d["true_dx"] = [CLASS_ORDER[i] for i in d["y_true"]]
    cols = ["image_id", "lesion_id", "true_dx", "pred_dx", "confidence", "decision", "review_reasons", "tier", "passes"]
    return d.sort_values("confidence", ascending=False)[cols].head(top)

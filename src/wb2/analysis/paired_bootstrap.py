"""Paired, lesion-grouped bootstrap CIs for method differences
(implementation.md §3 C6, §6 "Confidence intervals").

Resampling unit is the lesion, not the image: HAM10000 has several images
per lesion, so image resampling understates variance. Each resample draws
lesion_ids with replacement and takes *all* images of each drawn lesion,
then scores both methods on the same rows — a paired comparison.

V1's image-level bootstrap (`src.eval.bootstrap`, 1,000 resamples) is a
different estimator and stays labeled separately in every table.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.eval.metrics import NUM_CLASSES
from src.wb2.analysis.evaluate import method_summary

DEFAULT_N_RESAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_METRICS = ("macro_auroc", "macro_f1", "ece_10bin", "brier", "passes_mean", "review_rate", "mel_miss_all")


def lesion_resample_indices(lesion_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices for one lesion-grouped resample (all rows of each drawn lesion)."""
    codes, uniques = pd.factorize(lesion_ids)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    drawn = rng.integers(0, len(uniques), size=len(uniques))
    return np.concatenate([order[bounds[g]:bounds[g + 1]] for g in drawn])


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, metrics: tuple[str, ...] = DEFAULT_METRICS,
                     n_resamples: int = DEFAULT_N_RESAMPLES, seed: int = DEFAULT_SEED, ci_level: float = 95.0,
                     summarize: Callable[[pd.DataFrame], dict] = method_summary) -> pd.DataFrame:
    """CI of (metric(b) - metric(a)) over lesion-grouped resamples.

    `a` and `b` are one method's labeled decisions each, for the same images.
    Resamples missing a class are skipped and counted (AUROC undefined), as
    in the V1 bootstrap.
    """
    a = a.sort_values("image_id").reset_index(drop=True)
    b = b.sort_values("image_id").reset_index(drop=True)
    if not a["image_id"].equals(b["image_id"]):
        raise ValueError("paired bootstrap needs both methods on exactly the same images")

    rng = np.random.default_rng(seed)
    lesions = a["lesion_id"].to_numpy()
    y = a["y_true"].to_numpy()
    point_a, point_b = summarize(a), summarize(b)
    diffs = {m: [] for m in metrics}
    skipped = 0
    for _ in range(n_resamples):
        idx = lesion_resample_indices(lesions, rng)
        if len(np.unique(y[idx])) < NUM_CLASSES:
            skipped += 1
            continue
        sa, sb = summarize(a.iloc[idx]), summarize(b.iloc[idx])
        for m in metrics:
            diffs[m].append(sb[m] - sa[m])

    alpha = (100.0 - ci_level) / 2.0
    rows = []
    for m in metrics:
        arr = np.asarray(diffs[m], dtype=float)
        arr = arr[np.isfinite(arr)]
        lo, hi = (np.percentile(arr, alpha), np.percentile(arr, 100 - alpha)) if len(arr) else (np.nan, np.nan)
        rows.append({
            "metric": m,
            "method_a": a["method"].iloc[0],
            "method_b": b["method"].iloc[0],
            "a": point_a[m],
            "b": point_b[m],
            "diff_b_minus_a": point_b[m] - point_a[m],
            "ci_low": float(lo),
            "ci_high": float(hi),
            "ci_excludes_zero": bool(len(arr) and (lo > 0 or hi < 0)),
            "n_resamples_used": len(arr),
            "n_skipped_missing_class": skipped,
            "resampling_unit": "lesion_id",
            "seed": seed,
        })
    return pd.DataFrame(rows)

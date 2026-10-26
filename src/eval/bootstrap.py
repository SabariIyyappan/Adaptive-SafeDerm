"""Bootstrap confidence intervals (paper §3.7.6, PLAN.md §5.3 item 3).

1000 resamples of the test set with replacement, percentile 95% interval,
seed 42 (PLAN.md §4.5 — seed 42 everywhere).

A resample that happens to omit an entire class can't be scored one-vs-rest,
so it is **skipped and counted**; the count is reported in `metrics.json`
and `summary.md` rather than silently dropped, per §5.3 item 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.eval.metrics import NUM_CLASSES, macro_metrics

DEFAULT_N_RESAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_CI = 95.0


@dataclass
class BootstrapResult:
    point_estimate: dict[str, float]
    ci_low: dict[str, float] = field(default_factory=dict)
    ci_high: dict[str, float] = field(default_factory=dict)
    samples: dict[str, np.ndarray] = field(default_factory=dict)
    n_resamples_requested: int = DEFAULT_N_RESAMPLES
    n_resamples_used: int = 0
    n_skipped_missing_class: int = 0
    ci_level: float = DEFAULT_CI
    seed: int = DEFAULT_SEED

    def as_dict(self) -> dict:
        """JSON-serializable summary (drops the raw sample arrays)."""
        return {
            "ci_level": self.ci_level,
            "seed": self.seed,
            "n_resamples_requested": self.n_resamples_requested,
            "n_resamples_used": self.n_resamples_used,
            "n_skipped_missing_class": self.n_skipped_missing_class,
            "metrics": {
                name: {
                    "point": self.point_estimate[name],
                    "ci_low": self.ci_low.get(name),
                    "ci_high": self.ci_high.get(name),
                }
                for name in self.point_estimate
            },
        }

    def format_ci(self, name: str, digits: int = 4) -> str:
        """e.g. '0.9264 [0.9139, 0.9378]' — the paper's presentation style."""
        p = self.point_estimate[name]
        lo, hi = self.ci_low.get(name), self.ci_high.get(name)
        if lo is None or hi is None:
            return f"{p:.{digits}f}"
        return f"{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def bootstrap_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric_names: tuple[str, ...] = ("macro_auroc", "macro_f1"),
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    ci_level: float = DEFAULT_CI,
) -> BootstrapResult:
    """Percentile bootstrap CI over resamples of the evaluation set.

    Resampling is at the **image** level, matching the paper's description
    ("1000 resamples of the test set"). Note this treats images as
    independent even though some share a lesion; the split guarantees a
    lesion never crosses splits, so within the test set this is the
    paper's own convention and we follow it.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point = macro_metrics(y_true, probs)
    collected: dict[str, list[float]] = {name: [] for name in metric_names}
    skipped = 0

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_true = y_true[idx]
        # One-vs-rest AUROC is undefined for a class with no positives (or
        # no negatives) in the resample — skip the whole resample so every
        # reported metric comes from the same set of resamples.
        if len(np.unique(resampled_true)) < NUM_CLASSES:
            skipped += 1
            continue
        m = macro_metrics(resampled_true, probs[idx])
        for name in metric_names:
            collected[name].append(m[name])

    alpha = (100.0 - ci_level) / 2.0
    result = BootstrapResult(
        point_estimate={name: point[name] for name in metric_names},
        n_resamples_requested=n_resamples,
        n_resamples_used=n_resamples - skipped,
        n_skipped_missing_class=skipped,
        ci_level=ci_level,
        seed=seed,
    )
    for name in metric_names:
        arr = np.asarray(collected[name], dtype=float)
        result.samples[name] = arr
        if len(arr):
            result.ci_low[name] = float(np.percentile(arr, alpha))
            result.ci_high[name] = float(np.percentile(arr, 100.0 - alpha))
    return result

"""WB2 controller figures (implementation.md §3 C5). Same house style as the
V1 suite (`src.eval.style`): takeaway title, caption under every figure,
direct labels, solid hairline grids. Every title carries the run status
(MEASURED / PILOT ONLY / SYNTHETIC) so no figure can be mistaken for
another kind of evidence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.classes import CHART_INK, MALIGNANCY_COLOR, SERIES_2, SINGLE_SERIES_BLUE
from src.eval.metrics import expected_calibration_error
from src.eval.style import annotate_bars, finish_figure, grid_xy, grid_y, new_figure
from src.wb2.controller.policies import PROB_COLUMNS, TIERS

TIER_COLOR = {
    "low": MALIGNANCY_COLOR["benign"],
    "medium": MALIGNANCY_COLOR["pre-cancerous"],
    "high": "#eb6834",
    "critical": MALIGNANCY_COLOR["malignant"],
}
METHOD_LABEL = {
    "det_T1": "Deterministic T=1",
    "fixed_T10": "Fixed T=10",
    "fixed_T20": "Fixed T=20",
    "fixed_T50": "Fixed T=50",
    "fixed_T50+referral": "Fixed T=50 + entropy referral",
    "variance_adaptive": "Variance-only adaptive",
    "risk_aware_adaptive": "Risk-aware adaptive (proposed)",
}
ADAPTIVE = ("variance_adaptive", "risk_aware_adaptive")


def _label(m: str) -> str:
    return METHOD_LABEL.get(m, m)


def plot_tier_distribution(labeled: pd.DataFrame, out_path: Path, status: str, split: str) -> Path:
    d = labeled[labeled["method"] == "risk_aware_adaptive"]
    counts = [int((d["tier"] == t).sum()) for t in TIERS]
    reviewed = [int(((d["tier"] == t) & d["review"]).sum()) for t in TIERS]
    fig, ax = new_figure(figsize=(6, 3.6))
    bars = ax.bar(TIERS, counts, color=[TIER_COLOR[t] for t in TIERS])
    annotate_bars(ax, bars, [f"{c} ({r} review)" for c, r in zip(counts, reviewed)], fmt="{}")
    grid_y(ax)
    ax.set_ylabel("Images")
    ax.set_title(f"Risk-aware controller: final model-risk tier — {split}, {status}", fontsize=9)
    return finish_figure(fig, out_path, caption=(
        "Tier is a model risk/attention tier, not a pathology severity rating. "
        f"n={len(d)}; counts in parentheses ended NEEDS REVIEW."))


def plot_stop_distribution(labeled: pd.DataFrame, out_path: Path, status: str, split: str) -> Path:
    fig, ax = new_figure(figsize=(7, 3.6))
    budgets = np.arange(5, 55, 5)
    width = 1.8
    for offset, (m, color) in zip((-width / 2, width / 2), zip(ADAPTIVE, (SERIES_2, SINGLE_SERIES_BLUE))):
        d = labeled[labeled["method"] == m]
        counts = [int((d["passes"] == b).sum()) for b in budgets]
        ax.bar(budgets + offset, counts, width=width, color=color, label=f"{_label(m)} (mean {d['passes'].mean():.1f})")
    grid_y(ax)
    ax.set_xticks(budgets)
    ax.set_xlabel("Stochastic head passes used at stop")
    ax.set_ylabel("Images")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"Where the adaptive policies stop — {split}, {status}", fontsize=9)
    return finish_figure(fig, out_path, caption=(
        "Offline prefix replay of the same saved 50-draw bank; pass savings are algorithmic, not wall-clock."))


def plot_compute_vs_risk(table: pd.DataFrame, out_path: Path, status: str, split: str) -> Path:
    fig, axes = new_figure(1, 2, figsize=(9, 3.8))
    for ax, col, ylabel in (
        (axes[0], "accepted_error", "Error rate among TRUSTED results"),
        (axes[1], "mel_miss_all", "Accepted melanoma misses / all melanoma"),
    ):
        t = table.assign(accepted_error=1 - table["accepted_accuracy"])
        for _, r in t.iterrows():
            adaptive = r["method"] in ADAPTIVE
            ax.scatter(r["passes_mean"], r[col], s=40, zorder=3,
                       color=SINGLE_SERIES_BLUE if r["method"] == "risk_aware_adaptive" else (SERIES_2 if adaptive else CHART_INK["muted"]))
            ax.annotate(_label(r["method"]), (r["passes_mean"], r[col]), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points", color=CHART_INK["primary"])
        grid_xy(ax)
        ax.set_xlabel("Mean stochastic head passes per image")
        ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle(f"Compute vs risk — {split}, {status}", fontsize=9)
    return finish_figure(fig, out_path, caption=(
        "Methods without referral accept every image. Melanoma numerator/denominator are in the comparison table."))


def plot_coverage_vs_risk(curve: pd.DataFrame, table: pd.DataFrame, out_path: Path, status: str, split: str) -> Path:
    fig, axes = new_figure(1, 2, figsize=(9, 3.8))
    t = table.assign(accepted_error=1 - table["accepted_accuracy"])
    for ax, ccol, tcol, ylabel in (
        (axes[0], "error_rate", "accepted_error", "Error rate among accepted"),
        (axes[1], "mel_miss_all", "mel_miss_all", "Accepted melanoma misses / all melanoma"),
    ):
        ax.plot(curve["coverage"], curve[ccol], color=CHART_INK["muted"], linewidth=1.5,
                label="Fixed T=50, refer highest entropy first")
        for m, color in (("fixed_T50+referral", CHART_INK["secondary"]), ("variance_adaptive", SERIES_2),
                         ("risk_aware_adaptive", SINGLE_SERIES_BLUE)):
            r = t[t["method"] == m]
            if len(r):
                ax.scatter(r["coverage"], r[tcol], color=color, s=45, zorder=3, label=_label(m))
        grid_xy(ax)
        ax.set_xlabel("Accepted coverage (1 - review rate)")
        ax.set_ylabel(ylabel, fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"Coverage vs risk at matched review budgets — {split}, {status}", fontsize=9)
    return finish_figure(fig, out_path, caption=(
        "Points below the grey curve accept the same share of images with less risk than entropy referral on fixed T=50."))


def plot_reliability(labeled: pd.DataFrame, out_path: Path, status: str, split: str,
                     methods: tuple[str, ...] = ("det_T1", "fixed_T50", "risk_aware_adaptive")) -> Path:
    fig, ax = new_figure(figsize=(5, 4.2))
    ax.plot([0, 1], [0, 1], color=CHART_INK["gridline"], linewidth=1)
    for m, color in zip(methods, (CHART_INK["muted"], SERIES_2, SINGLE_SERIES_BLUE)):
        d = labeled[labeled["method"] == m]
        if d.empty:
            continue
        ece, bins = expected_calibration_error(d["y_true"].to_numpy(int), d[PROB_COLUMNS].to_numpy(float))
        b = bins[bins["count"] > 0]
        ax.plot(b["confidence"], b["accuracy"], marker="o", markersize=3, color=color, label=f"{_label(m)} (ECE {ece:.3f})")
    grid_xy(ax)
    ax.set_xlabel("Mean confidence (max probability)")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"Reliability, 10 equal-width bins — {split}, {status}", fontsize=9)
    return finish_figure(fig, out_path, caption="Uncalibrated unless the run manifest names a calibration_id.")

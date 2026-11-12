"""Shared evaluation pieces for the V2-V5 result packages.

The V1 figure module (`src/eval/figures.py`) bakes V1-specific findings into
its captions ("best epoch is the last epoch", "single pass"), so later
versions use these generic plots whose titles and captions are computed
from the data they show. Metrics, bootstrap and table helpers are reused
unchanged from the V1 suite.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda import CLASS_ORDER
from src.eval.bootstrap import BootstrapResult, bootstrap_metrics
from src.eval.classes import CHART_INK, NUM_CLASSES, SEQUENTIAL_CMAP, SERIES_2, SINGLE_SERIES_BLUE, full_names
from src.eval.metrics import (
    brier_score,
    confusion_matrices,
    expected_calibration_error,
    load_predictions,
    macro_metrics,
    per_class_table,
    probs_and_labels,
    roc_curves,
)
from src.eval.style import annotate_bars, finish_figure, grid_xy, grid_y, new_figure, wrap_class_labels

MEL = CLASS_ORDER.index("mel")
DF = CLASS_ORDER.index("df")
NV = CLASS_ORDER.index("nv")


def json_default(obj):
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


@dataclass
class Evaluation:
    """Everything one prediction set yields, computed once."""

    y_true: np.ndarray
    probs: np.ndarray
    metrics: dict
    per_class: pd.DataFrame
    ece: float
    brier: float
    boot: BootstrapResult

    def headline(self) -> dict:
        pc = self.per_class.set_index("dx")
        return {
            "macro_auroc": self.metrics["macro_auroc"],
            "macro_auroc_ci": [self.boot.ci_low.get("macro_auroc"), self.boot.ci_high.get("macro_auroc")],
            "macro_f1": self.metrics["macro_f1"],
            "macro_f1_ci": [self.boot.ci_low.get("macro_f1"), self.boot.ci_high.get("macro_f1")],
            "balanced_accuracy": self.metrics["balanced_accuracy"],
            "overall_accuracy": self.metrics["overall_accuracy"],
            "ece_10bin": self.ece,
            "brier": self.brier,
            "mel_sensitivity": float(pc.loc["mel", "recall"]),
            "mel_specificity": float(pc.loc["mel", "specificity"]),
            "df_sensitivity": float(pc.loc["df", "recall"]),
            "nv_accuracy_eq17": float(pc.loc["nv", "accuracy_eq17"]),
            "nv_recall": float(pc.loc["nv", "recall"]),
            "n": self.metrics["n"],
        }


def evaluate(y_true: np.ndarray, probs: np.ndarray, n_resamples: int = 1000, seed: int = 42) -> Evaluation:
    ece, _ = expected_calibration_error(y_true, probs)
    return Evaluation(
        y_true=y_true,
        probs=probs,
        metrics=macro_metrics(y_true, probs),
        per_class=per_class_table(y_true, probs),
        ece=ece,
        brier=brier_score(y_true, probs),
        boot=bootstrap_metrics(y_true, probs, n_resamples=n_resamples, seed=seed),
    )


def evaluate_file(path: Path, n_resamples: int = 1000, seed: int = 42) -> Evaluation:
    probs, y_true = probs_and_labels(load_predictions(path))
    return evaluate(y_true, probs, n_resamples=n_resamples, seed=seed)


def fmt_ci(point: float, lo: float | None, hi: float | None, digits: int = 4) -> str:
    if lo is None or hi is None:
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ------------------------------------------------------------------ figures
def plot_training_curves(history: pd.DataFrame, best_epoch: int, out_path: Path, run_label: str) -> Path:
    fig, (ax_loss, ax_metric) = new_figure(1, 2, figsize=(11, 4.2))
    ax_loss.plot(history["epoch"], history["train_loss"], color=SINGLE_SERIES_BLUE, linewidth=2, label="Train loss")
    ax_loss.plot(history["epoch"], history["val_loss"], color=SERIES_2, linewidth=2, label="Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title(f"{run_label}: training and validation loss", fontsize=10, color=CHART_INK["primary"])
    ax_loss.legend(frameon=False, fontsize=8)
    grid_y(ax_loss)

    ax_metric.plot(history["epoch"], history["val_macro_f1"], color=SINGLE_SERIES_BLUE, linewidth=2, label="Val macro F1")
    ax_metric.plot(history["epoch"], history["val_macro_auroc"], color=SERIES_2, linewidth=2, label="Val macro AUROC")
    ax_metric.axvline(best_epoch, color=CHART_INK["muted"], linewidth=1)
    best = history.loc[history["epoch"] == best_epoch].iloc[0]
    ax_metric.annotate(f"best epoch {best_epoch}\nF1 {best['val_macro_f1']:.4f}", (best_epoch, best["val_macro_f1"]),
                       textcoords="offset points", xytext=(-60 if best_epoch > len(history) / 2 else 10, -30),
                       fontsize=7.5, color=CHART_INK["primary"],
                       arrowprops=dict(arrowstyle="-", color=CHART_INK["muted"], linewidth=0.8))
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score")
    ax_metric.set_ylim(0, 1)
    ax_metric.set_title("Validation metrics (checkpoint selected on val macro F1)", fontsize=10, color=CHART_INK["primary"])
    ax_metric.legend(frameon=False, fontsize=8, loc="lower right")
    grid_y(ax_metric)

    first5 = history.loc[history["epoch"] <= 5, "train_loss"]
    caption = (
        f"Train loss {history['train_loss'].iloc[0]:.4f} at epoch 1 -> {first5.iloc[-1]:.4f} at epoch {int(len(first5))} "
        f"(paper: 0.8728 -> 0.1483). Validation loss range {history['val_loss'].min():.3f}-{history['val_loss'].max():.3f} "
        f"(paper: 0.53-0.66). Best epoch {best_epoch}/{len(history)} (paper: 26)."
    )
    return finish_figure(fig, out_path, caption)


def plot_confusion_matrix(y_true: np.ndarray, probs: np.ndarray, out_path: Path, title: str) -> Path:
    counts, normalized = confusion_matrices(y_true, probs)
    labels = wrap_class_labels(full_names(), width=11)
    fig, axes = new_figure(1, 2, figsize=(14, 6))
    for ax, matrix, sub, is_float, vmax in (
        (axes[0], counts, "image counts", False, counts.max()),
        (axes[1], normalized, "row-normalized (recall per true class)", True, 1.0),
    ):
        im = ax.imshow(matrix, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=vmax)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{title} — {sub}", fontsize=10, color=CHART_INK["primary"])
        ax.grid(False)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.2f}" if is_float else f"{int(val)}", ha="center", va="center", fontsize=7,
                        color="white" if val > vmax * 0.55 else CHART_INK["primary"])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    off = counts.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    caption = (f"Largest off-diagonal cell: true {CLASS_ORDER[i]} predicted as {CLASS_ORDER[j]} ({int(off[i, j])} images). "
               f"Diagonal recall ranges {np.diag(normalized).min():.2f}-{np.diag(normalized).max():.2f}.")
    return finish_figure(fig, out_path, caption)


def plot_per_class(per_class: pd.DataFrame, out_path: Path, title: str) -> Path:
    x = np.arange(NUM_CLASSES)
    width = 0.27
    fig, ax = new_figure(figsize=(12, 4.8))
    series = (("accuracy_eq17", "Accuracy (Eq. 17)", SINGLE_SERIES_BLUE),
              ("recall", "Sensitivity / recall", SERIES_2),
              ("specificity", "Specificity", CHART_INK["muted"]))
    for k, (col, label, color) in enumerate(series):
        bars = ax.bar(x + (k - 1) * width, per_class[col], width, color=color, label=label)
        annotate_bars(ax, bars, per_class[col], fmt="{:.2f}", fontsize=6)
    ax.axhline(0.70, color=CHART_INK["secondary"], linewidth=1)
    ax.annotate("70%", (NUM_CLASSES - 0.5, 0.70), fontsize=7, color=CHART_INK["secondary"], va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_class_labels(full_names(), width=11), fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=10, color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3)
    grid_y(ax)
    n_over = int((per_class["accuracy_eq17"] >= 0.70).sum())
    mel = per_class.set_index("dx").loc["mel"]
    caption = (f"{n_over}/7 classes have Eq. 17 accuracy >= 70%. Melanoma sensitivity {mel['recall']:.3f}, "
               f"specificity {mel['specificity']:.3f} (paper: 0.808 / 0.962).")
    return finish_figure(fig, out_path, caption)


def plot_roc(y_true: np.ndarray, probs: np.ndarray, out_path: Path, title: str) -> Path:
    from src.eval.classes import CLASS_COLOR

    curves = roc_curves(y_true, probs)
    fig, ax = new_figure(figsize=(7, 6))
    for cls in CLASS_ORDER:
        c = curves[cls]
        ax.plot(c["fpr"], c["tpr"], color=CLASS_COLOR[cls], linewidth=1.6, label=f"{cls}  AUC {c['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], color=CHART_INK["muted"], linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=10, color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    grid_xy(ax)
    worst = min(CLASS_ORDER, key=lambda k: curves[k]["auroc"])
    caption = f"One-vs-rest ROC per class. Lowest per-class AUC: {worst} ({curves[worst]['auroc']:.3f})."
    return finish_figure(fig, out_path, caption)


def plot_reliability(y_true: np.ndarray, probs_by_label: dict[str, np.ndarray], out_path: Path, title: str,
                     caption: str) -> Path:
    """Reliability curves for one or more prediction sets on the same axes."""
    colors = [SINGLE_SERIES_BLUE, SERIES_2, CHART_INK["secondary"]]
    fig, axes = new_figure(2, 1, figsize=(7, 7.4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax, ax_hist = axes
    ax.plot([0, 1], [0, 1], color=CHART_INK["muted"], linewidth=1, label="Perfect calibration")
    width = 0.1 / (len(probs_by_label) + 0.5)
    for k, (label, probs) in enumerate(probs_by_label.items()):
        ece, bins = expected_calibration_error(y_true, probs)
        valid = bins["count"] > 0
        ax.plot(bins.loc[valid, "confidence"], bins.loc[valid, "accuracy"], marker="o", markersize=5,
                color=colors[k], linewidth=2, label=f"{label}  ECE {ece:.4f}")
        centers = (bins["bin_lo"] + bins["bin_hi"]) / 2
        ax_hist.bar(centers + (k - (len(probs_by_label) - 1) / 2) * width, bins["count"], width=width, color=colors[k])
    ax.set_ylabel("Observed accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=10, color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    grid_xy(ax)
    ax_hist.set_xlabel("Predicted confidence (max probability), 10 equal-width bins")
    ax_hist.set_ylabel("Images")
    ax_hist.set_xlim(0, 1)
    grid_y(ax_hist)
    return finish_figure(fig, out_path, caption)


def run_package(ev: Evaluation, history: pd.DataFrame, best_epoch: int, out_dir: Path, run_label: str,
                prefix: str = "") -> list[Path]:
    """Standard per-model figure set (paper Figs 2, 4, 5 + ROC)."""
    fdir = out_dir / "figures"
    return [
        plot_training_curves(history, best_epoch, fdir / f"{prefix}training_curves.png", run_label),
        plot_confusion_matrix(ev.y_true, ev.probs, fdir / f"{prefix}confusion_matrix.png", run_label),
        plot_per_class(ev.per_class, fdir / f"{prefix}per_class.png", f"{run_label}: per-class accuracy, sensitivity, specificity"),
        plot_roc(ev.y_true, ev.probs, fdir / f"{prefix}roc_curves.png", f"{run_label}: one-vs-rest ROC"),
    ]

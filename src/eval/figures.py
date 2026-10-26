"""Evaluation figures (PLAN.md §5.3 item 4, plus the diagnostic figures the
V1 result package needs to explain its own numbers).

Core, required by the plan:
  1. training_curves.png         (paper Fig. 2 style)
  2. confusion_matrix.png        (paper Fig. 5 style)
  3. per_class_accuracy.png      (paper Fig. 4 style)
  4. per_class_sens_spec.png
  5. roc_curves.png

Diagnostic depth:
  6. pr_curves.png               — more honest than ROC under 58:1 imbalance
  7. error_flow.png              — where the mel over-predictions come from
  8. confidence_distributions.png
  9. reliability_diagram.png     — single-pass (T=1) calibration; V3 adds T=50
 10. per_class_confidence_ecdf.png
 11. bootstrap_distributions.png
 12. val_test_agreement.png
 13. ours_vs_paper.png

Design rules (the plan's Step 2) are enforced here: full diagnostic class
names everywhere, a takeaway title on every figure, a caption line, values
direct-labeled, one blue for nominal bar charts, the 7-slot categorical
order only where color carries identity, no dual-axis anywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda import CLASS_ORDER
from src.eval.classes import (
    CHART_INK,
    CLASS_COLOR,
    CLASS_FULL_NAME,
    MALIGNANCY_COLOR,
    NUM_CLASSES,
    SEQUENTIAL_CMAP,
    SERIES_2,
    SINGLE_SERIES_BLUE,
    full_names,
    malignancy_of,
)
from src.eval.metrics import (
    confusion_matrices,
    expected_calibration_error,
    pr_curves,
    predictive_entropy,
    roc_curves,
)
from src.eval.style import annotate_bars, finish_figure, grid_xy, grid_y, new_figure, wrap_class_labels

# Paper reference values (PLAN.md §2) used as comparison marks on figures.
PAPER_TEST_AUROC = 0.9404
PAPER_TEST_F1 = 0.7308
PAPER_MEL_SENSITIVITY = 0.808
PAPER_MEL_SPECIFICITY = 0.962
PAPER_NV_ACCURACY = 0.698
PAPER_BEST_EPOCH = 26


# ---------------------------------------------------------------- 1. training
def plot_training_curves(history: pd.DataFrame, out_path: Path, best_epoch: int) -> Path:
    """Paper Fig. 2 equivalent: loss panel + validation-metric panel.

    Deliberately two panels rather than one dual-axis plot (loss and the
    0-1 metrics have different scales; a dual axis would invent a
    correlation). The best-epoch marker sits at the right edge here
    because our best epoch is the last one — annotated explicitly so it
    doesn't read as a plotting bug.
    """
    fig, axes = new_figure(1, 2, figsize=(11, 4.2))
    ax_loss, ax_metric = axes

    ax_loss.plot(history["epoch"], history["train_loss"], color=SINGLE_SERIES_BLUE, linewidth=2, label="Train loss")
    ax_loss.plot(history["epoch"], history["val_loss"], color=SERIES_2, linewidth=2, label="Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Weighted cross-entropy loss")
    ax_loss.set_title("Loss: train falls fast, validation plateaus", fontsize=10, color=CHART_INK["primary"])
    ax_loss.legend(frameon=False, fontsize=8)
    grid_y(ax_loss)

    ax_metric.plot(history["epoch"], history["val_macro_f1"], color=SINGLE_SERIES_BLUE, linewidth=2, label="Val macro F1")
    ax_metric.plot(history["epoch"], history["val_macro_auroc"], color=SERIES_2, linewidth=2, label="Val macro AUROC")
    ax_metric.axvline(best_epoch, color=CHART_INK["muted"], linewidth=1)
    best_row = history.loc[history["epoch"] == best_epoch].iloc[0]
    ax_metric.annotate(
        f"best epoch {best_epoch}\nF1 {best_row['val_macro_f1']:.4f}",
        (best_epoch, best_row["val_macro_f1"]),
        textcoords="offset points", xytext=(-58, -28), fontsize=7.5,
        color=CHART_INK["primary"],
        arrowprops=dict(arrowstyle="-", color=CHART_INK["muted"], linewidth=0.8),
    )
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score")
    ax_metric.set_ylim(0, 1)
    ax_metric.set_title("Validation metrics still rising at epoch 30", fontsize=10, color=CHART_INK["primary"])
    ax_metric.legend(frameon=False, fontsize=8, loc="lower right")
    grid_y(ax_metric)

    caption = (
        f"Best epoch is {best_epoch}/{len(history)} — the LAST epoch, so the marker sits at the right edge: the model was "
        f"still improving when training stopped (the paper peaked at epoch {PAPER_BEST_EPOCH}). "
        "Train and validation loss are not directly comparable: train loss is computed on the class-balanced "
        "resampled stream, validation loss on the natural distribution."
    )
    return finish_figure(fig, out_path, caption)


# ------------------------------------------------------- 2. confusion matrix
def plot_confusion_matrix(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> Path:
    """Counts and row-normalized side by side, one-hue sequential ramp."""
    counts, normalized = confusion_matrices(y_true, probs)
    labels = wrap_class_labels(full_names(), width=11)

    fig, axes = new_figure(1, 2, figsize=(14, 6))
    for ax, matrix, title, fmt, vmax in (
        (axes[0], counts, "Confusion matrix — image counts", "{:d}", counts.max()),
        (axes[1], normalized, "Row-normalized — recall per true class", "{:.2f}", 1.0),
    ):
        im = ax.imshow(matrix, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=vmax)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title, fontsize=10, color=CHART_INK["primary"])
        ax.grid(False)
        # Annotate every cell; flip ink to white on dark cells so nothing is unreadable.
        threshold = vmax * 0.55
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = matrix[i, j]
                ax.text(
                    j, i, fmt.format(val if fmt.endswith("f}") else int(val)),
                    ha="center", va="center", fontsize=7,
                    color="white" if val > threshold else CHART_INK["primary"],
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_nv_to_mel = int(counts[CLASS_ORDER.index("nv"), CLASS_ORDER.index("mel")])
    caption = (
        f"The diagonal holds, so there is no collapse into the majority class. The largest single off-diagonal cell is "
        f"true nv predicted as mel ({n_nv_to_mel} images) — the dual imbalance correction pushing the model off nv, "
        "which is the clinically safe direction (over-calling melanoma rather than missing it)."
    )
    return finish_figure(fig, out_path, caption)


# ----------------------------------------------------- 3. per-class accuracy
def plot_per_class_accuracy(per_class: pd.DataFrame, out_path: Path) -> Path:
    """Paper Fig. 4 style, with BOTH accuracy definitions (PLAN.md §13 item 4).

    Two series (Eq. 17 vs recall), so slots 1-2 rather than a 7-hue ramp —
    the categories are nominal and the bar length already encodes the value.
    """
    x = np.arange(NUM_CLASSES)
    width = 0.38
    fig, ax = new_figure(figsize=(11, 5))

    bars_eq17 = ax.bar(x - width / 2, per_class["accuracy_eq17"], width,
                       color=SINGLE_SERIES_BLUE, label="Accuracy, Eq. 17  (TP+TN)/N")
    bars_recall = ax.bar(x + width / 2, per_class["recall"], width,
                         color=SERIES_2, label="Recall / sensitivity  TP/(TP+FN)")
    annotate_bars(ax, bars_eq17, per_class["accuracy_eq17"], fmt="{:.3f}", fontsize=6.5)
    annotate_bars(ax, bars_recall, per_class["recall"], fmt="{:.3f}", fontsize=6.5)

    ax.axhline(0.70, color=CHART_INK["muted"], linewidth=1)
    # Sits just above the line at the left edge, where the tallest bars aren't,
    # so it never overlaps a bar or its value label.
    ax.annotate("70% reference line (paper Fig. 4)", (-0.42, 0.712),
                ha="left", fontsize=7, color=CHART_INK["secondary"])

    ax.set_xticks(x)
    ax.set_xticklabels(wrap_class_labels(full_names(), width=12), fontsize=7.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Per-class accuracy: Eq. 17 vs recall — the paper's 69.8% for nv matches Eq. 17, not recall",
        fontsize=10, color=CHART_INK["primary"],
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    grid_y(ax)

    nv = per_class.loc[per_class["dx"] == "nv"].iloc[0]
    caption = (
        f"Resolves PLAN.md §13 item 4: for nv, Eq. 17 gives {nv['accuracy_eq17']:.3f} and recall {nv['recall']:.3f}. "
        f"The paper reports {PAPER_NV_ACCURACY:.3f} for nv, so its \"per-class accuracy\" is Eq. 17's one-vs-rest form. "
        "Eq. 17 is inflated for rare classes by the large true-negative pool, which is why recall is shown beside it."
    )
    return finish_figure(fig, out_path, caption)


# ------------------------------------------------- 4. sensitivity/specificity
def plot_per_class_sens_spec(per_class: pd.DataFrame, out_path: Path) -> Path:
    x = np.arange(NUM_CLASSES)
    width = 0.38
    fig, ax = new_figure(figsize=(11, 5))

    bars_sens = ax.bar(x - width / 2, per_class["recall"], width, color=SINGLE_SERIES_BLUE, label="Sensitivity (recall)")
    bars_spec = ax.bar(x + width / 2, per_class["specificity"], width, color=SERIES_2, label="Specificity")
    annotate_bars(ax, bars_sens, per_class["recall"], fmt="{:.3f}", fontsize=6.5)
    annotate_bars(ax, bars_spec, per_class["specificity"], fmt="{:.3f}", fontsize=6.5)

    # Mark the paper's melanoma reference values — the clinically critical class.
    mel_idx = CLASS_ORDER.index("mel")
    ax.scatter([mel_idx - width / 2], [PAPER_MEL_SENSITIVITY], marker="_", s=420,
               color=CHART_INK["primary"], zorder=5, linewidths=1.8)
    ax.scatter([mel_idx + width / 2], [PAPER_MEL_SPECIFICITY], marker="_", s=420,
               color=CHART_INK["primary"], zorder=5, linewidths=1.8)
    ax.annotate(
        f"paper: mel sens {PAPER_MEL_SENSITIVITY:.3f} / spec {PAPER_MEL_SPECIFICITY:.3f}",
        (mel_idx, PAPER_MEL_SPECIFICITY + 0.04), ha="center", fontsize=7, color=CHART_INK["primary"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(wrap_class_labels(full_names(), width=12), fontsize=7.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Sensitivity vs specificity: melanoma sensitivity nearly matches the paper; its specificity is the gap",
        fontsize=10, color=CHART_INK["primary"],
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    grid_y(ax)

    mel = per_class.loc[per_class["dx"] == "mel"].iloc[0]
    caption = (
        f"Ours: mel sensitivity {mel['recall']:.3f} (paper {PAPER_MEL_SENSITIVITY:.3f}) but specificity "
        f"{mel['specificity']:.3f} (paper {PAPER_MEL_SPECIFICITY:.3f}) — the model over-predicts melanoma. "
        "Black dashes mark the paper's values. Missing preprocessing (V2) and MC-Dropout averaging (V3) are the "
        "attributable causes; see summary.md."
    )
    return finish_figure(fig, out_path, caption)


# --------------------------------------------------------------- 5. ROC curves
def plot_roc_curves(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> Path:
    """One place the 7-slot categorical order genuinely applies: each curve
    is a distinct entity, so color carries identity here.
    """
    curves = roc_curves(y_true, probs)
    fig, ax = new_figure(figsize=(7.5, 6.5))

    for cls in CLASS_ORDER:
        if cls not in curves:
            continue
        c = curves[cls]
        ax.plot(c["fpr"], c["tpr"], color=CLASS_COLOR[cls], linewidth=1.8,
                label=f"{CLASS_FULL_NAME[cls]} — AUROC {c['auroc']:.3f}")
    if "macro" in curves:
        ax.plot(curves["macro"]["fpr"], curves["macro"]["tpr"], color=CHART_INK["primary"],
                linewidth=2.4, linestyle=(0, (4, 2)),
                label=f"Macro average — AUROC {curves['macro']['auroc']:.4f}")
    ax.plot([0, 1], [0, 1], color=CHART_INK["muted"], linewidth=1, label="Chance")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("One-vs-rest ROC curves — every class well above chance", fontsize=10, color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    grid_xy(ax)

    caption = (
        "Macro AUROC is the paper's principal metric. All 7 classes separate strongly, which is why AUROC "
        f"({curves.get('macro', {}).get('auroc', float('nan')):.4f}) is much closer to the paper's {PAPER_TEST_AUROC} "
        "than macro F1 is: AUROC is threshold-free, while F1 is hurt by the argmax decision boundary shifting "
        "toward rare classes."
    )
    return finish_figure(fig, out_path, caption)


# ---------------------------------------------------------------- 6. PR curves
def plot_pr_curves(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> Path:
    """Precision-recall per class — the honest view under 58:1 imbalance,
    where ROC's large true-negative pool flatters the majority class.
    """
    curves = pr_curves(y_true, probs)
    fig, ax = new_figure(figsize=(7.5, 6.5))

    for cls in CLASS_ORDER:
        if cls not in curves:
            continue
        c = curves[cls]
        ax.plot(c["recall"], c["precision"], color=CLASS_COLOR[cls], linewidth=1.8,
                label=f"{CLASS_FULL_NAME[cls]} — AP {c['ap']:.3f} (base {c['baseline']:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curves — the imbalance-honest view", fontsize=10, color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    grid_xy(ax)

    caption = (
        "\"base\" is each class's prevalence — the precision a random classifier would get, and the right "
        "yardstick for average precision. Rare classes have very low baselines, so even modest AP is a large "
        "lift; ROC alone would hide this."
    )
    return finish_figure(fig, out_path, caption)


# --------------------------------------------------------------- 7. error flow
def plot_error_flow(y_true: np.ndarray, probs: np.ndarray, out_path: Path,
                    focus: str = "mel") -> Path:
    """Makes the melanoma over-prediction finding visual (plan Step 4 item 7).

    Left: the true-class composition of everything predicted `focus`.
    Right: where the true-`focus` images actually went.
    """
    counts, _ = confusion_matrices(y_true, probs)
    fi = CLASS_ORDER.index(focus)
    labels = wrap_class_labels(full_names(), width=11)

    fig, axes = new_figure(1, 2, figsize=(12.5, 5))

    # Left: composition of the `focus` prediction column.
    column = counts[:, fi]
    colors = [MALIGNANCY_COLOR[malignancy_of(c)] if i != fi else SINGLE_SERIES_BLUE
              for i, c in enumerate(CLASS_ORDER)]
    bars = axes[0].bar(range(NUM_CLASSES), column, color=colors)
    annotate_bars(axes[0], bars, column, fmt="{:.0f}")
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_xlabel("True class of images predicted as melanoma")
    axes[0].set_ylabel("Images")
    axes[0].set_title(
        f"Of {int(column.sum())} melanoma predictions, only {int(column[fi])} are truly melanoma",
        fontsize=10, color=CHART_INK["primary"],
    )
    grid_y(axes[0])

    # Right: where true-focus rows went.
    row = counts[fi, :]
    bars2 = axes[1].bar(range(NUM_CLASSES), row, color=[SINGLE_SERIES_BLUE if i == fi else CHART_INK["muted"]
                                                        for i in range(NUM_CLASSES)])
    annotate_bars(axes[1], bars2, row, fmt="{:.0f}")
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_xlabel("Predicted class for true-melanoma images")
    axes[1].set_ylabel("Images")
    axes[1].set_title(
        f"True melanoma is mostly caught: {int(row[fi])}/{int(row.sum())} correct",
        fontsize=10, color=CHART_INK["primary"],
    )
    grid_y(axes[1])

    nv_idx = CLASS_ORDER.index("nv")
    caption = (
        f"Left panel, bar colour marks malignancy (red malignant, yellow pre-cancerous, green benign). "
        f"{int(column[nv_idx])} benign nevi are predicted as melanoma — this is what drives mel specificity down "
        "to 0.75, and it is the direct consequence of the paper's dual imbalance correction giving mel ~36x nv's "
        "effective training weight. Errors point toward over-caution, not missed cancer."
    )
    return finish_figure(fig, out_path, caption)


# ----------------------------------------------------- 8. confidence / 10. ecdf
def plot_confidence_distributions(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> Path:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true

    fig, axes = new_figure(1, 2, figsize=(12, 4.5))

    bins = np.linspace(1 / NUM_CLASSES, 1.0, 26)
    axes[0].hist(confidence[correct], bins=bins, color=SINGLE_SERIES_BLUE, alpha=0.85, label="Correct")
    axes[0].hist(confidence[~correct], bins=bins, color=SERIES_2, alpha=0.75, label="Incorrect")
    axes[0].set_xlabel("Predicted probability of the chosen class")
    axes[0].set_ylabel("Images")
    axes[0].set_title("Confidence separates correct from incorrect", fontsize=10, color=CHART_INK["primary"])
    axes[0].legend(frameon=False, fontsize=8)
    grid_y(axes[0])

    entropy = predictive_entropy(probs)
    axes[1].hist(entropy[correct], bins=26, color=SINGLE_SERIES_BLUE, alpha=0.85, label="Correct")
    axes[1].hist(entropy[~correct], bins=26, color=SERIES_2, alpha=0.75, label="Incorrect")
    axes[1].set_xlabel("Predictive entropy (nats)")
    axes[1].set_ylabel("Images")
    axes[1].set_title("Entropy is low for correct, spread for incorrect", fontsize=10, color=CHART_INK["primary"])
    axes[1].legend(frameon=False, fontsize=8)
    grid_y(axes[1])

    caption = (
        f"Mean confidence {confidence[correct].mean():.3f} on correct vs {confidence[~correct].mean():.3f} on "
        "incorrect predictions. This separation is the precondition for V3's entropy-based referral to work: "
        "uncertain cases are disproportionately the wrong ones, so referring them should raise accuracy on what "
        "remains. Single-pass entropy here; V3 decomposes it over T=50 MC Dropout passes."
    )
    return finish_figure(fig, out_path, caption)


def plot_per_class_confidence_ecdf(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> Path:
    fig, ax = new_figure(figsize=(8, 5.5))
    for idx, cls in enumerate(CLASS_ORDER):
        mask = y_true == idx
        if not mask.any():
            continue
        conf = np.sort(probs[mask].max(axis=1))
        ax.step(conf, np.arange(1, len(conf) + 1) / len(conf), where="post",
                color=CLASS_COLOR[cls], linewidth=1.8,
                label=f"{CLASS_FULL_NAME[cls]} (n={int(mask.sum())})")
    ax.set_xlabel("Predicted probability of the chosen class")
    ax.set_ylabel("Cumulative fraction of the class")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-class confidence distribution (ECDF)", fontsize=10, color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    grid_xy(ax)
    caption = (
        "A curve that rises early means the model is habitually unsure about that class. Curves hugging the "
        "right edge are classes the model commits to confidently."
    )
    return finish_figure(fig, out_path, caption)


# --------------------------------------------------------- 9. reliability/ECE
def plot_reliability_diagram(y_true: np.ndarray, probs: np.ndarray, out_path: Path, n_bins: int = 10) -> Path:
    ece, bins = expected_calibration_error(y_true, probs, n_bins=n_bins)
    centers = (bins["bin_lo"] + bins["bin_hi"]) / 2

    fig, axes = new_figure(2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax, ax_hist = axes

    ax.plot([0, 1], [0, 1], color=CHART_INK["muted"], linewidth=1, label="Perfect calibration")
    valid = bins["count"] > 0
    ax.plot(bins.loc[valid, "confidence"], bins.loc[valid, "accuracy"], marker="o", markersize=6,
            color=SINGLE_SERIES_BLUE, linewidth=2, label="Observed")
    for _, row in bins.loc[valid].iterrows():
        ax.annotate(f"{row['accuracy']:.2f}", (row["confidence"], row["accuracy"]),
                    textcoords="offset points", xytext=(4, -9), fontsize=6.5, color=CHART_INK["secondary"])
    ax.set_ylabel("Observed accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Reliability diagram — single pass (T=1), ECE = {ece:.4f}", fontsize=10, color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    grid_xy(ax)

    ax_hist.bar(centers, bins["count"], width=1.0 / n_bins * 0.9, color=CHART_INK["muted"])
    annotate_bars(ax_hist, ax_hist.patches, bins["count"], fmt="{:.0f}", fontsize=6)
    ax_hist.set_xlabel("Predicted confidence")
    ax_hist.set_ylabel("Images")
    ax_hist.set_xlim(0, 1)
    grid_y(ax_hist)

    caption = (
        f"Points below the diagonal mean over-confidence. ECE = {ece:.4f} on a single pass; the paper reports "
        "0.1891 at T=1 and 0.1456 at T=50, so V3's MC Dropout averaging is expected to improve this. "
        "The lower panel shows how many images land in each confidence bin."
    )
    return finish_figure(fig, out_path, caption)


# ------------------------------------------------------- 11. bootstrap spread
def plot_bootstrap_distributions(boot, out_path: Path) -> Path:
    names = [n for n in ("macro_auroc", "macro_f1") if n in boot.samples and len(boot.samples[n])]
    fig, axes = new_figure(1, len(names), figsize=(5.8 * len(names), 4.3))
    axes = np.atleast_1d(axes)

    pretty = {"macro_auroc": "Macro AUROC", "macro_f1": "Macro F1"}
    paper = {"macro_auroc": PAPER_TEST_AUROC, "macro_f1": PAPER_TEST_F1}

    for ax, name in zip(axes, names):
        samples = boot.samples[name]
        ax.hist(samples, bins=40, color=SINGLE_SERIES_BLUE, alpha=0.9)
        for value, color, label in (
            (boot.point_estimate[name], CHART_INK["primary"], f"Point {boot.point_estimate[name]:.4f}"),
            (boot.ci_low[name], CHART_INK["secondary"], f"2.5% {boot.ci_low[name]:.4f}"),
            (boot.ci_high[name], CHART_INK["secondary"], f"97.5% {boot.ci_high[name]:.4f}"),
        ):
            ax.axvline(value, color=color, linewidth=1.4)
            ax.annotate(label, (value, ax.get_ylim()[1] * 0.96), rotation=90, fontsize=6.5,
                        ha="right", va="top", color=color)
        ax.axvline(paper[name], color=SERIES_2, linewidth=1.8)
        ax.annotate(f"paper {paper[name]}", (paper[name], ax.get_ylim()[1] * 0.5), rotation=90,
                    fontsize=6.5, ha="right", va="center", color=SERIES_2)
        ax.set_xlabel(pretty[name])
        ax.set_ylabel("Bootstrap resamples")
        ax.set_title(f"{pretty[name]}: {boot.format_ci(name)}", fontsize=10, color=CHART_INK["primary"])
        grid_y(ax)

    caption = (
        f"{boot.n_resamples_used} of {boot.n_resamples_requested} resamples used "
        f"({boot.n_skipped_missing_class} skipped for missing a class), seed {boot.seed}. "
        "The 95% interval is a measured percentile of this distribution, not a symmetric error bar. "
        "Where the orange paper line falls outside the interval, the gap is statistically real."
    )
    return finish_figure(fig, out_path, caption)


# ---------------------------------------------------- 12. val/test agreement
def plot_val_test_agreement(val_per_class: pd.DataFrame, test_per_class: pd.DataFrame, out_path: Path) -> Path:
    x = np.arange(NUM_CLASSES)
    width = 0.38
    fig, axes = new_figure(1, 2, figsize=(13, 4.8))

    for ax, metric, title in (
        (axes[0], "recall", "Recall agrees between validation and test"),
        (axes[1], "auroc_ovr", "Per-class AUROC agrees between validation and test"),
    ):
        b1 = ax.bar(x - width / 2, val_per_class[metric], width, color=SINGLE_SERIES_BLUE, label="Validation")
        b2 = ax.bar(x + width / 2, test_per_class[metric], width, color=SERIES_2, label="Test")
        annotate_bars(ax, b1, val_per_class[metric], fmt="{:.2f}", fontsize=6)
        annotate_bars(ax, b2, test_per_class[metric], fmt="{:.2f}", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(wrap_class_labels(full_names(), width=11), fontsize=7)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=10, color=CHART_INK["primary"])
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        grid_y(ax)

    caption = (
        "Validation and test track each other closely, so the test split is not anomalous and the reported "
        "test metrics are not a lucky draw. Checkpoint selection used validation macro F1 only, so test "
        "remains untouched by model selection."
    )
    return finish_figure(fig, out_path, caption)


# -------------------------------------------------------------- 13. vs paper
def plot_ours_vs_paper(metrics: dict, boot, out_path: Path) -> Path:
    labels = ["Macro AUROC", "Macro F1"]
    ours = [metrics["macro_auroc"], metrics["macro_f1"]]
    paper = [PAPER_TEST_AUROC, PAPER_TEST_F1]
    err = [
        [ours[0] - boot.ci_low["macro_auroc"], ours[1] - boot.ci_low["macro_f1"]],
        [boot.ci_high["macro_auroc"] - ours[0], boot.ci_high["macro_f1"] - ours[1]],
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = new_figure(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, ours, width, color=SINGLE_SERIES_BLUE, yerr=err, capsize=4,
                ecolor=CHART_INK["primary"], label="Ours — V1, single pass, no preprocessing")
    b2 = ax.bar(x + width / 2, paper, width, color=SERIES_2,
                label="Paper — T=50 MC Dropout, with preprocessing")
    annotate_bars(ax, b1, ours, fmt="{:.4f}", fontsize=8, offset=0.03)
    annotate_bars(ax, b2, paper, fmt="{:.4f}", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Ours vs paper — AUROC nearly matches; F1 is the gap V2/V3 should close",
                 fontsize=10, color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    grid_y(ax)

    caption = (
        "Not a like-for-like comparison, by design: V1 deliberately omits hair removal + CLAHE (V2) and "
        "MC Dropout averaging (V3), and uses provisional hyperparameters. Error bars are our 95% bootstrap CI. "
        f"AUROC is within {abs(ours[0] - paper[0]):.4f} of the paper; macro F1 is "
        f"{abs(ours[1] - paper[1]):.4f} below, which is the expected V1 shortfall."
    )
    return finish_figure(fig, out_path, caption)

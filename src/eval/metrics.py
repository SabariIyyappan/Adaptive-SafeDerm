"""Metric suite (paper §3.7, PLAN.md §5.3 item 2).

Pure functions over numpy arrays — no file I/O beyond `load_predictions`,
so every metric is directly unit-testable against a hand-computed toy
example (PLAN.md §5.3 item 6).

Conventions are pinned to match Member B's training loop exactly
(`src/model/train.py:80-83`) so our numbers reproduce B's `history.csv`:
  - macro F1 with `average="macro"`, `zero_division=0`;
  - macro AUROC one-vs-rest with an explicit `labels=range(7)`.

Per-class "accuracy" is reported **both** ways (PLAN.md §5.3 item 2, §13
item 4): Eq. 17's one-vs-rest (TP+TN)/N, and recall. See
`results/v1/summary.md` for which one matches the paper's reported values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.data.eda import CLASS_ORDER
from src.eval.classes import CLASS_FULL_NAME, NUM_CLASSES, malignancy_of

PROB_COLUMNS = [f"prob_{c}" for c in range(NUM_CLASSES)]
C3_REQUIRED_COLUMNS = ["image_id", "split", "y_true", *PROB_COLUMNS]


def load_predictions(path: Path, expected_rows: int | None = None) -> pd.DataFrame:
    """Read and validate a C3 predictions file (CONTRACTS.md C3).

    Validates the contract on read rather than trusting it, mirroring how
    `src.model.dataset.HAM10000Split` validates C2 — a malformed
    predictions file should fail here with an actionable message, not
    produce a plausible-looking but wrong figure.
    """
    df = pd.read_csv(path)

    missing = [c for c in C3_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is not a valid C3 predictions file — missing columns: {missing}. "
            f"Expected (CONTRACTS.md C3): {C3_REQUIRED_COLUMNS} [+ optional 'run_id']"
        )

    if len(df) == 0:
        raise ValueError(f"{path} has no rows")

    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(
            f"{path} has {len(df)} rows but the split it claims to cover has "
            f"{expected_rows}. C3 requires one row per image in the split."
        )

    probs = df[PROB_COLUMNS].to_numpy(dtype=float)
    row_sums = probs.sum(axis=1)
    bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-4))[0]
    if len(bad):
        raise ValueError(
            f"{path}: {len(bad)} probability rows do not sum to 1.0 (atol=1e-4); "
            f"first offenders at positions {bad[:5].tolist()} with sums "
            f"{row_sums[bad[:5]].round(6).tolist()}"
        )

    labels = df["y_true"].to_numpy()
    out_of_range = labels[(labels < 0) | (labels >= NUM_CLASSES)]
    if len(out_of_range):
        raise ValueError(
            f"{path}: y_true has {len(out_of_range)} values outside 0..{NUM_CLASSES - 1} "
            f"(e.g. {out_of_range[:5].tolist()}) — check the C1 label map was applied"
        )

    return df


def probs_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract (probs NxC, y_true N) from a validated C3 frame."""
    return df[PROB_COLUMNS].to_numpy(dtype=float), df["y_true"].to_numpy(dtype=int)


def macro_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Headline metrics (paper §3.7). Macro AUROC is the principal metric."""
    y_pred = probs.argmax(axis=1)
    labels = list(range(NUM_CLASSES))

    macro_auroc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)

    # Balanced accuracy = mean per-class recall. Reported alongside overall
    # accuracy because under 58:1 imbalance overall accuracy is dominated
    # by `nv` and is context-only (PLAN.md §5.3 item 2).
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class_recall = np.diag(cm) / cm.sum(axis=1)
    balanced_acc = float(np.nanmean(per_class_recall))

    return {
        "macro_auroc": float(macro_auroc),
        "macro_f1": float(macro_f1),
        "overall_accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": balanced_acc,
        "n": int(len(y_true)),
    }


def per_class_table(y_true: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    """One row per class, with BOTH accuracy definitions (PLAN.md §13 item 4).

    `accuracy_eq17` is the paper's Eq. 17 one-vs-rest accuracy (TP+TN)/N;
    `recall` is sensitivity. These differ sharply for the majority class,
    which is what resolves the paper's ambiguity.
    """
    y_pred = probs.argmax(axis=1)
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = int(cm.sum())

    rows = []
    for idx, cls in enumerate(CLASS_ORDER):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = n - tp - fn - fp
        support = tp + fn

        rows.append(
            {
                "label": idx,
                "dx": cls,
                "class_name": CLASS_FULL_NAME[cls],
                "malignancy": malignancy_of(cls),
                "support": support,
                "n_predicted": int(cm[:, idx].sum()),
                "accuracy_eq17": (tp + tn) / n if n else float("nan"),
                "recall": tp / support if support else float("nan"),
                "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
                "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
                "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
                "auroc_ovr": _one_vs_rest_auroc(y_true, probs, idx),
            }
        )
    return pd.DataFrame(rows)


def _one_vs_rest_auroc(y_true: np.ndarray, probs: np.ndarray, class_idx: int) -> float:
    binary_true = (y_true == class_idx).astype(int)
    if binary_true.min() == binary_true.max():
        return float("nan")  # class absent (or only class) — ovr AUROC undefined
    return float(roc_auc_score(binary_true, probs[:, class_idx]))


def confusion_matrices(y_true: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(counts, row-normalized). Row-normalized rows sum to 1 where support>0."""
    y_pred = probs.argmax(axis=1)
    counts = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = counts / counts.sum(axis=1, keepdims=True)
    return counts, np.nan_to_num(normalized)


def roc_curves(y_true: np.ndarray, probs: np.ndarray) -> dict[str, dict]:
    """Per-class one-vs-rest ROC curves plus the macro-average curve."""
    curves: dict[str, dict] = {}
    all_fpr = np.unique(np.concatenate([
        roc_curve((y_true == i).astype(int), probs[:, i])[0]
        for i in range(NUM_CLASSES)
        if len(np.unique((y_true == i).astype(int))) > 1
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    n_valid = 0

    for idx, cls in enumerate(CLASS_ORDER):
        binary_true = (y_true == idx).astype(int)
        if len(np.unique(binary_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(binary_true, probs[:, idx])
        curves[cls] = {"fpr": fpr, "tpr": tpr, "auroc": float(roc_auc_score(binary_true, probs[:, idx]))}
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        n_valid += 1

    if n_valid:
        mean_tpr /= n_valid
        curves["macro"] = {
            "fpr": all_fpr,
            "tpr": mean_tpr,
            "auroc": float(np.trapezoid(mean_tpr, all_fpr)),
        }
    return curves


def pr_curves(y_true: np.ndarray, probs: np.ndarray) -> dict[str, dict]:
    """Per-class precision-recall curves. More informative than ROC under
    heavy imbalance, where a large TN pool flatters the majority class.
    """
    curves: dict[str, dict] = {}
    for idx, cls in enumerate(CLASS_ORDER):
        binary_true = (y_true == idx).astype(int)
        if len(np.unique(binary_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(binary_true, probs[:, idx])
        # Average precision via the step-wise sum (sklearn's AP convention).
        ap = float(-np.sum(np.diff(recall) * precision[:-1]))
        curves[cls] = {"precision": precision, "recall": recall, "ap": ap, "baseline": float(binary_true.mean())}
    return curves


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> tuple[float, pd.DataFrame]:
    """ECE with equal-width bins on max-probability confidence (paper §3.7,
    Eqs. 15-16). Returns (ece, per-bin frame) so the reliability diagram
    and the scalar come from the same binning.

    V1 reports the single-pass (T=1) value only; V3 adds the T=50
    comparison from the same function.
    """
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin closes on the right so confidence == 1.0 is included.
        in_bin = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if count:
            bin_acc = float(correct[in_bin].mean())
            bin_conf = float(confidence[in_bin].mean())
            ece += (count / n) * abs(bin_acc - bin_conf)
        else:
            bin_acc = float("nan")
            bin_conf = float("nan")
        rows.append({"bin_lo": lo, "bin_hi": hi, "count": count, "accuracy": bin_acc, "confidence": bin_conf})

    return float(ece), pd.DataFrame(rows)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot target."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def confidence_summary(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Mean max-probability overall / on correct / on incorrect predictions.

    A model whose confidence separates correct from incorrect is the
    precondition for V3's entropy-based referral to work at all, so this
    is the V1 hook into that work.
    """
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    return {
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else float("nan"),
        "mean_confidence_incorrect": float(confidence[~correct].mean()) if (~correct).any() else float("nan"),
    }


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each predictive distribution, in nats (paper §3.4.3).

    V1 uses it only descriptively (single-pass); V3 decomposes it into
    aleatoric/epistemic over T=50 MC Dropout passes.
    """
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)

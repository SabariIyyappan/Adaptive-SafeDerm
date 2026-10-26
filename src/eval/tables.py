"""Result tables (PLAN.md §5.3 item 5).

Markdown via `DataFrame.to_markdown()`, matching how `src/data/verify_split.py`
writes its reports (requires `tabulate`, pinned in `env/requirements.txt`).

  - headline metrics with CI, beside a paper column;
  - per-class table in the paper's Table 5 format, with BOTH accuracy defs;
  - training milestones (paper Table 3 style);
  - split fairness;
  - the imbalance-correction weight table that explains the mel behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.eda import CLASS_ORDER
from src.eval.classes import CLASS_FULL_NAME, NUM_CLASSES, malignancy_of
from src.eval.figures import (
    PAPER_MEL_SENSITIVITY,
    PAPER_MEL_SPECIFICITY,
    PAPER_NV_ACCURACY,
    PAPER_TEST_AUROC,
    PAPER_TEST_F1,
)

MILESTONE_EPOCHS = [1, 5, 10, 15, 20, 25, 30]


def headline_table(metrics: dict, boot, ece: float, brier: float) -> pd.DataFrame:
    """Ours (with CI) beside the paper's values, clearly labelled as
    non-comparable protocols.
    """
    return pd.DataFrame([
        {
            "Metric": "Macro AUROC (principal)",
            "Ours — V1, single pass, no preprocessing": boot.format_ci("macro_auroc"),
            "Paper — T=50 MC Dropout, with preprocessing": f"{PAPER_TEST_AUROC} [0.9351, 0.9455]",
        },
        {
            "Metric": "Macro F1",
            "Ours — V1, single pass, no preprocessing": boot.format_ci("macro_f1"),
            "Paper — T=50 MC Dropout, with preprocessing": f"{PAPER_TEST_F1} [0.7109, 0.7497]",
        },
        {
            "Metric": "ECE (10 bins)",
            "Ours — V1, single pass, no preprocessing": f"{ece:.4f}",
            "Paper — T=50 MC Dropout, with preprocessing": "0.1456 (T=50) · 0.1891 (T=1)",
        },
        {
            "Metric": "Brier score",
            "Ours — V1, single pass, no preprocessing": f"{brier:.4f}",
            "Paper — T=50 MC Dropout, with preprocessing": "not reported numerically",
        },
        {
            "Metric": "Overall accuracy (context only)",
            "Ours — V1, single pass, no preprocessing": f"{metrics['overall_accuracy']:.4f}",
            "Paper — T=50 MC Dropout, with preprocessing": "not reported",
        },
        {
            "Metric": "Balanced accuracy (mean recall)",
            "Ours — V1, single pass, no preprocessing": f"{metrics['balanced_accuracy']:.4f}",
            "Paper — T=50 MC Dropout, with preprocessing": "not reported",
        },
    ])


def per_class_markdown_table(per_class: pd.DataFrame) -> pd.DataFrame:
    """Paper Table 5 format, with a paper-value column where the text quotes one."""
    paper_values = {
        "mel": f"sens {PAPER_MEL_SENSITIVITY:.3f} / spec {PAPER_MEL_SPECIFICITY:.3f}",
        "nv": f"accuracy {PAPER_NV_ACCURACY:.3f} (Eq. 17)",
    }
    out = per_class.copy()
    out["Class"] = out["dx"].map(CLASS_FULL_NAME)
    out["Category"] = out["dx"].map(malignancy_of)
    out["Paper (quoted in text)"] = out["dx"].map(paper_values).fillna("—")
    cols = {
        "Class": "Class",
        "Category": "Category",
        "support": "Support",
        "n_predicted": "Predicted",
        "accuracy_eq17": "Accuracy (Eq. 17)",
        "recall": "Recall / sensitivity",
        "specificity": "Specificity",
        "precision": "Precision",
        "f1": "F1",
        "auroc_ovr": "AUROC (ovr)",
        "Paper (quoted in text)": "Paper (quoted in text)",
    }
    out = out[list(cols)].rename(columns=cols)
    for c in ("Accuracy (Eq. 17)", "Recall / sensitivity", "Specificity", "Precision", "F1", "AUROC (ovr)"):
        out[c] = out[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    return out


def training_milestones_table(history: pd.DataFrame, best_epoch: int) -> pd.DataFrame:
    """Paper Table 3 style: fixed milestone epochs plus the best epoch."""
    epochs = sorted(set(MILESTONE_EPOCHS + [best_epoch]) & set(history["epoch"]))
    rows = []
    for epoch in epochs:
        row = history.loc[history["epoch"] == epoch].iloc[0]
        rows.append({
            "Epoch": int(epoch),
            "Train loss": f"{row['train_loss']:.4f}",
            "Val loss": f"{row['val_loss']:.4f}",
            "Val macro F1": f"{row['val_macro_f1']:.4f}",
            "Val macro AUROC": f"{row['val_macro_auroc']:.4f}",
            "LR": f"{row['lr']:.2e}",
            "Note": "best (selected checkpoint)" if epoch == best_epoch else "",
        })
    return pd.DataFrame(rows)


def split_fairness_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The evidence that the split carries no class or sampling bias."""
    from src.eval.eda import SPLIT_ORDER, split_proportion_tables

    image_pct, lesion_pct = split_proportion_tables(splits)
    rows = []
    for cls in CLASS_ORDER:
        row = {"Class": CLASS_FULL_NAME[cls]}
        for name in SPLIT_ORDER:
            row[f"{name} % (img)"] = f"{image_pct.loc[cls, name]:.2f}"
        row["max drift vs train (img, pp)"] = (
            f"{max(abs(image_pct.loc[cls, s] - image_pct.loc[cls, 'train']) for s in ('val', 'test')):.3f}"
        )
        row["max drift vs train (lesion, pp)"] = (
            f"{max(abs(lesion_pct.loc[cls, s] - lesion_pct.loc[cls, 'train']) for s in ('val', 'test')):.3f}"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def imbalance_correction_table(train_split: pd.DataFrame) -> pd.DataFrame:
    """Why the model over-predicts melanoma.

    The paper applies two corrections (§3.2.5) that COMPOUND: the sampler
    equalizes how often each class is drawn, and the class-weighted loss
    then re-weights those same examples again. This table makes the
    combined effective weight per class explicit.
    """
    counts = train_split["label"].value_counts().sort_index()
    n = counts.reindex(range(NUM_CLASSES)).fillna(0).to_numpy(dtype=float)
    inv = 1.0 / n
    loss_weight = inv / inv.sum()              # paper Eq. 2
    raw_share = n / n.sum()
    sampler_share = 1.0 / NUM_CLASSES          # sampler equalizes to uniform
    sampler_boost = sampler_share / raw_share
    combined = sampler_boost * loss_weight
    nv_idx = CLASS_ORDER.index("nv")
    relative = combined / combined[nv_idx]

    return pd.DataFrame({
        "Class": [CLASS_FULL_NAME[c] for c in CLASS_ORDER],
        "Train images": n.astype(int),
        "Raw share %": (100 * raw_share).round(2),
        "Sampler boost": [f"{b:.2f}x" for b in sampler_boost],
        "Loss weight w_c": loss_weight.round(4),
        "Combined, relative to nv": [f"{r:,.0f}x" for r in relative],
    })


def bootstrap_table(boot) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Metric": name.replace("macro_", "macro ").replace("_", " "),
            "Point estimate": f"{boot.point_estimate[name]:.4f}",
            f"{boot.ci_level:.0f}% CI": f"[{boot.ci_low[name]:.4f}, {boot.ci_high[name]:.4f}]",
            "CI width": f"{boot.ci_high[name] - boot.ci_low[name]:.4f}",
        }
        for name in boot.point_estimate
    ])


def write_tables(tables: dict[str, pd.DataFrame], out_dir) -> list:
    """Write each table as both .md and .csv; returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in tables.items():
        md = out_dir / f"{name}.md"
        md.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
        csv = out_dir / f"{name}.csv"
        df.to_csv(csv, index=False)
        written += [md, csv]
    return written

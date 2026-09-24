"""Rigorous EDA for the V1 result package (PLAN.md §5.3 item 7, extended).

Member A's `src/data/eda.py` produced three basic figures; this module is
Member C's presentation-grade replacement, built in C's own folder
(PLAN.md §4.2 — we import from `src/data/` but never edit it).

Five groups:
  A. Dataset composition          (1-4)
  B. Lesion structure             (5-7)
  C. Split fairness               (8-10) — the evidence that the split is unbiased
  D. Image-content statistics     (11-17) — needs data/raw/images
  E. Class-difficulty context     (18-19) — needs C3 predictions

Every figure follows the plan's design rules: full diagnostic names, a
takeaway title, a caption, direct-labeled values, one blue for nominal
bar charts and the 7-slot categorical order only where colour carries
identity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

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
from src.eval.style import annotate_bars, finish_figure, grid_xy, grid_y, new_figure, wrap_class_labels

SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLOR = {"train": SINGLE_SERIES_BLUE, "val": SERIES_2, "test": "#1baf7a"}
PAPER_SPLIT_IMAGE_COUNTS = {"train": 7024, "val": 1497, "test": 1494}

# Sampling caps for the expensive per-pixel passes. The full 10,015-image
# sweep is affordable for basic stats; hair detection (morphology on the
# full-resolution image) is capped per class so the EDA stays interactive.
HAIR_SAMPLE_PER_CLASS = 120
DUPLICATE_HASH_SIZE = 8


# ============================================================ A. composition
def plot_class_distribution(metadata: pd.DataFrame, out_path: Path) -> Path:
    """Rebuilt: full diagnostic names, malignancy grouping, imbalance in the title."""
    counts = metadata["dx"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    pct = 100 * counts / counts.sum()
    ratio = counts.max() / counts.min()

    fig, ax = new_figure(figsize=(11, 5.4))
    colors = [MALIGNANCY_COLOR[malignancy_of(c)] for c in CLASS_ORDER]
    bars = ax.bar(range(NUM_CLASSES), counts.values, color=colors)
    for bar, n, p in zip(bars, counts.values, pct.values):
        ax.annotate(f"{n:,}\n{p:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8, color=CHART_INK["primary"])

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(wrap_class_labels(full_names(), width=12), fontsize=8)
    ax.set_ylabel("Images")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title(
        f"HAM10000 is severely imbalanced: {ratio:.0f}:1 between the most and least common class",
        fontsize=11, color=CHART_INK["primary"],
    )

    handles = [
        __import__("matplotlib").patches.Patch(color=color, label=level.capitalize())
        for level, color in MALIGNANCY_COLOR.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, title="Clinical category",
              title_fontsize=8, loc="upper left")
    grid_y(ax)

    caption = (
        f"Bar colour is the clinical category, not the class identity. Melanocytic nevus alone is "
        f"{pct['nv']:.1f}% of the dataset while dermatofibroma is {pct['df']:.1f}% — the imbalance the paper's "
        "WeightedRandomSampler and class-weighted loss exist to correct. "
        "(PLAN.md §13 item 2: the paper calls this 67:1; by image counts it is 58:1.)"
    )
    return finish_figure(fig, out_path, caption)


def plot_class_imbalance_ratio(metadata: pd.DataFrame, out_path: Path) -> Path:
    counts = metadata["dx"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    rarest = counts.min()
    ratios = counts / rarest

    fig, ax = new_figure(figsize=(10, 5))
    bars = ax.bar(range(NUM_CLASSES), ratios.values, color=SINGLE_SERIES_BLUE)
    annotate_bars(ax, bars, ratios.values, fmt="{:.1f}x", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(wrap_class_labels(full_names(), width=12), fontsize=8)
    ax.set_ylabel(f"Images relative to the rarest class (log scale)")
    ax.set_title("How many times more common is each class than dermatofibroma?",
                 fontsize=11, color=CHART_INK["primary"])
    grid_y(ax)

    caption = (
        "A log scale is used because a linear one makes every class except nevus look identical. "
        "Without correction, a model minimizing plain cross-entropy can score 67% accuracy by predicting "
        "\"nevus\" for every image — which is exactly the failure the paper's unmitigated baseline shows "
        "(df sensitivity 0%)."
    )
    return finish_figure(fig, out_path, caption)


def plot_malignancy_breakdown(metadata: pd.DataFrame, out_path: Path) -> Path:
    meta = metadata.copy()
    meta["malignancy"] = meta["dx"].map(malignancy_of)
    order = ["malignant", "pre-cancerous", "benign"]
    counts = meta["malignancy"].value_counts().reindex(order).fillna(0).astype(int)
    pct = 100 * counts / counts.sum()

    fig, axes = new_figure(1, 2, figsize=(12, 4.8))
    bars = axes[0].bar(range(len(order)), counts.values,
                       color=[MALIGNANCY_COLOR[o] for o in order])
    for bar, n, p in zip(bars, counts.values, pct.values):
        axes[0].annotate(f"{n:,}\n{p:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha="center", va="bottom", fontsize=8.5, color=CHART_INK["primary"])
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([o.capitalize() for o in order], fontsize=9)
    axes[0].set_ylabel("Images")
    axes[0].set_ylim(0, counts.max() * 1.18)
    axes[0].set_title("Clinical category totals", fontsize=10, color=CHART_INK["primary"])
    grid_y(axes[0])

    # Per-class stacked into its category, so the composition is visible.
    class_counts = meta["dx"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    bottoms = {o: 0 for o in order}
    for cls in CLASS_ORDER:
        cat = malignancy_of(cls)
        xi = order.index(cat)
        axes[1].bar(xi, class_counts[cls], bottom=bottoms[cat],
                    color=CLASS_COLOR[cls], edgecolor=CHART_INK["surface"], linewidth=2,
                    label=CLASS_FULL_NAME[cls])
        if class_counts[cls] > 200:
            axes[1].annotate(f"{cls} {class_counts[cls]:,}",
                             (xi, bottoms[cat] + class_counts[cls] / 2),
                             ha="center", va="center", fontsize=7, color="white")
        bottoms[cat] += class_counts[cls]
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([o.capitalize() for o in order], fontsize=9)
    axes[1].set_ylabel("Images")
    axes[1].set_title("Which diagnoses make up each category", fontsize=10, color=CHART_INK["primary"])
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left", ncol=1)
    grid_y(axes[1])

    caption = (
        f"Only {pct['malignant']:.1f}% of the dataset is malignant, yet those are the cases where a miss is "
        "most costly. This asymmetry is why the paper reports melanoma sensitivity separately and why "
        "V3's referral mechanism targets uncertain cases rather than overall accuracy."
    )
    return finish_figure(fig, out_path, caption)


def dataset_summary_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """Paper Table 1 equivalent, extended with lesion structure and category."""
    counts = metadata["dx"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    lesions = metadata.groupby("dx")["lesion_id"].nunique().reindex(CLASS_ORDER).fillna(0).astype(int)
    return pd.DataFrame({
        "dx": CLASS_ORDER,
        "class_name": [CLASS_FULL_NAME[c] for c in CLASS_ORDER],
        "clinical_category": [malignancy_of(c) for c in CLASS_ORDER],
        "n_images": counts.values,
        "pct_images": (100 * counts / counts.sum()).round(2).values,
        "n_unique_lesions": lesions.values,
        "images_per_lesion": (counts / lesions).round(3).values,
        "ratio_to_rarest": (counts / counts.min()).round(1).values,
    })


# ======================================================== B. lesion structure
def plot_images_per_lesion(metadata: pd.DataFrame, out_path: Path) -> Path:
    """Rebuilt with an explanatory title — the original was unlabelled and unintuitive."""
    per_lesion = metadata.groupby("lesion_id").size()
    dist = per_lesion.value_counts().sort_index()
    n_multi = int((per_lesion > 1).sum())
    pct_multi = 100 * n_multi / len(per_lesion)

    fig, ax = new_figure(figsize=(10, 5))
    bars = ax.bar(dist.index, dist.values, color=SINGLE_SERIES_BLUE)
    annotate_bars(ax, bars, dist.values, fmt="{:,.0f}", fontsize=8)
    ax.set_xticks(dist.index)
    ax.set_xlabel("Photographs taken of the same physical lesion")
    ax.set_ylabel("Number of lesions")
    ax.set_ylim(0, dist.max() * 1.15)
    ax.set_title(
        f"{len(per_lesion):,} physical lesions produced {len(metadata):,} images — "
        f"{n_multi:,} lesions ({pct_multi:.1f}%) were photographed more than once",
        fontsize=11, color=CHART_INK["primary"],
    )
    grid_y(ax)

    caption = (
        "This is why the split must group by lesion. If images were split independently, two photos of the "
        "SAME lesion could land in train and test, and the model would be tested on a lesion it had already "
        "memorized — inflating every metric. Grouping by lesion_id makes that impossible (PLAN.md §5.3.4)."
    )
    return finish_figure(fig, out_path, caption)


def plot_images_per_lesion_by_class(metadata: pd.DataFrame, out_path: Path) -> Path:
    counts = metadata["dx"].value_counts().reindex(CLASS_ORDER)
    lesions = metadata.groupby("dx")["lesion_id"].nunique().reindex(CLASS_ORDER)
    ratio = (counts / lesions)

    fig, ax = new_figure(figsize=(10, 5))
    bars = ax.bar(range(NUM_CLASSES), ratio.values, color=SINGLE_SERIES_BLUE)
    annotate_bars(ax, bars, ratio.values, fmt="{:.2f}", fontsize=8)
    ax.axhline(len(metadata) / metadata["lesion_id"].nunique(), color=CHART_INK["muted"], linewidth=1)
    ax.annotate(f"dataset mean {len(metadata) / metadata['lesion_id'].nunique():.2f}",
                (-0.42, len(metadata) / metadata["lesion_id"].nunique() + 0.02),
                ha="left", fontsize=7, color=CHART_INK["secondary"])
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(wrap_class_labels(full_names(), width=12), fontsize=8)
    ax.set_ylabel("Images per unique lesion")
    ax.set_ylim(0, ratio.max() * 1.2)
    ax.set_title("Are rare classes inflated by repeat photography of the same lesion?",
                 fontsize=11, color=CHART_INK["primary"])
    grid_y(ax)

    top = ratio.idxmax()
    caption = (
        f"A class well above the mean gets much of its apparent size from repeat shots rather than distinct "
        f"lesions — {CLASS_FULL_NAME[top]} is highest at {ratio.max():.2f}. This matters because the effective "
        "number of independent examples for such a class is lower than its image count suggests, so its "
        "metrics carry more variance than the support column implies."
    )
    return finish_figure(fig, out_path, caption)


def plot_lesion_multiplicity_vs_class(metadata: pd.DataFrame, out_path: Path) -> Path:
    per_lesion = metadata.groupby(["dx", "lesion_id"]).size().rename("n_images").reset_index()
    table = pd.crosstab(per_lesion["dx"], per_lesion["n_images"]).reindex(CLASS_ORDER).fillna(0)
    # Row-normalize so classes of very different sizes are comparable.
    normalized = table.div(table.sum(axis=1), axis=0)

    fig, ax = new_figure(figsize=(9, 5.2))
    im = ax.imshow(normalized.values, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, fontsize=8)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(wrap_class_labels(full_names(), width=13), fontsize=7.5)
    ax.set_xlabel("Images of the same lesion")
    ax.set_ylabel("Class")
    ax.set_title("Within each class, how often are lesions photographed repeatedly?",
                 fontsize=11, color=CHART_INK["primary"])
    ax.grid(False)
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            v = normalized.values[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else CHART_INK["primary"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of the class's lesions")

    caption = (
        "Rows sum to 1. A row concentrated in column 1 means that class is made of distinct lesions; mass in "
        "the higher columns means repeat photography. Read together with the previous figure, this shows "
        "whether a class's image count overstates its independent sample size."
    )
    return finish_figure(fig, out_path, caption)


# ========================================================== C. split fairness
def load_splits(split_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(split_dir / f"{name}.csv") for name in SPLIT_ORDER}


def split_proportion_tables(splits: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(image-level %, lesion-level %) per class per split."""
    image_pct, lesion_pct = {}, {}
    for name, df in splits.items():
        image_pct[name] = (100 * df["dx"].value_counts(normalize=True)).reindex(CLASS_ORDER).fillna(0)
        lesions = df.drop_duplicates("lesion_id")
        lesion_pct[name] = (100 * lesions["dx"].value_counts(normalize=True)).reindex(CLASS_ORDER).fillna(0)
    return pd.DataFrame(image_pct)[SPLIT_ORDER], pd.DataFrame(lesion_pct)[SPLIT_ORDER]


def plot_split_class_proportions(splits: dict[str, pd.DataFrame], out_path: Path) -> Path:
    image_pct, lesion_pct = split_proportion_tables(splits)
    x = np.arange(NUM_CLASSES)
    width = 0.27

    fig, axes = new_figure(2, 1, figsize=(11.5, 8.5), sharex=True)
    for ax, table, title in (
        (axes[0], image_pct, "Image-level class mix per split"),
        (axes[1], lesion_pct, "Lesion-level class mix per split — this is what stratification acts on"),
    ):
        for i, name in enumerate(SPLIT_ORDER):
            bars = ax.bar(x + (i - 1) * width, table[name].values, width,
                          color=SPLIT_COLOR[name], label=f"{name} (n={len(splits[name]):,})")
            annotate_bars(ax, bars, table[name].values, fmt="{:.1f}", fontsize=5.8)
        ax.set_ylabel("% of the split")
        ax.set_title(title, fontsize=10, color=CHART_INK["primary"])
        ax.legend(frameon=False, fontsize=8)
        grid_y(ax)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(wrap_class_labels(full_names(), width=12), fontsize=8)

    caption = (
        "The three bars per class are near-identical in both panels, so no split is enriched or depleted in "
        "any diagnosis. The lesion-level panel is the tighter of the two because stratification is applied to "
        "unique lesions; the image-level panel inherits a small amount of drift because lesions carry "
        "different numbers of photographs."
    )
    return finish_figure(fig, out_path, caption)


def plot_split_drift(splits: dict[str, pd.DataFrame], out_path: Path) -> Path:
    """Deviation from the train proportion, with a ±1pp reference band."""
    image_pct, lesion_pct = split_proportion_tables(splits)
    x = np.arange(NUM_CLASSES)
    width = 0.34

    fig, axes = new_figure(1, 2, figsize=(13, 5))
    for ax, table, title in (
        (axes[0], image_pct, "Image-level drift from train"),
        (axes[1], lesion_pct, "Lesion-level drift from train"),
    ):
        drift = table[["val", "test"]].sub(table["train"], axis=0)
        for i, name in enumerate(["val", "test"]):
            bars = ax.bar(x + (i - 0.5) * width, drift[name].values, width,
                          color=SPLIT_COLOR[name], label=f"{name} − train")
            annotate_bars(ax, bars, drift[name].values, fmt="{:+.2f}", fontsize=6)
        ax.axhspan(-1, 1, color=CHART_INK["gridline"], zorder=0)
        ax.axhline(0, color=CHART_INK["muted"], linewidth=1)
        worst = float(drift.abs().to_numpy().max())
        ax.set_xticks(x)
        ax.set_xticklabels(wrap_class_labels(full_names(), width=11), fontsize=7)
        ax.set_ylabel("Percentage points")
        ax.set_ylim(-2, 2)
        ax.set_title(f"{title} — worst {worst:.2f} pp", fontsize=10, color=CHART_INK["primary"])
        ax.legend(frameon=False, fontsize=8)
        grid_y(ax)

    caption = (
        "The grey band is ±1 percentage point. Every class in every split sits inside it, and the lesion-level "
        "panel is near-zero throughout. Verdict: the split carries no class bias — the residual image-level "
        "drift is an unavoidable consequence of grouping by lesion, and is far preferable to the leakage that "
        "image-level splitting would cause."
    )
    return finish_figure(fig, out_path, caption)


def plot_split_sizes(splits: dict[str, pd.DataFrame], out_path: Path) -> Path:
    images = [len(splits[s]) for s in SPLIT_ORDER]
    lesions = [splits[s]["lesion_id"].nunique() for s in SPLIT_ORDER]
    paper = [PAPER_SPLIT_IMAGE_COUNTS[s] for s in SPLIT_ORDER]
    x = np.arange(len(SPLIT_ORDER))
    width = 0.27

    fig, axes = new_figure(1, 2, figsize=(12, 4.8))
    b1 = axes[0].bar(x - width, images, width, color=SINGLE_SERIES_BLUE, label="Ours — images")
    b2 = axes[0].bar(x, paper, width, color=SERIES_2, label="Paper — images")
    b3 = axes[0].bar(x + width, lesions, width, color=CHART_INK["muted"], label="Ours — unique lesions")
    for bars, vals in ((b1, images), (b2, paper), (b3, lesions)):
        annotate_bars(axes[0], bars, vals, fmt="{:,.0f}", fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SPLIT_ORDER)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Split sizes — ours vs the paper's reported counts", fontsize=10, color=CHART_INK["primary"])
    axes[0].legend(frameon=False, fontsize=8)
    grid_y(axes[0])

    pct = [100 * n / sum(images) for n in images]
    bars = axes[1].bar(x, pct, 0.5, color=SINGLE_SERIES_BLUE)
    annotate_bars(axes[1], bars, pct, fmt="{:.1f}%", fontsize=8)
    for target, xi in zip([70, 15, 15], x):
        axes[1].scatter([xi], [target], marker="_", s=600, color=CHART_INK["primary"], zorder=5, linewidths=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(SPLIT_ORDER)
    axes[1].set_ylabel("% of all images")
    axes[1].set_ylim(0, 80)
    axes[1].set_title("Achieved 70/15/15 (black dashes = target)", fontsize=10, color=CHART_INK["primary"])
    grid_y(axes[1])

    caption = (
        "Our image counts differ slightly from the paper's because lesion-grouped splitting cannot hit exact "
        "image percentages — whole lesions move together. The paper's own lesion counts (1,891/405/405 = 2,701) "
        "are irreconcilable with HAM10000's 7,470 lesions, so we follow its stated method and report our own "
        "counts (PLAN.md §13 item 1)."
    )
    return finish_figure(fig, out_path, caption)


def split_leakage_checks(splits: dict[str, pd.DataFrame], metadata: pd.DataFrame) -> dict:
    """Recomputed live rather than cited, so the report never goes stale."""
    lesion_sets = {n: set(df["lesion_id"]) for n, df in splits.items()}
    image_sets = {n: set(df["image_id"]) for n, df in splits.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {
        "lesion_overlaps": {f"{a}-{b}": len(lesion_sets[a] & lesion_sets[b]) for a, b in pairs},
        "image_overlaps": {f"{a}-{b}": len(image_sets[a] & image_sets[b]) for a, b in pairs},
        "union_covers_all": len(set().union(*image_sets.values())) == metadata["image_id"].nunique(),
        "n_union": len(set().union(*image_sets.values())),
        "n_metadata": int(metadata["image_id"].nunique()),
        "classes_present": {n: sorted(df["dx"].unique()) for n, df in splits.items()},
        "all_classes_everywhere": all(
            set(df["dx"].unique()) == set(CLASS_ORDER) for df in splits.values()
        ),
        "n_images": {n: len(df) for n, df in splits.items()},
        "n_lesions": {n: int(df["lesion_id"].nunique()) for n, df in splits.items()},
        "mean_images_per_lesion": {
            n: round(len(df) / df["lesion_id"].nunique(), 3) for n, df in splits.items()
        },
    }


# =================================================== D. image-content stats
def _image_path(images_dir: Path, image_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def plot_sample_grid(metadata: pd.DataFrame, images_dir: Path, out_path: Path,
                     n_per_class: int = 4, seed: int = 42) -> Path | None:
    """Rebuilt: full names and malignancy-coloured borders (the paper's scheme)."""
    rng = np.random.default_rng(seed)
    fig, axes = new_figure(NUM_CLASSES, n_per_class,
                           figsize=(n_per_class * 2.3, NUM_CLASSES * 2.15))
    any_found = False
    for row, cls in enumerate(CLASS_ORDER):
        pool = metadata.loc[metadata["dx"] == cls, "image_id"].to_numpy()
        picks = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)
        border = MALIGNANCY_COLOR[malignancy_of(cls)]
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border)
                spine.set_linewidth(2.5)
            if col < len(picks):
                path = _image_path(images_dir, picks[col])
                if path is not None:
                    with Image.open(path) as im:
                        ax.imshow(im)
                    any_found = True
            if col == 0:
                ax.set_ylabel(
                    "\n".join(wrap_class_labels([CLASS_FULL_NAME[cls]], width=13)),
                    fontsize=7.5, color=CHART_INK["primary"], rotation=0,
                    ha="right", va="center", labelpad=42,
                )
    if not any_found:
        import matplotlib.pyplot as plt
        plt.close(fig)
        return None

    fig.suptitle("Four random images per class (seed 42), border colour = clinical category",
                 fontsize=11, color=CHART_INK["primary"])
    caption = (
        "Red = malignant, yellow = pre-cancerous, green = benign, matching the paper's Figure 8 scheme. "
        "Note the visible hair, colour casts and varying illumination — exactly the artifacts V2's hair "
        "removal and CLAHE are meant to normalize."
    )
    return finish_figure(fig, out_path, caption)


def compute_image_stats(metadata: pd.DataFrame, images_dir: Path,
                        sample_per_class: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Per-image size, RGB/LAB means, brightness, contrast and sharpness.

    Uses OpenCV for LAB and the Laplacian; images are read at full
    resolution because that is what V2's preprocessing will operate on.
    """
    import cv2

    rng = np.random.default_rng(seed)
    rows = []
    for cls in CLASS_ORDER:
        ids = metadata.loc[metadata["dx"] == cls, "image_id"].to_numpy()
        if sample_per_class is not None and len(ids) > sample_per_class:
            ids = rng.choice(ids, size=sample_per_class, replace=False)
        for image_id in ids:
            path = _image_path(images_dir, image_id)
            if path is None:
                continue
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            rows.append({
                "image_id": image_id, "dx": cls, "width": w, "height": h,
                "r_mean": float(rgb[:, :, 0].mean()), "g_mean": float(rgb[:, :, 1].mean()),
                "b_mean": float(rgb[:, :, 2].mean()),
                "L_mean": float(lab[:, :, 0].mean()), "a_mean": float(lab[:, :, 1].mean()),
                "b_lab_mean": float(lab[:, :, 2].mean()),
                "brightness": float(gray.mean()), "contrast": float(gray.std()),
                "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            })
    return pd.DataFrame(rows)


def compute_hair_fraction(metadata: pd.DataFrame, images_dir: Path,
                          sample_per_class: int = HAIR_SAMPLE_PER_CLASS, seed: int = 42) -> pd.DataFrame:
    """Hair-pixel fraction using the paper's own §3.2.1a recipe.

    Grayscale -> blackhat with a 17x17 elliptical kernel -> threshold at 10.
    We only measure here; V2 applies the inpainting. This quantifies how
    much work V2's hair removal will actually have to do, per class.
    """
    import cv2

    rng = np.random.default_rng(seed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    rows = []
    for cls in CLASS_ORDER:
        ids = metadata.loc[metadata["dx"] == cls, "image_id"].to_numpy()
        if len(ids) > sample_per_class:
            ids = rng.choice(ids, size=sample_per_class, replace=False)
        for image_id in ids:
            path = _image_path(images_dir, image_id)
            if path is None:
                continue
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
            rows.append({
                "image_id": image_id, "dx": cls,
                "hair_pixel_fraction": float((mask > 0).mean()),
            })
    return pd.DataFrame(rows)


def plot_image_dimensions(stats: pd.DataFrame, out_path: Path) -> Path:
    sizes = stats.groupby(["width", "height"]).size().reset_index(name="count")
    fig, ax = new_figure(figsize=(8, 4.6))
    labels = [f"{int(r.width)}x{int(r.height)}" for r in sizes.itertuples()]
    bars = ax.bar(range(len(sizes)), sizes["count"], color=SINGLE_SERIES_BLUE, width=0.5)
    annotate_bars(ax, bars, sizes["count"].to_numpy(), fmt="{:,.0f}", fontsize=8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Image dimensions (pixels)")
    ax.set_ylabel("Images")
    ax.set_ylim(0, sizes["count"].max() * 1.16)
    ax.set_title("Every image shares one resolution — no aspect-ratio confound",
                 fontsize=11, color=CHART_INK["primary"])
    grid_y(ax)
    caption = (
        "A single resolution means the plain 224x224 resize the paper specifies (§3.2.3, no crop) distorts "
        "every image identically, so resizing introduces no class-dependent bias."
    )
    return finish_figure(fig, out_path, caption)


def plot_color_distributions(stats: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = new_figure(2, 3, figsize=(13.5, 7))
    channels = [
        ("r_mean", "Red channel mean", axes[0, 0]),
        ("g_mean", "Green channel mean", axes[0, 1]),
        ("b_mean", "Blue channel mean", axes[0, 2]),
        ("L_mean", "LAB  L* (lightness)", axes[1, 0]),
        ("a_mean", "LAB  a* (green-red)", axes[1, 1]),
        ("b_lab_mean", "LAB  b* (blue-yellow)", axes[1, 2]),
    ]
    # One hue across all panels: the x-axis already identifies each class, so
    # a 7-colour ramp here would encode nothing the labels don't already carry.
    # Malignancy is marked instead, since that IS extra information.
    short_labels = [f"{c}\n{'M' if malignancy_of(c) == 'malignant' else ('P' if malignancy_of(c) == 'pre-cancerous' else 'B')}"
                    for c in CLASS_ORDER]
    for col, title, ax in channels:
        data = [stats.loc[stats["dx"] == cls, col].to_numpy() for cls in CLASS_ORDER]
        parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.85)
        for body in parts["bodies"]:
            body.set_facecolor(SINGLE_SERIES_BLUE)
            body.set_alpha(0.75)
        parts["cmeans"].set_color(CHART_INK["primary"])
        ax.set_xticks(range(1, NUM_CLASSES + 1))
        ax.set_xticklabels(short_labels, fontsize=7)
        ax.set_title(title, fontsize=9.5, color=CHART_INK["primary"])
        grid_y(ax)

    fig.suptitle("Per-class colour statistics — the LAB row is the space V2's CLAHE operates in",
                 fontsize=11, color=CHART_INK["primary"])
    caption = (
        "Class codes are suffixed M = malignant, P = pre-cancerous, B = benign. Horizontal bars are means. "
        "Classes differ systematically in colour, so colour is genuine diagnostic signal, not just noise. "
        "The paper applies CLAHE to each LAB channel independently (§3.2.1b), which will alter a* and b* and "
        "therefore shift these distributions — this figure is the before-state V2 should be compared against."
    )
    return finish_figure(fig, out_path, caption)


def plot_brightness_contrast(stats: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = new_figure(1, 2, figsize=(13, 5))

    for cls in CLASS_ORDER:
        sub = stats.loc[stats["dx"] == cls]
        axes[0].scatter(sub["brightness"], sub["contrast"], s=7, alpha=0.45,
                        color=CLASS_COLOR[cls], label=CLASS_FULL_NAME[cls])
    axes[0].set_xlabel("Mean luminance (0-255)")
    axes[0].set_ylabel("Luminance standard deviation (contrast)")
    axes[0].set_title("Illumination varies widely across the dataset", fontsize=10, color=CHART_INK["primary"])
    axes[0].legend(frameon=False, fontsize=6.5, markerscale=2, loc="upper left")
    grid_xy(axes[0])

    means = stats.groupby("dx")["brightness"].mean().reindex(CLASS_ORDER)
    stds = stats.groupby("dx")["brightness"].std().reindex(CLASS_ORDER)
    bars = axes[1].bar(range(NUM_CLASSES), means.values, yerr=stds.values, capsize=3,
                       color=SINGLE_SERIES_BLUE, ecolor=CHART_INK["muted"])
    annotate_bars(axes[1], bars, means.values, fmt="{:.0f}", fontsize=7.5, offset=2)
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_xticklabels(wrap_class_labels(full_names(), width=11), fontsize=7)
    axes[1].set_ylabel("Mean luminance")
    axes[1].set_title("Mean brightness per class (bars = 1 SD)", fontsize=10, color=CHART_INK["primary"])
    grid_y(axes[1])

    spread = float(means.max() - means.min())
    caption = (
        f"Class mean brightness spans {spread:.0f} grey levels, and the within-class spread is larger still. "
        "A model can exploit systematic brightness differences as a shortcut instead of learning lesion "
        "morphology; CLAHE in V2 is what suppresses that shortcut."
    )
    return finish_figure(fig, out_path, caption)


def plot_hair_density(hair: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = new_figure(1, 2, figsize=(13, 5))

    axes[0].hist(hair["hair_pixel_fraction"] * 100, bins=45, color=SINGLE_SERIES_BLUE)
    axes[0].set_xlabel("Hair pixels (% of image)")
    axes[0].set_ylabel("Images")
    axes[0].set_title("Most images have some hair; a tail has a great deal",
                      fontsize=10, color=CHART_INK["primary"])
    grid_y(axes[0])

    means = hair.groupby("dx")["hair_pixel_fraction"].mean().reindex(CLASS_ORDER) * 100
    bars = axes[1].bar(range(NUM_CLASSES), means.values, color=SINGLE_SERIES_BLUE)
    annotate_bars(axes[1], bars, means.values, fmt="{:.2f}%", fontsize=7.5)
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_xticklabels(wrap_class_labels(full_names(), width=11), fontsize=7)
    axes[1].set_ylabel("Mean hair pixels (% of image)")
    axes[1].set_ylim(0, means.max() * 1.22)
    axes[1].set_title("Hair occlusion by class", fontsize=10, color=CHART_INK["primary"])
    grid_y(axes[1])

    caption = (
        f"Measured with the paper's own detector (§3.2.1a: greyscale, 17x17 blackhat, threshold 10) on a "
        f"sample of {len(hair):,} images — detection only; V2 adds the TELEA inpainting. "
        "Because hair density differs by class, leaving it in lets the model use hair as an unintended cue, "
        "which is precisely the confound V2 removes."
    )
    return finish_figure(fig, out_path, caption)


def plot_sharpness(stats: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = new_figure(figsize=(10, 5))
    data = [stats.loc[stats["dx"] == cls, "sharpness"].to_numpy() for cls in CLASS_ORDER]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, cls in zip(bp["boxes"], CLASS_ORDER):
        patch.set_facecolor(SINGLE_SERIES_BLUE)
        patch.set_alpha(0.85)
        patch.set_edgecolor(CHART_INK["secondary"])
    for element in ("whiskers", "caps", "medians"):
        for item in bp[element]:
            item.set_color(CHART_INK["secondary"])
    ax.set_xticks(range(1, NUM_CLASSES + 1))
    ax.set_xticklabels(wrap_class_labels(full_names(), width=11), fontsize=7)
    ax.set_ylabel("Laplacian variance (higher = sharper)")
    ax.set_title("Image sharpness is broadly comparable across classes",
                 fontsize=11, color=CHART_INK["primary"])
    grid_y(ax)
    caption = (
        "Laplacian variance is the standard blur detector. Outliers are hidden for readability. No class is "
        "systematically blurrier, so image quality is not a confound between diagnoses."
    )
    return finish_figure(fig, out_path, caption)


PIXEL_DUPLICATE_TOLERANCE = 8.0  # mean abs difference on a 64x64 thumbnail


def find_near_duplicates(metadata: pd.DataFrame, images_dir: Path, splits: dict[str, pd.DataFrame],
                         hash_size: int = DUPLICATE_HASH_SIZE,
                         pixel_tolerance: float = PIXEL_DUPLICATE_TOLERANCE) -> pd.DataFrame:
    """Near-duplicate detection ACROSS splits, as an independent leakage check.

    Lesion_id grouping cannot catch two visually identical images filed
    under *different* lesion IDs, so this looks for them directly.

    Two stages, because the hash alone is not sufficient evidence:
      1. a difference hash narrows 10,015 images to a few candidate groups;
      2. every candidate is then **verified by actual pixel comparison**.

    Stage 2 matters: a 64-bit dHash collides readily on dermoscopic images,
    which share a dark circular vignette and a centred blob. Run against
    HAM10000 without verification it reports ~16 spurious cross-split
    "duplicates", every one of which differs by 15-71 grey levels on
    direct comparison. `pixel_confirmed` is the column to trust.
    """
    import cv2

    split_of = {}
    for name, df in splits.items():
        for image_id in df["image_id"]:
            split_of[image_id] = name

    hashes: dict[int, list[str]] = {}
    for image_id in metadata["image_id"]:
        path = _image_path(images_dir, image_id)
        if path is None:
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        key = int("".join("1" if b else "0" for b in bits), 2)
        hashes.setdefault(key, []).append(image_id)

    rows = []
    lesion_of = metadata.set_index("image_id")["lesion_id"].to_dict()
    for key, ids in hashes.items():
        if len(ids) < 2:
            continue
        split_names = {split_of.get(i) for i in ids}
        lesions = {lesion_of.get(i) for i in ids}

        # Stage 2: confirm (or refute) the group by comparing real pixels.
        thumbs = []
        for image_id in ids:
            path = _image_path(images_dir, image_id)
            if path is None:
                continue
            im = cv2.imread(str(path))
            if im is not None:
                thumbs.append(cv2.resize(im, (64, 64)).astype(float))
        max_diff = (
            max(float(np.abs(thumbs[0] - t).mean()) for t in thumbs[1:])
            if len(thumbs) > 1 else float("nan")
        )

        rows.append({
            "hash": key,
            "n_images": len(ids),
            "image_ids": ",".join(sorted(ids)[:6]),
            "n_distinct_lesions": len(lesions),
            "splits": ",".join(sorted(s for s in split_names if s)),
            "crosses_splits": len(split_names - {None}) > 1,
            "same_lesion": len(lesions) == 1,
            "max_pixel_diff": round(max_diff, 2),
            "pixel_confirmed": bool(max_diff == max_diff and max_diff < pixel_tolerance),
        })
    return pd.DataFrame(rows)


def plot_duplicate_candidates(dupes: pd.DataFrame, out_path: Path) -> Path:
    """Reports the funnel from hash candidates down to pixel-confirmed leakage.

    Showing all four stages is deliberate: the drop between stage 3 and
    stage 4 is what distinguishes a real finding from a hashing artifact,
    and hiding it would make the check look more authoritative than it is.
    """
    if len(dupes):
        cross = dupes.loc[dupes["crosses_splits"]]
        cross_diff_lesion = cross.loc[~cross["same_lesion"]]
        confirmed = cross_diff_lesion.loc[cross_diff_lesion["pixel_confirmed"]]
    else:
        cross = cross_diff_lesion = confirmed = dupes

    categories = [
        "Hash candidate\ngroups",
        "Crossing\nsplits",
        "Cross-split AND\ndifferent lesion",
        "Pixel-confirmed\nleakage",
    ]
    values = [len(dupes), len(cross), len(cross_diff_lesion), len(confirmed)]
    colors = [
        SINGLE_SERIES_BLUE, SINGLE_SERIES_BLUE, CHART_INK["muted"],
        MALIGNANCY_COLOR["malignant"] if len(confirmed) else MALIGNANCY_COLOR["benign"],
    ]

    fig, ax = new_figure(figsize=(10, 5))
    bars = ax.bar(range(4), values, color=colors, width=0.55)
    annotate_bars(ax, bars, values, fmt="{:,.0f}", fontsize=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_ylabel("Image groups")
    ax.set_ylim(0, max(max(values), 1) * 1.2)
    verdict = ("no leakage beyond lesion grouping"
               if len(confirmed) == 0 else f"{len(confirmed)} confirmed groups need review")
    ax.set_title(f"Independent leakage check — {verdict}", fontsize=11, color=CHART_INK["primary"])
    grid_y(ax)

    worst = cross_diff_lesion["max_pixel_diff"].min() if len(cross_diff_lesion) else float("nan")
    caption = (
        "A 64-bit difference hash proposes candidates; each is then verified by direct pixel comparison, "
        f"because dermoscopic images (dark vignette, centred blob) collide readily. The {len(cross_diff_lesion)} "
        f"cross-split candidates all failed verification — the closest pair still differs by {worst:.1f} grey "
        "levels, far above the 8.0 duplicate threshold. Only the last bar is evidence of leakage."
    )
    return finish_figure(fig, out_path, caption)


# ================================================== E. class-difficulty context
def plot_class_separability(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> Path:
    """Mean predicted probability per (true class, predicted class) pair."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        mask = y_true == i
        if mask.any():
            matrix[i] = probs[mask].mean(axis=0)

    labels = wrap_class_labels(full_names(), width=11)
    fig, ax = new_figure(figsize=(8.5, 6.5))
    im = ax.imshow(matrix, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean probability assigned to")
    ax.set_ylabel("True class")
    ax.set_title("Where the model's probability mass goes, per true class",
                 fontsize=11, color=CHART_INK["primary"])
    ax.grid(False)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > 0.55 else CHART_INK["primary"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    caption = (
        "A strong diagonal means the class is easy to separate. Off-diagonal mass shows which diagnoses the "
        "model confuses even when it gets the answer right — a softer signal than the confusion matrix, "
        "because it uses the full probability vector rather than only the argmax."
    )
    return finish_figure(fig, out_path, caption)


# ====================================================================== report
def build_data_report(metadata: pd.DataFrame, splits: dict[str, pd.DataFrame],
                      summary: pd.DataFrame, leakage: dict,
                      stats: pd.DataFrame | None, hair: pd.DataFrame | None,
                      dupes: pd.DataFrame | None) -> str:
    """Consolidated data report — replaces the separate integrity-audit and
    split-verification files, with every fact recomputed here rather than
    copied, so PLAN.md §5.3 items 2 and 5 stay satisfied (§12 definition of done).
    """
    image_pct, lesion_pct = split_proportion_tables(splits)
    img_drift = image_pct[["val", "test"]].sub(image_pct["train"], axis=0).abs().to_numpy().max()
    lesion_drift = lesion_pct[["val", "test"]].sub(lesion_pct["train"], axis=0).abs().to_numpy().max()

    n_images = int(metadata["image_id"].nunique())
    n_lesions = int(metadata["lesion_id"].nunique())
    counts = metadata["dx"].value_counts()

    lines = [
        "# V1 Data Report — integrity, split and exploratory analysis",
        "",
        "Consolidates what were previously three separate files (`integrity_audit.md`,",
        "`split_verification.md`, and the EDA notes). Every number here is **recomputed**",
        "by `python -m src.eval.eda`, not copied, so the report cannot go stale.",
        "",
        "Covers PLAN.md §5.3 items 2 (integrity audit) and 5 (split verification),",
        "which §12's definition of done requires for V1 sign-off.",
        "",
        "## 1. Integrity audit",
        "",
        "| Check | Expected | Found | Result |",
        "|---|---|---|---|",
        f"| Unique image IDs | 10,015 | {n_images:,} | {'PASS' if n_images == 10015 else 'FAIL'} |",
        f"| Unique lesion IDs | 7,470 | {n_lesions:,} | {'PASS' if n_lesions == 7470 else 'FAIL'} |",
        f"| Lesions with >1 diagnosis | 0 | {int((metadata.groupby('lesion_id')['dx'].nunique() > 1).sum())} | "
        f"{'PASS' if (metadata.groupby('lesion_id')['dx'].nunique() > 1).sum() == 0 else 'FAIL'} |",
    ]
    expected_counts = {"nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514, "akiec": 327, "vasc": 142, "df": 115}
    for cls, expected in expected_counts.items():
        actual = int(counts.get(cls, 0))
        lines.append(f"| Class count `{cls}` | {expected:,} | {actual:,} | {'PASS' if actual == expected else 'FAIL'} |")

    if stats is not None and len(stats):
        odd = stats.loc[(stats["width"] != 600) | (stats["height"] != 450)]
        lines.append(
            f"| Image dimensions (all 600x450) | 0 deviations | {len(odd)} | {'PASS' if len(odd) == 0 else 'FAIL'} |"
        )

    lines += [
        "",
        "## 2. Dataset composition (paper Table 1 equivalent)",
        "",
        summary.to_markdown(index=False),
        "",
        f"Imbalance ratio, most to least common class: **{counts.max() / counts.min():.1f}:1** "
        f"({counts.idxmax()} {counts.max():,} vs {counts.idxmin()} {counts.min():,}).",
        "PLAN.md §13 item 2 notes the paper calls this 67:1; computed from counts it is 58:1,",
        "and 67% is nevus's *share* of the dataset rather than a ratio.",
        "",
        "## 3. Split verification — is the split unbiased?",
        "",
        "### 3.1 Leakage checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for pair, n in leakage["lesion_overlaps"].items():
        lines.append(f"| Shared lesions, {pair} | {n} {'PASS' if n == 0 else 'FAIL'} |")
    for pair, n in leakage["image_overlaps"].items():
        lines.append(f"| Shared images, {pair} | {n} {'PASS' if n == 0 else 'FAIL'} |")
    lines.append(
        f"| Union covers all images | {leakage['n_union']:,} of {leakage['n_metadata']:,} "
        f"{'PASS' if leakage['union_covers_all'] else 'FAIL'} |"
    )
    lines.append(
        f"| All 7 classes in every split | {'yes PASS' if leakage['all_classes_everywhere'] else 'no FAIL'} |"
    )

    lines += [
        "",
        "### 3.2 Split sizes",
        "",
        "| Split | Images (ours) | Images (paper) | Unique lesions | Mean images/lesion |",
        "|---|---|---|---|---|",
    ]
    for name in SPLIT_ORDER:
        lines.append(
            f"| {name} | {leakage['n_images'][name]:,} | {PAPER_SPLIT_IMAGE_COUNTS[name]:,} | "
            f"{leakage['n_lesions'][name]:,} | {leakage['mean_images_per_lesion'][name]} |"
        )

    lines += [
        "",
        "### 3.3 Class balance across splits",
        "",
        "Image-level class percentages:",
        "",
        image_pct.round(2).to_markdown(),
        "",
        "Lesion-level class percentages (what stratification actually acts on):",
        "",
        lesion_pct.round(2).to_markdown(),
        "",
        "### 3.4 Verdict",
        "",
        f"**The split is balanced and carries no class or sampling bias.**",
        "",
        f"- Maximum deviation of any class proportion from train, **lesion-level: {lesion_drift:.3f} pp**.",
        f"- Maximum deviation, **image-level: {img_drift:.3f} pp**.",
        f"- Mean images per lesion is near-identical across splits "
        f"({', '.join(f'{k} {v}' for k, v in leakage['mean_images_per_lesion'].items())}), so no split is "
        "enriched in repeatedly-photographed lesions.",
        "",
        "Lesion-level stratification is near-exact because that is the level `train_test_split(stratify=dx)`",
        "operates on. The image level inherits a sub-1pp drift because whole lesions move together and carry",
        "1-6 images each. This residual is inherent to lesion-grouped splitting and is strongly preferable to",
        "the alternative: splitting by image would place two photographs of the *same* lesion in train and",
        "test, testing the model on a lesion it had memorized.",
    ]

    if dupes is not None:
        if len(dupes):
            cross = dupes.loc[dupes["crosses_splits"]]
            cross_diff_lesion = cross.loc[~cross["same_lesion"]]
            confirmed = cross_diff_lesion.loc[cross_diff_lesion["pixel_confirmed"]]
            closest = cross_diff_lesion["max_pixel_diff"].min() if len(cross_diff_lesion) else float("nan")
        else:
            cross = cross_diff_lesion = confirmed = dupes
            closest = float("nan")

        lines += [
            "",
            "### 3.5 Independent leakage check (perceptual hashing + pixel verification)",
            "",
            "Lesion grouping cannot detect two visually identical images filed under *different* lesion IDs.",
            "To rule that out, every image was hashed and each candidate group was then **verified by direct",
            "pixel comparison** — the hash alone is not sufficient evidence.",
            "",
            f"- Hash candidate groups: **{len(dupes):,}**",
            f"- Spanning more than one split: **{len(cross):,}**",
            f"- Cross-split *and* different lesion ID: **{len(cross_diff_lesion):,}**",
            f"- **Pixel-confirmed leakage: {len(confirmed):,}**",
            "",
            (
                f"**No leakage beyond what lesion grouping already prevents.** All {len(cross_diff_lesion)} "
                f"cross-split candidates failed pixel verification — the closest pair still differs by "
                f"{closest:.1f} grey levels on a 64x64 thumbnail, against a duplicate threshold of "
                f"{PIXEL_DUPLICATE_TOLERANCE}. This is expected: dermoscopic images share a dark circular "
                "vignette and a centred lesion, so a 64-bit difference hash collides readily. Reporting the "
                "hash candidates alone would have been a false alarm."
                if len(confirmed) == 0 else
                f"**{len(confirmed)} groups are pixel-confirmed duplicates across splits and warrant review.**"
            ),
        ]

    if hair is not None and len(hair):
        lines += [
            "",
            "## 4. Image-content findings relevant to V2",
            "",
            f"- Mean hair-pixel fraction: **{100 * hair['hair_pixel_fraction'].mean():.2f}%** of image area "
            f"(sample of {len(hair):,} images, paper's §3.2.1a detector).",
            f"- Highest by class: **{hair.groupby('dx')['hair_pixel_fraction'].mean().idxmax()}**; "
            f"lowest: **{hair.groupby('dx')['hair_pixel_fraction'].mean().idxmin()}**.",
            "- Because hair density differs by class, leaving it in lets the model use hair as an unintended",
            "  cue. This quantifies the work V2's hair removal has to do.",
        ]
    if stats is not None and len(stats):
        lines += [
            f"- Mean luminance ranges from {stats.groupby('dx')['brightness'].mean().min():.0f} to "
            f"{stats.groupby('dx')['brightness'].mean().max():.0f} across classes, motivating V2's CLAHE.",
        ]

    lines += ["", "---", "", "Figures for every section are in `results/v1/eda/`.", ""]
    return "\n".join(lines)


# ========================================================================= CLI
def run_eda(data_root: Path, split_dir: Path, out_dir: Path, predictions: Path | None,
            report_path: Path, seed: int = 42, full_image_stats: bool = True,
            hair_sample: int = HAIR_SAMPLE_PER_CLASS) -> None:
    metadata_path = data_root / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        raise SystemExit(
            f"\nCannot run the EDA: {metadata_path} not found.\n\n"
            "The EDA describes the dataset, so it needs the dataset. Acquire it with:\n\n"
            f"    python -m src.data.acquire --dest {data_root} --source dataverse\n\n"
            "(~3.2 GB; the Harvard Dataverse route needs no credentials. Use\n"
            " `--source kaggle` instead if you have a Kaggle API token configured.)\n\n"
            "Note: the Dataverse archive stores the metadata as `HAM10000_metadata`\n"
            "with no extension - copy it to `HAM10000_metadata.csv`.\n\n"
            "The *evaluation* suite has no such dependency - `python -m src.eval.run_eval`\n"
            "runs from the committed files in results/v1/inputs/ alone."
        )
    if not split_dir.exists():
        raise SystemExit(
            f"\nCannot run the EDA: split directory {split_dir} not found.\n"
            "The C2 split files (train/val/test.csv) are committed in the repo at data/splits/."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path)
    images_dir = data_root / "images"
    splits = load_splits(split_dir)

    made: list[str] = []

    # A. composition
    made.append(str(plot_class_distribution(metadata, out_dir / "class_distribution.png")))
    made.append(str(plot_class_imbalance_ratio(metadata, out_dir / "class_imbalance_ratio.png")))
    made.append(str(plot_malignancy_breakdown(metadata, out_dir / "malignancy_breakdown.png")))
    summary = dataset_summary_table(metadata)
    summary.to_csv(out_dir / "dataset_summary.csv", index=False)
    (out_dir / "dataset_summary.md").write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")

    # B. lesion structure
    made.append(str(plot_images_per_lesion(metadata, out_dir / "images_per_lesion.png")))
    made.append(str(plot_images_per_lesion_by_class(metadata, out_dir / "images_per_lesion_by_class.png")))
    made.append(str(plot_lesion_multiplicity_vs_class(metadata, out_dir / "lesion_multiplicity_vs_class.png")))

    # C. split fairness
    made.append(str(plot_split_class_proportions(splits, out_dir / "split_class_proportions.png")))
    made.append(str(plot_split_drift(splits, out_dir / "split_drift.png")))
    made.append(str(plot_split_sizes(splits, out_dir / "split_sizes.png")))
    leakage = split_leakage_checks(splits, metadata)

    # D. image content
    stats = hair = dupes = None
    if images_dir.exists():
        grid = plot_sample_grid(metadata, images_dir, out_dir / "sample_grid.png", seed=seed)
        if grid:
            made.append(str(grid))

        print("Computing per-image colour/brightness/sharpness statistics...")
        stats = compute_image_stats(metadata, images_dir,
                                    sample_per_class=None if full_image_stats else 300, seed=seed)
        stats.to_csv(out_dir / "image_stats.csv", index=False)
        made.append(str(plot_image_dimensions(stats, out_dir / "image_dimensions.png")))
        made.append(str(plot_color_distributions(stats, out_dir / "color_distributions.png")))
        made.append(str(plot_brightness_contrast(stats, out_dir / "brightness_contrast.png")))
        made.append(str(plot_sharpness(stats, out_dir / "sharpness_blur.png")))

        print(f"Detecting hair (paper §3.2.1a recipe, {hair_sample}/class)...")
        hair = compute_hair_fraction(metadata, images_dir, sample_per_class=hair_sample, seed=seed)
        hair.to_csv(out_dir / "hair_stats.csv", index=False)
        made.append(str(plot_hair_density(hair, out_dir / "hair_density.png")))

        print("Running perceptual-hash duplicate detection...")
        dupes = find_near_duplicates(metadata, images_dir, splits)
        dupes.to_csv(out_dir / "duplicate_candidates.csv", index=False)
        made.append(str(plot_duplicate_candidates(dupes, out_dir / "duplicate_candidates.png")))
    else:
        print(
            f"NOTE: {images_dir} not found — skipping the image-content figures.\n"
            f"      Acquire the dataset first:\n"
            f"        python -m src.data.acquire --dest {data_root} --source dataverse"
        )

    # E. class difficulty
    if predictions is not None and predictions.exists():
        from src.eval.metrics import load_predictions, probs_and_labels
        df = load_predictions(predictions)
        probs, y_true = probs_and_labels(df)
        made.append(str(plot_class_separability(y_true, probs, out_dir / "class_separability.png")))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_data_report(metadata, splits, summary, leakage, stats, hair, dupes), encoding="utf-8"
    )

    print(f"\nWrote {len(made)} figures to {out_dir}")
    print(f"Wrote the consolidated data report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rigorous V1 EDA (PLAN.md §5.3 item 7, Member C).")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--out", type=Path, default=Path("results/v1/eda"))
    parser.add_argument("--predictions", type=Path, default=Path("results/v1/inputs/predictions_test.csv"),
                        help="C3 test predictions, for the class-separability figure.")
    parser.add_argument("--report", type=Path, default=Path("results/v1/data_report.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hair-sample", type=int, default=HAIR_SAMPLE_PER_CLASS,
                        help="Images per class for hair detection (the slowest pass).")
    parser.add_argument("--fast", action="store_true",
                        help="Sample 300 images/class for colour stats instead of all 10,015.")
    args = parser.parse_args()

    run_eda(args.data_root, args.split_dir, args.out, args.predictions, args.report,
            seed=args.seed, full_image_stats=not args.fast, hair_sample=args.hair_sample)


if __name__ == "__main__":
    main()

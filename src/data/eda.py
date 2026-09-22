"""EDA figures and dataset summary table (PLAN.md §5.3.7).

Produces, under `--out`:
  - class_distribution.png   (counts and %)
  - images_per_lesion_hist.png
  - sample_grid.png          (4 images per class; skipped if images aren't on disk)
  - dataset_summary.csv      (paper Table 1 equivalent)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_COLOR = "#3B6FA0"  # single brand-neutral blue; consistent across figures


def plot_class_distribution(metadata: pd.DataFrame, out_path: Path) -> None:
    counts = metadata["dx"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    pct = 100 * counts / counts.sum()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(CLASS_ORDER, counts.values, color=CLASS_COLOR)
    ax.set_ylabel("Image count")
    ax.set_xlabel("Class (dx)")
    ax.set_title("HAM10000 class distribution")
    for bar, n, p in zip(bars, counts.values, pct.values):
        ax.annotate(f"{n}\n({p:.1f}%)", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_images_per_lesion(metadata: pd.DataFrame, out_path: Path) -> None:
    per_lesion = metadata.groupby("lesion_id").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    max_n = int(per_lesion.max())
    ax.hist(per_lesion.values, bins=range(1, max_n + 2), align="left", color=CLASS_COLOR, edgecolor="white")
    ax.set_xlabel("Images per lesion")
    ax.set_ylabel("Number of lesions")
    ax.set_title("Distribution of images per lesion")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_grid(metadata: pd.DataFrame, images_dir: Path, out_path: Path, n_per_class: int = 4) -> bool:
    """Returns False (and writes nothing) if the images directory has no
    usable images, so callers can skip this figure gracefully before real
    data is acquired.
    """
    if not images_dir.exists():
        return False

    fig, axes = plt.subplots(len(CLASS_ORDER), n_per_class, figsize=(n_per_class * 2, len(CLASS_ORDER) * 2))
    any_found = False
    for row, cls in enumerate(CLASS_ORDER):
        sample_ids = metadata.loc[metadata["dx"] == cls, "image_id"].head(n_per_class)
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(sample_ids):
                image_id = sample_ids.iloc[col]
                candidates = [images_dir / f"{image_id}{ext}" for ext in (".jpg", ".jpeg", ".png")]
                found = next((c for c in candidates if c.exists()), None)
                if found is not None:
                    with Image.open(found) as im:
                        ax.imshow(im)
                    any_found = True
            if col == 0:
                ax.text(-0.15, 0.5, cls, transform=ax.transAxes, ha="right", va="center", fontsize=10)
    fig.suptitle("Sample images per class (4 each)")
    fig.tight_layout()
    if any_found:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return any_found


def dataset_summary_table(metadata: pd.DataFrame) -> pd.DataFrame:
    counts = metadata["dx"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    pct = 100 * counts / counts.sum()
    n_lesions_per_class = metadata.groupby("dx")["lesion_id"].nunique().reindex(CLASS_ORDER).fillna(0).astype(int)
    return pd.DataFrame({
        "dx": CLASS_ORDER,
        "n_images": counts.values,
        "pct_images": pct.round(2).values,
        "n_unique_lesions": n_lesions_per_class.values,
    })


def run_eda(data_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
    images_dir = data_root / "images"

    plot_class_distribution(metadata, out_dir / "class_distribution.png")
    plot_images_per_lesion(metadata, out_dir / "images_per_lesion_hist.png")
    made_grid = plot_sample_grid(metadata, images_dir, out_dir / "sample_grid.png")
    if not made_grid:
        print(f"NOTE: skipped sample_grid.png — no images found under {images_dir}")

    summary = dataset_summary_table(metadata)
    summary.to_csv(out_dir / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V1 EDA figures (PLAN.md §5.3.7).")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--splits", type=Path, default=None, help="Unused for now; reserved for split-aware EDA.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run_eda(args.data_root, args.out)


if __name__ == "__main__":
    main()

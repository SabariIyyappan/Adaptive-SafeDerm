"""Architecture figure, equivalent to paper Figure 1 (PLAN.md §5.3.9).

A block diagram of EfficientNet-B0 (ImageNet-pretrained backbone, features
kept, original classifier removed) feeding the paper's custom head:
Linear(1280->256) -> ReLU -> Dropout(0.3) -> Linear(256->7).

Needs no dataset — safe to generate immediately.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

BLOCKS = [
    ("Input\n224×224×3", "#E8EEF5", "#3B6FA0"),
    ("EfficientNet-B0\nbackbone\n(IMAGENET1K_V1)\nconv stack + GAP", "#D8E4F0", "#3B6FA0"),
    ("Features\n1280-dim", "#E8EEF5", "#3B6FA0"),
    ("Linear\n1280 → 256", "#FBEfe0", "#C77B22"),
    ("ReLU", "#FBEfe0", "#C77B22"),
    ("Dropout\np = 0.3", "#FBEfe0", "#C77B22"),
    ("Linear\n256 → 7", "#FBEfe0", "#C77B22"),
    ("Softmax\n7 classes", "#E8F0E8", "#3E7A3E"),
]


def draw_architecture(out_path: Path) -> None:
    n = len(BLOCKS)
    box_w, box_h, gap = 1.5, 1.0, 0.55
    total_w = n * box_w + (n - 1) * gap

    fig, ax = plt.subplots(figsize=(total_w * 0.9 + 1, 2.6))
    ax.set_xlim(-0.5, total_w + 0.5)
    ax.set_ylim(-1.2, box_h + 0.8)
    ax.axis("off")

    x = 0.0
    centers = []
    for label, facecolor, edgecolor in BLOCKS:
        box = FancyBboxPatch(
            (x, 0), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor,
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, box_h / 2, label, ha="center", va="center", fontsize=8.5)
        centers.append(x + box_w)
        x += box_w + gap

    for cx in centers[:-1]:
        ax.add_patch(FancyArrow(cx, box_h / 2, gap - 0.1, 0, width=0.02, head_width=0.14, head_length=0.1,
                                 length_includes_head=True, color="#555555"))

    # Group brackets: "backbone (frozen structure, trainable weights)" vs "head (all trainable, new)"
    ax.annotate("", xy=(box_w + gap + box_w + gap + box_w, -0.35), xytext=(0, -0.35),
                arrowprops=dict(arrowstyle="-", color="#3B6FA0", lw=1.2))
    ax.text((box_w + gap + box_w + gap + box_w) / 2, -0.55, "backbone", ha="center", fontsize=8, color="#3B6FA0")

    head_start = 3 * box_w + 3 * gap
    head_end = total_w
    ax.annotate("", xy=(head_end, -0.35), xytext=(head_start, -0.35),
                arrowprops=dict(arrowstyle="-", color="#C77B22", lw=1.2))
    ax.text((head_start + head_end) / 2, -0.55, "custom head (§3.3.2, Eqs. 4–5)", ha="center", fontsize=8, color="#C77B22")

    ax.set_title("EfficientNet-B0 + dropout head — architecture (paper Figure 1 equivalent, PLAN.md §5.3.9)",
                 fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the V1 architecture figure (PLAN.md §5.3.9).")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    draw_architecture(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

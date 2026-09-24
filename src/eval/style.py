"""Shared figure styling (PLAN.md §5.3 item 4).

Every figure in `src/eval/` goes through `new_figure` / `finish_figure` so
the whole deck is visually consistent with Member A's `src/data/eda.py`
figures (Agg backend, dpi 150, top/right spines hidden).

The design rules these helpers enforce come from the plan's Step 2:
  - a title that states the takeaway, not just the variable names;
  - a caption line under every figure, so a slide reader needs no paper;
  - full diagnostic class names on axes, never bare codes like `mel`;
  - recessive solid hairline grid; no dashed gridlines.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.classes import CHART_INK

DPI = 150


def new_figure(*args, **kwargs):
    """`plt.subplots` with the house style applied."""
    fig, axes = plt.subplots(*args, **kwargs)
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        _style_axes(ax)
    return fig, axes


def _style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(CHART_INK["muted"])
    ax.tick_params(colors=CHART_INK["secondary"], labelsize=8)
    ax.set_facecolor(CHART_INK["surface"])


def grid_y(ax) -> None:
    """Recessive horizontal hairline grid, behind the marks. Solid, never dashed."""
    ax.grid(axis="y", color=CHART_INK["gridline"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def grid_xy(ax) -> None:
    ax.grid(color=CHART_INK["gridline"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def finish_figure(fig, out_path: Path, caption: str | None = None) -> Path:
    """Add the caption, tighten, save at DPI, close. Returns the path."""
    if caption:
        fig.text(
            0.5, 0.005, caption, ha="center", va="bottom",
            fontsize=7.5, color=CHART_INK["secondary"], wrap=True,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=CHART_INK["surface"])
    plt.close(fig)
    return out_path


def wrap_class_labels(names: list[str], width: int = 14) -> list[str]:
    """Break long diagnostic names onto two lines so x-tick labels don't
    collide (`Basal cell carcinoma (bcc)` is too wide for a 7-category axis).
    """
    out = []
    for name in names:
        # Split the '(code)' onto its own line, then wrap the words above it.
        if "(" in name:
            label, code = name.rsplit(" (", 1)
            code = "(" + code
        else:
            label, code = name, ""
        words = label.split()
        lines, current = [], ""
        for w in words:
            candidate = f"{current} {w}".strip()
            if len(candidate) > width and current:
                lines.append(current)
                current = w
            else:
                current = candidate
        if current:
            lines.append(current)
        if code:
            lines.append(code)
        out.append("\n".join(lines))
    return out


def annotate_bars(ax, bars, values, fmt: str = "{:.0f}", fontsize: int = 7, offset: float = 0.0) -> None:
    """Direct-label every bar. Values on bars are the mitigation for the
    palette's sub-3:1 contrast slots (the dataviz relief rule) and mean no
    value is chart-only.
    """
    for bar, val in zip(bars, values):
        if val is None or (isinstance(val, float) and val != val):  # NaN
            continue
        ax.annotate(
            fmt.format(val),
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + offset),
            ha="center", va="bottom", fontsize=fontsize, color=CHART_INK["primary"],
        )

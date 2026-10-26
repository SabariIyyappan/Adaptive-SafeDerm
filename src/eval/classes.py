"""Shared class metadata for the evaluation and EDA suites (PLAN.md §5.3,
Member C). Single source of truth so every figure/table uses the same
class order, full diagnostic names and malignancy grouping — the fix for
figures that showed bare codes like ``mel`` with no explanation.

Reuses `src.data.eda.CLASS_ORDER` and `src.data.labels.EXPECTED_HAM10000_LABEL_MAP`
as the canonical class order/index so this module can never drift from
Member A's C1 label map (PLAN.md §4.2 — we don't edit `src/data/`, only
import from it).
"""
from __future__ import annotations

from src.data.eda import CLASS_ORDER
from src.data.labels import EXPECTED_HAM10000_LABEL_MAP

# Sanity: CLASS_ORDER (used for figures) must agree with the C1 label map
# (used for prediction column indices prob_0..prob_6) on both membership
# and index. If A's label map ever changes, this fails loudly instead of
# silently mislabeling every figure.
assert [EXPECTED_HAM10000_LABEL_MAP[c] for c in CLASS_ORDER] == list(range(7)), (
    "CLASS_ORDER (src.data.eda) is out of sync with the C1 label map "
    "(src.data.labels.EXPECTED_HAM10000_LABEL_MAP)"
)

NUM_CLASSES = 7

# Full diagnostic names, in C1 label-map order (index == label int).
CLASS_FULL_NAME = {
    "akiec": "Actinic keratosis (akiec)",
    "bcc": "Basal cell carcinoma (bcc)",
    "bkl": "Benign keratosis (bkl)",
    "df": "Dermatofibroma (df)",
    "mel": "Melanoma (mel)",
    "nv": "Melanocytic nevus (nv)",
    "vasc": "Vascular lesion (vasc)",
}

# The paper's own clinical grouping (PLAN.md §8.3, Figure 8 caption):
# red = malignant (mel, bcc), yellow = pre-cancerous (akiec), green = benign
# (bkl, df, nv, vasc). akiec (actinic keratosis) is a precursor lesion, not
# yet cancer, hence its own category.
MALIGNANT = {"mel", "bcc"}
PRE_CANCEROUS = {"akiec"}
BENIGN = {"bkl", "df", "nv", "vasc"}


def malignancy_of(cls: str) -> str:
    if cls in MALIGNANT:
        return "malignant"
    if cls in PRE_CANCEROUS:
        return "pre-cancerous"
    if cls in BENIGN:
        return "benign"
    raise ValueError(f"unknown class '{cls}'")


# Status-style colors for the malignancy grouping (dataviz skill's status
# palette, PLAN.md-adjacent — used only as a marker/annotation, never as
# the sole channel, per the figure design rules in the plan).
MALIGNANCY_COLOR = {
    "malignant": "#d03b3b",     # critical (status palette)
    "pre-cancerous": "#fab219",  # warning
    "benign": "#0ca30c",        # good
}

# Validated categorical palette (dataviz skill, 7-slot order — passed the
# lightness band, chroma floor, CVD-separation and normal-vision-floor
# checks; see the plan's Step 2 for the validator run). Assigned in
# CLASS_ORDER order so color is stable across every figure that uses it.
CATEGORICAL_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
CLASS_COLOR = dict(zip(CLASS_ORDER, CATEGORICAL_PALETTE))

# Single-series blue — used for any bar chart over nominal categories
# (per-class accuracy, sensitivity, etc.), where a 7-hue ramp would be an
# anti-pattern (coloring nominal categories by value/rank).
SINGLE_SERIES_BLUE = "#2a78d6"
SERIES_2 = "#eb6834"  # second series in a 2-series comparison (e.g. sens vs spec)

# Sequential one-hue ramp (confusion matrices, heatmaps) — matplotlib
# colormap name, not a fixed list, since these need continuous interpolation.
SEQUENTIAL_CMAP = "Blues"

CHART_INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "gridline": "#e1e0d9",
    "surface": "#fcfcfb",
}


def full_names(order: list[str] = CLASS_ORDER) -> list[str]:
    return [CLASS_FULL_NAME[c] for c in order]

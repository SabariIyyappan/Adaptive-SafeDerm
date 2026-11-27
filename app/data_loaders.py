"""Shared paths, cached file readers and small UI helpers for the Streamlit app.

Every number the app shows comes from a file read here (or from the live
pipeline). Nothing is hard-coded, so each chart can carry a "Source:" caption
that points back to the file it was built from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.classes import CLASS_FULL_NAME, CLASS_ORDER  # noqa: E402

WB2 = ROOT / "results" / "wb2"
BANK = WB2 / "inputs" / "wb2-mc-v1b0-s42"
APP_DATA = ROOT / "app" / "data"
EXAMPLES_DIR = ROOT / "app" / "assets" / "examples"
CONTROLLER_CONFIG = ROOT / "configs" / "wb2_controller.yaml"

PROB_COLS = [f"prob_{i}" for i in range(len(CLASS_ORDER))]
SUSPICIOUS = ("mel", "bcc")

METHOD_LABEL = {
    "det_T1": "Deterministic T=1",
    "fixed_T10": "Fixed T=10",
    "fixed_T20": "Fixed T=20",
    "fixed_T50": "Fixed T=50",
    "fixed_T50+referral": "Fixed T=50 + entropy referral (20% budget)",
    "fixed_T50+referral@matched": "Fixed T=50 + entropy referral (matched)",
    "variance_adaptive": "Variance-only adaptive + referral",
    "risk_aware_adaptive": "Risk-aware adaptive (proposed)",
}

DISCLAIMER = (
    "**Research prototype, not a diagnosis.** Adaptive SafeDerm is a CMPE 295 "
    "research project on retrospective dermoscopic images. *TRUSTED RESULT* means "
    "the model's own computational checks passed; it never means benign or safe. "
    "The risk tier is a model-risk tier, not a pathology severity."
)


def rel(path: Path) -> str:
    """Repo-relative path with forward slashes, for captions."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def source_caption(*paths: Path, note: str | None = None) -> None:
    text = "Source: " + ", ".join(f"`{rel(p)}`" for p in paths)
    if note:
        text += f" — {note}"
    st.caption(text)


def disclaimer() -> None:
    st.warning(DISCLAIMER, icon="⚠️")


def class_label(code: str) -> str:
    return CLASS_FULL_NAME.get(code, code)


# --------------------------------------------------------------------------- readers

def table_path(name: str, split: str = "test", calibration: str = "calibrated") -> Path:
    return WB2 / calibration / "tables" / f"{name}_{split}.csv"


def decisions_path(split: str = "test", calibration: str = "calibrated") -> Path:
    return WB2 / calibration / "decisions" / f"decisions_{split}.csv.gz"


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def controller_config() -> dict:
    return yaml.safe_load(CONTROLLER_CONFIG.read_text(encoding="utf-8"))


def tau_s() -> float:
    return float(controller_config()["risk_aware_adaptive"]["tau_s"])


@st.cache_data(show_spinner="Loading decisions…")
def decisions(split: str = "test", calibration: str = "calibrated", method: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(decisions_path(split, calibration))
    if method is not None:
        df = df[df["method"] == method].reset_index(drop=True)
    return df


def probs_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df[PROB_COLS].to_numpy(dtype=float), df["y_true"].to_numpy(dtype=int)


def policy_comparison(split: str = "test", calibration: str = "calibrated") -> pd.DataFrame:
    """Method comparison table plus the matched-coverage referral baseline."""
    base = read_csv(table_path("policy_comparison", split, calibration))
    matched_path = WB2 / calibration / "tables" / "matched_coverage.csv"
    if matched_path.exists():
        matched = read_csv(matched_path)
        matched = matched[(matched["split"] == split) & (matched["method"] == "fixed_T50+referral@matched")]
        base = pd.concat([base, matched[[c for c in base.columns if c in matched.columns]]], ignore_index=True)
    return base


@st.cache_data(show_spinner=False)
def examples() -> pd.DataFrame:
    """Sai's bundled example images (app/assets/examples/examples.csv), if present."""
    meta = EXAMPLES_DIR / "examples.csv"
    if not meta.exists():
        return pd.DataFrame(columns=["image_id", "true_class", "why_chosen", "path"])
    df = pd.read_csv(meta)
    paths = []
    for image_id in df["image_id"]:
        hits = sorted(EXAMPLES_DIR.glob(f"{image_id}.*"))
        hits = [h for h in hits if h.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        paths.append(str(hits[0]) if hits else None)
    df["path"] = paths
    return df[df["path"].notna()].reset_index(drop=True)

"""Adaptive SafeDerm — Page 1: Upload & Analyze (the live prototype).

Run:  streamlit run app/streamlit_app.py

The page renders only the implementation.md §2 contract keys returned by
`analyze()`. It uses Soumya's `app/pipeline.py` when it imports, and otherwise
falls back to `app/stub_pipeline.py` behind a visible STUB banner.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from app import data_loaders as D  # noqa: E402
from src.eval.classes import CLASS_COLOR, CLASS_ORDER  # noqa: E402

st.set_page_config(page_title="Adaptive SafeDerm", page_icon="🩺", layout="wide")

REASON_TEXT = {
    "UNSTABLE_CLASS": "the top class kept changing between checkpoints",
    "HIGH_ENTROPY": "the calibrated prediction is too uncertain",
    "SUSPICIOUS_MARGIN": "the suspicious score (mel + bcc) is too close to the threshold",
    "CRITICAL_TIER": "critical model-risk tier (e.g. a benign answer with high suspicious mass)",
    "BUDGET_EXHAUSTED": "reached the 50-pass cap without meeting the stop rule",
}
TIER_COLOR = {"low": "#0ca30c", "medium": "#2a78d6", "high": "#fab219", "critical": "#d03b3b"}


# --------------------------------------------------------------------------- pipeline

@st.cache_resource(show_spinner="Loading pipeline…")
def load_analyze(checkpoint: str):
    """Return (analyze, source, error). Real pipeline first, stub as fallback."""
    if checkpoint:
        os.environ["SAFEDERM_CKPT"] = checkpoint
    try:
        from app.pipeline import analyze  # Soumya (A)
        return analyze, "pipeline", None
    except Exception as exc:  # missing module, missing checkpoint, import error
        from app.stub_pipeline import analyze
        return analyze, "stub", f"{type(exc).__name__}: {exc}"


def overlay_image(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Use Sai's `overlay()` once ported; jet-colormap fallback until then."""
    try:
        from src.model.gradcam import overlay
        return overlay(image, cam, alpha=alpha)
    except (ImportError, AttributeError):
        from matplotlib import colormaps
        base = image.convert("RGB").resize(cam.shape[::-1])
        heat = (colormaps["jet"](np.clip(cam, 0, 1))[..., :3] * 255).astype(np.uint8)
        return Image.blend(base, Image.fromarray(heat), alpha)


# --------------------------------------------------------------------------- charts

def probs_chart(probs: dict[str, float], pred: str) -> go.Figure:
    codes = list(CLASS_ORDER)
    fig = go.Figure(go.Bar(
        x=[probs[c] for c in codes], y=[D.class_label(c) for c in codes], orientation="h",
        marker_color=[CLASS_COLOR[c] if c == pred else "#c9c8c2" for c in codes],
        text=[f"{probs[c]:.1%}" for c in codes], textposition="outside",
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=40, t=10, b=10),
                      xaxis=dict(range=[0, 1.1], tickformat=".0%", title="calibrated probability"),
                      yaxis=dict(autorange="reversed"))
    return fig


def gauge_chart(score: float, threshold: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#d03b3b" if score >= threshold else "#2a78d6"},
            "steps": [{"range": [0, threshold], "color": "#eef3fb"}, {"range": [threshold, 1], "color": "#fbeaea"}],
            "threshold": {"line": {"color": "#0b0b0b", "width": 3}, "value": threshold},
        },
        title={"text": f"Suspicious score p(mel)+p(bcc) · τ_s = {threshold:.2f}"},
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def trajectory_chart(trajectory: list[dict]) -> go.Figure:
    passes = [t["passes"] for t in trajectory]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=passes, y=[t["top_prob"] for t in trajectory], name="top-class probability",
                             mode="lines+markers", line_color="#2a78d6",
                             hovertext=[f"{t['pred_class']} · {t['tier']}" for t in trajectory]))
    fig.add_trace(go.Scatter(x=passes, y=[t["entropy"] for t in trajectory], name="entropy (nats)",
                             mode="lines+markers", line_color="#eb6834", yaxis="y2"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3),
                      xaxis=dict(title="MC head passes", range=[0, 52], dtick=5),
                      yaxis=dict(title="top prob", range=[0, 1]),
                      yaxis2=dict(title="entropy", overlaying="y", side="right", rangemode="tozero"))
    return fig


def latency_chart(lat: dict) -> go.Figure:
    parts = ["backbone", "mc_head", "gradcam"]
    fig = go.Figure(go.Bar(x=[lat.get(p, 0.0) for p in parts], y=["backbone (1×)", "MC head passes", "Grad-CAM"],
                           orientation="h", marker_color="#2a78d6",
                           text=[f"{lat.get(p, 0.0):.1f} ms" for p in parts], textposition="outside"))
    fig.update_layout(height=180, margin=dict(l=10, r=40, t=10, b=10), xaxis_title="ms")
    return fig


def badge(decision: str, reasons: list[str]) -> None:
    trusted = decision.upper().startswith("TRUSTED")
    color, text = ("#0ca30c", "TRUSTED RESULT") if trusted else ("#d03b3b", "NEEDS REVIEW")
    st.markdown(
        f"<div style='background:{color};color:white;padding:18px;border-radius:10px;"
        f"text-align:center;font-size:2rem;font-weight:700;letter-spacing:1px'>{text}</div>",
        unsafe_allow_html=True,
    )
    if trusted:
        st.caption("The model's own stability, entropy and risk-margin checks passed. This is not a statement that the lesion is benign.")
    else:
        for r in reasons:
            st.markdown(f"- **`{r}`**: {REASON_TEXT.get(r, '')}")


# --------------------------------------------------------------------------- layout

def sidebar() -> str:
    st.sidebar.title("Adaptive SafeDerm")
    ckpt = st.sidebar.text_input("Checkpoint path (best.pt)", value=os.environ.get("SAFEDERM_CKPT", ""),
                                 help="Or set the SAFEDERM_CKPT environment variable. Weights are never in Git.")
    st.sidebar.markdown("---")
    st.sidebar.markdown(D.DISCLAIMER)
    return ckpt


def pick_image() -> tuple[Image.Image | None, str | None]:
    upload = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])
    ex = D.examples()
    if not ex.empty:
        st.write("…or try a bundled test-set example:")
        cols = st.columns(len(ex))
        for col, row in zip(cols, ex.itertuples()):
            with col:
                st.image(row.path, width="stretch")
                if st.button(f"{row.image_id}", key=f"ex_{row.image_id}", help=f"true: {row.true_class} · {row.why_chosen}"):
                    st.session_state["example"] = row.path
                    st.session_state["example_label"] = f"{row.image_id} (true class: {row.true_class})"
    else:
        st.caption("No bundled examples yet (`app/assets/examples/`).")
    if upload is not None:
        st.session_state.pop("example", None)
        return Image.open(upload).convert("RGB"), upload.name
    if "example" in st.session_state:
        return Image.open(st.session_state["example"]).convert("RGB"), st.session_state["example_label"]
    return None, None


def show_result(image: Image.Image, name: str, res: dict, threshold: float) -> None:
    if res["mode"] == "replay":
        st.info("**REPLAY (saved logits):** this result replays the image's saved 50 MC logits through the live controller. No heatmap.")

    top = st.columns([2, 3])
    with top[0]:
        badge(res["decision"], res["reasons"])
        pred = res["pred_class"]
        st.metric("Predicted class", D.class_label(pred), f"{res['probs'][pred]:.1%} calibrated confidence", delta_color="off")
        if res["clinical_attention"]:
            st.error(f"**Clinical attention:** the prediction ({pred}) is in the suspicious group (mel, bcc).", icon="🚩")
        tier = res["tier"]
        st.markdown(
            f"Model-risk tier: <span style='background:{TIER_COLOR.get(tier, '#898781')};color:white;"
            f"padding:3px 10px;border-radius:6px;font-weight:600'>{tier.upper()}</span>",
            unsafe_allow_html=True,
        )
        st.metric("MC head passes used", f"{res['passes_used']} / 50")
    with top[1]:
        st.plotly_chart(probs_chart(res["probs"], pred), width="stretch")
        st.plotly_chart(gauge_chart(res["suspicious_score"], threshold), width="stretch")

    st.subheader("Where the model looked")
    img_cols = st.columns(2)
    with img_cols[0]:
        st.image(image.resize((224, 224)), caption=name, width="stretch")
    with img_cols[1]:
        cam = res.get("heatmap")
        if cam is None:
            st.info("Heatmap unavailable.")
        else:
            st.image(overlay_image(image.resize((224, 224)), np.asarray(cam)),
                     caption=f"Grad-CAM for predicted class `{pred}` (dropout off)", width="stretch")

    low = st.columns(2)
    with low[0]:
        st.subheader("How the prediction settled")
        st.plotly_chart(trajectory_chart(res["trajectory"]), width="stretch")
        st.caption("One point per 5-pass checkpoint; the controller stops at the first checkpoint that meets its rule (cap 50).")
    with low[1]:
        st.subheader("Latency")
        lat = res["latency_ms"]
        st.metric("Total", f"{lat.get('total', 0.0):.1f} ms")
        st.plotly_chart(latency_chart(lat), width="stretch")

    st.caption(f"Checkpoint SHA256: `{res['checkpoint_sha256']}` · mode: `{res['mode']}`")


def main() -> None:
    ckpt = sidebar()
    st.title("🩺 Adaptive SafeDerm — Upload & Analyze")
    D.disclaimer()

    analyze, source, err = load_analyze(ckpt)
    if source == "stub":
        st.error("**STUB DATA.** The real pipeline (`app/pipeline.py`) is not available, so the result below is a fixed "
                 "fake example for UI development. It does not come from the model.", icon="🚧")
        if err:
            st.caption(f"Pipeline import: {err}")

    image, name = pick_image()
    if image is None:
        st.info("Upload an image or pick an example to run the SafeDerm decision.")
        return
    with st.spinner("Running adaptive MC-Dropout controller…"):
        res = analyze(image)
    show_result(image, name, res, D.tau_s())


main()

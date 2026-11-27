"""Page 3: Project Overview — pipeline, paper vs ours, figure gallery, next steps.

Tables and figures come from `app/data/` (read-only copies of `second-main`
results, see app/data/SOURCES.md) and from the committed WB2 results.
No checkpoint needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import data_loaders as D  # noqa: E402

st.set_page_config(page_title="SafeDerm · Overview", page_icon="🧭", layout="wide")
FIG = D.APP_DATA / "figures"
cfg = D.controller_config()["risk_aware_adaptive"]
calib = D.read_json(D.BANK / "calibration.json")

st.title("🧭 Project Overview")
D.disclaimer()

st.markdown(
    "**Adaptive SafeDerm** classifies a dermoscopic image into one of 7 HAM10000 classes and then decides whether "
    "the model's answer can be trusted. Instead of always running 50 Monte-Carlo dropout passes, a **risk-aware "
    "controller** samples 5 passes at a time and stops early on easy, clearly benign cases. It keeps sampling, "
    "and refers the case for review, when the image is suspicious, unstable or close to the danger line."
)

# --------------------------------------------------------------------------- pipeline
st.header("The pipeline")
DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fillcolor="#eef3fb", color="#2a78d6"];
  edge [color="#52514e"];
  in  [label="1. Input\\n224×224 image"];
  cls [label="2. Classifier\\nEfficientNet-B0"];
  mc  [label="3. Adaptive uncertainty\\nMC Dropout, +5 passes"];
  cal [label="4. Calibrate\\ntemperature scaling"];
  gate[label="5. Risk gate\\ntier + checks", fillcolor="#fff4dc", color="#fab219"];
  e6a [label="6A. Easy case\\nstop early", fillcolor="#e6f6e6", color="#0ca30c"];
  e6b [label="6B. Explanation check\\n(planned, Spring)", style="rounded,dashed", fillcolor="#ffffff", color="#898781"];
  fin [label="7. Final safety controller\\nTRUSTED / NEEDS REVIEW", fillcolor="#fbeaea", color="#d03b3b"];
  in -> cls -> mc -> cal -> gate;
  gate -> mc [label="not settled", style=dashed, constraint=false];
  gate -> e6a [label="easy"];
  gate -> e6b [label="risky", style=dashed];
  e6a -> fin; e6b -> fin [style=dashed];
}
"""
st.graphviz_chart(DOT, width="stretch")

STAGES = [
    ("1. Input", "One dermoscopic image, resized to 224×224. The demo uses the V1 preprocessing path (no hair removal, "
                 "no CLAHE), because the controller was tuned on V1 inputs."),
    ("2. Classifier", "An ImageNet-pretrained EfficientNet-B0 fine-tuned on HAM10000. The backbone runs once per image; "
                      "a small head (256 units → Dropout(0.3) → 7 logits) gives the class scores."),
    ("3. Adaptive uncertainty", "Only the head dropout stays on at inference. The head is re-sampled 5 passes at a time, "
                                f"from a minimum of {cfg['min_passes']} up to a cap of 50, so easy images pay far less compute."),
    ("4. Calibrate", f"Per-pass logits are divided by one temperature (T = {calib['temperature']:.3f}, fitted on validation only) "
                     "so that a stated confidence matches how often the model is actually right."),
    ("5. Risk gate", f"After each batch it checks class stability, entropy, and the suspicious score p(mel)+p(bcc) against "
                     f"τ_s = {cfg['tau_s']}, then assigns a model-risk tier: low, medium, high or critical. High-tier cases "
                     f"need at least {cfg['min_passes_high']} passes."),
    ("6A. Stop early", "Stable, low-uncertainty cases far from the threshold stop at the minimum. This is where the compute saving comes from."),
    ("6B. Explanation check", "Planned for Spring: mask the Grad-CAM++ hot region, re-run the model, and compare against masking "
                              "a random region. Not built yet; every decision records `explanation_check = NOT_RUN`."),
    ("7. Final safety controller", "Returns TRUSTED RESULT, or NEEDS REVIEW with reasons (UNSTABLE_CLASS, HIGH_ENTROPY, "
                                   "SUSPICIOUS_MARGIN, CRITICAL_TIER, BUDGET_EXHAUSTED). A confident mel or bcc prediction "
                                   "always carries a clinical-attention flag."),
]
cols = st.columns(2)
for i, (name, text) in enumerate(STAGES):
    cols[i % 2].markdown(f"**{name}.** {text}")

# --------------------------------------------------------------------------- paper vs ours
st.header("Paper vs. ours")
left, right = st.columns(2)
with left:
    st.subheader("V3 replication: MC Dropout T=50 (test)")
    st.dataframe(D.read_csv(D.APP_DATA / "v3_table4_headline_T50.csv"), hide_index=True, width="stretch")
    D.source_caption(D.APP_DATA / "v3_table4_headline_T50.csv",
                     note="copy of second-main `results/v3/tables/table4_headline_T50.csv` (v2-main-b0 model, uncalibrated)")
with right:
    st.subheader("V3: entropy-based referral trade-off")
    ref = D.read_csv(D.APP_DATA / "v3_table6_referral.csv")
    st.dataframe(ref[["referral_rate", "n_accepted", "macro_auroc", "macro_f1", "accepted_accuracy"]],
                 hide_index=True, width="stretch")
    D.source_caption(D.APP_DATA / "v3_table6_referral.csv", note="τ taken from the test entropy distribution, as in the paper")

st.subheader("V5: backbone comparison (paper Table 7)")
left, right = st.columns([3, 2])
with left:
    st.dataframe(D.read_csv(D.APP_DATA / "v5_table7_comparison.csv"), hide_index=True, width="stretch")
    D.source_caption(D.APP_DATA / "v5_table7_comparison.csv")
with right:
    st.image(str(FIG / "v5_table7_params_vs_auroc.png"), width="stretch")
    D.source_caption(FIG / "v5_table7_params_vs_auroc.png")

st.caption("The replication model (v2-main-b0, with hair removal + CLAHE) differs from the V1 model the controller runs on, "
           "so these numbers are not compared like-for-like with page 2.")

# --------------------------------------------------------------------------- gallery
st.header("Figure gallery")
tabs = st.tabs(["EDA", "Preprocessing", "Grad-CAM vs. expert masks (V4)"])
with tabs[0]:
    for name in ["v1_class_distribution.png", "v1_split_class_proportions.png", "v1_hair_density.png", "v1_sample_grid.jpg"]:
        st.image(str(FIG / name), width="stretch")
        D.source_caption(FIG / name)
with tabs[1]:
    st.image(str(FIG / "v2_preprocessing_before_after.jpg"), width="stretch")
    D.source_caption(FIG / "v2_preprocessing_before_after.jpg", note="hair removal + CLAHE, replication branch (V2)")
with tabs[2]:
    st.dataframe(D.read_csv(D.APP_DATA / "v4_table10_iou.csv"), hide_index=True, width="stretch")
    D.source_caption(D.APP_DATA / "v4_table10_iou.csv", note="IoU between the binarised Grad-CAM map and the expert lesion mask")
    for name in ["v4_fig8_one_per_class.jpg", "v4_fig9_correct_vs_incorrect.jpg", "v4_iou_accepted_vs_referred.png"]:
        st.image(str(FIG / name), width="stretch")
        D.source_caption(FIG / name)
st.caption("Provenance of every copied file: `app/data/SOURCES.md`.")

# --------------------------------------------------------------------------- next
st.header("What's next (Spring 2027, CMPE 295B)")
st.markdown(
    "- **Fix the akiec gap:** add akiec to the attention group (most accepted melanoma misses were predicted akiec), "
    "then re-tune on val and re-test, on the V2 model.\n"
    "- **Stage 6B, the explanation check:** Grad-CAM++, mask the hot region, re-run, compare with a random-region mask.\n"
    "- **External stress test:** Derm7pt / PAD-UFES-20.\n"
    "- **The head-only dropout question:** MC passes barely change predictions on this model. Options are test-time "
    "augmentation or backbone stochasticity as the source of randomness, or reframing the contribution.\n"
    "- Online early-exit timing, MC-seed sensitivity, and the paper write-up."
)
D.disclaimer()

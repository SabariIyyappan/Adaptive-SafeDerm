"""Page 2: Model Evaluation dashboard.

Everything here is read live from committed result files; no checkpoint needed.
Every chart has a "Source:" caption naming the file(s) it was built from.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app import data_loaders as D  # noqa: E402
from src.eval.classes import CLASS_COLOR, CLASS_ORDER  # noqa: E402
from src.eval.metrics import confusion_matrices, expected_calibration_error, roc_curves  # noqa: E402

st.set_page_config(page_title="SafeDerm · Evaluation", page_icon="📊", layout="wide")

PROPOSED = "risk_aware_adaptive"
TRUSTED, REVIEW = "TRUSTED RESULT", "NEEDS REVIEW"
DECISION_COLOR = {TRUSTED: "#0ca30c", REVIEW: "#d03b3b"}
SUMMARY_COLS = {
    "macro_auroc": "Macro AUROC", "macro_f1": "Macro F1", "balanced_accuracy": "Bal. acc", "ece_10bin": "ECE",
    "brier": "Brier", "passes_mean": "Mean passes", "review_rate": "Review rate",
    "accepted_accuracy": "Acc. of accepted", "mel_miss_all_str": "Mel missed & accepted / all mel",
    "mel_miss_accepted_str": "… / accepted mel", "suspicious_false_reassurance_str": "Suspicious falsely reassured",
}

st.title("📊 Model Evaluation")
D.disclaimer()

c1, c2 = st.columns(2)
split = c1.radio("Split", ["test", "val"], horizontal=True,
                 help="The controller was tuned on val and locked before the single test replay. Test is primary.")
calib = c2.radio("Analysis", ["calibrated", "uncalibrated"], horizontal=True,
                 help="Calibrated (T≈2.007, fitted on val) was declared primary before the test replay.")

comp = D.policy_comparison(split, calib)
row = comp.set_index("method").loc[PROPOSED]
comp_path = D.table_path("policy_comparison", split, calib)

# --------------------------------------------------------------------------- metric cards
st.header(f"Risk-aware controller · {split} · {calib}")
cards = st.columns(4)
cards[0].metric("Macro AUROC", f"{row.macro_auroc:.4f}")
cards[1].metric("Macro F1", f"{row.macro_f1:.4f}")
cards[2].metric("Balanced accuracy", f"{row.balanced_accuracy:.4f}")

unc = D.read_csv(D.table_path("policy_comparison", split, "uncalibrated")).set_index("method").loc["fixed_T50"]
cal = D.read_csv(D.table_path("policy_comparison", split, "calibrated")).set_index("method").loc["fixed_T50"]
cards[3].metric("ECE, fixed T=50", f"{unc.ece_10bin:.4f} → {cal.ece_10bin:.4f}", "temperature scaling", delta_color="off")

cards = st.columns(4)
cards[0].metric("Mean head passes", f"{row.passes_mean:.1f} / 50", f"median {row.passes_median:.0f} · p95 {row.passes_p95:.0f}", delta_color="off")
cards[1].metric("Review rate", f"{row.review_rate:.1%}")
cards[2].metric("Accepted melanoma misses", row.mel_miss_all_str, f"{row.mel_miss_accepted_str} of accepted mel", delta_color="off")
cards[3].metric("Accuracy of accepted", f"{row.accepted_accuracy:.4f}")
calib_json = D.read_json(D.BANK / "calibration.json")
D.source_caption(comp_path, D.table_path("policy_comparison", split, "uncalibrated"),
                 note=f"T = {calib_json['temperature']:.3f} fitted on val (val ECE {calib_json['val_before']['ece_10bin']:.4f} → "
                      f"{calib_json['val_after']['ece_10bin']:.4f}, `{D.rel(D.BANK / 'calibration.json')}`)")

# --------------------------------------------------------------------------- comparison table
st.header("Method comparison")
tbl = comp[["method", *SUMMARY_COLS]].copy()
tbl["method"] = tbl["method"].map(lambda m: D.METHOD_LABEL.get(m, m))
tbl = tbl.rename(columns={"method": "Method", **SUMMARY_COLS})
st.dataframe(
    tbl.style.format({"Macro AUROC": "{:.4f}", "Macro F1": "{:.4f}", "Bal. acc": "{:.4f}", "ECE": "{:.4f}",
                      "Brier": "{:.4f}", "Mean passes": "{:.1f}", "Review rate": "{:.1%}", "Acc. of accepted": "{:.4f}"})
    .apply(lambda r: ["font-weight:700" if "proposed" in r["Method"] else "" for _ in r], axis=1),
    hide_index=True, width="stretch",
)
st.caption("AUROC/F1/bal. acc/ECE/Brier use all images (no referral); safety columns count TRUSTED results only. "
           "The matched baseline's entropy threshold is set on val to the controller's val review rate.")
D.source_caption(comp_path, D.WB2 / calib / "tables" / "matched_coverage.csv")

# --------------------------------------------------------------------------- compute vs safety, coverage vs risk
left, right = st.columns(2)
with left:
    st.subheader("Compute vs. safety")
    fig = go.Figure()
    for r in comp.itertuples():
        proposed = r.method == PROPOSED
        fig.add_trace(go.Scatter(
            x=[r.passes_mean], y=[r.mel_missed_accepted], mode="markers+text", name=D.METHOD_LABEL.get(r.method, r.method),
            text=[D.METHOD_LABEL.get(r.method, r.method).replace(" + ", "<br>+ ")], textposition="top center",
            marker=dict(size=16 if proposed else 11, color="#d03b3b" if proposed else "#2a78d6",
                        symbol="star" if proposed else "circle"),
            hovertemplate="%{x:.1f} passes · %{y} accepted mel misses<br>review %{customdata:.1%}<extra></extra>",
            customdata=[r.review_rate],
        ))
    fig.update_layout(height=430, showlegend=False, margin=dict(t=10),
                      xaxis_title="mean stochastic head passes per image", yaxis_title="melanomas missed & accepted (TRUSTED)")
    st.plotly_chart(fig, width="stretch")
    D.source_caption(comp_path)

with right:
    st.subheader("Coverage vs. risk")
    cr_path = D.table_path("coverage_risk_T50", split, calib)
    cr = D.read_csv(cr_path)
    metric = st.radio("Risk measure", ["melanoma misses accepted", "error rate of accepted"], horizontal=True)
    ycol, yval = (("mel_missed_accepted", row.mel_missed_accepted) if metric.startswith("mel")
                  else ("error_rate", 1 - row.accepted_accuracy))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cr.coverage, y=cr[ycol], mode="lines", name="Fixed T=50, highest-entropy-first referral",
                             line=dict(color="#898781", width=2)))
    fig.add_trace(go.Scatter(x=[row.coverage], y=[yval], mode="markers", name="Risk-aware controller",
                             marker=dict(size=16, color="#d03b3b", symbol="star")))
    fig.update_layout(height=400, margin=dict(t=10), legend=dict(orientation="h", y=-0.25),
                      xaxis=dict(title="coverage (fraction TRUSTED)", tickformat=".0%"), yaxis_title=metric)
    st.plotly_chart(fig, width="stretch")
    D.source_caption(cr_path, comp_path)

# --------------------------------------------------------------------------- stop passes / tiers
dec_path = D.decisions_path(split, calib)
rk = D.decisions(split, calib, PROPOSED)
left, right = st.columns(2)
with left:
    st.subheader("Where the controller stops")
    counts = rk.groupby(["passes", "decision"]).size().unstack(fill_value=0).reindex(columns=[TRUSTED, REVIEW], fill_value=0)
    counts = counts.reindex(range(10, 55, 5), fill_value=0)
    fig = go.Figure([go.Bar(x=counts.index, y=counts[d], name=d, marker_color=DECISION_COLOR[d]) for d in counts.columns])
    fig.update_layout(barmode="stack", height=380, margin=dict(t=10), legend=dict(orientation="h", y=-0.25),
                      xaxis=dict(title="head passes at stop", dtick=5), yaxis_title="images")
    st.plotly_chart(fig, width="stretch")
    D.source_caption(dec_path, note=f"method = {PROPOSED}")
with right:
    st.subheader("Final model-risk tier")
    tiers = rk.groupby(["tier", "decision"]).size().unstack(fill_value=0).reindex(columns=[TRUSTED, REVIEW], fill_value=0)
    tiers = tiers.reindex(["low", "medium", "high", "critical"], fill_value=0)
    fig = go.Figure([go.Bar(x=tiers.index, y=tiers[d], name=d, marker_color=DECISION_COLOR[d]) for d in tiers.columns])
    fig.update_layout(barmode="stack", height=380, margin=dict(t=10), legend=dict(orientation="h", y=-0.25),
                      xaxis_title="tier", yaxis_title="images")
    st.plotly_chart(fig, width="stretch")
    D.source_caption(dec_path, note=f"method = {PROPOSED}. With the locked h_lo the low/medium split is not informative.")

reasons = rk["review_reasons"].dropna().str.split("|").explode().value_counts()
st.caption("Review reasons among NEEDS REVIEW cases (one case can have several): "
           + ", ".join(f"`{k}` {v}" for k, v in reasons.items()))

# --------------------------------------------------------------------------- reliability
st.header("Calibration: reliability diagram")
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration", line=dict(color="#898781", dash="dash")))
rel_paths = []
for c, color in (("uncalibrated", "#eb6834"), ("calibrated", "#2a78d6")):
    t50 = D.decisions(split, c, "fixed_T50")
    probs, y = D.probs_labels(t50)
    ece, bins = expected_calibration_error(y, probs)
    bins = bins[bins["count"] > 0]
    fig.add_trace(go.Scatter(x=bins.confidence, y=bins.accuracy, mode="lines+markers", line_color=color,
                             name=f"{'before' if c == 'uncalibrated' else 'after'} temperature scaling (ECE {ece:.4f})",
                             customdata=bins["count"], hovertemplate="conf %{x:.2f} · acc %{y:.2f} · n=%{customdata}<extra></extra>"))
    rel_paths.append(D.decisions_path(split, c))
fig.update_layout(height=450, margin=dict(t=10), xaxis=dict(title="mean confidence (10 equal-width bins)", range=[0, 1]),
                  yaxis=dict(title="accuracy", range=[0, 1]), legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, width="stretch")
D.source_caption(*rel_paths, note="method = fixed_T50; ECE via `src.eval.metrics.expected_calibration_error`")

# --------------------------------------------------------------------------- confusion + ROC
probs, y = D.probs_labels(rk)
left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    norm = st.toggle("Row-normalised", value=True)
    counts, normalized = confusion_matrices(y, probs)
    z = normalized if norm else counts
    fig = go.Figure(go.Heatmap(
        z=z, x=list(CLASS_ORDER), y=list(CLASS_ORDER), colorscale="Blues", showscale=False,
        text=[[f"{v:.2f}" if norm else f"{int(v)}" for v in r] for r in z], texttemplate="%{text}",
        customdata=counts, hovertemplate="true %{y} → pred %{x}<br>n=%{customdata}<extra></extra>",
    ))
    fig.update_layout(height=450, margin=dict(t=10), xaxis_title="predicted", yaxis=dict(title="true", autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    D.source_caption(dec_path, note=f"method = {PROPOSED} (calibrated probabilities, all images)" if calib == "calibrated"
                     else f"method = {PROPOSED}")
with right:
    st.subheader("Per-class ROC (one-vs-rest)")
    curves = roc_curves(y, probs)
    fig = go.Figure()
    for cls in CLASS_ORDER:
        if cls in curves:
            fig.add_trace(go.Scatter(x=curves[cls]["fpr"], y=curves[cls]["tpr"], mode="lines", line_color=CLASS_COLOR[cls],
                                     name=f"{cls} (AUROC {curves[cls]['auroc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#898781", dash="dash"), showlegend=False))
    fig.update_layout(height=450, margin=dict(t=10), xaxis_title="false positive rate", yaxis_title="true positive rate")
    st.plotly_chart(fig, width="stretch")
    D.source_caption(dec_path, note=f"method = {PROPOSED}")

# --------------------------------------------------------------------------- melanoma audit
st.header("Melanoma safety audit")
audit_path = D.table_path("melanoma_audit", split, calib)
audit = D.read_csv(audit_path)
grid = pd.crosstab(audit["correct"].map({True: "Predicted mel (correct)", False: "Predicted another class (missed)"}),
                   audit["decision"]).reindex(index=["Predicted mel (correct)", "Predicted another class (missed)"],
                                              columns=[TRUSTED, REVIEW], fill_value=0)
left, right = st.columns([1, 1])
with left:
    st.dataframe(grid, width="stretch")
    st.caption(f"All {len(audit)} {split} melanomas, risk-aware controller. The bottom-left cell is false reassurance.")
with right:
    missed = audit[(~audit["correct"]) & (audit["decision"] == TRUSTED)]
    by_pred = missed["pred_dx"].value_counts()
    fig = go.Figure(go.Bar(x=by_pred.values, y=[D.class_label(c) for c in by_pred.index], orientation="h",
                           marker_color=[CLASS_COLOR.get(c, "#2a78d6") for c in by_pred.index], text=by_pred.values,
                           textposition="outside"))
    fig.update_layout(height=250, margin=dict(t=30, r=30), title="Accepted melanoma misses were predicted as…",
                      yaxis=dict(autorange="reversed"), xaxis_title="melanomas")
    st.plotly_chart(fig, width="stretch")
D.source_caption(audit_path)
if "akiec" in by_pred.index:
    st.info(f"**Adverse finding:** {by_pred['akiec']} of the {len(missed)} accepted melanoma misses were predicted akiec, which "
            "is outside the declared suspicious group (mel, bcc). The Spring plan adds akiec to the attention group "
            "and re-tunes on val.")
with st.expander("Per-melanoma detail"):
    st.dataframe(audit, hide_index=True, width="stretch")

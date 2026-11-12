"""V3 result package — MC Dropout uncertainty, calibration and referral
(PLAN.md §7, paper §3.4, §4.2, §4.4-4.6).

    python -m src.eval.run_v3

Reads only the committed C4 arrays in `results/v3/inputs/` (mc_val.npz,
mc_test.npz, mc_report.json, checkpoint.json) and writes `results/v3/`:
Table 4 (T=50 headline + T=1), the T-budget sweep, uncertainty summary,
Table 6 (referral), the C8 referral decisions, Figures 3, 6, 7 and the T=50
per-class package (Table 5 / Figs 4-5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda import CLASS_ORDER
from src.eval import tables as T
from src.eval.classes import CHART_INK, SERIES_2, SINGLE_SERIES_BLUE
from src.eval.common import (
    evaluate,
    fmt_ci,
    plot_confusion_matrix,
    plot_per_class,
    plot_reliability,
    plot_roc,
    write_json,
)
from src.eval.style import finish_figure, grid_y, new_figure
from src.eval.uncertainty import (
    REFERRAL_RATES,
    accept_mask,
    bootstrap_scalar,
    budget_sweep,
    decompose,
    ece_of,
    referral_table,
    referral_threshold,
)

PAPER = {"auroc": "0.9404 [0.9351, 0.9455]", "f1": "0.7308 [0.7109, 0.7497]",
         "ece50": "0.1456 [0.1311, 0.1601]", "ece1": "0.1891 [0.1722, 0.2057]",
         "total": 0.3209, "epistemic": 0.0113}
PAPER_REFERRAL_GAINS = {0.10: (0.77, 2.57), 0.30: (1.64, 7.91)}


def load_c4(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def plot_entropy_histogram(total: np.ndarray, correct: np.ndarray, out_path: Path) -> Path:
    fig, ax = new_figure(figsize=(9, 4.6))
    bins = np.linspace(0, max(total.max(), 1e-6), 40)
    ax.hist(total[correct], bins=bins, color=SINGLE_SERIES_BLUE, alpha=0.85, label=f"Correct (n={int(correct.sum())})")
    ax.hist(total[~correct], bins=bins, color=SERIES_2, alpha=0.75, label=f"Incorrect (n={int((~correct).sum())})")
    ax.set_xlabel("Total predictive entropy H[p̄] (nats), T=50")
    ax.set_ylabel("Test images")
    ax.set_title("Total entropy: correct vs incorrect predictions (paper Fig. 6)", fontsize=10,
                 color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=8)
    grid_y(ax)
    caption = (f"Median entropy: correct {np.median(total[correct]):.3f}, incorrect {np.median(total[~correct]):.3f} nats. "
               "Paper: correct predictions concentrate near 0, incorrect spread over 0-1.4 nats.")
    return finish_figure(fig, out_path, caption)


def plot_decomposition(unc: dict, out_path: Path) -> Path:
    fig, axes = new_figure(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    bins = np.linspace(0, max(unc["total"].max(), 1e-6), 40)
    ax.hist(unc["aleatoric"], bins=bins, color=SINGLE_SERIES_BLUE, alpha=0.85, label="Aleatoric (mean per-pass entropy)")
    ax.hist(unc["epistemic"], bins=bins, color=SERIES_2, alpha=0.85, label="Epistemic (mutual information)")
    ax.set_xlabel("nats")
    ax.set_ylabel("Test images")
    ax.set_title("Aleatoric vs epistemic uncertainty", fontsize=10, color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=8)
    grid_y(ax)
    ax = axes[1]
    ax.scatter(unc["total"], unc["epistemic"], s=5, color=SINGLE_SERIES_BLUE, alpha=0.5)
    ax.set_xlabel("Total entropy (nats)")
    ax.set_ylabel("Epistemic (nats)")
    ax.set_title("Epistemic vs total entropy per image", fontsize=10, color=CHART_INK["primary"])
    grid_y(ax)
    caption = (f"Means: total {unc['total'].mean():.4f}, aleatoric {unc['aleatoric'].mean():.4f}, epistemic "
               f"{unc['epistemic'].mean():.4f} nats (paper: total 0.3209, epistemic 0.0113).")
    return finish_figure(fig, out_path, caption)


def plot_referral(table: pd.DataFrame, out_path: Path, val_table: pd.DataFrame | None = None) -> Path:
    fig, axes = new_figure(1, 2, figsize=(12, 4.6))
    r = 100 * table["referral_rate"]
    for ax, metric, lo, hi, label in ((axes[0], "macro_auroc", "auroc_ci_low", "auroc_ci_high", "Macro AUROC"),
                                      (axes[1], "macro_f1", "f1_ci_low", "f1_ci_high", "Macro F1")):
        ax.fill_between(r, table[lo], table[hi], color=SINGLE_SERIES_BLUE, alpha=0.18, linewidth=0)
        ax.plot(r, table[metric], marker="o", color=SINGLE_SERIES_BLUE, linewidth=2, label="tau from test entropy (paper)")
        if val_table is not None:
            ax.plot(100 * val_table["actual_referral_rate"], val_table[metric], marker="s", color=SERIES_2,
                    linewidth=1.5, label="tau from validation entropy")
        for x, y in zip(r, table[metric]):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=6.5)
        ax.set_xlabel("Referral rate r (%)")
        ax.set_ylabel(f"Accepted-subset {label}")
        ax.set_title(f"Accepted-subset {label} vs referral rate", fontsize=10, color=CHART_INK["primary"])
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        grid_y(ax)
    last = table.iloc[-1]
    caption = (f"Shaded band: 95% bootstrap CI within the accepted subset. At r=30%: AUROC {last['macro_auroc']:.4f} "
               f"({last['auroc_gain_pct']:+.2f}%), F1 {last['macro_f1']:.4f} ({last['f1_gain_pct']:+.2f}%); "
               "paper: +1.64% / +7.91%.")
    return finish_figure(fig, out_path, caption)


def run(inputs: Path, out_dir: Path, n_resamples: int, seed: int) -> dict:
    val, test = load_c4(inputs / "mc_val.npz"), load_c4(inputs / "mc_test.npz")
    mc_report = json.loads((inputs / "mc_report.json").read_text())
    y = test["y_true"]
    unc = decompose(test["probs_mc"])
    p50, p1 = unc["mean_probs"], test["probs_det"].astype(float)
    val_unc = decompose(val["probs_mc"])

    print(f"V3: {len(y)} test images, T={int(test['T'])}")
    ev50, ev1 = evaluate(y, p50, n_resamples, seed), evaluate(y, p1, n_resamples, seed)
    ece50 = bootstrap_scalar(y, p50, ece_of, n_resamples, seed)
    ece1 = bootstrap_scalar(y, p1, ece_of, n_resamples, seed)

    table4 = pd.DataFrame([
        {"Metric": "Macro AUROC", "Ours T=50": fmt_ci(ev50.metrics["macro_auroc"], *[ev50.boot.ci_low["macro_auroc"], ev50.boot.ci_high["macro_auroc"]]),
         "Ours T=1 (deterministic)": fmt_ci(ev1.metrics["macro_auroc"], ev1.boot.ci_low["macro_auroc"], ev1.boot.ci_high["macro_auroc"]),
         "Paper T=50": PAPER["auroc"]},
        {"Metric": "Macro F1", "Ours T=50": fmt_ci(ev50.metrics["macro_f1"], ev50.boot.ci_low["macro_f1"], ev50.boot.ci_high["macro_f1"]),
         "Ours T=1 (deterministic)": fmt_ci(ev1.metrics["macro_f1"], ev1.boot.ci_low["macro_f1"], ev1.boot.ci_high["macro_f1"]),
         "Paper T=50": PAPER["f1"]},
        {"Metric": "ECE (10 bins)", "Ours T=50": fmt_ci(*ece50[:3]), "Ours T=1 (deterministic)": fmt_ci(*ece1[:3]),
         "Paper T=50": f"{PAPER['ece50']} (T=1: {PAPER['ece1']})"},
        {"Metric": "Brier score", "Ours T=50": f"{ev50.brier:.4f}", "Ours T=1 (deterministic)": f"{ev1.brier:.4f}",
         "Paper T=50": "not reported numerically"},
        {"Metric": "Balanced accuracy", "Ours T=50": f"{ev50.metrics['balanced_accuracy']:.4f}",
         "Ours T=1 (deterministic)": f"{ev1.metrics['balanced_accuracy']:.4f}", "Paper T=50": "not reported"},
        {"Metric": "Mean total entropy (nats)", "Ours T=50": f"{unc['total'].mean():.4f}", "Ours T=1 (deterministic)": "—",
         "Paper T=50": f"{PAPER['total']}"},
        {"Metric": "Mean aleatoric (nats)", "Ours T=50": f"{unc['aleatoric'].mean():.4f}", "Ours T=1 (deterministic)": "—",
         "Paper T=50": "not reported"},
        {"Metric": "Mean epistemic (nats)", "Ours T=50": f"{unc['epistemic'].mean():.4f}", "Ours T=1 (deterministic)": "—",
         "Paper T=50": f"{PAPER['epistemic']}"},
    ])

    print("  referral (test-entropy tau, as paper) ...")
    ref = referral_table(y, p50, unc["total"], n_resamples=n_resamples, seed=seed)
    print("  referral (validation-entropy tau, leakage-free variant) ...")
    ref_val = referral_table(y, p50, unc["total"], tau_source=val_unc["total"], n_resamples=n_resamples, seed=seed)
    sweep = budget_sweep(y, test["probs_mc"].astype(float), p1)

    correct = p50.argmax(1) == y
    unc_summary = pd.DataFrame([
        {"Quantity": q, "Mean": unc[k].mean(), "Median": np.median(unc[k]), "Std": unc[k].std(),
         "Mean (correct)": unc[k][correct].mean(), "Mean (incorrect)": unc[k][~correct].mean()}
        for q, k in (("Total entropy", "total"), ("Aleatoric", "aleatoric"), ("Epistemic", "epistemic"))
    ]).round(4)

    # C8 referral decisions (paper's test-entropy tau), long format: one row per image per r.
    rows = []
    for r in REFERRAL_RATES:
        tau = referral_threshold(unc["total"], r)
        acc = accept_mask(unc["total"], tau)
        rows.append(pd.DataFrame({"image_id": test["image_ids"], "referral_rate": r, "tau": tau,
                                  "entropy_total": unc["total"], "entropy_aleatoric": unc["aleatoric"],
                                  "entropy_epistemic": unc["epistemic"], "y_true": y, "y_pred": p50.argmax(1),
                                  "accepted": acc}))
    c8 = pd.concat(rows, ignore_index=True)
    (out_dir / "decisions").mkdir(parents=True, exist_ok=True)
    c8.to_csv(out_dir / "decisions" / "referral_decisions_c8.csv", index=False, float_format="%.6f")

    # T=50 per-image predictions in C3 format, so the V1 suite can consume them.
    c3 = pd.DataFrame({"image_id": test["image_ids"], "split": "test", "y_true": y,
                       **{f"prob_{c}": p50[:, c] for c in range(7)}, "run_id": "v2-main-b0@T50"})
    c3.to_csv(out_dir / "decisions" / "predictions_test_T50.csv", index=False, float_format="%.8f")

    fdir = out_dir / "figures"
    plot_reliability(y, {"T=1 (deterministic)": p1, "T=50 MC Dropout": p50}, fdir / "fig3_reliability_T1_vs_T50.png",
                     "Reliability: T=1 vs T=50 (paper Fig. 3)",
                     f"ECE T=1 {ece1[0]:.4f} -> T=50 {ece50[0]:.4f} (paper 0.1891 -> 0.1456). Points below the diagonal "
                     "are over-confident; lower panel shows images per confidence bin.")
    plot_entropy_histogram(unc["total"], correct, fdir / "fig6_entropy_histogram.png")
    plot_decomposition(unc, fdir / "uncertainty_decomposition.png")
    plot_referral(ref, fdir / "fig7_referral_tradeoff.png", val_table=ref_val)
    plot_confusion_matrix(y, p50, fdir / "fig5_confusion_matrix_T50.png", "v2-main-b0, T=50")
    plot_per_class(ev50.per_class, fdir / "fig4_per_class_T50.png", "v2-main-b0, T=50: per-class accuracy, sensitivity, specificity")
    plot_roc(y, p50, fdir / "roc_curves_T50.png", "v2-main-b0, T=50: one-vs-rest ROC")

    fmt = lambda df: df.map(lambda v: f"{v:.4f}" if isinstance(v, float) else v)
    T.write_tables({
        "table4_headline_T50": table4,
        "table5_per_class_T50": T.per_class_markdown_table(ev50.per_class),
        "table6_referral": fmt(ref),
        "table6b_referral_val_tau": fmt(ref_val),
        "mc_budget_sweep": fmt(sweep),
        "uncertainty_summary": unc_summary,
    }, out_dir / "tables")

    checks = [
        {"check": "Epistemic much smaller than aleatoric (paper 0.011 vs 0.32)",
         "value": f"{unc['epistemic'].mean():.4f} vs {unc['aleatoric'].mean():.4f}",
         "passed": bool(unc["epistemic"].mean() < 0.25 * unc["aleatoric"].mean())},
        {"check": "Epistemic >= 0 for every image (numerical tol 1e-6)",
         "value": f"min {unc['epistemic'].min():.2e}", "passed": bool(unc["epistemic"].min() > -1e-6)},
        {"check": "ECE at T=50 <= ECE at T=1", "value": f"{ece50[0]:.4f} vs {ece1[0]:.4f}", "passed": bool(ece50[0] <= ece1[0])},
        {"check": "Accepted-subset AUROC rises monotonically with r",
         "value": " -> ".join(f"{v:.4f}" for v in ref["macro_auroc"]),
         "passed": bool(np.all(np.diff(ref["macro_auroc"]) >= -1e-9))},
        {"check": "r=30% AUROC CI does not overlap r=0 CI (paper: no overlap)",
         "value": f"r=0 [{ref['auroc_ci_low'].iloc[0]:.4f}, {ref['auroc_ci_high'].iloc[0]:.4f}] vs r=30 "
                  f"[{ref['auroc_ci_low'].iloc[-1]:.4f}, {ref['auroc_ci_high'].iloc[-1]:.4f}]",
         "passed": bool(ref["auroc_ci_low"].iloc[-1] > ref["auroc_ci_high"].iloc[0])},
        {"check": "MC passes vary (dropout active) and BN unchanged",
         "value": f"mean per-pass std {mc_report['splits']['test']['mean_per_pass_std']:.4f}; BN unchanged "
                  f"{mc_report['batchnorm_unchanged']}",
         "passed": bool(mc_report["splits"]["test"]["mean_per_pass_std"] > 0 and mc_report["batchnorm_unchanged"])},
    ]

    payload = {
        "run_id": "v2-main-b0", "T": int(test["T"]), "seed": int(test["seed"]),
        "checkpoint_sha256": str(test["checkpoint_sha256"]), "mc_report": mc_report,
        "test_T50": {**ev50.headline(), "ece_ci": list(ece50[1:3]), "bootstrap": ev50.boot.as_dict()},
        "test_T1_deterministic": {**ev1.headline(), "ece_ci": list(ece1[1:3]), "bootstrap": ev1.boot.as_dict()},
        "uncertainty_means": {k: float(unc[k].mean()) for k in ("total", "aleatoric", "epistemic")},
        "referral_test_tau": ref.to_dict(orient="records"),
        "referral_val_tau": ref_val.to_dict(orient="records"),
        "budget_sweep": sweep.to_dict(orient="records"),
        "sanity_checks": checks,
    }
    write_json(payload, out_dir / "metrics.json")
    (out_dir / "summary.md").write_text(build_summary(payload, table4, ref, ref_val, sweep, unc_summary, checks),
                                        encoding="utf-8")
    print(f"Wrote {out_dir}; sanity checks {sum(c['passed'] for c in checks)}/{len(checks)} pass")
    return payload


def build_summary(p, table4, ref, ref_val, sweep, unc_summary, checks) -> str:
    def row(r):
        return ref.loc[np.isclose(ref["referral_rate"], r)].iloc[0]
    r10, r30 = row(0.10), row(0.30)
    fmt = lambda df: df.map(lambda v: f"{v:.4f}" if isinstance(v, float) else v)
    lines = [
        "# V3 — MC Dropout Uncertainty + Entropy-Based Referral",
        "",
        "Generated by `python -m src.eval.run_v3` from the committed C4 arrays in `results/v3/inputs/`.",
        f"Model: locked `v2-main-b0` (sha256 `{p['checkpoint_sha256'][:16]}…`), T={p['T']} MC Dropout passes, seed {p['seed']}.",
        "Only the head `Dropout(0.3)` is stochastic; BatchNorm and EfficientNet's StochasticDepth stay in eval mode.",
        f"Backbone features are computed once and the head sampled T times — verified identical to full forward passes "
        f"(max |diff| {p['mc_report']['verify_max_abs_diff']:.1e} on 32 images).",
        "",
        "## Table 4 — headline test results",
        "",
        table4.to_markdown(index=False),
        "",
        "## MC budget sweep (mean of the first T passes of the same 50-pass bank)",
        "",
        fmt(sweep).to_markdown(index=False),
        "",
        "## Uncertainty decomposition (nats, test set, T=50)",
        "",
        unc_summary.to_markdown(index=False),
        "",
        "## Table 6 — referral trade-off (tau = (1-r)-quantile of TEST entropy, as in the paper)",
        "",
        fmt(ref[["referral_rate", "tau", "n_accepted", "macro_auroc", "auroc_ci_low", "auroc_ci_high", "macro_f1",
                 "f1_ci_low", "f1_ci_high", "auroc_gain_pct", "f1_gain_pct"]]).to_markdown(index=False),
        "",
        f"Gains vs r=0: r=10% AUROC {r10['auroc_gain_pct']:+.2f}%, F1 {r10['f1_gain_pct']:+.2f}% (paper +0.77% / +2.57%); "
        f"r=30% AUROC {r30['auroc_gain_pct']:+.2f}%, F1 {r30['f1_gain_pct']:+.2f}% (paper +1.64% / +7.91%).",
        "",
        "**Methodological note (PLAN.md §13 item 5).** The paper takes tau from the test set's own entropy distribution. "
        "Table 6b repeats the analysis with tau from the *validation* entropy distribution (no test information); the "
        "actual test referral rate then differs slightly from the nominal r.",
        "",
        fmt(ref_val[["referral_rate", "tau", "actual_referral_rate", "n_accepted", "macro_auroc", "macro_f1",
                     "auroc_gain_pct", "f1_gain_pct"]]).to_markdown(index=False),
        "",
        "## Sanity checks (PLAN.md §7.5)",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
        "## Figures",
        "",
        "- `figures/fig3_reliability_T1_vs_T50.png` — paper Fig. 3",
        "- `figures/fig6_entropy_histogram.png` — paper Fig. 6",
        "- `figures/fig7_referral_tradeoff.png` — paper Fig. 7",
        "- `figures/fig4_per_class_T50.png`, `figures/fig5_confusion_matrix_T50.png`, `figures/roc_curves_T50.png`",
        "- `figures/uncertainty_decomposition.png`",
        "",
        "C8 referral decisions for every r: `decisions/referral_decisions_c8.csv` (consumed by V4).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 result package (PLAN.md §7).")
    parser.add_argument("--inputs", type=Path, default=Path("results/v3/inputs"))
    parser.add_argument("--out", type=Path, default=Path("results/v3"))
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.inputs, args.out, args.n_resamples, args.seed)


if __name__ == "__main__":
    main()

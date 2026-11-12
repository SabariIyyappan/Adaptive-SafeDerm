"""V5 result package — comparative baselines (paper §4.7, Table 7) and the
final "Paper vs Ours" replication report (PLAN.md §9).

    python -m src.eval.run_v5

Reads `results/v5/inputs/<run_id>/` (C3/C5/C6 + params.json per backbone),
the V2 main-model single-pass predictions and the V1-V4 `metrics.json`
files, and writes `results/v5/` (Table 7, param-vs-AUROC chart, per-backbone
training curves) and `results/final_report/report.md`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import tables as T
from src.eval.classes import CHART_INK, SERIES_2, SINGLE_SERIES_BLUE
from src.eval.common import evaluate_file, fmt_ci, plot_training_curves, write_json
from src.eval.style import finish_figure, grid_xy, new_figure

BACKBONES = [
    ("v5-mobilenetv3", "MobileNetV3-Large", "mobilenet_v3_large"),
    ("v5-resnet50", "ResNet-50", "resnet50"),
    ("v5-densenet121", "DenseNet-121", "densenet121"),
]


def count_params(arch: str) -> tuple[int, int]:
    from src.model.model import build_model, count_parameters

    return count_parameters(build_model(pretrained=False, arch=arch))


def plot_params_vs_auroc(rows: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = new_figure(figsize=(8.5, 5))
    for _, r in rows.iterrows():
        color = SERIES_2 if r["Model"].startswith("EfficientNet") else SINGLE_SERIES_BLUE
        ax.errorbar(r["params_m"], r["macro_auroc"], yerr=[[r["macro_auroc"] - r["auroc_lo"]], [r["auroc_hi"] - r["macro_auroc"]]],
                    fmt="o", color=color, capsize=4, markersize=8)
        ax.annotate(f"{r['Model']}\n{r['macro_auroc']:.4f}", (r["params_m"], r["macro_auroc"]), textcoords="offset points",
                    xytext=(8, -4), fontsize=7.5, color=CHART_INK["primary"])
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("Test macro AUROC (single pass), 95% CI")
    ax.set_title("Accuracy vs model size (paper Table 7)", fontsize=10, color=CHART_INK["primary"])
    grid_xy(ax)
    caption = ("All backbones: same C9 cache, split, augmentation, dual imbalance correction, provisional hyperparameters, "
               "30 epochs and the same 256-unit dropout head; single-pass evaluation. Orange = our EfficientNet-B0.")
    return finish_figure(fig, out_path, caption)


def run(inputs: Path, v2_inputs: Path, out_dir: Path, n_resamples: int, seed: int) -> dict:
    b0_params = count_params("efficientnet_b0")
    v3 = json.loads(Path("results/v3/metrics.json").read_text()) if Path("results/v3/metrics.json").exists() else None
    rows, histories = [], {}
    entries = [("v2-main-b0", "EfficientNet-B0 (ours, main)", "efficientnet_b0", v2_inputs / "v2-main-b0")] + \
              [(rid, name, arch, inputs / rid) for rid, name, arch in BACKBONES]
    for rid, name, arch, d in entries:
        if not (d / "predictions_test.csv").exists():
            print(f"  [skip] {rid}: no predictions")
            continue
        print(f"  evaluating {rid}")
        ev = evaluate_file(d / "predictions_test.csv", n_resamples, seed)
        h = ev.headline()
        total, trainable = count_params(arch)
        ck = json.loads((d / "checkpoint.json").read_text())
        histories[rid] = (pd.read_csv(d / "history.csv"), int(ck["best_epoch"]), name)
        rows.append({"run_id": rid, "Model": name, "params": total, "trainable_params": trainable,
                     "params_m": total / 1e6, "best_epoch": int(ck["best_epoch"]),
                     "macro_auroc": h["macro_auroc"], "auroc_lo": h["macro_auroc_ci"][0], "auroc_hi": h["macro_auroc_ci"][1],
                     "macro_f1": h["macro_f1"], "f1_lo": h["macro_f1_ci"][0], "f1_hi": h["macro_f1_ci"][1],
                     "balanced_accuracy": h["balanced_accuracy"], "mel_sensitivity": h["mel_sensitivity"],
                     "ece_10bin": h["ece_10bin"], "uncertainty": "MC Dropout (V3)" if rid == "v2-main-b0" else "no"})
    res = pd.DataFrame(rows)

    table7 = pd.DataFrame([{
        "Model": r["Model"], "Parameters": f"{r['params'] / 1e6:.2f} M",
        "B0 size ratio": f"{b0_params[0] / r['params']:.2f}",
        "Macro AUROC [95% CI]": fmt_ci(r["macro_auroc"], r["auroc_lo"], r["auroc_hi"]),
        "Macro F1 [95% CI]": fmt_ci(r["macro_f1"], r["f1_lo"], r["f1_hi"]),
        "Mel sens.": f"{r['mel_sensitivity']:.3f}", "Uncertainty estimate": r["uncertainty"],
    } for _, r in res.iterrows()])
    if v3:
        t = v3["test_T50"]
        table7.loc[len(table7)] = {"Model": "EfficientNet-B0 (ours) + MC Dropout T=50", "Parameters": f"{b0_params[0] / 1e6:.2f} M",
                                   "B0 size ratio": "1.00",
                                   "Macro AUROC [95% CI]": fmt_ci(t["macro_auroc"], *t["macro_auroc_ci"]),
                                   "Macro F1 [95% CI]": fmt_ci(t["macro_f1"], *t["macro_f1_ci"]),
                                   "Mel sens.": f"{t['mel_sensitivity']:.3f}", "Uncertainty estimate": "yes (T=50)"}
    table7.loc[len(table7)] = {"Model": "Paper: EfficientNet-B0 + MC Dropout T=50", "Parameters": "~4.3 M (paper)",
                               "B0 size ratio": "—", "Macro AUROC [95% CI]": "0.9404 [0.9351, 0.9455]",
                               "Macro F1 [95% CI]": "0.7308 [0.7109, 0.7497]", "Mel sens.": "0.808",
                               "Uncertainty estimate": "yes"}
    table7.loc[len(table7)] = {"Model": "ISIC 2018 winner (literature)", "Parameters": "—", "B0 size ratio": "—",
                               "Macro AUROC [95% CI]": "different protocol — context only, not trained",
                               "Macro F1 [95% CI]": "—", "Mel sens.": "—", "Uncertainty estimate": "—"}

    plot_params_vs_auroc(res, out_dir / "figures" / "table7_params_vs_auroc.png")
    for rid, (hist, best, name) in histories.items():
        if rid != "v2-main-b0":
            plot_training_curves(hist, best, out_dir / "figures" / f"training_curves_{rid}.png", name)
    T.write_tables({"table7_comparison": table7}, out_dir / "tables")
    res.to_csv(out_dir / "tables" / "comparison_raw.csv", index=False)

    r = res.set_index("run_id")
    claims = []
    for rid, paper_ratio in (("v5-resnet50", "about 1/5"), ("v5-densenet121", "about 2/3")):
        if rid in r.index:
            ratio = r.loc["v2-main-b0", "params"] / r.loc[rid, "params"]
            gap = r.loc["v2-main-b0", "macro_auroc"] - r.loc[rid, "macro_auroc"]
            claims.append({"claim": f"B0 is competitive with {r.loc[rid, 'Model']} at {paper_ratio} of its parameters",
                           "measured param ratio": f"{ratio:.2f}", "AUROC B0 - other": f"{gap:+.4f}",
                           "CIs overlap": bool(r.loc["v2-main-b0", "auroc_lo"] <= r.loc[rid, "auroc_hi"]
                                               and r.loc[rid, "auroc_lo"] <= r.loc["v2-main-b0", "auroc_hi"])})
    claims = pd.DataFrame(claims)
    payload = {"comparison": res.to_dict(orient="records"), "claims": claims.to_dict(orient="records")}
    write_json(payload, out_dir / "metrics.json")
    lines = [
        "# V5 — Comparative Baselines (paper Table 7)",
        "",
        "Generated by `python -m src.eval.run_v5`. Every backbone uses the `v2-main-b0` recipe with the same 256-unit "
        "dropout head on its pooled features; only the backbone differs. Single-pass evaluation (the paper's baselines "
        "give no uncertainty estimate). ResNet-50 and DenseNet-121 trained with mixed precision to fit a 6 GB GPU.",
        "",
        table7.to_markdown(index=False),
        "",
        "## The paper's efficiency claim",
        "",
        claims.to_markdown(index=False) if len(claims) else "(comparison runs missing)",
        "",
        "Figures: `figures/table7_params_vs_auroc.png`, `figures/training_curves_<run>.png`.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")
    return payload


# ------------------------------------------------------------ final report
def _load(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def final_report(out: Path) -> Path:
    v1, v2, v3, v4, v5 = (_load(f"results/v{i}/metrics.json") for i in range(1, 6))
    f = lambda x, d=4: "—" if x is None else f"{x:.{d}f}"
    rows = []

    def add(item, paper, ours, where):
        rows.append({"Quantity": item, "Paper": paper, "Ours": ours, "Source": where})

    if v3:
        t50, t1 = v3["test_T50"], v3["test_T1_deterministic"]
        add("Test macro AUROC (T=50)", "0.9404 [0.9351, 0.9455]", fmt_ci(t50["macro_auroc"], *t50["macro_auroc_ci"]), "V3 Table 4")
        add("Test macro F1 (T=50)", "0.7308 [0.7109, 0.7497]", fmt_ci(t50["macro_f1"], *t50["macro_f1_ci"]), "V3 Table 4")
        add("ECE T=50", "0.1456 [0.1311, 0.1601]", fmt_ci(t50["ece_10bin"], *t50["ece_ci"]), "V3 Table 4")
        add("ECE T=1", "0.1891 [0.1722, 0.2057]", fmt_ci(t1["ece_10bin"], *t1["ece_ci"]), "V3 Table 4")
        u = v3["uncertainty_means"]
        add("Mean total / epistemic entropy (nats)", "0.3209 / 0.0113", f"{u['total']:.4f} / {u['epistemic']:.4f}", "V3")
        add("Melanoma sensitivity / specificity (T=50)", "0.808 / 0.962",
            f"{t50['mel_sensitivity']:.3f} / {t50['mel_specificity']:.3f}", "V3 Table 5")
        add("nv accuracy, Eq. 17 (T=50)", "0.698", f(t50["nv_accuracy_eq17"], 3), "V3 Table 5")
        ref = {round(r["referral_rate"], 2): r for r in v3["referral_test_tau"]}
        for rr, (pa, pf) in ((0.10, (0.77, 2.57)), (0.30, (1.64, 7.91))):
            add(f"Referral r={int(rr * 100)}% gain AUROC / F1", f"+{pa}% / +{pf}%",
                f"{ref[rr]['auroc_gain_pct']:+.2f}% / {ref[rr]['f1_gain_pct']:+.2f}%", "V3 Table 6")
        add("Accepted AUROC at r=30%", "0.9568 [0.9525, 0.9609]",
            fmt_ci(ref[0.30]["macro_auroc"], ref[0.30]["auroc_ci_low"], ref[0.30]["auroc_ci_high"]), "V3 Table 6")
    if v2:
        abl = {a["run_id"]: a for a in v2["ablation"]}
        if "v2-abl-nocorr" in abl:
            n = abl["v2-abl-nocorr"]
            add("Unmitigated baseline AUROC / F1", "0.8038 / 0.3600", f"{n['macro_auroc']:.4f} / {n['macro_f1']:.4f}", "V2 Table 8 (i)")
            add("Unmitigated mel / df sensitivity", "0.305 / 0.000", f"{n['mel_sensitivity']:.3f} / {n['df_sensitivity']:.3f}", "V2 Table 8 (i)")
        for rid, (pa, pf) in (("v2-abl-sampler", (1.8, 7.0)), ("v2-abl-loss", (1.0, 10.9))):
            if rid in abl:
                add(f"Both corrections vs {abl[rid]['label'].split(') ')[1]}", f"+{pa}% AUROC / +{pf}% F1",
                    f"{abl[rid]['delta_auroc_vs_main_pct']:+.1f}% / {abl[rid]['delta_f1_vs_main_pct']:+.1f}%", "V2 Table 8")
        if "v2-abl-nopretrain" in abl:
            add("ImageNet pretraining gain (AUROC)", "+6-8%", f"{abl['v2-abl-nopretrain']['delta_auroc_vs_main_pct']:+.1f}%", "V2 Table 8 (v)")
        add("Best epoch (main model)", "26", str(v2["main"]["checkpoint"]["best_epoch"]), "V2")
    if v4:
        add("Grad-CAM IoU accepted / referred", "0.61 / 0.41", f"{v4['iou_accepted']:.3f} / {v4['iou_referred']:.3f}", "V4 Table 10")
    if v5:
        for c in v5["claims"]:
            add(c["claim"], "yes", f"param ratio {c['measured param ratio']}, AUROC diff {c['AUROC B0 - other']}, "
                f"CIs overlap: {c['CIs overlap']}", "V5 Table 7")

    checks = []
    for name, m in (("V1", v1), ("V2", v2), ("V3", v3), ("V4", v4)):
        for c in (m or {}).get("sanity_checks", []):
            checks.append({"Version": name, "Check": c["check"], "Value": c["value"], "Passed": c["passed"]})

    deviations = Path("docs/deviations.md").read_text(encoding="utf-8") if Path("docs/deviations.md").exists() else ""
    lines = [
        "# Final Replication Report — Randhawa et al. (2026), BioMedInformatics 6(4):53",
        "",
        "*Skin Lesion Classification in Low-Resource Settings Using Lightweight CNNs with Uncertainty Estimation.*",
        "Generated by `python -m src.eval.run_v5` from the `metrics.json` of every version. Edge-latency benchmarks "
        "(paper §4.9, Table 9) are out of scope by team decision.",
        "",
        "## What was built",
        "",
        "| Version | Paper sections | Result package |",
        "|---|---|---|",
        "| V1 | §3.1-3.3, §3.7 | `results/v1/` — trained baseline (raw images), single-pass evaluation |",
        "| V2 | §3.2.1, §3.6, §4.1, §4.3, §4.8 | `results/v2/` — hair removal + CLAHE cache, locked `v2-main-b0`, Table 8 ablation |",
        "| V3 | §3.4, §4.2, §4.4-4.6 | `results/v3/` — MC Dropout T=50, uncertainty decomposition, calibration, referral |",
        "| V4 | §3.5, Table 10, Figs 8-11 | `results/v4/` — Grad-CAM + IoU vs HAM10000 lesion masks |",
        "| V5 | §4.7, Table 7 | `results/v5/` — MobileNetV3-Large / ResNet-50 / DenseNet-121 comparison |",
        "",
        "## Paper vs Ours",
        "",
        pd.DataFrame(rows).to_markdown(index=False) if rows else "(no results yet)",
        "",
        "## Sanity checks across versions",
        "",
        pd.DataFrame(checks).to_markdown(index=False) if checks else "(none)",
        "",
        "## Paper inconsistencies found (PLAN.md §13)",
        "",
        "1. Split lesion counts (1,891/405/405) don't add up to HAM10000's 7,470 lesions; we report our own 7002/1532/1481 images.",
        "2. The stated 67:1 imbalance ratio is 58:1 by class counts (nv 6705 / df 115).",
        "3. Fig. 4's caption (all classes > 70%) contradicts the text (nv 69.8%).",
        "4. Per-class \"accuracy\" is Eq. 17's one-vs-rest form, not recall (resolved in V1).",
        "5. The referral threshold tau is a quantile of TEST-set entropy; V3 also reports a validation-tau variant.",
        "6. Cross-reference slips (§3.2.1 cites \"Section 5\"; Fig. 8/9 cite \"Section 3.1\").",
        "7. Mask coverage: the HAM10000 Dataverse lesion masks cover all 1,481 of our test images.",
        "8. \"T=1\" is ambiguous; we use the deterministic (dropout-off) pass.",
        "9. The Grad-CAM target class isn't stated; we use the predicted class.",
        "",
        "## Limitations",
        "",
        "- Paper Tables 2, 7, 8 and 10 were not readable in our copy, so optimizer/LR/batch/weight decay are provisional "
        "(AdamW, 1e-4, 32, 1e-4); the MobileNetV3 variant and Table 10's referral rate are our defaults.",
        "- One training seed per configuration; differences smaller than the bootstrap CIs are not evidence of a real effect.",
        "- Bootstrap CIs resample images, as the paper does; images of the same lesion are correlated, so CIs are somewhat optimistic.",
        "- ResNet-50 and DenseNet-121 used mixed precision (6 GB GPU); the B0 and MobileNetV3 runs are fp32.",
        "- Grad-CAM overlap with a lesion mask is not evidence of clinically correct reasoning.",
        "",
        "## Deviation log (docs/deviations.md)",
        "",
        deviations,
    ]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 result package + final report (PLAN.md §9).")
    parser.add_argument("--inputs", type=Path, default=Path("results/v5/inputs"))
    parser.add_argument("--v2-inputs", type=Path, default=Path("results/v2/inputs"))
    parser.add_argument("--out", type=Path, default=Path("results/v5"))
    parser.add_argument("--report-out", type=Path, default=Path("results/final_report"))
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.inputs, args.v2_inputs, args.out, args.n_resamples, args.seed)
    final_report(args.report_out)


if __name__ == "__main__":
    main()

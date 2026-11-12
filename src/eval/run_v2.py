"""V2 result package — paper-faithful training + imbalance ablation
(PLAN.md §6, paper §3.2.1, §3.6, §4.1, §4.3, §4.8).

    python -m src.eval.run_v2 [--qc]        # --qc needs data/raw + data/cache_prep224

Reads per-run C3/C5/C6 artifacts copied to `results/v2/inputs/<run_id>/`
and writes `results/v2/`: the main-model single-pass package (Figs 2, 4, 5,
Tables 3, 5), Table 8 ablation with deltas vs the paper, the V1->V2
consistency check, and (with --qc) the preprocessing QC figures.
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
from src.eval.common import evaluate_file, fmt_ci, run_package, write_json
from src.eval.style import annotate_bars, finish_figure, grid_y, new_figure

MAIN = "v2-main-b0"
ABLATIONS = [
    ("v2-abl-nocorr", "(i) no correction", False, False, True),
    ("v2-abl-sampler", "(ii) sampler only", True, False, True),
    ("v2-abl-loss", "(iii) loss weights only", False, True, True),
    (MAIN, "(iv) sampler + loss weights (MAIN)", True, True, True),
    ("v2-abl-nopretrain", "(v) both, no ImageNet pretraining", True, True, False),
]
# Paper §4.2-4.3 / §4.8 (text): the unmitigated baseline and the relative gains of (iv).
PAPER_NOCORR = {"macro_auroc": 0.8038, "macro_f1": 0.3600, "mel_sensitivity": 0.305, "df_sensitivity": 0.0}
PAPER_GAIN_VS = {"v2-abl-sampler": (1.8, 7.0), "v2-abl-loss": (1.0, 10.9)}


# ------------------------------------------------------------- QC figures
def qc_figures(raw_dir: Path, cache_dir: Path, out_dir: Path, seed: int = 42) -> dict:
    import cv2
    from PIL import Image

    from src.data.preprocess import clahe_lab, hair_mask, remove_hair

    manifest = pd.read_csv(cache_dir / "manifest.csv")
    hairy = manifest.sort_values("hair_fraction", ascending=False)
    rng = np.random.default_rng(seed)
    # Weighted toward hairy images (PLAN.md §6.3 item 3): 4 of the top-2% hair images + 2 random.
    pick = list(rng.choice(hairy["image_id"].head(len(hairy) // 50), 4, replace=False)) + \
        list(rng.choice(manifest["image_id"], 2, replace=False))

    fig, axes = new_figure(len(pick), 4, figsize=(13, 3.1 * len(pick)))
    for r, image_id in enumerate(pick):
        rgb = np.asarray(Image.open(raw_dir / "images" / f"{image_id}.jpg").convert("RGB"))
        mask = hair_mask(rgb)
        no_hair, _ = remove_hair(rgb)
        final = np.asarray(Image.open(cache_dir / f"{image_id}.png"))
        overlay = rgb.copy()
        overlay[mask > 0] = (255, 0, 0)
        hf = float(manifest.loc[manifest["image_id"] == image_id, "hair_fraction"].iloc[0])
        for c, (img, title) in enumerate(((rgb, f"{image_id} original"), (overlay, f"hair mask {100 * hf:.1f}%"),
                                          (no_hair, "after TELEA inpaint"), (final, "after CLAHE + resize 224"))):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    finish_figure(fig, out_dir / "figures" / "preprocessing_before_after.png",
                  "Rows 1-4: among the hairiest 2% of images; rows 5-6: random. Red = blackhat hair mask "
                  "(17x17 disk, threshold 10). CLAHE on each LAB channel (clip 2.0, 8x8).")

    fig, ax = new_figure(figsize=(8, 4.2))
    ax.hist(100 * manifest["hair_fraction"], bins=60, color=SINGLE_SERIES_BLUE)
    ax.set_xlabel("Hair-mask pixel fraction (%)")
    ax.set_ylabel("Images")
    ax.set_title("Hair-mask coverage across all 10,015 images", fontsize=10, color=CHART_INK["primary"])
    grid_y(ax)
    finish_figure(fig, out_dir / "figures" / "hair_fraction_histogram.png",
                  f"Median {100 * manifest['hair_fraction'].median():.1f}%, 95th percentile "
                  f"{100 * manifest['hair_fraction'].quantile(0.95):.1f}%. The paper's threshold of 10 also marks fine "
                  "dark texture and lesion borders, not only hairs — the inpainting smooths those too.")

    # Geometry check (V4 mask alignment depends on it): cache is exactly 224x224 everywhere.
    sizes = {Image.open(cache_dir / f"{i}.png").size for i in rng.choice(manifest["image_id"], 200, replace=False)}
    stats = {
        "n_images": int(len(manifest)), "n_ok": int((manifest["status"] == "ok").sum()),
        "hair_fraction_mean": float(manifest["hair_fraction"].mean()),
        "hair_fraction_median": float(manifest["hair_fraction"].median()),
        "hair_fraction_p95": float(manifest["hair_fraction"].quantile(0.95)),
        "processing_time_s_mean": float(manifest["processing_time_s"].mean()),
        "processing_time_s_total_cpu": float(manifest["processing_time_s"].sum()),
        "cache_sizes_sampled": sorted(sizes),
    }
    write_json(stats, out_dir / "preprocessing_stats.json")
    return stats


# ---------------------------------------------------------------- ablation
def plot_ablation(rows: pd.DataFrame, out_path: Path) -> Path:
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = new_figure(figsize=(12, 5))
    b1 = ax.bar(x - width / 2, rows["macro_auroc"], width, color=SINGLE_SERIES_BLUE, label="Macro AUROC")
    b2 = ax.bar(x + width / 2, rows["macro_f1"], width, color=SERIES_2, label="Macro F1")
    annotate_bars(ax, b1, rows["macro_auroc"], fmt="{:.3f}", fontsize=7)
    annotate_bars(ax, b2, rows["macro_f1"], fmt="{:.3f}", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (MAIN)", "\n(MAIN)").replace(", no", ",\nno") for l in rows["label"]], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Test score (single pass)")
    ax.set_title("Imbalance ablation (paper Table 8): what each correction contributes", fontsize=10,
                 color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    grid_y(ax)
    caption = ("All runs share the C9 cache, split, augmentation, provisional hyperparameters, seed 42 and 30 epochs; "
               "only the named factor differs. Single-pass (dropout off) test evaluation.")
    return finish_figure(fig, out_path, caption)


def run(inputs: Path, out_dir: Path, n_resamples: int, seed: int, v1_inputs: Path) -> dict:
    evals, histories, ckpts = {}, {}, {}
    for run_id, *_ in ABLATIONS:
        d = inputs / run_id
        if not (d / "predictions_test.csv").exists():
            print(f"  [skip] {run_id}: no predictions yet")
            continue
        print(f"  evaluating {run_id}")
        evals[run_id] = evaluate_file(d / "predictions_test.csv", n_resamples, seed)
        histories[run_id] = pd.read_csv(d / "history.csv")
        ckpts[run_id] = json.loads((d / "checkpoint.json").read_text())

    main_ev = evals[MAIN]
    best = int(ckpts[MAIN]["best_epoch"])
    run_package(main_ev, histories[MAIN], best, out_dir, "v2-main-b0 (single pass)")

    rows = []
    for run_id, label, sampler, weights, pretrained in ABLATIONS:
        if run_id not in evals:
            continue
        h = evals[run_id].headline()
        rows.append({"run_id": run_id, "label": label, "sampler": sampler, "class_weighted_loss": weights,
                     "imagenet_pretrained": pretrained, "best_epoch": int(ckpts[run_id]["best_epoch"]), **h})
    abl = pd.DataFrame(rows)
    m = abl.set_index("run_id").loc[MAIN]
    abl["delta_auroc_vs_main_pct"] = 100 * (m["macro_auroc"] - abl["macro_auroc"]) / abl["macro_auroc"]
    abl["delta_f1_vs_main_pct"] = 100 * (m["macro_f1"] - abl["macro_f1"]) / abl["macro_f1"]
    plot_ablation(abl, out_dir / "figures" / "table8_ablation.png")

    table8 = pd.DataFrame([{
        "Configuration": r["label"],
        "Macro AUROC [95% CI]": fmt_ci(r["macro_auroc"], *r["macro_auroc_ci"]),
        "Macro F1 [95% CI]": fmt_ci(r["macro_f1"], *r["macro_f1_ci"]),
        "Mel sens.": f"{r['mel_sensitivity']:.3f}",
        "Df sens.": f"{r['df_sensitivity']:.3f}",
        "Nv acc. (Eq. 17)": f"{r['nv_accuracy_eq17']:.3f}",
        "MAIN gain AUROC / F1": "—" if r["run_id"] == MAIN else
            f"{r['delta_auroc_vs_main_pct']:+.1f}% / {r['delta_f1_vs_main_pct']:+.1f}%",
        "Paper gain AUROC / F1": (f"+{PAPER_GAIN_VS[r['run_id']][0]}% / +{PAPER_GAIN_VS[r['run_id']][1]}%"
                                  if r["run_id"] in PAPER_GAIN_VS else "—"),
    } for _, r in abl.iterrows()])

    nocorr_cmp = None
    if "v2-abl-nocorr" in evals:
        n = abl.set_index("run_id").loc["v2-abl-nocorr"]
        nocorr_cmp = pd.DataFrame([{"Metric": k, "Ours (i) no correction": f"{n[k]:.4f}", "Paper pre-fix baseline": v}
                                   for k, v in PAPER_NOCORR.items()])

    # V1 -> V2 internal consistency (pipeline check, not a new result).
    v1 = evaluate_file(v1_inputs / "predictions_test.csv", n_resamples, seed).headline()
    m_h = main_ev.headline()
    consistency = pd.DataFrame([{"Metric": k, "V1 v1-b0-baseline (raw JPEG)": f"{v1[k]:.4f}",
                                 "V2 v2-main-b0 (hair removal + CLAHE)": f"{m_h[k]:.4f}",
                                 "Difference": f"{m_h[k] - v1[k]:+.4f}"}
                                for k in ("macro_auroc", "macro_f1", "balanced_accuracy", "ece_10bin", "brier",
                                          "mel_sensitivity", "df_sensitivity", "nv_accuracy_eq17")])

    tables = {
        "table8_ablation": table8,
        "table5_per_class_main": T.per_class_markdown_table(main_ev.per_class),
        "table3_training_milestones_main": T.training_milestones_table(histories[MAIN], best),
        "v1_vs_v2_consistency": consistency,
    }
    if nocorr_cmp is not None:
        tables["nocorr_vs_paper_baseline"] = nocorr_cmp
    T.write_tables(tables, out_dir / "tables")
    abl.to_csv(out_dir / "tables" / "ablation_raw.csv", index=False)

    h = histories[MAIN]
    checks = [
        {"check": "Run (i) shows minority-class collapse (df sensitivity near 0)",
         "value": f"df sens {abl.set_index('run_id').loc['v2-abl-nocorr', 'df_sensitivity']:.3f}" if "v2-abl-nocorr" in evals else "n/a",
         "passed": bool("v2-abl-nocorr" in evals and abl.set_index('run_id').loc['v2-abl-nocorr', 'df_sensitivity'] < 0.3)},
        {"check": "(iv) beats (ii) and (iii) on macro F1",
         "value": ", ".join(f"{r}: {abl.set_index('run_id').loc[r, 'macro_f1']:.4f}" for r in (MAIN, "v2-abl-sampler", "v2-abl-loss") if r in evals),
         "passed": bool(all(m["macro_f1"] > abl.set_index('run_id').loc[r, "macro_f1"] for r in ("v2-abl-sampler", "v2-abl-loss") if r in evals))},
        {"check": "Train loss drops sharply in first 5 epochs (paper 0.87 -> 0.15)",
         "value": f"{h['train_loss'].iloc[0]:.4f} -> {h['train_loss'].iloc[4]:.4f}",
         "passed": bool(h["train_loss"].iloc[4] < 0.5 * h["train_loss"].iloc[0])},
    ]

    payload = {"main": {"run_id": MAIN, "checkpoint": ckpts[MAIN], **m_h, "bootstrap": main_ev.boot.as_dict(),
                        "per_class": main_ev.per_class.to_dict(orient="records")},
               "ablation": abl.to_dict(orient="records"), "v1_reference": v1, "checkpoints": ckpts,
               "sanity_checks": checks}
    write_json(payload, out_dir / "metrics.json")
    (out_dir / "summary.md").write_text(build_summary(payload, tables, checks, h, best), encoding="utf-8")
    print(f"Wrote {out_dir}; sanity checks {sum(c['passed'] for c in checks)}/{len(checks)} pass")
    return payload


def build_summary(p, tables, checks, history, best) -> str:
    m = p["main"]
    stats_path = Path("results/v2/preprocessing_stats.json")
    pre = json.loads(stats_path.read_text()) if stats_path.exists() else None
    lines = [
        "# V2 — Paper-Faithful Training + Imbalance Ablation",
        "",
        "Generated by `python -m src.eval.run_v2` from `results/v2/inputs/<run_id>/` (C3/C5/C6 per run).",
        "All V2 runs: EfficientNet-B0 + dropout head on the C9 hair-removed + CLAHE cache, frozen V1 split, "
        "paper augmentation, AdamW lr 1e-4 / wd 1e-4 / batch 32 / cosine / 30 epochs (PROVISIONAL Table 2 values — "
        "the paper's table was not accessible), seed 42, checkpoint = best validation macro F1. Single-pass evaluation; "
        "the T=50 headline is V3.",
        "",
        "## Preprocessing (C9)",
        "",
    ]
    if pre:
        lines += [f"{pre['n_ok']}/{pre['n_images']} images processed. Hair-mask fraction: mean "
                  f"{100 * pre['hair_fraction_mean']:.1f}%, median {100 * pre['hair_fraction_median']:.1f}%, 95th pct "
                  f"{100 * pre['hair_fraction_p95']:.1f}%. Mean {1000 * pre['processing_time_s_mean']:.0f} ms/image. "
                  f"Cache geometry: {pre['cache_sizes_sampled']}. Figures: `figures/preprocessing_before_after.png`, "
                  "`figures/hair_fraction_histogram.png`.", ""]
    lines += [
        f"## Main model `v2-main-b0` (locked for V3/V4/V5)",
        "",
        f"Best epoch {best}/30, val macro F1 {m['checkpoint']['selection_metric_value']:.4f}, "
        f"checkpoint sha256 `{m['checkpoint']['best_checkpoint_sha256']}`.",
        "",
        f"Test (single pass): macro AUROC **{fmt_ci(m['macro_auroc'], *m['macro_auroc_ci'])}**, macro F1 "
        f"**{fmt_ci(m['macro_f1'], *m['macro_f1_ci'])}**, balanced accuracy {m['balanced_accuracy']:.4f}, ECE "
        f"{m['ece_10bin']:.4f}, Brier {m['brier']:.4f}, melanoma sensitivity {m['mel_sensitivity']:.3f} / specificity "
        f"{m['mel_specificity']:.3f}.",
        "",
        "### Table 3 — training milestones",
        "",
        tables["table3_training_milestones_main"].to_markdown(index=False),
        "",
        "### Table 5 — per-class (single pass)",
        "",
        tables["table5_per_class_main"].to_markdown(index=False),
        "",
        "## Table 8 — imbalance ablation",
        "",
        tables["table8_ablation"].to_markdown(index=False),
        "",
        "\"MAIN gain\" = relative improvement of (iv) over that row; the paper reports (iv) over sampler-only "
        "+1.8% AUROC / +7.0% F1 and over loss-only +1.0% / +10.9%.",
        "",
    ]
    if "nocorr_vs_paper_baseline" in tables:
        lines += ["### Run (i) vs the paper's unmitigated baseline", "",
                  tables["nocorr_vs_paper_baseline"].to_markdown(index=False), ""]
    lines += [
        "## V1 -> V2 consistency check (pipeline check, not a new result)",
        "",
        tables["v1_vs_v2_consistency"].to_markdown(index=False),
        "",
        "## Sanity checks (PLAN.md §6.5)",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
        "## Figures",
        "",
        "- `figures/training_curves.png` (Fig. 2), `figures/per_class.png` (Fig. 4), `figures/confusion_matrix.png` "
        "(Fig. 5), `figures/roc_curves.png`",
        "- `figures/table8_ablation.png`",
        "- `figures/preprocessing_before_after.png`, `figures/hair_fraction_histogram.png`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 result package (PLAN.md §6).")
    parser.add_argument("--inputs", type=Path, default=Path("results/v2/inputs"))
    parser.add_argument("--out", type=Path, default=Path("results/v2"))
    parser.add_argument("--v1-inputs", type=Path, default=Path("results/v1/inputs"))
    parser.add_argument("--qc", action="store_true", help="also build preprocessing QC figures (needs data/)")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache_prep224"))
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.qc:
        qc_figures(args.raw_dir, args.cache_dir, args.out, args.seed)
    run(args.inputs, args.out, args.n_resamples, args.seed, args.v1_inputs)


if __name__ == "__main__":
    main()

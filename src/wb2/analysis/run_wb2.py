"""Replay every compared method on one split and write the WB2 evidence
package (implementation.md §3 C4–C7, WB2-O1/O2).

    python -m src.wb2.analysis.run_wb2 --bank-dir <bank dir> --split val \
        --config configs/wb2_controller.yaml --status MEASURED

Guards (§0 rules 2–4):
  - the config must be `locked: true` and tuned on `val`;
  - on `test`, the config's val bank must come from the same checkpoint;
  - the calibration passed here must be the one the config was tuned with;
  - a SYNTHETIC bank can never write into results/wb2/;
  - a pilot subset (`--allow-subset`) must be labeled PILOT ONLY.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.wb2.analysis import evaluate as E
from src.wb2.analysis import figures as F
from src.wb2.analysis.paired_bootstrap import DEFAULT_N_RESAMPLES, paired_bootstrap
from src.wb2.controller.bank import load_bank
from src.wb2.controller.replay import (
    attach_labels,
    head_and_backbone_passes,
    load_temperature,
    policies_from_config,
    replay_bank,
)
from src.wb2.provenance import environment, git_state, now_iso

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "results" / "wb2"
STATUSES = ("MEASURED", "PILOT ONLY")
FIXED_METHODS = ("det_T1", "fixed_T10", "fixed_T20", "fixed_T50")
PAIRS = (
    ("fixed_T50", "risk_aware_adaptive"),
    ("fixed_T50", "variance_adaptive"),
    ("fixed_T50+referral", "risk_aware_adaptive"),
    ("variance_adaptive", "risk_aware_adaptive"),
    ("det_T1", "fixed_T50"),
)
HEADLINE_COLUMNS = [
    "method", "status", "n", "macro_auroc", "macro_f1", "balanced_accuracy", "ece_10bin", "brier", "nll",
    "mel_recall", "suspicious_recall", "passes_mean", "passes_median", "passes_p95", "pct_at_cap",
    "review_rate", "coverage", "accepted_accuracy", "mel_miss_all_str", "mel_miss_accepted_str",
    "suspicious_false_reassurance_str",
]


def check_config(cfg: dict, bank, cal_id: str) -> None:
    if not cfg.get("locked") or cfg.get("tuned_on_split") != "val":
        raise SystemExit("config is not a locked, validation-tuned config — run src.wb2.controller.tune first")
    if cfg.get("calibration", {}).get("calibration_id") != cal_id:
        raise SystemExit(
            f"config was tuned with calibration {cfg.get('calibration', {}).get('calibration_id')!r} "
            f"but this run uses {cal_id!r}; thresholds would not transfer")
    if cfg["val_bank"].get("synthetic") != bank.is_synthetic:
        raise SystemExit("config tuned on a synthetic bank cannot be applied to a real bank (or vice versa)")
    if bank.split == "test" and cfg["val_bank"].get("checkpoint_sha256") != bank.manifest.get("checkpoint_sha256"):
        raise SystemExit("val (tuning) and test banks come from different checkpoints")


def write_report(path: Path, split: str, status: str, bank, cfg: dict, table: pd.DataFrame,
                 paired: pd.DataFrame, class_counts: dict, cal_id: str) -> None:
    head = table[[c for c in HEADLINE_COLUMNS if c in table.columns]]
    lines = [
        f"# WB2 controller report — {split} ({status})",
        "",
        f"Generated {now_iso()} by `python -m src.wb2.analysis.run_wb2`. Auto-generated — the curated",
        "Workbook text lives in `results/wb2/summary.md`.",
        "",
        f"- Images: **{bank.n_images}**; class counts {class_counts}",
        f"- Checkpoint SHA256: `{bank.manifest.get('checkpoint_sha256')}`",
        f"- Calibration: `{cal_id}`; config tuned on val at {cfg.get('tuned_at')} "
        f"(risk-aware feasible on val: {cfg['val_selection']['risk_aware_feasible']})",
        f"- Feature reuse in sampling: {bank.manifest.get('feature_reuse', 'unknown')} — pass counts are "
        "stochastic *head* draws; savings are algorithmic (offline replay), not measured wall-clock.",
        "- TRUSTED RESULT means the model's computational checks passed, never 'clinically safe/benign'. "
        "Explanation verifier: NOT RUN (planned V4).",
        "",
        "## Method comparison",
        "",
        head.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired lesion-grouped bootstrap (b − a, 95% percentile CI)",
        "",
        paired.to_markdown(index=False, floatfmt=".4f") if len(paired) else "_not run_",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="WB2 offline policy replay + evaluation package.")
    parser.add_argument("--bank-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "wb2_controller.yaml")
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--temperature-json", type=Path, default=None)
    parser.add_argument("--status", choices=STATUSES, default="MEASURED")
    parser.add_argument("--allow-subset", action="store_true", help="predeclared pilot subset (requires --status 'PILOT ONLY')")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_RESAMPLES)
    args = parser.parse_args()

    started = now_iso()
    split_csv = args.split_csv or REPO / "data" / "splits" / f"{args.split}.csv"
    if args.allow_subset and args.status != "PILOT ONLY":
        raise SystemExit("a subset run must be labeled --status 'PILOT ONLY'")
    bank = load_bank(args.bank_dir, args.split, split_csv=split_csv,
                     allow_partial=args.allow_partial, allow_subset=args.allow_subset)
    status = args.status
    if bank.is_synthetic:
        if args.out_dir.resolve() == DEFAULT_OUT.resolve():
            raise SystemExit("refusing to write SYNTHETIC-bank outputs into results/wb2/")
        status = "SYNTHETIC - NOT MODEL RESULTS"

    tau, cal_id = load_temperature(args.temperature_json)
    cfg = yaml.safe_load(args.config.read_text())
    check_config(cfg, bank, cal_id)

    decisions = replay_bank(bank, policies_from_config(cfg), tau)
    decisions = head_and_backbone_passes(decisions, bank.manifest)
    run_id = f"wb2_{args.split}_{cal_id}"
    decisions.insert(0, "run_id", run_id)
    decisions.insert(1, "checkpoint_sha256", bank.manifest.get("checkpoint_sha256"))
    decisions.insert(2, "calibration_id", cal_id)
    labeled = attach_labels(decisions, bank)

    out = args.out_dir
    tables, figs = out / "tables", out / "figures"
    for d in (out / "decisions", tables, figs):
        d.mkdir(parents=True, exist_ok=True)
    sfx = args.split
    labeled.to_csv(out / "decisions" / f"decisions_{sfx}.csv.gz", index=False)

    table = E.comparison_table(labeled, status)
    table.to_csv(tables / f"policy_comparison_{sfx}.csv", index=False)
    table[table["method"].isin(FIXED_METHODS)].to_csv(tables / f"fixed_budget_{sfx}.csv", index=False)
    E.stop_distribution(labeled).to_csv(tables / f"stop_distribution_{sfx}.csv", index=False)
    E.tier_distribution(labeled).to_csv(tables / f"tier_distribution_{sfx}.csv", index=False)
    E.melanoma_audit(labeled, "risk_aware_adaptive").to_csv(tables / f"melanoma_audit_{sfx}.csv", index=False)
    E.confident_errors(labeled, "risk_aware_adaptive").to_csv(tables / f"confident_errors_{sfx}.csv", index=False)
    curve = E.coverage_risk_curve(labeled[labeled["method"] == "fixed_T50"])
    curve.to_csv(tables / f"coverage_risk_T50_{sfx}.csv", index=False)

    paired_rows = []
    if args.n_bootstrap > 0:
        for a, b in PAIRS:
            da, db = labeled[labeled["method"] == a], labeled[labeled["method"] == b]
            if len(da) and len(db):
                paired_rows.append(paired_bootstrap(da, db, n_resamples=args.n_bootstrap))
    paired = pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    paired.to_csv(tables / f"paired_bootstrap_{sfx}.csv", index=False)

    F.plot_tier_distribution(labeled, figs / f"controller_tiers_{sfx}.png", status, sfx)
    F.plot_stop_distribution(labeled, figs / f"controller_stop_passes_{sfx}.png", status, sfx)
    F.plot_compute_vs_risk(table, figs / f"controller_compute_vs_risk_{sfx}.png", status, sfx)
    F.plot_coverage_vs_risk(curve, table, figs / f"controller_coverage_vs_risk_{sfx}.png", status, sfx)
    F.plot_reliability(labeled, figs / f"controller_reliability_{sfx}.png", status, sfx)

    class_counts = E.class_counts(labeled)
    write_report(out / f"report_{sfx}.md", sfx, status, bank, cfg, table, paired, class_counts, cal_id)
    manifest = {
        "run_id": run_id,
        "status": status,
        "split": sfx,
        "n_images": bank.n_images,
        "class_counts": class_counts,
        "bank_dir": str(args.bank_dir),
        "bank_files_sha256": bank.files_sha256,
        "bank_manifest": bank.manifest,
        "nonfinite_rows": len(bank.nonfinite_rows),
        "config": str(args.config),
        "config_tuned_at": cfg.get("tuned_at"),
        "calibration_id": cal_id,
        "temperature": tau,
        "bootstrap": {"n_resamples": args.n_bootstrap, "seed": 42, "unit": "lesion_id"},
        "git": git_state(),
        "environment": environment(),
        "started_at": started,
        "finished_at": now_iso(),
    }
    (out / f"manifest_{sfx}.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[{status}] wrote {sfx} package for {bank.n_images} images to {out}")


if __name__ == "__main__":
    main()

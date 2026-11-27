"""Matched-coverage comparison: fixed T=50 + entropy referral whose threshold is
set on **val** to the risk-aware controller's val review rate, then applied
unchanged to test (implementation.md §3 C3 "matched coverage").

    python -m src.wb2.analysis.matched_coverage --bank-dir <bank dir> \
        --config configs/wb2_controller.yaml --temperature-json <calibration.json> \
        --out-dir results/wb2/calibrated

Disclosure: this comparison was added after the first test replay, because the
20%-budget referral baseline reviewed more images than the controller. Its
threshold is a function of the val bank and the already-locked config only;
no test quantity enters it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.wb2.analysis.evaluate import method_summary
from src.wb2.analysis.paired_bootstrap import DEFAULT_N_RESAMPLES, paired_bootstrap
from src.wb2.controller.bank import load_bank
from src.wb2.controller.features import prefix_stats
from src.wb2.controller.policies import FixedT, RiskAware
from src.wb2.controller.replay import attach_labels, load_temperature
from src.wb2.controller.tune import entropy_threshold_for_budget

REPO = Path(__file__).resolve().parents[3]
METHOD = "fixed_T50+referral@matched"


def _labeled(policy, stats, bank, name=None) -> pd.DataFrame:
    d = policy.decide(stats)
    if name:
        d["method"] = name
    d.insert(0, "image_id", bank.ids["image_id"].to_numpy())
    return attach_labels(d, bank)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bank-dir", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--temperature-json", type=Path, default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_RESAMPLES)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tau, cal_id = load_temperature(args.temperature_json)
    if not cfg.get("locked") or cfg["calibration"]["calibration_id"] != cal_id:
        raise SystemExit("needs the locked config tuned with this same calibration")
    val = load_bank(args.bank_dir, "val", split_csv=REPO / "data" / "splits" / "val.csv")
    test = load_bank(args.bank_dir, "test", split_csv=REPO / "data" / "splits" / "test.csv")
    risk = RiskAware(**cfg["risk_aware_adaptive"])

    val_stats = prefix_stats(val.logits, tau)
    val_risk_review = float(risk.decide(val_stats)["review"].mean())
    theta = entropy_threshold_for_budget(val_stats.entropy[:, -1], val_risk_review)

    rows, paired = [], []
    for split, bank in (("val", val), ("test", test)):
        stats = val_stats if split == "val" else prefix_stats(test.logits, tau)
        base = _labeled(FixedT(50, referral_entropy=theta), stats, bank, METHOD)
        ctrl = _labeled(risk, stats, bank)
        for d in (base, ctrl):
            rows.append({"split": split, "method": d["method"].iloc[0], "status": "MEASURED",
                         "referral_entropy_from_val": theta, "val_budget": val_risk_review, **method_summary(d)})
        if args.n_bootstrap > 0:
            paired.append(paired_bootstrap(base, ctrl, n_resamples=args.n_bootstrap).assign(split=split))

    tables = args.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables / "matched_coverage.csv", index=False)
    if paired:
        pd.concat(paired, ignore_index=True).to_csv(tables / "matched_coverage_paired_bootstrap.csv", index=False)
    print(f"matched-coverage tables written to {tables} (val budget {val_risk_review:.4f}, theta {theta:.4f})")


if __name__ == "__main__":
    main()

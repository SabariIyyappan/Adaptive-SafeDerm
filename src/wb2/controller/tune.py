"""Validation-only threshold selection -> locked controller config
(implementation.md §0 rule 4, §3 C4).

    python -m src.wb2.controller.tune --bank-dir <dir with val_* files> \
        --out configs/wb2_controller.yaml --grid-out results/wb2/tables/tuning_grid_val.csv

Only the `val` bank is ever loaded here. The objective is predeclared below
and written into the config so it is fixed before any test replay:

  1. Review budget R (default 0.20). The fixed-T50 baseline and the variance
     comparator get an entropy referral threshold that refers the top-R most
     uncertain val images (matched coverage).
  2. Variance comparator: smallest mean passes with val macro F1 (all images,
     no referral) >= T50 macro F1 - f1_tolerance.
  3. Risk-aware controller: smallest mean passes subject to
       val review rate <= R,
       val accepted-melanoma misses / all mel <= the T50+referral baseline's,
       val macro F1 (all images) >= T50 macro F1 - f1_tolerance.
     Ties -> fewer melanoma misses, then lower review rate. If nothing is
     feasible the least-violating config is kept and `feasible: false` is
     recorded — reported, not hidden.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.wb2.analysis.evaluate import method_summary
from src.wb2.controller.bank import LogitBank, load_bank
from src.wb2.controller.features import prefix_stats
from src.wb2.controller.policies import FixedT, RiskAware, VarianceConvergence
from src.wb2.controller.replay import load_temperature
from src.wb2.provenance import git_state, now_iso

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "configs" / "wb2_controller.yaml"

DEFAULT_REVIEW_BUDGET = 0.20
DEFAULT_F1_TOLERANCE = 0.01

# Grid v2 (2026-10-27): v1's val optimum sat on the edges (h_hi=0.7, h_lo=0.15,
# delta_h=0.05, eps_p=0.05), so the ranges were widened once — on val, before
# any test replay — and then frozen.
VARIANCE_GRID = {
    "eps_p": [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15],
    "eps_var": [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02],
}
RISK_GRID = {
    "tau_s": [0.1, 0.2, 0.3, 0.4, 0.5],
    "h_lo": [0.05, 0.1, 0.15, 0.25, 0.35],
    "h_hi": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "margin": [0.0, 0.05, 0.1],
    "delta_h": [0.01, 0.02, 0.05, 0.1, 0.2],
}
FIXED_PARAMS = {"min_passes": 10, "min_passes_high": 20}
OBJECTIVE_TEXT = (
    "Risk-aware: minimize val mean passes s.t. val review rate <= review_budget, val accepted-mel misses / "
    "all mel <= fixed-T50+entropy-referral at the same budget, val macro F1 >= T50 macro F1 - f1_tolerance; "
    "ties -> fewer mel misses, lower review. Variance comparator: minimize val mean passes s.t. macro F1 "
    "constraint; referral threshold refers the top review_budget by final entropy."
)


def entropy_threshold_for_budget(entropy: np.ndarray, budget: float) -> float:
    """Threshold theta such that `entropy >= theta` refers ~budget of the images."""
    m = int(np.floor(budget * len(entropy)))
    if m <= 0:
        return float("inf")
    return float(np.sort(entropy)[::-1][m - 1])


def _labeled(decisions: pd.DataFrame, bank: LogitBank) -> pd.DataFrame:
    # Decisions come back row-aligned with bank.ids; labels joined only to score.
    out = decisions.copy()
    out["y_true"] = bank.y_true
    return out


def tune(bank: LogitBank, tau: float = 1.0, review_budget: float = DEFAULT_REVIEW_BUDGET,
         f1_tolerance: float = DEFAULT_F1_TOLERANCE, variance_grid: dict = VARIANCE_GRID,
         risk_grid: dict = RISK_GRID) -> tuple[dict, pd.DataFrame]:
    if bank.split != "val":
        raise ValueError(f"tuning is validation-only; got a '{bank.split}' bank")
    stats = prefix_stats(bank.logits, tau)

    t50 = method_summary(_labeled(FixedT(50).decide(stats), bank))
    theta50 = entropy_threshold_for_budget(stats.entropy[:, -1], review_budget)
    t50_ref = method_summary(_labeled(FixedT(50, referral_entropy=theta50).decide(stats), bank))
    f1_floor = t50["macro_f1"] - f1_tolerance
    mel_ceiling = t50_ref["mel_miss_all"]

    grid_rows = []
    best_var = None
    for eps_p, eps_var in itertools.product(variance_grid["eps_p"], variance_grid["eps_var"]):
        policy = VarianceConvergence(eps_p=eps_p, eps_var=eps_var, min_passes=FIXED_PARAMS["min_passes"])
        s = method_summary(_labeled(policy.decide(stats), bank))
        ok = s["macro_f1"] >= f1_floor
        grid_rows.append({"policy": "variance_adaptive", "eps_p": eps_p, "eps_var": eps_var, "feasible": ok, **_brief(s)})
        key = (not ok, s["passes_mean"], -s["macro_f1"])
        if best_var is None or key < best_var[0]:
            best_var = (key, policy)
    var_policy = best_var[1]
    var_dec = var_policy.decide(stats)
    var_theta = entropy_threshold_for_budget(var_dec["entropy"].to_numpy(), review_budget)

    best_risk = None
    keys = list(risk_grid)
    for values in itertools.product(*(risk_grid[k] for k in keys)):
        params = dict(zip(keys, values))
        if params["h_lo"] >= params["h_hi"]:
            continue
        policy = RiskAware(**params, **FIXED_PARAMS)
        s = method_summary(_labeled(policy.decide(stats), bank))
        violations = (
            max(0.0, s["mel_miss_all"] - mel_ceiling),
            max(0.0, s["review_rate"] - review_budget),
            max(0.0, f1_floor - s["macro_f1"]),
        )
        ok = not any(violations)
        grid_rows.append({"policy": "risk_aware_adaptive", **params, "feasible": ok, **_brief(s)})
        key = (not ok, violations if not ok else (0, 0, 0), s["passes_mean"], s["mel_miss_all"], s["review_rate"])
        if best_risk is None or key < best_risk[0]:
            best_risk = (key, policy, s, ok)
    _, risk_policy, risk_summary, risk_ok = best_risk

    config = {
        "locked": True,
        "tuned_on_split": "val",
        "tuned_at": now_iso(),
        "git": git_state(),
        "val_bank": {
            "files_sha256": bank.files_sha256,
            "checkpoint_sha256": bank.manifest.get("checkpoint_sha256"),
            "n_images": bank.n_images,
            "synthetic": bank.is_synthetic,
        },
        "objective": {
            "review_budget": review_budget,
            "f1_tolerance": f1_tolerance,
            "text": OBJECTIVE_TEXT,
        },
        "fixed_T50_referral": {"referral_entropy": theta50},
        "variance_adaptive": {**var_policy.params(), "referral_entropy": var_theta},
        "risk_aware_adaptive": risk_policy.params(),
        "val_selection": {
            "risk_aware_feasible": bool(risk_ok),
            "T50": _brief(t50),
            "T50_referral": _brief(t50_ref),
            "risk_aware": _brief(risk_summary),
            "grid_size": len(grid_rows),
        },
    }
    return config, pd.DataFrame(grid_rows)


def _brief(s: dict) -> dict:
    keys = ["macro_f1", "macro_auroc", "passes_mean", "review_rate", "mel_miss_all", "mel_missed_accepted", "mel_total"]
    return {k: (float(s[k]) if isinstance(s[k], (float, np.floating)) else int(s[k])) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only tuning of WB2 stop/review thresholds.")
    parser.add_argument("--bank-dir", required=True, type=Path)
    parser.add_argument("--split-csv", type=Path, default=REPO / "data" / "splits" / "val.csv")
    parser.add_argument("--temperature-json", type=Path, default=None, help="WB2-I4 record; omit = uncalibrated")
    parser.add_argument("--review-budget", type=float, default=DEFAULT_REVIEW_BUDGET)
    parser.add_argument("--f1-tolerance", type=float, default=DEFAULT_F1_TOLERANCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--grid-out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing locked config (only before any test replay)")
    args = parser.parse_args()

    bank = load_bank(args.bank_dir, "val", split_csv=args.split_csv)
    if bank.is_synthetic and args.out.resolve() == DEFAULT_CONFIG.resolve():
        raise SystemExit("refusing to write a config tuned on a SYNTHETIC bank to configs/wb2_controller.yaml")
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; it may already have been used on test. Pass --overwrite only if it has not.")

    tau, cal_id = load_temperature(args.temperature_json)
    config, grid = tune(bank, tau, args.review_budget, args.f1_tolerance)
    config["calibration"] = {"calibration_id": cal_id, "temperature": tau, "applied_to": "per_pass_logits_before_softmax"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote locked config to {args.out} (risk-aware feasible on val: {config['val_selection']['risk_aware_feasible']})")
    if args.grid_out:
        args.grid_out.parent.mkdir(parents=True, exist_ok=True)
        grid.to_csv(args.grid_out, index=False)
        print(f"Wrote {len(grid)} grid rows to {args.grid_out}")


if __name__ == "__main__":
    main()

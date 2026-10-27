"""Member A's fixed-budget cross-check on a delivered bank (implementation.md E0/E1).

    python -m src.wb2.sampling.fixed_budget --bank-dir results/wb2/inputs/wb2-mc-v1b0-s42 --split val

Deterministic T=1 and MC T=10/20/50, where T=k is the mean softmax of the
**first k passes of the same bank** (never an independent sample). If a
val-fitted `calibration.json` exists, the calibrated rows apply its T
unchanged. This is a sanity table for the handoff; Sabari owns the final
`results/wb2/tables/` comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.eval.metrics import brier_score, expected_calibration_error, macro_metrics
from src.wb2.sampling.calibrate import mc_probs, nll

MEL = 4
BUDGETS = (10, 20, 50)


def row(method: str, y_true: np.ndarray, probs: np.ndarray, head_passes: int, calibrated: bool) -> dict:
    ece, _ = expected_calibration_error(y_true, probs, n_bins=10)
    pred = probs.argmax(axis=1)
    mel = y_true == MEL
    return {
        "method": method,
        **macro_metrics(y_true, probs),
        "ece_10bin": ece,
        "brier": brier_score(y_true, probs),
        "nll": nll(y_true, probs),
        "mel_recall": float((pred[mel] == MEL).mean()) if mel.any() else float("nan"),
        "mel_missed": int((pred[mel] != MEL).sum()),
        "mel_total": int(mel.sum()),
        "head_passes": head_passes,
        "backbone_passes": 1,
        "calibrated": calibrated,
    }


def fixed_budget_table(bank_dir: Path, split: str) -> pd.DataFrame:
    manifest = json.loads((bank_dir / f"{split}_manifest.json").read_text())
    mc = np.load(bank_dir / f"{split}_mc_logits.npz")["logits"]
    det = np.load(bank_dir / f"{split}_det_logits.npz")["logits"]
    y_true = pd.read_csv(bank_dir / f"{split}_ids.csv")["y_true"].to_numpy(dtype=int)

    rows = [row("deterministic_T1", y_true, softmax(det.astype(np.float64), axis=1), 0, False)]
    rows += [row(f"mc_T{t}", y_true, mc_probs(mc[:, :t]), t, False) for t in BUDGETS if t <= mc.shape[1]]

    calib_path = bank_dir / "calibration.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
        assert calib["fitted_on"] == "val" and calib["applied_to"] == "per_pass_logits_before_softmax"
        rows += [row(f"mc_T{t}_temp", y_true, mc_probs(mc[:, :t], calib["temperature"]), t, True)
                 for t in BUDGETS if t <= mc.shape[1]]

    table = pd.DataFrame(rows)
    table.insert(0, "split", split)
    table.insert(1, "status", "MEASURED" if manifest["status"] == "COMPLETE" and not manifest.get("pilot_subset")
                 else ("PILOT ONLY" if manifest.get("pilot_subset") else manifest["status"]))
    table["checkpoint_sha256"] = manifest["checkpoint_sha256"][:12]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-budget MC cross-check for a WB2 bank.")
    parser.add_argument("--bank-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    args = parser.parse_args()

    table = fixed_budget_table(args.bank_dir, args.split)
    out = args.bank_dir / f"fixed_budget_{args.split}.csv"
    table.to_csv(out, index=False)
    print(table[["method", "macro_auroc", "macro_f1", "balanced_accuracy", "ece_10bin", "brier",
                 "mel_recall", "mel_missed", "mel_total"]].to_string(index=False, float_format="%.4f"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

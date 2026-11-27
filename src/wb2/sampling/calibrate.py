"""Validation-only temperature scaling for the MC predictive distribution (WB2-I4, implementation.md A3).

    python -m src.wb2.sampling.calibrate --bank-dir results/wb2/inputs/wb2-mc-v1b0-s42

Frozen rule: one scalar T > 0 divides every per-pass logit before softmax;
the predictive distribution is the mean of those softmaxes over all passes
in the bank (50). T minimises mean NLL of that distribution on **val only**.
The same transform must be applied unchanged to test. T is fitted for the
50-pass mean and is not claimed optimal for T=10/20 or for the
deterministic logits.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from src.eval.metrics import brier_score, expected_calibration_error

APPLIED_TO = "per_pass_logits_before_softmax"
LOG_T_BOUNDS = (np.log(0.05), np.log(20.0))


def mc_probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """(N, T, 7) logits -> (N, 7) mean softmax over passes, after dividing logits by `temperature`."""
    return softmax(logits.astype(np.float64) / temperature, axis=-1).mean(axis=1)


def nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, None)).mean())


def calibration_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    ece, _ = expected_calibration_error(y_true, probs, n_bins=10)
    return {"nll": nll(y_true, probs), "ece_10bin": ece, "brier": brier_score(y_true, probs)}


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Minimise val NLL over log T (bounded, so T stays positive and finite)."""
    result = minimize_scalar(lambda log_t: nll(y_true, mc_probs(logits, float(np.exp(log_t)))),
                             bounds=LOG_T_BOUNDS, method="bounded", options={"xatol": 1e-5})
    return float(np.exp(result.x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit WB2 temperature on the validation bank only.")
    parser.add_argument("--bank-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="default: <bank-dir>/calibration.json")
    args = parser.parse_args()

    manifest = json.loads((args.bank_dir / "val_manifest.json").read_text())
    if manifest["split"] != "val":
        raise SystemExit("Temperature may only be fitted on the validation bank.")
    if manifest["status"] != "COMPLETE":
        raise SystemExit(f"Validation bank status is {manifest['status']}; refusing to fit on an incomplete bank.")

    logits = np.load(args.bank_dir / "val_mc_logits.npz")["logits"]
    y_true = pd.read_csv(args.bank_dir / "val_ids.csv")["y_true"].to_numpy(dtype=int)

    temperature = fit_temperature(logits, y_true)
    record = {
        "calibration_id": f"{manifest['run_id']}-temp-val",
        "temperature": temperature,
        "fitted_on": "val",
        "applied_to": APPLIED_TO,
        "aggregation": f"mean of softmax(logits / T) over all {logits.shape[1]} passes",
        "objective": "mean negative log-likelihood on val (scipy bounded scalar search over log T)",
        "search_bounds_T": [float(np.exp(b)) for b in LOG_T_BOUNDS],
        "n_passes_fit": int(logits.shape[1]),
        "n_val_images": int(len(y_true)),
        "val_before": calibration_metrics(y_true, mc_probs(logits, 1.0)),
        "val_after": calibration_metrics(y_true, mc_probs(logits, temperature)),
        "bank_run_id": manifest["run_id"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "note": "Fitted for the 50-pass mean only; not claimed optimal for other T. Test data never used.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out = args.out or args.bank_dir / "calibration.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"T = {temperature:.4f}  val NLL {record['val_before']['nll']:.4f} -> {record['val_after']['nll']:.4f}, "
          f"ECE {record['val_before']['ece_10bin']:.4f} -> {record['val_after']['ece_10bin']:.4f}  -> {out}")


if __name__ == "__main__":
    main()

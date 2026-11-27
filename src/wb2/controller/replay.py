"""Offline prefix replay of every compared method over one logit bank
(implementation.md §3 C4, WB2-O1).

All MC methods read the *same* ordered draws: fixed T=10/20 are prefixes
of the T=50 bank and the adaptive policies stop at a prefix of it, so method
differences come from the policy alone (§1 "same stochastic draws").

Pass savings measured here are algorithmic/simulated — wall-clock claims
belong to the separately timed online path (§5 E3 interpretation warning).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.wb2.controller.bank import LogitBank
from src.wb2.controller.features import prefix_stats
from src.wb2.controller.policies import FixedT, RiskAware, VarianceConvergence

UNCALIBRATED_ID = "uncalibrated"


def load_temperature(path: Path | None) -> tuple[float, str]:
    """Read the frozen WB2-I4 calibration record -> (tau, calibration_id).

    Expected keys: calibration_id, temperature (>0), fitted_on ("val"),
    applied_to ("per_pass_logits_before_softmax"). None -> uncalibrated.
    """
    if path is None:
        return 1.0, UNCALIBRATED_ID
    rec = json.loads(Path(path).read_text())
    if rec.get("fitted_on") != "val":
        raise ValueError(f"{path}: temperature must be fitted on val only, got fitted_on={rec.get('fitted_on')!r}")
    if rec.get("applied_to") != "per_pass_logits_before_softmax":
        raise ValueError(
            f"{path}: applied_to={rec.get('applied_to')!r}; the controller applies temperature to "
            "per-pass logits before softmax — agree the order with the sampling owner first"
        )
    tau = float(rec["temperature"])
    if not tau > 0:
        raise ValueError(f"{path}: temperature must be > 0, got {tau}")
    return tau, str(rec["calibration_id"])


def policies_from_config(cfg: dict) -> list:
    """The fixed set of compared MC methods, thresholds taken from a tuned config."""
    return [
        FixedT(10),
        FixedT(20),
        FixedT(50),
        FixedT(50, referral_entropy=cfg["fixed_T50_referral"]["referral_entropy"]),
        VarianceConvergence(**cfg["variance_adaptive"]),
        RiskAware(**cfg["risk_aware_adaptive"]),
    ]


def replay_bank(bank: LogitBank, policies: list, tau: float = 1.0) -> pd.DataFrame:
    """Run deterministic T=1 (if the bank has WB2-I3 logits) and every policy.

    Returns the decisions frame *without* labels; join them for evaluation
    only via `attach_labels`.
    """
    frames = []
    if bank.det_logits is not None:
        det_stats = prefix_stats(bank.det_logits[:, None, :], tau, checkpoints=(1,))
        frames.append(_with_ids(FixedT(1, name="det_T1").decide(det_stats), bank))
    stats = prefix_stats(bank.logits, tau)
    for policy in policies:
        frames.append(_with_ids(policy.decide(stats), bank))
    return pd.concat(frames, ignore_index=True)


def _with_ids(decisions: pd.DataFrame, bank: LogitBank) -> pd.DataFrame:
    decisions.insert(0, "image_id", bank.ids["image_id"].to_numpy())
    return decisions


def attach_labels(decisions: pd.DataFrame, bank: LogitBank) -> pd.DataFrame:
    """Evaluation-only join of lesion_id / y_true (never used to make a decision)."""
    out = decisions.merge(bank.ids[["image_id", "lesion_id", "y_true"]], on="image_id", how="left", validate="many_to_one")
    if out["y_true"].isna().any():
        raise ValueError("some decisions have no label row — ID misalignment")
    out["y_true"] = out["y_true"].astype(int)
    return out


def head_and_backbone_passes(decisions: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Separate stochastic head draws from backbone forwards (§6 Efficiency).

    With feature reuse the backbone runs once per image regardless of T;
    without it every draw is a full forward. Unknown -> NaN, not a guess.
    """
    out = decisions.copy()
    out["head_passes"] = out["passes"]
    reuse = manifest.get("feature_reuse")
    det = out["method"] == "det_T1"
    if reuse is True:
        out["backbone_passes"] = 1
    elif reuse is False:
        out["backbone_passes"] = out["passes"]
    else:
        out["backbone_passes"] = np.where(det, 1, np.nan)
    return out

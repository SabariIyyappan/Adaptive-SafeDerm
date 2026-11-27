"""Stopping / review policies replayed over a saved logit bank
(implementation.md §3 C2–C3, §5 E1/E3).

Every policy maps a `PrefixStats` (label-free, causal) to one decision per
image. A policy picks its stop checkpoint as the *first* checkpoint whose
stop condition holds, using only statistics at or before that checkpoint,
so replaying the saved 50-draw bank is equivalent to running the policy
online on the same draw order (the "offline prefix replay" of §3 C4).

Two output flags, intentionally separate:
  - `review`: the binary research output — False = TRUSTED RESULT (the
    model's computational checks passed), True = NEEDS REVIEW. TRUSTED never
    means clinically safe/benign (§6).
  - `clinical_attention`: the final prediction is in the suspicious group
    (mel/bcc). A stable, confident suspicious prediction can be TRUSTED by the
    computational checks, but it always carries this flag — it is never
    presented as an autonomous benign result (§3 C2).

Policies:
  FixedT            — mean of the first T stochastic draws (T=1 on the
                      deterministic bank = V1-style single pass); optional
                      entropy referral threshold for matched-coverage review.
  VarianceConvergence — generic MC-CP-*inspired* comparator: stop when the
                      mean-probability vector and mean per-class variance stop
                      changing between successive checkpoints. Not a faithful
                      MC-CP reproduction (no conformal step).
  RiskAware         — the proposed controller: four model-risk tiers from
                      suspicious score and entropy, stop only when class,
                      uncertainty and the suspicious/benign margin are all
                      settled; unresolved cases end at the cap as NEEDS REVIEW.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.data.eda import CLASS_ORDER
from src.wb2.controller.features import NUM_CLASSES, SUSPICIOUS_IDX, PrefixStats

TIERS = ("low", "medium", "high", "critical")
LOW, MEDIUM, HIGH, CRITICAL = range(4)

# Review reasons (implementation.md §3 C3). Joined with "|" in the decisions frame.
NONFINITE = "NONFINITE"
UNSTABLE_CLASS = "UNSTABLE_CLASS"
UNSTABLE_UNCERTAINTY = "UNSTABLE_UNCERTAINTY"
HIGH_ENTROPY = "HIGH_ENTROPY"
SUSPICIOUS_MARGIN = "SUSPICIOUS_MARGIN"
CRITICAL_TIER = "CRITICAL_TIER"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"

# Stop reasons.
STOP_FIXED = "fixed_budget"
STOP_CONVERGED = "converged"
STOP_BUDGET = "budget_exhausted"
STOP_NONFINITE = "nonfinite"

# The occlusion / Grad-CAM++ verifier is planned V4 work (§5 E5); every
# decision states that it did not run rather than implying it passed.
EXPLANATION_CHECK = "NOT_RUN"

PROB_COLUMNS = [f"prob_{c}" for c in range(NUM_CLASSES)]


def _first_true(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: (index of first True, any True). Rows with no True get the last index."""
    any_true = mask.any(axis=1)
    first = np.where(any_true, mask.argmax(axis=1), mask.shape[1] - 1)
    return first, any_true


def _frame(method: str, stats: PrefixStats, j: np.ndarray, stop_reason: np.ndarray,
           review: np.ndarray, reasons: list[str], tier: np.ndarray | None) -> pd.DataFrame:
    rows = np.arange(stats.n_images)
    probs = stats.mean_probs[rows, j]
    pred = stats.pred[rows, j]
    out = pd.DataFrame(
        {
            "method": method,
            "passes": np.asarray(stats.checkpoints)[j],
            "stop_reason": stop_reason,
            "review": review.astype(bool),
            "decision": np.where(review, "NEEDS REVIEW", "TRUSTED RESULT"),
            "review_reasons": reasons,
            "tier": [TIERS[t] for t in tier] if tier is not None else "",
            "pred": pred,
            "pred_dx": [CLASS_ORDER[p] for p in pred],
            "clinical_attention": np.isin(pred, SUSPICIOUS_IDX),
            "suspicious_score": stats.suspicious[rows, j],
            "attention_score": stats.attention[rows, j],
            "entropy": stats.entropy[rows, j],
            "mutual_info": stats.mutual_info[rows, j],
            "explanation_check": EXPLANATION_CHECK,
        }
    )
    for c in range(NUM_CLASSES):
        out[PROB_COLUMNS[c]] = probs[:, c]
    return out


def _join(flag_lists: list[tuple[str, np.ndarray]], n: int) -> list[str]:
    reasons = [[] for _ in range(n)]
    for name, mask in flag_lists:
        for i in np.flatnonzero(mask):
            reasons[i].append(name)
    return ["|".join(r) for r in reasons]


@dataclass(frozen=True)
class FixedT:
    """Mean of the first T draws. `referral_entropy` (normalized, 0..1) is the
    optional validation-tuned review gate; None = no referral."""

    T: int
    referral_entropy: float | None = None
    name: str | None = None

    @property
    def method(self) -> str:
        base = self.name or f"fixed_T{self.T}"
        return base if self.referral_entropy is None else f"{base}+referral"

    def decide(self, stats: PrefixStats) -> pd.DataFrame:
        if self.T not in stats.checkpoints:
            raise ValueError(f"T={self.T} is not one of the checkpoints {stats.checkpoints}")
        n = stats.n_images
        j = np.full(n, stats.checkpoints.index(self.T))
        rows = np.arange(n)
        nonfinite = stats.nonfinite[rows, j]
        high_h = (
            stats.entropy[rows, j] >= self.referral_entropy
            if self.referral_entropy is not None
            else np.zeros(n, dtype=bool)
        )
        review = nonfinite | high_h
        stop = np.where(nonfinite, STOP_NONFINITE, STOP_FIXED)
        reasons = _join([(NONFINITE, nonfinite), (HIGH_ENTROPY, high_h)], n)
        return _frame(self.method, stats, j, stop, review, reasons, None)


@dataclass(frozen=True)
class VarianceConvergence:
    """Generic convergence comparator. Stops at the first checkpoint >= min_passes
    where delta_p < eps_p and delta_var < eps_var. Review only via the optional
    entropy referral gate (or non-finite input); hitting the cap does not by
    itself force review — the % at cap is reported instead."""

    eps_p: float
    eps_var: float
    referral_entropy: float | None = None
    min_passes: int = 10
    method: str = "variance_adaptive"

    def decide(self, stats: PrefixStats) -> pd.DataFrame:
        n = stats.n_images
        k = np.asarray(stats.checkpoints)[None, :]
        with np.errstate(invalid="ignore"):
            converged = (k >= self.min_passes) & (stats.delta_p < self.eps_p) & (stats.delta_var < self.eps_var)
        stop_mask = converged | stats.nonfinite
        j, stopped = _first_true(stop_mask)
        rows = np.arange(n)
        nonfinite = stats.nonfinite[rows, j]
        stop = np.where(nonfinite, STOP_NONFINITE, np.where(stopped, STOP_CONVERGED, STOP_BUDGET))
        high_h = (
            stats.entropy[rows, j] >= self.referral_entropy
            if self.referral_entropy is not None
            else np.zeros(n, dtype=bool)
        )
        review = nonfinite | high_h
        reasons = _join([(NONFINITE, nonfinite), (HIGH_ENTROPY, high_h)], n)
        return _frame(self.method, stats, j, stop, review, reasons, None)

    def params(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "method"}


@dataclass(frozen=True)
class RiskAware:
    """The proposed controller (implementation.md §3 C2). Thresholds are
    validation-selected; entropies are normalized by ln 7.

    Tier at each checkpoint, in priority order (a *model risk/attention* tier,
    not a pathology severity rating):
      critical — benign-class prediction while suspicious mass s >= tau_s
                 (false-reassurance risk), or a suspicious prediction that is
                 unstable (class flip / tie / entropy still moving) or has
                 entropy >= h_hi;
      high     — a suspicious prediction, or entropy >= h_hi;
      medium   — entropy >= h_lo, or |s - tau_s| < margin;
      low      — otherwise.

    Stop (TRUSTED) at the first checkpoint where k >= min_passes (min_passes_high
    for high tier), the class is stable since the previous checkpoint and not
    tied, |delta entropy| < delta_h, entropy < h_hi, |s - tau_s| >= margin and
    the tier is not critical. The first checkpoint never counts as stable
    because there is no previous checkpoint to compare against. Otherwise draw
    5 more; at the cap the case is NEEDS REVIEW with the failing checks as
    reasons plus BUDGET_EXHAUSTED. Non-finite logits stop immediately as
    NEEDS REVIEW.
    """

    tau_s: float
    h_lo: float
    h_hi: float
    margin: float
    delta_h: float
    min_passes: int = 10
    min_passes_high: int = 20
    method: str = "risk_aware_adaptive"

    def tiers(self, stats: PrefixStats) -> np.ndarray:
        pred_susp = np.isin(stats.pred, SUSPICIOUS_IDX)
        with np.errstate(invalid="ignore"):
            moving_h = stats.delta_h >= self.delta_h  # nan (first checkpoint) -> False
        unstable = stats.flip | stats.tie | moving_h
        high_h = stats.entropy >= self.h_hi
        near = np.abs(stats.suspicious - self.tau_s) < self.margin

        critical = (~pred_susp & (stats.suspicious >= self.tau_s)) | (pred_susp & (unstable | high_h))
        high = ~critical & (pred_susp | high_h)
        medium = ~critical & ~high & ((stats.entropy >= self.h_lo) | near)
        tier = np.full(stats.pred.shape, LOW)
        tier[medium] = MEDIUM
        tier[high] = HIGH
        tier[critical] = CRITICAL
        return tier

    def _checks(self, stats: PrefixStats) -> dict[str, np.ndarray]:
        with np.errstate(invalid="ignore"):
            settled_h = stats.delta_h < self.delta_h  # nan -> False: no cross-checkpoint claim at j=0
        return {
            UNSTABLE_CLASS: stats.flip | stats.tie,
            UNSTABLE_UNCERTAINTY: ~settled_h,
            HIGH_ENTROPY: stats.entropy >= self.h_hi,
            SUSPICIOUS_MARGIN: np.abs(stats.suspicious - self.tau_s) < self.margin,
        }

    def decide(self, stats: PrefixStats) -> pd.DataFrame:
        n = stats.n_images
        tier = self.tiers(stats)
        checks = self._checks(stats)
        k = np.asarray(stats.checkpoints)[None, :]
        min_k = np.where(tier == HIGH, self.min_passes_high, self.min_passes)
        failing = np.zeros_like(tier, dtype=bool)
        for mask in checks.values():
            failing |= mask
        ok = (k >= min_k) & ~failing & (tier != CRITICAL) & ~stats.nonfinite

        j, stopped = _first_true(ok | stats.nonfinite)
        rows = np.arange(n)
        nonfinite = stats.nonfinite[rows, j]
        exhausted = ~stopped
        final_tier = tier[rows, j]
        stop = np.where(nonfinite, STOP_NONFINITE, np.where(exhausted, STOP_BUDGET, STOP_CONVERGED))
        review = nonfinite | exhausted

        flags = [(NONFINITE, nonfinite)]
        for name, mask in checks.items():
            flags.append((name, exhausted & mask[rows, j]))
        flags.append((CRITICAL_TIER, exhausted & (final_tier == CRITICAL)))
        flags.append((BUDGET_EXHAUSTED, exhausted))
        reasons = _join(flags, n)
        return _frame(self.method, stats, j, stop, review, reasons, final_tier)

    def params(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "method"}

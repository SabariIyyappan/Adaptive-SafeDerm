"""Load and validate the WB2 stochastic logit banks (implementation.md §4,
WB2-I1/I2/I3). Proposed file layout, one directory per run, per split:

    {split}_mc_logits.npz   key `logits`: float32 N x 50 x 7 raw logits,
                            pass axis in 0-based draw order
    {split}_det_logits.npz  key `logits`: float32 N x 7 dropout-OFF logits (WB2-I3)
    {split}_ids.csv         image_id, lesion_id, y_true — row-aligned with both arrays
    {split}_manifest.json   checkpoint_sha256, seed, transform, split_csv_sha256,
                            class_order, status (COMPLETE | PARTIAL | FAILED),
                            n_images, n_passes, stochastic_modules, feature_reuse,
                            started_at, finished_at, synthetic (bool)

Validation fails loudly rather than producing plausible-looking numbers
from a misaligned bank, mirroring `src.eval.metrics.load_predictions`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda import CLASS_ORDER
from src.wb2.controller.features import MAX_PASSES, NUM_CLASSES

ID_COLUMNS = ["image_id", "lesion_id", "y_true"]
STATUSES = ("COMPLETE", "PARTIAL", "FAILED")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class LogitBank:
    split: str
    logits: np.ndarray                 # N x P x C
    ids: pd.DataFrame                  # image_id, lesion_id, y_true
    manifest: dict
    det_logits: np.ndarray | None = None
    files_sha256: dict[str, str] = field(default_factory=dict)
    nonfinite_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def n_images(self) -> int:
        return len(self.ids)

    @property
    def is_synthetic(self) -> bool:
        return bool(self.manifest.get("synthetic", False))

    @property
    def y_true(self) -> np.ndarray:
        return self.ids["y_true"].to_numpy(dtype=int)


def validate_arrays(logits: np.ndarray, ids: pd.DataFrame, det_logits: np.ndarray | None = None,
                    n_passes: int = MAX_PASSES) -> np.ndarray:
    """Shape/alignment checks. Returns row indices containing non-finite logits
    (allowed but reported; the policies send them to NEEDS REVIEW)."""
    missing = [c for c in ID_COLUMNS if c not in ids.columns]
    if missing:
        raise ValueError(f"ids table is missing columns {missing}; expected {ID_COLUMNS}")
    if logits.ndim != 3 or logits.shape[1:] != (n_passes, NUM_CLASSES):
        raise ValueError(f"MC logits must be N x {n_passes} x {NUM_CLASSES}, got {logits.shape}")
    if logits.shape[0] != len(ids):
        raise ValueError(f"{logits.shape[0]} logit rows but {len(ids)} id rows — not row-aligned")
    if ids["image_id"].duplicated().any():
        dups = ids.loc[ids["image_id"].duplicated(), "image_id"].tolist()[:5]
        raise ValueError(f"duplicate image_id rows in bank, e.g. {dups}")
    labels = ids["y_true"].to_numpy()
    if ((labels < 0) | (labels >= NUM_CLASSES)).any():
        raise ValueError(f"y_true outside 0..{NUM_CLASSES - 1}")
    if det_logits is not None and det_logits.shape != (len(ids), NUM_CLASSES):
        raise ValueError(f"deterministic logits must be N x {NUM_CLASSES}, got {det_logits.shape}")
    return np.flatnonzero(~np.isfinite(logits).all(axis=(1, 2)))


def check_against_split(ids: pd.DataFrame, split_csv: Path) -> None:
    """The bank must cover exactly the images of the frozen split, with the same labels."""
    split = pd.read_csv(split_csv)
    bank_ids, split_ids = set(ids["image_id"]), set(split["image_id"])
    if bank_ids != split_ids:
        raise ValueError(
            f"bank image_ids differ from {split_csv}: {len(bank_ids - split_ids)} extra, "
            f"{len(split_ids - bank_ids)} missing (a pilot subset must be loaded with allow_subset=True)"
        )
    _check_labels(ids, split, split_csv)


def _check_labels(ids: pd.DataFrame, split: pd.DataFrame, split_csv: Path) -> None:
    merged = ids.merge(split[["image_id", "lesion_id", "label"]], on="image_id", suffixes=("", "_split"))
    if len(merged) != len(ids):
        raise ValueError(f"{len(ids) - len(merged)} bank image_ids are not in {split_csv}")
    bad_label = merged["y_true"] != merged["label"]
    if bad_label.any():
        raise ValueError(f"{int(bad_label.sum())} y_true values disagree with {split_csv} (C1 label map?)")
    bad_lesion = merged["lesion_id"] != merged["lesion_id_split"]
    if bad_lesion.any():
        raise ValueError(f"{int(bad_lesion.sum())} lesion_id values disagree with {split_csv}")


def load_bank(bank_dir: Path, split: str, split_csv: Path | None = None,
              allow_partial: bool = False, allow_subset: bool = False) -> LogitBank:
    bank_dir = Path(bank_dir)
    mc_path = bank_dir / f"{split}_mc_logits.npz"
    det_path = bank_dir / f"{split}_det_logits.npz"
    ids_path = bank_dir / f"{split}_ids.csv"
    manifest_path = bank_dir / f"{split}_manifest.json"
    for p in (mc_path, ids_path, manifest_path):
        if not p.exists():
            raise FileNotFoundError(f"bank file missing: {p}")

    manifest = json.loads(manifest_path.read_text())
    status = manifest.get("status")
    if status not in STATUSES:
        raise ValueError(f"{manifest_path}: status must be one of {STATUSES}, got {status!r}")
    if status == "FAILED" or (status == "PARTIAL" and not allow_partial):
        raise ValueError(f"{manifest_path}: bank status is {status}; pass allow_partial=True only for a disclosed pilot")
    if manifest.get("class_order", CLASS_ORDER) != CLASS_ORDER:
        raise ValueError(f"{manifest_path}: class_order {manifest.get('class_order')} != C1 order {CLASS_ORDER}")

    with np.load(mc_path) as z:
        logits = z["logits"]
    det = None
    if det_path.exists():
        with np.load(det_path) as z:
            det = z["logits"]
    ids = pd.read_csv(ids_path, dtype={"image_id": str, "lesion_id": str})

    nonfinite = validate_arrays(logits, ids, det)
    if split_csv is not None:
        if allow_subset:
            _check_labels(ids, pd.read_csv(split_csv), split_csv)
        else:
            check_against_split(ids, split_csv)

    hashes = {p.name: sha256_file(p) for p in (mc_path, det_path, ids_path, manifest_path) if p.exists()}
    return LogitBank(split, logits, ids, manifest, det, hashes, nonfinite)


def save_bank(bank_dir: Path, split: str, logits: np.ndarray, ids: pd.DataFrame, manifest: dict,
              det_logits: np.ndarray | None = None) -> None:
    """Writer for the same layout (used by tests and by anyone producing a bank)."""
    bank_dir = Path(bank_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)
    validate_arrays(logits, ids, det_logits, n_passes=logits.shape[1])
    np.savez(bank_dir / f"{split}_mc_logits.npz", logits=logits.astype(np.float32))
    if det_logits is not None:
        np.savez(bank_dir / f"{split}_det_logits.npz", logits=det_logits.astype(np.float32))
    ids[ID_COLUMNS].to_csv(bank_dir / f"{split}_ids.csv", index=False)
    (bank_dir / f"{split}_manifest.json").write_text(json.dumps(manifest, indent=2))

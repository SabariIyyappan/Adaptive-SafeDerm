"""C2 — lesion-grouped, dx-stratified 70/15/15 split (PLAN.md §5.3.4).

The paper describes this as a one-line "70/15/15 split stratified by dx,
random_state=42, applied to unique lesions." We interpret and implement it
as the two-step procedure PLAN.md §5.3.4 specifies (and §13 item 1 logs
this as our documented interpretation, since the paper's own lesion counts
don't add up — see `docs/deviations.md`):

  1. reduce to a one-row-per-lesion table (one dx per lesion);
  2. split lesions 70% / 30%, stratified by dx, seed 42;
  3. split the 30% into two halves (15% / 15%), stratified by dx, seed 42;
  4. map lesions back to all of their images.

Every lesion's images go to exactly one split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.labels import EXPECTED_HAM10000_LABEL_MAP, build_label_map

SEED = 42


def lesion_level_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per lesion_id. Asserts each lesion has exactly one dx —
    integrity_audit.py already checks this, but split.py must not silently
    proceed on a dataset where that check would have failed.
    """
    dx_per_lesion = metadata.groupby("lesion_id")["dx"].nunique()
    bad = dx_per_lesion[dx_per_lesion > 1]
    if len(bad) > 0:
        raise ValueError(f"{len(bad)} lesion_ids have more than one distinct dx: {list(bad.index[:5])}")
    return metadata.drop_duplicates(subset="lesion_id")[["lesion_id", "dx"]].reset_index(drop=True)


def build_split(
    metadata: pd.DataFrame,
    seed: int = SEED,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """Returns `metadata` with a `split` column ('train'/'val'/'test') and
    a `label` column (via the C1 label map), split at the lesion level.
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("train/val/test fractions must sum to 1.0")

    lesions = lesion_level_table(metadata)

    train_lesions, rest_lesions = train_test_split(
        lesions, test_size=(val_frac + test_frac), stratify=lesions["dx"], random_state=seed,
    )
    # Split the remainder in half (proportionally, in case val_frac != test_frac).
    rest_test_share = test_frac / (val_frac + test_frac)
    val_lesions, test_lesions = train_test_split(
        rest_lesions, test_size=rest_test_share, stratify=rest_lesions["dx"], random_state=seed,
    )

    split_by_lesion = {}
    for lesion_id in train_lesions["lesion_id"]:
        split_by_lesion[lesion_id] = "train"
    for lesion_id in val_lesions["lesion_id"]:
        split_by_lesion[lesion_id] = "val"
    for lesion_id in test_lesions["lesion_id"]:
        split_by_lesion[lesion_id] = "test"

    out = metadata.copy()
    out["split"] = out["lesion_id"].map(split_by_lesion)
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), "lesion_id"].unique()
        raise ValueError(f"{len(missing)} lesion_ids in metadata were not assigned to a split: {missing[:5]}")

    label_map = build_label_map(out["dx"])
    if label_map != EXPECTED_HAM10000_LABEL_MAP:
        raise ValueError(
            f"Built label map {label_map} does not match the expected HAM10000 map "
            f"{EXPECTED_HAM10000_LABEL_MAP}. Refusing to write a split with an unexpected label map."
        )
    out["label"] = out["dx"].map(label_map)

    return out[["image_id", "lesion_id", "dx", "label", "split"]]


def write_split_files(split_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        subset = split_df[split_df["split"] == name].drop(columns="split")
        subset.to_csv(out_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen C2 lesion-grouped split (PLAN.md §5.3.4).")
    parser.add_argument("--data-root", required=True, type=Path, help="Dir containing HAM10000_metadata.csv")
    parser.add_argument("--out", required=True, type=Path, help="Output dir, e.g. data/splits")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    metadata = pd.read_csv(args.data_root / "HAM10000_metadata.csv")
    split_df = build_split(metadata, seed=args.seed)
    write_split_files(split_df, args.out)
    counts = split_df["split"].value_counts()
    print(f"Wrote split files to {args.out}:\n{counts}")


if __name__ == "__main__":
    main()

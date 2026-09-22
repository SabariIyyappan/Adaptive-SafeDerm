"""Split verification (PLAN.md §5.3.5).

Confirms:
  - zero lesion overlap between splits;
  - zero image overlap;
  - the union of splits is all images in the metadata;
  - all 7 classes appear in every split.

Produces a per-split class-count/proportion table and an "Ours vs Paper"
image-count table (paper: 7024 / 1497 / 1494 — PLAN.md §5.3.5, §13 item 1).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PAPER_SPLIT_IMAGE_COUNTS = {"train": 7024, "val": 1497, "test": 1494}
PAPER_LESION_COUNTS_NOTE = (
    "Paper reports 1,891 / 405 / 405 unique lesions (2,701 total) for train/val/test, "
    "which is far short of HAM10000's ~7,470 unique lesion_ids. See PLAN.md §13 item 1 — "
    "we follow the stated split *method* and report our own counts here rather than "
    "trying to match the paper's inconsistent lesion totals."
)


@dataclass
class VerificationResult:
    passed: bool
    checks: dict = field(default_factory=dict)
    class_table: pd.DataFrame | None = None
    ours_vs_paper: pd.DataFrame | None = None
    anomalies: list = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# V1 Split Verification (PLAN.md §5.3.5)", ""]
        lines.append(f"**Overall: {'PASS' if self.passed else 'FAIL — see anomalies below'}**")
        lines.append("")
        lines.append("| Check | Result |")
        lines.append("|---|---|")
        for name, val in self.checks.items():
            lines.append(f"| {name} | {val} |")
        if self.class_table is not None:
            lines.append("")
            lines.append("## Per-split class counts and proportions")
            lines.append("")
            lines.append(self.class_table.to_markdown())
        if self.ours_vs_paper is not None:
            lines.append("")
            lines.append("## Ours vs Paper — image counts per split")
            lines.append("")
            lines.append(self.ours_vs_paper.to_markdown(index=False))
            lines.append("")
            lines.append(f"> {PAPER_LESION_COUNTS_NOTE}")
        if self.anomalies:
            lines.append("")
            lines.append("## Anomalies")
            for a in self.anomalies:
                lines.append(f"- {a}")
        return "\n".join(lines) + "\n"


def verify_split(splits: dict[str, pd.DataFrame], full_metadata: pd.DataFrame) -> VerificationResult:
    """`splits` maps split name -> its DataFrame (as written by split.py)."""
    anomalies: list[str] = []
    checks: dict = {}

    # Lesion overlap.
    lesion_sets = {name: set(df["lesion_id"]) for name, df in splits.items()}
    lesion_overlaps = {}
    names = list(lesion_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = lesion_sets[names[i]] & lesion_sets[names[j]]
            if overlap:
                lesion_overlaps[f"{names[i]}∩{names[j]}"] = len(overlap)
    checks["lesion overlap between splits"] = lesion_overlaps or "0 (none)"
    if lesion_overlaps:
        anomalies.append(f"lesion_id overlap found between splits: {lesion_overlaps}")

    # Image overlap.
    image_sets = {name: set(df["image_id"]) for name, df in splits.items()}
    image_overlaps = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = image_sets[names[i]] & image_sets[names[j]]
            if overlap:
                image_overlaps[f"{names[i]}∩{names[j]}"] = len(overlap)
    checks["image_id overlap between splits"] = image_overlaps or "0 (none)"
    if image_overlaps:
        anomalies.append(f"image_id overlap found between splits: {image_overlaps}")

    # Union == all images.
    union = set().union(*image_sets.values())
    all_images = set(full_metadata["image_id"])
    missing_from_union = all_images - union
    extra_in_union = union - all_images
    checks["union covers all metadata images"] = (
        "yes" if not missing_from_union and not extra_in_union
        else f"missing={len(missing_from_union)}, extra={len(extra_in_union)}"
    )
    if missing_from_union:
        anomalies.append(f"{len(missing_from_union)} images in metadata are in no split")
    if extra_in_union:
        anomalies.append(f"{len(extra_in_union)} images in splits are not in metadata")

    # All 7 classes in every split.
    all_classes = set(full_metadata["dx"].unique())
    for name, df in splits.items():
        present = set(df["dx"].unique())
        missing_classes = all_classes - present
        checks[f"classes missing from '{name}'"] = missing_classes or "none"
        if missing_classes:
            anomalies.append(f"split '{name}' is missing classes: {missing_classes}")

    # Per-split class count/proportion table.
    rows = []
    for name, df in splits.items():
        counts = df["dx"].value_counts()
        total = len(df)
        for cls in sorted(all_classes):
            n = int(counts.get(cls, 0))
            rows.append({"split": name, "dx": cls, "count": n, "pct_of_split": round(100 * n / total, 2) if total else 0.0})
    class_table = pd.DataFrame(rows).pivot(index="dx", columns="split", values=["count", "pct_of_split"])

    # Ours vs paper.
    ours_counts = {name: len(df) for name, df in splits.items()}
    ours_vs_paper = pd.DataFrame(
        [
            {"split": name, "ours": ours_counts.get(name, 0), "paper": PAPER_SPLIT_IMAGE_COUNTS.get(name, "—")}
            for name in ("train", "val", "test")
        ]
    )

    return VerificationResult(
        passed=(len(anomalies) == 0),
        checks=checks,
        class_table=class_table,
        ours_vs_paper=ours_vs_paper,
        anomalies=anomalies,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the C2 split (PLAN.md §5.3.5).")
    parser.add_argument("--splits", required=True, type=Path, help="Dir with train.csv/val.csv/test.csv")
    parser.add_argument("--data-root", required=True, type=Path, help="Dir containing HAM10000_metadata.csv")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    splits = {name: pd.read_csv(args.splits / f"{name}.csv") for name in ("train", "val", "test")}
    full_metadata = pd.read_csv(args.data_root / "HAM10000_metadata.csv")

    result = verify_split(splits, full_metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result.to_markdown())
    print(result.to_markdown())
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

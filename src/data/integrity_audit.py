"""Integrity audit (PLAN.md §5.3.2).

Confirms, against `data_root/HAM10000_metadata.csv` and `data_root/images/`:
  - exactly N unique image_ids, each with an image file present;
  - M unique lesion_ids, each with exactly one dx;
  - class counts match `expected_class_counts`;
  - every image is `expected_size`.

Defaults are the paper's real HAM10000 numbers. Tests override them via
`expected_*` kwargs so the exact same logic can be checked against a small
synthetic fixture.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image

EXPECTED_IMAGE_COUNT = 10_015
EXPECTED_LESION_COUNT = 7_470
EXPECTED_CLASS_COUNTS = {
    "nv": 6705,
    "mel": 1113,
    "bkl": 1099,
    "bcc": 514,
    "akiec": 327,
    "vasc": 142,
    "df": 115,
}
EXPECTED_IMAGE_SIZE = (600, 450)  # (width, height)


@dataclass
class AuditResult:
    passed: bool
    checks: dict = field(default_factory=dict)
    anomalies: list = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# V1 Integrity Audit (PLAN.md §5.3.2)", ""]
        lines.append(f"**Overall: {'PASS' if self.passed else 'FAIL — see anomalies below'}**")
        lines.append("")
        lines.append("| Check | Result |")
        lines.append("|---|---|")
        for name, val in self.checks.items():
            lines.append(f"| {name} | {val} |")
        if self.anomalies:
            lines.append("")
            lines.append("## Anomalies")
            for a in self.anomalies:
                lines.append(f"- {a}")
        return "\n".join(lines) + "\n"


def run_integrity_audit(
    data_root: Path,
    expected_image_count: int = EXPECTED_IMAGE_COUNT,
    expected_lesion_count: int = EXPECTED_LESION_COUNT,
    expected_class_counts: dict | None = None,
    expected_image_size: tuple[int, int] = EXPECTED_IMAGE_SIZE,
    check_image_sizes: bool = True,
) -> AuditResult:
    expected_class_counts = expected_class_counts or EXPECTED_CLASS_COUNTS
    metadata_path = data_root / "HAM10000_metadata.csv"
    images_dir = data_root / "images"

    anomalies: list[str] = []
    checks: dict = {}

    df = pd.read_csv(metadata_path)
    required_cols = {"image_id", "lesion_id", "dx"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{metadata_path} is missing required columns: {missing_cols}")

    # 1. Unique image_ids, each with an image file present.
    n_images_meta = df["image_id"].nunique()
    checks["unique image_ids in metadata"] = f"{n_images_meta} (expected {expected_image_count})"
    if n_images_meta != expected_image_count:
        anomalies.append(f"metadata has {n_images_meta} unique image_ids, expected {expected_image_count}")

    missing_files = []
    if images_dir.exists():
        for image_id in df["image_id"].unique():
            candidates = [images_dir / f"{image_id}{ext}" for ext in (".jpg", ".jpeg", ".png")]
            if not any(c.exists() for c in candidates):
                missing_files.append(image_id)
    else:
        anomalies.append(f"images directory does not exist: {images_dir}")
    checks["image files missing on disk"] = len(missing_files)
    if missing_files:
        anomalies.append(f"{len(missing_files)} image_ids are missing their file in {images_dir} (e.g. {missing_files[:5]})")

    # 2. Unique lesion_ids, each with exactly one dx.
    n_lesions = df["lesion_id"].nunique()
    checks["unique lesion_ids"] = f"{n_lesions} (expected {expected_lesion_count})"
    if n_lesions != expected_lesion_count:
        anomalies.append(f"metadata has {n_lesions} unique lesion_ids, expected {expected_lesion_count}")

    dx_per_lesion = df.groupby("lesion_id")["dx"].nunique()
    inconsistent_lesions = dx_per_lesion[dx_per_lesion > 1]
    checks["lesions with >1 distinct dx"] = len(inconsistent_lesions)
    if len(inconsistent_lesions) > 0:
        anomalies.append(
            f"{len(inconsistent_lesions)} lesion_ids have more than one distinct dx: "
            f"{list(inconsistent_lesions.index[:5])}"
        )

    # 3. Class counts.
    actual_counts = df["dx"].value_counts().to_dict()
    checks["class counts (image-level)"] = json.dumps(actual_counts, sort_keys=True)
    for cls, expected_n in expected_class_counts.items():
        actual_n = actual_counts.get(cls, 0)
        if actual_n != expected_n:
            anomalies.append(f"class '{cls}': got {actual_n} images, expected {expected_n}")
    unexpected_classes = set(actual_counts) - set(expected_class_counts)
    if unexpected_classes:
        anomalies.append(f"unexpected dx classes present: {unexpected_classes}")

    # 4. Image sizes.
    if check_image_sizes and images_dir.exists():
        bad_sizes = []
        for image_id in df["image_id"].unique():
            candidates = [images_dir / f"{image_id}{ext}" for ext in (".jpg", ".jpeg", ".png")]
            found = next((c for c in candidates if c.exists()), None)
            if found is None:
                continue
            with Image.open(found) as im:
                if im.size != expected_image_size:
                    bad_sizes.append((image_id, im.size))
        checks["images not matching expected size"] = len(bad_sizes)
        if bad_sizes:
            anomalies.append(
                f"{len(bad_sizes)} images are not {expected_image_size} (e.g. {bad_sizes[:5]})"
            )

    return AuditResult(passed=(len(anomalies) == 0), checks=checks, anomalies=anomalies)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V1 integrity audit (PLAN.md §5.3.2).")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--skip-image-sizes", action="store_true", help="Faster; skips opening every image.")
    args = parser.parse_args()

    result = run_integrity_audit(args.data_root, check_image_sizes=not args.skip_image_sizes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result.to_markdown())
    print(result.to_markdown())
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

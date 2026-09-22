"""Acquire HAM10000 (PLAN.md §5.3.1).

Two sources, in the order the plan prefers them:

1. Kaggle: `kmader/skin-cancer-mnist-ham10000` — requires a phone-verified
   Kaggle account and `~/.kaggle/kaggle.json` (or KAGGLE_USERNAME /
   KAGGLE_KEY env vars).
2. Harvard Dataverse fallback: `doi:10.7910/DVN/DBW86T`.

This script does the real download + unzip + merge; it does not need real
data to import cleanly, so it's covered by the unit tests via a
`--dry-run` flag that skips the network call but exercises the same
merge/layout logic on fixture files.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"
DATAVERSE_DOI = "doi:10.7910/DVN/DBW86T"
DATAVERSE_API = "https://dataverse.harvard.edu/api/access/dataset/:persistentId/"


def download_via_kaggle(dest_raw_zip_dir: Path) -> None:
    """Requires the `kaggle` CLI (`pip install kaggle`) and valid credentials."""
    dest_raw_zip_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(dest_raw_zip_dir),
        "--unzip",
    ]
    subprocess.run(cmd, check=True)


def download_via_dataverse(dest_raw_zip_dir: Path) -> None:
    """Requires network access to dataverse.harvard.edu."""
    dest_raw_zip_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-L", "-o", str(dest_raw_zip_dir / "dataverse_ham10000.zip"),
        f"{DATAVERSE_API}?persistentId={DATAVERSE_DOI}",
    ]
    subprocess.run(cmd, check=True)
    with zipfile.ZipFile(dest_raw_zip_dir / "dataverse_ham10000.zip") as zf:
        zf.extractall(dest_raw_zip_dir)


def merge_image_parts(raw_dir: Path, images_out: Path) -> int:
    """Merge `HAM10000_images_part_1/` and `_part_2/` (or loose files) into
    a single flat `images/` directory keyed by image_id. Returns the count
    of images placed.
    """
    images_out.mkdir(parents=True, exist_ok=True)
    n = 0
    candidates = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.jpeg"))
    # Don't re-copy files that are already inside the destination.
    candidates = [p for p in candidates if images_out not in p.parents]
    for src in candidates:
        dst = images_out / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire HAM10000 (PLAN.md §5.3.1).")
    parser.add_argument("--dest", required=True, type=Path, help="e.g. data/raw")
    parser.add_argument("--source", choices=["kaggle", "dataverse"], default="kaggle")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip the network call; only run merge_image_parts on whatever is already in --dest "
             "(used by the test suite against a fixture).",
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        if args.source == "kaggle":
            download_via_kaggle(args.dest)
        else:
            download_via_dataverse(args.dest)

    metadata_candidates = list(args.dest.rglob("HAM10000_metadata.csv"))
    if not metadata_candidates:
        print(
            f"WARNING: no HAM10000_metadata.csv found under {args.dest} after acquisition. "
            "Check the download succeeded.",
            file=sys.stderr,
        )
    else:
        target = args.dest / "HAM10000_metadata.csv"
        if metadata_candidates[0] != target:
            shutil.copy2(metadata_candidates[0], target)

    n = merge_image_parts(args.dest, args.dest / "images")
    print(f"Merged {n} image files into {args.dest / 'images'}")


if __name__ == "__main__":
    main()

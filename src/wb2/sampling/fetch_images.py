"""Fetch only the HAM10000 images a WB2 sampling run needs (implementation.md A1).

    python -m src.wb2.sampling.fetch_images --splits val test --images-dir data/raw/images

`src.data.acquire` downloads and merges the full dataset, keeping zips and
copies side by side (~6 GB peak). WB2 inference only reads val + test
(3,013 images), so this downloads one Dataverse zip at a time, extracts just
the image_ids listed in the frozen C2 split files, then deletes the zip.
Images land in the same flat `data/raw/images/<image_id>.jpg` layout that
`src.model.dataset.HAM10000Split` reads, so V1's eval transform is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

DATAVERSE_FILE_API = "https://dataverse.harvard.edu/api/access/datafile/"
# Dataverse file ids for doi:10.7910/DVN/DBW86T (listed via the dataset API, 2026-10-27).
IMAGE_ZIPS = {
    "HAM10000_images_part_1.zip": 3172585,
    "HAM10000_images_part_2.zip": 3172584,
}


def wanted_image_ids(split_dir: Path, splits: list[str]) -> set[str]:
    ids: set[str] = set()
    for split in splits:
        ids.update(pd.read_csv(split_dir / f"{split}.csv")["image_id"].astype(str))
    return ids


def extract_wanted(zip_path: Path, wanted: set[str], images_dir: Path) -> int:
    """Extract members whose stem is in `wanted`, flattened into `images_dir`."""
    images_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename)
            if name.suffix.lower() != ".jpg" or name.stem not in wanted:
                continue
            dst = images_dir / f"{name.stem}.jpg"
            if not dst.exists():
                dst.write_bytes(zf.read(info))
            n += 1
    return n


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch HAM10000 images for selected splits only.")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/images"))
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--keep-zips", action="store_true")
    args = parser.parse_args()

    wanted = wanted_image_ids(args.split_dir, args.splits)
    have = {p.stem for p in args.images_dir.glob("*.jpg")} if args.images_dir.exists() else set()
    missing = wanted - have
    print(f"{len(wanted)} images needed for {args.splits}; {len(missing)} missing")

    record = {}
    for zip_name, file_id in IMAGE_ZIPS.items():
        if not missing:
            break
        zip_path = args.download_dir / zip_name
        args.download_dir.mkdir(parents=True, exist_ok=True)
        # -C - resumes a partial download after a dropped connection.
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(zip_path), f"{DATAVERSE_FILE_API}{file_id}"], check=True)
        record[zip_name] = {"bytes": zip_path.stat().st_size, "sha256": sha256_of(zip_path)}
        n = extract_wanted(zip_path, missing, args.images_dir)
        print(f"{zip_name}: extracted {n} images")
        if not args.keep_zips:
            zip_path.unlink()
        missing -= {p.stem for p in args.images_dir.glob("*.jpg")}

    if missing:
        raise SystemExit(f"Still missing {len(missing)} images, e.g. {sorted(missing)[:5]}")
    (args.download_dir / "wb2_fetch_record.json").write_text(json.dumps(record, indent=2))
    print(f"All {len(wanted)} images present in {args.images_dir}")


if __name__ == "__main__":
    main()

"""Optional V1 speed-up: a plain-resize 224×224 cache (PLAN.md §5.3.8).

Plain resize only, no crop, no other processing — this is purely an I/O
optimization for B's data loader. B's pipeline must work without it.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from PIL import Image

TARGET_SIZE = (224, 224)


def build_raw_cache(data_root: Path, out_dir: Path) -> pd.DataFrame:
    metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
    images_dir = data_root / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_id in metadata["image_id"]:
        candidates = [images_dir / f"{image_id}{ext}" for ext in (".jpg", ".jpeg", ".png")]
        src = next((c for c in candidates if c.exists()), None)
        if src is None:
            rows.append({"image_id": image_id, "status": "missing_source"})
            continue
        t0 = time.perf_counter()
        with Image.open(src) as im:
            im = im.convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)
            im.save(out_dir / f"{image_id}.png")
        rows.append({"image_id": image_id, "status": "ok", "processing_time_s": time.perf_counter() - t0})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the optional V1 raw-resize cache (PLAN.md §5.3.8).")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="e.g. data/cache_raw224")
    args = parser.parse_args()

    manifest = build_raw_cache(args.data_root, args.out)
    n_ok = (manifest["status"] == "ok").sum()
    print(f"Cached {n_ok}/{len(manifest)} images to {args.out}")


if __name__ == "__main__":
    main()

"""C10 — ground-truth lesion masks aligned to the 224×224 model input
(PLAN.md §8.3 Member A, paper §3.5).

    python -m src.data.masks --zip data/raw/seg_dl/HAM10000_segmentations_lesion_tschandl.zip \
        --split-dir data/splits --out data/masks224

Source: the HAM10000 Harvard Dataverse record's
`HAM10000_segmentations_lesion_tschandl.zip` (file id 3838943), one
600×450 binary PNG per image. Each mask gets exactly the image pipeline's
geometric transform — a plain resize to 224×224 (no crop) — using
nearest-neighbour interpolation, then is binarized. Hair removal and CLAHE
don't move pixels, so image and mask stay aligned.

Writes `<out>/<image_id>.png` (0/255) and `manifest.csv` (C10):
image_id, split, source, has_mask, lesion_fraction.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

TARGET_SIZE = (224, 224)
SOURCE = "HAM10000_segmentations_lesion_tschandl (Harvard Dataverse doi:10.7910/DVN/DBW86T)"


def align_mask(mask: Image.Image) -> np.ndarray:
    """Binary uint8 {0, 1} mask at 224×224, nearest-neighbour."""
    resized = mask.convert("L").resize(TARGET_SIZE, Image.NEAREST)
    return (np.asarray(resized) > 127).astype(np.uint8)


def build_masks(zip_path: Path, split_dir: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = pd.concat([pd.read_csv(split_dir / f"{s}.csv").assign(split=s) for s in ("train", "val", "test")])
    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(n).name.replace("_segmentation.png", ""): n for n in zf.namelist() if n.endswith(".png")}
        rows = []
        for image_id, split in zip(splits["image_id"], splits["split"]):
            name = members.get(image_id)
            if name is None:
                rows.append({"image_id": image_id, "split": split, "source": SOURCE, "has_mask": False,
                             "lesion_fraction": np.nan})
                continue
            mask = align_mask(Image.open(io.BytesIO(zf.read(name))))
            Image.fromarray(mask * 255).save(out_dir / f"{image_id}.png")
            rows.append({"image_id": image_id, "split": split, "source": SOURCE, "has_mask": True,
                         "lesion_fraction": float(mask.mean())})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def load_mask(masks_dir: Path, image_id: str) -> np.ndarray:
    return (np.asarray(Image.open(masks_dir / f"{image_id}.png")) > 127).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build C10 aligned masks (PLAN.md §8.3).")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    m = build_masks(args.zip, args.split_dir, args.out)
    for split, g in m.groupby("split"):
        print(f"{split}: {int(g['has_mask'].sum())}/{len(g)} images have a mask")


if __name__ == "__main__":
    main()

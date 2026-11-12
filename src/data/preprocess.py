"""C9 — the V2 preprocessed image cache (PLAN.md §6.3, paper §3.2.1).

    python -m src.data.preprocess --data-root data/raw --out data/cache_prep224 --workers 16

Per image, on the ORIGINAL full-resolution file (before any resize):
  (a) hair removal: grayscale -> blackhat with a 17x17 elliptical (disk)
      structuring element -> binary threshold at 10 -> TELEA inpainting of
      the colour image with radius 3;
  (b) illumination normalization: RGB -> LAB, CLAHE (clip 2.0, tile 8x8) on
      EACH of L, a, b independently (paper wording) -> back to RGB;
  (c) plain resize to 224x224 (bilinear, no crop), saved as lossless PNG so
      no JPEG re-compression artifacts are introduced.

Writes `manifest.csv` (C9): image_id, hair_fraction, processing_time_s, status.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TARGET_SIZE = (224, 224)
BLACKHAT_KERNEL = 17
HAIR_THRESHOLD = 10
INPAINT_RADIUS = 3
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)


def hair_mask(rgb: np.ndarray) -> np.ndarray:
    """Binary (0/255) hair mask from the blackhat transform, paper §3.2.1a."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KERNEL, BLACKHAT_KERNEL))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, HAIR_THRESHOLD, 255, cv2.THRESH_BINARY)
    return mask


def remove_hair(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = hair_mask(rgb)
    return cv2.inpaint(rgb, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA), mask


def clahe_lab(rgb: np.ndarray) -> np.ndarray:
    """CLAHE on each LAB channel independently, paper §3.2.1b."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    channels = [clahe.apply(lab[:, :, c]) for c in range(3)]
    return cv2.cvtColor(np.stack(channels, axis=2), cv2.COLOR_LAB2RGB)


def preprocess_rgb(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Full C9 transform on one full-resolution RGB uint8 image.

    Returns the 224x224 RGB result and the hair-pixel fraction.
    """
    no_hair, mask = remove_hair(rgb)
    normalized = clahe_lab(no_hair)
    resized = cv2.resize(normalized, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    return resized, float((mask > 0).mean())


def _process_one(args: tuple[str, str, str]) -> dict:
    image_id, src, out_dir = args
    t0 = time.perf_counter()
    bgr = cv2.imread(src, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"image_id": image_id, "status": "unreadable"}
    out, hair_fraction = preprocess_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cv2.imwrite(str(Path(out_dir) / f"{image_id}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return {
        "image_id": image_id,
        "hair_fraction": hair_fraction,
        "processing_time_s": time.perf_counter() - t0,
        "status": "ok",
    }


def build_prep_cache(data_root: Path, out_dir: Path, workers: int = 1) -> pd.DataFrame:
    metadata = pd.read_csv(data_root / "HAM10000_metadata.csv")
    images_dir = data_root / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    rows = []
    for image_id in metadata["image_id"]:
        src = images_dir / f"{image_id}.jpg"
        if not src.exists():
            rows.append({"image_id": image_id, "status": "missing_source"})
            continue
        jobs.append((image_id, str(src), str(out_dir)))

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows.extend(pool.map(_process_one, jobs, chunksize=32))
    else:
        rows.extend(_process_one(j) for j in jobs)

    manifest = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the C9 hair-removed + CLAHE cache (PLAN.md §6.3).")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="e.g. data/cache_prep224")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    t0 = time.perf_counter()
    manifest = build_prep_cache(args.data_root, args.out, workers=args.workers)
    n_ok = int((manifest["status"] == "ok").sum())
    print(f"Preprocessed {n_ok}/{len(manifest)} images to {args.out} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()

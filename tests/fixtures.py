"""Builds a tiny synthetic HAM10000-shaped fixture so Member A's pipeline
logic (label map, split, integrity audit, verification) can be exercised
for real, without the actual 10,015-image dataset (which this environment
cannot download — see data/README.md).

The fixture mirrors HAM10000's real shape: 7 classes with a realistic
imbalance ratio, a mix of single- and multi-image lesions, correct
600x450 JPEGs.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
# Scaled-down but proportionally similar to the real class imbalance
# (real: nv 6705, mel 1113, bkl 1099, bcc 514, akiec 327, vasc 142, df 115).
LESIONS_PER_CLASS = {"nv": 60, "mel": 15, "bkl": 15, "bcc": 10, "akiec": 8, "vasc": 6, "df": 6}


def build_fixture_data_root(tmp_path: Path) -> Path:
    rng = random.Random(0)
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    lesion_counter = 0
    image_counter = 0
    for cls, n_lesions in LESIONS_PER_CLASS.items():
        for _ in range(n_lesions):
            lesion_id = f"HAM_{lesion_counter:07d}"
            lesion_counter += 1
            n_images = rng.choice([1, 1, 1, 2, 3])  # most lesions have 1 image, some have more
            for _ in range(n_images):
                image_id = f"ISIC_{image_counter:07d}"
                image_counter += 1
                Image.new("RGB", (600, 450), color=(rng.randrange(256), rng.randrange(256), rng.randrange(256))).save(
                    images_dir / f"{image_id}.jpg"
                )
                rows.append({"image_id": image_id, "lesion_id": lesion_id, "dx": cls})

    metadata = pd.DataFrame(rows)
    metadata.to_csv(tmp_path / "HAM10000_metadata.csv", index=False)
    return tmp_path


class TempFixture:
    """Context manager: yields a fresh fixture data_root in a temp dir."""

    def __enter__(self) -> Path:
        self._tmpdir = tempfile.TemporaryDirectory()
        return build_fixture_data_root(Path(self._tmpdir.name))

    def __exit__(self, *exc):
        self._tmpdir.cleanup()
        return False

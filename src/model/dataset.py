"""HAM10000 dataset + transforms (PLAN.md §5.3, Member B task 2, paper §3.2.2-3.2.4).

Reads a C2 split file (`image_id,lesion_id,dx,label`) and loads the matching
image from `<images_dir>/<image_id>.jpg`. V1 has no preprocessing cache yet
(hair removal + CLAHE land in V2 as C9) — this reads whatever is in
`images_dir`, which may be the raw acquired images or A's optional V1
raw-resize cache (CONTRACTS.md "V1-only" cache); either works since both are
plain images, and training must work without the cache.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def build_train_transform() -> transforms.Compose:
    """Train-only augmentation, exactly per paper §3.2.4 / PLAN.md §5.3.2."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Val/test: resize + normalize only, no augmentation (paper §3.2.2-3.2.3)."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class HAM10000Split(Dataset):
    """One split ('train'/'val'/'test') read from a C2 split CSV.

    `label` already comes from A's C1 map (baked into the split CSV), so this
    class trusts it rather than recomputing a label map — recomputing here
    would risk silently drifting from C1 if the two ever disagreed.
    """

    def __init__(self, split_csv: Path, images_dir: Path, transform: transforms.Compose):
        self.frame = pd.read_csv(split_csv)
        required = {"image_id", "lesion_id", "dx", "label"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{split_csv} is missing C2 columns: {missing}")
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[idx]
        image_path = self.images_dir / f"{row['image_id']}.jpg"
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, int(row["label"]), str(row["image_id"])

    @property
    def labels(self) -> pd.Series:
        return self.frame["label"]

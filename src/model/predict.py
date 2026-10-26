"""Export C3 single-pass predictions from a trained checkpoint (PLAN.md §5.3.10).

    python -m src.model.predict --split-dir data/splits --images-dir data/raw/images \
        --checkpoint runs/v1-b0-baseline/best.pt --split test --run-id v1-b0-baseline \
        --out runs/v1-b0-baseline/predictions_test.csv

V1 is single-pass only (dropout off) — MC Dropout (C4, T=50) is a V3 task.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.model.dataset import HAM10000Split, build_eval_transform
from src.model.model import NUM_CLASSES, build_model


@torch.no_grad()
def predict_split(checkpoint_path: Path, split_dir: Path, images_dir: Path, split: str, device: str = "cpu") -> pd.DataFrame:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = HAM10000Split(split_dir / f"{split}.csv", images_dir, build_eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    image_ids: list[str] = []
    y_true: list[int] = []
    probs: list[np.ndarray] = []
    for images, labels, batch_image_ids in loader:
        logits = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        y_true.extend(labels.tolist())
        image_ids.extend(batch_image_ids)

    probs_arr = np.concatenate(probs, axis=0)
    out = pd.DataFrame(
        {
            "image_id": image_ids,
            "split": split,
            "y_true": y_true,
            **{f"prob_{c}": probs_arr[:, c] for c in range(NUM_CLASSES)},
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export C3 predictions (PLAN.md §5.3.10).")
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = predict_split(args.checkpoint, args.split_dir, args.images_dir, args.split, device=args.device)
    out["run_id"] = args.run_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} predictions to {args.out}")


if __name__ == "__main__":
    main()

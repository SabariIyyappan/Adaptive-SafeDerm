"""C7 — Grad-CAM + IoU engine (paper §3.5, Eqs. 11-12; PLAN.md §8.3 Member B).

    python -m src.model.gradcam --checkpoint runs/v2-main-b0/best.pt \
        --split-dir data/splits --images-dir data/cache_prep224 --image-ext png \
        --masks-dir data/masks224 --split test --out-dir runs/v2-main-b0

Per image, model in eval mode (dropout OFF):
  - hooks on `model.features[-1]` (EfficientNet-B0: 1280×7×7);
  - target class = the PREDICTED class (paper doesn't state it; logged in
    docs/deviations.md);
  - alpha_k = spatial mean of d y_c / d A_k; CAM = ReLU(sum_k alpha_k A_k);
  - bilinear upsample to 224×224, normalize to [0, 1];
  - binarize at 50% of the map's own maximum; IoU against the C10 mask.

Writes `gradcam_<split>.csv` (C7: image_id, y_true, pred_class,
target_class, iou, cam_area, mask_area) and `gradcam_<split>.npz` with the
raw 7×7 CAMs (upsampling is deterministic, so the 224 map regenerates).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.masks import load_mask
from src.model.dataset import HAM10000Split, build_eval_transform
from src.model.mc_dropout import load_checkpoint_model

CAM_THRESHOLD = 0.5
INPUT_SIZE = 224


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model
        self._acts = None
        self._grads = None
        layer.register_forward_hook(self._save_acts)
        layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _module, _inp, out):
        self._acts = out

    def _save_grads(self, _module, _grad_in, grad_out):
        self._grads = grad_out[0]

    def __call__(self, images: torch.Tensor, target: torch.Tensor | None = None):
        """Returns (raw CAMs B×h×w, logits B×7, target B)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        if target is None:
            target = logits.argmax(dim=1)
        logits.gather(1, target[:, None]).sum().backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cams = F.relu((weights * self._acts).sum(dim=1))
        return cams.detach(), logits.detach(), target


def upsample_normalize(cams: torch.Tensor | np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Bilinear upsample raw CAMs to size×size and min-max normalize to [0, 1]."""
    t = torch.as_tensor(cams, dtype=torch.float32)
    up = F.interpolate(t[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    lo = up.amin(dim=(1, 2), keepdim=True)
    hi = up.amax(dim=(1, 2), keepdim=True)
    return ((up - lo) / (hi - lo).clamp_min(1e-12)).numpy()


def binarize(cam: np.ndarray, threshold: float = CAM_THRESHOLD) -> np.ndarray:
    """Threshold at `threshold` × the map's own maximum (paper §3.5)."""
    return cam >= threshold * cam.max()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def run(checkpoint: Path, split_dir: Path, images_dir: Path, image_ext: str, masks_dir: Path, split: str,
        out_dir: Path, device: str, batch_size: int = 32) -> pd.DataFrame:
    model, _ = load_checkpoint_model(checkpoint, device)
    for p in model.parameters():
        p.requires_grad_(True)
    cam_engine = GradCAM(model, model.features[-1])
    ds = HAM10000Split(split_dir / f"{split}.csv", images_dir, build_eval_transform(), image_ext=image_ext)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    rows, raw = [], []
    for images, labels, image_ids in loader:
        cams, logits, target = cam_engine(images.to(device))
        raw.append(cams.cpu().numpy())
        maps = upsample_normalize(cams.cpu())
        for k, image_id in enumerate(image_ids):
            mask_path = masks_dir / f"{image_id}.png"
            cam_bin = binarize(maps[k])
            mask = load_mask(masks_dir, image_id) if mask_path.exists() else None
            rows.append({
                "image_id": image_id, "y_true": int(labels[k]), "pred_class": int(logits[k].argmax()),
                "target_class": int(target[k]), "iou": iou(cam_bin, mask) if mask is not None else np.nan,
                "cam_area": float(cam_bin.mean()), "mask_area": float(mask.mean()) if mask is not None else np.nan,
                "cam_index": len(rows),
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"gradcam_{split}.csv", index=False)
    np.savez_compressed(out_dir / f"gradcam_{split}.npz", cams=np.concatenate(raw).astype(np.float32),
                        image_ids=df["image_id"].to_numpy())
    print(f"{split}: {len(df)} CAMs, mean IoU {df['iou'].mean():.4f} over {int(df['iou'].notna().sum())} masked images")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="C7 Grad-CAM + IoU (PLAN.md §8.3).")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--image-ext", default="png")
    parser.add_argument("--masks-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args.checkpoint, args.split_dir, args.images_dir, args.image_ext, args.masks_dir, args.split, args.out_dir,
        args.device)


if __name__ == "__main__":
    main()

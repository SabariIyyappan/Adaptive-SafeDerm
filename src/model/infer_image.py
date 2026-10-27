"""Run the trained checkpoint on a single image and print per-class scores.

    python -m src.model.infer_image --image data/raw/images/ISIC_0024306.jpg \
        --checkpoint runs/v1-b0-baseline/best.pt

Loads the same EfficientNet-B0 + dropout-head architecture used in training
(src/model/model.py) and the same eval-time resize/normalize transform used
by predict.py (src/model/dataset.py), so scores are directly comparable to
runs/v1-b0-baseline/predictions_test.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.data.labels import EXPECTED_HAM10000_LABEL_MAP
from src.model.dataset import build_eval_transform
from src.model.model import build_model

# int label -> class name, inverse of EXPECTED_HAM10000_LABEL_MAP
IDX_TO_CLASS = {idx: name for name, idx in EXPECTED_HAM10000_LABEL_MAP.items()}

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


@torch.no_grad()
def predict_image(image_path: Path, checkpoint_path: Path, device: str = "cpu") -> dict[str, float]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = build_eval_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    return {IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(probs))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a single image with the trained checkpoint.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", default=Path("runs/v1-b0-baseline/best.pt"), type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scores = predict_image(args.image, args.checkpoint, device=args.device)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    print(f"Image: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    print(f"{'class':<8} {'description':<45} {'score'}")
    for cls, score in ranked:
        print(f"{cls:<8} {CLASS_DESCRIPTIONS[cls]:<45} {score:.4f}")

    top_cls, top_score = ranked[0]
    print()
    print(f"Predicted: {top_cls} ({CLASS_DESCRIPTIONS[top_cls]}) - {top_score:.1%} confidence")


if __name__ == "__main__":
    main()

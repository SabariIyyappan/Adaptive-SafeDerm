"""E0: do the new deterministic logits reproduce V1's committed predictions? (implementation.md §5 E0)

    python -m src.wb2.sampling.e0_check --bank-dir results/wb2/inputs/wb2-mc-v1b0-s42 --split test

Compares softmax(det_logits) row-by-row (joined on image_id) against
`results/v1/inputs/predictions_{split}.csv`. Writes `e0_{split}.json` with
argmax agreement and max/mean absolute probability difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.eval.metrics import PROB_COLUMNS, load_predictions

TOLERANCE = 1e-3  # GPU/MPS/CPU numerics; Sai measured 3.2e-4 on 16 images


def compare(bank_dir: Path, split: str, v1_dir: Path) -> dict:
    ids = pd.read_csv(bank_dir / f"{split}_ids.csv")
    det = softmax(np.load(bank_dir / f"{split}_det_logits.npz")["logits"].astype(np.float64), axis=1)
    v1 = load_predictions(v1_dir / f"predictions_{split}.csv").set_index("image_id").loc[ids["image_id"]]
    v1_probs = v1[PROB_COLUMNS].to_numpy(dtype=float)
    diff = np.abs(det - v1_probs)
    return {
        "split": split,
        "n_images": int(len(ids)),
        "labels_match": bool((v1["y_true"].to_numpy() == ids["y_true"].to_numpy()).all()),
        "argmax_agreement": float((det.argmax(1) == v1_probs.argmax(1)).mean()),
        "argmax_disagreements": int((det.argmax(1) != v1_probs.argmax(1)).sum()),
        "max_abs_prob_diff": float(diff.max()),
        "mean_abs_prob_diff": float(diff.mean()),
        "tolerance": TOLERANCE,
        "within_tolerance": bool(diff.max() <= TOLERANCE),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E0 reproduction check against committed V1 predictions.")
    parser.add_argument("--bank-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--v1-dir", type=Path, default=Path("results/v1/inputs"))
    args = parser.parse_args()

    result = compare(args.bank_dir, args.split, args.v1_dir)
    out = args.bank_dir / f"e0_{args.split}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

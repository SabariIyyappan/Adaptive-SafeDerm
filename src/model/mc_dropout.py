"""C4 — MC Dropout inference engine (paper §3.4.2, PLAN.md §7.3 Member B).

    python -m src.model.mc_dropout --checkpoint runs/v2-main-b0/best.pt \
        --split-dir data/splits --images-dir data/cache_prep224 --image-ext png \
        --splits val test --T 50 --seed 42 --out-dir runs/v2-main-b0

Mode handling: the whole model goes to eval mode, then ONLY the head's
nn.Dropout is switched back to train mode. BatchNorm keeps its running
statistics and EfficientNet-B0's 16 StochasticDepth blocks stay
deterministic (they are not Dropout layers).

Speed-up (PLAN.md §7.3 item 4): the head is the only stochastic part, so the
pooled backbone features are computed once per image and the head is
sampled T times. `verify_feature_reuse` checks on real images that this
reproduces full forward passes exactly under the same RNG seed.

Writes `mc_<split>.npz` per split (C4):
    image_ids (N,), y_true (N,), logits_mc (T,N,7), logits_det (N,7),
    probs_mc (T,N,7), probs_det (N,7), T, seed, checkpoint_sha256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.dataset import HAM10000Split, build_eval_transform
from src.model.model import arch_of, build_model, pooled_features
from src.model.train import file_sha256


def enable_head_dropout(model: torch.nn.Module) -> list[str]:
    """eval() everywhere, then train() on head Dropout only. Returns the
    names of modules left in training mode (should be exactly the head
    dropout) so callers can assert on it.
    """
    model.eval()
    for module in model.classifier.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
    return [name for name, m in model.named_modules() if m.training]


def load_checkpoint_model(checkpoint_path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False, arch=arch_of(ckpt)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def extract_features(model, loader, device) -> tuple[torch.Tensor, list[str], list[int]]:
    model.eval()
    feats, ids, labels = [], [], []
    for images, y, image_ids in loader:
        feats.append(pooled_features(model, images.to(device)))
        ids.extend(image_ids)
        labels.extend(y.tolist())
    return torch.cat(feats), ids, labels


@torch.no_grad()
def sample_head(model, features: torch.Tensor, T: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (logits_mc T×N×7, logits_det N×7)."""
    model.eval()
    logits_det = model.classifier(features)
    active = enable_head_dropout(model)
    assert all(name.startswith("classifier") for name in active), active
    torch.manual_seed(seed)
    logits_mc = torch.stack([model.classifier(features) for _ in range(T)])
    model.eval()
    return logits_mc, logits_det


@torch.no_grad()
def verify_feature_reuse(model, images: torch.Tensor, T: int = 5, seed: int = 0) -> float:
    """Max |difference| between full-network MC passes and head-only passes
    on cached features, drawn with the same seed. Should be ~0 (float noise).
    """
    enable_head_dropout(model)
    torch.manual_seed(seed)
    full = torch.stack([model(images) for _ in range(T)])
    model.eval()
    feats = pooled_features(model, images)
    head, _ = sample_head(model, feats, T, seed)
    return float((full - head).abs().max())


def bn_state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "running_" in k or "num_batches" in k}


def run(checkpoint: Path, split_dir: Path, images_dir: Path, image_ext: str, splits: list[str], T: int,
        seed: int, out_dir: Path, device: str, batch_size: int = 64) -> dict:
    model, _ = load_checkpoint_model(checkpoint, device)
    sha = file_sha256(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    bn_before = bn_state(model)
    report = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha, "T": T, "seed": seed, "splits": {}}

    for split in splits:
        ds = HAM10000Split(split_dir / f"{split}.csv", images_dir, build_eval_transform(), image_ext=image_ext)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        if "verify_max_abs_diff" not in report:
            first_images = torch.stack([ds[i][0] for i in range(32)]).to(device)
            report["verify_max_abs_diff"] = verify_feature_reuse(model, first_images)
        feats, ids, labels = extract_features(model, loader, device)
        logits_mc, logits_det = sample_head(model, feats, T, seed)
        probs_mc = torch.softmax(logits_mc, dim=-1)
        probs_det = torch.softmax(logits_det, dim=-1)

        per_pass_std = float(probs_mc.std(dim=0).mean())
        mean_vs_det = float((probs_mc.mean(dim=0) - probs_det).abs().mean())
        np.savez_compressed(
            out_dir / f"mc_{split}.npz",
            image_ids=np.array(ids), y_true=np.array(labels, dtype=np.int64),
            logits_mc=logits_mc.cpu().numpy().astype(np.float32), logits_det=logits_det.cpu().numpy().astype(np.float32),
            probs_mc=probs_mc.cpu().numpy().astype(np.float32), probs_det=probs_det.cpu().numpy().astype(np.float32),
            T=T, seed=seed, checkpoint_sha256=sha,
        )
        report["splits"][split] = {
            "n": len(ids),
            "order_matches_split_csv": ids == ds.frame["image_id"].tolist(),
            "mean_per_pass_std": per_pass_std,
            "mean_abs_mc_mean_minus_det": mean_vs_det,
            "all_finite": bool(torch.isfinite(logits_mc).all() and torch.isfinite(logits_det).all()),
        }
        print(f"{split}: N={len(ids)} T={T} mean per-pass std {per_pass_std:.4f}, |mean-det| {mean_vs_det:.4f}")

    bn_after = bn_state(model)
    report["batchnorm_unchanged"] = all(torch.equal(bn_before[k], bn_after[k]) for k in bn_before)
    (out_dir / "mc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"feature-reuse check max|diff| {report['verify_max_abs_diff']:.2e}; BN unchanged: {report['batchnorm_unchanged']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="C4 MC Dropout predictions (PLAN.md §7.3).")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--image-ext", default="png")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args.checkpoint, args.split_dir, args.images_dir, args.image_ext, args.splits, args.T, args.seed,
        args.out_dir, args.device)


if __name__ == "__main__":
    main()

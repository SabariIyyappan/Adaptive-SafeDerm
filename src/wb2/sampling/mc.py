"""MC-Dropout inference adapter around the unchanged V1 model (implementation.md A2).

Only the classifier-head `Dropout(0.3)` (`classifier.2`) is made stochastic:
`model.eval()` first, then `model.classifier[2].train()`. `model.train()` is
never called — it would also enable torchvision's 16 StochasticDepth modules
and put BatchNorm in batch-statistics mode, updating its running stats
(docs/wb2_checkpoint_handoff.md, "MC-Dropout activation").

Because that dropout sits after global average pooling, the 1280-d backbone
features can be computed once per image and reused for every head draw
(`feature_reuse=True`): T MC passes = 1 backbone pass + T head passes. The
`feature_reuse=False` path re-runs the full network per pass; it exists for
the timing study and the equivalence test, and consumes the same RNG draws
in the same order, so both paths give identical logits under one seed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from src.model.model import build_model

HEAD_DROPOUT_MODULE = "classifier.2"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v1(checkpoint_path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load the V1 checkpoint strictly; returns (model in eval mode, checkpoint info)."""
    # weights_only=False: the checkpoint stores its training config dict next to the weights.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    info = {"epoch": ckpt.get("epoch"), "checkpoint_sha256": sha256_of(checkpoint_path)}
    return model, info


def enable_head_dropout(model: nn.Module) -> nn.Module:
    """Eval mode everywhere except the head Dropout. Idempotent."""
    model.eval()
    model.classifier[2].train()
    return model


def stochastic_modules(model: nn.Module) -> list[str]:
    """Names of leaf modules currently in training mode (expected: ['classifier.2'] in MC mode)."""
    return [name for name, m in model.named_modules() if name and m.training and not list(m.children())]


@torch.no_grad()
def backbone_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """EfficientNet-B0 features + avgpool + flatten — everything before `model.classifier`."""
    return torch.flatten(model.avgpool(model.features(images)), 1)


@torch.no_grad()
def deterministic_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Dropout fully off (T=1 deterministic), shape (B, 7)."""
    model.eval()
    return model(images)


@torch.no_grad()
def mc_logits(model: nn.Module, images: torch.Tensor, n_passes: int, feature_reuse: bool = True) -> torch.Tensor:
    """Ordered stochastic head-dropout logits, shape (B, n_passes, 7).

    Callers seed the global RNG (torch.manual_seed) before calling; pass t
    always consumes the t-th dropout mask, regardless of `feature_reuse`.
    """
    enable_head_dropout(model)
    try:
        if feature_reuse:
            feats = backbone_features(model, images)
            passes = [model.classifier(feats) for _ in range(n_passes)]
        else:
            passes = [model(images) for _ in range(n_passes)]
    finally:
        model.eval()
    return torch.stack(passes, dim=1)

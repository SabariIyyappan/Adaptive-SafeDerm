"""EfficientNet-B0 + the paper's dropout head (paper §3.3, PLAN.md §5.3.3),
plus the V5 comparison backbones (paper §4.7, PLAN.md §9.3).

Keeps the ImageNet-pretrained convolutional feature stack and global average
pooling, and replaces the *entire* original classifier (including its
built-in dropout) with Linear(features->256) -> ReLU -> Dropout(0.3) ->
Linear(256->7). All parameters are trainable (§3.3.3: no freeze-then-unfreeze
stage).

V5 puts the SAME head on each comparison backbone's pooled features, so only
the backbone differs (PLAN.md §9.3 shared protocol). The head always lives
at `model.classifier` (ResNet's `fc` is replaced by Identity), so MC Dropout
and feature reuse treat every architecture identically.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    densenet121,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet50,
)

NUM_CLASSES = 7
BACKBONE_FEATURES = 1280
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.3

ARCHITECTURES = ("efficientnet_b0", "mobilenet_v3_large", "resnet50", "densenet121")


def build_head(in_features: int) -> nn.Sequential:
    # Dropout sits only where §3.3.2 puts it: right before the final linear
    # layer, never after the logits.
    return nn.Sequential(
        nn.Linear(in_features, HEAD_HIDDEN),
        nn.ReLU(inplace=True),
        nn.Dropout(p=HEAD_DROPOUT),
        nn.Linear(HEAD_HIDDEN, NUM_CLASSES),
    )


class _ResNetWithHead(nn.Module):
    """ResNet-50 with `fc` removed and the shared head at `.classifier`."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = build_head(in_features)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def build_model(pretrained: bool = True, arch: str = "efficientnet_b0") -> nn.Module:
    if arch == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        # torchvision's own classifier is Sequential(Dropout, Linear) —
        # replaced wholesale.
        model.classifier = build_head(BACKBONE_FEATURES)
        return model
    if arch == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        # torchvision's MobileNetV3 classifier is Linear(960->1280)+Hardswish+
        # Dropout+Linear; replaced wholesale so the head matches B0's.
        model.classifier = build_head(model.classifier[0].in_features)
        return model
    if arch == "resnet50":
        return _ResNetWithHead(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None))
    if arch == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier = build_head(model.classifier.in_features)
        return model
    raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHITECTURES}")


def pooled_features(model: nn.Module, x):
    """Backbone output after global pooling, i.e. the head's input.

    Everything before `model.classifier` is deterministic in eval mode, so
    MC Dropout can compute this once per image and sample only the head
    (PLAN.md §7.3 item 4).
    """
    import torch
    import torch.nn.functional as F

    if isinstance(model, _ResNetWithHead):
        return model.forward_features(x)
    if hasattr(model, "avgpool"):  # EfficientNet, MobileNetV3
        return torch.flatten(model.avgpool(model.features(x)), 1)
    # DenseNet: features -> ReLU -> adaptive avg pool (torchvision's forward)
    out = F.relu(model.features(x), inplace=False)
    return torch.flatten(F.adaptive_avg_pool2d(out, (1, 1)), 1)


def arch_of(checkpoint: dict) -> str:
    """Architecture a checkpoint was trained with (V1 checkpoints predate the key)."""
    return checkpoint.get("config", {}).get("arch", "efficientnet_b0")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

"""EfficientNet-B0 + the paper's dropout head (paper §3.3, PLAN.md §5.3.3).

Keeps the ImageNet-pretrained convolutional feature stack and global average
pooling, and replaces the *entire* original classifier (including its
built-in dropout) with Linear(1280->256) -> ReLU -> Dropout(0.3) ->
Linear(256->7). All parameters are trainable (§3.3.3: no freeze-then-unfreeze
stage).
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

NUM_CLASSES = 7
BACKBONE_FEATURES = 1280
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.3


def build_model(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    # torchvision's own classifier is Sequential(Dropout, Linear) — replaced
    # wholesale so the paper's dropout sits only where §3.3.2 puts it: right
    # before the final linear layer, never after the logits.
    model.classifier = nn.Sequential(
        nn.Linear(BACKBONE_FEATURES, HEAD_HIDDEN),
        nn.ReLU(inplace=True),
        nn.Dropout(p=HEAD_DROPOUT),
        nn.Linear(HEAD_HIDDEN, NUM_CLASSES),
    )
    return model

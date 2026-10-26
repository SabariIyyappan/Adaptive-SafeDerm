"""Dual imbalance correction (paper §3.2.5, Eq. 2, PLAN.md §5.3.4).

Both corrections are computed from **train-split counts only**, per the
plan — using val/test counts here would leak split information into
training.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def class_counts(labels: Sequence[int], num_classes: int) -> list[int]:
    counter = Counter(labels)
    counts = [counter.get(c, 0) for c in range(num_classes)]
    if any(c == 0 for c in counts):
        raise ValueError(f"class_counts got a zero-count class among train labels: {counts}")
    return counts


def class_weights_for_loss(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """w_c = (1/n_c) / sum_k(1/n_k) — paper Eq. 2, for CrossEntropyLoss(weight=...)."""
    counts = class_counts(labels, num_classes)
    inv = [1.0 / n for n in counts]
    total = sum(inv)
    return torch.tensor([w / total for w in inv], dtype=torch.float32)


def build_weighted_sampler(labels: Sequence[int], num_classes: int) -> WeightedRandomSampler:
    """Per-sample weight w_i = 1/n_c of that sample's class, sampling with
    replacement, `num_samples` equal to the train size (replaces shuffling).
    """
    counts = class_counts(labels, num_classes)
    sample_weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )

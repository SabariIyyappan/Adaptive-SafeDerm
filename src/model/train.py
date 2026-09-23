"""Training loop for `v1-b0-baseline` (PLAN.md §5.3, Member B tasks 5-9).

    python -m src.model.train --split-dir data/splits --images-dir data/raw/images \
        --config configs/v1-b0-baseline.yaml --out-dir runs/v1-b0-baseline

Produces, under `--out-dir`:
    history.csv       C5 — one row per epoch
    last.pt           checkpoint from the most recent epoch
    best.pt           checkpoint with the best validation macro F1 so far
    checkpoint.json   C6 — run_id, best epoch, selection metric, config ref, file hash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.model.dataset import HAM10000Split, build_eval_transform, build_train_transform
from src.model.imbalance import build_weighted_sampler, class_weights_for_loss
from src.model.model import NUM_CLASSES, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_epoch_train(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels, _image_ids in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def run_epoch_eval(model, loader, criterion, device) -> tuple[float, float, float]:
    """Returns (loss, macro_f1, macro_auroc) — dropout off (deterministic)."""
    model.eval()
    total_loss = 0.0
    n = 0
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
    for images, labels, _image_ids in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.cpu().tolist())

    probs = np.concatenate(all_probs, axis=0)
    preds = probs.argmax(axis=1)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    # roc_auc_score needs every class present in y_true to score one-vs-rest for it;
    # V1's val/test splits are guaranteed (C2) to contain all 7 classes.
    macro_auroc = roc_auc_score(all_labels, probs, multi_class="ovr", average="macro", labels=list(range(NUM_CLASSES)))
    return total_loss / n, macro_f1, macro_auroc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(
    split_dir: Path,
    images_dir: Path,
    config: dict,
    out_dir: Path,
    device: str = "cpu",
    num_workers: int = 0,
) -> None:
    set_seed(config["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = HAM10000Split(split_dir / "train.csv", images_dir, build_train_transform())
    val_ds = HAM10000Split(split_dir / "val.csv", images_dir, build_eval_transform())

    train_labels = train_ds.labels.tolist()
    sampler = build_weighted_sampler(train_labels, NUM_CLASSES)
    class_weights = class_weights_for_loss(train_labels, NUM_CLASSES).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], sampler=sampler, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers
    )

    model = build_model(pretrained=config.get("pretrained", True)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    history_rows = []
    best_val_macro_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, config["epochs"] + 1):
        train_loss = run_epoch_train(model, train_loader, optimizer, criterion, device)
        val_loss, val_macro_f1, val_macro_auroc = run_epoch_eval(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
                "val_macro_auroc": val_macro_auroc,
                "lr": lr,
            }
        )
        print(
            f"epoch {epoch}/{config['epochs']}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_macro_f1={val_macro_f1:.4f}  val_macro_auroc={val_macro_auroc:.4f}"
        )

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}, out_dir / "last.pt")
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}, out_dir / "best.pt"
            )

    pd.DataFrame(history_rows).to_csv(out_dir / "history.csv", index=False)

    checkpoint_meta = {
        "run_id": config["run_id"],
        "best_epoch": best_epoch,
        "selection_metric": "val_macro_f1",
        "selection_metric_value": best_val_macro_f1,
        "config_reference": config.get("_config_path"),
        "best_checkpoint_sha256": file_sha256(out_dir / "best.pt"),
    }
    with open(out_dir / "checkpoint.json", "w") as f:
        json.dump(checkpoint_meta, f, indent=2)
    print(f"Best epoch {best_epoch} (val_macro_f1={best_val_macro_f1:.4f}). Wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v1-b0-baseline (PLAN.md §5.3, Member B).")
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    config["_config_path"] = str(args.config)
    train(args.split_dir, args.images_dir, config, args.out_dir, device=args.device, num_workers=args.num_workers)


if __name__ == "__main__":
    main()

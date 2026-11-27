"""Measured inference-cost microbenchmark (WB2-I4 timing record, implementation.md A4).

    python -m src.wb2.sampling.timing --config configs/wb2_sampling.yaml --split val --n-images 50

Times, on the same warmed model and the same first `n_images` images of the
split (in split order):
  - preprocessing alone (JPEG open + eval transform), per image;
  - deterministic T=1 (one full forward);
  - MC T in {10, 20, 50} with feature reuse (1 backbone + T head passes);
  - MC T=50 naive (T full-network passes).
Latency is per image at batch size 1 (p50/p95 over images x repeats), plus
throughput at the config batch size. Accelerator work is synchronised before
each clock read. Every number here is measured on this machine; nothing is
derived from saved logits.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.model.dataset import HAM10000Split, build_eval_transform
from src.wb2.sampling.bank import hardware_description, load_config, resolve_device
from src.wb2.sampling.mc import deterministic_logits, load_v1, mc_logits


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_call(fn, device: str) -> float:
    synchronize(device)
    start = time.perf_counter()
    fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0


def summarize(times_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(times_ms)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean()), "n_timings": int(len(arr))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure WB2 inference latency.")
    parser.add_argument("--config", type=Path, default=Path("configs/wb2_sampling.yaml"))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--n-images", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, help="default: <out_dir>/timing.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    model, info = load_v1(Path(cfg["checkpoint"]), device)
    dataset = HAM10000Split(Path(cfg["split_dir"]) / f"{args.split}.csv", Path(cfg["images_dir"]), build_eval_transform())
    transform = build_eval_transform()
    paths = [dataset.images_dir / f"{iid}.jpg" for iid in dataset.frame["image_id"].iloc[: args.n_images]]

    prep_times = [time_call(lambda p=p: transform(Image.open(p).convert("RGB")), "cpu") for p in paths]
    images = torch.stack([dataset[i][0] for i in range(len(paths))]).to(device)
    torch.manual_seed(cfg["seed"])

    methods = [("deterministic_T1", 1, None)]
    methods += [(f"mc_T{t}_feature_reuse", t, True) for t in (10, 20, 50)]
    methods += [("mc_T50_naive_full_network", 50, False)]

    def runner(n_passes, feature_reuse, batch):
        if feature_reuse is None:
            return lambda: deterministic_logits(model, batch)
        return lambda: mc_logits(model, batch, n_passes, feature_reuse)

    records = []
    for name, n_passes, feature_reuse in methods:
        for _ in range(args.warmup):
            runner(n_passes, feature_reuse, images[:1])()
        per_image = [time_call(runner(n_passes, feature_reuse, images[i:i + 1]), device)
                     for _ in range(args.repeats) for i in range(len(images))]
        batch = images[: cfg["batch_size"]]
        batch_ms = [time_call(runner(n_passes, feature_reuse, batch), device) for _ in range(args.repeats)]
        records.append({
            "method": name,
            "n_passes": n_passes,
            "feature_reuse": feature_reuse,
            "backbone_passes_per_image": 1 if feature_reuse in (None, True) else n_passes,
            "head_passes_per_image": n_passes,
            "latency_batch1": summarize(per_image),
            "throughput_batch": {"batch_size": int(len(batch)),
                                 "images_per_s": float(len(batch) / (np.median(batch_ms) / 1000.0))},
        })
        print(f"{name:28s} p50 {records[-1]['latency_batch1']['p50_ms']:8.2f} ms  "
              f"p95 {records[-1]['latency_batch1']['p95_ms']:8.2f} ms")

    record = {
        "status": "MEASURED",
        "run_id": cfg["run_id"],
        "checkpoint_sha256": info["checkpoint_sha256"],
        "device": device,
        "hardware": hardware_description(device),
        "torch_version": torch.__version__,
        "split": args.split,
        "n_images": len(paths),
        "image_selection": f"first {len(paths)} rows of data/splits/{args.split}.csv",
        "warmup_calls_per_method": args.warmup,
        "repeats": args.repeats,
        "inference_excludes_preprocessing": True,
        "preprocessing_per_image_cpu": summarize(prep_times),
        "methods": records,
        "note": "Latency is model inference on preloaded tensors; add preprocessing for end-to-end. "
                "Feature-reuse MC runs the backbone once; naive MC runs it once per pass.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out = args.out or Path(cfg["out_dir"]) / "timing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

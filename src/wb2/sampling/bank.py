"""Generate a WB2 stochastic logit bank for one split (contracts WB2-I1/I2/I3).

    python -m src.wb2.sampling.bank --config configs/wb2_sampling.yaml --split val

Writes, under the config's `out_dir` (one directory per sampling run):
    {split}_mc_logits.npz   key `logits`, float32 (N, n_passes, 7) — raw logits, axis 1 = 0-based pass order
    {split}_det_logits.npz  key `logits`, float32 (N, 7) — dropout fully off (deterministic T=1)
    {split}_ids.csv         image_id, lesion_id, y_true — row i matches row i of both arrays
    {split}_manifest.json   provenance + status (COMPLETE / PARTIAL / FAILED)

Work is done in fixed-size shards cached under `shard_dir` (git-ignored
`runs/`), so an interrupted run resumes where it stopped and still produces
the same draws: the global RNG is reseeded to `seed + shard_index` at the
start of every shard. Draws therefore depend on (seed, shard_size,
batch_size, device) — all recorded in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader, Subset

from src.model.dataset import HAM10000Split, build_eval_transform
from src.wb2.sampling.mc import deterministic_logits, enable_head_dropout, load_v1, mc_logits, stochastic_modules

CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
EXPECTED_ROWS = {"val": 1532, "test": 1481}
TRANSFORM_DESCRIPTION = (
    "src.model.dataset.build_eval_transform: Resize((224,224)) -> ToTensor -> "
    "Normalize(ImageNet mean/std); raw HAM10000 JPEGs, no hair removal / CLAHE (V1 path)"
)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware_description(device: str) -> str:
    if device == "cuda":
        accel = torch.cuda.get_device_name(0)
    elif device == "mps":
        accel = "Apple MPS"
    else:
        accel = "CPU only"
    return f"{accel}; {platform.platform()}; {platform.processor() or platform.machine()}"


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True).stdout.strip()
        return f"{sha}{'+dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_shard(model, dataset, indices: list[int], shard_seed: int, cfg: dict, device: str) -> dict:
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg.get("num_workers", 0))
    torch.manual_seed(shard_seed)
    mc, det, ids = [], [], []
    for images, _labels, image_ids in loader:
        images = images.to(device)
        det.append(deterministic_logits(model, images).float().cpu().numpy())
        mc.append(mc_logits(model, images, cfg["n_passes"], cfg["feature_reuse"]).float().cpu().numpy())
        ids.extend(image_ids)
    return {"mc": np.concatenate(mc).astype(np.float32), "det": np.concatenate(det).astype(np.float32),
            "image_ids": np.array(ids)}


def generate_bank(cfg: dict, split: str, image_ids: list[str] | None = None) -> dict:
    """Run (or resume) one split and write the four delivery files. Returns the manifest."""
    device = resolve_device(cfg.get("device", "auto"))
    split_csv = Path(cfg["split_dir"]) / f"{split}.csv"
    out_dir = Path(cfg["out_dir"])
    shard_dir = Path(cfg["shard_dir"]) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    model, info = load_v1(Path(cfg["checkpoint"]), device)
    if cfg.get("expected_checkpoint_sha256") and info["checkpoint_sha256"] != cfg["expected_checkpoint_sha256"]:
        raise SystemExit(f"Checkpoint SHA256 {info['checkpoint_sha256']} != expected {cfg['expected_checkpoint_sha256']}")

    dataset = HAM10000Split(split_csv, Path(cfg["images_dir"]), build_eval_transform())
    frame = dataset.frame
    if image_ids is not None:  # predeclared pilot subset, kept in split-file order
        wanted = set(image_ids)
        positions = [i for i, iid in enumerate(frame["image_id"]) if iid in wanted]
    else:
        positions = list(range(len(frame)))

    shard_size = cfg["shard_size"]
    shards = [positions[i:i + shard_size] for i in range(0, len(positions), shard_size)]
    done: list[dict] = []
    error = None
    try:
        for k, indices in enumerate(shards):
            path = shard_dir / f"shard_{k:04d}.npz"
            expected_ids = frame["image_id"].iloc[indices].to_numpy()
            if path.exists():
                cached = dict(np.load(path))
                if np.array_equal(cached["image_ids"], expected_ids):
                    done.append(cached)
                    continue
            result = run_shard(model, dataset, indices, cfg["seed"] + k, cfg, device)
            assert np.array_equal(result["image_ids"], expected_ids), "DataLoader order drifted from split order"
            np.savez(path, **result)
            done.append(result)
            print(f"[{split}] shard {k + 1}/{len(shards)} done ({sum(len(d['image_ids']) for d in done)} images)")
    except (Exception, KeyboardInterrupt) as exc:  # keep whatever finished; status says so
        error = f"{type(exc).__name__}: {exc}"
        print(f"[{split}] stopped early: {error}")

    n_expected = len(positions)
    mc = np.concatenate([d["mc"] for d in done]) if done else np.zeros((0, cfg["n_passes"], 7), np.float32)
    det = np.concatenate([d["det"] for d in done]) if done else np.zeros((0, 7), np.float32)
    got_ids = np.concatenate([d["image_ids"] for d in done]) if done else np.array([], dtype=str)
    rows = frame.set_index("image_id").loc[got_ids].reset_index()

    if not np.isfinite(mc).all() or not np.isfinite(det).all():
        error = (error or "") + " non-finite logits"
    status = "COMPLETE" if len(got_ids) == n_expected and not error else ("FAILED" if len(got_ids) == 0 or "non-finite" in (error or "") else "PARTIAL")

    np.savez_compressed(out_dir / f"{split}_mc_logits.npz", logits=mc)
    np.savez_compressed(out_dir / f"{split}_det_logits.npz", logits=det)
    pd.DataFrame({"image_id": rows["image_id"], "lesion_id": rows["lesion_id"], "y_true": rows["label"].astype(int)}).to_csv(
        out_dir / f"{split}_ids.csv", index=False
    )

    enable_head_dropout(model)
    stochastic = stochastic_modules(model)
    model.eval()
    manifest = {
        "run_id": cfg["run_id"],
        "split": split,
        "status": status,
        "error": error,
        "synthetic": False,
        "pilot_subset": image_ids is not None,
        "n_images": int(len(got_ids)),
        "n_images_expected": int(n_expected),
        "n_passes": int(cfg["n_passes"]),
        "class_order": CLASS_ORDER,
        "logit_kind": "raw logits (pre-softmax), uncalibrated",
        "checkpoint_sha256": info["checkpoint_sha256"],
        "checkpoint_epoch": info["epoch"],
        "seed": int(cfg["seed"]),
        "seed_policy": "torch.manual_seed(seed + shard_index) at the start of each shard",
        "shard_size": int(shard_size),
        "batch_size": int(cfg["batch_size"]),
        "transform": TRANSFORM_DESCRIPTION,
        "split_csv": str(split_csv),
        "split_csv_sha256": file_sha256(split_csv),
        "stochastic_modules": stochastic,
        "feature_reuse": bool(cfg["feature_reuse"]),
        "backbone_passes_per_image": 1 if cfg["feature_reuse"] else int(cfg["n_passes"]),
        "head_passes_per_image": int(cfg["n_passes"]),
        "deterministic_logits": "separate file; model.eval() only, dropout off",
        "device": device,
        "hardware": hardware_description(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "git_commit": git_commit(),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / f"{split}_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def validate_bank(out_dir: Path, split: str, split_csv: Path | None = None) -> dict:
    """Check a delivered bank against the WB2-I1/I2/I3 layout; raises ValueError on any mismatch.

    Mirrors the checks Sabari's `src/wb2/controller/bank.py:load_bank` performs,
    so a bank can be verified before handoff without his branch.
    """
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / f"{split}_manifest.json").read_text())
    mc = np.load(out_dir / f"{split}_mc_logits.npz")["logits"]
    det = np.load(out_dir / f"{split}_det_logits.npz")["logits"]
    ids = pd.read_csv(out_dir / f"{split}_ids.csv")

    problems = []
    n = len(ids)
    if list(ids.columns) != ["image_id", "lesion_id", "y_true"]:
        problems.append(f"ids columns {list(ids.columns)}")
    if mc.dtype != np.float32 or det.dtype != np.float32:
        problems.append(f"dtypes mc={mc.dtype} det={det.dtype}")
    if mc.shape != (n, manifest["n_passes"], 7):
        problems.append(f"mc shape {mc.shape} != ({n}, {manifest['n_passes']}, 7)")
    if det.shape != (n, 7):
        problems.append(f"det shape {det.shape} != ({n}, 7)")
    if not (np.isfinite(mc).all() and np.isfinite(det).all()):
        problems.append("non-finite logits")
    if manifest["n_images"] != n:
        problems.append(f"manifest n_images {manifest['n_images']} != {n} rows")
    if manifest["class_order"] != CLASS_ORDER:
        problems.append(f"class_order {manifest['class_order']}")
    if manifest["stochastic_modules"] != ["classifier.2"]:
        problems.append(f"stochastic_modules {manifest['stochastic_modules']}")
    if manifest["status"] not in {"COMPLETE", "PARTIAL", "FAILED"}:
        problems.append(f"status {manifest['status']}")
    if split_csv is not None:
        split_df = pd.read_csv(split_csv)
        if manifest["split_csv_sha256"] != file_sha256(split_csv):
            problems.append("split_csv_sha256 does not match the split file")
        if manifest["status"] == "COMPLETE" and not manifest.get("pilot_subset"):
            if not (ids["image_id"].tolist() == split_df["image_id"].tolist()
                    and ids["y_true"].tolist() == split_df["label"].tolist()
                    and ids["lesion_id"].tolist() == split_df["lesion_id"].tolist()):
                problems.append("ids do not match the split file row-for-row")
        else:
            merged = ids.merge(split_df, on="image_id", how="left")
            if merged["label"].isna().any() or (merged["label"] != merged["y_true"]).any():
                problems.append("ids contain images/labels not in the split file")
    if problems:
        raise ValueError(f"{out_dir}/{split}: " + "; ".join(problems))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a WB2 MC-Dropout logit bank (WB2-I1/I2/I3).")
    parser.add_argument("--config", type=Path, default=Path("configs/wb2_sampling.yaml"))
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--pilot-ids", type=Path, help="CSV with an image_id column: predeclared PILOT ONLY subset")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_ids = pd.read_csv(args.pilot_ids)["image_id"].tolist() if args.pilot_ids else None
    manifest = generate_bank(cfg, args.split, image_ids)
    validate_bank(Path(cfg["out_dir"]), args.split, Path(cfg["split_dir"]) / f"{args.split}.csv")
    print(f"[{args.split}] {manifest['status']}: {manifest['n_images']}/{manifest['n_images_expected']} images -> {cfg['out_dir']}")


if __name__ == "__main__":
    main()

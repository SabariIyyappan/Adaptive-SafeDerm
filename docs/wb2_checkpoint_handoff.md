# wb2_checkpoint_handoff.md — V1 checkpoint handoff (WB2-H0, Sai → Soumya)

`implementation.md` §3 tasks B1/B2. Owner: Sai Teja Nomula (Member B). Date: 2026-10-27.
The weights are **not** in Git (`runs/` is git-ignored); they are shared out-of-band.

## Checkpoint identity

| Field | Value |
|---|---|
| File | `best.pt` (from `runs/v1-b0-baseline/` on Sai's laptop) |
| Location | [Google Drive folder](https://drive.google.com/drive/folders/1gN7yWLa3MJDYdqdxJFo0TrHJro3BvhLl?usp=drive_link) (also uploaded: `checkpoint.json`, `history.csv`) |
| Size | 17,629,275 bytes |
| SHA256 | `901a73a6a689cfb4617d3897152fcb33c73d40bbbf190978218fff4614f9ae0d` (matches `results/v1/inputs/checkpoint.json` and `docs/run_registry.md`) |
| Run ID / config | `v1-b0-baseline` / `configs/v1-b0-baseline.yaml` (seed 42, AdamW lr 1e-4, wd 1e-4, batch 32, 30 epochs, cosine) |
| Epoch | 30 of 30 (best by val macro F1 = 0.5839; no early stop) |
| Training commit | `a919142` |
| Training hardware | NVIDIA RTX 3060 Laptop GPU, 6 GB, ~35 min |
| `last.pt` | Also epoch 30 with tensor-for-tensor identical weights; SHA256 differs (`dfdb695f…92d1`) only because torch's zip format embeds the file name. Use `best.pt`. |

## Load configuration

- Format: `{"epoch": int, "model_state_dict": ..., "config": dict}` (`src/model/train.py`).
- Load: `build_model(pretrained=False)` then `load_state_dict(ckpt["model_state_dict"])` (as in `src/model/predict.py`). With torch ≥ 2.6, pass `weights_only=False` to `torch.load` because `config` is a plain dict saved alongside the weights.
- Architecture (`src/model/model.py`): torchvision EfficientNet-B0 features + avgpool, then `classifier = Sequential(Linear(1280,256), ReLU, Dropout(0.3), Linear(256,7))`.
- Labels (C1, `src/data/labels.py`): akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6.
- Eval preprocessing (`build_eval_transform`, `src/model/dataset.py`): Resize((224,224)) → ToTensor → Normalize(ImageNet mean (0.485,0.456,0.406), std (0.229,0.224,0.225)). Raw images from `data/raw/images/`, no hair removal/CLAHE (V1 is un-preprocessed).

## MC-Dropout activation (B2)

- The model has **17** stochastic modules: the intended head `Dropout(0.3)` at `classifier.2`, **plus 16 `StochasticDepth` modules** inside the MBConv blocks (torchvision default), and the backbone has BatchNorm.
- **Do not call `model.train()`** for MC sampling. That would enable StochasticDepth and switch BatchNorm to batch statistics, updating its running stats.
- Safe recipe: `model.eval()` then `model.classifier[2].train()`. Only `classifier.2` is then in train mode.
- Because the only intended stochasticity sits after global pooling, the 1280-d backbone features can be computed once per image and reused across T head draws. In that implementation, "T=50 MC passes" means 1 backbone pass + 50 head passes, not 50 full-network passes. Report the two counts separately (§5 E1, §6 Efficiency).

## Load check — MEASURED (2026-10-27, Sai's laptop)

Environment: Python 3.14.4, torch 2.14.0+cu126, torchvision 0.29.0+cu126, RTX 3060 Laptop GPU; repo at `2623795`.

| Check | Result |
|---|---|
| SHA256 of `best.pt` | matches the table above |
| `load_state_dict(strict=True)` | all keys matched |
| Output on 4×3×224×224 dummy batch | shape (4, 7), all finite |
| Eval mode, two passes | identical (max abs diff 0.0) |
| Head-dropout-only mode, two passes | differ (max abs logit diff 0.245), finite |
| BatchNorm running stats after MC passes | unchanged |
| 16 real test images vs committed `results/v1/inputs/predictions_test.csv` | argmax 16/16 agree; max abs prob diff 3.2e-4 (GPU/library numerics) |

This is Sai's own sanity check. **Soumya's independent load and inference verification (A1/E0) is still required.**

## Caveats

- Weights and credentials must never be committed. `runs.7z` in Sai's working tree is not git-ignored. Do not stage it.
- No new training was done for this handoff.

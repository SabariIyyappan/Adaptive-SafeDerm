# Member B → Member C: V1 Handoff

**From:** Member B (Model, Training, Predictions) · **To:** Member C (Evaluation, Analysis & Reporting)
**Date:** 2026-09-23 · **Branch:** `member-b-v1` → `second-main`

## TL;DR

`v1-b0-baseline` is trained on the **real** HAM10000 dataset and its predictions are ready for your evaluation suite. Nothing here is synthetic/dummy data anymore.

## What's built and run (merged/committed)

1. **Model** — EfficientNet-B0 (ImageNet-pretrained) + the paper's head: Linear(1280→256) → ReLU → Dropout(0.3) → Linear(256→7). All params trainable. `src/model/model.py`
2. **Data pipeline** — resize 224, train-only augmentation (flips, ±15° rotation, color jitter), ImageNet normalization. `src/model/dataset.py`
3. **Dual imbalance correction** — WeightedRandomSampler + class-weighted cross-entropy, both from train-split counts only. `src/model/imbalance.py`
4. **Training loop** — 30 epochs, no early stopping, cosine LR, AdamW (lr 1e-4, wd 1e-4, batch 32 — provisional Table 2 values, see `docs/deviations.md`). `src/model/train.py`
5. **Real run, `v1-b0-baseline`** — trained on the real, frozen C2 split (`data/splits/`), RTX 3060 laptop GPU (6GB), ~35 min. Full row in `docs/run_registry.md`.

## What you actually need: the C3 files

```
runs/v1-b0-baseline/predictions_val.csv
runs/v1-b0-baseline/predictions_test.csv
```

**Not in git** (per `.gitignore` — `runs/` is never committed). You'll need to regenerate them yourself once you pull `member-b-v1` / `second-main`:

```bash
pip install -r env/requirements.txt
python -m src.model.predict --split-dir data/splits --images-dir data/raw/images \
    --checkpoint runs/v1-b0-baseline/best.pt --split val --run-id v1-b0-baseline \
    --out runs/v1-b0-baseline/predictions_val.csv
python -m src.model.predict --split-dir data/splits --images-dir data/raw/images \
    --checkpoint runs/v1-b0-baseline/best.pt --split test --run-id v1-b0-baseline \
    --out runs/v1-b0-baseline/predictions_test.csv
```

**Problem:** `runs/v1-b0-baseline/best.pt` (the checkpoint) is also gitignored and only exists on my machine. You'll need either the checkpoint file itself (I'll share it via the team's shared storage — Drive/Kaggle dataset, per PLAN.md §3) or you can retrain from scratch with the exact same command (seed 42 makes it reproducible, though not bit-identical across different GPUs):

```bash
python -m src.model.train --split-dir data/splits --images-dir data/raw/images \
    --config configs/v1-b0-baseline.yaml --out-dir runs/v1-b0-baseline --device cuda
```

**C3 schema** (exactly per `CONTRACTS.md`):

| column | notes |
|---|---|
| `image_id` | e.g. `ISIC_0024306` |
| `split` | `"val"` or `"test"` |
| `y_true` | int, 0–6, via C1 |
| `prob_0` … `prob_6` | softmax probabilities, verified to sum to 1.0 |
| `run_id` | `"v1-b0-baseline"` |

Row counts: val = 1532, test = 1481 (matches `data/splits/val.csv` / `test.csv` exactly).

## C5 — training history (also needed for your Figure 2 / Table 3)

```
runs/v1-b0-baseline/history.csv
```
Columns: `epoch, train_loss, val_loss, val_macro_f1, val_macro_auroc, lr` — 30 rows. Best epoch was **30/30** (val macro F1 0.5839), so there's no early "peak epoch" to mark — the curve is still gently improving at epoch 30, unlike the paper's epoch 26 peak. Worth calling out in your training-curves writeup.

## C6 — checkpoint metadata

```
runs/v1-b0-baseline/checkpoint.json
```
Has `run_id`, `best_epoch`, `selection_metric` (`val_macro_f1`), its value, the config path, and the checkpoint's sha256 (for the run registry / reproducibility).

## Headline numbers you'll be sanity-checking against

| Metric | Our V1 (single-pass) | Paper (T=50, w/ preprocessing) |
|---|---|---|
| Val macro AUROC | 0.9233 | 0.9404 (test, T=50) |
| Val macro F1 | 0.5839 | 0.7308 (test, T=50) |

**Don't be alarmed the F1 gap looks bigger than the AUROC gap** — that's expected for V1: no hair-removal/CLAHE preprocessing (V2) and no MC Dropout T=50 averaging (V3) yet. Per PLAN.md §5.6's sanity checks: AUROC well above the unmitigated baseline (0.80) and in the region of 0.94 ✓, all 7 classes predicted (no collapse into `nv` — I checked: predicted counts range 21–565 across classes) ✓, val macro F1 trending upward over epochs ✓.

## Deviations you should know about (full detail in `docs/deviations.md`)

- Provisional Table 2 hyperparameters (AdamW, lr 1e-4, wd 1e-4, batch 32) — paper's real Table 2 wasn't readable; update `configs/v1-b0-baseline.yaml` and retrain if the PDF surfaces later.
- Trained on a 6GB GPU (plan recommends ≥8GB) — worked fine for B0 at batch 32, no OOM, but keep in mind for V2/V5 when models get bigger.

## What I need from you back

Once you've run your eval suite, `results/v1/` needs: figures (training curves, confusion matrix, per-class bars, ROC curves), `metrics.json`, `summary.md` — per PLAN.md §5.3 Member C tasks. A needs those by Day 2 midday for the deck.

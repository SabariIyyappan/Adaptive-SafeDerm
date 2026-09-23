# PLAN.md — Versioned Replication Plan

**Paper being replicated:** Randhawa, Suddala, Hemal, Meesala, Rahman, Biswas, Batumalay & Naik, *"Skin Lesion Classification in Low-Resource Settings Using Lightweight CNNs with Uncertainty Estimation,"* BioMedInformatics 2026, 6(4):53. https://doi.org/10.3390/biomedinformatics6040053

**Scope decisions (agreed):**
- **Faithful replication only.** No additions beyond what the paper does.
- **Edge-latency benchmarks (paper §4.9, Table 9) are skipped.**
- **V1 is due in 2 days** and must show a trained baseline plus core metrics and graphs.

**This document contains no code.** It gives the plan, the structure, the task split, the prerequisites, and the expected outputs for every version.

---

## Status (updated 2026-09-23)

**V1 — Member A: done.** Real HAM10000 acquired and full pipeline run. PR: https://github.com/SabariIyyappan/Adaptive-SafeDerm/pull/1 (`member-a-v1` → `second-main`, merged)

- Done: C1 label map, acquisition script (Kaggle primary, Harvard Dataverse fallback), integrity audit, the lesion-grouped dx-stratified 70/15/15 split (C2), split verification, EDA figures, and the architecture figure (paper Figure 1 equivalent). All unit-tested against a synthetic HAM10000-shaped fixture — 16/16 tests passing.
- **Real dataset acquired and run:** the real 10,015-image HAM10000 dataset was downloaded via `src.data.acquire --source kaggle` and run through the full pipeline. Integrity audit **PASS** (exact match: 10,015 images, 7,470 lesions, class counts identical to the paper's reported distribution). Split verification **PASS** (zero lesion/image leakage, every class present in every split). The real, frozen C2 split is committed at `data/splits/{train,val,test}.csv` (7002/1532/1481 images). EDA outputs at `results/v1/eda/`.
- **Done when checklist:** all met — split files pass every verification check against the real dataset, B has confirmed the real split loads and trains correctly.

**V1 — Member B: done.** Real `v1-b0-baseline` trained on the real HAM10000 dataset; see `docs/run_registry.md`.

- Done: `env/requirements.txt` (shared pinned spec), the data pipeline (resize + train-only augmentation + ImageNet normalization, `src/model/dataset.py`), the EfficientNet-B0 + 256-unit dropout head model (`src/model/model.py`), dual imbalance correction — WeightedRandomSampler and class-weighted cross-entropy (`src/model/imbalance.py`), the training loop with per-epoch history (C5), checkpointing (C6) and cosine LR (`src/model/train.py`), and single-pass C3 prediction export (`src/model/predict.py`). Provisional Table-2 hyperparameters are pinned in `configs/v1-b0-baseline.yaml` and logged in `docs/deviations.md`.
- **Real run complete:** `v1-b0-baseline` trained for the full 30 epochs on the real split (RTX 3060 laptop GPU, 6GB VRAM, ~35 min) and exported C3 predictions for val and test. Results: best epoch 30/30, val macro F1 0.5839, val macro AUROC 0.9233 (single-pass, no preprocessing — see `docs/run_registry.md` for why this isn't directly comparable to the paper's final T=50 numbers). All four of §5.3's "done when" checks pass: 30-row history, prediction counts match split sizes, probabilities sum to 1, all 7 classes present.
- **Handoff to C:** see `results/v1/member_b_handoff.md` for exactly what to consume and how.
- **Next:** Member C runs the evaluation suite on `runs/v1-b0-baseline/predictions_{val,test}.csv` to produce `results/v1/` figures, tables and `metrics.json` for the deck.

---

## 0. How this plan is organized

The replication is split into **5 versions**. Each version ends in a *complete, presentable result package*: a working pipeline, trained or evaluated artifacts, figures, tables and metrics. No version ends at "preprocessing done".

| Version | Name | Paper sections covered | Headline output | Suggested duration |
|---|---|---|---|---|
| **V1** | Trained baseline + core evaluation | §3.1, §3.2.2–3.2.5, §3.3, §3.7.1–3.7.2, §3.7.5–3.7.6 | Trained EfficientNet-B0, single-pass test metrics, core graphs | **2 days (prof demo)** |
| **V2** | Paper-faithful training + imbalance ablation | §3.2.1, §3.6, §4.1, §4.3, §4.8 | Locked "main model", paper Table 8 ablation | 5–7 days |
| **V3** | MC Dropout uncertainty + referral | §3.4, §3.7.3–3.7.4, §4.2, §4.4–4.6 | Paper Tables 4 and 6, Figures 3, 6 and 7 | 5–7 days |
| **V4** | Grad-CAM explainability + IoU | §3.5, Table 10, Figures 8–11 | Quantitative IoU plus qualitative CAM figures | 5–7 days |
| **V5** | Comparative baselines + final replication report | §4.7, Table 7 | MobileNetV3 / ResNet-50 / DenseNet-121 comparison and the full "Paper vs Ours" report | 7–10 days |

Each version section below follows the same template:
1. Goal and scope
2. Pipeline structure
3. Tasks per member, with prerequisites
4. Intra-version dependencies
5. Result package: what gets shown
6. Paper reference values and sanity checks
7. Risks and fallbacks

---

## 1. What the paper specifies (reference sheet)

Every setting the replication must match is listed here. **TBD** means the value is only in a paper table that wasn't readable in the text version (see §14).

| Component | Paper specification | Source | Status |
|---|---|---|---|
| Dataset | HAM10000, 10,015 dermoscopic images, 7 classes | §3.1 | Known |
| Label map | Alphabetical, built on the full dataset before splitting: akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6 | §3.2.2 | Known |
| Split | Group by `lesion_id`; 70/15/15 split stratified by `dx`, `random_state=42`, applied to unique lesions; each lesion's images go to exactly one split | §3.1 | Known (reported counts inconsistent, see §13) |
| Hair removal | Grayscale → blackhat morphology with 17×17 disk structuring element → binary threshold 10 → OpenCV TELEA inpainting, radius 3. Applied to train, val and test, **before resizing** | §3.2.1a | Known |
| Illumination normalization | CLAHE on **each channel of LAB** independently, clip limit 2.0, tile grid 8×8 | §3.2.1b | Known |
| Resize | 224×224 (plain resize, no crop) | §3.2.3 | Known |
| Normalization | ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) | §3.2.3 | Known |
| Augmentation (train only) | Horizontal flip p=0.5, vertical flip p=0.5, rotation ±15°, colour jitter (brightness 0.2, contrast 0.2, saturation 0.2, hue 0.1) | §3.2.4 | Known |
| Imbalance correction (1) | WeightedRandomSampler, per-sample weight w_i = 1/n_c | §3.2.5a | Known |
| Imbalance correction (2) | Class-weighted cross-entropy, w_c = (1/n_c) / Σ_k (1/n_k) | §3.2.5b, Eq. 2 | Known |
| Backbone | torchvision EfficientNet-B0, `IMAGENET1K_V1` weights, 1280-dim features | §3.3.1, §3.3.3 | Known |
| Head | Linear(1280→256) → ReLU → Dropout(p=0.3) → Linear(256→7). Dropout sits **before the final linear layer**, never after the logits | §3.3.2, Eqs. 4–5 | Known |
| Fine-tuning | Entire network trained end-to-end (no freeze-then-unfreeze stage) | §3.3.3 | Known |
| Epochs | 30, no early stopping | §4.1 | Known |
| LR schedule | Cosine annealing | §3.6 | Known (exact parameters TBD) |
| Optimizer, learning rate, batch size, weight decay | — | Table 2 | **TBD** |
| Checkpoint selection | Best checkpoint was epoch 26; the Figure 2 caption ties it to validation macro-F1 | §4.1–4.2, Fig. 2 | Inferred: **best validation macro-F1** |
| MC Dropout | T=50 passes; only Dropout layers active, BatchNorm frozen in eval mode | §3.4.2 | Known |
| Uncertainty | Total = entropy of mean prediction; aleatoric = mean per-pass entropy; epistemic = total − aleatoric (nats) | §3.4.3, Eqs. 8–10 | Known |
| Referral | Refer when H > τ, with τ = (1−r)-quantile of **test-set** entropy, r ∈ {0, .05, .10, .15, .20, .25, .30} | §3.4.4 | Known |
| Grad-CAM | Hooks on `model.features[-1]` (1280×7×7), bilinear upsample to 224, threshold at 50% of the map's max, IoU against the ground-truth mask | §3.5, Eqs. 11–12 | Known (target class not stated, see §13) |
| Metrics | Macro AUROC one-vs-rest (principal metric), macro F1, ECE with 10 bins, Brier score, per-class accuracy (Eq. 17), sensitivity, specificity | §3.7 | Known |
| Confidence intervals | 95% bootstrap, 1000 resamples of the test set, for AUROC, macro F1, ECE and referral AUROC | §3.7.6 | Known |

---

## 2. Paper target values (for "Ours vs Paper" tables)

| Quantity | Paper value | Where |
|---|---|---|
| Test macro AUROC (T=50) | **0.9404** [0.9351, 0.9455] | §4.2 |
| Test macro F1 (T=50) | **0.7308** [0.7109, 0.7497] | §4.2 |
| ECE, T=50 | 0.1456 [0.1311, 0.1601] | §4.2 |
| ECE, T=1 | 0.1891 [0.1722, 0.2057] | §4.2 |
| Mean total entropy / mean epistemic | 0.3209 / 0.0113 nats | §3.4.3, §4.2 |
| Melanoma sensitivity / specificity | 80.8% / 96.2% | §4.3 |
| Per-class accuracy | 6 of 7 classes ≥70%; nv = 69.8% | §4.3 |
| Unmitigated baseline (no imbalance correction) | AUROC 0.8038, F1 0.3600, mel sensitivity 30.5%, df sensitivity 0% | §4.2–4.3 |
| Training dynamics | Train loss 0.8728 → 0.1483 within 5 epochs; val loss oscillates 0.53–0.66; best epoch 26 | §4.1 |
| Referral at r=10% | AUROC +0.77%, macro F1 +2.57% | §4.6 |
| Referral at r=30% | AUROC 0.9568 [0.9525, 0.9609] (+1.64%), macro F1 0.8099 (+7.91%) | §4.6 |
| Ablation: both corrections vs sampler-only | +1.8% AUROC, +7.0% macro F1 | §4.8 |
| Ablation: both corrections vs loss-only | +1.0% AUROC, +10.9% macro F1 | §4.8 |
| Ablation: largest single gains | ImageNet pretraining +6–8% AUROC; weighted sampler +4–5% AUROC | §4.8 |
| Grad-CAM IoU | 0.61 (accepted predictions), 0.41 (referred) | Abstract, Conclusion |
| Comparison | Competitive with ResNet-50 and DenseNet-121 at about 1/5 and 2/3 of their parameter counts | §4.7 |

---

## 3. Team roles and how to assign people

Roles stay **fixed across all versions**, so nobody has to re-learn a different part of the codebase each week.

| Role | Owns | Assign to the person who… |
|---|---|---|
| **A — Data, Preprocessing & Masks** | Dataset acquisition, split, preprocessing cache, ground-truth masks, V1 results deck, and one training run in V2 and V5 | …has good CPU, disk and internet, plus a Kaggle account. Needs a GPU only for single B0 / MobileNetV3 runs (free Kaggle is enough). |
| **B — Model, Training & Inference** | Model definition, training loop, main checkpoints, MC Dropout inference, Grad-CAM engine | …has the **most reliable GPU access** (Colab Pro, a local NVIDIA GPU with ≥8 GB, or university HPC). B's runs sit on the critical path of V1 and V2. |
| **C — Evaluation, Analysis & Reporting** | Metric suite, bootstrap CIs, all plots, referral analysis, ablation and comparison tables, final report | …is comfortable with statistics and plotting. Mostly CPU-only, but needs a 12–16 GB GPU for the V2 ablation runs and the V5 DenseNet-121 run. |

### Compute profiles (referenced in every task)

| Profile | Specification | Examples that qualify | Typical time |
|---|---|---|---|
| **P-CPU** | Any laptop: 4+ cores, ≥8 GB RAM, ≥10 GB free disk, Python 3.10+ | Any recent laptop | — |
| **P-CPU-MC** | Multi-core CPU (8+ threads preferred), ≥8 GB RAM, ≥15 GB disk | Laptop/desktop, or a Kaggle/Colab CPU session | Hair removal + CLAHE on 10,015 images at 600×450: roughly 20–60 min parallelized |
| **P-GPU-S** | NVIDIA GPU with ≥8 GB VRAM (16 GB preferred), a CUDA version supported by current PyTorch, ≥12 GB system RAM, persistent storage for checkpoints | Kaggle T4 / P100 (free, weekly quota, 12 h session cap), Colab T4, local RTX 3060 12 GB / 2070 8 GB | EfficientNet-B0 or MobileNetV3, 30 epochs at 224 px, batch 32: about 45–90 min on a T4 (faster with a pre-resized cache) |
| **P-GPU-L** | NVIDIA GPU with **≥12 GB VRAM (16 GB preferred)**, ≥16 GB system RAM | Kaggle T4 / P100 16 GB, Colab T4/L4/A100, local RTX 3080 12 GB+ | ResNet-50: about 1.5–2.5 h; DenseNet-121: about 2–3 h (30 epochs, T4) |
| **P-GPU-INF** | Any CUDA GPU ≥4 GB. CPU fallback works but is slower | Anything above, or a laptop CPU | MC Dropout / Grad-CAM over about 3,000 images: minutes on GPU, up to about 1 h on CPU |

**Accounts every member needs before V1 starts:**
- A **phone-verified Kaggle account**. Verification is required for GPU and internet access in Kaggle notebooks, and can take time, so do it on Day 1, Hour 0. Check your remaining weekly GPU quota on your own account.
- Access to one shared storage location (Google Drive folder or a private Kaggle dataset) for splits, caches, checkpoints and prediction files.
- Access to the shared Git repository.

> **Fill in before starting:** A = Soumya · B = ________ · C = ________

---

## 4. Working agreements (how we avoid overlap and merge conflicts)

### 4.1 Repository structure

```
repo/
├── PLAN.md
├── CONTRACTS.md            ← data contracts (owner: A; changes announced to team)
├── env/                    ← pinned environment spec (owner: B)
├── configs/                ← one config file per run (owner: whoever runs it)
├── data/                   ← NOT in git; README + checksums only (owner: A)
│   ├── README.md
│   ├── splits/             ← frozen train/val/test files (C2)
│   ├── cache_raw224/       ← V1 optional speed-up cache
│   ├── cache_prep224/      ← V2+ hair-removed + CLAHE cache (C9)
│   └── masks224/           ← V4 ground-truth masks (C10)
├── src/
│   ├── data/               ← owner: A (split, preprocessing, mask alignment)
│   ├── model/              ← owner: B (model, training, MC inference, Grad-CAM)
│   └── eval/               ← owner: C (metrics, bootstrap, plots, referral, tables)
├── runs/                   ← NOT in git; checkpoints + prediction files per run ID
├── results/
│   ├── v1/ v2/ v3/ v4/ v5/ ← figures, tables, metrics JSON per version
│   └── final_report/
└── docs/
    ├── run_registry.md     ← every training/inference run, one row each
    └── deviations.md       ← every place we differ from the paper, with reason
```

### 4.2 File ownership rule
**Each person edits only the folders they own.** Other members *use* those modules but don't modify them. If a change is needed, ask the owner. This removes merge conflicts almost entirely. `CONTRACTS.md` has a single owner (A); changes to it are announced in the team channel before anyone relies on them.

### 4.3 Data contracts (the interfaces between members)

Define these on Day 1, Hour 0. They are what let everyone work in parallel.

| ID | Artifact | Producer → Consumer | Required content |
|---|---|---|---|
| **C1** | Label map | A → all | The fixed alphabetical mapping from §1 |
| **C2** | Split files | A → B, C | One file per split with `image_id`, `lesion_id`, `dx`, `label` |
| **C3** | Single-pass predictions | B → C | Per image: `image_id`, `split`, `y_true`, `prob_0`…`prob_6` (sums to 1), `run_id` |
| **C4** | MC Dropout predictions | B → A, C | Per-pass probabilities shaped T×N×7, plus an index of `image_id` and `y_true` in the same order, plus deterministic (dropout-off) probabilities N×7, `run_id`, seed |
| **C5** | Training history | B → C | One row per epoch: `epoch`, `train_loss`, `val_loss`, `val_macro_f1`, `val_macro_auroc`, `lr` |
| **C6** | Checkpoint metadata | B → all | `run_id`, epoch, selection metric value, config reference, file hash |
| **C7** | Grad-CAM outputs | B → C | Per image: `image_id`, predicted class, target class used, IoU, path to the stored 224×224 CAM |
| **C8** | Referral decisions | C → C (V4), report | Per image and per r: entropy, τ, accepted/referred flag |
| **C9** | Preprocessed image cache | A → B, C | 224×224 lossless images plus a manifest (`image_id`, hair-pixel fraction, processing time) |
| **C10** | Aligned GT masks | A → B, C | 224×224 binary masks plus a manifest (`image_id`, source, coverage flag) |

### 4.4 "Dummy-first" rule
Every consumer first builds and tests their part against a **dummy file that follows the contract**: random but valid probabilities, fake histories, synthetic masks. Real artifacts get dropped in at integration time. This is what makes the tasks inside a version independent.

### 4.5 Reproducibility conventions
- **Seed 42** everywhere (split, sampler, augmentation, bootstrap), matching the paper's `random_state=42`.
- Every run gets a `run_id` (e.g. `v2-main-b0`, `v2-abl-nocorr`), a config file, and a row in `docs/run_registry.md`: who ran it, where, GPU type, duration, git commit, final metrics.
- **The split is generated once in V1 and frozen for every later version.** Never regenerate it.
- Checkpoints are saved **every epoch** (last) plus best-by-val-macro-F1, to persistent storage, so a notebook disconnect loses at most one epoch.
- Every difference from the paper goes into `docs/deviations.md` with a reason.

---

## 5. V1 — Trained Baseline + Core Evaluation (2 days, professor demo)

### 5.1 Goal and scope
Show a **working, trained EfficientNet-B0** built with the paper's architecture, split, augmentation and dual imbalance correction, evaluated with the paper's core metrics and graphs on the untouched test split.

| In V1 | Deferred (and to which version) |
|---|---|
| Lesion-grouped 70/15/15 split with leakage verification | Hair removal + CLAHE → V2 |
| EfficientNet-B0 plus the paper's 256-unit dropout head | Final Table 2 hyperparameters → V2 |
| Paper augmentation and ImageNet normalization | Ablation study → V2 |
| WeightedRandomSampler + class-weighted cross-entropy | MC Dropout (T=50), ECE, entropy, referral → V3 |
| 30 epochs, cosine LR, best-val-macro-F1 checkpoint | Grad-CAM + IoU → V4 |
| **Single-pass** (dropout off) test evaluation | Comparative CNNs → V5 |
| Macro AUROC, macro F1, per-class accuracy / sensitivity / specificity, confusion matrix, ROC curves, training curves, bootstrap CIs | — |

### 5.2 Pipeline structure (V1)

```
HAM10000 raw ──► integrity audit ──► label map (C1) ──► lesion-grouped split (C2, frozen)
                                                              │
         ┌────────────────────────────────────────────────────┘
         ▼
 resize 224 ─► train-only augmentation ─► ImageNet normalize
         ▼
 EfficientNet-B0 (IMAGENET1K_V1) ─► Linear 1280→256 ─► ReLU ─► Dropout 0.3 ─► Linear 256→7
         ▼
 training: WeightedRandomSampler + class-weighted CE, cosine LR, 30 epochs
         ▼
 best-val-macro-F1 checkpoint (C6) + history (C5)
         ▼
 single-pass predictions on val/test (C3)
         ▼
 evaluation suite ─► metrics + CIs ─► figures & tables ─► V1 results deck
```

### 5.3 Tasks

#### Member A — Data, Split, EDA, V1 Deck
**Prerequisites:** P-CPU; phone-verified Kaggle account; internet; about 10 GB free disk.

1. **Acquire HAM10000**: both image parts plus `HAM10000_metadata.csv`, from Kaggle (`kmader/skin-cancer-mnist-ham10000`) or the Harvard Dataverse record. In `data/README.md`, record the source, download date, file sizes and checksums.
2. **Integrity audit.** Confirm:
   - exactly 10,015 unique `image_id`s, each with an image file;
   - 7,470 unique `lesion_id`s, each with exactly one `dx`;
   - class counts nv 6705, mel 1113, bkl 1099, bcc 514, akiec 327, vasc 142, df 115;
   - all images are 600×450.
   Record any anomaly.
3. **Label map (C1):** alphabetical, built from the full dataset before splitting, exactly as in §1.
4. **Split (C2), following the paper's method:**
   - reduce to a one-row-per-lesion table;
   - split lesions 70% / 30%, stratified by `dx`, seed 42;
   - split the 30% into two halves (15% / 15%), stratified by `dx`, seed 42;
   - map lesions back to all their images.
   Write this two-step procedure into `deviations.md` as our interpretation of the paper's one-line description.
5. **Split verification report.** Confirm:
   - zero lesion overlap between splits;
   - zero image overlap;
   - the union of splits is all 10,015 images;
   - **all 7 classes appear in every split.**
   Produce a per-split class-count and class-proportion table. Also produce an "Ours vs Paper" count table (paper: 7024 / 1497 / 1494 images) and note that the paper's lesion counts don't add up to 7,470 (see §13).
6. **Publish the split** to shared storage **by Day 1, about Hour 3–4**, and announce it. From that point on it is frozen.
7. **EDA figures:**
   - class distribution bar chart (counts and %);
   - histogram of images per lesion;
   - sample grid with 4 images per class;
   - dataset summary table (paper Table 1 equivalent).
8. **Optional speed-up:** a raw 224×224 resized cache (plain resize, no crop) so B's data loading is faster. B must be able to train without it; this is purely an optimization.
9. **Architecture figure** equivalent to paper Figure 1: a block diagram of B0 plus the dropout head, drawn with any drawing tool.
10. **Day 2:** assemble the **V1 results deck (6–8 slides)**. A owns the dataset/split/architecture slides and drops in C's result figures. Also prepare a static PDF backup of every figure.

**Outputs:** `data/README.md`, the C1 label map, C2 split files, split verification report, EDA figures, architecture figure, V1 deck.
**Done when:** the split files pass every verification check, have been published, and B and C have confirmed they can load them.

#### Member B — Model, Training, Predictions
**Prerequisites:** **P-GPU-S** (most reliable GPU on the team); persistent checkpoint storage; the environment pinned on Day 1.

1. **Environment:** Python 3.10+, a current PyTorch/torchvision that provides `EfficientNet_B0_Weights.IMAGENET1K_V1`, scikit-learn, pandas, NumPy, Pillow, OpenCV, matplotlib. Pin the versions in `env/` and share them. Everyone uses the same spec.
2. **Data pipeline, per paper §3.2.2–3.2.4:**
   - read the image and resize to 224×224;
   - training only: horizontal flip p=0.5, vertical flip p=0.5, rotation ±15°, colour jitter (0.2 / 0.2 / 0.2 / 0.1);
   - convert to tensor and apply ImageNet normalization;
   - validation and test: resize and normalize only.
3. **Model, per §3.3:** torchvision EfficientNet-B0 with ImageNet weights. Keep the convolutional feature stack and global average pooling. **Replace the whole original classifier** (including its built-in dropout) with Linear 1280→256 → ReLU → Dropout 0.3 → Linear 256→7. All parameters are trainable.
4. **Dual imbalance correction, per §3.2.5:**
   - WeightedRandomSampler: weight 1/n_c per sample, using **train-split** counts, sampling with replacement, number of samples equal to the train size (the sampler replaces shuffling);
   - cross-entropy class weights (1/n_c)/Σ(1/n_k), also from train counts.
5. **Optimization:** 30 epochs, no early stopping, cosine annealing over the 30 epochs. Optimizer, learning rate, batch size and weight decay come from paper Table 2 **(TBD)**. Until then use **provisional defaults**: AdamW, LR 1e-4, weight decay 1e-4, batch 32, cosine to about 0. Record them as provisional in `deviations.md`.
6. **Per-epoch logging (C5):** train loss, val loss (same weighted criterion), val macro F1, val macro AUROC, learning rate.
7. **Checkpointing (C6):** save the last checkpoint every epoch plus the best by **validation macro F1**, to persistent storage.
8. **Smoke test before the split lands:** 2 epochs on a random 500-image subset. Confirm the loss decreases, and measure GPU memory and time per epoch. Throw this run away.
9. **Full V1 run** (`v1-b0-baseline`) on the frozen split, started as soon as C2 is published on Day 1. Expected about 45–90 min on a T4.
10. **Export (C3):** single-pass predictions on val and test with the model in eval mode (dropout **off**); the history file (C5); the best-checkpoint metadata (C6); the run config. Add a row to the run registry. **Hand off by Day 2, about 10:00.**

**Outputs:** best checkpoint, C3 val/test prediction files, C5 history, config, run-registry entry.
**Done when:**
- the history file has 30 rows;
- prediction row counts equal the split sizes;
- probabilities sum to 1;
- all 7 classes appear among the predictions.

#### Member C — Evaluation Suite and Figures
**Prerequisites:** P-CPU only.

1. **Dummy data:** generate a fake C3 predictions file with realistic class frequencies and random valid probabilities, and a fake 30-epoch C5 history. Build everything below against these before real results exist.
2. **Metric suite, per §3.7:**
   - macro AUROC, one-vs-rest (principal metric);
   - macro F1;
   - per class: accuracy exactly as Eq. 17 (TP+TN)/(all), sensitivity/recall, specificity, precision, support;
   - overall accuracy, for context only;
   - confusion matrix as counts and row-normalized.
   **Report per-class accuracy both as Eq. 17 and as recall** (see §13, item 4).
3. **Bootstrap CIs, per §3.7.6:** 1000 resamples of the test set, percentile 95% interval, seed 42, for macro AUROC and macro F1. If a resample is missing a class, skip it and report how many were skipped.
4. **Figures:**
   - training curves: train/val loss, and val macro F1 / AUROC, with the best epoch marked (paper Fig. 2 style);
   - 7×7 confusion matrix as counts and normalized (Fig. 5 style);
   - per-class accuracy bars with a 70% reference line (Fig. 4 style);
   - per-class sensitivity and specificity grouped bars;
   - one-vs-rest ROC curves for the 7 classes plus the macro average.
5. **Tables:**
   - headline metrics with CI, next to a paper column;
   - per-class table in the paper's Table 5 format;
   - training milestones table (paper Table 3 style): epochs 1, 5, 10, 15, 20, 25, 30, and the best epoch.
6. **Verify the suite:** compare each metric against a reference implementation on dummy data, and against a hand-computed 3-class toy example.
7. **Day 2:** run the suite on B's real files and write `results/v1/` (figures, `metrics.json`, `summary.md`). Hand the figures to A for the deck.

**Outputs:** `src/eval/` suite, `results/v1/` figures, tables and metrics.
**Done when:** every figure and table is regenerated from real predictions with one command, and the numbers pass the sanity checks in §5.6.

### 5.4 Intra-version dependencies (V1)
- **B needs C2 from A** by about Day 1 Hour 3–4. The gap is covered by B's smoke test, so B is never blocked.
- **C needs C3 and C5 from B** on Day 2 morning. The gap is covered by the dummy-first rule, so C is never blocked.
- **A needs C's figures** on Day 2 midday for the deck. A fills the dataset and split slides first.

### 5.5 Two-day schedule

| Time | A | B | C |
|---|---|---|---|
| **Day 1, H0–H1** | Kickoff for all: agree contracts C1–C3, C5, C6; create repo and shared storage; Kaggle phone verification; skim paper §3 together | (same) | (same) |
| **Day 1, H1–H4** | Download, audit, split, verification → **publish C2** | Environment, data pipeline, model, smoke test | Dummy data; metric suite |
| **Day 1, H4–H8** | EDA figures, split report, optional cache, architecture figure | **Start full 30-epoch run**; monitor; check the val curve | Bootstrap, all figures and tables on dummy data; verification |
| **Day 1, evening / overnight** | — | Run finishes; export C3/C5/C6 | — |
| **Day 2, AM** | Deck skeleton with dataset/split slides | **Hand off artifacts (by 10:00)**; update run registry | Run the suite on real outputs → `results/v1/` |
| **Day 2, midday** | Integration review for all: sanity checks (§5.6) | (same) | (same) |
| **Day 2, PM** | Final deck + PDF backup; rehearsal with all | Prepare to explain training setup | Prepare to explain metrics |

### 5.6 V1 result package (what the professor sees)

1. Dataset overview: class distribution, 58:1 imbalance by counts, lesion multiplicity, sample grid.
2. Split table (ours vs paper) and the leakage check ("0 shared lesions").
3. Architecture figure.
4. Training curves.
5. **Headline test metrics with 95% CI:** macro AUROC and macro F1.
6. Per-class table: accuracy (Eq. 17), recall, specificity; per-class bar chart.
7. Confusion matrix.
8. ROC curves.
9. "Ours vs Paper" slide, clearly labelled: paper = preprocessing + T=50; V1 = no preprocessing, single pass.
10. Roadmap slide: V2–V5.

**Sanity checks before showing anything:**
- **Macro AUROC should sit well above the paper's unmitigated baseline (0.80) and in the region of its final 0.94.** If it's below about 0.80, check label mapping, normalization, and whether the sampler and loss weights used train counts.
- All 7 classes are predicted at least once.
- The confusion matrix doesn't collapse into the nv column.
- Validation macro F1 trends upward over epochs.

### 5.7 Risks and fallbacks (V1)

| Risk | Fallback |
|---|---|
| Kaggle/Colab session drops mid-run | Checkpoint every epoch; resume from the last one |
| Run not finished by Day 2 09:00 | Present the best checkpoint so far, labelled "N/30 epochs" |
| GPU quota exhausted | Switch to another member's Kaggle account with the same config (that's why everyone verifies on Day 1) |
| Slow data loading | Use A's 224×224 raw cache |
| Metrics look implausible | Run the §5.6 checks in order; the label map is the most common cause |

### 5.8 Prep-ahead for V2 (after the demo, A only)
Build the **preprocessed cache (C9)** described in V2 below, so every V2 training run can start immediately.

---

## 6. V2 — Paper-Faithful Training + Imbalance Ablation

### 6.1 Goal and scope
Add the paper's artifact-removal preprocessing and final hyperparameters, produce the **locked main model** (`v2-main-b0`), and reproduce the **imbalance ablation (Table 8)**. Evaluation is still single-pass here; the paper's T=50 headline numbers come in V3 from this same checkpoint.

### 6.2 Pipeline structure (V2)

```
raw 600×450 ─► hair removal (blackhat 17×17, thr 10, TELEA r=3) ─► CLAHE (LAB, each channel, 2.0, 8×8)
           ─► resize 224 ─► lossless cache (C9)
                              │
          ┌───────────────────┼─────────────────────┬──────────────────────┐
          ▼                   ▼                     ▼                      ▼
   (iv) MAIN: sampler   (i) no correction   (ii) sampler only    (iii) loss-weights only
        + loss weights      [A]                  [C]                    [C]
        [B]
          │              (v) optional: no ImageNet pretraining [B], if Table 8 lists it
          ▼
   all runs ─► single-pass predictions (C3) ─► V1 eval suite ─► Table 8 + main-model figures
```

All runs share the same split, cache, augmentation, hyperparameters, seed and 30 epochs. **Only the toggled factor differs.**

### 6.3 Tasks

#### Member A — Preprocessing cache + ablation run (i)
**Prerequisites:** P-CPU-MC for the cache (done during the prep-ahead window); **P-GPU-S** for one run of about 1–1.5 h.

1. **Preprocessing (C9), exactly per §3.2.1**, applied to the **original full-resolution** image *before* resizing:
   - grayscale conversion → blackhat morphological transform with a 17×17 disk (elliptical) structuring element → binary threshold at 10 → TELEA inpainting of the colour image with radius 3;
   - convert to LAB → CLAHE with clip 2.0 and tile 8×8 on **each of the L, a, b channels independently** (as the paper states) → convert back to RGB;
   - resize to 224×224 and store **losslessly** (PNG), so JPEG re-compression artifacts aren't introduced.
   Apply identically to train, val and test.
2. **Manifest:** per-image hair-pixel fraction and processing time; total runtime.
3. **Quality control:**
   - before/after grid, weighted toward hairy images;
   - hair-mask overlays for about 20 images;
   - histogram of hair-pixel fraction;
   - manual spot-check of 50 random images;
   - confirmation that the geometry is unchanged, which V4 mask alignment depends on.
4. **Ablation run (i), "no correction":** plain shuffling with unweighted cross-entropy; everything else identical to the main run. This reproduces the paper's "pre-fix baseline" (AUROC 0.8038, F1 0.36, mel sensitivity 30.5%, df sensitivity 0%). Export C3/C5, and evaluate with C's suite (read-only use).

**Done when:** the cache covers all 10,015 images, QC figures are in `results/v2/`, and run (i) is evaluated and registered.

#### Member B — Main run (iv) + model lock
**Prerequisites:** **P-GPU-S** (most reliable); about 1.5 h per run.

1. **Hyperparameters:** replace the provisional V1 values with **paper Table 2** values if the PDF is available. Otherwise keep the provisional values and note that in `deviations.md`.
2. **Main run `v2-main-b0`:** the full paper recipe on the C9 cache, with sampler **and** class-weighted loss, 30 epochs, cosine LR, best by validation macro F1.
3. **Export:** C3 val/test predictions, C5 history, C6 metadata.
4. **Lock the model:** tag `v2-main-b0` as **the** model for V3 and V4, and record its file hash in the registry. It must not be retrained afterwards; later versions depend on exactly this checkpoint.
5. **Optional run (v), "no ImageNet pretraining"** (random initialization, all else identical), **only if paper Table 8 contains this row.** The text credits pretraining with +6–8% AUROC, but it's unclear whether Table 8 includes it.

**Done when:** `v2-main-b0` is locked and registered, and its predictions and history have been handed to C.

#### Member C — Ablation runs (ii) and (iii) + ablation analysis
**Prerequisites:** **P-GPU-S** for two sequential runs (about 2–3 h total); P-CPU for the analysis.

1. **Run (ii), "sampler only":** WeightedRandomSampler with unweighted cross-entropy.
2. **Run (iii), "loss only":** shuffled loader with class-weighted cross-entropy.
3. **Evaluate every V2 run** — (i), (ii), (iii), (iv) and (v) if present — with the suite.
4. **Table 8 reproduction.** Rows are the runs; columns are:
   - macro AUROC [CI] and macro F1 [CI];
   - melanoma sensitivity, dermatofibroma sensitivity, nv accuracy;
   - Δ against (iv).
   Compare our Δs to the paper's (+1.8% / +1.0% AUROC and +7.0% / +10.9% F1 for "both" against sampler-only and loss-only). Also produce an ablation bar chart.
5. **Main-model single-pass package:**
   - Figure 2 (training curves) and Table 3 (milestones) for `v2-main-b0`;
   - per-class Table 5 and Figure 4;
   - confusion matrix (Figure 5).
   Label all of these "single pass"; V3 updates them to T=50.
6. **Internal consistency check:** compare `v1-b0-baseline` with `v2-main-b0`. They differ only in preprocessing and final hyperparameters. This is reported as a check on our pipeline, not as a new result.

**Done when:** `results/v2/` contains Table 8, the ablation chart, and the main-model figures and tables.

### 6.4 Intra-version dependencies (V2)
- **Every run needs the C9 cache.** It's built during A's prep-ahead window after V1, so V2 starts with it ready.
- The runs are otherwise fully independent and can happen in parallel on three separate GPU sessions.
- C's Table 8 assembly waits for all run outputs, but C builds the table layout against V1-format dummy files first.

### 6.5 V2 result package
- Preprocessing before/after figure and hair statistics.
- Hyperparameter table (our Table 2).
- `v2-main-b0` training curves and milestone table.
- Single-pass test metrics with CI.
- Per-class table and chart, and confusion matrix.
- **Table 8 ablation with "Ours vs Paper" deltas.**
- Updated `deviations.md`.

**Sanity checks:**
- Run (i) should show a clear minority-class collapse, like the paper's (df sensitivity near 0).
- Run (iv) should beat (ii) and (iii) on macro F1.
- Train loss should drop sharply in the first 5 epochs (paper: 0.87 → 0.15).

### 6.6 Risks and fallbacks (V2)

| Risk | Fallback |
|---|---|
| Preprocessing slower than expected | Parallelize across cores, or split the image list between two machines and merge the manifests |
| CLAHE on the a/b channels produces odd colours | Keep it anyway — it's faithful to the paper — and show the QC grid in the report |
| Not enough GPU quota for 4–5 runs | Prioritize (iv) → (i) → (ii) → (iii) → (v) |

---

## 7. V3 — MC Dropout Uncertainty + Entropy-Based Referral

### 7.1 Goal and scope
Using the locked `v2-main-b0`, reproduce:
- the paper's **T=50 headline results** (Table 4);
- the **uncertainty decomposition**;
- **calibration** (ECE with T=1 vs T=50, reliability diagram, Brier score);
- the **entropy distribution** (Figure 6);
- the **referral tradeoff** (Table 6, Figure 7).

### 7.2 Pipeline structure (V3)

```
v2-main-b0 (locked) ─► eval mode ─► re-enable ONLY nn.Dropout (head) ─► 50 passes on val & test (C4)
                                   └► deterministic single pass (dropout off) (C4, "T=1")
                                                   │
                     ┌─────────────────────────────┴──────────────────────────────┐
                     ▼                                                            ▼
   uncertainty & calibration [A]                                  referral analysis [C]
   entropy / aleatoric / epistemic,                                τ = (1−r)-quantile of test entropy,
   ECE (T=1 vs T=50), Brier, reliability diagram,                  accepted-subset AUROC/F1 + CIs,
   entropy histogram, Table 4                                      Table 6, Fig. 7, decisions (C8)
```

### 7.3 Tasks

#### Member B — MC Dropout inference engine
**Prerequisites:** P-GPU-INF (a CPU works with the feature-caching approach below).

1. **Mode handling, per §3.4.2:** put the whole model in eval mode, then switch **only the Dropout layers** back to stochastic mode. BatchNorm stays frozen, and so do EfficientNet-B0's internal *stochastic-depth* layers. They are not Dropout layers and must stay deterministic.
2. **Passes:** T=50 stochastic passes over validation and test; store the per-pass softmax probabilities (T×N×7) with the image index and labels (C4). Record the seed.
3. **"T=1" reference:** one deterministic pass with dropout off, also stored in C4. Record this interpretation in `deviations.md` (see §13, item 8).
4. **Allowed speed-up (mathematically identical):** the only stochastic layer is in the head, so backbone features can be computed once per image and the head sampled 50 times. Before relying on this, verify on 32 images that it matches full forward passes.
5. **Validation checks:**
   - passes differ from each other (non-zero variance);
   - the 50-pass mean is close to the deterministic probabilities;
   - row counts and order match the split files.

**Done when:** C4 files for val and test are delivered and validated.

#### Member A — Uncertainty decomposition and calibration
**Prerequisites:** P-CPU. Start against a dummy C4 file.

1. **Per image, per Eqs. 7–10:** mean prediction p̄; total entropy H[p̄] in nats; aleatoric = mean of per-pass entropies; epistemic = total − aleatoric. Assert that epistemic ≥ 0, up to numerical tolerance.
2. **Summary statistics:** means of total, aleatoric and epistemic uncertainty (paper: 0.3209 total, 0.0113 epistemic nats), and their distributions.
3. **Calibration, per Eqs. 15–16:**
   - ECE with 10 equal-width bins on the max-probability confidence of p̄, **for T=50 and for T=1**, each with bootstrap CI (paper: 0.1456 vs 0.1891);
   - Brier score.
4. **Figures:**
   - **Figure 3** reliability diagram: per-bin points against the diagonal, plus a bin-count histogram;
   - **Figure 6** entropy histogram split into correct vs incorrect predictions (paper: correct predictions concentrated near 0 nats, incorrect ones spread over 0–1.4).
5. **Table 4 (final headline):** macro AUROC [CI], macro F1 [CI], ECE [CI], Brier, and mean entropies, all computed from p̄. Also regenerate Table 5, Figure 4 and Figure 5 from p̄ using C's suite.

**Done when:** `results/v3/` contains Table 4, the ECE comparison, Figures 3 and 6, and the T=50 per-class package.

#### Member C — Referral mechanism and tradeoff
**Prerequisites:** P-CPU. Start against a dummy C4 file.

1. **Thresholds, per §3.4.4:** for each r ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}, τ = the (1−r)-quantile of the **test-set** total entropy (as the paper does). Accept a prediction if H ≤ τ.
2. **Accepted-subset metrics:** n accepted, macro AUROC and macro F1, each with 1000-resample bootstrap CIs drawn within the accepted subset.
3. **Table 6** plus **Figure 7** (curve with a shaded 95% CI band).
4. **Checks:**
   - is the curve monotonic?
   - does the r=0.30 CI overlap the r=0 CI? (The paper reports no overlap.)
   - % improvements at r=10% and r=30%, compared with the paper's +0.77% / +2.57% and +1.64% / +7.91%.
5. **Referral decisions file (C8)** for every r. V4 needs it for the accepted-vs-referred IoU comparison. The r the paper uses for that split is TBD from Table 10; default to r=0.30 and record the choice.

**Done when:** `results/v3/` contains Table 6, Figure 7, and the C8 file.

### 7.4 Intra-version dependencies (V3)
- A and C both consume B's C4, and both develop against dummy C4 files until it arrives. B's task is short (well under a day), so the wait is minimal.
- A and C are independent of each other.

### 7.5 V3 result package
- **Table 4 (T=50) with CIs.**
- ECE for T=1 vs T=50.
- Figure 3 reliability diagram.
- Figure 6 entropy histogram.
- **Table 6 + Figure 7 referral tradeoff.**
- Updated Table 5, Figure 4 and Figure 5 at T=50.
- Uncertainty summary statistics.
- "Ours vs Paper" for all of the above.

**Sanity checks:**
- Epistemic uncertainty should be much smaller than aleatoric (paper: 0.011 vs 0.32).
- ECE at T=50 should be at or below ECE at T=1.
- Accepted-subset AUROC should rise as r increases.

---

## 8. V4 — Grad-CAM Explainability + Quantitative IoU

### 8.1 Goal and scope
Reproduce the Grad-CAM analysis in §3.5, **Table 10** (IoU against ground-truth lesion masks, accepted vs referred) and **Figures 8–11**.

### 8.2 Pipeline structure (V4)

```
GT lesion masks ─► align to 224×224 (same geometry as images) ─► C10          [A]
v2-main-b0 ─► Grad-CAM @ features[-1] (1280×7×7) ─► bilinear ↑224 ─► threshold 50% of max
           ─► IoU vs C10 mask ─► C7                                           [B]
C7 + C8 (V3 referral) + entropies ─► Table 10, Figs 8–11                      [C]
```

### 8.3 Tasks

#### Member A — Ground-truth masks
**Prerequisites:** P-CPU; about 1–2 GB disk; internet.

1. **Source:** the HAM10000 Harvard Dataverse record includes a **lesion segmentation mask archive** for HAM10000 (the file name contains "segmentations_lesion"). Download it and verify that it covers the test split; the paper implies all 1,494 test images have masks.
   **Fallback:** the ISIC 2018 Task 1 masks. Most of those are *not* HAM10000 image IDs, so coverage would be low; report it as a deviation.
2. **Alignment:** give each mask exactly the same geometric transform as its image (resize to 224×224 with nearest-neighbour, then binarize). Hair removal and CLAHE don't change geometry (verified in V2).
3. **QC:** mask-outline overlays on preprocessed images for 30 random test images, and a coverage report (masks found / test images).
4. **Publish C10** (masks plus manifest).

#### Member B — Grad-CAM and IoU engine
**Prerequisites:** P-GPU-INF recommended; CPU is fine (about 15–30 min for the test set).

1. **Grad-CAM, per Eq. 11:**
   - hooks on `model.features[-1]`;
   - model in eval mode (dropout off);
   - target class = **predicted class** (the paper doesn't say; this is the standard convention, logged in `deviations.md`);
   - channel weights = spatial mean of the gradients; ReLU of the weighted sum; bilinear upsampling to 224×224; normalization to [0, 1].
2. **Binarization:** threshold at **50% of each map's own maximum** (§3.5).
3. **IoU, per Eq. 12,** against the C10 mask. Test the IoU logic on synthetic masks first: identical masks give 1, disjoint masks give 0.
4. **Output C7:** per test image, the predicted class, the target class, the IoU, and the stored CAM.

#### Member C — Explainability analysis and figures
**Prerequisites:** P-CPU. Start against dummy C7 files.

1. **Join** C7 with the V3 referral decisions (C8) and entropies.
2. **Table 10:** mean ± SD IoU overall, **accepted vs referred** (paper: 0.61 vs 0.41), and per class.
3. **Figures, with a documented example-selection rule** (e.g. highest-confidence correct example per class, seeded random for ties) so nothing is cherry-picked:
   - **Figure 8:** one example per class, original image and CAM overlay, with entropy and ACCEPT/REFER label. Border colours follow the paper: red for malignant (mel, bcc), yellow for pre-cancerous (akiec), green for benign (bkl, df, nv, vasc).
   - **Figure 9:** correct vs incorrect predictions for MEL, BCC, AKIEC and NV.
   - **Figure 10:** per-class grid, 7 rows of original plus overlay, with entropy.
   - **Figure 11:** examples ordered by increasing entropy.

### 8.4 Intra-version dependencies (V4)
- B's IoU needs A's C10 masks. B builds and tests the Grad-CAM engine first, and A's task is short, so the wait is small.
- C develops against dummy C7 files; the C8 file already exists from V3.

### 8.5 V4 result package
- **Table 10 (IoU overall / accepted / referred / per class) against the paper.**
- Figures 8–11.
- Mask coverage report.
- Deviations: CAM target class, and the mask source if the fallback was used.

**Sanity checks:**
- Accepted-prediction IoU should exceed referred-prediction IoU.
- CAMs for confident correct predictions should visibly sit on the lesion.

---

## 9. V5 — Comparative Baselines + Final Replication Report

### 9.1 Goal and scope
Reproduce **Table 7**: MobileNetV3, ResNet-50 and DenseNet-121 trained under the **same preprocessing, split and evaluation protocol**, compared on macro AUROC, macro F1 and parameter count. Then compile the **final replication report**. Per the paper, the ISIC 2018 winner row is quoted from literature for context only and is not trained.

### 9.2 Pipeline structure (V5)

```
C9 cache + frozen split + V2 recipe (same aug, dual correction, 30 ep, cosine, best val macro-F1)
   ├─► MobileNetV3  [A]  ─┐
   ├─► ResNet-50    [B]  ─┼─► single-pass predictions (C3) ─► eval suite ─► Table 7
   └─► DenseNet-121 [C]  ─┘                                     + params count
                                                        ─► Final replication report [C, sections from A & B]
```

### 9.3 Tasks

**Shared protocol for all three runs:**
- ImageNet-pretrained torchvision weights;
- identical recipe to `v2-main-b0`;
- the same 256-unit dropout head placed on each backbone's pooled features, so only the backbone differs (confirm against Table 7 when the PDF is available);
- single-pass evaluation (the paper says the baselines give no uncertainty estimates);
- report total and trainable parameter counts.

#### Member A — MobileNetV3 + report sections
**Prerequisites:** **P-GPU-S**; about 1 h.
1. Train `v5-mobilenetv3`. The variant (Large vs Small) is TBD from Table 7; default to **Large** and log the choice.
2. Evaluate with the suite and count parameters.
3. Write the final-report sections on **data, split, preprocessing and masks**.

#### Member B — ResNet-50 + report sections
**Prerequisites:** **P-GPU-L** (≥12 GB VRAM); about 1.5–2.5 h.
1. Train `v5-resnet50`, then evaluate and count parameters.
2. Write the final-report sections on **model, training, MC Dropout and Grad-CAM**.

#### Member C — DenseNet-121 + Table 7 + final report assembly
**Prerequisites:** **P-GPU-L** (≥12 GB VRAM); about 2–3 h; then P-CPU.
1. Train `v5-densenet121`, then evaluate and count parameters.
2. **Table 7:** method, parameter count, macro AUROC [CI], macro F1 [CI], and whether it provides uncertainty. Include our B0 (from V3 and V2) and the literature ISIC 2018 row marked "different protocol, context only". Check the paper's claim that B0 is competitive with ResNet-50 and DenseNet-121 at about 1/5 and 2/3 of their parameter counts.
3. **Final replication report,** assembled from all sections:
   - "Paper vs Ours" for Tables 1, 3–8 and 10 and Figures 1–11 (Table 9 excluded by decision);
   - the full `deviations.md`;
   - the paper inconsistencies we found (§13);
   - limitations.

**If a member has less than 12 GB VRAM:** use mixed precision, or reduce the batch size and log it as a deviation. Better, swap runs with a member who has a 16 GB session. Kaggle's free T4/P100 have 16 GB, so anyone with a Kaggle account qualifies.

### 9.4 Intra-version dependencies (V5)
- None between members for the training runs.
- C's Table 7 and report assembly wait for the other two runs and sections; C builds the templates first.

### 9.5 V5 result package
- **Table 7** plus a parameter-count vs AUROC chart.
- The **final replication report**, which covers all tables and figures from V1–V4 in consolidated form.
- The complete run registry and deviation log.

---

## 10. Cross-version artifact flow

| Artifact | Created in | Used in |
|---|---|---|
| C1 label map, C2 frozen split | V1 (A) | V2–V5 |
| Evaluation suite | V1 (C) | V2–V5 |
| C9 preprocessed cache | V1 prep-ahead (A) | V2, V3, V4, V5 |
| `v2-main-b0` locked checkpoint | V2 (B) | V3, V4, V5 (as our B0 row) |
| C4 MC predictions, entropies | V3 (B, A) | V3, V4 |
| C8 referral decisions | V3 (C) | V4 |
| C10 aligned masks | V4 (A) | V4 |
| Recipe config | V2 (B) | V5 |

---

## 11. Per-version prerequisites at a glance

| Version | A needs | B needs | C needs |
|---|---|---|---|
| V1 | P-CPU, Kaggle, ~10 GB disk | **P-GPU-S** (most reliable) | P-CPU |
| V2 | P-CPU-MC (cache) + P-GPU-S (1 run) | **P-GPU-S** (1–2 runs) | P-GPU-S (2 runs) + P-CPU |
| V3 | P-CPU | P-GPU-INF (or CPU) | P-CPU |
| V4 | P-CPU, ~2 GB disk | P-GPU-INF (or CPU) | P-CPU |
| V5 | P-GPU-S (MobileNetV3) | **P-GPU-L** (ResNet-50) | **P-GPU-L** (DenseNet-121) + P-CPU |

---

## 12. Definition of done (every version)

- [ ] Every artifact in the version's result package exists in `results/vN/` and regenerates from stored predictions with one command.
- [ ] Every run is in `docs/run_registry.md`, and every departure from the paper is in `docs/deviations.md`.
- [ ] An "Ours vs Paper" table exists for every paper number covered by the version.
- [ ] The version's sanity checks pass, or failures are documented with an explanation.
- [ ] A short results deck or summary page for the professor is ready.

---

## 13. Paper inconsistencies found while reading (and how we handle them)

1. **Split counts don't add up.** The paper reports 1,891 / 405 / 405 unique lesions (2,701 total), but HAM10000 has about 7,470 unique lesion IDs. Its image counts (7024 / 1497 / 1494) are plausible for a 70/15/15 split. → **We follow the stated method and report our own counts next to theirs.**
2. **Imbalance ratio.** The paper calls it 67:1 (nv vs df). By class counts it is 6705 / 115 ≈ 58:1; "67" looks like nv's share (≈67%). → **We report the ratio computed from counts.**
3. **Figure 4 caption vs text.** The caption says all classes exceed 70%; the text says six of seven, with nv at 69.8%. → **We report our own numbers.**
4. **The per-class "accuracy" definition is ambiguous.** Eq. 17 is one-vs-rest accuracy, (TP+TN)/all. For the majority class, a value of 69.8% looks more like recall than one-vs-rest accuracy. → **We report both, and state which one matches the paper's values.**
5. **The referral threshold is taken from the test set.** τ is a quantile of test-set entropy. → **We replicate this exactly as stated and mention it as a methodological note in the final report.**
6. **Cross-reference slips.** §3.2.1 cites "Section 5" for Grad-CAM evidence; the Figure 8/9 captions cite "Section 3.1" for hair removal (it is §3.2.1). These are harmless.
7. **Mask coverage.** The paper implies ISIC masks for all 1,494 test images. → **A verifies coverage in V4 and reports the actual count.**
8. **"T=1" ECE is ambiguous.** It could mean one stochastic pass or a deterministic pass. → **We use the deterministic pass (dropout off) and log it.**
9. **Grad-CAM target class isn't stated.** → **We use the predicted class and log it.**

---

## 14. Open items that need the paper PDF (Tables were not readable in the text version)

| Item | Needed by | Current handling |
|---|---|---|
| **Table 2**: optimizer, learning rate, batch size, weight decay, cosine schedule parameters | V1 (provisional), V2 (final) | Provisional: AdamW, LR 1e-4, WD 1e-4, batch 32, cosine over 30 epochs |
| Table 1 exact class table | V1 | Build from our own counts |
| Table 3 milestone epochs and values | V2 | Report epochs 1, 5, 10, …, 30 and the best epoch |
| Table 5 per-class paper values | V2, V3 | Use the values quoted in the text (mel 80.8% / 96.2%, nv 69.8%) |
| Table 6 full rows | V3 | Use the values quoted in the text (r = 0, 10%, 30%) |
| Table 7: MobileNetV3 variant, head used for the baselines, numbers | V5 | MobileNetV3-Large + the same 256-unit head |
| Table 8 rows (is "no pretraining" included?) | V2 | Run (v) is optional |
| Table 10 values and the r used for accepted/referred | V4 | r = 0.30 |

**Action:** upload the PDF (or paste Tables 2, 7, 8 and 10) and this plan can be updated so every TBD becomes an exact value.

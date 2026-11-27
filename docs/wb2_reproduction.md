# wb2_reproduction.md — reproducing the Workbook 2 experiments

Owners: Member A (sampling: this section), Member C (controller and analysis: separate section).
The status labels used below are **MEASURED**, **PILOT ONLY**, **PLANNED**, **NOT RUN** and **FAILED**.

## A. Stochastic sampling (Member A, `src/wb2/sampling/`)

### Inputs

| Input | Source | Identity |
|---|---|---|
| Checkpoint | Sai's V1 `best.pt` (handoff: `docs/wb2_checkpoint_handoff.md` on `wb2-sai-handoff`), placed at `runs/v1-b0-baseline/best.pt` (git-ignored) | SHA256 `901a73a6a689cfb4617d3897152fcb33c73d40bbbf190978218fff4614f9ae0d`, epoch 30 |
| Splits | `data/splits/{val,test}.csv` (frozen C2, unchanged) | 1,532 val / 1,481 test images |
| Images | Raw HAM10000 JPEGs from Dataverse `doi:10.7910/DVN/DBW86T`, val + test only | `data/raw/images/` (git-ignored); zip hashes in `data/raw/wb2_fetch_record.json` |
| Transform | `src.model.dataset.build_eval_transform`: Resize 224×224 → ToTensor → ImageNet normalize | Same as V1. No hair removal and no CLAHE. |
| Config | `configs/wb2_sampling.yaml` | run_id `wb2-mc-v1b0-s42`, seed 42, 50 passes |

### Steps

```bash
python3 -m venv .venv && .venv/bin/pip install -r env/requirements.txt
.venv/bin/python -m src.wb2.sampling.fetch_images --splits val test      # ~2.8 GB download, keeps only 3,013 images
.venv/bin/python -m src.wb2.sampling.bank --config configs/wb2_sampling.yaml --split val
.venv/bin/python -m src.wb2.sampling.e0_check --bank-dir results/wb2/inputs/wb2-mc-v1b0-s42 --split val
.venv/bin/python -m src.wb2.sampling.calibrate --bank-dir results/wb2/inputs/wb2-mc-v1b0-s42   # val only
.venv/bin/python -m src.wb2.sampling.bank --config configs/wb2_sampling.yaml --split test
.venv/bin/python -m src.wb2.sampling.timing --config configs/wb2_sampling.yaml --split val --n-images 50
.venv/bin/python -m src.wb2.sampling.fixed_budget --bank-dir results/wb2/inputs/wb2-mc-v1b0-s42 --split val
.venv/bin/python -m unittest discover -s tests -t .
```

`bank` is resumable. Shards are cached under `runs/wb2-mc-v1b0-s42/shards/`, so re-running the same command continues an interrupted run and produces identical draws. Each bank is validated against its split file before the command reports success.

### MC-Dropout semantics

- The model is put in `model.eval()`, then `model.classifier[2].train()` is applied. Only the head `Dropout(0.3)` is stochastic; the manifest records `stochastic_modules: ["classifier.2"]`. `model.train()` is never called, so StochasticDepth stays off and BatchNorm statistics stay frozen. Unit tests check both.
- Backbone features are reused (`feature_reuse: true`): each image gets **1 backbone pass + 50 head passes**. The full-network path (`feature_reuse: false`) produces identical logits under the same seed (unit test). Only the timing study uses it.
- Seed policy: `torch.manual_seed(42 + shard_index)` at the start of each 256-image shard. The draws also depend on batch size (64) and device; both are recorded in the manifest.
- T=10 and T=20 are the first 10 and 20 passes of the same 50-pass bank.
- The deterministic T=1 logits come from `model.eval()` alone, stored in a separate file. They are not a stochastic draw.

### Outputs (`results/wb2/inputs/wb2-mc-v1b0-s42/`)

| File | Contract | Status |
|---|---|---|
| `val_mc_logits.npz`, `val_det_logits.npz`, `val_ids.csv`, `val_manifest.json` | WB2-I1 / I3 | **MEASURED**, COMPLETE, 1,532 / 1,532 |
| `test_*` (same four files) | WB2-I2 / I3 | **MEASURED**, COMPLETE, 1,481 / 1,481 |
| `calibration.json` | WB2-I4: scalar T fitted on val NLL, applied to per-pass logits before softmax, 50-pass mean | **MEASURED** |
| `timing.json` | WB2-I4: p50/p95 at batch 1, throughput at batch 50; backbone and head passes counted separately | **MEASURED** |
| `e0_{val,test}.json` | E0: deterministic logits vs committed V1 predictions | **MEASURED** (outside the 1e-3 tolerance, see below) |
| `fixed_budget_{val,test}.csv` | Member A cross-check of T=1/10/20/50 (Sabari owns the final tables) | **MEASURED** |

Logit arrays are float32, about 2 MB per split (3.9 MB in total, committed). `logits` holds raw pre-softmax values with class order `akiec, bcc, bkl, df, mel, nv, vasc`.

### Run record (2026-10-27)

All runs used the checkpoint with SHA256 `901a73a6…f9ae0d` (verified on arrival) on an Apple M2 (MPS, macOS 15.5) with torch 2.8.0 and torchvision 0.23.0. Timestamps are in UTC. The val bank ran 23:18:30–23:18:52; calibration was fitted at 23:19:18, **before** the test bank existed; the test bank ran 23:19:20–23:19:38.

**E0 reproduction.** On val, 1,531 of 1,532 argmaxes agree with V1 (median |Δp| 1.9e-4, max 0.0058). On test, 1,480 of 1,481 agree (mean |Δp| 1.2e-4, max 0.0070). The max gap exceeds the predeclared 1e-3 tolerance; the tolerance was not changed after seeing the result. CPU and MPS give identical outputs, so the gap comes from library versions, not the device: Sai ran torch 2.14 / torchvision 0.29, and this run used torch 2.8 / torchvision 0.23. Each split's single argmax flip is a near-tie; on val, the flipped image has bcc 0.403 vs bkl 0.408 in V1 and 0.406 vs 0.405 here. Test deterministic macro AUROC is 0.9264, identical to V1's published value; macro F1 is 0.6232 vs V1's 0.6213.

**Calibration (val only).** T = 2.007. On val: NLL 1.171 → 0.945, ECE 0.226 → 0.065, Brier 0.587 → 0.506.

**Fixed budgets on test (T=k uses the first k passes of the bank; uncalibrated / with val-fitted T):**

| Method | Macro AUROC | Macro F1 | Balanced acc | ECE | Brier | Melanoma missed / total |
|---|---|---|---|---|---|---|
| Deterministic T=1 | 0.9264 | 0.6232 | 0.7302 | 0.2213 | 0.5595 | 37 / 165 |
| MC T=10 | 0.9261 | 0.6129 | 0.7189 | 0.2194 | 0.5545 | 38 / 165 |
| MC T=20 | 0.9261 | 0.6196 | 0.7291 | 0.2152 | 0.5545 | 38 / 165 |
| MC T=50 | 0.9273 | 0.6219 | 0.7275 | 0.2156 | 0.5540 | 36 / 165 |
| MC T=50 + temperature | 0.9304 | 0.6219 | 0.7275 | 0.0568 | 0.4849 | 36 / 165 |

These are single-run point estimates with no confidence intervals. Sabari's analysis owns the paired comparisons and conclusions.

**Timing** (50 val images, batch 1, 10 warmup calls, 3 repeats, model inference only on preloaded tensors):

| Path | Backbone passes | Head passes | p50 ms | p95 ms |
|---|---|---|---|---|
| Deterministic T=1 | 1 | 1 | 9.4 | 10.6 |
| MC T=10, feature reuse | 1 | 10 | 11.2 | 12.1 |
| MC T=20, feature reuse | 1 | 20 | 13.0 | 14.3 |
| MC T=50, feature reuse | 1 | 50 | 17.7 | 19.1 |
| MC T=50, naive full network | 50 | 50 | 413.6 | 444.0 |

Preprocessing (JPEG decode + transform, CPU) adds a p50 of 4.8 ms per image. Batch-50 throughput figures in `timing.json` come from only 3 repeats and are noisy; use the batch-1 latencies.

### Caveats

- Savings from stopping MC early are savings in **head** passes. The backbone runs once whether T is 10 or 50: going from T=50 to T=10 saves about 6.5 ms per image (17.7 → 11.2 ms), while feature reuse itself is roughly a 23× speedup over the naive path.
- The temperature is fitted for the 50-pass mean and is not claimed to be optimal for T=10, T=20 or the deterministic logits. It is never fitted on test.
- Library versions shift probabilities slightly compared with V1's committed predictions (see E0 above). The device does not.


## C. Controller, comparators and analysis (Member C, `src/wb2/controller/`, `src/wb2/analysis/`)

The bank format in section A was agreed with Member A on 2026-10-27. `src/wb2/controller/bank.py` validates every bank on load and refuses to continue on any of:
- a shape other than N × 50 × 7;
- IDs or labels that don't match `data/splits/{split}.csv`;
- duplicate IDs;
- a class order other than C1's;
- a `PARTIAL` run without explicit pilot opt-in, or any `FAILED` run.

Rows with non-finite values are counted and reported. Both delivered banks pass, with 0 non-finite rows.

### Controller design (Workbook Chapter 7)

**Inputs at inference time:** the ordered per-pass logits of one image and the frozen temperature τ_T. Policies never receive `y_true`; labels are joined only afterwards, for evaluation. A unit test shuffles the labels and checks that every decision stays identical.

**Statistics at checkpoint k ∈ {5, 10, …, 50}** (`features.py`), computed from passes 0..k−1 only. A unit test checks that changing later passes never changes earlier statistics.
- p̄_k: mean of softmax(logits/τ_T) over the first k passes.
- pred_k = argmax p̄_k. `tie` = the top two probabilities are equal.
- H_k: entropy of p̄_k ÷ ln 7, so it lies in 0..1. Expected entropy and mutual information are also recorded.
- s_k = p̄_mel + p̄_bcc, the suspicious score. akiec is reported separately as an attention score; it is **not** in the suspicious group.
- Changes since the previous checkpoint:
  - class flip;
  - ΔH = |H_k − H_{k−5}|;
  - Δp = ‖p̄_k − p̄_{k−5}‖∞;
  - change in mean per-class variance.

  At k = 5 there is no previous checkpoint, so nothing is "stable" there and the minimum is 10 passes.

**Model-risk tier.** This is a model risk/attention tier, not a pathology severity. The first matching rule wins:

| Tier | Rule |
|---|---|
| critical | (benign-class prediction **and** s ≥ τ_s), the false-reassurance case, **or** (mel/bcc prediction **and** (unstable or H ≥ h_hi)) |
| high | mel/bcc prediction, **or** H ≥ h_hi |
| medium | H ≥ h_lo, **or** \|s − τ_s\| < m |
| low | otherwise |

"Unstable" means the class flipped, the top two are tied, or ΔH ≥ δ_H.

**Stop and review** (`policies.RiskAware`):
1. Draw passes 5 at a time, up to 50.
2. Stop as **TRUSTED RESULT** at the first checkpoint where every condition holds:
   - k ≥ 10 (k ≥ 20 for the high tier);
   - no class flip and no tie;
   - ΔH < δ_H;
   - H < h_hi;
   - |s − τ_s| ≥ m;
   - the tier is not critical.
3. If 50 passes are reached without stopping, the result is **NEEDS REVIEW**. The reasons list each failed check (`UNSTABLE_CLASS`, `UNSTABLE_UNCERTAINTY`, `HIGH_ENTROPY`, `SUSPICIOUS_MARGIN`, `CRITICAL_TIER`) plus `BUDGET_EXHAUSTED`.
4. A non-finite logit gives NEEDS REVIEW `NONFINITE` at once.
5. `explanation_check = NOT_RUN` on every decision. The occlusion / Grad-CAM++ verifier is planned V4 work and is never claimed.
6. A separate `clinical_attention` flag marks every mel or bcc prediction.
   - A stable, confident suspicious prediction may be TRUSTED by the computational checks, but it always carries this flag and is never presented as benign or safe.
   - This is a declared design choice. If every suspicious prediction went to review instead, V1's suspicious predictions alone would be about 37% of test images, and no review budget could be met.

**Comparators**, all replayed on the same ordered draws:
- deterministic T=1 (the det logits);
- fixed T = 10 / 20 / 50, where T=10 and T=20 are prefixes of the T=50 bank;
- fixed T=50 + entropy referral (refer H ≥ θ, with θ set on val to the review budget);
- **variance-only adaptive**: stop at the first k ≥ 10 where Δp < ε_p and Δ(mean variance) < ε_var, cap 50, plus an entropy-referral gate tuned the same way. This comparator is MC-CP-*inspired*: it has no conformal step and is not an MC-CP reproduction.

### Validation-only tuning protocol

Declared before tuning: the review budget is R = 0.20 and the macro-F1 tolerance is 0.01.
- **Risk-aware controller:** minimize mean passes on val, subject to all three of:
  - review rate ≤ R;
  - accepted-melanoma misses ÷ all melanoma ≤ that of fixed T=50 + referral at the same R;
  - macro F1 ≥ T=50 macro F1 − 0.01.

  Ties go to fewer melanoma misses, then the lower review rate.
- **Variance comparator:** minimize mean passes subject to the macro-F1 constraint; its referral threshold refers the top R by final entropy.

**Primary vs secondary analysis**, declared before any test replay:
- **Primary:** calibrated, using Member A's val-fitted T = 2.007 (`calibration.json`), applied to per-pass logits before softmax.
- **Secondary:** uncalibrated.

Each has its own config, tuned on val only and locked: `configs/wb2_controller.yaml` and `configs/wb2_controller_uncalibrated.yaml`.

**Tuning history (full disclosure).**
- Grid v1 had 457 settings. Its val optimum hit the grid edges: h_hi = 0.7, h_lo = 0.15, δ_H = 0.05, ε_p = 0.05.
- The ranges were widened **once**, on val, before any test replay, to grid v2 (2,306 settings, defined in `tune.py`). The configs were then frozen.
- Grid v2 still selects edge values for h_hi (0.9 calibrated, 0.8 uncalibrated) and h_lo (0.05). These were left as they are rather than searched further.
  - **h_lo has no effect on decisions:** the medium tier follows the same stopping rule as low, so h_lo only changes the tier label, and the tie goes to the first grid value.
  - **A high h_hi means the entropy cap rarely binds:** stopping is driven by class/uncertainty stability and the suspicious-score margin.
- Every evaluated setting is saved in `results/wb2/{calibrated,uncalibrated}/tables/tuning_grid_val.csv`.

Safeguards in the tools:
- `tune.py` refuses non-val banks.
- `run_wb2.py` refuses to run when:
  - the config is unlocked;
  - the calibration doesn't match the one used for tuning;
  - the val and test banks come from different checkpoints;
  - a synthetic bank would be written into `results/wb2/`.

### Commands (from the repo root)

```bash
python -m unittest discover -s tests -t .
D=results/wb2/inputs/wb2-mc-v1b0-s42
# 1. tune on val only (writes the locked configs)
python -m src.wb2.controller.tune --bank-dir $D --temperature-json $D/calibration.json \
    --out configs/wb2_controller.yaml --grid-out results/wb2/calibrated/tables/tuning_grid_val.csv
python -m src.wb2.controller.tune --bank-dir $D \
    --out configs/wb2_controller_uncalibrated.yaml --grid-out results/wb2/uncalibrated/tables/tuning_grid_val.csv
# 2. replay + evaluate (val, then test once)
for s in val test; do
  python -m src.wb2.analysis.run_wb2 --bank-dir $D --split $s --config configs/wb2_controller.yaml \
      --temperature-json $D/calibration.json --out-dir results/wb2/calibrated --status MEASURED
  python -m src.wb2.analysis.run_wb2 --bank-dir $D --split $s --config configs/wb2_controller_uncalibrated.yaml \
      --out-dir results/wb2/uncalibrated --status MEASURED
done
```

(Re-running `tune` on an existing config requires `--overwrite`. Use it only if that config has not been used on test.)

Each `results/wb2/{calibrated,uncalibrated}/` directory contains:
- `decisions/decisions_{split}.csv.gz`: WB2-O1, one row per image × method;
- `tables/`, holding:
  - the policy comparison and fixed-budget tables;
  - stop-pass and tier distributions;
  - the paired lesion-grouped bootstrap (1,000 resamples, seed 42, resampling whole lesions);
  - the melanoma audit and confident errors;
  - the coverage–risk curve;
- `figures/controller_*`;
- `manifest_{split}.json`;
- `report_{split}.md`.

The lesion bootstrap is a different estimator from V1's image-level bootstrap and is labeled separately.

### Interpretation limits

- Pass savings from replaying saved logits are **algorithmic**, counted in stochastic *head* draws. With feature reuse the backbone runs once per image regardless of T. Per section A, going from T=50 to T=10 saves about 6.5 ms per image on Member A's M2. Any wall-clock claim must use those measured timings, not the pass counts.
- The temperature is fitted for the 50-pass mean only. The adaptive policies apply the same T at every checkpoint, so it is not claimed optimal for shorter prefixes.
- TRUSTED RESULT means the model's computational checks passed. It never means clinically safe or benign.

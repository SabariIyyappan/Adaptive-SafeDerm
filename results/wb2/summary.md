# WB2 Results Summary — risk-aware adaptive MC-Dropout controller

**Owner:** Sabari (Member C). Sampling inputs come from Soumya (Member A); the checkpoint handoff is from Sai (Member B). Written 2026-10-27.

Every number below comes from the CSVs in `results/wb2/{calibrated,uncalibrated}/tables/`; regenerate them with the commands in `docs/wb2_reproduction.md` §C.

**Status labels:** **MEASURED** = executed on the real V1 checkpoint and the full frozen split. **NOT RUN** / **PLANNED** = not executed. No synthetic numbers appear here.

| Item | Status |
|---|---|
| Fixed MC budgets T = 1 / 10 / 20 / 50 (E1) | **MEASURED**, full val (1,532) and test (1,481) |
| Temperature scaling (E2), T = 2.007 fitted on val only | **MEASURED** |
| Variance-only adaptive comparator and risk-aware controller (E3) | **MEASURED** by offline prefix replay, full val and test |
| Safety / melanoma audit (E4) | **MEASURED**, all 165 test melanomas |
| Online (early-exit) timing of the adaptive controller | **NOT RUN**. Pass savings are algorithmic; see §6. |
| Occlusion / Grad-CAM++ explanation verifier (E5) | **NOT RUN**, planned V4. Every decision records `explanation_check = NOT_RUN`. |

## 1. Setup

**Model and data:**
- Checkpoint: the V1 EfficientNet-B0 `best.pt` (SHA256 `901a73a6…f9ae0d`, epoch 30), unchanged. No retraining.
- Data: the frozen lesion-grouped split; the test set has 165 melanomas.
- Preprocessing: the V1 path, with no hair removal and no CLAHE.

**Stochastic sampling** (Soumya, see `docs/wb2_reproduction.md` §A):
- Only the head `Dropout(0.3)` is active. StochasticDepth is off and BatchNorm is frozen.
- 50 ordered draws per image, with 1 backbone pass and 50 head passes (backbone features reused).
- Seed 42 + shard index.
- Fixed T and both adaptive policies all read prefixes of **the same** 50 draws.

**Tuning:** validation only.
- Review budget R = 20% and macro-F1 tolerance 0.01, both declared before tuning.
- Configs locked before any test replay: `configs/wb2_controller.yaml` (calibrated) and `configs/wb2_controller_uncalibrated.yaml`.
- The calibrated analysis was declared **primary** before the test replay.
- The grid was widened once, on val, before test; the history is in `docs/wb2_reproduction.md` §C.

**Locked thresholds (primary):**
- τ_s = 0.30, h_lo = 0.05, h_hi = 0.90, m = 0.05, δ_H = 0.10.
- Minimum passes: 10, or 20 for the high tier. Cap: 50.
- Variance comparator: ε_p = 0.075, ε_var = 0.002.

**Statistics:**
- Paired bootstrap that resamples whole **lesions** (1,000 resamples, seed 42, 0 skipped).
- This is a different estimator from V1's image-level bootstrap and is not mixed with it.

## 2. Primary results — test, calibrated (T = 2.007)

| Method | Macro AUROC | Macro F1 | Bal. acc | ECE | Brier | Mean passes (median / p95) | Review rate | Accuracy of accepted | Mel missed & accepted / all mel | … / accepted mel | Suspicious (mel+bcc) falsely reassured |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Deterministic T=1 | 0.9300 | 0.6232 | 0.7302 | 0.0576 | 0.4854 | 1 | 0% | 0.6226 | 37/165 | 37/165 | 48/236 |
| Fixed T=10 | 0.9297 | 0.6135 | 0.7198 | 0.0578 | 0.4849 | 10 | 0% | 0.6172 | 38/165 | 38/165 | 49/236 |
| Fixed T=20 | 0.9298 | 0.6194 | 0.7289 | 0.0562 | 0.4852 | 20 | 0% | 0.6199 | 38/165 | 38/165 | 49/236 |
| Fixed T=50 | 0.9304 | 0.6219 | 0.7275 | 0.0568 | 0.4849 | 50 | 0% | 0.6205 | 36/165 | 36/165 | 47/236 |
| Fixed T=50 + entropy referral (20% budget) | 0.9304 | 0.6219 | 0.7275 | 0.0568 | 0.4849 | 50 | 18.9% | 0.6736 | 28/165 | 28/135 | 37/236 |
| Fixed T=50 + entropy referral (matched to controller)¹ | 0.9304 | 0.6219 | 0.7275 | 0.0568 | 0.4849 | 50 | 13.0% | 0.6563 | 30/165 | 30/145 | 40/236 |
| Variance-only adaptive + referral | 0.9297 | 0.6135 | 0.7198 | 0.0578 | 0.4849 | 10.0 (10 / 10) | 18.8% | 0.6692 | 29/165 | 29/135 | 38/236 |
| **Risk-aware adaptive (proposed)** | 0.9297 | 0.6134 | 0.7200 | 0.0594 | 0.4854 | **18.3** (10 / 50) | **11.3%** | 0.6253 | **22/165** | **22/149** | **26/236** |

How to read the columns:
- AUROC, F1, balanced accuracy, ECE and Brier are computed over **all** images, with no referral. The safety columns count only TRUSTED (accepted) results.
- "Mel missed & accepted" is a true melanoma that was TRUSTED but predicted as another class. It is shown over two denominators that are never mixed: all melanomas, and accepted melanomas.
- Deterministic T=1 uses the same val-fitted temperature for comparability. That temperature was fitted for the 50-pass mean, so it isn't claimed optimal for T=1.

¹ The matched-coverage baseline was added **after** the first test replay, because the 20%-budget baseline reviews more images than the controller does. Its threshold is set on **val** to the controller's val review rate (13.6%), using only the val bank and the locked config (`src/wb2/analysis/matched_coverage.py`). No test quantity enters it.

### Paired differences on test (risk-aware − baseline, 95% lesion-grouped CI)

| Baseline | Δ mel missed & accepted / all mel | Δ review rate | Δ mean passes | Δ macro F1 | Δ macro AUROC | Δ ECE |
|---|---|---|---|---|---|---|
| Fixed T=50, no referral | **−8.5 pp [−14.4, −3.8]** | +11.3 pp [+9.7, +13.1] | **−31.7 [−32.4, −31.0]** | −0.0085 [−0.0190, −0.0004] | −0.0007 [−0.0015, −0.0000] | +0.0026 [−0.0059, +0.0097] |
| Fixed T=50 + referral, matched | **−4.9 pp [−10.7, −0.6]** | −1.6 pp [−3.8, +0.6] | −31.7 | −0.0085 [−0.0190, −0.0004] | −0.0007 | +0.0026 |
| Fixed T=50 + referral, 20% budget | −3.6 pp [−9.3, +0.6] (CI includes 0) | −7.6 pp [−10.0, −5.1] | −31.7 | −0.0085 | −0.0007 | +0.0026 |
| Variance-only adaptive + referral | −4.2 pp [−10.1, 0.0] (CI touches 0) | −7.4 pp [−10.0, −5.1] | +8.3 [+7.6, +9.0] | −0.0001 [−0.0029, +0.0026] | +0.0001 | +0.0016 |

### Secondary: uncalibrated (separately tuned and locked config)

Results go the same way:
- The risk-aware controller uses 18.3 mean passes and reviews 11.2% of images.
- It accepts 22/165 melanomas missed (22/151 accepted).
- Fixed T=50 + referral, matched on val, reviews 11.1% and accepts **33/165** missed.
- Paired Δ mel-miss: **−6.7 pp [−12.6, −2.0]**, with Δ review +0.1 pp [−1.9, +2.2].

Full tables are in `results/wb2/uncalibrated/tables/`.

## 3. Answers to the research questions

**Q1. Fixed MC budgets vs deterministic T=1.** On this checkpoint, MC-Dropout gives **no measurable benefit**.
- On test, every paired difference between deterministic T=1 and MC T=50 has a CI containing 0:
  - ΔAUROC +0.0004 [−0.0002, +0.0011];
  - ΔF1 −0.0013 [−0.0057, +0.0028];
  - ΔECE −0.0008 [−0.0087, +0.0054].
- T=10 and T=20 are within about 0.01 macro F1 of T=50.
- The likely reason: the only stochastic module is a single head dropout layer after global pooling, so the 50 draws vary little.
- This doesn't reproduce the paper's T=50 gain. Their model and preprocessing differ (V2/V3), and these numbers are not compared to theirs as like-for-like.

**Q2. Can a risk-aware 5–50-pass controller cut passes while keeping melanoma safety?** On this test set, **yes, within the stated limits.**
- Against fixed T=50 it uses **63% fewer stochastic head passes** (18.3 vs 50).
- It refers 11.3% of images.
- It accepts **fewer** missed melanomas than fixed T=50 with entropy referral at the same review rate: 22 vs 30 of 165. The CI excludes 0; with uncalibrated configs it's 22 vs 33.
- **Costs:**
  - macro F1 is lower by 0.0085, a small amount whose CI excludes 0;
  - accuracy among accepted images is lower (0.625 vs 0.656 at matched coverage). The controller spends its review budget on suspicious or unstable cases rather than on general uncertainty.
- Against the 20%-budget referral baseline, the melanoma difference is **not significant**.

**Q3. Risk-aware vs generic variance-convergence stopping.**
- The variance comparator stopped at the 10-pass minimum for **every** image, on both val and test. With head-only dropout, the mean prediction stops changing almost immediately, so a convergence rule simply becomes "T=10" here.
- The risk-aware controller spends extra passes (mean 18.3) on high, critical and borderline-margin cases. It reviews less (11.3% vs 18.8%) and accepts fewer missed melanomas (22 vs 29), but the CI on that difference touches 0.
- This comparator is MC-CP-*inspired* only. It has no conformal step and is not an MC-CP reproduction.

**Q4. Calibration and latency.**
- **Calibration:** the val-fitted temperature cuts test ECE for T=50 from 0.2156 to 0.0568, and Brier from 0.5540 to 0.4849 (Soumya's measurement, reproduced here). The same direction held on val (0.226 → 0.065).
- **Latency:** the measured savings are mostly in *head* passes. On Soumya's M2, p50 per-image latency is 17.7 ms for T=50 with feature reuse and 11.2 ms for T=10, against 413.6 ms for the naive full-network T=50.
- So **feature reuse** is where the wall-clock gain comes from. Stopping early saves at most about 6.5 ms per image on top of that.
- The controller's own end-to-end latency was **NOT RUN** (§6).

## 4. Controller behaviour on test (primary)

**Where it stops:**
- 759 images at 10 passes.
- 539 at 20.
- 168 at the 50-pass cap.
- 15 at 15–40.

**Final tier:** low 17, medium 792, high 549, critical 123.
- With the locked h_lo = 0.05, almost every non-suspicious case is labelled medium. h_lo only changes the label, not the decision, so the low/medium split is **not informative** (see `docs/wb2_reproduction.md` §C).
- 123 of 168 reviews (73%) are critical-tier.

**Review reasons** among the 168 NEEDS REVIEW cases (one case can have several):
- `BUDGET_EXHAUSTED` 168;
- `CRITICAL_TIER` 123;
- `SUSPICIOUS_MARGIN` 106;
- `UNSTABLE_CLASS` 1;
- `HIGH_ENTROPY` 1.

Review is driven mainly by the suspicious-vs-benign margin and the "benign prediction with suspicious mass" (critical) rule, not by entropy.

## 5. Safety audit and adverse findings (E4)

**All 165 test melanomas, risk-aware controller, calibrated** (`tables/melanoma_audit_test.csv`):

| | TRUSTED RESULT | NEEDS REVIEW |
|---|---|---|
| Predicted mel (correct) | 127 | 2 |
| Predicted another class (missed) | **22** | 14 |

The 22 accepted misses were predicted as:
- **akiec 12**
- bkl 6
- bcc 3
- vasc 1

**Main adverse finding:** 12 of the 22 accepted melanoma misses were predicted **akiec**.
- By the design declared up front, akiec (pre-cancerous) is **outside** the suspicious group, so an akiec prediction with low mel+bcc mass passes as non-suspicious.
- Some of these are confident: ISIC_0024554 was TRUSTED as akiec at 0.982 confidence after 10 passes.
- The 3 cases predicted bcc still carry the `clinical_attention` flag, so **19 melanomas were accepted with no attention flag at all.**
- **Recommendation for the next iteration** (not applied, because it would be tuning on test): add akiec to a declared attention group, or include p_akiec in the suspicious score. Then re-tune on val and re-test.

**Other confident errors** (`tables/confident_errors_test.csv`): two images of one nevus lesion (HAM_0006150) were predicted bcc at 0.99 and TRUSTED as high tier. These are false alarms that carry the attention flag, not false reassurance.

**Model-level ceiling:** 36 of 165 test melanomas (22%) are missed by fixed T=50 before any review. The controller can only defer errors; it can't fix the classifier.

## 6. Limitations

1. **Pass savings are algorithmic.** They come from replaying saved logits offline and count only stochastic head draws. The backbone runs once per image in every MC method. No online early-exit implementation was timed.
2. **One checkpoint, one seed, one split.** No MC-seed sensitivity study was run. The melanoma comparisons rest on 165 test melanomas, so the CIs are wide; the matched-coverage CIs only just exclude 0.
3. **Tuning on the same validation set** that chose V1's checkpoint (val macro F1). The grid was widened once on val, and some thresholds sit at grid edges (h_hi, h_lo); disclosed in §C of the reproduction doc.
4. **The matched-coverage baseline was added after the first test replay.** Its threshold uses val only, but the comparison itself was chosen afterwards (§2, footnote 1).
5. **The temperature** is fitted for the 50-pass mean and applied at every prefix length and to T=1. It isn't claimed optimal for those.
6. **Small deviation from V1's committed predictions** (E0, max |Δp| 0.007), caused by torch/torchvision version differences; documented by Soumya. V1's published numbers stay a separate baseline and are not mixed into these tables.
7. **Not a clinical result.** This is a retrospective dermoscopic-image research prototype. TRUSTED RESULT means the model's computational checks passed, never clinically safe or benign. The tier is a model risk/attention tier, not a pathology severity.

## 7. Figures (`results/wb2/{calibrated,uncalibrated}/figures/`)

| File | Caption |
|---|---|
| `controller_coverage_vs_risk_test.png` | Accepted error and accepted-melanoma misses against coverage. Grey curve: fixed T=50 with highest-entropy-first referral. The risk-aware point lies below the curve for melanoma misses but above it for overall error. |
| `controller_compute_vs_risk_test.png` | Mean stochastic head passes against error among accepted images and accepted-melanoma misses, for every method. |
| `controller_stop_passes_test.png` | Pass budget at stop. The variance comparator stops every image at 10; the controller stops at 10, 20 or the 50 cap. |
| `controller_tiers_test.png` | Final model-risk tier counts, with how many in each tier went to review. |
| `controller_reliability_test.png` | Reliability diagrams (10 bins) for deterministic T=1, fixed T=50 and the controller. |

Same-named `_val` figures show the validation replay.

## 8. Workbook chapter feed

- **Chapter 6 (evaluation):** §1–§5 and the figures above; the metric definitions are in `src/wb2/analysis/evaluate.py`.
- **Chapter 7 (design):** the algorithm, tier table, stop rule and tuning protocol in `docs/wb2_reproduction.md` §C; the module layout under `src/wb2/`.
- **Chapter 8 (progress, dated 2026-10-27):**
  - Sai handed over the checkpoint.
  - Soumya produced the sampling banks, calibration and timing.
  - Sabari built and tested the controller, comparator, tuner and evaluation, then ran the val tuning, locked the configs and did one test replay per config.
  - Adverse finding: akiec melanoma misses (§5).
  - Next steps: the akiec attention group, online early-exit timing, MC-seed sensitivity, and the E5 explanation verifier.
- **Chapter 9 (dependencies):**
  - The real chain was: checkpoint → val bank → calibration / val tuning → locked configs → test bank → test replay.
  - Every MEASURED item above is complete; the NOT RUN items are listed in the status table at the top.

# deviations.md — every place we differ from the paper, with reason

Format: one entry per deviation, added by whoever makes the call (PLAN.md §4.5).

---

### V1 — Split method interpretation
**Section:** §3.1 / PLAN.md §5.3.4, §13 item 1
**What:** The paper describes the split as a single 70/15/15 stratified split of unique lesions, `random_state=42`. Its own reported lesion counts (1,891/405/405 = 2,701 total) don't match HAM10000's ~7,470 unique lesions, so the exact procedure that produced its numbers can't be reproduced directly.
**What we did:** Implemented as the two-step procedure PLAN.md specifies: (1) split lesions 70% train / 30% remainder, stratified by `dx`, seed 42; (2) split the 30% remainder in half (15%/15%), stratified by `dx`, seed 42. This is our documented interpretation of the paper's one-line description.
**Why:** It's the standard way to turn a single three-way stratified split into two `sklearn.train_test_split` calls, and it satisfies every property the paper's split needs to have (grouped by lesion, stratified by class, seed 42).

### V1 — Provisional Table 2 hyperparameters
**Section:** §4.1 / PLAN.md §5.3, Table 2 (TBD — see PLAN.md §14)
**What:** Paper Table 2 (optimizer, LR, batch size, weight decay, cosine schedule parameters) was not readable in the text version of the paper.
**What we did:** Used the plan's provisional defaults, pinned in `configs/v1-b0-baseline.yaml`: AdamW, LR 1e-4, weight decay 1e-4, batch size 32, cosine annealing with `T_max=30` (no warm restarts), 30 epochs, no early stopping, seed 42. These will be replaced with the exact Table 2 values in V2 once the paper PDF is available.
**Why:** PLAN.md §5.3.5 explicitly calls for provisional defaults here rather than blocking V1 on the missing table.

### V1 — Model/training code developed and unit-tested before the real dataset/GPU were available
**Section:** §3.3, §3.2.5 / PLAN.md §5.3 Member B tasks
**What:** `src/model/` (dataset, model, imbalance correction, training loop, prediction export) was first written and validated in an environment with a CPU-only PyTorch install and no real HAM10000 images.
**What we did:** Built and unit-tested every component — dataset loading against A's real C2 split format, the EfficientNet-B0 + dropout head architecture, the class-weight and WeightedRandomSampler formulas, and a full 2-epoch training + checkpointing + C3-export run — against the same synthetic HAM10000-shaped fixture Member A's tests use (`tests/fixtures.py`). The real 30-epoch `v1-b0-baseline` run followed once the real dataset and a GPU became available in the same session — see the entry below.
**Why:** Per the plan's "dummy-first" rule (§4.4), so the training/export logic is verified correct independent of when the real split and a GPU session are available.

### V1 — Real dataset acquired and full run completed on a sub-spec GPU (6GB VRAM)
**Section:** PLAN.md §3 (compute profiles), §5.3
**What:** The plan's P-GPU-S profile calls for "≥8 GB VRAM (16 GB preferred)". The real `v1-b0-baseline` run instead used a local NVIDIA RTX 3060 Laptop GPU with 6 GB VRAM, at the plan's provisional batch size of 32.
**What we did:** Ran the full 30-epoch training unmodified (no batch-size reduction, no mixed precision) — it fit in 6 GB without OOM and completed in ~35 minutes, faster than the paper's 45-90 min T4 estimate. Also: this session's network access reached both Kaggle and the Harvard Dataverse (unlike the environment the pipeline code was originally developed in), so the real 10,015-image dataset was downloaded via `src.data.acquire --source kaggle` and A's full pipeline (integrity audit, split, verify, EDA) was re-run against it in this same session; every check passed with results matching the paper's reported class counts and expected image counts exactly.
**Why:** Recording the actual hardware used, since it's below the plan's stated GPU recommendation, in case later versions (V2's ablation runs, V5's ResNet-50/DenseNet-121, which need ≥12 GB) hit VRAM limits this run didn't.

### V1 — RESOLVED: the paper's per-class "accuracy" is Eq. 17, not recall
**Section:** §3.7, Eq. 17 / PLAN.md §13 item 4
**What:** PLAN.md flagged the paper's per-class "accuracy" as ambiguous — Eq. 17 defines
one-vs-rest accuracy (TP+TN)/N, but the reported value for `nv` (69.8%) looked more like recall.
The plan's instruction was to report both and state which matches.
**What we found:** On our test split, `nv` scores **0.708 under Eq. 17** and **0.567 under recall**.
The paper's 0.698 matches Eq. 17. **This resolves the open item: the paper's per-class accuracy is
Eq. 17's one-vs-rest form.**
**What we do:** Both columns appear in every per-class table and in `per_class_accuracy.png`,
because Eq. 17 is strongly inflated for rare classes by the large true-negative pool
(dermatofibroma: 0.994 on Eq. 17 vs 0.800 on recall). Reporting Eq. 17 alone would overstate
minority-class performance. Asserted in `tests/test_eval.py` so a refactor can't silently swap them.

### V1 — Melanoma over-prediction is the specified design, not a defect
**Section:** §3.2.5 / PLAN.md §5.6
**What:** Our melanoma specificity is 0.754 against the paper's 0.962, while sensitivity is close
(0.776 vs 0.808). Of 452 melanoma predictions, only 128 are truly melanoma; **291 are benign nevi**.
**Why it happens:** The paper's two imbalance corrections compound multiplicatively. The
WeightedRandomSampler equalizes how often each class is drawn (nv 0.21x, mel 1.29x, df 13.2x) and
the class-weighted loss then re-weights those same examples again (Eq. 2). Combined, melanoma
carries ~36x nevus's effective training weight and dermatofibroma ~3,790x, pushing the decision
boundary hard off the majority class.
**What we do:** Report it rather than tune it away — this is faithful replication, and the errors
run in the clinically safe direction (only 1 true melanoma in the test set was called a benign
nevus). Documented with the weight table and an error-flow figure in `results/v1/summary.md` §4.
Explicitly **not** overfitting: train loss (0.037) and val loss (1.25) are computed on different
distributions — the resampled class-balanced stream vs the natural one — and validation macro F1
rises monotonically through epoch 30. Re-check after V2's preprocessing and V3's T=50 averaging.

### V1 — Evaluation inputs committed under `results/v1/inputs/`
**Section:** PLAN.md §4.1 (repo structure), §12 (definition of done)
**What:** PLAN.md §4.1 says `runs/` is never in git. That makes the V1 results non-reproducible
from a clean clone, since the predictions the eval suite consumes live only there.
**What we did:** Copied the four small C3/C5/C6 artifacts (~370 KB total: both prediction files,
`history.csv`, `checkpoint.json`) to `results/v1/inputs/`, with a README recording the source run,
producing commit and checkpoint sha256. Checkpoints (~17 MB each) and images stay excluded.
**Why:** §12 requires every artifact to "regenerate from stored predictions with one command".
Without this, reproducing V1 needs a 3 GB download and a GPU; with it, `python -m src.eval.run_eval`
runs anywhere. The provenance chain back to `best.pt` is preserved via the recorded hash.

### V1 — Consolidated the V1 report files
**Section:** PLAN.md §5.3 items 2 and 5, §4.2 (file ownership)
**What:** `results/v1/` held three thin, overlapping generated reports (`integrity_audit.md`,
`split_verification.md`, `member_b_handoff.md`) plus `deck_outline.md`. Two had mojibake (`§`
written as `?`) from a Windows console encoding issue.
**What we did:** Replaced them with a single `results/v1/data_report.md`, generated by
`src/eval/eda.py` with every fact **recomputed live** rather than copied, so it cannot go stale.
It covers everything §5.3 items 2 and 5 require: the full integrity audit, all leakage checks,
per-split class counts and proportions, and the ours-vs-paper split table. The handoff note was
transient (a message to Member C, since consumed) and the deck outline is superseded by the
generated figures.
**Note on ownership:** §4.2 assigns `src/data/` to Member A and we have not touched it. Removing
generated *reports* under `results/v1/` while preserving their content is a reporting change, which
is Member C's role — but Member A authored those files and should be told.

### V1 — Near-duplicate leakage check requires pixel verification, not hashing alone
**Section:** PLAN.md §5.3.5 (split verification)
**What:** We added an independent leakage check beyond lesion grouping: lesion_id cannot catch two
visually identical images filed under different lesion IDs. A 64-bit difference hash over all
10,015 images flagged 37 candidate groups, 16 of them crossing splits with different lesion IDs.
**What we found:** **All 16 are hash collisions, not duplicates.** Direct pixel comparison shows the
closest pair still differs by 15.0 grey levels on a 64x64 thumbnail (threshold for a true duplicate:
8.0); the rest differ by up to 71. Dermoscopic images share a dark circular vignette and a centred
blob, so a coarse hash collides readily.
**What we do:** The detector now verifies every candidate by pixel comparison and reports
`pixel_confirmed`. **Confirmed cross-split leakage: 0.** Reporting the hash candidates alone would
have been a false alarm, so the funnel from candidates to confirmed is shown explicitly in
`duplicate_candidates.png`.

### V1 — Dataset acquisition: Dataverse used instead of Kaggle (superseded entry)
**Section:** §3.1 / PLAN.md §5.3.1
**What:** An earlier note here recorded that the development environment could reach neither Kaggle
nor the Harvard Dataverse, so the pipeline was only exercised against a synthetic fixture. That is
no longer true and the entry is kept only for history.
**Current state:** The real dataset has been acquired twice — first by Member B via Kaggle (commit
`63f8348`), and again by Member C for the image-level EDA. Member C's machine had **no Kaggle
credentials** (`~/.kaggle/kaggle.json` absent, `kaggle` CLI not installed), so the
**Harvard Dataverse fallback** was used instead:

```bash
python -m src.data.acquire --dest data/raw --source dataverse
```

It needs no credentials. The 3.2 GB archive also contains
`HAM10000_segmentations_lesion_tschandl.zip` — **the V4 ground-truth lesion masks** (PLAN.md §8.3
Member A item 1) — which has been retained at `data/raw/`, so V4's mask acquisition is already done.
**Verification:** A's integrity audit was re-run against the re-downloaded copy and passed exactly:
10,015 images, 7,470 lesions, paper-matching class counts, all 600×450. The two downloads are the
same dataset, so Member C's EDA statistics describe the images Member B actually trained on.
**One gotcha:** the Dataverse archive stores metadata as `HAM10000_metadata` with no extension, and
nests the images in two inner zips. `acquire.py` handles the merge, but the metadata file must be
copied to `HAM10000_metadata.csv` for the downstream tools.

---

### V2 — D-V2-1: Table 2 hyperparameters still provisional
**Section:** §3.6, Table 2 / PLAN.md §6.3 Member B item 1
**What:** The paper PDF's tables were still not accessible (the publisher site returns HTTP 403 to automated fetches), so the exact Table 2 values could not be read.
**What we did:** Kept the V1 provisional values for every V2-V5 run (AdamW, LR 1e-4, weight decay 1e-4, batch 32, cosine `T_max=30`, 30 epochs, seed 42), as PLAN.md §6.3 allows.
**Why:** Changing hyperparameters without the source values would be a guess; keeping them fixed makes V1 -> V2 a clean preprocessing-only comparison.

### V2 — D-V2-2: Preprocessing implementation details the paper leaves open
**Section:** §3.2.1
**What:** The paper names the operations but not every library detail.
**What we did:** `src/data/preprocess.py` — OpenCV `MORPH_ELLIPSE` 17x17 as the "disk" element, `MORPH_BLACKHAT` on grayscale, `THRESH_BINARY` at 10, `INPAINT_TELEA` radius 3, then RGB->LAB, `createCLAHE(2.0, (8,8))` applied to **each** of L, a, b (paper wording), LAB->RGB, bilinear resize to 224x224, lossless PNG. Applied to train, val and test identically, on the full-resolution image before resizing.
**Observed:** At threshold 10 the mask covers a median 9.7% of pixels (95th percentile 31%) — it marks fine dark skin texture and lesion-border structure as well as hairs, and the inpainting smooths those too (`results/v2/figures/preprocessing_before_after.png`). Faithful to the stated parameters; reported, not tuned.

### V2 — D-V2-3: Data loading single-process on Windows
**What:** Multi-worker `DataLoader`s crashed on Windows ("worker exited unexpectedly") after the first epoch.
**What we did:** All V2/V5 runs use `--num-workers 0`. The 224x224 PNG cache makes this cheap; it doesn't affect results (same sampler/augmentation RNG stream for a given seed).

### V2 — D-V2-4: Ablation (v) "no ImageNet pretraining" included
**What:** It's unclear whether paper Table 8 contains a no-pretraining row (the text credits pretraining with +6-8% AUROC).
**What we did:** Ran it as `v2-abl-nopretrain` (random init, both corrections) and report it as a fifth row.

### V3 — D-V3-1: "T=1" is the deterministic dropout-off pass
**Section:** §4.2 / PLAN.md §13 item 8
**What we did:** ECE "T=1" uses one deterministic pass with dropout off. The MC budget sweep table also reports the mean of the first T stochastic passes for T = 1, 5, 10, 20, 30, 50 from the same 50-pass bank.

### V3 — D-V3-2: Head-only MC sampling over cached features
**Section:** §3.4.2 / PLAN.md §7.3 item 4
**What we did:** Only the head `Dropout(0.3)` is stochastic, so pooled backbone features are computed once and the head sampled 50 times. Verified on 32 real images against full-network passes under the same seed (max |diff| recorded in `results/v3/inputs/mc_report.json`). EfficientNet-B0's 16 StochasticDepth blocks and all BatchNorm stay in eval mode.

### V3 — D-V3-3: Supplementary validation-tau referral
**Section:** §3.4.4 / PLAN.md §13 item 5
**What:** The paper sets tau from the test-set entropy distribution. Table 6 replicates that exactly; Table 6b additionally reports tau from the validation entropy distribution, which uses no test information. Reported side by side; the paper-faithful Table 6 is the headline.

### V4 — D-V4-1: Mask source and alignment
**Section:** §3.5 / PLAN.md §8.3
**What we did:** Masks from the HAM10000 Harvard Dataverse file `HAM10000_segmentations_lesion_tschandl.zip` (file id 3838943). Coverage is 10,015/10,015 images (1,481/1,481 test). Each mask is resized to 224x224 with nearest-neighbour (same plain resize as the images) and binarized.

### V4 — D-V4-2: Grad-CAM target class and referral rate
**Section:** §3.5, Table 10 / PLAN.md §13 item 9, §7.3 C item 5
**What we did:** Target class = predicted class. Accepted vs referred split at r = 30% (Table 10's r is unreadable in our copy); IoU at every r is also reported.

### V5 — D-V5-1: MobileNetV3 variant
**What we did:** MobileNetV3-**Large** (Table 7's variant unreadable). Its torchvision classifier is replaced wholesale by the shared 960->256->Dropout(0.3)->7 head.

### V5 — D-V5-2: Mixed precision for ResNet-50 and DenseNet-121
**Section:** PLAN.md §9.3 ("If a member has less than 12 GB VRAM: use mixed precision...")
**What we did:** The only available GPU is a 6 GB RTX 3060 Laptop, so `v5-resnet50` and `v5-densenet121` train with `torch.autocast` + `GradScaler` (`amp: true`) at the unchanged batch size 32. EfficientNet-B0 and MobileNetV3 runs are fp32.

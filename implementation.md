# Adaptive SafeDerm — Workbook 2 Implementation & Experiment Execution Plan

**Status:** execution specification for human team members and their AI coding agents; not a claim that the proposed system is complete.  
**Repository:** https://github.com/SabariIyyappan/Adaptive-SafeDerm (work from the current `main` branch; `second-main` was merged).  
**Planning date:** Monday, October 26, 2026 (America/Los_Angeles).  
**Submission deadline:** Thursday, October 29, 2026, **8:00 AM Pacific**. Target a submission-ready PDF by **Wednesday, October 28, 8:00 PM Pacific**, retaining overnight contingency.  
**Project type:** CMPE 295A **research**, not a product-development workbook.  
**People:** Gandhi Soumya Atluri (Soumya), Sabari Iyyappan Duraipandian (Sabari), Sai Teja Nomula (Sai).  
**Advisor:** Dr. Vidhyacharan Bhaskar.  
**Primary objective:** provide measured baseline experiments and a reproducible *early implementation of the proposed risk-aware controller* for Workbook 2 Chapters 6–9. The goal is credible, clearly delimited evidence, not a finished clinical product or an invented performance gain.

> **Instructions to coding agents:** Treat this document as the execution contract. Before changing anything, inspect the current `main` working tree, `PLAN.md`, `CONTRACTS.md`, `docs/run_registry.md`, `docs/deviations.md`, `configs/v1-b0-baseline.yaml`, the relevant `src/model/` and `src/eval/` modules, `tests/`, and `results/v1/summary.md`. Confirm the current commit hash in the run manifest. The online inspection that informed this plan is a snapshot; repository files may change. If an existing implementation differs from this plan, reuse functioning code, document the difference, and do not silently overwrite it.

---

## 0. Non-negotiable rules for all AI coding agents

1. **NO AI attribution in Git history.** AI coding agents are tools, not project members or coauthors. **Never** add `Co-authored-by:`, `Signed-off-by:`, `Generated-by:`, author lines, trailer metadata, or commit messages naming an AI assistant, coding agent, bot, model, or vendor as an author/coauthor. Do not configure a bot identity or impersonate another person. Any commit must be reviewed, approved, and authored by the responsible human using that human's own legitimate Git identity. Prefer leaving changes uncommitted for human review; if a human explicitly authorizes an agent to run a commit, verify the author/committer and all trailers before committing. Never push or merge without a human's explicit instruction.
2. **No fabricated results.** Do not invent metric values, GPU timings, successful tests, checkpoint files, plots, branches, commit hashes, or run completions. Clearly label every outcome as *MEASURED*, *PLANNED*, *FAILED*, *PILOT ONLY*, or *NOT RUN*. Synthetic fixture numbers must never appear in Workbook 2 as model performance.
3. **Do not overwrite V1.** Keep `data/splits/` immutable; do not regenerate the existing lesion-grouped split, edit existing V1 results, or train a substitute model without making the changed checkpoint provenance explicit. Do not silently combine V1 and new runs into one undifferentiated table.
4. **No test-set tuning or leakage.** Fit temperature and set stopping/referral thresholds only using the *validation* split. Lock all hyperparameters/thresholds before running or inspecting the final test comparison. Image IDs and true labels may be used for evaluation, never at inference time to choose passes, tiers, or referral decisions.
5. **No unapproved clinical claims.** This is a retrospective dermoscopic-image research/decision-support prototype, not a diagnostic or clinically validated medical device. Never call an explanation 'verified correct' merely because an occlusion changes a prediction. NEVER infer a ground-truth diagnosis from heatmaps.
6. **No unauthorized new scope.** During these three days, do not spend the critical path on retraining all V2/V5 models, a website, external dataset acquisition, full Grad-CAM++ infrastructure, or exhaustive paper reproduction. These remain documented future work unless the must-have gates finish early.
7. **Work independently.** Each human/agent works on a separate feature branch/worktree, owns separate paths, does not rewrite shared modules without owner approval, and hands over *versioned artifacts with an agreed schema*. Avoid cross-branch cherry-pick conflicts. Changes to `CONTRACTS.md`, existing evaluation APIs, and pinned dependencies require team coordination.
8. **Preserve provenance.** Every experiment artifact records model/checkpoint hash, source commit, config and random seed, split, dataset/preprocessing version, GPU/CPU hardware, installed dependency versions, start/end time, number of images, and status. Protect large/raw datasets and private storage locations from Git commits.
9. **Report adverse findings.** If adaptive stopping degrades melanoma safety, calibration worsens ECE, or runtime does not improve, show that honestly and explain how it informs the next iteration. Full marks are best pursued through sound methodology and real execution, not a positive result at any cost.

## 1. What is actually in `main` and why this is the fastest path

The accessible repository and Workbook 1 confirm the following V1 assets:

- `src/model/model.py`: EfficientNet-B0, seven outputs, 1280 → 256 → ReLU → Dropout(0.3) → 7 head, trained with ImageNet starting weights.
- `src/model/predict.py`: loads an existing checkpoint, produces deterministic **single-pass** predictions with dropout disabled; its documentation explicitly reserves MC-Dropout for a later phase.
- `src/model/dataset.py`, training/evaluation modules, the frozen grouped split, and `configs/v1-b0-baseline.yaml` already provide reusable data/model interfaces.
- `results/v1/summary.md`, `results/v1/inputs/`, and `docs/run_registry.md`: prior measured test macro AUROC **0.9264**, macro F1 **0.6213**, ECE **0.2219**, balanced accuracy **0.7271**, on **1,481 test images**. The validation set has **1,532** images; training has **7,002**. These are *single-pass, non-preprocessed V1* results, not results of the proposed controller.
- `src/eval/` already supports the original evaluation suite and bootstrapping; Workbook 1 reports **53 unit tests passing** at the time of submission. Re-run tests rather than assuming they still pass on the current branch.
- The original `PLAN.md` is explicitly a **faithful replication-only plan**. **This `implementation.md` supersedes that scope restriction for the short Workbook 2 proposed-project experiment only**, without deleting or rewriting the replication roadmap.

**Confirmed team handoff fact (October 26):** **Sai Teja Nomula retains the trained checkpoint from the previous V1 run because he performed the training.** The weight file is not assumed to be checked into Git. Its accessible location, readability, checkpoint hash, label mapping, and compatibility with current `main` must still be independently verified by Soumya before new experiments. A checkpoint metadata artifact is **not** a replacement for the trained weights. **Make Sai-to-Soumya transfer the first dependency gate; do not start a new training run merely to bypass the handoff.**

**Why not retrain now?** The assignment asks for baseline experiments and an early proposed solution. Reusing the exact V1 checkpoint lets us compare different inference policies with the *same model, images, preprocessing, and stochastic draws*. This isolates the effect of adaptive inference; retraining a different classifier would confound that comparison and waste scarce time.

### Target experiment questions

- **Q1 (baseline):** How do fixed MC-Dropout budgets T = 10, 20, 50 change AUROC, macro F1, ECE, melanoma sensitivity and forward-pass cost relative to deterministic T = 1?
- **Q2 (novel core):** At a matched model and review policy, can a **risk-aware** 5-to-50-pass controller reduce average dropout passes relative to T = 50 while maintaining or appropriately deferring potentially unsafe cases?
- **Q3 (fair prior art):** Does a clinical-risk-aware stop rule differ usefully from a **generic variance-convergence adaptive stop rule** under comparable budgets and coverage?
- **Q4 (secondary):** Does validation-fitted temperature scaling improve test calibration? Is actual end-to-end latency reduced, or are savings mainly in stochastic head passes?

The primary hypothesis may be **rejected**. The Workbook 2 value is the reproducible experiment and honest analysis.

---

## 2. Hard deliverables and priorities

| Priority | Minimum deliverable | Artifact / evidence | Owner | Acceptance condition |
| --- | --- | --- | --- | --- |
| P0 | V1 checkpoint + split/environment handoff | Checkpoint hash, path/size, load check, data path and hardware note; human-owned record in `docs/` | Sai (checkpoint owner; tiny first-day transfer), Soumya verifies | Sai transfers the actual V1 trained weights from his previous training run; Soumya independently loads those SAME weights and confirms the 7-class label mapping; otherwise raise blocker immediately |
| P0 | Single shared stochastic prediction bank | `results/wb2/inputs/` or non-Git shared storage; validation and test per-pass logits with manifests | Soumya | Each image has an ordered set of up to 50 raw stochastic logits; correct image IDs/labels, seed and SHA; all finite |
| P0 | Fixed-budget comparisons | `results/wb2/tables/fixed_budget.csv` plus evaluation config | Soumya produces predictions; Sabari evaluates | T=1 deterministic and fixed MC T=10/20/50 reported on identical held-out split; status and coverage stated |
| P0 | Executable early proposed controller | New isolated `src/wb2/controller/` module + synthetic-unit tests and a validation-tuned locked config | Sabari | Runs per-image prefix 5,10,...,50 with four risk tiers and clear TRUSTED / NEEDS REVIEW reasons; does not inspect labels |
| P0 | Matched controller-vs-baseline evaluation | `results/wb2/tables/` and figures with test metrics, risk-tier distribution, average passes, review rates, melanoma miss counts | Sabari | Risk-aware vs fixed T=50 and simple generic adaptive stopping on the same stochastic predictions; no test tuning |
| P0 | Workbook evidence package | `results/wb2/summary.md`, `results/wb2/manifest.*`, `docs/wb2_reproduction.md` | Sabari (results); Soumya (sampling reproduction) | Another person can identify exact input, output, run command documented in prose, seed, limitations and actual results |
| P1 | Temperature-scaled comparisons | Validation-fitted scalar, before/after ECE/NLL/Brier, raw and calibrated tables | Soumya fits; Sabari reports | Valid validation-only fit; no test-set temperature selection |
| P1 | Actual runtime microbenchmark | Device, warmup, p50/p95, batch size, cold/hot path and compute-accounting note | Soumya | Timings are **measured**, not estimated from saved logits |
| P2 | Small occlusion/Grad-CAM++ design spike | Optional short qualitative appendix, *not* a safety claim | Soumya only if P0/P1 done | Explicitly labeled preliminary; do not let this block submission |

**Minimum success threshold:** by Wednesday afternoon, have **one real checkpoint**, **at least one measured fixed-budget comparison**, **a runnable risk-aware controller prototype**, and **a quantitative or clearly labeled pilot comparison**. If the full 50-pass test bank cannot finish, use a predeclared held-out **subset of test images**, stratified to include melanoma and other classes, never selected by favorable model behavior; report sample count, IDs/protocol, and **PILOT ONLY**. V1's full test results remain a separate baseline.

---

## 3. Parallel ownership and strict workload boundaries

### Member A: Gandhi Soumya Atluri — stochastic inference & evidence generation

**Responsibility:** data/checkpoint consumption, inference experiment generation, calibration, practical runtime measurements. **No dependency on Sabari's controller code.** Work independently in a `wb2-soumya-inference` feature branch/worktree.

**Owned paths (new work):** `src/wb2/sampling/`, `tests/wb2_sampling/`, `configs/wb2_sampling.*`, `results/wb2/inputs/` or private shared equivalent, `results/wb2/figures/sampling_*`, sampling-specific sections of `docs/wb2_reproduction.md`. Do not modify Sai's `src/model/` unless essential and separately reviewed.

**A1 — Initial audit and feasibility (Monday):** Verify raw HAM10000 images are accessible, load V1 split CSVs without regenerating, confirm label mapping from `CONTRACTS.md`, confirm V1 checkpoint and its SHA256 with Sai, establish working CUDA (or CPU fallback), and run the original evaluation smoke check. Confirm all inputs are the *original un-preprocessed V1 path* so scores compare meaningfully to V1.

**A2 — Stochastic logit generation (Monday into Tuesday):** Build an isolated inference adapter around the existing V1 model. Load checkpoint once; set the full model to evaluation mode and activate **only the intended Dropout layer(s)** while keeping BatchNorm and other layers in eval mode. Run a deterministic dropout-OFF prediction path separately. Generate **50 ordered stochastic logits per image**, preserving pass index. Produce **validation first** and then test after gating. Use common draws for all T=10,20,50 and both adaptive algorithms; the T=10 result is the prefix of the T=50 bank, never an independently lucky sample. Include support for resumable shard output to survive a notebook/GPU cutoff.

**A3 — Calibration (Tuesday, after validation logits):** Define one frozen temperature scaling rule and fit exactly one positive scalar on validation **only** by minimizing mean negative log likelihood of the adopted MC predictive distribution. Choose and document whether temperature is applied to per-pass logits before softmax/averaging; use precisely the same transformation in validation and test. Record the pre/post-calibration NLL, ECE and Brier score. Never use `test.csv` to select temperature; never describe a temperature fitted for T=50 as automatically optimal for every other T.

**A4 — Inference-cost study (Tuesday):** Record raw T budget, actual expensive backbone evaluations, stochastic head evaluations, GPU batch size, hardware and warmup. **Crucial architectural caveat:** in V1, the explicitly introduced dropout is in the *classifier head*, after the feature extractor. If cached backbone features are reused across 50 dropout-head draws, '50 MC passes' does **not** mean 50 full EfficientNet backbone runs, and savings in head passes may yield a much smaller wall-clock improvement. Compare or at least document the naive repeated-full-network and optimized feature-reuse implementations, and report latency on the genuine end-to-end path. Do not sell an offline-replay speedup as measured end-to-end latency.

**A5 — Handoff (by Tuesday evening):** Give Sabari a schema-valid validation prediction bank **as soon as available** (test bank next), a small synthetic contract fixture, frozen calibration metadata, and clear human-readable reproduction steps. Do not wait for pretty plots to hand over data. If GPU runs fail, preserve partial files with explicit `FAILED`/`PARTIAL` status rather than masking missing samples.

**Definition of done:** 50-pass ordered logits (or explicitly documented subset), independent deterministic logits, correct split and checkpoint mapping, complete schema/provenance, readable handoff, and at least one repeatable timing record. Soumya does **not** own tier thresholds or the final research conclusions.

### Member C: Sabari Iyyappan Duraipandian — risk-aware controller, baselines, statistics, Workbook evidence

**Responsibility:** design and implement the proposed decision algorithm, variance-only comparator, evaluation, plots, detailed system-method documentation, and workbook-ready evidence. **Starts immediately with a schema fixture; does not wait for GPU output.** Work in `wb2-sabari-controller` feature branch/worktree.

**Owned paths (new work):** `src/wb2/controller/`, `src/wb2/analysis/`, `tests/wb2_controller/`, `tests/wb2_analysis/`, `configs/wb2_controller.*`, `results/wb2/tables/`, `results/wb2/figures/controller_*`, `results/wb2/summary.md`, design/reproduction text in `docs/wb2_reproduction.md`. Avoid editing `src/model/`, the original `src/eval/`, and Soumya-owned sampling files unless approved.

**C1 — Early design specification (Monday):** Write a controller interface whose only inference-time inputs are sequential stochastic logits/probabilities and a frozen calibration value; it must never read the true label. Define outputs per image: decision at every 5-pass checkpoint, final pass count, stop reason, tier, final class and risk score, review flag and textual review reason. Use the fixed class map akiec/bcc/bkl/df/mel/nv/vasc and document clinical-risk-group mapping; melanoma is always reported separately. Classify `mel` and `bcc` as suspicious for a conservative *research screening* group; `akiec` is pre-cancerous and should be either included in a declared broader attention group or reported separately, not silently relabeled as confirmed cancer. State this choice in the methods table.

**C2 — Two executable stopping policies (Monday–Tuesday, using synthetic fixtures):**

- **Generic variance-only adaptive baseline:** start with 5 passes, check only a clearly defined prediction/variance convergence statistic at 5-pass checkpoints (e.g., change in predictive class-probability variance and/or mean-probability vector between successive windows), stop if stabilized under a validation-chosen threshold; max 50. This is an *MC-CP-inspired convergence comparator*, **not** a claimed faithful reproduction of the entire MC-CP conformal method unless its official code/algorithm has actually been matched.
- **Proposed risk-aware adaptive controller:** 5 passes initially; add 5 if any material instability, uncertainty, or unsafe suspicious/benign margin remains; cap 50. At each checkpoint track predicted-class stability across successive checkpoints, change in total predictive entropy or its estimate, and a predeclared suspicious-vs-benign decision margin. At the first 5-pass checkpoint, where no previous checkpoint exists, either enforce a documented minimum of 10 passes for sequential-stability checks or use a predeclared within-batch stability criterion; do not claim cross-checkpoint convergence from only one checkpoint. Assign a risk tier **low / medium / high / critical** from explicit validation-selected thresholds. A low/medium, stable, sufficiently separated case may stop early; high/critical cases require additional passes up to a stop/cap rule and must be routed to **NEEDS REVIEW** if unresolved or high-stakes. A stable high-confidence suspicious prediction should still receive a review/clinical-attention flag, not be marketed as an autonomous benign diagnosis.

**C3 — Safety policy and comparison fairness (Tuesday):** Define a binary research output **TRUSTED RESULT** vs **NEEDS REVIEW**, with explicit reasons: unstable class, high entropy, suspicious/benign margin, high/critical tier, maximal budget exhausted, or optional explanation flag. Until the actual perturbation verifier exists, do **not** claim the explanation check was executed. Avoid changing the review target opportunistically: use a stated validation-set review budget or safety constraint; include fixed-T50 with an equivalently validation-tuned referral policy at matched coverage for a defensible safety/compute comparison. Also compare no-referral metrics separately so the reader sees the underlying predictive accuracy.

**C4 — Threshold tuning and test execution (Tuesday–Wednesday):** Use validation bank for the stopping/decision thresholds, save the chosen config and tuning objective *before* test evaluation. Replay each test sample's first 5,10,...,50 draws against identical precomputed logits. This **offline prefix replay** is a legitimate way to test budget allocation and prediction quality; call pass savings **simulated/algorithmic** until Soumya's real online path is timed. Evaluate all policies on the same test images and checkpoint, using the same stochastic-pass order.

**C5 — Complete tables and error audit (Wednesday):** Produce one detailed comparison covering deterministic T=1, fixed T=10/20/50, variance-only adaptive, risk-aware adaptive, and optional calibrated variants. Include macro AUROC (OvR), macro F1, balanced accuracy, ECE (10 bins), Brier, melanoma recall, suspicious-class recall, average/median/p95 stochastic passes, review rate, accepted-set accuracy, and melanoma false-negative **counts** among TRUSTED results. Show both (a) missed accepted melanoma as a fraction of **all melanoma cases** and (b) miss rate **conditional on melanoma accepted**, with numerator/denominator displayed; don't mix denominators. Include coverage-versus-risk and compute-versus-risk plots, tier histogram, passing-budget distribution, and a small case study of concerning errors.

**C6 — Statistical discipline (Wednesday):** Compute paired confidence intervals for differences where feasible; prefer lesion-group bootstrap because multiple images may share a lesion. Keep the prior V1 image-bootstrap method labeled separately for comparability. Do not claim statistically significant improvements without an appropriate paired test/interval. If the pilot is small, provide exact counts and uncertainty warnings rather than a falsely precise percentage. Report failure cases, especially confident but wrong melanoma predictions.

**C7 — Workbook integration feed (Wednesday):** Write `results/wb2/summary.md` with clearly labeled **MEASURED / PILOT ONLY / PLANNED** entries, figures with captions, architecture and algorithm descriptions, validation-only tuning details, and exact limitations. Deliver chapter-ready content for Chapter 6 (evaluation), Chapter 7 (system design and reproduction), Chapter 8 (dated progress / contribution evidence), and Chapter 9 (dependency facts). The final PDF assembly is a **human team activity**, not evidence that a controller experiment was run.

**Definition of done:** Two functioning adaptive policies, validation-locked settings, independent unit tests, correct fixed-budget results, safety accounting, readable figures/tables, and honest controller execution status. Sabari does **not** own retraining or raw-image dataset acquisition.

### Member B: Sai Teja Nomula — **very small task; finish first, then exams**

**Maximum human time budget: 60–90 minutes total, no later than Tuesday, October 27 (ideally Monday). No ongoing integration, no new training runs, no responsibility for blocker resolution after handoff.** Work in an isolated `wb2-sai-handoff` branch only if a file change is useful; otherwise share an artifact handoff note and finish.

**B1 — Transfer the checkpoint YOU trained (30–45 minutes):** Sai is the confirmed custodian of the previous V1 trained checkpoint. Locate the **actual best-model weight file from Sai's V1 training run** (`best.pt` or its real filename), not just checkpoint metadata or exported predictions. Transfer it to a private/shared storage location that Soumya can access **today**, and send the exact file name, access instructions (without credentials), file size, SHA256 hash, checkpoint epoch, run/config ID, model/head definition, seven-class label order, input resize/normalization, and original hardware details. If possible, Sai performs a quick read/load check; **Soumya owns the independent load + tiny inference verification** after transfer. Do not upload model weights or credentials to Git; do not initiate retraining. If transfer fails, notify Soumya immediately with the specific blocker and exit after the brief handoff window.

**B2 — Very brief read-only technical note (15–30 minutes):** Verify existing V1 architecture has the classifier-head `Dropout(0.3)`, and tell Soumya the safe activation point for stochastic inference while keeping BatchNorm frozen. Confirm the existing checkpoint loader shape and V1 prediction preprocessor, optionally record a one-paragraph `docs/wb2_checkpoint_handoff.md`. No new MC engine, no calibration, no writing full tests, no repeated meetings.

**B3 — Exit condition:** Message Soumya and Sabari with checkpoint location/access, hash, expected load configuration and caveats; get a one-line acknowledgement from Soumya that the actual file is accessible, then close out by Tuesday. All downstream work is owned by Soumya and Sabari. **Do not assign Sai last-minute debugging or report formatting.**

**If Sai becomes unavailable before the transfer:** Soumya checks any previously shared checkpoint copies and the run registry immediately, but does **not** assume Git contains the trained weights. Escalate the missing model artifact as a P0 blocker; use committed V1 predictions for documented baseline results while continuing controller development on synthetic fixtures. The project cannot wait for Sai's exam to end.

---

## 4. Shared artifact contract — agree within the first two hours

This deliberately adds **Workbook 2-specific** contracts without mutating the original C1–C10 definitions in `CONTRACTS.md`.

| ID | Producer → consumer | Logical artifact | Required content and validation |
| --- | --- | --- | --- |
| WB2-H0 | Sai → Soumya | Checkpoint handoff manifest | Hash, full checkpoint path, epoch, source config, label mapping, architecture identifier, whether load and tiny inference passed; no credentials |
| WB2-I1 | Soumya → Sabari | Validation stochastic logit bank | Shape **N_val × 50 × 7** with `image_id`, `lesion_id`, `y_true` aligned 1:1, 0-based ordered pass indices, logits (not mislabeled probabilities), checkpoint hash, seed, image transform and split hash |
| WB2-I2 | Soumya → Sabari | Test stochastic logit bank | Same **N_test × 50 × 7** schema; test never used to fit temperature or thresholds |
| WB2-I3 | Soumya → Sabari | Deterministic baseline logits | One dropout-OFF logit vector of length 7 per image; same IDs and transform; T=1 deterministic is distinct from one stochastic draw |
| WB2-I4 | Soumya → Sabari | Frozen calibration and timing record | Scalar T > 0, validation objective, temperature application order, metrics before/after, timing hardware/batch/warmup and exact sampling implementation |
| WB2-O1 | Sabari → team | Controller decisions | For each image and method: run ID, checkpoint hash, calibration ID, final 7-class probabilities, predicted class, suspicious score, tier, stop pass count, stop reason, review flag/reason, and label only for evaluation export |
| WB2-O2 | Sabari → team | Evaluation tables and plots | Method + full/pilot status, dataset size and class counts, metric definitions, confusion counts, uncertainty and confidence intervals, mean passes, review coverage, paired comparisons |
| WB2-R1 | Both → team | Reproduction/provenance note | Dependency versions, exact run and evaluation commands in *documentation* (not in this planning document), seed policy, paths and sizes, artifact hashes, configs, hardware, warnings, run registry updates |

**Format choices:** use structured efficient array storage for N×50×7 logits with a separate ID/metadata table; do not publish a gigantic human-unreadable 50-pass CSV. The uncompressed 1,532×50×7 float32 validation bank is about 2.1 MB and the 1,481×50×7 test bank about 2.1 MB, excluding metadata. Logits must be finite; each softmax probability vector must sum to 1 within numerical tolerance. Logits cannot be compared as if they are already calibrated probabilities. Use deterministic array ordering and preserve class column names. Paths above are *target names*, not claims that those files presently exist.

**Implementation tests agents must create:** exact checkpoint load; no image/lesion overlap across existing splits; label-map identity; constant image logit shape; identical results across methods for the same prefix draws; MC stochastic variation when dropout is enabled; BatchNorm statistics unchanged in MC mode; deterministic T=1 reproducible; pass counts limited to 5,10,...,50; stable stopping; budget exhaustion; low/medium/high/critical reachable using synthetic fixtures; review reason consistency; perturbation **not run** flagged correctly; logits/temperature invariants; no ground-truth leakage into stopping; JSON/table schema validation; no test thresholds in a training config. Preserve and rerun the original test suite.

---

## 5. Precise experiment protocol

### E0. Baseline provenance and reproduction check — MUST

**Input:** V1 trained checkpoint, fixed HAM10000 split, unchanged original V1 raw-image evaluation transform.  
**Output:** test/val deterministic logits, cross-check against the committed V1 `results/v1/inputs/` probabilities where possible, with tolerance and documented pipeline differences.  
**Acceptance:** baseline checkpoint actually loads; no data leakage; checkpoint/data versions are named. The published V1 numbers can be quoted from `results/v1/summary.md` as **previously measured** even before new runs, but never treated as evidence that the proposed controller works.

### E1. Fixed MC-Dropout budget study — MUST

Run the V1 checkpoint with **T=10, T=20, T=50 stochastic draws** on validation then test; keep **deterministic T=1** separate. Evaluate mean softmax probabilities from the *first T draws* (all budgets share the same saved draw order). Report macro AUROC, macro F1, ECE, Brier, melanoma sensitivity, all-class counts and compute (passes and measured time where available). The MC procedure must use dropout stochasticity but **not change BatchNorm running statistics**. Explain whether the model's only active stochasticity is classifier-head dropout and whether head-feature reuse is employed.

A direct comparison of uncalibrated fixed T=50 with V1 T=1 is the most important genuinely new baseline table. Do not compare these against the P1 paper's T=50 number as if preprocessing/checkpoint/training hyperparameters were identical.

### E2. Calibration — STRONGLY RECOMMENDED

Fit scalar temperature on **validation** for the declared MC prediction aggregation rule. Evaluate raw vs calibrated on test with validation-selected temperature unchanged. Report NLL and ECE alongside Brier; recognize that ECE depends on binning and may not improve for every subgroup. If the budget is tight, do not block E1/E3 on calibration: label the initial controller comparison **uncalibrated pilot**, then add calibrated results if ready.

### E3. Two adaptive policies — MUST for the strongest proposal evidence

**Variance-only comparator:** 5 initial draws, add 5 per checkpoint until validation-chosen variance-convergence rule passes or T=50 is reached.  
**Risk-aware proposed policy:** same 5-to-50 pass budget, but adds class-prediction stability, uncertainty stability, and suspicious/benign margin, with explicit four-tier priority and review decision.  
**Design preference:** compare using identical stochastic draws and same validation tuning objective, and include analysis at matched review/coverage. Do not tune risk thresholds on observed test melanoma misses.

For every method report: stopped at 5/10/.../50 distribution, average dropout draws, proportion forced to max, risk-tier distribution, classification metrics, review/accepted coverage and melanoma missed-and-accepted count; compare to T=50 fixed both with no review and a properly matched validation-calibrated review gate. V1's seven-class label taxonomy must be preserved, not reduced secretly to a malignant-only classifier.

**Interpretation warning:** an offline controller run on saved 50-draw tensors is an **evaluation of stopping policy**, not proof that the online implementation saved end-to-end time. The online implementation, if built, must actually avoid remaining draws after stop and be benchmarked separately. Quantify both average stochastic dropout passes and real backbone computations so computational claims are meaningful.

### E4. Safety-focused error audit — MUST

At minimum inspect all **true melanoma** images (full test if available; otherwise state pilot size), count melanoma predictions as other classes, record whether incorrect cases are **TRUSTED** or **NEEDS REVIEW**, and inspect cases with high confident wrong predictions. Report false reassurance among accepted cases, coverage and safe deferral. Always print raw numerator/denominator. Use conservative language: the tier is a **model risk/attention tier**, not an actual pathology severity rating.

### E5. Optional explainability spike — NOT ON CRITICAL PATH

The proposed complete design invokes Grad-CAM++ and targeted occlusion only for high/critical cases. The Workbook 2 system design should **specify** the module interface, mask fraction as validation-tuned (e.g., 15–25% candidate range), comparator occlusion (random/background/control), re-inference cost, and future faithful explanation tests. If time permits, run 5–20 **illustrative** high/critical cases with controls and report as preliminary only. **Do not describe a mask-induced probability drop as proof of medical correctness.** If it is not implemented, mark the verifier as *planned V4*.

### What is intentionally deferred beyond Workbook 2

New V2 preprocessing/retraining/imbalance ablations, architecture swaps, external Derm7pt/PAD-UFES-20/Fitzpatrick17k/DDI validation, UI/FastAPI, full ground-truth mask IoU evaluation, and complete paper reproduction. Record each as future work and represent them in the two-semester schedule, not as finished.

---

## 6. Research measurement definitions and guardrails

**Dataset:** HAM10000, 10,015 total dermoscopic images, frozen grouped **7,002 train / 1,532 validation / 1,481 test** split from the repository. Seed **42** for the default run. Tune on validation; test once per frozen policy. If compute permits, report variation over additional MC seeds as a sensitivity study **after** the principal locked comparison, not in place of it.

**Core classification:** seven-class macro AUROC (one-vs-rest), macro F1, balanced accuracy, confusion matrix, per-class recall/sensitivity and specificity, 10-bin ECE, multiclass Brier, NLL. Do not confuse one-vs-rest class accuracy with class sensitivity; Workbook 1 documents this ambiguity.

**Safety:** explicitly declare suspicious grouping (at minimum **mel + bcc**; describe `akiec` as pre-cancerous separately or in a separately declared attention tier). Melanoma FN among accepted = true `mel` with an accepted non-`mel` predicted class; report count, fraction of *all* true melanoma, and separately conditional fraction of *accepted* true melanoma. Report both *review rate* and *accepted coverage*, and compare risk at matching coverage when possible. A correctly flagged suspicious prediction may still require human clinical follow-up; `TRUSTED RESULT` means the model's computational checks passed, **never** 'clinically safe/benign'.

**Efficiency:** average/median/p95 stochastic head draws, distribution across stop points, review workload, CPU/GPU throughput and per-image p50/p95 latency. Runtime protocol requires repeated timing runs on the same machine and image mix with warmed model; distinguish inference-only and preprocessing time. With feature caching, list **backbone forward count separately from head-stochastic draw count**. Compute savings calculated from skipped draws must be labeled an algorithmic proxy, not wall-clock speedup.

**Confidence intervals:** 95% intervals for principal metrics and **paired method differences** if feasible. Bootstrap by lesion ID where practical to respect correlated images, with seed and iteration count logged; note historical V1 uses 1,000 image bootstrap resamples. In tiny pilot sets, show exact support and caution rather than unsupported significance language.

**Sensitivity / ablations worth doing only if time remains:** drop malignant-risk margin (uncertainty-only), drop stability condition, vary starting/minimum pass budget, compare review threshold at matched coverage, and measure optimized vs unoptimized inference path. Choose one interpretable ablation ahead of the flashy UI.

**Stop-rule spec quality bar:** every tier and threshold has an explicit formula/definition, validation tuning procedure, expected effect, permitted actions and fallback. Distinguish output class stability, entropy-estimate stability, and clinical-risk margin; do not call them interchangeable. State how ties, missing logits, nonfinite predictions and max-budget exhaustion are handled (conservative NEEDS REVIEW).

---

## 7. Three-day execution schedule and dependency gates

**Monday, Oct 26 — build and unblock (Day 1)**

- **First 2 hours:** all inspect `main` and agree WB2-I1/I2/I3 fields; Sai immediately transfers his existing trained V1 checkpoint and shares its identity/hash; Soumya verifies she can independently load the transferred weights and secures image/GPU access; Sabari builds synthetic fixtures and drafts the controller interface and review definitions. Lock file ownership and a shared checkpoint identity.
- **By Monday evening:** Sai is finished; Soumya can generate validation 50-pass logits (or a proof-of-life subset); Sabari has variance-only and four-tier controller unit tests passing against fake data and has documented threshold-search objectives. If checkpoint missing, activate fallback immediately.

**Tuesday, Oct 27 — produce experimental evidence (Day 2)**

- **Morning:** Soumya publishes validated WB2-I1 with SHA and begins test WB2-I2; Sabari tunes both policies *only on WB2-I1*. Calibration fitting happens on validation only.
- **Afternoon:** Soumya finishes T=10/20/50 shared-bank inference and timing microbenchmark; Sabari freezes calibration/stop/review settings and executes test comparison once WB2-I2 is ready.
- **Evening gate:** at least one real fixed-budget results table, one functioning adaptive-policy replay, full manifests, and a list of remaining blocks. If full test is infeasible, lock and disclose the predeclared test pilot set rather than spend all of Wednesday waiting.

**Wednesday, Oct 28 — verify, document and submit (Day 3)**

- **Morning:** Sabari finalizes safety/error breakdown, paired comparisons, plots and chapter-ready summary; Soumya checks sampling reproducibility and timed-path semantics. Run both new and original tests, and check Git provenance.
- **By 2:00 PM:** team integrates real results and methodology into **Research** Workbook Chapters 6 and 7; Chapter 8 updates individually attributed progress with **all newly changed text bold** relative to Workbook 1; Chapter 9 updates progress and includes detailed **Gantt and PERT dependency charts spanning both semesters**, individual owners, at least 10 granular tasks and milestone dependencies.
- **By 5:00 PM:** restore/retain prior Chapters 1–5 from Workbook 1, update claims in Chapters 3–5 only where new evidence justifies changes, cross-check table numbers with saved CSV/JSON and captions, export PDF.
- **By 8:00 PM:** visual PDF review (all pages, diagrams, tables, bold-change requirement, references), advisor-facing executive sanity check, final submission-ready PDF. **Submit before Thursday 8:00 AM Pacific; do not rely on the later 'available until' date.**

### Dependency / critical-path map

**Checkpoint availability → shared validation stochastic logits → validation-fitted temperature and stopping thresholds → locked test stochastic logits → matched policy evaluation → Workbook figures/tables → final PDF.**

In parallel without waiting: Sabari synthetic-fixture controller implementation; Workbook architecture/pseudocode writing; Gantt and PERT diagrams; Chapters 1–5 preservation; team role tracking. Sai exits after the checkpoint handoff; Soumya and Sabari own all subsequent integration.

### Failure and time-pressure fallbacks

- **V1 checkpoint unavailable:** Soumya attempts retrieval from the existing training machine/shared storage and uses `docs/run_registry.md` + checkpoint metadata. If irretrievably lost, a clearly named *new* checkpoint can be trained only if feasible; results must be identified as a different run, or fall back to V1 previously measured experiments and synthetic controller tests marked **no actual proposed-model test**. Do not make Sai retrain.
- **GPU not available:** first build and test controller using synthetic saved-logit fixtures; use CPU/full-test only if feasible. Evaluate on a predeclared stratified pilot sample and disclose coverage. Never pass off model-free synthetic arrays as real screening results.
- **Full T50 expensive:** checkpoint/resume per-image or per-shard; finish validation first, then stratified pilot test; once complete, extend to full test if GPU permits. Do not compare methods over different image subsets.
- **Calibration delayed:** perform clearly labeled uncalibrated comparisons, leave scalar absent, never claim calibrated confidence.
- **Too little time for MC-CP exact:** use named generic variance-convergence comparator; cite MC-CP as inspiration and say exact conformalized replication is future work.
- **Controller not ready:** still submit fixed T experiments plus V1 with precise design and a clear limitations log; do not fabricate risk-tier result charts.
- **Metrics disagree:** audit ID alignment, label map, softmax temperature order, dropout activation, checkpoint hash, split, and whether historical V1 vs new run preprocessing differs before drawing scientific conclusions.

---

## 8. Workbook 2 evidence-to-rubric mapping (what humans must submit)

| Rubric item (9 pts each) | Concrete evidence required from this plan | Common avoidable deduction |
| --- | --- | --- |
| **Chapter 6 — Evaluation Methodology** | Real V1 baseline; fixed T=10/20/50 measured experiment; frozen datasets/testbed/hyperparameters; statistical definitions; comparison table and figures; ideally controller vs variance-only method | Only listing intended metrics with no executed baseline; test-set threshold tuning; no sample counts |
| **Chapter 7 — System Design/Methodology** | Detailed seven-stage architecture; distinct four-tier state transitions; stopping algorithm pseudocode in the WORKBOOK; threshold logic; data/interface contracts; execution commands/provenance tutorial in reproduction doc; evidence of runnable early prototype | Architecture is merely a generic block diagram; no reproducible steps or tested controller |
| **Chapter 8 — Implementation Plan and Progress** | Dated implementation ledger: V1 evidence, new WB2 experiment PRs/tests/checkpoint hashes, what failed, individual ownership (Soumya/Sabari/Sai) and remaining planned tasks; **bold all changed text from the prior submission** | Claiming planned tasks are complete; vague roles; failing bold-change instruction |
| **Chapter 9 — Project Schedule** | **Gantt + PERT/dependency graph for both semesters**; clearly identified per-person task owners; dependencies, dates/versions, ≥10 granular tasks, critical path and updated status; **bold changes** | Only a version-level 5-row table or a first-semester Gantt; no PERT or task ownership |

**Rubric reality check:** No single experiment guarantees 36/36. The grading also explicitly requires a detailed clear design, honest implementation progress, and properly laid-out **two-semester** Gantt + PERT charts. Code completion is only one part of the PDF submission.

## 9. Exact evidence package checklist (do not stop with console output)

- [ ] Checkpoint is verifiably loadable; SHA256, original V1 run ID and selected checkpoint epoch recorded.
- [ ] Same V1 grouped split and preprocessing used across all *new* methods; test is untouched during tuning.
- [ ] Current branch commit hash, software versions, hardware, random seeds and running configuration recorded.
- [ ] Deterministic T=1 **and** stochastic fixed T=10/20/50 are clearly distinguished.
- [ ] Real prediction bank has finite logits, correct row counts, class alignment and ordered pass indices.
- [ ] Fixed T methods use identical prefixes of the same stochastic 50-draw bank.
- [ ] Two controller policies have actual unit tests, seeded fixtures and failure-case behavior.
- [ ] Temperature is fitted on validation only; uncalibrated results are labeled when relevant.
- [ ] Risk-tier/stop/review thresholds were frozen on validation **before** test reporting.
- [ ] Mean passes, referral/accepted rates and melanoma miss **counts and denominators** are included.
- [ ] The end-to-end latency claim differentiates backbone passes from inexpensive head passes.
- [ ] Results and comparisons have tables, legible plots, captions, sample sizes and limitations.
- [ ] Existing V1 numbers are not described as results of proposed risk-aware inference.
- [ ] No omitted or failed experiment is silently represented as complete.
- [ ] Original V1 unit tests and new WB2 tests have logged results (failures included).
- [ ] The team has human-reviewed Git identity/trailers; **no AI agent listed as author or coauthor**.
- [ ] Workbook uses the research template, keeps Chapters 1–5 intact/updated, completes Chapters 6–9.
- [ ] Chapters 8 and 9 clearly **bold additions/changes** relative to Workbook 1.
- [ ] Gantt **and** PERT span two semesters with at least 10 tasks and individual assignments.
- [ ] Final PDF opens correctly, renders cleanly, contains actual figures and is uploaded before the deadline.

---

## 10. Source and reference links for agents

**Repository (current main; inspect before editing):**

- Repository root: https://github.com/SabariIyyappan/Adaptive-SafeDerm
- Existing replication plan and status: https://github.com/SabariIyyappan/Adaptive-SafeDerm/blob/main/PLAN.md
- Data/split contract: https://github.com/SabariIyyappan/Adaptive-SafeDerm/blob/main/CONTRACTS.md
- Checkpoint/model definition: https://github.com/SabariIyyappan/Adaptive-SafeDerm/blob/main/src/model/model.py
- Existing deterministic predictor: https://github.com/SabariIyyappan/Adaptive-SafeDerm/blob/main/src/model/predict.py
- Existing evaluation entry point: https://github.com/SabariIyyappan/Adaptive-SafeDerm/blob/main/src/eval/run_eval.py
- V1 config: https://github.com/SabariIyyappan/Adaptive-SafeDerm/blob/main/configs/v1-b0-baseline.yaml
- V1 actual results: https://github.com/SabariIyyappan/Adaptive-SafeDerm/blob/main/results/v1/summary.md
- Run provenance: https://github.com/SabariIyyappan/Adaptive-SafeDerm/blob/main/docs/run_registry.md
- Recorded deviations: https://github.com/SabariIyyappan/Adaptive-SafeDerm/blob/main/docs/deviations.md

**Scientific / data references:**

- Primary replicated paper, Randhawa et al. (2026): https://doi.org/10.3390/biomedinformatics6040053
- Prior art (not to misrepresent): Bethell, Gerasimou & Calinescu, *Robust Uncertainty Quantification Using Conformalised Monte Carlo Prediction*, AAAI 2024: https://doi.org/10.1609/aaai.v38i19.30084
- Authors' MC-CP reference implementation: https://github.com/team-daniel/MC-CP (read before claiming exact MC-CP reproduction)
- Original HAM10000 dataset paper: https://doi.org/10.1038/sdata.2018.161
- HAM10000 Dataverse (as listed in Workbook 1): https://doi.org/10.7910/DVN/DBW86T
- Temperature scaling foundation (Guo et al., ICML 2017): https://proceedings.mlr.press/v70/guo17a.html
- Grad-CAM++ foundation (Chattopadhyay et al., WACV 2018): https://doi.org/10.1109/WACV.2018.00097
- Calibrated selective referral example (Yu, 2026; as recorded in the project pitch): https://doi.org/10.1038/s41598-026-64721-3
- PyTorch Dropout behavior (important for inference semantics): https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- PyTorch BatchNorm behavior: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

**Assignment source of truth:** uploaded `Workbook1_AdaptiveSafeDerm.pdf` (previous submission), `Project Workbook Template - research v2.docx` (Research template), project abstract/pitch and the current Canvas rubric. The Canvas upload is **PDF only**; the later Canvas availability date does not override the Thursday 8 AM due time.

**End of implementation plan.** The file intentionally contains **no executable source code, shell commands or pseudocode blocks**; coding agents must implement and document their own code in the repository after inspecting current files, preserving the responsibilities and acceptance gates above.

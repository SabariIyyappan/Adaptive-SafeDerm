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

### V1 — Real dataset not yet acquired in this environment
**Section:** §3.1 / PLAN.md §5.3.1
**What:** The environment this V1 pipeline code was developed in has no network route to Kaggle (`kaggle.com`) or the Harvard Dataverse (`dataverse.harvard.edu`) — both return policy-denied (403) at the network egress layer.
**What we did:** Wrote and unit-tested the full acquisition → integrity-audit → split → verify → EDA pipeline against a synthetic fixture that mirrors HAM10000's shape (7 classes, realistic imbalance, lesion-to-image ratios). `acquire.py` is real, working code (Kaggle CLI primary, Dataverse fallback) that has not yet been run against the real dataset. `data/README.md` has the exact commands to run once on a machine with normal internet access.
**Why:** So the split/audit/EDA logic is verified correct (leakage checks, stratification, class coverage) independent of when/where the real download happens, per the plan's own "dummy-first" rule (§4.4).

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
**What we did:** Not yet applicable to Member A's scope (owned by Member B — PLAN.md §5.3 Member B task 5). Logged here as a placeholder pointer; B's actual provisional values (AdamW, LR 1e-4, WD 1e-4, batch 32) belong in this file once B's training code lands.

### V1 — Real dataset not yet acquired in this environment
**Section:** §3.1 / PLAN.md §5.3.1
**What:** The environment this V1 pipeline code was developed in has no network route to Kaggle (`kaggle.com`) or the Harvard Dataverse (`dataverse.harvard.edu`) — both return policy-denied (403) at the network egress layer.
**What we did:** Wrote and unit-tested the full acquisition → integrity-audit → split → verify → EDA pipeline against a synthetic fixture that mirrors HAM10000's shape (7 classes, realistic imbalance, lesion-to-image ratios). `acquire.py` is real, working code (Kaggle CLI primary, Dataverse fallback) that has not yet been run against the real dataset. `data/README.md` has the exact commands to run once on a machine with normal internet access.
**Why:** So the split/audit/EDA logic is verified correct (leakage checks, stratification, class coverage) independent of when/where the real download happens, per the plan's own "dummy-first" rule (§4.4).

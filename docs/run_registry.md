# run_registry.md — every training/inference run, one row each

PLAN.md §4.5: who ran it, where, GPU type, duration, git commit, final metrics.

| run_id | who | where | GPU | duration | git commit | final metrics | notes |
|---|---|---|---|---|---|---|---|
| `v1-b0-baseline` | Member B | Sai Teja's laptop (local) | NVIDIA RTX 3060 Laptop, 6GB VRAM | ~35 min (30 epochs) | `a919142` | best epoch 30/30 (no early stopping); val macro F1 0.5839, val macro AUROC 0.9233 (single-pass); test-set predictions exported | Real HAM10000 (10,015 images), frozen C2 split from A's pipeline run the same session. Provisional Table 2 hyperparameters (`configs/v1-b0-baseline.yaml`). No preprocessing (hair removal/CLAHE is V2); single-pass eval (MC Dropout is V3) — so these numbers are not directly comparable to the paper's final T=50 headline (0.9404 AUROC / 0.7308 F1), only to its V1-equivalent single-pass baseline. |

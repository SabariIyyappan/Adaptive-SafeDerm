# run_registry.md — every training/inference run, one row each

PLAN.md §4.5: who ran it, where, GPU type, duration, git commit, final metrics.

| run_id | who | where | GPU | duration | git commit | final metrics | notes |
|---|---|---|---|---|---|---|---|
| `v1-b0-baseline` | Member B | Sai Teja's laptop (local) | NVIDIA RTX 3060 Laptop, 6GB VRAM | ~35 min (30 epochs) | `a919142` | best epoch 30/30 (no early stopping); **val** macro F1 0.5839, macro AUROC 0.9233; **test** macro F1 0.6213 [0.5795, 0.6590], macro AUROC 0.9264 [0.9098, 0.9405], balanced acc 0.7271, ECE 0.2219, Brier 0.5596 (all single-pass) | Real HAM10000 (10,015 images), frozen C2 split from A's pipeline run the same session. Provisional Table 2 hyperparameters (`configs/v1-b0-baseline.yaml`). No preprocessing (hair removal/CLAHE is V2); single-pass eval (MC Dropout is V3) — so these numbers are not directly comparable to the paper's final T=50 headline (0.9404 AUROC / 0.7308 F1), only to its V1-equivalent single-pass baseline. |

## Evaluation runs

| eval of | who | inputs | seed | outputs | notes |
|---|---|---|---|---|---|
| `v1-b0-baseline` | Member C | `results/v1/inputs/` (committed C3/C5/C6) | 42 | `results/v1/`: 13 figures, 18 EDA figures, 6 tables, `metrics.json`, `summary.md`, `data_report.md` | `python -m src.eval.run_eval` + `python -m src.eval.eda`. Bootstrap 1000 resamples, 0 skipped. All 4 PLAN.md §5.6 sanity checks pass. Test metrics above come from this run. |

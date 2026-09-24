# `results/v1/` — V1 result package

Everything the V1 professor demo needs (PLAN.md §5.6), regenerated from
committed inputs by two commands.

## Contents

| Path | What it is | Produced by |
|---|---|---|
| `summary.md` | **Start here.** Headline metrics vs paper, per-class results, split-fairness verdict, the melanoma diagnosis, sanity checks | C |
| `data_report.md` | Integrity audit, split verification and EDA findings — all recomputed, not copied | C |
| `metrics.json` | Every number, machine-readable | C |
| `figures/` | 13 evaluation figures | C |
| `eda/` | 18 dataset figures + summary tables | C |
| `tables/` | 6 result tables (.md and .csv) | C |
| `inputs/` | The committed C3/C5/C6 artifacts the suite reads ([provenance](inputs/README.md)) | B → C |
| `architecture_figure.png` | Paper Figure 1 equivalent | A |

## Reproducing

```bash
pip install -r env/requirements.txt

# Evaluation — needs nothing but this repo (no dataset, GPU or checkpoint):
python -m src.eval.run_eval

# EDA — the image-content figures need the dataset:
python -m src.data.acquire --dest data/raw --source dataverse   # ~3.2 GB, no credentials needed
python -m src.eval.eda
```

`run_eval` reads only `inputs/`, so the headline numbers, all 13 evaluation
figures and every table regenerate on a clean clone. `eda` additionally needs
`data/raw/images/`; without it the metadata- and split-based figures are still
produced and the image-content ones are skipped with a message.

Tests: `python -m unittest discover -s tests` (53 pass; `tests/test_model.py`
additionally needs `torch`, which Member C's CPU-only environment does not have).

## Headline results (test split, single pass)

| Metric | Ours | Paper (T=50, with preprocessing) |
|---|---|---|
| Macro AUROC | **0.9264** [0.9098, 0.9405] | 0.9404 |
| Macro F1 | **0.6213** [0.5795, 0.6590] | 0.7308 |
| Balanced accuracy | 0.7271 | not reported |
| ECE (10 bins) | 0.2219 | 0.1891 (T=1) · 0.1456 (T=50) |

Not a like-for-like comparison by design: V1 omits hair removal + CLAHE (V2)
and MC Dropout averaging (V3), and uses provisional Table 2 hyperparameters.
See `summary.md` §5 for each gap and its attributable cause.

## Key findings

1. **PLAN.md §13 item 4 is resolved.** The paper's per-class "accuracy" is
   Eq. 17's one-vs-rest form, not recall — `nv` scores 0.708 under Eq. 17 and
   0.567 under recall against the paper's reported 0.698.
2. **The split is unbiased**, measured rather than asserted: class proportions
   deviate from train by ≤0.081 pp at the lesion level and ≤0.806 pp at the
   image level, with zero lesion/image overlap and zero pixel-confirmed
   duplicate leakage.
3. **The melanoma over-prediction is the paper's design working as specified**,
   not a training defect — the two imbalance corrections compound to give
   melanoma ~36× nevus's effective weight. Errors run in the clinically safe
   direction.

## "Done when" (PLAN.md §5.3)

- [x] Integrity audit, split, split verification and EDA implemented and unit-tested
- [x] Split files pass every verification check against the real 10,015-image dataset
- [x] B confirmed the real split files load and train correctly (`v1-b0-baseline`)
- [x] C confirmed the real split files load for evaluation
- [x] Every figure and table regenerates from real predictions with one command
- [x] All four §5.6 sanity checks pass (recorded in `summary.md` §6 and `metrics.json`)

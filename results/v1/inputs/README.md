# `results/v1/inputs/` — committed evaluation inputs

These four files are the **only** inputs the V1 evaluation suite needs. They are
committed (unlike everything else under `runs/`, which `.gitignore` excludes)
so that `python -m src.eval.run_eval` regenerates every figure, table and metric
from a clean clone — no dataset, no GPU, no checkpoint required. That is what
PLAN.md §12 means by "regenerates from stored predictions with one command".

Total size ~370 KB.

| File | Contract | Contents |
|---|---|---|
| `predictions_test.csv` | C3 | 1,481 rows — `image_id, split, y_true, prob_0..prob_6, run_id` |
| `predictions_val.csv` | C3 | 1,532 rows, same schema |
| `history.csv` | C5 | 30 rows — `epoch, train_loss, val_loss, val_macro_f1, val_macro_auroc, lr` |
| `checkpoint.json` | C6 | `run_id`, best epoch, selection metric, config reference, checkpoint sha256 |

## Provenance

Copied verbatim from `runs/v1-b0-baseline/`, produced by Member B.

- **Run ID:** `v1-b0-baseline`
- **Produced by git commit:** `a919142` (*Member B V1: model, training loop, imbalance correction, C3 export*)
- **Training hardware:** NVIDIA RTX 3060 Laptop, 6 GB VRAM, ~35 min for 30 epochs
- **Config:** `configs/v1-b0-baseline.yaml` (provisional Table 2 hyperparameters — see `docs/deviations.md`)
- **Best epoch:** 30/30, selected by validation macro F1 = 0.5839484357569342
- **Source checkpoint sha256:** `901a73a6a689cfb4617d3897152fcb33c73d40bbbf190978218fff4614f9ae0d`
  (the `best.pt` these predictions came from; the weights themselves stay out of git)

The full run record is in `docs/run_registry.md`.

## Validation on load

`src.eval.metrics.load_predictions` re-checks the C3 contract every time it
reads these files: required columns present, row count against the split size,
every probability row summing to 1.0 (atol 1e-4), and all labels inside 0–6.
A corrupted or truncated file fails loudly rather than producing a
plausible-looking but wrong figure.

## Regenerating them

These files came from a trained checkpoint. To rebuild them you need the
dataset and `best.pt`:

```bash
python -m src.model.predict --split-dir data/splits --images-dir data/raw/images \
    --checkpoint runs/v1-b0-baseline/best.pt --split test --run-id v1-b0-baseline \
    --out runs/v1-b0-baseline/predictions_test.csv
```

You do **not** need to do this to reproduce the V1 results — that is the point
of committing them here.

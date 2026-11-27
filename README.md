# Adaptive SafeDerm

Risk-aware adaptive MC-Dropout for dermoscopic skin-lesion classification (CMPE 295, SJSU).
Research results: `results/wb2/summary.md`.

> **Research prototype, not a diagnosis.** "TRUSTED RESULT" means the model's own checks passed. It never means benign or safe.

## Run the app

```bash
pip install -r env/requirements.txt
streamlit run app/streamlit_app.py
```

The app has three pages:

| Page | Needs the checkpoint? | What it shows |
|---|---|---|
| **Upload & Analyze** (`app/streamlit_app.py`) | yes, for live results | Upload an image or pick an example; shows the TRUSTED / NEEDS REVIEW decision, reasons, tier, calibrated probabilities, Grad-CAM, passes used and latency |
| **Evaluation** (`app/pages/2_Evaluation.py`) | no | Metric cards, method comparison, compute vs safety, coverage vs risk, stop passes, tiers, reliability, confusion matrix, ROC, melanoma audit |
| **Overview** (`app/pages/3_Overview.py`) | no | Pipeline diagram, paper vs ours, figure gallery, next steps |

**Checkpoint.** The V1 `best.pt` (SHA256 `901a73a6…f9ae0d`) is shared privately and is never committed. Point the app at it either way:

```bash
export SAFEDERM_CKPT=/path/to/best.pt        # Windows PowerShell: $env:SAFEDERM_CKPT="C:\path\best.pt"
```

or paste the path into the sidebar field. Without a checkpoint, page 1 shows a clearly labelled **STUB** result and pages 2–3 work normally.

Every number on pages 2–3 is read from files in `results/wb2/`, `configs/wb2_controller.yaml` and `app/data/` (provenance in `app/data/SOURCES.md`); each chart names its source file.

# V1 results deck — outline (PLAN.md §5.3 item 10, §5.6)

Member A owns the dataset/split/architecture slides (1–3, 9 partial, 10) and drops in C's figures for the rest (4–8). This is a slide-by-slide outline, not the final deck — export to PDF once real numbers land (PLAN.md §5.6 lists the full 10-slide package).

1. **Title** — Replicating "Skin Lesion Classification in Low-Resource Settings..." (Randhawa et al., BioMedInformatics 2026) — V1: trained baseline + core evaluation.
2. **Dataset overview** — HAM10000, 10,015 images, 7 classes, ~58:1 imbalance (nv vs df) by count. Insert `eda/class_distribution.png`, `eda/images_per_lesion_hist.png`, `eda/sample_grid.png`.
3. **Split** — lesion-grouped 70/15/15, stratified by dx, seed 42. Insert the "Ours vs Paper" table from `split_verification.md`, and state the leakage check result ("0 shared lesions" — PLAN.md §5.6 item 2).
4. *(C)* Architecture figure — `architecture_figure.png` (done, no dependency on C).
5. *(C)* Training curves.
6. *(C)* Headline test metrics with 95% CI: macro AUROC, macro F1.
7. *(C)* Per-class table (accuracy, recall, specificity) + bar chart.
8. *(C)* Confusion matrix.
9. *(C)* ROC curves.
10. **"Ours vs Paper"** — clearly labelled: paper = preprocessing + T=50; V1 = no preprocessing, single pass.
11. **Roadmap** — V2 (imbalance ablation) → V3 (MC Dropout + referral) → V4 (Grad-CAM + IoU) → V5 (comparative baselines + final report).

Slide 4's figure is ready now (`architecture_figure.png`). Slides 2–3's figures/tables are ready to generate the moment real data is acquired (`data/README.md`). Slides 5–9 need Member B's training run + Member C's evaluation suite output before they can be filled in.

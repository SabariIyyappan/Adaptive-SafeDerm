# WB2 controller report — test (MEASURED)

Generated 2026-10-28T06:09:38+00:00 by `python -m src.wb2.analysis.run_wb2`. Auto-generated — the curated
Workbook text lives in `results/wb2/summary.md`.

- Images: **1481**; class counts {'akiec': 46, 'bcc': 71, 'bkl': 168, 'df': 20, 'mel': 165, 'nv': 992, 'vasc': 19}
- Checkpoint SHA256: `901a73a6a689cfb4617d3897152fcb33c73d40bbbf190978218fff4614f9ae0d`
- Calibration: `uncalibrated`; config tuned on val at 2026-10-28T06:01:41+00:00 (risk-aware feasible on val: True)
- Feature reuse in sampling: True — pass counts are stochastic *head* draws; savings are algorithmic (offline replay), not measured wall-clock.
- TRUSTED RESULT means the model's computational checks passed, never 'clinically safe/benign'. Explanation verifier: NOT RUN (planned V4).

## Method comparison

| method              | status   |    n |   macro_auroc |   macro_f1 |   balanced_accuracy |   ece_10bin |   brier |    nll |   mel_recall |   suspicious_recall |   passes_mean |   passes_median |   passes_p95 |   pct_at_cap |   review_rate |   coverage |   accepted_accuracy | mel_miss_all_str   | mel_miss_accepted_str   | suspicious_false_reassurance_str   |
|:--------------------|:---------|-----:|--------------:|-----------:|--------------------:|------------:|--------:|-------:|-------------:|--------------------:|--------------:|----------------:|-------------:|-------------:|--------------:|-----------:|--------------------:|:-------------------|:------------------------|:-----------------------------------|
| det_T1              | MEASURED | 1481 |        0.9264 |     0.6232 |              0.7302 |      0.2213 |  0.5595 | 1.1597 |       0.7758 |              0.7966 |        1.0000 |          1.0000 |       1.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6226 | 37/165             | 37/165                  | 48/236                             |
| fixed_T10           | MEASURED | 1481 |        0.9261 |     0.6129 |              0.7189 |      0.2194 |  0.5545 | 1.1340 |       0.7697 |              0.7924 |       10.0000 |         10.0000 |      10.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6165 | 38/165             | 38/165                  | 49/236                             |
| fixed_T20           | MEASURED | 1481 |        0.9261 |     0.6196 |              0.7291 |      0.2152 |  0.5545 | 1.1336 |       0.7697 |              0.7924 |       20.0000 |         20.0000 |      20.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6205 | 38/165             | 38/165                  | 49/236                             |
| fixed_T50           | MEASURED | 1481 |        0.9273 |     0.6219 |              0.7275 |      0.2156 |  0.5540 | 1.1297 |       0.7818 |              0.8008 |       50.0000 |         50.0000 |      50.0000 |       1.0000 |        0.0000 |     1.0000 |              0.6205 | 36/165             | 36/165                  | 47/236                             |
| fixed_T50+referral  | MEASURED | 1481 |        0.9273 |     0.6219 |              0.7275 |      0.2156 |  0.5540 | 1.1297 |       0.7818 |              0.8008 |       50.0000 |         50.0000 |      50.0000 |       1.0000 |        0.1843 |     0.8157 |              0.6813 | 27/165             | 27/135                  | 36/236                             |
| variance_adaptive   | MEASURED | 1481 |        0.9261 |     0.6129 |              0.7189 |      0.2194 |  0.5545 | 1.1340 |       0.7697 |              0.7924 |       10.0000 |         10.0000 |      10.0000 |       0.0000 |        0.1924 |     0.8076 |              0.6831 | 28/165             | 28/135                  | 37/236                             |
| risk_aware_adaptive | MEASURED | 1481 |        0.9261 |     0.6134 |              0.7200 |      0.2183 |  0.5548 | 1.1343 |       0.7818 |              0.8008 |       18.2714 |         10.0000 |      50.0000 |       0.1121 |        0.1121 |     0.8879 |              0.6251 | 22/165             | 22/151                  | 26/236                             |

## Paired lesion-grouped bootstrap (b − a, 95% percentile CI)

| metric       | method_a           | method_b            |       a |       b |   diff_b_minus_a |   ci_low |   ci_high | ci_excludes_zero   |   n_resamples_used |   n_skipped_missing_class | resampling_unit   |   seed |
|:-------------|:-------------------|:--------------------|--------:|--------:|-----------------:|---------:|----------:|:-------------------|-------------------:|--------------------------:|:------------------|-------:|
| macro_auroc  | fixed_T50          | risk_aware_adaptive |  0.9273 |  0.9261 |          -0.0012 |  -0.0024 |   -0.0002 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50          | risk_aware_adaptive |  0.6219 |  0.6134 |          -0.0085 |  -0.0190 |   -0.0004 | True               |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50          | risk_aware_adaptive |  0.2156 |  0.2183 |           0.0026 |  -0.0017 |    0.0057 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50          | risk_aware_adaptive |  0.5540 |  0.5548 |           0.0008 |  -0.0002 |    0.0020 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50          | risk_aware_adaptive | 50.0000 | 18.2714 |         -31.7286 | -32.4266 |  -31.0314 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50          | risk_aware_adaptive |  0.0000 |  0.1121 |           0.1121 |   0.0951 |    0.1307 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50          | risk_aware_adaptive |  0.2182 |  0.1333 |          -0.0848 |  -0.1437 |   -0.0384 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | fixed_T50          | variance_adaptive   |  0.9273 |  0.9261 |          -0.0012 |  -0.0025 |   -0.0002 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50          | variance_adaptive   |  0.6219 |  0.6129 |          -0.0090 |  -0.0194 |    0.0000 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50          | variance_adaptive   |  0.2156 |  0.2194 |           0.0038 |  -0.0021 |    0.0085 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50          | variance_adaptive   |  0.5540 |  0.5545 |           0.0005 |  -0.0012 |    0.0022 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50          | variance_adaptive   | 50.0000 | 10.0000 |         -40.0000 | -40.0000 |  -40.0000 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50          | variance_adaptive   |  0.0000 |  0.1924 |           0.1924 |   0.1702 |    0.2154 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50          | variance_adaptive   |  0.2182 |  0.1697 |          -0.0485 |  -0.0844 |   -0.0174 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | fixed_T50+referral | risk_aware_adaptive |  0.9273 |  0.9261 |          -0.0012 |  -0.0024 |   -0.0002 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50+referral | risk_aware_adaptive |  0.6219 |  0.6134 |          -0.0085 |  -0.0190 |   -0.0004 | True               |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50+referral | risk_aware_adaptive |  0.2156 |  0.2183 |           0.0026 |  -0.0017 |    0.0057 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50+referral | risk_aware_adaptive |  0.5540 |  0.5548 |           0.0008 |  -0.0002 |    0.0020 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50+referral | risk_aware_adaptive | 50.0000 | 18.2714 |         -31.7286 | -32.4266 |  -31.0314 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50+referral | risk_aware_adaptive |  0.1843 |  0.1121 |          -0.0722 |  -0.0946 |   -0.0523 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50+referral | risk_aware_adaptive |  0.1636 |  0.1333 |          -0.0303 |  -0.0839 |    0.0068 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | variance_adaptive  | risk_aware_adaptive |  0.9261 |  0.9261 |           0.0000 |  -0.0004 |    0.0005 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | variance_adaptive  | risk_aware_adaptive |  0.6129 |  0.6134 |           0.0005 |  -0.0026 |    0.0036 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | variance_adaptive  | risk_aware_adaptive |  0.2194 |  0.2183 |          -0.0012 |  -0.0053 |    0.0033 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | variance_adaptive  | risk_aware_adaptive |  0.5545 |  0.5548 |           0.0003 |  -0.0011 |    0.0016 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | variance_adaptive  | risk_aware_adaptive | 10.0000 | 18.2714 |           8.2714 |   7.5734 |    8.9686 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | variance_adaptive  | risk_aware_adaptive |  0.1924 |  0.1121 |          -0.0804 |  -0.1023 |   -0.0601 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | variance_adaptive  | risk_aware_adaptive |  0.1697 |  0.1333 |          -0.0364 |  -0.0927 |    0.0064 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | det_T1             | fixed_T50           |  0.9264 |  0.9273 |           0.0009 |  -0.0001 |    0.0023 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | det_T1             | fixed_T50           |  0.6232 |  0.6219 |          -0.0013 |  -0.0057 |    0.0028 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | det_T1             | fixed_T50           |  0.2213 |  0.2156 |          -0.0057 |  -0.0117 |   -0.0006 | True               |               1000 |                         0 | lesion_id         |     42 |
| brier        | det_T1             | fixed_T50           |  0.5595 |  0.5540 |          -0.0056 |  -0.0067 |   -0.0044 | True               |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | det_T1             | fixed_T50           |  1.0000 | 50.0000 |          49.0000 |  49.0000 |   49.0000 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | det_T1             | fixed_T50           |  0.0000 |  0.0000 |           0.0000 |   0.0000 |    0.0000 | False              |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | det_T1             | fixed_T50           |  0.2242 |  0.2182 |          -0.0061 |  -0.0196 |    0.0000 | False              |               1000 |                         0 | lesion_id         |     42 |

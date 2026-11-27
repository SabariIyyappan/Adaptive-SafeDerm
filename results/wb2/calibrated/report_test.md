# WB2 controller report — test (MEASURED)

Generated 2026-10-28T06:07:42+00:00 by `python -m src.wb2.analysis.run_wb2`. Auto-generated — the curated
Workbook text lives in `results/wb2/summary.md`.

- Images: **1481**; class counts {'akiec': 46, 'bcc': 71, 'bkl': 168, 'df': 20, 'mel': 165, 'nv': 992, 'vasc': 19}
- Checkpoint SHA256: `901a73a6a689cfb4617d3897152fcb33c73d40bbbf190978218fff4614f9ae0d`
- Calibration: `wb2-mc-v1b0-s42-temp-val`; config tuned on val at 2026-10-28T06:01:05+00:00 (risk-aware feasible on val: True)
- Feature reuse in sampling: True — pass counts are stochastic *head* draws; savings are algorithmic (offline replay), not measured wall-clock.
- TRUSTED RESULT means the model's computational checks passed, never 'clinically safe/benign'. Explanation verifier: NOT RUN (planned V4).

## Method comparison

| method              | status   |    n |   macro_auroc |   macro_f1 |   balanced_accuracy |   ece_10bin |   brier |    nll |   mel_recall |   suspicious_recall |   passes_mean |   passes_median |   passes_p95 |   pct_at_cap |   review_rate |   coverage |   accepted_accuracy | mel_miss_all_str   | mel_miss_accepted_str   | suspicious_false_reassurance_str   |
|:--------------------|:---------|-----:|--------------:|-----------:|--------------------:|------------:|--------:|-------:|-------------:|--------------------:|--------------:|----------------:|-------------:|-------------:|--------------:|-----------:|--------------------:|:-------------------|:------------------------|:-----------------------------------|
| det_T1              | MEASURED | 1481 |        0.9300 |     0.6232 |              0.7302 |      0.0576 |  0.4854 | 0.9135 |       0.7758 |              0.7966 |        1.0000 |          1.0000 |       1.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6226 | 37/165             | 37/165                  | 48/236                             |
| fixed_T10           | MEASURED | 1481 |        0.9297 |     0.6135 |              0.7198 |      0.0578 |  0.4849 | 0.9141 |       0.7697 |              0.7924 |       10.0000 |         10.0000 |      10.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6172 | 38/165             | 38/165                  | 49/236                             |
| fixed_T20           | MEASURED | 1481 |        0.9298 |     0.6194 |              0.7289 |      0.0562 |  0.4852 | 0.9150 |       0.7697 |              0.7924 |       20.0000 |         20.0000 |      20.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6199 | 38/165             | 38/165                  | 49/236                             |
| fixed_T50           | MEASURED | 1481 |        0.9304 |     0.6219 |              0.7275 |      0.0568 |  0.4849 | 0.9137 |       0.7818 |              0.8008 |       50.0000 |         50.0000 |      50.0000 |       1.0000 |        0.0000 |     1.0000 |              0.6205 | 36/165             | 36/165                  | 47/236                             |
| fixed_T50+referral  | MEASURED | 1481 |        0.9304 |     0.6219 |              0.7275 |      0.0568 |  0.4849 | 0.9137 |       0.7818 |              0.8008 |       50.0000 |         50.0000 |      50.0000 |       1.0000 |        0.1891 |     0.8109 |              0.6736 | 28/165             | 28/135                  | 37/236                             |
| variance_adaptive   | MEASURED | 1481 |        0.9297 |     0.6135 |              0.7198 |      0.0578 |  0.4849 | 0.9141 |       0.7697 |              0.7924 |       10.0000 |         10.0000 |      10.0000 |       0.0000 |        0.1877 |     0.8123 |              0.6692 | 29/165             | 29/135                  | 38/236                             |
| risk_aware_adaptive | MEASURED | 1481 |        0.9297 |     0.6134 |              0.7200 |      0.0594 |  0.4854 | 0.9150 |       0.7818 |              0.8008 |       18.3153 |         10.0000 |      50.0000 |       0.1134 |        0.1134 |     0.8866 |              0.6253 | 22/165             | 22/149                  | 26/236                             |

## Paired lesion-grouped bootstrap (b − a, 95% percentile CI)

| metric       | method_a           | method_b            |       a |       b |   diff_b_minus_a |   ci_low |   ci_high | ci_excludes_zero   |   n_resamples_used |   n_skipped_missing_class | resampling_unit   |   seed |
|:-------------|:-------------------|:--------------------|--------:|--------:|-----------------:|---------:|----------:|:-------------------|-------------------:|--------------------------:|:------------------|-------:|
| macro_auroc  | fixed_T50          | risk_aware_adaptive |  0.9304 |  0.9297 |          -0.0007 |  -0.0015 |   -0.0000 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50          | risk_aware_adaptive |  0.6219 |  0.6134 |          -0.0085 |  -0.0190 |   -0.0004 | True               |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50          | risk_aware_adaptive |  0.0568 |  0.0594 |           0.0026 |  -0.0059 |    0.0097 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50          | risk_aware_adaptive |  0.4849 |  0.4854 |           0.0004 |  -0.0003 |    0.0012 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50          | risk_aware_adaptive | 50.0000 | 18.3153 |         -31.6847 | -32.3539 |  -30.9886 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50          | risk_aware_adaptive |  0.0000 |  0.1134 |           0.1134 |   0.0970 |    0.1314 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50          | risk_aware_adaptive |  0.2182 |  0.1333 |          -0.0848 |  -0.1437 |   -0.0384 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | fixed_T50          | variance_adaptive   |  0.9304 |  0.9297 |          -0.0008 |  -0.0017 |   -0.0000 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50          | variance_adaptive   |  0.6219 |  0.6135 |          -0.0084 |  -0.0191 |    0.0003 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50          | variance_adaptive   |  0.0568 |  0.0578 |           0.0010 |  -0.0080 |    0.0108 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50          | variance_adaptive   |  0.4849 |  0.4849 |          -0.0000 |  -0.0010 |    0.0010 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50          | variance_adaptive   | 50.0000 | 10.0000 |         -40.0000 | -40.0000 |  -40.0000 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50          | variance_adaptive   |  0.0000 |  0.1877 |           0.1877 |   0.1648 |    0.2096 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50          | variance_adaptive   |  0.2182 |  0.1758 |          -0.0424 |  -0.0802 |   -0.0075 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | fixed_T50+referral | risk_aware_adaptive |  0.9304 |  0.9297 |          -0.0007 |  -0.0015 |   -0.0000 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50+referral | risk_aware_adaptive |  0.6219 |  0.6134 |          -0.0085 |  -0.0190 |   -0.0004 | True               |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50+referral | risk_aware_adaptive |  0.0568 |  0.0594 |           0.0026 |  -0.0059 |    0.0097 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50+referral | risk_aware_adaptive |  0.4849 |  0.4854 |           0.0004 |  -0.0003 |    0.0012 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50+referral | risk_aware_adaptive | 50.0000 | 18.3153 |         -31.6847 | -32.3539 |  -30.9886 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50+referral | risk_aware_adaptive |  0.1891 |  0.1134 |          -0.0756 |  -0.1002 |   -0.0511 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50+referral | risk_aware_adaptive |  0.1697 |  0.1333 |          -0.0364 |  -0.0933 |    0.0058 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | variance_adaptive  | risk_aware_adaptive |  0.9297 |  0.9297 |           0.0001 |  -0.0003 |    0.0005 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | variance_adaptive  | risk_aware_adaptive |  0.6135 |  0.6134 |          -0.0001 |  -0.0029 |    0.0026 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | variance_adaptive  | risk_aware_adaptive |  0.0578 |  0.0594 |           0.0016 |  -0.0065 |    0.0069 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | variance_adaptive  | risk_aware_adaptive |  0.4849 |  0.4854 |           0.0004 |  -0.0003 |    0.0012 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | variance_adaptive  | risk_aware_adaptive | 10.0000 | 18.3153 |           8.3153 |   7.6461 |    9.0114 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | variance_adaptive  | risk_aware_adaptive |  0.1877 |  0.1134 |          -0.0743 |  -0.0997 |   -0.0509 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | variance_adaptive  | risk_aware_adaptive |  0.1758 |  0.1333 |          -0.0424 |  -0.1007 |    0.0000 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | det_T1             | fixed_T50           |  0.9300 |  0.9304 |           0.0004 |  -0.0002 |    0.0011 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | det_T1             | fixed_T50           |  0.6232 |  0.6219 |          -0.0013 |  -0.0057 |    0.0028 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | det_T1             | fixed_T50           |  0.0576 |  0.0568 |          -0.0008 |  -0.0087 |    0.0054 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | det_T1             | fixed_T50           |  0.4854 |  0.4849 |          -0.0005 |  -0.0011 |    0.0001 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | det_T1             | fixed_T50           |  1.0000 | 50.0000 |          49.0000 |  49.0000 |   49.0000 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | det_T1             | fixed_T50           |  0.0000 |  0.0000 |           0.0000 |   0.0000 |    0.0000 | False              |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | det_T1             | fixed_T50           |  0.2242 |  0.2182 |          -0.0061 |  -0.0196 |    0.0000 | False              |               1000 |                         0 | lesion_id         |     42 |

# WB2 controller report — val (MEASURED)

Generated 2026-10-28T06:05:43+00:00 by `python -m src.wb2.analysis.run_wb2`. Auto-generated — the curated
Workbook text lives in `results/wb2/summary.md`.

- Images: **1532**; class counts {'akiec': 51, 'bcc': 77, 'bkl': 157, 'df': 19, 'mel': 170, 'nv': 1034, 'vasc': 24}
- Checkpoint SHA256: `901a73a6a689cfb4617d3897152fcb33c73d40bbbf190978218fff4614f9ae0d`
- Calibration: `uncalibrated`; config tuned on val at 2026-10-28T06:01:41+00:00 (risk-aware feasible on val: True)
- Feature reuse in sampling: True — pass counts are stochastic *head* draws; savings are algorithmic (offline replay), not measured wall-clock.
- TRUSTED RESULT means the model's computational checks passed, never 'clinically safe/benign'. Explanation verifier: NOT RUN (planned V4).

## Method comparison

| method              | status   |    n |   macro_auroc |   macro_f1 |   balanced_accuracy |   ece_10bin |   brier |    nll |   mel_recall |   suspicious_recall |   passes_mean |   passes_median |   passes_p95 |   pct_at_cap |   review_rate |   coverage |   accepted_accuracy | mel_miss_all_str   | mel_miss_accepted_str   | suspicious_false_reassurance_str   |
|:--------------------|:---------|-----:|--------------:|-----------:|--------------------:|------------:|--------:|-------:|-------------:|--------------------:|--------------:|----------------:|-------------:|-------------:|--------------:|-----------:|--------------------:|:-------------------|:------------------------|:-----------------------------------|
| det_T1              | MEASURED | 1532 |        0.9233 |     0.5836 |              0.6880 |      0.2322 |  0.5932 | 1.2012 |       0.7588 |              0.8016 |        1.0000 |          1.0000 |       1.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6031 | 41/170             | 41/170                  | 49/247                             |
| fixed_T10           | MEASURED | 1532 |        0.9250 |     0.5836 |              0.6858 |      0.2265 |  0.5878 | 1.1759 |       0.7588 |              0.8016 |       10.0000 |         10.0000 |      10.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6018 | 41/170             | 41/170                  | 49/247                             |
| fixed_T20           | MEASURED | 1532 |        0.9245 |     0.5823 |              0.6877 |      0.2262 |  0.5877 | 1.1729 |       0.7588 |              0.8016 |       20.0000 |         20.0000 |      20.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6018 | 41/170             | 41/170                  | 49/247                             |
| fixed_T50           | MEASURED | 1532 |        0.9247 |     0.5822 |              0.6863 |      0.2256 |  0.5873 | 1.1711 |       0.7588 |              0.8016 |       50.0000 |         50.0000 |      50.0000 |       1.0000 |        0.0000 |     1.0000 |              0.6025 | 41/170             | 41/170                  | 49/247                             |
| fixed_T50+referral  | MEASURED | 1532 |        0.9247 |     0.5822 |              0.6863 |      0.2256 |  0.5873 | 1.1711 |       0.7588 |              0.8016 |       50.0000 |         50.0000 |      50.0000 |       1.0000 |        0.1997 |     0.8003 |              0.6672 | 26/170             | 26/132                  | 36/247                             |
| variance_adaptive   | MEASURED | 1532 |        0.9250 |     0.5836 |              0.6858 |      0.2265 |  0.5878 | 1.1759 |       0.7588 |              0.8016 |       10.0000 |         10.0000 |      10.0000 |       0.0000 |        0.1997 |     0.8003 |              0.6648 | 24/170             | 24/133                  | 34/247                             |
| risk_aware_adaptive | MEASURED | 1532 |        0.9246 |     0.5809 |              0.6840 |      0.2277 |  0.5877 | 1.1738 |       0.7647 |              0.8057 |       18.8446 |         20.0000 |      50.0000 |       0.1221 |        0.1208 |     0.8792 |              0.6028 | 23/170             | 23/153                  | 29/247                             |

## Paired lesion-grouped bootstrap (b − a, 95% percentile CI)

| metric       | method_a           | method_b            |       a |       b |   diff_b_minus_a |   ci_low |   ci_high | ci_excludes_zero   |   n_resamples_used |   n_skipped_missing_class | resampling_unit   |   seed |
|:-------------|:-------------------|:--------------------|--------:|--------:|-----------------:|---------:|----------:|:-------------------|-------------------:|--------------------------:|:------------------|-------:|
| macro_auroc  | fixed_T50          | risk_aware_adaptive |  0.9247 |  0.9246 |          -0.0001 |  -0.0015 |    0.0012 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50          | risk_aware_adaptive |  0.5822 |  0.5809 |          -0.0013 |  -0.0063 |    0.0019 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50          | risk_aware_adaptive |  0.2256 |  0.2277 |           0.0022 |  -0.0035 |    0.0062 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50          | risk_aware_adaptive |  0.5873 |  0.5877 |           0.0004 |  -0.0008 |    0.0015 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50          | risk_aware_adaptive | 50.0000 | 18.8446 |         -31.1554 | -31.7964 |  -30.4577 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50          | risk_aware_adaptive |  0.0000 |  0.1208 |           0.1208 |   0.1043 |    0.1389 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50          | risk_aware_adaptive |  0.2412 |  0.1353 |          -0.1059 |  -0.1638 |   -0.0568 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | fixed_T50          | variance_adaptive   |  0.9247 |  0.9250 |           0.0002 |  -0.0013 |    0.0019 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50          | variance_adaptive   |  0.5822 |  0.5836 |           0.0014 |  -0.0063 |    0.0084 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50          | variance_adaptive   |  0.2256 |  0.2265 |           0.0009 |  -0.0046 |    0.0067 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50          | variance_adaptive   |  0.5873 |  0.5878 |           0.0004 |  -0.0011 |    0.0019 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50          | variance_adaptive   | 50.0000 | 10.0000 |         -40.0000 | -40.0000 |  -40.0000 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50          | variance_adaptive   |  0.0000 |  0.1997 |           0.1997 |   0.1781 |    0.2235 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50          | variance_adaptive   |  0.2412 |  0.1412 |          -0.1000 |  -0.1607 |   -0.0479 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | fixed_T50+referral | risk_aware_adaptive |  0.9247 |  0.9246 |          -0.0001 |  -0.0015 |    0.0012 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50+referral | risk_aware_adaptive |  0.5822 |  0.5809 |          -0.0013 |  -0.0063 |    0.0019 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50+referral | risk_aware_adaptive |  0.2256 |  0.2277 |           0.0022 |  -0.0035 |    0.0062 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50+referral | risk_aware_adaptive |  0.5873 |  0.5877 |           0.0004 |  -0.0008 |    0.0015 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50+referral | risk_aware_adaptive | 50.0000 | 18.8446 |         -31.1554 | -31.7964 |  -30.4577 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50+referral | risk_aware_adaptive |  0.1997 |  0.1208 |          -0.0790 |  -0.1014 |   -0.0553 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50+referral | risk_aware_adaptive |  0.1529 |  0.1353 |          -0.0176 |  -0.0690 |    0.0333 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | variance_adaptive  | risk_aware_adaptive |  0.9250 |  0.9246 |          -0.0004 |  -0.0012 |    0.0003 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | variance_adaptive  | risk_aware_adaptive |  0.5836 |  0.5809 |          -0.0027 |  -0.0092 |    0.0036 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | variance_adaptive  | risk_aware_adaptive |  0.2265 |  0.2277 |           0.0013 |  -0.0037 |    0.0042 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | variance_adaptive  | risk_aware_adaptive |  0.5878 |  0.5877 |          -0.0001 |  -0.0013 |    0.0011 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | variance_adaptive  | risk_aware_adaptive | 10.0000 | 18.8446 |           8.8446 |   8.2036 |    9.5423 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | variance_adaptive  | risk_aware_adaptive |  0.1997 |  0.1208 |          -0.0790 |  -0.1003 |   -0.0568 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | variance_adaptive  | risk_aware_adaptive |  0.1412 |  0.1353 |          -0.0059 |  -0.0575 |    0.0460 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | det_T1             | fixed_T50           |  0.9233 |  0.9247 |           0.0014 |   0.0002 |    0.0029 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | det_T1             | fixed_T50           |  0.5836 |  0.5822 |          -0.0014 |  -0.0044 |    0.0007 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | det_T1             | fixed_T50           |  0.2322 |  0.2256 |          -0.0066 |  -0.0101 |   -0.0036 | True               |               1000 |                         0 | lesion_id         |     42 |
| brier        | det_T1             | fixed_T50           |  0.5932 |  0.5873 |          -0.0058 |  -0.0071 |   -0.0047 | True               |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | det_T1             | fixed_T50           |  1.0000 | 50.0000 |          49.0000 |  49.0000 |   49.0000 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | det_T1             | fixed_T50           |  0.0000 |  0.0000 |           0.0000 |   0.0000 |    0.0000 | False              |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | det_T1             | fixed_T50           |  0.2412 |  0.2412 |           0.0000 |   0.0000 |    0.0000 | False              |               1000 |                         0 | lesion_id         |     42 |

# WB2 controller report — val (MEASURED)

Generated 2026-10-28T06:03:50+00:00 by `python -m src.wb2.analysis.run_wb2`. Auto-generated — the curated
Workbook text lives in `results/wb2/summary.md`.

- Images: **1532**; class counts {'akiec': 51, 'bcc': 77, 'bkl': 157, 'df': 19, 'mel': 170, 'nv': 1034, 'vasc': 24}
- Checkpoint SHA256: `901a73a6a689cfb4617d3897152fcb33c73d40bbbf190978218fff4614f9ae0d`
- Calibration: `wb2-mc-v1b0-s42-temp-val`; config tuned on val at 2026-10-28T06:01:05+00:00 (risk-aware feasible on val: True)
- Feature reuse in sampling: True — pass counts are stochastic *head* draws; savings are algorithmic (offline replay), not measured wall-clock.
- TRUSTED RESULT means the model's computational checks passed, never 'clinically safe/benign'. Explanation verifier: NOT RUN (planned V4).

## Method comparison

| method              | status   |    n |   macro_auroc |   macro_f1 |   balanced_accuracy |   ece_10bin |   brier |    nll |   mel_recall |   suspicious_recall |   passes_mean |   passes_median |   passes_p95 |   pct_at_cap |   review_rate |   coverage |   accepted_accuracy | mel_miss_all_str   | mel_miss_accepted_str   | suspicious_false_reassurance_str   |
|:--------------------|:---------|-----:|--------------:|-----------:|--------------------:|------------:|--------:|-------:|-------------:|--------------------:|--------------:|----------------:|-------------:|-------------:|--------------:|-----------:|--------------------:|:-------------------|:------------------------|:-----------------------------------|
| det_T1              | MEASURED | 1532 |        0.9267 |     0.5836 |              0.6880 |      0.0717 |  0.5061 | 0.9449 |       0.7588 |              0.8016 |        1.0000 |          1.0000 |       1.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6031 | 41/170             | 41/170                  | 49/247                             |
| fixed_T10           | MEASURED | 1532 |        0.9272 |     0.5837 |              0.6858 |      0.0618 |  0.5059 | 0.9461 |       0.7588 |              0.8016 |       10.0000 |         10.0000 |      10.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6018 | 41/170             | 41/170                  | 49/247                             |
| fixed_T20           | MEASURED | 1532 |        0.9270 |     0.5825 |              0.6879 |      0.0628 |  0.5059 | 0.9461 |       0.7588 |              0.8016 |       20.0000 |         20.0000 |      20.0000 |       0.0000 |        0.0000 |     1.0000 |              0.6025 | 41/170             | 41/170                  | 49/247                             |
| fixed_T50           | MEASURED | 1532 |        0.9271 |     0.5822 |              0.6863 |      0.0645 |  0.5056 | 0.9454 |       0.7588 |              0.8016 |       50.0000 |         50.0000 |      50.0000 |       1.0000 |        0.0000 |     1.0000 |              0.6025 | 41/170             | 41/170                  | 49/247                             |
| fixed_T50+referral  | MEASURED | 1532 |        0.9271 |     0.5822 |              0.6863 |      0.0645 |  0.5056 | 0.9454 |       0.7588 |              0.8016 |       50.0000 |         50.0000 |      50.0000 |       1.0000 |        0.1997 |     0.8003 |              0.6623 | 23/170             | 23/130                  | 32/247                             |
| variance_adaptive   | MEASURED | 1532 |        0.9272 |     0.5837 |              0.6858 |      0.0618 |  0.5059 | 0.9461 |       0.7588 |              0.8016 |       10.0000 |         10.0000 |      10.0000 |       0.0000 |        0.1997 |     0.8003 |              0.6623 | 24/170             | 24/132                  | 32/247                             |
| risk_aware_adaptive | MEASURED | 1532 |        0.9271 |     0.5810 |              0.6841 |      0.0613 |  0.5059 | 0.9460 |       0.7647 |              0.8057 |       19.3995 |         20.0000 |      50.0000 |       0.1371 |        0.1364 |     0.8636 |              0.6017 | 22/170             | 22/152                  | 27/247                             |

## Paired lesion-grouped bootstrap (b − a, 95% percentile CI)

| metric       | method_a           | method_b            |       a |       b |   diff_b_minus_a |   ci_low |   ci_high | ci_excludes_zero   |   n_resamples_used |   n_skipped_missing_class | resampling_unit   |   seed |
|:-------------|:-------------------|:--------------------|--------:|--------:|-----------------:|---------:|----------:|:-------------------|-------------------:|--------------------------:|:------------------|-------:|
| macro_auroc  | fixed_T50          | risk_aware_adaptive |  0.9271 |  0.9271 |          -0.0001 |  -0.0013 |    0.0010 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50          | risk_aware_adaptive |  0.5822 |  0.5810 |          -0.0012 |  -0.0060 |    0.0020 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50          | risk_aware_adaptive |  0.0645 |  0.0613 |          -0.0032 |  -0.0095 |    0.0047 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50          | risk_aware_adaptive |  0.5056 |  0.5059 |           0.0003 |  -0.0004 |    0.0011 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50          | risk_aware_adaptive | 50.0000 | 19.3995 |         -30.6005 | -31.2731 |  -29.8660 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50          | risk_aware_adaptive |  0.0000 |  0.1364 |           0.1364 |   0.1191 |    0.1556 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50          | risk_aware_adaptive |  0.2412 |  0.1294 |          -0.1118 |  -0.1679 |   -0.0604 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | fixed_T50          | variance_adaptive   |  0.9271 |  0.9272 |           0.0001 |  -0.0013 |    0.0013 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50          | variance_adaptive   |  0.5822 |  0.5837 |           0.0014 |  -0.0063 |    0.0086 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50          | variance_adaptive   |  0.0645 |  0.0618 |          -0.0027 |  -0.0109 |    0.0080 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50          | variance_adaptive   |  0.5056 |  0.5059 |           0.0003 |  -0.0007 |    0.0012 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50          | variance_adaptive   | 50.0000 | 10.0000 |         -40.0000 | -40.0000 |  -40.0000 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50          | variance_adaptive   |  0.0000 |  0.1997 |           0.1997 |   0.1774 |    0.2231 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50          | variance_adaptive   |  0.2412 |  0.1412 |          -0.1000 |  -0.1553 |   -0.0479 | True               |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | fixed_T50+referral | risk_aware_adaptive |  0.9271 |  0.9271 |          -0.0001 |  -0.0013 |    0.0010 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | fixed_T50+referral | risk_aware_adaptive |  0.5822 |  0.5810 |          -0.0012 |  -0.0060 |    0.0020 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | fixed_T50+referral | risk_aware_adaptive |  0.0645 |  0.0613 |          -0.0032 |  -0.0095 |    0.0047 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | fixed_T50+referral | risk_aware_adaptive |  0.5056 |  0.5059 |           0.0003 |  -0.0004 |    0.0011 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | fixed_T50+referral | risk_aware_adaptive | 50.0000 | 19.3995 |         -30.6005 | -31.2731 |  -29.8660 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | fixed_T50+referral | risk_aware_adaptive |  0.1997 |  0.1364 |          -0.0633 |  -0.0892 |   -0.0360 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | fixed_T50+referral | risk_aware_adaptive |  0.1353 |  0.1294 |          -0.0059 |  -0.0641 |    0.0552 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | variance_adaptive  | risk_aware_adaptive |  0.9272 |  0.9271 |          -0.0001 |  -0.0006 |    0.0003 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | variance_adaptive  | risk_aware_adaptive |  0.5837 |  0.5810 |          -0.0026 |  -0.0091 |    0.0034 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | variance_adaptive  | risk_aware_adaptive |  0.0618 |  0.0613 |          -0.0005 |  -0.0068 |    0.0063 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | variance_adaptive  | risk_aware_adaptive |  0.5059 |  0.5059 |           0.0000 |  -0.0007 |    0.0007 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | variance_adaptive  | risk_aware_adaptive | 10.0000 | 19.3995 |           9.3995 |   8.7269 |   10.1340 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | variance_adaptive  | risk_aware_adaptive |  0.1997 |  0.1364 |          -0.0633 |  -0.0876 |   -0.0355 | True               |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | variance_adaptive  | risk_aware_adaptive |  0.1412 |  0.1294 |          -0.0118 |  -0.0743 |    0.0505 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_auroc  | det_T1             | fixed_T50           |  0.9267 |  0.9271 |           0.0004 |  -0.0002 |    0.0012 | False              |               1000 |                         0 | lesion_id         |     42 |
| macro_f1     | det_T1             | fixed_T50           |  0.5836 |  0.5822 |          -0.0014 |  -0.0044 |    0.0007 | False              |               1000 |                         0 | lesion_id         |     42 |
| ece_10bin    | det_T1             | fixed_T50           |  0.0717 |  0.0645 |          -0.0072 |  -0.0125 |    0.0004 | False              |               1000 |                         0 | lesion_id         |     42 |
| brier        | det_T1             | fixed_T50           |  0.5061 |  0.5056 |          -0.0005 |  -0.0011 |    0.0001 | False              |               1000 |                         0 | lesion_id         |     42 |
| passes_mean  | det_T1             | fixed_T50           |  1.0000 | 50.0000 |          49.0000 |  49.0000 |   49.0000 | True               |               1000 |                         0 | lesion_id         |     42 |
| review_rate  | det_T1             | fixed_T50           |  0.0000 |  0.0000 |           0.0000 |   0.0000 |    0.0000 | False              |               1000 |                         0 | lesion_id         |     42 |
| mel_miss_all | det_T1             | fixed_T50           |  0.2412 |  0.2412 |           0.0000 |   0.0000 |    0.0000 | False              |               1000 |                         0 | lesion_id         |     42 |

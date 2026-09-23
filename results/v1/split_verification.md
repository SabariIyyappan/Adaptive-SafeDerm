# V1 Split Verification (PLAN.md §5.3.5)

**Overall: PASS**

| Check | Result |
|---|---|
| lesion overlap between splits | 0 (none) |
| image_id overlap between splits | 0 (none) |
| union covers all metadata images | yes |
| classes missing from 'train' | none |
| classes missing from 'val' | none |
| classes missing from 'test' | none |

## Per-split class counts and proportions

| dx    |   ('count', 'test') |   ('count', 'train') |   ('count', 'val') |   ('pct_of_split', 'test') |   ('pct_of_split', 'train') |   ('pct_of_split', 'val') |
|:------|--------------------:|---------------------:|-------------------:|---------------------------:|----------------------------:|--------------------------:|
| akiec |                  46 |                  230 |                 51 |                       3.11 |                        3.28 |                      3.33 |
| bcc   |                  71 |                  366 |                 77 |                       4.79 |                        5.23 |                      5.03 |
| bkl   |                 168 |                  774 |                157 |                      11.34 |                       11.05 |                     10.25 |
| df    |                  20 |                   76 |                 19 |                       1.35 |                        1.09 |                      1.24 |
| mel   |                 165 |                  778 |                170 |                      11.14 |                       11.11 |                     11.1  |
| nv    |                 992 |                 4679 |               1034 |                      66.98 |                       66.82 |                     67.49 |
| vasc  |                  19 |                   99 |                 24 |                       1.28 |                        1.41 |                      1.57 |

## Ours vs Paper — image counts per split

| split   |   ours |   paper |
|:--------|-------:|--------:|
| train   |   7002 |    7024 |
| val     |   1532 |    1497 |
| test    |   1481 |    1494 |

> Paper reports 1,891 / 405 / 405 unique lesions (2,701 total) for train/val/test, which is far short of HAM10000's ~7,470 unique lesion_ids. See PLAN.md §13 item 1 — we follow the stated split *method* and report our own counts here rather than trying to match the paper's inconsistent lesion totals.

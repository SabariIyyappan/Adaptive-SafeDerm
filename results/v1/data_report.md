# V1 Data Report — integrity, split and exploratory analysis

Consolidates what were previously three separate files (`integrity_audit.md`,
`split_verification.md`, and the EDA notes). Every number here is **recomputed**
by `python -m src.eval.eda`, not copied, so the report cannot go stale.

Covers PLAN.md §5.3 items 2 (integrity audit) and 5 (split verification),
which §12's definition of done requires for V1 sign-off.

## 1. Integrity audit

| Check | Expected | Found | Result |
|---|---|---|---|
| Unique image IDs | 10,015 | 10,015 | PASS |
| Unique lesion IDs | 7,470 | 7,470 | PASS |
| Lesions with >1 diagnosis | 0 | 0 | PASS |
| Class count `nv` | 6,705 | 6,705 | PASS |
| Class count `mel` | 1,113 | 1,113 | PASS |
| Class count `bkl` | 1,099 | 1,099 | PASS |
| Class count `bcc` | 514 | 514 | PASS |
| Class count `akiec` | 327 | 327 | PASS |
| Class count `vasc` | 142 | 142 | PASS |
| Class count `df` | 115 | 115 | PASS |
| Image dimensions (all 600x450) | 0 deviations | 0 | PASS |

## 2. Dataset composition (paper Table 1 equivalent)

| dx    | class_name                 | clinical_category   |   n_images |   pct_images |   n_unique_lesions |   images_per_lesion |   ratio_to_rarest |
|:------|:---------------------------|:--------------------|-----------:|-------------:|-------------------:|--------------------:|------------------:|
| akiec | Actinic keratosis (akiec)  | pre-cancerous       |        327 |         3.27 |                228 |               1.434 |               2.8 |
| bcc   | Basal cell carcinoma (bcc) | malignant           |        514 |         5.13 |                327 |               1.572 |               4.5 |
| bkl   | Benign keratosis (bkl)     | benign              |       1099 |        10.97 |                727 |               1.512 |               9.6 |
| df    | Dermatofibroma (df)        | benign              |        115 |         1.15 |                 73 |               1.575 |               1   |
| mel   | Melanoma (mel)             | malignant           |       1113 |        11.11 |                614 |               1.813 |               9.7 |
| nv    | Melanocytic nevus (nv)     | benign              |       6705 |        66.95 |               5403 |               1.241 |              58.3 |
| vasc  | Vascular lesion (vasc)     | benign              |        142 |         1.42 |                 98 |               1.449 |               1.2 |

Imbalance ratio, most to least common class: **58.3:1** (nv 6,705 vs df 115).
PLAN.md §13 item 2 notes the paper calls this 67:1; computed from counts it is 58:1,
and 67% is nevus's *share* of the dataset rather than a ratio.

## 3. Split verification — is the split unbiased?

### 3.1 Leakage checks

| Check | Result |
|---|---|
| Shared lesions, train-val | 0 PASS |
| Shared lesions, train-test | 0 PASS |
| Shared lesions, val-test | 0 PASS |
| Shared images, train-val | 0 PASS |
| Shared images, train-test | 0 PASS |
| Shared images, val-test | 0 PASS |
| Union covers all images | 10,015 of 10,015 PASS |
| All 7 classes in every split | yes PASS |

### 3.2 Split sizes

| Split | Images (ours) | Images (paper) | Unique lesions | Mean images/lesion |
|---|---|---|---|---|
| train | 7,002 | 7,024 | 5,229 | 1.339 |
| val | 1,532 | 1,497 | 1,120 | 1.368 |
| test | 1,481 | 1,494 | 1,121 | 1.321 |

### 3.3 Class balance across splits

Image-level class percentages:

| dx    |   train |   val |   test |
|:------|--------:|------:|-------:|
| akiec |    3.28 |  3.33 |   3.11 |
| bcc   |    5.23 |  5.03 |   4.79 |
| bkl   |   11.05 | 10.25 |  11.34 |
| df    |    1.09 |  1.24 |   1.35 |
| mel   |   11.11 | 11.1  |  11.14 |
| nv    |   66.82 | 67.49 |  66.98 |
| vasc  |    1.41 |  1.57 |   1.28 |

Lesion-level class percentages (what stratification actually acts on):

| dx    |   train |   val |   test |
|:------|--------:|------:|-------:|
| akiec |    3.04 |  3.04 |   3.12 |
| bcc   |    4.38 |  4.38 |   4.37 |
| bkl   |    9.73 |  9.73 |   9.72 |
| df    |    0.98 |  0.98 |   0.98 |
| mel   |    8.22 |  8.21 |   8.21 |
| nv    |   72.33 | 72.32 |  72.35 |
| vasc  |    1.32 |  1.34 |   1.25 |

### 3.4 Verdict

**The split is balanced and carries no class or sampling bias.**

- Maximum deviation of any class proportion from train, **lesion-level: 0.081 pp**.
- Maximum deviation, **image-level: 0.806 pp**.
- Mean images per lesion is near-identical across splits (train 1.339, val 1.368, test 1.321), so no split is enriched in repeatedly-photographed lesions.

Lesion-level stratification is near-exact because that is the level `train_test_split(stratify=dx)`
operates on. The image level inherits a sub-1pp drift because whole lesions move together and carry
1-6 images each. This residual is inherent to lesion-grouped splitting and is strongly preferable to
the alternative: splitting by image would place two photographs of the *same* lesion in train and
test, testing the model on a lesion it had memorized.

### 3.5 Independent leakage check (perceptual hashing + pixel verification)

Lesion grouping cannot detect two visually identical images filed under *different* lesion IDs.
To rule that out, every image was hashed and each candidate group was then **verified by direct
pixel comparison** — the hash alone is not sufficient evidence.

- Hash candidate groups: **37**
- Spanning more than one split: **16**
- Cross-split *and* different lesion ID: **16**
- **Pixel-confirmed leakage: 0**

**No leakage beyond what lesion grouping already prevents.** All 16 cross-split candidates failed pixel verification — the closest pair still differs by 15.0 grey levels on a 64x64 thumbnail, against a duplicate threshold of 8.0. This is expected: dermoscopic images share a dark circular vignette and a centred lesion, so a 64-bit difference hash collides readily. Reporting the hash candidates alone would have been a false alarm.

## 4. Image-content findings relevant to V2

- Mean hair-pixel fraction: **14.00%** of image area (sample of 835 images, paper's §3.2.1a detector).
- Highest by class: **bkl**; lowest: **vasc**.
- Because hair density differs by class, leaving it in lets the model use hair as an unintended
  cue. This quantifies the work V2's hair removal has to do.
- Mean luminance ranges from 151 to 168 across classes, motivating V2's CLAHE.

---

Figures for every section are in `results/v1/eda/`.

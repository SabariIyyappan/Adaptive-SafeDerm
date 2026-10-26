# CONTRACTS.md — Data Contracts

**Owner: Member A.** Changes to this file are announced to the team before anyone relies on them (PLAN.md §4.3).

This file specifies the exact schema of every artifact Member A produces for other members to consume. It is the executable-adjacent counterpart to PLAN.md §4.3 — column names, dtypes and file locations are fixed here so B and C can build against dummy files that match these contracts exactly.

## C1 — Label map

**Producer:** A. **Consumers:** all.
**File:** `data/label_map.json`
**Content:** alphabetical mapping, built from the full dataset before splitting.

```json
{"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}
```

## C2 — Split files

**Producer:** A. **Consumers:** B, C.
**Files:** `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`
**Columns:**

| column | dtype | notes |
|---|---|---|
| `image_id` | str | e.g. `ISIC_0024306` |
| `lesion_id` | str | e.g. `HAM_0000118` |
| `dx` | str | one of `akiec,bcc,bkl,df,mel,nv,vasc` |
| `label` | int | via C1, 0–6 |

**Guarantees:** every `lesion_id` appears in exactly one split; the union of the three files is every `image_id` in the dataset; every split contains at least one image of every class; generated with `random_state=42`.

## C9 — Preprocessed image cache (V2+, prep-ahead after V1)

**Producer:** A. **Consumers:** B, C.
**Location:** `data/cache_prep224/<image_id>.png`
**Manifest:** `data/cache_prep224/manifest.csv` — columns `image_id, hair_pixel_fraction, processing_time_s`.

## C10 — Aligned ground-truth masks (V4)

**Producer:** A. **Consumers:** B, C.
**Location:** `data/masks224/<image_id>.png`
**Manifest:** `data/masks224/manifest.csv` — columns `image_id, source, coverage_flag`.

---

## V1-only, not yet a numbered contract

**Raw resized cache** (optional speed-up, §5.3.8): `data/cache_raw224/<image_id>.png`, plain resize, no crop, no other processing. B does not require this — training must work without it.

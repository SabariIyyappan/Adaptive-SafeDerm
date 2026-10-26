# data/ — not tracked in git (except this file, checksums, and small manifests)

Owner: Member A.

## Status

**Real HAM10000 images are not yet present in this environment.** The acquisition script (`src/data/acquire.py`) is written and ready to run, but the machine this repo was built on has no network route to Kaggle or the Harvard Dataverse. Whoever runs the acquisition step (Kaggle notebook, a laptop with a phone-verified Kaggle account, or any machine with normal internet) should:

```bash
pip install -r src/data/requirements.txt
python -m src.data.acquire --dest data/raw --source kaggle
# or, if Kaggle is unavailable:
python -m src.data.acquire --dest data/raw --source dataverse
```

Then fill in the fields below and commit this file (images themselves stay untracked — see `.gitignore`).

## Source

| Field | Value |
|---|---|
| Dataset | HAM10000 ("Human Against Machine with 10000 training images") |
| Primary source | Kaggle: `kmader/skin-cancer-mnist-ham10000` |
| Fallback source | Harvard Dataverse: `doi:10.7910/DVN/DBW86T` |
| Download date | _TBD — fill in after running `acquire.py`_ |
| `HAM10000_metadata.csv` size / sha256 | _TBD_ |
| `HAM10000_images_part_1.zip` size / sha256 | _TBD_ |
| `HAM10000_images_part_2.zip` size / sha256 | _TBD_ |
| Total images after unzip | _TBD, expected 10,015_ |

## Expected layout after acquisition

```
data/raw/
├── HAM10000_metadata.csv
└── images/                 # both parts merged, flat, by image_id.jpg
```

## Directory map

| Path | Contents | In git? |
|---|---|---|
| `data/raw/` | Original HAM10000 images + metadata | No |
| `data/splits/` | Frozen C2 split files | **Yes** (small CSVs) |
| `data/label_map.json` | C1 label map | **Yes** |
| `data/cache_raw224/` | V1 optional resize cache | No |
| `data/cache_prep224/` | V2+ hair-removed + CLAHE cache | No |
| `data/masks224/` | V4 ground-truth masks | No |

## Next step once real images are present

Run, in order:

```bash
python -m src.data.integrity_audit --data-root data/raw --out results/v1/integrity_audit.md
python -m src.data.split --data-root data/raw --out data/splits --seed 42
python -m src.data.verify_split --splits data/splits --data-root data/raw --out results/v1/split_verification.md
python -m src.data.eda --data-root data/raw --splits data/splits --out results/v1/eda
```

Each script is unit-tested against a small synthetic fixture in `tests/` (no real images required), so the logic is verified even before real data lands — see `tests/test_data_pipeline.py`.

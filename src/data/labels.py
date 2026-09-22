"""C1 — the fixed alphabetical label map (PLAN.md §1, §5.3.3).

The map must be built from the full set of classes present in the dataset,
before any splitting happens, and it is alphabetical. For HAM10000's 7
`dx` classes this is fixed and always resolves to the same mapping:

    akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def build_label_map(classes: Iterable[str]) -> dict[str, int]:
    """Alphabetical class -> int label map, built from *all* classes seen.

    Raises if `classes` is empty, since a label map from zero classes is
    always a bug upstream (e.g. an empty or mis-read metadata file).
    """
    unique_sorted = sorted(set(classes))
    if not unique_sorted:
        raise ValueError("build_label_map got no classes — check the input metadata")
    return {cls: idx for idx, cls in enumerate(unique_sorted)}


# The paper's 7 HAM10000 classes (PLAN.md §1). Used as the expected/reference
# map so callers can assert the *actual* full-dataset map matches it exactly.
EXPECTED_HAM10000_LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}


def write_label_map(label_map: dict[str, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(label_map, f, indent=2, sort_keys=False)
        f.write("\n")


def load_label_map(path: Path) -> dict[str, int]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description="Build C1 label map from HAM10000 metadata.")
    parser.add_argument("--metadata", required=True, type=Path, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--out", required=True, type=Path, help="Output path, e.g. data/label_map.json")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    label_map = build_label_map(df["dx"])
    if label_map != EXPECTED_HAM10000_LABEL_MAP:
        raise SystemExit(
            f"Label map built from {args.metadata} does not match the paper's expected "
            f"7-class map.\n  built:    {label_map}\n  expected: {EXPECTED_HAM10000_LABEL_MAP}\n"
            "This usually means the metadata file has extra/missing/misspelled dx values."
        )
    write_label_map(label_map, args.out)
    print(f"Wrote label map to {args.out}: {label_map}")

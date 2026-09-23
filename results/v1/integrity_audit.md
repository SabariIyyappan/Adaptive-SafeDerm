# V1 Integrity Audit (PLAN.md §5.3.2)

**Overall: PASS**

| Check | Result |
|---|---|
| unique image_ids in metadata | 10015 (expected 10015) |
| image files missing on disk | 0 |
| unique lesion_ids | 7470 (expected 7470) |
| lesions with >1 distinct dx | 0 |
| class counts (image-level) | {"akiec": 327, "bcc": 514, "bkl": 1099, "df": 115, "mel": 1113, "nv": 6705, "vasc": 142} |
| images not matching expected size | 0 |

# Source provenance

Exact bytes, URLs, sizes, SHA-256 hashes and retrieval times are in [`outputs/tables/source_manifest.csv`](../outputs/tables/source_manifest.csv).

| Source | Version pinned | Coverage | Role |
|---|---|---|---|
| MMIST-LUNG GitHub | commit `430670ce` (2026-09-17) | 1,365 patients | Audited benchmark: labels, features, folds, code, paper PDF |
| GDC `/cases` API | Data Release 46.0 (2026-08-10) | 1,365/1,365 | **Primary source**: direct upstream of MMIST's clinical export |
| TCGA-CDR, Supplemental Table S1 | GDC PanCanAtlas file `1b5f413e-…` | 1,026/1,026 TCGA | Alternative release, curated endpoints |
| LinkedOmics CPTAC pan-cancer freeze v1.2 | S3 object, hash in manifest | 218/339 CPTAC | Alternative release |
| cBioPortal PanCancer Atlas | API pull, hash in manifest | TCGA | Downloaded for cross-checking; derived from TCGA-CDR and not used in results |
| I3LUNG Zenodo record 17535424 | `I3LUNG_DATA.zip`, hash in manifest | 2,075 patients (separate population) | Independent label-provenance case study |

## Why GDC is treated as the upstream

- MMIST's column names follow the GDC export convention (`cases.submitter_id`, `demographic.days_to_death`).
- `encode-csv.ipynb` reads GDC-style `clinical.tsv` and `exposure.tsv` files.
- Death days match GDC exactly for 399/399 TCGA and 89/90 CPTAC patients.

## Limits of "source-linked"

- GDC, TCGA-CDR and LinkedOmics describe the **same participants** and share upstream submissions. Agreement between them shows a label is consistent with the record. It does not independently adjudicate clinical truth.
- This is outcome provenance checking, not external validation on a new population.
- GDC changes between releases. One CPTAC death in MMIST is absent from GDC DR46 and one was added. Both are flagged and excluded in a sensitivity analysis.

## Benchmark details that could not be verified

- **Fold alignment:** `folds.npy` is reproduced exactly (1,365/1,365) by `StratifiedKFold(5, shuffle=True, random_state=42)` on the released `vital_status_12`, with patients ordered as sorted TCGA followed by CPTAC in `diagnosis.csv` order (`outputs/tables/fold_provenance.csv`). So the folds are stratified on the audited labels, as the paper describes. The released `PatientDataset.py` sorts all patients by ID, which would apply the same indices to different patients. Which alignment the authors ran cannot be determined from the repository, so both are analysed.
- **Original pipeline:** `config.yaml` reads `../lung/features/pretty/CPTAC+TCGA/*-encoded.csv`, which are not released under those paths, and targets `vital_status_dss`. The 12-month configuration used for the paper's Table 4 is not in the repository. The archived checkpoints under `benchmark-code/experiments/` have feature names from a different (Portuguese-language) gastric dataset.

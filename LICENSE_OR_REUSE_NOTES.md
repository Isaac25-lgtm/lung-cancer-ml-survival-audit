# Licence and data-reuse notes

## Code

All code in `src/`, `dashboard/` and `tests/` is released under the MIT licence (see `LICENSE`).

## Data: nothing patient-level is committed

`data/raw/` and `data/interim/` are git-ignored. Run `python -m src.download_data` to fetch the inputs from their original hosts. Exact bytes are pinned by SHA-256 in `outputs/tables/source_manifest.csv`.

| Source | Status checked 2026-11-05 | What this repo does |
|---|---|---|
| MMIST-LUNG GitHub (`ritacmendes/MMIST-LUNG` @ `430670ce`) | **No licence file.** We treat it as all rights reserved for redistribution. | Downloads at runtime; publishes only aggregate counts and metrics |
| GDC API, Data Release 46.0 (TCGA-LUAD, TCGA-LUSC, CPTAC-3 open-access clinical) | Open access under the NIH GDC data-sharing policies; citation of GDC and the projects is required | Queries at runtime; aggregates only |
| TCGA-CDR (Liu et al., Cell 2018) | Public supplementary table, GDC PanCanAtlas publication page | Downloads at runtime; aggregates only |
| LinkedOmics CPTAC pan-cancer freeze v1.2 | Public S3 bucket; cite the CPTAC pan-cancer resources | Downloads at runtime; aggregates only |
| cBioPortal PanCancer Atlas (LUAD, LUSC) | Public API | Downloaded for cross-checking only |
| I3LUNG (Zenodo record 17535424) | **CC BY-NC 4.0** (non-commercial) | Downloads `I3LUNG_DATA.zip` at runtime; only aggregate audit counts are published |

### Identifiers

Patient identifiers (TCGA barcodes, CPTAC case IDs) are public, de-identified project identifiers. Even so, no committed table, figure or dashboard view lists individual patients. Version-difference cases are described only in aggregate.

### Ethics

- This is secondary analysis of public, de-identified, open-access data.
- Whether institutional ethics review or a formal exemption is needed depends on the author's institution and the target journal. Isaac should confirm this with their institution before submission.
- No data from Uganda Cancer Institute or any other institution were requested or used.

### Disclosure

- These findings concern a public benchmark's labels.
- Contacting the MMIST-LUNG authors (responsible disclosure) is recommended before publication. It has **not** been done, and should be done only on Isaac's explicit instruction.

# Dashboard guide

## Run locally

```bash
streamlit run dashboard/app.py
```

The app reads only the aggregate tables committed in `outputs/tables/`, so it works from a fresh clone with no data download and no credentials. If a required table is missing it shows an error instead of any results.

## Pages

| Tab | What it shows |
|---|---|
| Overview | The question, headline counts, and findings ordered by strength of evidence |
| Provenance | Sources, hashes, linkage, death-time agreement, reconciled version differences |
| Label agreement | Released vs source-derived status under each outcome definition, how the labels were generated |
| Survival & censoring | Kaplan–Meier curves, released fractions, reasons status is not ascertainable |
| Model evaluation | Experiments A, A2 and B, the cohort-risk mechanism, model comparison across label sets |
| Robustness | Sensitivity analyses, fold reconstruction, censoring handling, controlled leakage, falsification checks, DSS check, I3LUNG |
| Literature & reproducibility | Closest prior work, commands, table hashes compared with the run manifest |

## Checks run

- 7 tabs rendered in Chrome at 1300 px and 390 px wide: no exceptions, no console errors, no horizontal page overflow.
- Screens for the Overview, Label agreement and Model evaluation tabs were inspected at desktop width, and the first screen of the Overview and Label agreement tabs at phone width. Other tabs were checked by the automated counts only.
- `tests/test_dashboard_smoke.py` compares headline numbers with the tables, scans the tables for patient identifiers, and checks that a missing table is reported.
- Dark mode was not visually tested.

## Hosting

The app has not been deployed. Streamlit Community Cloud can host it from this GitHub repository with `dashboard/app.py` as the entry file. Check its current terms and pricing first. No secrets are needed. Use `requirements-lock.txt` as the dependency file.

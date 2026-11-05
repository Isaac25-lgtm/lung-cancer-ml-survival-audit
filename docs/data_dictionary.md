# Data dictionary

## Released MMIST-LUNG fields used (`features/demographics.csv`, `features/diagnosis.csv`)

| Field | Meaning as released | Role here |
|---|---|---|
| `cases.submitter_id` | GDC case submitter ID (TCGA barcode or CPTAC case ID) | Linkage key |
| `vital_status_12` | 1 = "alive at 12 months" per the paper (Table 1: 1,169 alive / 196 deceased) | **Audited label**. Analysed as `y_released = 1 - vital_status_12` |
| `vital_status_dss` | 0 = disease-specific death (any time) in TCGA; constant 1 in CPTAC | Benchmark DSS target; quarantined leakage field |
| `demographic.days_to_death` | Days from index to death (GDC) | Used to reverse-engineer the label rules; forbidden as a predictor |
| `days_to_last_fup` | Equals the death day for deaths; missing for all 249 CPTAC patients with no recorded death | Forbidden as a predictor |
| 15 clinical features | Age bin, ethnicity, sex, race, pack-years, smoking status, years smoked, AJCC pathological M/N/stage/T, AJCC edition, ICD-10, ICD-O-3 morphology, prior malignancy (ordinal and nominal codes) | `CLIN` predictor set |

Encoding maps are taken from `encode-csv.ipynb`. Pathological stage is assigned after surgery, so `CLIN` is "known at the time of pathological staging", not strictly at the diagnostic visit.

## Source fields (GDC Data Release 46.0, `/cases` API)

| Field | Use |
|---|---|
| `project.project_id` | Verifies cohort and histology (TCGA-LUAD, TCGA-LUSC, CPTAC-3) |
| `index_date` | Time origin: Diagnosis (1,363) or Sample Procurement (2) |
| `demographic.vital_status` | Alive / Dead / Not Reported |
| `demographic.days_to_death` | Death time |
| `demographic.cause_of_death` | Used in the DSS audit |
| `diagnoses.days_to_last_follow_up`, `follow_ups.days_to_follow_up` | Last contact = maximum of both |

## Derived variables (`src/derive_outcomes.py`, `src/link_sources.py`)

| Variable | Definition |
|---|---|
| `y_gdc_gt` | **Primary** three-state 365-day status from GDC: 1 = early death, 0 = alive beyond 365 days, NaN = not ascertainable. A death on day 365 counts as early death. |
| `y_gdc_ge` | As above, but a death on day 365 counts as a survivor |
| `y_cdr_*` | From TCGA-CDR `OS` / `OS.time` |
| `y_lo_*` | From LinkedOmics `OS_event` / `OS_days`; strict rule requires a non-missing event flag |
| `y_alt_*` | CDR for TCGA, LinkedOmics for CPTAC |
| `y_consensus_*` | Value where GDC and the alternative release agree; NaN otherwise |
| `y_mmistfields_*` | Derived only from fields released in MMIST |
| `y_rule_tcga`, `y_rule_notebook` | The two reverse-engineered release rules |
| `os_time_gdc`, `os_event_gdc` | Right-censored overall survival for Kaplan–Meier, IPCW and C-index |
| `flag_*` | Version and provenance flags (death removed or added upstream, negative time, non-diagnosis index, dead without a date) |

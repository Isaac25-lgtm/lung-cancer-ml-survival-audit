# Analysis protocol (v1.1, 2026-11-05; v1.0 frozen earlier the same day)

Machine-readable version: [`configs/analysis.yaml`](../configs/analysis.yaml). Deviations and clarifications are logged in [`changelog.md`](changelog.md).

## Status of this protocol

This protocol was written **after** the exploratory feasibility work in the handover, **after** we had reverse-engineered the released-label rules, and **after** we had seen the GDC versus MMIST death-time agreement. It was frozen **before** any model in `src/experiments.py` was fitted. It is prospective relative to the modelling analysis. It is **not** a preregistration of data already seen.

## Question

In MMIST-LUNG, how do source-linked 12-month survival outcomes compare with the released benchmark labels, and how does the choice of outcome affect the evaluation of machine-learning predictions on the same patients?

## Population, index date, horizon

- **Population:** all 1,365 MMIST-LUNG patients at commit `430670ce` (TCGA-LUAD 522, TCGA-LUSC 504, CPTAC-3 339).
- **Index date:** diagnosis, as given by the GDC `index_date`. Two CPTAC cases are indexed at sample procurement. They are flagged and excluded in a sensitivity analysis.
- **Horizon:** 365 days.
- **Intended use of any model:** none clinical. Models are instruments for measuring how much the label choice matters.

## Outcome (three states)

| State | Primary rule (`gt`) |
|---|---|
| Early death (`y = 1`) | GDC `vital_status == Dead` and `0 <= days_to_death <= 365` |
| Alive beyond horizon (`y = 0`) | death after day 365, **or** not dead with last contact `>= 365` |
| Not ascertainable (`NaN`) | everything else: censored before day 365, no follow-up, dead with no date, negative times |

- **Last contact** is the maximum of `diagnoses.days_to_last_follow_up` and `follow_ups.days_to_follow_up`.
- **Boundary sensitivity (`ge`):** a death on day 365 counts as a survivor.
- **Alternative source releases:** TCGA-CDR for TCGA, and LinkedOmics for CPTAC with a strict non-missing event flag. A consensus outcome uses only patients on whom GDC and the alternative release agree.

## Comparators, models and splits

- **Labels compared:** the released `vital_status_12` (analysed as `y_released = 1 - vital_status_12`) against the source-derived outcome.
- **Models:**
  - prevalence only
  - L2 logistic regression (C = 1)
  - histogram gradient boosting (fixed hyperparameters, an XGBoost analogue)
  - stage-only logistic regression
- **Features:** the 15 released clinical features (`CLIN`). Outcome and follow-up fields are blocked in code (`src/models.py::check_features`).
- **Splits:** the released `folds.npy`, applied in the order it was generated (sorted TCGA, then CPTAC in `diagnosis.csv` order), which reproduces the file exactly. v1.0 specified ascending `submitter_id` order, now reported as a sensitivity analysis (changelog item 8). Experiment B uses 50 repeats of 5-fold CV stratified on cohort × outcome state.
- **Preprocessing:** fit inside training folds only. No hyperparameter tuning.

## Primary contrast

**Experiment A.** Logistic regression is trained on released labels in the released folds. Its out-of-fold scores are held fixed, and are evaluated on the 1,154 ascertainable patients. The estimand is:

ΔAUROC = AUROC(scores, y_source) − AUROC(scores, y_released)

on identical patients, using **fold-averaged** AUROC (see the changelog for this clarification). Uncertainty comes from a 2,000-replicate bootstrap over patients, stratified by fold × cohort and conditional on the fitted scores (seed 20261105).

## Secondary and sensitivity analyses

| Analysis | What it does |
|---|---|
| Experiment A, other metrics | Brier score, balanced accuracy (threshold = training-fold prevalence), AUPRC, calibration intercept and slope; also the benchmark setting (released labels, all 1,365) |
| A2 | Train on TCGA, test on CPTAC, with fixed scores |
| B | Training-label effect: B1 released/all, B2 released/ascertainable, B3 source/ascertainable |
| C | Censoring: IPCW AUROC and Brier at 365 days, sharp partial-identification bounds, imputation scenarios, Harrell C |
| D | Quarantined leakage negative control |
| Sensitivities | `ge` boundary, alternative sources, consensus outcome, excluding flagged cases, per cohort, an alternative fold assignment |

## Falsification checks

- AUROC under a label-permutation null.
- The prevalence model's fold-averaged AUROC should equal 0.5.
- Flipping the released labels of CPTAC recorded deaths should recover the source status.
- Cohort identifiability from the clinical features.
- Fold overlap and partition checks.

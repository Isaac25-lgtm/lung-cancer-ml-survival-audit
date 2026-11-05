# Protocol v2: replication and confirmatory tests

**Frozen:** 2026-11-05, and committed to the public repository **before** any MMIST-ccRCC label value, ccRCC source record or new model result was examined. The commit timestamp is the record.

**Authors:** Omoding Isaac, Bob Tumushime.

## Why v2 exists

Version 1 left three weaknesses:

- It covered a single benchmark.
- Its pre-specified evaluation-label contrast was weak: −0.034 (95% CI −0.068 to −0.003).
- Its stronger machine-learning findings (training-label effect, cohort-risk reversal, pair composition) were post-hoc.

Version 2 turns those post-hoc findings into **stated predictions**. It tests them on a **second benchmark** that has not been analysed, and re-estimates the training-label effect with **full model-refitting uncertainty**. Version 1 results are not changed or reinterpreted.

## What had been seen before freezing

- **MMIST-ccRCC paper** (arXiv:2405.01658): its text says patients were selected for "a follow-up of 12-months" and that "543 survived after 12 months (88%)", out of 618 patients (189 CPTAC-CCRCC, 429 TCGA-KIRC).
- **Released CSV** `clinical+genomic_split.csv` (repo `Multi-Modal-IST/multi-modal-ist.github.io`, file commit `92386bb`): column names and row count (618) only. **No values.**
- **No kidney GDC records** and **no new model outputs.**

## Data

| Item | Specification |
|---|---|
| Benchmark | MMIST-ccRCC `clinical+genomic_split.csv`, pinned to repo HEAD `c3f1a066` |
| Source | GDC Data Release 46.0 `/cases` for every submitter ID in the benchmark (projects TCGA-KIRC and CPTAC-3) |
| Alternative source | TCGA-CDR (KIRC), for sensitivity analysis |
| Outcome | Same three-state 365-day status as v1 (`src/derive_outcomes.py`, convention `gt`) |
| Released label | `vital_status_12`. Its orientation is fixed by the paper's statement that 88% survived: the class with prevalence near 0.88 is "survivor". |
| Features | All columns except `case_id`, `vital_status_12` and `Split` |
| Stage-only comparator | `ajcc_path_tumor_stage` |

## Confirmatory hypotheses

**C1: label provenance (ccRCC).** This is descriptive and deterministic, with the decision rule fixed in advance:

- **Released label per cohort.** Compute its agreement with GDC-derived status among ascertainable patients, with Wilson 95% CIs.
- **Agreement among recorded deaths.** Compute it separately for patients with a recorded death.
- **Candidate rules.** Report how many labels each of the two v1 rules reproduces (TCGA rule; notebook rule).
- **Decision:** "inversion replicated" if agreement among ascertainable CPTAC recorded deaths is below 10% **and** TCGA agreement is above 90%. "No inversion" if both cohorts exceed 90%. Otherwise "mixed".
- **Unknown status.** Report the number of patients whose 12-month status is not ascertainable, and how they were labelled. The paper's inclusion criterion predicts **few**.

**C2: training-label effect on MMIST-LUNG with full uncertainty.**

- **Estimand:** ψ_B = AUROC(model trained on source labels, Y) − AUROC(model trained on released labels, Y), for **logistic regression** in the released folds, fold-averaged, on ascertainable patients.
- **Uncertainty:** 200 bootstrap replicates. Patients are resampled within each fold's training and test sets, both models are **refitted** on each resampled training set, and both are evaluated on the resampled test set. Seed 2026110502.
- **Prediction:** ψ_B > 0. **Test:** the lower bound of the 95% percentile interval is above 0.

**C3: training-label effect on ccRCC.**

- **When it runs:** only if C1 finds at least 20 ascertainable patients whose released label disagrees with the source. Otherwise it is reported as not applicable.
- **Design:** the same estimand and refit bootstrap as C2. Folds are 5-fold stratified CV on cohort × source state (seed 2026110503), because the released 497/121 split leaves too few test events.
- **Prediction:** ψ_B > 0.

**Multiplicity.** Holm adjustment is applied across C2 and C3, at family-wise α = 0.05, one-sided.

## Secondary directional predictions

These are stated in advance, but they are not part of the confirmatory family.

- **S1, ccRCC, conditional on inversion:** the fixed-score evaluation-label contrast ψ_A is negative for logistic regression trained on released labels.
- **S2, ccRCC, conditional on inversion:** in the pair-level AUROC decomposition of ψ_A, the change in pair mix accounts for more than 50% of the logistic gap, and changes in concordance account for more than 50% of the stage-only gap.
- **S3, ccRCC, conditional on inversion:** mean predicted risk of released-trained logistic regression is higher in CPTAC than in TCGA, and the source-label event rate is lower in CPTAC than under released labels.
- **S4, MMIST-LUNG, new model families:** ψ_B > 0 also holds for **XGBoost** (the benchmark's own classifier type) and a **multilayer perceptron**, using the refit bootstrap with 100 replicates. Settings are fixed: XGBoost with 300 trees, depth 3, learning rate 0.05, subsample 0.8; MLP with one hidden layer of 64 units, alpha 1e-3, early stopping, `max_iter` 500. Seeds are fixed.

## Reporting rules

- Every hypothesis is reported whatever its result, including null and contrary findings.
- Any deviation from this protocol is dated in `docs/changelog.md`.
- Version 1 results remain as reported. A v2 result does not replace a v1 result.

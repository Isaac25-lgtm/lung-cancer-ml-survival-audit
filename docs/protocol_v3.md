# Protocol v3: revised training-label analysis after external review

**Written:** 2026-11-05, and committed **before** the analyses below were run.

**Status:** these are **revised estimates on the same MMIST-LUNG cohort**. The training-label effect has already been observed (v1, post-hoc; v2, C2), so these are not independent confirmation. They correct how that effect is defined and how its uncertainty is estimated.

## Why

An external review identified two problems with v2 C2:

1. **The comparison changed two things at once.** It compared a model trained on source labels among ascertainable patients with a model trained on released labels among *all* training patients, so both the labels and the training population changed. It is valid as a comparison of two practical training strategies, but it does not isolate the effect of relabelling.
2. **The bootstrap was inconsistent across folds.** It resampled each fold's training and test sets separately, so one patient could receive different multiplicities in different folds within the same replicate. The one-sided p-value came from counting bootstrap estimates at or below zero, not from a null distribution.

## Estimands

Throughout, models are evaluated by fold-averaged AUROC against source-derived labels, on ascertainable test patients, in the released folds (reconstructed generation order).

| Name | Comparison | Question |
|---|---|---|
| **Relabelling effect (primary)** | B3 − B2: source-label model vs released-label model, **both trained on the same ascertainable patients** | What does changing only the training labels do? |
| Training-strategy comparison | B3 − B1: source labels on ascertainable patients vs released labels on all patients (the v2 C2 contrast) | How do two practical strategies compare? |

## Uncertainty

Patient-level bootstrap with a **consistent multiplicity**:

- **Draw once per replicate.** Each patient's multiplicity is drawn once, by resampling with replacement within each released fold, which preserves fold sizes.
- **Carry it everywhere.** That multiplicity is used in every training set the patient belongs to and in the patient's test fold.
- **Refit every model.** All model variants are refitted on every replicate.

| Setting | Value |
|---|---|
| Replicates | 1,000 for logistic regression; 300 for histogram gradient boosting (runtime) |
| Interval | 95% percentile interval |
| Seed | 2026110504 |
| Also reported | Share of replicates with a positive difference |

**No p-values** are reported. These intervals describe sampling uncertainty given the fixed folds. They do not represent independent replication.

## Also changed in this revision

- **IPCW boundary.** Patients censored exactly at day 365 are now known survivors, matching the outcome derivation, with weight 1/G(365−).
- **IPCW Brier denominator.** The denominator is now the patients with a usable time-to-event record.
- **Regeneration.** All affected tables are regenerated.

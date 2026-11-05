# Methods notes

The full specification is in [protocol.md](protocol.md). These notes cover implementation choices.

## Fixed-horizon status

`src/derive_outcomes.py::status_at_horizon` returns 1 (early death), 0 (alive beyond the horizon) or NaN. A patient censored at day *c* ≥ 365 is known to survive beyond the horizon under both boundary conventions, because T > c ≥ 365. Dead patients without a valid date are left unknown, since their last contact may pre-date death.

## Metrics

Orientation: `y = 1` is early death, and the score is P(early death).

- **Primary metric:** fold-averaged AUROC, matching the benchmark's reporting of means across folds. Pooled out-of-fold values are reported alongside it.
- **Balanced accuracy:** uses the training-fold event prevalence as a pre-specified threshold. The benchmark tuned its thresholds, so these values are not directly comparable to its table.
- **Brier differences** across label sets are driven largely by prevalence and are not read as evidence of quality.

## Bootstrap

Patients are resampled within fold × cohort strata, 2,000 replicates (seed 20261105), with scores fixed. These intervals are conditional on the fitted models: they do not include model-fitting uncertainty. Experiment B ranges across CV repeats reflect split variability only, so they are complemented by the patient-level bootstrap in `exp_b_benchmark_folds.csv`.

## Censoring-aware evaluation

- **IPCW:** cases (T ≤ 365 with an event) get weight 1/G(T−) and controls (T > 365) get weight 1/G(365), where G is the reverse Kaplan–Meier estimate, stratified by cohort in the main analysis. Positivity is not a concern here: G(365) is 0.877 in TCGA and 0.834 in CPTAC, and the maximum weight is 1.20.
- **Sharp AUROC bounds:** for a fixed number *k* of unknowns completed as events, AUROC is minimised by assigning the event label to the *k* lowest-scored unknowns and maximised by assigning it to the *k* highest-scored. The code scans all *k*, and the result is validated against brute force in tests.

## Models

- **Logistic regression:** C = 1, ordinal features median-imputed with missing indicators and scaled, nominal features one-hot encoded with a missing category.
- **`HistGradientBoostingClassifier`:** fixed hyperparameters, no tuning, native missing-value handling.
- **Stage-only:** one-hot pathological stage.

`src/models.py::check_features` raises an error if any outcome or follow-up field enters a predictor set, unless the quarantined leakage experiment opts in explicitly.

## Security

`folds.npy` is a pickled object array. It is loaded with an allowlist unpickler (`src/folds.py`) that permits only NumPy reconstruction, and this is tested with a malicious payload. External notebooks and code were read as text and never executed.

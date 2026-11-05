# Changelog and protocol deviations

All dates are 2026-11-05 unless stated otherwise.

1. **Primary source changed from LinkedOmics (218 patients) to GDC Data Release 46.0 (all 1,365).**
   - **Why:** the MMIST column names (`cases.submitter_id`, `demographic.days_to_death`) show the benchmark was exported from GDC. GDC is therefore the direct upstream and covers every patient. LinkedOmics covers only 218 CPTAC patients.
   - **What remains:** LinkedOmics and TCGA-CDR are kept as alternative-release sensitivity sources.
   - **Timing:** decided before any model in `src/experiments.py` was fitted.
2. **Handover reproduction first.** All 35 handover quantities reproduce from freshly downloaded bytes (`outputs/tables/handover_reproduction.csv`). Exploratory rules were replicated as written, then revised:
   - The LinkedOmics strict event-flag requirement made no difference: 177 patients either way.
3. **Fold-averaged rather than pooled out-of-fold AUROC for the primary contrast.**
   - **Why:** the benchmark paper reports means across folds. A smoke test had also shown a pooled AUROC of 0.465 for the prevalence-only model.
   - **Correction:** that 0.465 was largely caused by the mis-aligned folds described in item 8. Under the correct alignment the pooled value is 0.498, so the artefact does not justify the choice. Fold-averaging is retained only for consistency with the benchmark.
   - **Timing:** this choice was made **after** seeing smoke-test pooled AUROCs for the logistic model (released 0.624, source 0.582). Pooled values are reported alongside fold-averaged ones, and the two differ by about 0.001 in the primary contrast.
4. **`auroc_bounds` bug fixed before use.**
   - **Bug:** an identity comparison on array slices always evaluated False, which mis-built the minimising completion.
   - **Fix:** the test now brute-forces 8 random seeds. No result used the buggy version.
5. **AUROC reimplemented** as a mid-rank Mann–Whitney statistic for speed. It is tested equal to sklearn to 1e-12.
6. **`lifelines` dropped.** It failed to build (`autograd-gamma` has no wheel). Kaplan–Meier, reverse-KM censoring weights and Harrell's C are implemented in `src/stats_utils.py` and `src/metrics.py`, with unit tests.
7. **I3LUNG used only as an independent label-provenance audit,** not for model transfer. Its population and endpoint are incompatible with MMIST-LUNG.
8. **Protocol v1.1: fold alignment corrected (deviation from v1.0).**
   - **Found:** `folds.npy` is exactly `StratifiedKFold(5, shuffle=True, random_state=42)` on the released `vital_status_12`, with patients ordered as sorted TCGA followed by CPTAC in `diagnosis.csv` order (1,365/1,365 fold assignments reproduced). v1.0 applied the indices in ascending-ID order, which gave a valid leak-free partition but not the assignment the file was generated for.
   - **Inconsistency in the source:** the released `PatientDataset.py` sorts patients by ID, so applying the file through the released code gives the ascending-ID alignment. Which alignment the authors ran cannot be determined.
   - **Decision:** the generation order became primary because it is the only one that reproduces the file and matches the paper's description of folds stratified on 12-month status. Ascending-ID order is reported as a sensitivity analysis.
   - **Timing:** the v1.0 result (ΔAUROC −0.042, 95% CI −0.074 to −0.010) had been seen. The choice rests on the fold-reconstruction evidence and was made before any result under the new alignment was computed. Both results are reported.
   - **Correction:** an earlier note that the folds were not balanced on the released label was wrong. It was an artefact of the mis-aligned order.
   - **Effect:** the primary contrast weakened from −0.042 (−0.074 to −0.010) to −0.034 (−0.068 to −0.003), and the gradient-boosting contrast now crosses zero. Earlier wording calling the contrast robust was withdrawn.
9. **Post-hoc analyses added after seeing the v1.1 results,** and labelled as post-hoc wherever reported:
   - `exp_a_cohort_mechanism.csv`: mean predicted risk by cohort, to explain why within-cohort contrasts are near zero while the pooled contrast is not.
   - `exp_b_benchmark_folds.csv`: patient-level paired bootstrap of the training-label effect in the released folds, because Experiment B's repeat ranges reflect split variability only.
   - `exp_a_model_ranking.csv`: model-gap interaction, added after noticing that point-estimate rankings changed between label sets.
10. **Figure 7 caption corrected.** A draft claimed the score distribution showed the model had learned the inversion. It does not, and the caption now says so.

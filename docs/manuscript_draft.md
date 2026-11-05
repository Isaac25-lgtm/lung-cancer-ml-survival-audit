# Manuscript draft material (PRELIMINARY: not submission-ready)

Written for a short communication or benchmark-audit article. Every number traces to `outputs/tables/`. Author roles, ethics statement and venue formatting are left for real collaborators to complete.

**Authors:** Omoding Isaac¹ (corresponding, omodingisaac111@gmail.com), Echodu Moses², Bob Tumushime³, Noah Okiror⁴ and Daniel Baguma⁴.

¹ Data Scientist, Ministry of Health. ² Co-investigator, Uganda Cancer Institute. ³ Data Scientist, Uganda Bureau of Statistics (UBOS). ⁴ Malaria Consortium.

Add contributor roles (CRediT), any further collaborators, and a statement that the views are the authors' own before submission.

**Title:** Survival Label Inconsistencies and Machine Learning Evaluation in Lung Cancer: A Source-Linked Reanalysis of the MMIST-LUNG Benchmark

## Structured abstract (draft)

**Background.** Public multimodal benchmarks combine cohorts with different clinical conventions. Fixed-horizon survival labels built from such data can be incorrect in ways that standard cross-validation cannot detect.

**Methods.** We linked all 1,365 patients in the MMIST-LUNG benchmark (1,026 TCGA, 339 CPTAC) to upstream Genomic Data Commons records (Data Release 46.0), with TCGA-CDR and LinkedOmics as alternative releases. We derived a three-state 365-day outcome (early death, alive beyond 365 days, not ascertainable), reverse-engineered the rules generating the released label, and quantified agreement. Using the released folds and 15 clinical features, we compared identical out-of-fold predictions from surrogate models against released and source-derived labels on the same patients, and separately examined the effect of training labels and censoring handling.

**Results.** Two opposite-polarity rules reproduced the released label exactly (TCGA 1,026/1,026; CPTAC 339/339). Among ascertainable patients, TCGA labels were fully consistent with the source (0/894 inconsistent), whereas 88/260 CPTAC labels (33.8%; 95% CI 28.4–39.8) were inconsistent, including all 62 released early-death labels and all 26 source early deaths. A further 211 patients (15.5%) had unascertainable 12-month status and were released as survivors. The labels reversed the apparent cohort risk: early-death rates in CPTAC versus TCGA were 23.8% versus 15.0% under released labels and 10.0% versus 15.0% under source-derived labels. With predictions held fixed, logistic regression AUROC was 0.634 against released and 0.600 against source-derived labels (difference −0.034; 95% CI −0.068 to −0.003), an effect that was small and not robust across sensitivity analyses. In post-hoc analyses, training on source-derived labels improved AUROC against those labels by 0.049 (95% CI 0.015 to 0.082), while the same comparison judged by released labels favoured the released-trained model by 0.073 (95% CI 0.032 to 0.116). An identical audit of the independent I3LUNG resource found no inconsistencies.

**Conclusions.** Released 12-month labels in MMIST-LUNG reflect a cohort-specific polarity inversion and treat unascertainable outcomes as survival. The inversion reverses apparent cohort risk and could mislead model training and selection, although its effect on evaluation of fixed predictions was small. Source-linked label audits are inexpensive and should precede benchmarking on combined clinical cohorts.

## Article outline

1. **Introduction.** Fixed-horizon endpoints in public benchmarks; why cross-validation cannot detect label errors; closest prior work (label errors in benchmarks; leakage and censoring audits in oncology); the question.
2. **Methods.**
   - Sources and versions.
   - Deterministic linkage.
   - Outcome definition and boundary conventions.
   - Label-rule reproduction.
   - Experiments A to D: fixed-score evaluation, transport, training labels, censoring, controlled leakage.
   - Metrics, bootstrap design and falsification checks.
   - I3LUNG case study.
   - TRIPOD+AI items where applicable.
3. **Results.**
   - Fig 1: label agreement.
   - Fig 2: polarity mechanism.
   - Fig 3: Kaplan–Meier and censoring.
   - Fig 4: fixed-score contrast.
   - Fig 5: training labels.
   - Fig 6: censoring handling.
   - Fig 8: cohort-risk mechanism (post-hoc).
4. **Discussion.**
   - **Mechanism:** the features identify the cohort perfectly, so a model learns the cohort-level risk ordering in its training labels, and the labels reverse that ordering.
   - **Lead with the audit:** the deterministic label finding is the main result. The pre-specified evaluation contrast is small and fragile; state this plainly, and label the larger training-label effects as post-hoc.
   - **Why the direction depends on training:** Experiment A versus A2.
   - **Censoring mattered less than polarity here.**
   - **Limitations (below).**
   - **Recommendations for benchmark curators:** publish label-generating code that runs, keep an explicit unknown state, and audit labels per cohort.
5. **Suggested correction for benchmark users.** Derive the label from GDC time and event fields; leave censored-before-horizon patients unknown.

## Tables

| Table | Source file |
|---|---|
| 1. Sources, versions, coverage | `source_manifest.csv`, `linkage_summary.csv` |
| 2. Rule reproduction | `rule_reproduction.csv` |
| 3. Agreement by cohort and histology | `discordance_summary.csv` |
| 4. Experiment A metrics and paired differences | `exp_a_metrics.csv`, `exp_a_paired_deltas.csv` |
| 5. Training-label effect | `exp_b_summary.csv`, `exp_b_paired.csv`, `exp_b_benchmark_folds.csv` |
| 6. Cohort-risk mechanism | `exp_a_cohort_mechanism.csv` |
| S1. Sensitivity analyses | `exp_sensitivity.csv` |
| S2. Censoring handling | `exp_c_censoring.csv` |
| S3. Controlled leakage | `exp_d_leakage.csv` |
| S4. DSS label check | `dss_audit.csv` |
| S5. I3LUNG | `i3lung_label_audit.csv` |
| S6. Fold reconstruction | `fold_provenance.csv` |

## Limitations to state

- A single benchmark; not a claim about lung cancer datasets generally.
- Source records are not adjudicated truth. TCGA, CPTAC and GDC share upstream submissions.
- Surrogate models only; the authors' MLP/Transformer pipeline and published numbers were not re-run.
- Few events in CPTAC (26). Per-cohort AUROC estimates are imprecise, and the CPTAC-only contrast is inconclusive (+0.011; −0.206 to +0.140).
- Experiment A intervals are conditional on fitted scores and do not include model-fitting uncertainty. Experiment B ranges reflect split variability only.
- Complete-case analysis changes the estimand. IPCW assumes censoring is independent of the outcome within cohort, and sharp bounds (0.468–0.734) show how much this assumption matters.
- Pathological stage is available only after surgery, so the prediction moment is not strictly the diagnostic visit.
- The fold file and the released code imply different patient alignments, and which one the authors ran is unknown. The primary contrast moved from −0.042 to −0.034 between them.
- The primary contrast is fragile: its interval crosses zero when flagged cases are excluded and under a new fold split. Multiple contrasts were examined, and only Experiment A's logistic contrast was pre-specified as primary.
- Mechanism, training-effect bootstrap and model-gap interaction analyses are post-hoc.
- The fold-averaging choice was finalised after a smoke test of pooled values, and protocol v1.1 changed the fold alignment after v1.0 results were seen (see `docs/changelog.md`).

## Evidence ledger

| Statement | Type |
|---|---|
| Rules reproduce released labels exactly | Direct observation |
| Death days match GDC | Direct observation |
| CPTAC labels come from the notebook rule | Inference (exact reproduction; the notebook is described as not meant to run) |
| Inversion is an unintended encoding error | **Unresolved**; authors not yet contacted |
| Labels reverse the apparent cohort risk | Direct observation |
| Flexible models learn that reversal via cohort-identifiable features | Inference, supported by cohort AUROC of 1.000, mean predicted risk by cohort, and Experiment B (post-hoc) |
| The published MMIST numbers are inflated | **Not established**; pipeline not re-run |
| No prior report of this issue | Provisional; search was not systematic |

# Fixed-horizon survival labels in public multimodal cancer benchmarks: a source-linked audit of label provenance and its consequences for machine-learning evaluation

**Running title:** Survival label audit of multimodal cancer benchmarks

**Authors:** Omoding Isaac¹\*, Echodu Moses², Bob Tumushime³, Noah Okiror⁴, Daniel Baguma⁵

¹ Ministry of Health · ² Uganda Cancer Institute · ³ Uganda Bureau of Statistics (UBOS) · ⁴ Malaria Consortium · ⁵ Malaria Consortium

\*Corresponding author: Omoding Isaac, omodingisaac111@gmail.com

**Article type:** Research and applications (short report)

**Word count:** abstract 250; main text about 3,100 (including in-text tables; excluding Tables 1–3, figure legends and references)

---

## Abstract

**Objective.** To audit whether released fixed-horizon survival labels in public multimodal cancer benchmarks agree with upstream records, and how disagreement affects machine learning.

**Materials and Methods.** We linked all patients in MMIST-LUNG (n = 1,365) and MMIST-ccRCC (n = 618), each pooling TCGA and CPTAC cohorts, to Genomic Data Commons records and derived three-state 12-month status. I3LUNG (n = 2,075) was a negative control. We reconstructed label-generating rules and estimated evaluation-label and training-label effects on identical patients. A replication protocol was committed publicly before its data were examined.

**Results.** In MMIST-LUNG, two opposite-polarity rules reproduced all 1,365 labels. In CPTAC, 88 of 260 ascertainable labels (33.8%) contradicted the source, including every released early death. A further 211 patients with unknown status were labelled survivors. MMIST-ccRCC showed no inversion, but 18 unverifiable deaths, all with missing stage, inflated logistic AUROC from 0.791 to 0.832. I3LUNG was fully consistent. With predictions fixed, the evaluation-label effect was small (ΔAUROC −0.034; 95% CI −0.068 to −0.003). In a confirmatory test with models refitted on every bootstrap replicate, training on source-derived labels improved AUROC by 0.049 (95% CI 0.008–0.091), yet benchmark-style evaluation favoured the opposite model.

**Discussion.** Label errors were systematic, cohort-specific and invisible to cross-validation. They acted mainly through training and model selection.

**Conclusion.** Source-linked label audits are inexpensive and should precede benchmarking on pooled clinical cohorts.

**Keywords:** benchmark datasets; label noise; survival analysis; censoring; machine learning evaluation; data provenance; lung cancer; renal cell carcinoma

---

## Background and significance

Public multimodal benchmarks have become the main testing ground for clinical machine learning in oncology. Many pool cohorts from The Cancer Genome Atlas (TCGA) and the Clinical Proteomic Tumor Analysis Consortium (CPTAC), distributed through the Genomic Data Commons (GDC) [1–6]. A common endpoint is a binary fixed-horizon survival label, such as survival beyond 12 months.

Constructing such a label is not trivial. It depends on recorded death times, follow-up, cohort-specific field conventions, and how censoring is handled. Under right censoring, a patient last seen alive before the horizon has an unknown status, not a survivor status [7–10].

Two literatures bear on this problem without addressing it directly:

- **Benchmark label errors.** Work on label errors in machine-learning benchmarks has shown that test-set errors can change model rankings [11]. Those errors are detected statistically and verified by people, outside clinical survival data.
- **Censoring-aware evaluation.** Methods such as inverse-probability-of-censoring weighting (IPCW) and partial identification assume the time-and-event data feeding a label are correctly encoded [7–10,12].

In oncology, curated endpoint resources and data-readiness audits improve endpoint definitions within a resource [13,14], and leakage audits examine temporal misalignment [15]. None reconstructs how a released benchmark label was generated from its upstream records, and none separates the effect of that label on evaluation from its effect on training.

Internal validation cannot detect such errors. Cross-validation estimates performance against the released label, often on folds stratified by that label, so a model of the wrong target scores well. When clinical features identify the source cohort, a flexible model can also learn a cohort-specific labelling artefact as if it were biology.

## Objective

To audit the provenance of released 12-month survival labels in public multimodal cancer benchmarks against upstream clinical records, and to quantify the consequences for model evaluation, training and comparison using paired designs on identical patients.

## Materials and methods

### Data sources

**MMIST-LUNG** [16] contains 1,365 patients: 522 TCGA-LUAD, 504 TCGA-LUSC and 339 CPTAC-3 lung. It provides a 12-month label (`vital_status_12`), clinical features, and five-fold cross-validation folds (`folds.npy`), pinned to repository commit `430670ce`.

**MMIST-ccRCC** [17], from the same initiative, contains 618 patients (429 TCGA-KIRC and 189 CPTAC-CCRCC) with a 12-month label and a released train/test split, pinned to commit `c3f1a066`.

**Source records** came from GDC Data Release 46.0 (10 August 2026) [1]. Two alternative releases describing the same participants were used for sensitivity analyses: the TCGA Clinical Data Resource (TCGA-CDR) [13] and LinkedOmics CPTAC tables [18,19]. Primary cohort references are [2–6,20,21].

**I3LUNG** [22], an independent multicentre immunotherapy cohort (Zenodo record 17535424; CC BY-NC 4.0), served as a negative control. It was used only to audit its own labels against its own time and event fields.

Every input was downloaded programmatically, and its SHA-256 hash and retrieval time were recorded. All data are public and de-identified, and no institutional data were used.

### Linkage and outcome derivation

Patients were linked deterministically on submitter identifier, and every join was validated one-to-one. For each patient we derived a three-state status at τ = 365 days from diagnosis:

- **Early death:** a recorded death on or before day 365.
- **Survivor:** a death after day 365, or last contact on or after day 365.
- **Not ascertainable:** everything else.

Last contact was the latest of the diagnosis-level and follow-up-level contact days. Records marked dead without a date, and negative times, were treated as not ascertainable. We also tested the alternative boundary convention, in which a death on day 365 counts as survival.

### Label audit

For each benchmark and cohort we:

1. **Reconstructed the generating rule.** We searched for deterministic rules that reproduce the released label from recorded death times. The candidates included the rule printed in the MMIST-LUNG encoding notebook.
2. **Quantified agreement.** Agreement between released and source-derived status was computed among ascertainable patients, with Wilson 95% CIs.
3. **Classified each disagreement** as a contradiction (opposite known outcome) or a forced outcome (an unascertainable status released as a class).
4. **Checked cross-release consistency.** We also reconstructed the released `folds.npy` from its generating code.

### Study design and estimands

Analyses followed two protocols:

- **Protocol v1** was frozen before any model was fitted, but after the labels had been explored.
- **Protocol v2** was committed to the public repository before any MMIST-ccRCC value or new model result was examined. It stated confirmatory hypotheses for a replication and for a refit-bootstrap test of the training-label effect.

All deviations are logged. Let $\tilde Y$ denote the released label, $Y$ the source-derived label, and $\mathcal A$ the patients with ascertainable $Y$. The estimands were:

| Estimand | Definition | Isolates |
|---|---|---|
| ψ_A, evaluation-label effect (v1 primary) | AUROC(ŝ, Y) − AUROC(ŝ, Ỹ) on $\mathcal A$, scores ŝ from a model trained on Ỹ | Change of scoring labels with predictions fixed |
| ψ_B, training-label effect (v2 confirmatory, C2) | AUROC(ŝ_Y, Y) − AUROC(ŝ_Ỹ, Y) | Change of fitting labels |
| Model-comparison shift | (AUROC_stage − AUROC_logistic) under Y minus the same under Ỹ | Whether labels change the gap between models |
| Selection effect | AUROC under Ỹ on all patients vs on $\mathcal A$ | Exclusion of unascertainable patients |

### Models

Models served as measurement instruments. The original MMIST pipelines could not be re-run, because their configuration files reference unreleased paths. We therefore used declared surrogates with hyperparameters fixed in advance and no tuning:

- **Prevalence only.**
- **Stage-only logistic regression,** which cannot identify the cohort.
- **L2-penalised logistic regression,** with ordinal features median-imputed with missing indicators and scaled, and nominal features one-hot encoded.
- **Histogram gradient boosting.**
- **XGBoost and a one-hidden-layer multilayer perceptron,** added in protocol v2 to match the benchmark's model families.

Predictors were the released clinical features (15 in MMIST-LUNG, 17 in MMIST-ccRCC). Outcome and follow-up fields were blocked in code, and all preprocessing was fitted within training folds.

### Validation and uncertainty

- **MMIST-LUNG validation** used the released folds in their reconstructed generation order. Sensitivity analyses applied them in the order used by the released code and in a new stratified split. Repeated cross-validation (50 × 5-fold) and a TCGA-to-CPTAC transport design were secondary analyses.
- **MMIST-ccRCC** analyses used 5-fold cross-validation stratified on cohort and outcome state.
- **Metrics** were fold-averaged AUROC, Brier score, scaled Brier score, balanced accuracy at the training-fold prevalence, and calibration intercept and slope.
- **Censoring-aware analyses** used Kaplan–Meier estimation [9], IPCW AUROC and Brier score at τ with reverse Kaplan–Meier weights [7,8], Harrell's C [10], and sharp partial-identification bounds on AUROC over all completions of unascertainable labels.
- **Uncertainty for ψ_A** came from a stratified patient bootstrap (2,000 replicates) conditional on the fitted models.
- **Uncertainty for the confirmatory ψ_B** came from a refit bootstrap (200 replicates): patients were resampled within each fold, and both models were refitted on every replicate, so the intervals include model-fitting uncertainty.
- **Multiplicity:** confirmatory tests were Holm-adjusted.

Reporting follows applicable TRIPOD+AI items [23].

### Reproducibility and ethics

Code, protocols, aggregate results and a test suite (more than 80 tests) are public. A run from a fresh clone reproduced all source hashes and deterministic result tables. Patient-level data are downloaded at runtime and not redistributed.

This was secondary analysis of public, de-identified data. Ethics requirements should be confirmed with the authors' institutions.

## Results

### MMIST-LUNG: two labelling rules with opposite polarity

All 1,365 patients linked to GDC. Cohort assignment matched GDC project membership in every case, and recorded death days matched GDC for 399 of 399 TCGA patients and 89 of 90 CPTAC patients.

Two rules reproduced every released label (Figure 2):

- **TCGA (1,026 of 1,026):** survivor if death occurred after day 365 or no death was recorded.
- **CPTAC (339 of 339):** survivor if death occurred on or before day 365 or no death was recorded, the rule printed in the benchmark's encoding notebook.

The CPTAC rule reverses the label for every patient with a recorded death. Among 1,154 ascertainable patients (160 source early deaths), 88 labels (7.6%; 95% CI 6.2–9.3) contradicted the source (Table 1):

- **CPTAC:** 88 of 260 (33.8%; 95% CI 28.4–39.8). All 62 released early deaths had survived beyond day 365, and all 26 source early deaths were labelled survivors. For true early death, the released label therefore had sensitivity 0 and positive predictive value 0.
- **TCGA:** 0 of 894 (95% CI 0.0–0.4).

A further **211 patients (15.5%)** had unascertainable status and were all labelled survivors. Of these, 168 were censored before day 365, 27 had vital status "Not Reported", 9 were recorded dead without a date, 6 had no follow-up, and 1 had a negative death time.

**Why the headline rate hid the errors.** Kaplan–Meier 12-month survival was 86.8% (95% CI 84.7–88.5) against 85.6% released. The overall rate agreed only because errors offset (Figure 3): for CPTAC squamous carcinoma the released rate was 76.2% against a Kaplan–Meier estimate of 93.5%.

**Cross-release checks.** Status agreed across GDC and the alternative releases for 1,070 of 1,071 jointly ascertainable patients. The released disease-specific label was 1 for every CPTAC patient, including 42 whose cause of death was recorded as cancer-related.

**Fold file.** The released folds were reproduced exactly (1,365 of 1,365) by stratified 5-fold splitting on the released label with seed 42, in an order that differs from the one the released code applies.

### Replication: MMIST-ccRCC and a negative control

**MMIST-ccRCC.** As pre-specified (C1), MMIST-ccRCC showed **no inversion**:

- **Recorded deaths:** 30 of 31 CPTAC and 169 of 169 TCGA ascertainable deaths agreed with the source. Only 1 of 580 ascertainable labels was contradicted.
- **Forced outcomes:** 38 CPTAC patients (20.1% of the cohort) had unascertainable status despite the paper's 12-month follow-up criterion:
  - 20 were censored before day 365 and labelled survivors.
  - 18 were labelled early deaths: 17 had vital status "Not Reported" with no follow-up, and 1 had a negative death time.
- **Cross-release check:** none of the 18 appeared in the LinkedOmics CPTAC tables, where GDC and LinkedOmics agreed for all 82 jointly ascertainable patients.

Of the 75 released early deaths, therefore, 56 were confirmed, 18 were unverifiable and 1 was contradicted. Because C1 found fewer than 20 contradictions, the pre-specified training-effect test (C3) was not applicable.

**I3LUNG.** The negative control was fully consistent. Its 6-month label agreed with its own time and event fields for 1,972 of 1,972 labelled patients, and its 24-month label for 1,699 of 1,699. Patients censored before the horizon were left blank (Figure 1).

**A missingness shortcut in MMIST-ccRCC (exploratory).** All 18 patients with pathological stage coded as missing were labelled early deaths, so missing stage predicts the released label perfectly (Table 3).

- **Released test set:** 4 of its 16 early-death labels had missing stage, and 5 of the 16 were unverifiable.
- **Effect on benchmark metrics:** we trained models on released labels and scored the same out-of-fold predictions on all 618 patients versus the 580 ascertainable patients:
  - logistic-regression AUROC was 0.832 versus 0.791 (difference 0.042; 95% CI 0.010–0.078);
  - balanced accuracy was 0.759 versus 0.719 (0.040; 0.016–0.068);
  - for the stage-only model the AUROC difference was 0.063 (0.036–0.099).

### MMIST-LUNG: evaluation-label effect (pre-specified, v1)

With predictions held fixed, logistic regression trained on released labels scored AUROC 0.634 against released labels and 0.600 against source-derived labels, on the same 1,154 patients (ψ_A −0.034; 95% CI −0.068 to −0.003; Table 2, Figure 4).

**This effect was small and not robust:**

- **Gradient boosting:** its interval crossed zero (−0.021; −0.055 to +0.013).
- **Sensitivity analyses:** the interval also crossed zero when flagged records were excluded (−0.027; −0.056 to +0.004), under a new fold split (−0.029; −0.065 to +0.004), and within CPTAC alone (+0.011; −0.206 to +0.140).
- **Fold alignment:** applying the released folds in the released code's order gave −0.042 (−0.077 to −0.008).
- **Selection:** excluding unascertainable patients changed released-label AUROC only from 0.637 to 0.634.

**Mechanism.** Pooled AUROC is a pair-weighted average of concordance, $\sum_{a,b} w_{ab} c_{ab}$, over case cohort $a$ and control cohort $b$:

- **Pair mix:** the inverted labels doubled the share of CPTAC-case, TCGA-control pairs (25.1% vs 12.4%). These pairs are easy to order (concordance 0.773), because a model trained on released labels ranks CPTAC as higher risk: mean predicted risk was 18.9% versus 13.3%, although the source early-death rate was 10.0% in CPTAC versus 15.0% in TCGA.
- **Attribution:** a Kitagawa decomposition attributed 78% of the logistic gap to changes in pair mix. In contrast, 94% of the stage-only model's gap (+0.024) came from concordance.
- **Model comparison:** switching labels shifted the stage-only-minus-logistic AUROC gap by 0.060 (95% CI 0.026–0.098).
- **Calibration** against source labels was poor for the released-trained logistic model (slope 0.27; scaled Brier score −0.014) and good for the stage-only model (slope 1.01; +0.026).

These mechanism analyses were post-hoc.

### MMIST-LUNG: training-label effect (confirmatory, v2)

**C2 was confirmed.** In the released folds, logistic regression trained on source-derived labels outperformed the same model trained on released labels when both were judged against source labels: ψ_B +0.049 (95% CI +0.008 to +0.091; refit bootstrap, 200 replicates; one-sided p = 0.010; Holm-adjusted p = 0.010; Figure 5).

**Pre-specified secondary analyses (S4)** were only partly supported:

- **XGBoost:** ψ_B +0.051 (95% CI −0.002 to +0.093; one-sided p = 0.050). The effect points the same way and is the same size, but the interval narrowly includes zero.
- **Multilayer perceptron:** ψ_B +0.028 (95% CI −0.093 to +0.101; p = 0.43). The estimate was unstable with only 160 events.

**Supporting analyses:**

- **Repeated cross-validation:** the gain appeared in 50 of 50 repeats (mean +0.061). Within CPTAC, AUROC rose from 0.56 to 0.73.
- **Benchmark lens:** judged by released labels, the source-trained model scored 0.073 lower (95% CI 0.032–0.116), so benchmark-style selection would prefer the model trained on inverted labels.
- **Transport:** a model trained on TCGA alone, which never saw inverted labels, scored higher against source than released labels in CPTAC (ψ_A +0.169; 95% CI +0.018 to +0.312). The direction of the evaluation effect therefore depends on whether the inverted labels entered training.

### Censoring handling

For the source-trained logistic model, AUROC changed little across methods:

| Method | AUROC |
|---|---:|
| Complete case | 0.646 |
| IPCW (maximum weight 1.20; effective n 1,152) | 0.648 |
| Unknowns imputed as survivors | 0.641 |
| Unknowns imputed as deaths | 0.578 |
| Sharp bounds over all completions | 0.468–0.734 |

The wide bounds show how much rests on non-informative censoring.

## Discussion

### Principal findings

Two benchmarks from one public initiative failed in different ways, while an independent resource did not fail at all:

- **MMIST-LUNG:** a cohort-specific polarity inversion, reconstructable to a single line of curation code, together with forced survival for unknown outcomes.
- **MMIST-ccRCC:** no inversion, but forced outcomes in both directions. The unverifiable deaths shared a missingness signature that models can exploit, and they inflated the benchmark's own metrics.
- **I3LUNG:** left censored outcomes blank.

The audit therefore discriminates between resources rather than finding problems everywhere.

### Where label errors do their damage

The effect of label errors fell less on scoring fixed predictions (small and fragile) than on what models learn and which model wins. The confirmatory test showed a training-label effect of about 0.05 AUROC for logistic regression, and benchmark-style evaluation reversed the ranking. For XGBoost and a multilayer perceptron the effect pointed the same way but was not resolved.

The pair-level decomposition explains why. When features identify the cohort, a model learns the cohort-level risk ordering implied by its labels. Pooled metrics then reward that ordering through between-cohort pairs. The same mechanism applies to any pooled benchmark with cohort-conditional label errors, including site, scanner or era effects.

### Comparison with prior work

Northcutt et al. showed that random-looking test-set errors can reorder models [11]. The errors here are systematic, cohort-conditional and deterministic, so they cannot be found by noise-detection methods that assume errors are scattered. Leakage audits [15] and censoring-aware evaluation [7,8,12] address different failure points, and binarisation critiques [24] argue for time-to-event targets, which would have avoided forced outcomes but not the inversion. Unlike work that harmonises raw records [14], we audited labels already released as a benchmark.

### Recommendations for benchmark curators and users

1. Release the code that generates labels, and test it against source records per cohort.
2. Keep an explicit "not ascertainable" state rather than forcing outcomes.
3. Release time-to-event fields alongside binary labels.
4. Report agreement between labels and source records per cohort.
5. Check whether any feature pattern, including missingness, predicts the label perfectly.

### Limitations

- **Scope:** we audited two benchmarks from one initiative and one independent resource, which supports a statement about these resources, not prevalence.
- **Source records** share upstream submissions and are not adjudicated clinical truth. Unverifiable labels may reflect records we could not access.
- **Surrogate models:** the published benchmark numbers were not re-estimated.
- **Events and intervals:** CPTAC lung had only 26 source early deaths. The ψ_A intervals are conditional on fitted models.
- **Post-hoc analyses:** the mechanism and MMIST-ccRCC metric analyses were post-hoc. The fold alignment the original authors used could not be determined, and changing it moved ψ_A from −0.042 to −0.034.
- **Prediction moment:** pathological stage is known only after surgery.

### Conclusion

Released survival labels in public benchmarks can be internally consistent, pass cross-validation, and still encode the wrong outcome for an entire cohort. Linking labels back to source records is a small step that catches what model-centric evaluation cannot.

---

## Declarations

**Author contributions (CRediT).** *To be completed by the authors.*

**Funding.** *To be completed by the authors.*

**Competing interests.** *To be completed by the authors.*

**Data availability.** All data are public. Retrieval scripts, pinned versions and hashes are at https://github.com/Isaac25-lgtm/lung-cancer-ml-survival-audit. Patient-level data are not redistributed.

**Code availability.** MIT-licensed code, protocols (v1 and v2, with commit history), aggregate results, figures and an interactive dashboard are at the repository above.

**Use of AI tools.** Code and manuscript drafting were assisted by Claude Code (Anthropic) under the authors' direction. The authors take full responsibility for the content.

**Acknowledgements.** We thank the MMIST team for openly sharing their datasets and code. Results are in part based upon data generated by the TCGA Research Network (https://www.cancer.gov/tcga) and the NCI Clinical Proteomic Tumor Analysis Consortium.

---

## References

1. Grossman RL, Heath AP, Ferretti V, et al. Toward a shared vision for cancer genomic data. *N Engl J Med*. 2016;375(12):1109-1112. doi:10.1056/NEJMp1607591
2. The Cancer Genome Atlas Research Network. Comprehensive molecular profiling of lung adenocarcinoma. *Nature*. 2014;511(7511):543-550. doi:10.1038/nature13385
3. The Cancer Genome Atlas Research Network. Comprehensive genomic characterization of squamous cell lung cancers. *Nature*. 2012;489(7417):519-525. doi:10.1038/nature11404
4. Gillette MA, Satpathy S, Cao S, et al. Proteogenomic characterization reveals therapeutic vulnerabilities in lung adenocarcinoma. *Cell*. 2020;182(1):200-225.e35. doi:10.1016/j.cell.2020.06.013
5. Satpathy S, Krug K, Jean Beltran PM, et al. A proteogenomic portrait of lung squamous cell carcinoma. *Cell*. 2021;184(16):4348-4371.e40. doi:10.1016/j.cell.2021.07.016
6. Li Y, Dou Y, Da Veiga Leprevost F, et al. Proteogenomic data and resources for pan-cancer analysis. *Cancer Cell*. 2023;41(8):1397-1406. doi:10.1016/j.ccell.2023.06.009
7. Graf E, Schmoor C, Sauerbrei W, Schumacher M. Assessment and comparison of prognostic classification schemes for survival data. *Stat Med*. 1999;18(17-18):2529-2545.
8. Uno H, Cai T, Tian L, Wei LJ. Evaluating prediction rules for t-year survivors with censored regression models. *J Am Stat Assoc*. 2007;102(478):527-537. doi:10.1198/016214507000000149
9. Kaplan EL, Meier P. Nonparametric estimation from incomplete observations. *J Am Stat Assoc*. 1958;53(282):457-481. doi:10.1080/01621459.1958.10501452
10. Harrell FE, Califf RM, Pryor DB, Lee KL, Rosati RA. Evaluating the yield of medical tests. *JAMA*. 1982;247(18):2543-2546. doi:10.1001/jama.1982.03320430047030
11. Northcutt CG, Athalye A, Mueller J. Pervasive label errors in test sets destabilize machine learning benchmarks. NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2103.14749
12. Kvamme H, Borgan Ø. The Brier score under administrative censoring: problems and solutions. arXiv:1912.08581. 2019.
13. Liu J, Lichtenberg T, Hoadley KA, et al. An integrated TCGA pan-cancer clinical data resource to drive high-quality survival outcome analytics. *Cell*. 2018;173(2):400-416.e11. doi:10.1016/j.cell.2018.02.052
14. Ahsan MM, Raman S, Siddique Z. Harmonizing public lung adenocarcinoma data for reproducible machine learning. NeurIPS 2026 Workshop on AI Data Readiness for Scientific Discovery.
15. Tadj L, Sufian MA, Hamzi W, Ali M, Ali A, Hamzi B. Auditing data leakage and temporal misalignment in machine-learning prognosis of breast cancer: a reproducible censoring-aware reanalysis. *BioMedInformatics*. 2026;6(4):60. doi:10.3390/biomedinformatics6040060
16. Cordeiro Mendes R, Fonseca Verdelho MR, Santiago C, Barata C. Real-world multi-modal and longitudinal lung cancer dataset. arXiv:2609.05202. 2026.
17. Mota T, Verdelho MR, Bissoto A, Santiago C, Barata C. MMIST-ccRCC: a real world medical dataset for the development of multi-modal systems. DCA in MI Workshop, CVPR 2024. arXiv:2405.01658
18. Vasaikar SV, Straub P, Wang J, Zhang B. LinkedOmics: analyzing multi-omics data within and across 32 cancer types. *Nucleic Acids Res*. 2018;46(D1):D956-D963. doi:10.1093/nar/gkx1090
19. Clark DJ, Dhanasekaran SM, Petralia F, et al. Integrated proteogenomic characterization of clear cell renal cell carcinoma. *Cell*. 2019;179(4):964-983.e31. doi:10.1016/j.cell.2019.10.007
20. The Cancer Genome Atlas Research Network. Comprehensive molecular characterization of clear cell renal cell carcinoma. *Nature*. 2013;499(7456):43-49. doi:10.1038/nature12222
21. Zheng Y, Conrad RD, Green EJ, et al. Graph attention-based fusion of pathology images and gene expression for prediction of cancer survival. *IEEE Trans Med Imaging*. 2024;43(9):3085-3097. doi:10.1109/TMI.2024.3386108
22. Prelaj A, Miskovic V, Sacco M, et al. Clinical usability of an explainable AI decision support tool and evaluation of multimodal models in NSCLC. *Nat Med*. 2026;32(9):3235-3247. doi:10.1038/s41591-026-04488-2
23. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. doi:10.1136/bmj-2023-078378
24. Yadav S, Routman DM, Foong AYK. The cost of binarizing survival outcomes in clinical prognostic modeling. arXiv:2608.04046. 2026.

---

## Tables

**Table 1. Label provenance by resource and cohort.** Ascertainable = 12-month status known from the source record. Forced = unascertainable status released as a class.

| Resource | Cohort | n | Ascertainable | Contradicted (% of ascertainable) | Forced as survivor | Forced as death | Generating rule reproduces |
|---|---|---:|---:|---:|---:|---:|---|
| MMIST-LUNG | TCGA | 1,026 | 894 | 0 (0.0) | 132 | 0 | TCGA rule, 1,026/1,026 |
| MMIST-LUNG | CPTAC | 339 | 260 | 88 (33.8) | 79 | 0 | Notebook rule, 339/339 |
| MMIST-ccRCC | TCGA | 429 | 429 | 0 (0.0) | 0 | 0 | TCGA rule, 429/429 |
| MMIST-ccRCC | CPTAC | 189 | 151 | 1 (0.7) | 20 | 18 | TCGA rule, 171/189 |
| I3LUNG (6-month) | All | 2,075 | 1,972 | 0 (0.0) | 0 | 0 | Unknowns left blank (103) |

**Table 2. MMIST-LUNG fixed-prediction contrasts on 1,154 ascertainable patients** (released folds; 95% bootstrap CI conditional on fitted models).

| Model | AUROC vs released | AUROC vs source | ψ_A (95% CI) |
|---|---:|---:|---|
| Logistic regression | 0.634 | 0.600 | −0.034 (−0.068 to −0.003) |
| Gradient boosting | 0.624 | 0.604 | −0.021 (−0.055 to +0.013) |
| Stage only | 0.609 | 0.635 | +0.026 (−0.011 to +0.063) |

**Table 3. MMIST-ccRCC released early-death labels by verification status and split.**

| Split | Patients | Released early deaths | Confirmed | Unverifiable | Contradicted | With missing stage |
|---|---:|---:|---:|---:|---:|---:|
| All | 618 | 75 | 56 | 18 | 1 | 18 |
| Train | 497 | 59 | 45 | 13 | 1 | 14 |
| Test | 121 | 16 | 11 | 5 | 0 | 4 |

## Figure legends

**Figure 1. Label integrity across three resources.** Released fixed-horizon label compared with 12-month status in GDC (MMIST datasets) or with the dataset's own time and event fields (I3LUNG). (`outputs/figures/readme/0_three_datasets.png`)

**Figure 2. Opposite-polarity labelling in MMIST-LUNG.** Released label by recorded day of death, TCGA and CPTAC. (`fig2_polarity_mechanism`)

**Figure 3. Kaplan–Meier 12-month survival versus released survivor fractions**, by cohort and histology. (`3_hidden_in_average`)

**Figure 4. Fixed-prediction contrasts and sensitivity analyses.** Blue intervals exclude zero, gray intervals include it. (`6_primary_contrast`)

**Figure 5. Training-label effect.** Models trained on released or source labels, judged under each label set. (`5_wrong_winner`)

**Supplementary Figure S1. Cohort-risk reversal and pair-level AUROC decomposition.** (`4_cohort_flip`)

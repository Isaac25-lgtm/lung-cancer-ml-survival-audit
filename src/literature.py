"""Literature and novelty register (written to outputs/tables/literature_register.csv).

Search log (2026-11-05): web search (standard) for the exact dataset name/arXiv id,
its GitHub issues (1 open issue: Hugging Face hosting request, no label report),
'binary fixed-horizon survival censored treated as survivors', 'label errors test
sets benchmarks', 'CPTAC TCGA lung survival label inconsistency audit',
'TRIPOD+AI'. DOIs verified via api.crossref.org; arXiv ids via export.arxiv.org.
Entries 5-6 reviewed at abstract level only; entry 2 metadata only. A complete
systematic search (PubMed, Scopus, Google Scholar citation tracking of the
MMIST-LUNG preprint) is still required before submission.
"""
import pandas as pd

from src.paths import TABLES

ROWS = [
    dict(rank=1, title="Real-World Multi-Modal and Longitudinal Lung Cancer Dataset (MMIST-LUNG)",
         authors="Cordeiro Mendes R, Fonseca Verdelho MR, Santiago C, Barata C", year=2026,
         type="arXiv preprint v1 (2026-09-04); accepted ECCV 2026 DCA-MI workshop",
         url="https://arxiv.org/abs/2609.05202",
         population="1365 lung cancer patients: TCGA-LUAD/LUSC 1026, CPTAC-3 339",
         outcome="12-month OS (vital_status_12), DSS, dynamic-horizon DSS, discrete hazard",
         methods="Transformer/mean/max early fusion, late fusion; MLP and XGBoost; 5-fold CV; balanced accuracy",
         main_result="CLIN 12-month BAcc 56.7 (MLP) / 65.0 (XGB); multimodal more robust when stage removed",
         relation="EXACT dataset: the benchmark whose labels are audited.",
         our_critique="Released 12-month labels follow opposite-polarity rules by cohort (CPTAC inverted for recorded "
                      "deaths); patients without a recorded death are labelled survivors irrespective of follow-up; "
                      "the CPTAC DSS label is constant. Folds were stratified on these labels. The published numbers "
                      "were not re-run (original code reads non-released encoded paths); a declared surrogate is used."),
    dict(rank=2, title="Auditing Data Leakage and Temporal Misalignment in Machine-Learning Prognosis of Breast Cancer: "
                       "A Reproducible Censoring-Aware Reanalysis",
         authors="Tadj et al.", year=2026, type="Journal article, BioMedInformatics 6(4):60 (2026-08-14)",
         url="https://doi.org/10.3390/biomedinformatics6040060",
         population="Public SEER-derived breast cancer extract (IEEE DataPort), 4,023 women, 616 deaths; no patient identifiers", outcome="End-of-follow-up vital status (audited) and time to all-cause death (Weibull AFT)",
         methods="Reproduced an SVM workflow; row-hash overlap; split-first and baseline-only reanalyses; penalized Weibull AFT", main_result="Pre-split oversampling: 50.5% of test rows duplicated in training, AUROC 0.998, falling to 0.715 split-first; baseline-only SVM AUROC 0.745; adding survival months gives 0.895; AFT C-index 0.753",
         relation="Closest generic methodological audit, different cancer.",
         our_critique="Full text reviewed (2026-11-05). Closest generic audit, but of the modelling PIPELINE (resampling order, post-baseline predictor) on one unlinkable dataset; the authors state patient-level linkage was not possible. We audit the OUTCOME LABEL via source linkage, in pipelines without leakage. Complementary, not overlapping."),
    dict(rank=3, title="Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks",
         authors="Northcutt CG, Athalye A, Mueller J", year=2021,
         type="NeurIPS 2021 Datasets and Benchmarks (arXiv 2103.14749)", url="https://arxiv.org/abs/2103.14749",
         population="10 vision/NLP/audio benchmarks", outcome="Test-set label errors",
         methods="Confident learning plus human verification",
         main_result="About 3.3% average test-label error; rankings can flip at realistic error prevalence",
         relation="General benchmark label-error literature.",
         our_critique="Their errors are found statistically and confirmed by humans; ours are reconstructed "
                      "deterministically from code and upstream records in a clinical survival benchmark, are "
                      "systematic (inverted) rather than random, and are concentrated in one cohort."),
    dict(rank=4, title="Label-Space Mismatch Distorts Cross-Cohort Evaluation in Breast Cancer Molecular Subtyping",
         authors="Liu S, Min B, Zheng Z, Wang J, et al.", year=2026, type="arXiv preprint 2610.24962 (2026-10-18)",
         url="https://arxiv.org/abs/2610.24962", population="TCGA-BRCA, GSE21653", outcome="PAM50 subtype",
         methods="Label-space restriction, EM prior correction",
         main_result="Companion dataset page: TCGA-BRCA accuracy 73.8% to 79.7% under restriction",
         relation="Label definition distorting cross-cohort evaluation; different task.",
         our_critique="Adjacent concept, not survival, censoring or polarity. Do not conflate."),
    dict(rank=5, title="Harmonizing Public Lung Adenocarcinoma Data for Reproducible Machine Learning",
         authors="Ahsan MM, Raman S, Siddique Z", year=2026, type="NeurIPS 2026 workshop poster (AIDaR)",
         url="https://neurips.cc/virtual/2026/162364", population="TCGA-LUAD, 100-patient audit slice",
         outcome="Censoring-aware 1/2/3-year mortality", methods="Schema harmonization; random forest; Cox",
         main_result="RF AUROC 0.768 at 1 y vs 0.469 at 3 y; eligibility 85 to 46",
         relation="Closest lung-specific data-readiness work.",
         our_critique="Harmonizes raw TCGA-LUAD fields; does not audit a released multi-cohort benchmark label, CPTAC, "
                      "or label polarity. Abstract-level review only."),
    dict(rank=6, title="The Cost of Binarizing Survival Outcomes in Clinical Prognostic Modeling",
         authors="Yadav S, Routman DM, Foong AYK", year=2026, type="arXiv preprint 2608.04046 (2026-08-04)",
         url="https://arxiv.org/abs/2608.04046", population="Head and neck; breast, colorectal, kidney",
         outcome="Binarized vs time-to-event", methods="Survival-aware Bayesian network",
         main_result="Time-to-event scoring recovers features binarization misses (abstract)",
         relation="Methodological argument against binarized endpoints.",
         our_critique="Supports the censoring argument; does not address released-label provenance. Abstract only."),
    dict(rank=7, title="The Brier Score under Administrative Censoring: Problems and Solutions",
         authors="Kvamme H, Borgan O", year=2019, type="arXiv 1912.08581 (later JMLR)",
         url="https://arxiv.org/abs/1912.08581", population="Methodological", outcome="Brier score under censoring",
         methods="IPCW and administrative Brier",
         main_result="Removing censored individuals biases binary classifiers' apparent performance",
         relation="Method basis for Experiment C.",
         our_critique="We apply IPCW at 365 days and add sharp partial-identification bounds; no new method claimed."),
    dict(rank=8, title="An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome "
                       "Analytics (TCGA-CDR)",
         authors="Liu J, Lichtenberg T, Hoadley KA, et al.", year=2018, type="Journal article, Cell 173(2):400-416",
         url="https://doi.org/10.1016/j.cell.2018.02.052", population="11,160 TCGA patients",
         outcome="OS, DSS, DFI, PFI", methods="Curated endpoints and usage recommendations",
         main_result="Standard curated TCGA survival endpoints",
         relation="Alternative source release for TCGA (sensitivity analysis).",
         our_critique="Agrees with GDC on 365-day status for 893/894 jointly ascertainable TCGA patients."),
    dict(rank=9, title="Graph Attention-Based Fusion of Pathology Images and Gene Expression for Prediction of "
                       "Cancer Survival",
         authors="Zheng Y, Conrad RD, Green EJ, Burks EJ, Betke M, Beane JE, et al.", year=2024,
         type="Journal article, IEEE Trans Med Imaging", url="https://doi.org/10.1109/TMI.2024.3386108",
         population="TCGA training, CPTAC external testing", outcome="Survival",
         methods="Graph-attention multimodal fusion", main_result="TCGA to CPTAC multimodal survival modelling",
         relation="Shows generic TCGA-to-CPTAC ML is not novel.",
         our_critique="Our transport design (A2) is a label-evaluation device, not a new model."),
    dict(rank=10, title="Clinical usability of an explainable AI decision support tool and evaluation of multimodal "
                        "models in NSCLC (I3LUNG)",
         authors="Prelaj A, Miskovic V, et al.", year=2026, type="Journal article, Nature Medicine (2026-09)",
         url="https://doi.org/10.1038/s41591-026-04488-2",
         population="Advanced NSCLC on (chemo-)immunotherapy, multicentre", outcome="OS_6, OS_24, others",
         methods="Multimodal models, external validation, usability study",
         main_result="Public bundle: Zenodo record 17535424 (CC BY-NC 4.0)",
         relation="Independent case study for the audit procedure (negative control).",
         our_critique="OS_6 and OS_24 are fully consistent with the bundle's OS time/event fields and censored-before-"
                      "horizon cases are left missing. Not external validation of MMIST models."),
    dict(rank=11, title="TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use "
                        "regression or machine learning methods",
         authors="Collins GS, Moons KGM, Dhiman P, et al.", year=2024, type="Reporting guideline, BMJ 385:e078378",
         url="https://doi.org/10.1136/bmj-2023-078378", population="-", outcome="-", methods="27-item checklist",
         main_result="-", relation="Reporting guidance for model-evaluation components.",
         our_critique="Partly applicable: this study evaluates benchmark labels, not a model for clinical use."),
]


def run() -> pd.DataFrame:
    df = pd.DataFrame(ROWS)
    df.insert(1, "searched_utc", "2026-11-05")
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "literature_register.csv", index=False)
    return df


if __name__ == "__main__":
    print(len(run()), "entries")

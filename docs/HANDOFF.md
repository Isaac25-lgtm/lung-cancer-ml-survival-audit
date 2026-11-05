# Handoff: survival label audit of public multimodal cancer benchmarks

**Date:** 2026-11-05 · **Repository:** https://github.com/Isaac25-lgtm/lung-cancer-ml-survival-audit (branch `main`)

**Authors:** Omoding Isaac (corresponding, Ministry of Health; omodingisaac111@gmail.com), Echodu Moses (Uganda Cancer Institute), Bob Tumushime (Uganda Bureau of Statistics), Noah Okiror (Malaria Consortium), Daniel Baguma (Malaria Consortium)

This document lets a new collaborator, human or AI assistant, continue the work without re-deriving it. Read sections 1, 6 and 7 before changing anything.

---

## 1. What this project is

The project audits the 12-month survival labels released with public multimodal cancer benchmarks. It checks each label against upstream clinical records, then measures how label errors affect machine-learning training and evaluation.

The goal is a **publishable short research article**, with the repository serving as the reproducible evidence base. A submission draft exists:

- `docs/manuscript.md` (source)
- `outputs/manuscript/manuscript.docx` (generated)

## 2. Key results (all regenerated from `outputs/tables/`)

| Finding | Number | Status |
|---|---|---|
| MMIST-LUNG labels follow two opposite-polarity rules | TCGA 1,026/1,026; CPTAC 339/339 reproduced | Deterministic |
| MMIST-LUNG CPTAC labels contradict source records | 88 of 260 (33.8%; 95% CI 28.4–39.8); every released early death is wrong | Deterministic |
| Unknown 12-month status released as survivor (MMIST-LUNG) | 211 of 1,365 (15.5%) | Deterministic |
| MMIST-ccRCC replication | No inversion: recorded deaths agree 30/31 (CPTAC) and 169/169 (TCGA) | Pre-registered (C1) |
| MMIST-ccRCC forced outcomes | 38 CPTAC patients with unknown status: 20 labelled survivors, 18 labelled deaths (unverifiable, all with missing stage) | Deterministic plus exploratory |
| Unverifiable ccRCC deaths inflate benchmark metrics | Logistic AUROC 0.791 → 0.832 (+0.042; 0.010–0.078) | Exploratory |
| I3LUNG negative control | 6-month label 1,972/1,972 consistent; censored patients left blank | Deterministic |
| Evaluation-label effect, predictions fixed (MMIST-LUNG) | ΔAUROC −0.034 (−0.068 to −0.003); fragile across sensitivity analyses | Pre-specified (v1), **weak** |
| Training-label effect (MMIST-LUNG, refit bootstrap) | +0.049 (+0.008 to +0.091), p = 0.010 | **Confirmed** (v2, C2) |
| Same effect with XGBoost and MLP | +0.051 (−0.002 to +0.093); +0.028 (−0.093 to +0.101) | Secondary (S4); borderline and inconclusive |
| Mechanism: labels reverse the apparent cohort risk | CPTAC vs TCGA early-death rate 23.8% vs 15.0% (released) and 10.0% vs 15.0% (source); 78% of the AUROC gap comes from the mix of case-control pairs | Post-hoc |

A full narrative with every number is in `docs/findings.md`.

## 3. File map

| Path | Contents |
|---|---|
| `README.md` | Public, expert-facing summary with story charts, gap analysis, data and credits, and references |
| `docs/manuscript.md`, `outputs/manuscript/manuscript.docx` | **Submission draft** (250-word abstract, about 3,100 words, 25 references, 3 tables, 5 figures plus S1) |
| `docs/findings.md` | Full results, decision gates and publication assessment |
| `docs/protocol.md`, `docs/protocol_v2.md` | Pre-specified plans. v2 was committed (`f3d3e14`) before the ccRCC data were opened |
| `docs/changelog.md` | Every deviation and post-hoc addition, dated |
| `docs/methods.md`, `docs/data_dictionary.md`, `docs/source_provenance.md` | Methods detail, variables, sources and versions |
| `docs/manuscript_draft.md` | Older outline and evidence ledger, superseded by `manuscript.md` |
| `src/` | Pipeline (see section 4) |
| `tests/` | 87 tests: outcome rules, metrics, leakage guards, linkage, and README and manuscript numbers |
| `outputs/tables/` | All aggregate results (no patient identifiers) |
| `outputs/figures/` | Publication figures (SVG, PDF, PNG) plus `readme/` story charts and `captions.md` |
| `outputs/reports/` | `results.json`, `v2_lung_results.json`, `v2_ccrcc_results.json`, table hashes |
| `dashboard/` | Streamlit app, reading aggregate tables only |
| `configs/` | Pinned sources and the v1 analysis protocol (YAML) |
| `LICENSE`, `LICENSE_OR_REUSE_NOTES.md` | MIT for code; data reuse terms |

## 4. How to reproduce

Requires Python 3.11 or 3.12, about 15 minutes on a laptop, and internet access for public downloads.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-lock.txt   # on Linux/macOS: .venv/bin/pip
python -m src.run_analysis        # v1: download, link, audit, experiments A–D, figures
python -m src.v2 all              # v2: MMIST-ccRCC replication and confirmatory tests
python -m src.make_story_figures  # README charts
python -m src.build_manuscript_docx
python -m pytest
streamlit run dashboard/app.py
```

**Raw data is not included.** `data/raw/` is re-downloaded with SHA-256 pinning. MMIST-LUNG has no licence and I3LUNG is CC BY-NC 4.0, so only aggregates are redistributed.

## 5. Methods in one paragraph

Patients were linked one-to-one to GDC Data Release 46.0, with TCGA-CDR and LinkedOmics as alternative releases. Each patient gets a three-state 365-day status: early death, survivor, or not ascertainable. Released labels were compared per cohort and their generating rules reconstructed. The models are surrogates (logistic regression, gradient boosting, XGBoost, MLP, stage-only, prevalence), all on released clinical features with preprocessing fit inside folds, in the reconstructed released folds. The estimands are paired, on identical patients:

- **ψ_A:** the evaluation-label effect.
- **ψ_B:** the training-label effect.
- **Model-gap interaction:** whether labels change the gap between two models.

Uncertainty comes from a stratified patient bootstrap, plus a refit bootstrap for confirmatory tests. Censoring analyses use IPCW and sharp partial-identification bounds.

## 6. Integrity rules for anyone continuing

- **Do not re-analyse MMIST-LUNG to strengthen the weak primary contrast.** It is reported as weak. New claims need new pre-specified tests, ideally on new data.
- **Keep the labels "pre-specified", "confirmatory", "post-hoc" and "exploratory"** exactly as given in `docs/changelog.md`.
- **Do not claim the published MMIST numbers are wrong by a specific amount.** Their pipelines were not re-run; our models are surrogates.
- **Treat source records as upstream records, not adjudicated truth.** The 18 unverifiable ccRCC deaths may come from records we cannot access.
- **Do not claim clinical utility** or relevance to any particular population. No Uganda Cancer Institute or other institutional data were used.
- **Commit new hypotheses publicly before analysing new data,** as was done for protocol v2.
- **Keep numbers tested.** `tests/test_readme.py` and `tests/test_manuscript.py` check the documents against the tables.

## 7. Open tasks before submission

1. **Declarations:** fill in CRediT contributions, funding and competing interests (marked "to be completed" in the manuscript). All co-authors must approve.
2. **Affiliations:** confirm exact names and titles, and whether "Ministry of Health" should read "Ministry of Health, Uganda".
3. **Responsible disclosure:** contact the MMIST authors (corresponding contact listed on their project site) before submission. The findings are already public in the repository. Consider a preprint.
4. **Systematic literature search:** run PubMed, Scopus and Google Scholar, with citation tracking of arXiv 2609.05202 and 2405.01658. Tadj et al. (2026) has been read in full and is compared in the manuscript.
5. **Target journal:** the draft is in *JAMIA Open* style. Verify scope, word limits, figure rules (journals often want figures without embedded titles; the README charts have takeaway titles) and fees.
6. **Ethics:** confirm institutional requirements for secondary analysis of public, de-identified data.
7. **Optional strengthening:**
   - Re-run the authors' own MLP/Transformer pipelines to estimate the effect on published numbers.
   - Audit further public pooled-cohort benchmarks under a new pre-registered protocol.
   - Refit-bootstrap the remaining post-hoc estimates.

## 8. Commit history (key points)

| Commit | What |
|---|---|
| `47e9ec9` | Pipeline and v1 analysis |
| `f3d3e14` | **Protocol v2 frozen** (before ccRCC data) |
| `7b133a4` | Co-authors, ccRCC replication, manuscript draft |
| `2135926` | Tadj et al. comparison finalised |

`git log` holds the full history.

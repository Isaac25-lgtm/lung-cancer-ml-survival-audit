# Survival label audit of the MMIST-LUNG benchmark

A reproducible, source-linked audit of the 12-month survival labels in [MMIST-LUNG](https://github.com/ritacmendes/MMIST-LUNG), a public multimodal lung cancer benchmark of 1,365 patients, and of how those labels affect machine-learning evaluation.

**This is a research audit of a benchmark's labels. It is not a clinical tool, and not peer reviewed.**

## Main findings

**The label audit (deterministic):**

| Finding | Result |
|---|---|
| The released label comes from two rules with opposite polarity | TCGA 1,026/1,026 reproduced by *survivor if death after day 365 or no death recorded*; CPTAC 339/339 reproduced by *survivor if death on or before day 365 or no death recorded* (the rule in the authors' encoding notebook) |
| Agreement with upstream GDC records (365-day status) | TCGA 0 of 894 inconsistent; **CPTAC 88 of 260 inconsistent (33.8%)**. No CPTAC released early-death label is a true early death |
| Patients whose 12-month status cannot be ascertained | 211 of 1,365 (15.5%), all released as survivors |
| Negative control (I3LUNG) | Labels fully consistent with that dataset's own time and event fields; the audit raises no false alarm |

**The consequences for machine learning (surrogate models, released folds):**

| Finding | Result | Strength |
|---|---|---|
| Labels reverse which cohort looks higher-risk | Early-death rate, CPTAC vs TCGA: 23.8% vs 15.0% released, 10.0% vs 15.0% source-derived. Models trained on each label set learn that ordering | Strong; post-hoc |
| Training on source-derived labels | AUROC +0.049 (95% CI 0.015 to 0.082) judged by source labels, but −0.073 (−0.116 to −0.032) judged by released labels | Moderate to strong; post-hoc |
| Standing against a stage-only model | Switching labels shifts the stage-only minus logistic AUROC gap by +0.060 (0.026 to 0.098) | Moderate; post-hoc |
| Same predictions, different labels (**pre-specified primary contrast**) | Logistic AUROC 0.634 vs released, 0.600 vs source-derived (Δ −0.034, −0.068 to −0.003) | **Weak**: small, and some sensitivity analyses cross zero |

Full report: [docs/findings.md](docs/findings.md). **Verdict: conditional go**, best suited to a short communication or benchmark-audit paper.

### What this does not show

- The authors' published numbers were **not** re-run. Models here are declared surrogates (logistic regression and gradient boosting on the same features and folds).
- GDC is the upstream record, not adjudicated clinical truth.
- The leakage experiment is hypothetical and does not describe the authors' configuration.
- Nothing here speaks to clinical utility or to any other patient population.

## Quickstart

Requires Python 3.11 or 3.12. A full run takes about 15 minutes on a laptop CPU (the first run also downloads data), needs under 2 GB of RAM and about 100 MB of disk. Public data are downloaded without credentials.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-lock.txt   # on Linux/macOS: .venv/bin/pip

.venv/Scripts/python -m src.run_analysis             # download, audit, models, figures (--quick for 5 CV repeats)
.venv/Scripts/python -m pytest                       # tests
.venv/Scripts/streamlit run dashboard/app.py         # dashboard
```

The dashboard runs from the committed aggregate tables, so you can view it without downloading any data or re-running the analysis.

## Repository layout

| Path | Contents |
|---|---|
| `src/download_data.py`, `src/download_api.py` | Pinned downloads with SHA-256 manifest |
| `src/reproduce_handover.py` | Gate: reproduces the 35 feasibility-stage numbers |
| `src/link_sources.py`, `src/derive_outcomes.py` | Deterministic linkage and three-state outcome derivation |
| `src/audit_labels.py` | Rule reproduction, agreement, Kaplan–Meier anchors, DSS check |
| `src/experiments.py` | Experiments A to D, sensitivity analyses, falsification checks |
| `src/fold_provenance.py` | Reconstructs how the released `folds.npy` was generated |
| `src/i3lung.py` | Independent label-provenance case study |
| `src/make_figures.py` | Publication figures (SVG, PDF, PNG), captions and alt text |
| `dashboard/` | Streamlit app, reading aggregate tables only |
| `configs/` | Pinned sources and machine-readable protocol |
| `docs/` | Protocol, provenance, findings, methods, literature register, manuscript draft, changelog |
| `outputs/tables/`, `outputs/figures/` | Aggregate results (no patient identifiers) |

## Data

Raw and patient-level data are **not** in this repository. MMIST-LUNG has no licence file, so it is fetched at runtime. Only aggregate tables are committed. See [LICENSE_OR_REUSE_NOTES.md](LICENSE_OR_REUSE_NOTES.md).

## Citing

Cite the original resources: MMIST-LUNG (Cordeiro Mendes et al., arXiv:2609.05202), GDC, TCGA-CDR (Liu et al., Cell 2018), and I3LUNG (Prelaj et al., Nature Medicine 2026) if you use that case study.

Code: MIT licence. Author: Isaac Omoding.

"""Experiments A-D, sensitivity analyses and falsification checks.

Orientation: y = 1 early death (<=365 d), score = P(early death).
Primary contrast (configs/analysis.yaml): Experiment A, logistic regression,
fold-averaged AUROC against source-derived vs released labels on identical
ascertainable patients, in the released benchmark folds.

All outputs are aggregate. Writes outputs/tables/exp_*.csv and outputs/reports/results.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src import metrics as M
from src.folds import benchmark_order, load_folds
from src.io import ID
from src.link_sources import load_linked
from src.models import CLIN, STAGE, fit_predict, oof_predict
from src.paths import CONFIGS, REPORTS, TABLES

CFG = yaml.safe_load((CONFIGS / "analysis.yaml").read_text())
SEED = int(CFG["seed"])
REPS = int(CFG["bootstrap_reps"])
MODELS = {"logistic": CLIN, "hgb": CLIN, "stage_only": STAGE, "prevalence": CLIN}


def load() -> tuple[pd.DataFrame, list]:
    # Rows follow the order in which folds.npy was generated, so positional fold indices
    # refer to the patients they were created for (verified in src/fold_provenance.py).
    L = load_linked().set_index(ID).loc[benchmark_order()].reset_index()
    folds = load_folds()
    fold_id = np.empty(len(L), int)
    for k, (_, te) in enumerate(folds):
        fold_id[te] = k
    L["fold"] = fold_id
    return L, folds


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fold_avg(fn, y, s, fold, idx=None, **kw) -> float:
    """Mean of per-fold metric (benchmark convention). idx: optional bootstrap index."""
    y, s, fold = np.asarray(y, float), np.asarray(s, float), np.asarray(fold)
    if idx is not None:
        y, s, fold = y[idx], s[idx], fold[idx]
    vals = []
    for k in np.unique(fold):
        m = fold == k
        extra = {kk: (np.asarray(v)[idx] if idx is not None else np.asarray(v))[m] for kk, v in kw.items()}
        vals.append(fn(y[m], s[m], **extra))
    return float(np.nanmean(vals))


def _bacc(y, s, thr):
    return M.balanced_accuracy(y, s, float(np.nanmean(thr)))


def metric_row(label: str, y, s, fold, thr, mask=None) -> dict:
    y = np.asarray(y, float)
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    m &= ~np.isnan(y)
    yy, ss, ff, tt = y[m], s[m], fold[m], thr[m]
    cil, slope = M.calibration(yy, ss)
    return {
        "evaluation": label, "n": int(m.sum()), "events": int(yy.sum()),
        "auroc_foldavg": fold_avg(M.auroc, yy, ss, ff),
        "auroc_pooled": M.auroc(yy, ss),
        "brier_foldavg": fold_avg(M.brier, yy, ss, ff),
        "scaled_brier_pooled": M.scaled_brier(yy, ss),
        "bacc_foldavg": fold_avg(_bacc, yy, ss, ff, thr=tt),
        "auprc_pooled": M.auprc(yy, ss),
        "cal_intercept": cil, "cal_slope": slope,
    }


def paired_delta(y_a, y_b, s, fold, strata, fn, reps=REPS, seed=SEED, thr=None) -> dict:
    """Bootstrap the fold-averaged metric difference fn(y_a) - fn(y_b) on identical patients."""
    y_a, y_b, s = (np.asarray(v, float) for v in (y_a, y_b, s))
    kw = {} if thr is None else {"thr": thr}
    point_a = fold_avg(fn, y_a, s, fold, **kw)
    point_b = fold_avg(fn, y_b, s, fold, **kw)
    a, b = M.paired_bootstrap(lambda i: fold_avg(fn, y_a, s, fold, i, **kw),
                              lambda i: fold_avg(fn, y_b, s, fold, i, **kw),
                              len(s), strata, reps, seed)
    lo, hi = M.ci(a - b)
    return {"a": point_a, "b": point_b, "delta": point_a - point_b, "delta_ci_lo": lo, "delta_ci_hi": hi,
            "a_ci": M.ci(a), "b_ci": M.ci(b), "reps": reps}


# ---------------------------------------------------------------------------
# Experiment A: evaluation-label effect with fixed out-of-fold scores
# ---------------------------------------------------------------------------

def experiment_a(L, folds, src="y_gdc_gt", tag="primary"):
    rows, deltas, scores = [], [], {}
    known = L[src].notna().to_numpy()
    fold = L.fold.to_numpy()
    for kind, feats in MODELS.items():
        s, thr = oof_predict(L, L.y_released_death.to_numpy(), folds, kind, feats)
        scores[kind] = (s, thr)
        for label, y, mask in [("released labels, all patients (benchmark setting)", L.y_released_death, None),
                               ("released labels, ascertainable patients", L.y_released_death, known),
                               ("source-derived labels, ascertainable patients", L[src], known)]:
            r = metric_row(label, y, s, fold, thr, mask)
            rows.append({"analysis": tag, "source": src, "model": kind, "trained_on": "released", "cohort": "All", **r})
            for cohort in ["TCGA", "CPTAC"]:
                cm = (L.cohort == cohort).to_numpy() & (np.ones(len(L), bool) if mask is None else mask)
                yy = np.asarray(y, float)
                rows.append({"analysis": tag, "source": src, "model": kind, "trained_on": "released",
                             "cohort": cohort, "evaluation": label, "n": int(cm.sum()),
                             "events": int(np.nansum(yy[cm])), "auroc_pooled": M.auroc(yy[cm], s[cm]),
                             "brier_foldavg": np.nan, "auroc_foldavg": np.nan,
                             "scaled_brier_pooled": M.scaled_brier(yy[cm], s[cm]),
                             "auprc_pooled": M.auprc(yy[cm], s[cm])})
        # paired contrasts on identical ascertainable patients
        k = known
        strata = (fold[k] * 2 + (L.cohort[k] == "CPTAC").to_numpy()).astype(int)
        for metric, fn, kw in [("AUROC", M.auroc, {}), ("Brier", M.brier, {}),
                               ("Balanced accuracy", _bacc, {"thr": thr[k]})]:
            d = paired_delta(L[src][k], L.y_released_death[k], s[k], fold[k], strata, fn, **kw)
            deltas.append({"analysis": tag, "source": src, "model": kind, "metric": metric,
                           "contrast": "source - released (same patients, same scores)",
                           "n": int(k.sum()), "events_source": int(L[src][k].sum()),
                           "events_released": int(L.y_released_death[k].sum()),
                           "value_source": d["a"], "value_released": d["b"], "delta": d["delta"],
                           "ci_lo": d["delta_ci_lo"], "ci_hi": d["delta_ci_hi"], "reps": d["reps"]})
    return pd.DataFrame(rows), pd.DataFrame(deltas), scores


# ---------------------------------------------------------------------------
# Experiment A2: TCGA -> CPTAC transport with fixed scores
# ---------------------------------------------------------------------------

def experiment_a2(L):
    rows = []
    tr = L[L.cohort == "TCGA"]
    te = L[(L.cohort == "CPTAC")].reset_index(drop=True)
    for kind in ["logistic", "hgb", "stage_only"]:
        s, thr = fit_predict(tr, tr.y_gdc_gt.to_numpy(), te, kind, MODELS[kind])
        for subset, src in [("CPTAC ascertainable (GDC)", "y_gdc_gt"), ("CPTAC LinkedOmics-ascertainable", "y_lo_gt")]:
            k = te[src].notna().to_numpy()
            yS, yR, sk = te[src].to_numpy()[k], te.y_released_death.to_numpy()[k], s[k]
            strata = (te.histology[k] == "LUSC").to_numpy().astype(int)
            a, b = M.paired_bootstrap(lambda i: M.auroc(yS[i], sk[i]), lambda i: M.auroc(yR[i], sk[i]),
                                      len(sk), strata, REPS, SEED)
            ab, bb = M.paired_bootstrap(lambda i: M.brier(yS[i], sk[i]), lambda i: M.brier(yR[i], sk[i]),
                                        len(sk), strata, REPS, SEED)
            rows.append({"model": kind, "train": "TCGA ascertainable (n=%d, %d events)" % (
                tr.y_gdc_gt.notna().sum(), tr.y_gdc_gt.sum()), "test": subset, "n": int(k.sum()),
                "events_source": int(yS.sum()), "events_released": int(yR.sum()),
                "auroc_source": M.auroc(yS, sk), "auroc_released": M.auroc(yR, sk),
                "auroc_delta": M.auroc(yS, sk) - M.auroc(yR, sk),
                "auroc_delta_ci_lo": M.ci(a - b)[0], "auroc_delta_ci_hi": M.ci(a - b)[1],
                "brier_source": M.brier(yS, sk), "brier_released": M.brier(yR, sk),
                "brier_delta": M.brier(yS, sk) - M.brier(yR, sk),
                "brier_delta_ci_lo": M.ci(ab - bb)[0], "brier_delta_ci_hi": M.ci(ab - bb)[1],
                "prevalence_brier_source": M.brier(yS, np.full(len(yS), thr)),
                "prevalence_brier_released": M.brier(yR, np.full(len(yR), thr))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment B: training-label effect (repeated CV)
# ---------------------------------------------------------------------------

VARIANTS = {
    "B1 released labels, all training patients (benchmark practice)": ("y_released_death", False),
    "B2 released labels, ascertainable training patients only": ("y_released_death", True),
    "B3 source-derived labels, ascertainable training patients": ("y_gdc_gt", True),
}


def experiment_b(L, n_repeats=50, models=("logistic", "hgb")):
    known = L.y_gdc_gt.notna().to_numpy()
    state = np.where(L.y_gdc_gt.isna(), 2, L.y_gdc_gt.fillna(0)).astype(int)
    strat = (L.cohort == "CPTAC").to_numpy().astype(int) * 3 + state
    rows = []
    for r in range(n_repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=1000 + r)
        folds = list(skf.split(np.zeros(len(L)), strat))
        fold = np.empty(len(L), int)
        for k, (_, te) in enumerate(folds):
            fold[te] = k
        for kind in models:
            for vname, (ycol, known_only) in VARIANTS.items():
                s, thr = oof_predict(L, L[ycol].to_numpy(), folds, kind, MODELS[kind],
                                     train_mask=known if known_only else None)
                yS = L.y_gdc_gt.to_numpy()
                rows.append({"repeat": r, "model": kind, "variant": vname,
                             "auroc_source_known": fold_avg(M.auroc, yS[known], s[known], fold[known]),
                             "brier_source_known": fold_avg(M.brier, yS[known], s[known], fold[known]),
                             "auroc_released_all": fold_avg(M.auroc, L.y_released_death.to_numpy(), s, fold),
                             "brier_released_all": fold_avg(M.brier, L.y_released_death.to_numpy(), s, fold),
                             "auroc_source_cptac": M.auroc(yS[known & (L.cohort == "CPTAC").to_numpy()],
                                                           s[known & (L.cohort == "CPTAC").to_numpy()])})
    per = pd.DataFrame(rows)
    summ = []
    for (kind, v), g in per.groupby(["model", "variant"]):
        d = {"model": kind, "variant": v, "repeats": len(g)}
        for c in ["auroc_source_known", "brier_source_known", "auroc_released_all", "brier_released_all",
                  "auroc_source_cptac"]:
            d[c + "_mean"] = g[c].mean()
            d[c + "_p2.5"], d[c + "_p97.5"] = np.percentile(g[c], [2.5, 97.5])
        summ.append(d)
    # paired differences across repeats
    piv = per.pivot_table(index=["repeat", "model"], columns="variant",
                          values=["auroc_source_known", "brier_source_known", "auroc_released_all"])
    diffs = []
    names = list(VARIANTS)
    for kind in models:
        sub = piv.xs(kind, level="model")
        for metric in ["auroc_source_known", "brier_source_known", "auroc_released_all"]:
            for a, b in [(names[2], names[0]), (names[2], names[1]), (names[1], names[0])]:
                dd = sub[(metric, a)] - sub[(metric, b)]
                diffs.append({"model": kind, "metric": metric, "contrast": f"{a[:2]} - {b[:2]}",
                              "mean_delta": dd.mean(), "p2.5": np.percentile(dd, 2.5),
                              "p97.5": np.percentile(dd, 97.5), "share_positive": float((dd > 0).mean()),
                              "repeats": len(dd)})
    return per, pd.DataFrame(summ), pd.DataFrame(diffs)


# ---------------------------------------------------------------------------
# Experiment C: censoring-aware evaluation
# ---------------------------------------------------------------------------

def experiment_c(L, folds, scores_released):
    known = L.y_gdc_gt.notna().to_numpy()
    s_src, _ = oof_predict(L, L.y_gdc_gt.to_numpy(), folds, "logistic", CLIN, train_mask=known)
    t, e = L.os_time_gdc.to_numpy(), L.os_event_gdc.to_numpy()
    strata = (L.cohort == "CPTAC").to_numpy().astype(int)
    y_ipcw, w = M.ipcw_weights(t, e, 365, strata=strata)
    y_ipcw_pooled, w_pooled = M.ipcw_weights(t, e, 365)
    yS = L.y_gdc_gt.to_numpy()
    # consistency check: IPCW case/control labels agree with derived status where both defined
    both = ~np.isnan(y_ipcw) & ~np.isnan(yS)
    agree = float((y_ipcw[both] == yS[both]).mean())
    rows = []
    for name, s in [("logistic trained on released labels", scores_released["logistic"][0]),
                    ("logistic trained on source-derived labels", s_src),
                    ("stage-only trained on released labels", scores_released["stage_only"][0])]:
        lo, hi = M.auroc_bounds(yS, s)
        rows += [
            {"scores": name, "method": "complete case (ascertainable only)", "n_used": int(known.sum()),
             "auroc": M.auroc(yS, s), "brier": M.brier(yS, s)},
            {"scores": name, "method": "IPCW, censoring KM stratified by cohort",
             "n_used": int((w > 0).sum()), "auroc": M.weighted_auroc(y_ipcw, s, w),
             "brier": M.weighted_brier(y_ipcw, s, w)},
            {"scores": name, "method": "IPCW, pooled censoring KM", "n_used": int((w_pooled > 0).sum()),
             "auroc": M.weighted_auroc(y_ipcw_pooled, s, w_pooled),
             "brier": M.weighted_brier(y_ipcw_pooled, s, w_pooled)},
            {"scores": name, "method": "unknown imputed as survivor (released practice)", "n_used": len(L),
             "auroc": M.auroc(np.where(np.isnan(yS), 0, yS), s), "brier": M.brier(np.where(np.isnan(yS), 0, yS), s)},
            {"scores": name, "method": "unknown imputed as early death", "n_used": len(L),
             "auroc": M.auroc(np.where(np.isnan(yS), 1, yS), s), "brier": M.brier(np.where(np.isnan(yS), 1, yS), s)},
            {"scores": name, "method": "sharp bounds over all completions of unknowns", "n_used": len(L),
             "auroc": np.nan, "brier": np.nan, "auroc_lo": lo, "auroc_hi": hi},
            {"scores": name, "method": "Harrell C-index, OS (GDC), all usable follow-up",
             "n_used": int((~np.isnan(t)).sum()), "auroc": np.nan, "brier": np.nan,
             "c_index": M.harrell_c(t, e, s)},
        ]
    wpos = w[w > 0]
    diag = {"ipcw_label_agreement_with_derived_status": agree, "ipcw_max_weight": float(wpos.max()),
            "ipcw_mean_weight": float(wpos.mean()),
            "ipcw_effective_n": float(wpos.sum() ** 2 / np.sum(wpos ** 2)),
            "ipcw_G365_TCGA": float(1 / w[(strata == 0) & (y_ipcw == 0)][0]),
            "ipcw_G365_CPTAC": float(1 / w[(strata == 1) & (y_ipcw == 0)][0])}
    return pd.DataFrame(rows), diag


# ---------------------------------------------------------------------------
# Experiment D: quarantined leakage negative control (hypothetical configuration)
# ---------------------------------------------------------------------------

def experiment_d(L, folds):
    rows = []
    known = L.y_gdc_gt.notna().to_numpy()
    fold = L.fold.to_numpy()
    sets = {"CLIN (15 released clinical features)": CLIN,
            "CLIN + vital_status_dss [LEAKAGE]": CLIN + ["vital_status_dss"],
            "CLIN + days_to_last_fup [LEAKAGE]": CLIN + ["days_to_last_fup"]}
    for kind in ["logistic", "hgb"]:
        for name, feats in sets.items():
            s, thr = oof_predict(L, L.y_released_death.to_numpy(), folds, kind, feats,
                                 allow_leakage="LEAKAGE" in name)
            rows.append({"model": kind, "features": name,
                         "auroc_released_all_foldavg": fold_avg(M.auroc, L.y_released_death, s, fold),
                         "bacc_released_all_foldavg": fold_avg(_bacc, L.y_released_death, s, fold, thr=thr),
                         "auroc_source_known_foldavg": fold_avg(M.auroc, L.y_gdc_gt[known], s[known], fold[known])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensitivity analyses of the primary contrast and falsification checks
# ---------------------------------------------------------------------------

def sensitivity(L, folds, scores):
    rows = []
    s, thr = scores["logistic"]
    fold = L.fold.to_numpy()
    flagged = (L.flag_death_removed_upstream | L.flag_death_added_upstream | L.flag_negative_time |
               L.flag_non_diagnosis_index | L.flag_death_time_changed).astype(bool).to_numpy()
    specs = [
        ("primary: GDC, death on day 365 = early death (>)", "y_gdc_gt", None),
        ("boundary: death on day 365 = survivor (>=)", "y_gdc_ge", None),
        ("source: TCGA-CDR + LinkedOmics (strict event flag)", "y_alt_gt", None),
        ("source: consensus of GDC and alternative release", "y_consensus_gt", None),
        ("exclude version-changed / non-diagnosis-index / negative-time cases", "y_gdc_gt", ~flagged),
        ("TCGA only", "y_gdc_gt", (L.cohort == "TCGA").to_numpy()),
        ("CPTAC only", "y_gdc_gt", (L.cohort == "CPTAC").to_numpy()),
    ]
    for name, src, extra in specs:
        k = L[src].notna().to_numpy() & (np.ones(len(L), bool) if extra is None else extra)
        strata = (fold[k] * 2 + (L.cohort[k] == "CPTAC").to_numpy()).astype(int)
        d = paired_delta(L[src][k], L.y_released_death[k], s[k], fold[k], strata, M.auroc)
        rows.append({"sensitivity": name, "model": "logistic", "n": int(k.sum()),
                     "events_source": int(L[src][k].sum()), "events_released": int(L.y_released_death[k].sum()),
                     "auroc_source": d["a"], "auroc_released": d["b"], "delta": d["delta"],
                     "ci_lo": d["delta_ci_lo"], "ci_hi": d["delta_ci_hi"]})
    def alt_row(name, alt_folds):
        alt_fold = np.empty(len(L), int)
        for i, (_, te) in enumerate(alt_folds):
            alt_fold[te] = i
        s2, _ = oof_predict(L, L.y_released_death.to_numpy(), alt_folds, "logistic", CLIN)
        k = L.y_gdc_gt.notna().to_numpy()
        strata = (alt_fold[k] * 2 + (L.cohort[k] == "CPTAC").to_numpy()).astype(int)
        d = paired_delta(L.y_gdc_gt[k], L.y_released_death[k], s2[k], alt_fold[k], strata, M.auroc)
        return {"sensitivity": name, "model": "logistic", "n": int(k.sum()),
                "events_source": int(L.y_gdc_gt[k].sum()), "events_released": int(L.y_released_death[k].sum()),
                "auroc_source": d["a"], "auroc_released": d["b"], "delta": d["delta"],
                "ci_lo": d["delta_ci_lo"], "ci_hi": d["delta_ci_hi"]}

    # Released fold indices applied in ascending-ID order (what the released PatientDataset.py does)
    asc = np.argsort(L[ID].to_numpy(dtype=str))
    rows.append(alt_row("folds: released indices in ascending-ID order (released PatientDataset.py)",
                        [(asc[tr], asc[te]) for tr, te in folds]))
    # A new stratified split, unrelated to the released folds
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows.append(alt_row("folds: new stratified 5-fold split (seed 20261105)",
                        list(skf.split(np.zeros(len(L)), L.y_released_death))))
    return pd.DataFrame(rows)


def falsification(L, folds, scores):
    rows = []
    fold = L.fold.to_numpy()
    k = L.y_gdc_gt.notna().to_numpy()
    s = scores["logistic"][0]
    rng = np.random.default_rng(SEED)
    yk = L.y_gdc_gt.to_numpy()[k]
    null = np.array([fold_avg(M.auroc, rng.permutation(yk), s[k], fold[k]) for _ in range(500)])
    rows.append({"check": "label-permutation null, logistic scores, fold-avg AUROC (500 perms)",
                 "result": f"mean {null.mean():.3f}, 95% range {np.percentile(null, 2.5):.3f}-{np.percentile(null, 97.5):.3f}",
                 "expected": "centred on 0.5", "pass": bool(abs(null.mean() - 0.5) < 0.02)})
    sp = scores["prevalence"][0]
    fa = fold_avg(M.auroc, L.y_released_death, sp, fold)
    rows.append({"check": "prevalence-only model, fold-averaged AUROC", "result": f"{fa:.3f}",
                 "expected": "0.5 exactly", "pass": bool(abs(fa - 0.5) < 1e-9)})
    rows.append({"check": "prevalence-only model, POOLED out-of-fold AUROC",
                 "result": f"{M.auroc(L.y_released_death, sp):.3f}",
                 "expected": "close to 0.5; departs because training-fold prevalence differs slightly by fold",
                 "pass": True})
    c = L[(L.cohort == "CPTAC") & L["demographic.days_to_death"].notna() & L.y_gdc_gt.notna()]
    stable = ~c.flag_death_removed_upstream.astype(bool)
    flipped = 1 - c.y_released_death
    rows.append({"check": "CPTAC recorded deaths: flipping released label recovers source status "
                          "(excluding deaths removed upstream after the MMIST export)",
                 "result": f"{int((flipped[stable] == c.y_gdc_gt[stable]).sum())}/{int(stable.sum())} "
                           f"(+{int((~stable).sum())} version-changed case excluded)",
                 "expected": "all (polarity inversion)",
                 "pass": bool((flipped[stable] == c.y_gdc_gt[stable]).all())})
    # cohort identifiability from the released clinical features (why label rules can be learned)
    y_c = (L.cohort == "CPTAC").astype(int).to_numpy()
    sc, _ = oof_predict(L, y_c, folds, "logistic", CLIN)
    rows.append({"check": "cohort (CPTAC vs TCGA) predictable from the 15 clinical features, OOF AUROC",
                 "result": f"{M.auroc(y_c, sc):.4f}", "expected": "descriptive", "pass": True})
    # no overlap between train and test across released folds
    regen = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(L)), L.vital_status_12))
    same = sum(np.array_equal(np.sort(te), np.sort(rte)) for (_, te), (_, rte) in zip(folds, regen))
    rows.append({"check": "folds.npy regenerated from released labels with StratifiedKFold(seed 42)",
                 "result": f"{same}/5 folds identical", "expected": "5/5", "pass": bool(same == 5)})
    ov = sum(len(np.intersect1d(tr, te)) for tr, te in folds)
    rows.append({"check": "released folds: train/test overlap; test sets partition all patients",
                 "result": f"overlap {ov}; union {len(np.unique(np.concatenate([te for _, te in folds])))}",
                 "expected": "0; 1365", "pass": bool(ov == 0)})
    return pd.DataFrame(rows)


def model_ranking(L, scores):
    """Does the label set change which model looks best? Paired bootstrap of AUROC differences
    between models, fold-averaged, under each label set (same patients, same scores)."""
    fold = L.fold.to_numpy()
    rows = []
    pairs = [("stage_only", "logistic"), ("stage_only", "hgb"), ("hgb", "logistic")]
    for label, ycol, mask in [("released labels, all patients (benchmark setting)", "y_released_death", None),
                              ("source-derived labels, ascertainable patients", "y_gdc_gt", "known")]:
        k = L.y_gdc_gt.notna().to_numpy() if mask else np.ones(len(L), bool)
        y = L[ycol].to_numpy(float)[k]
        f = fold[k]
        strata = (f * 2 + (L.cohort[k] == "CPTAC").to_numpy()).astype(int)
        for m1, m2 in pairs:
            s1, s2 = scores[m1][0][k], scores[m2][0][k]
            a, b = M.paired_bootstrap(lambda i: fold_avg(M.auroc, y, s1, f, i),
                                      lambda i: fold_avg(M.auroc, y, s2, f, i), len(y), strata, REPS, SEED)
            p1, p2 = fold_avg(M.auroc, y, s1, f), fold_avg(M.auroc, y, s2, f)
            lo, hi = M.ci(a - b)
            rows.append({"evaluation": label, "n": int(k.sum()), "events": int(np.nansum(y)),
                         "contrast": f"{m1} - {m2}", "auroc_1": p1, "auroc_2": p2, "delta": p1 - p2,
                         "ci_lo": lo, "ci_hi": hi})
    # Interaction on identical (ascertainable) patients: does the label set change the model gap?
    k = L.y_gdc_gt.notna().to_numpy()
    yS, yR, f = L.y_gdc_gt.to_numpy(float)[k], L.y_released_death.to_numpy(float)[k], fold[k]
    strata = (f * 2 + (L.cohort[k] == "CPTAC").to_numpy()).astype(int)
    for m1, m2 in pairs:
        s1, s2 = scores[m1][0][k], scores[m2][0][k]

        def gap(y, i=None):
            return fold_avg(M.auroc, y, s1, f, i) - fold_avg(M.auroc, y, s2, f, i)

        a, b = M.paired_bootstrap(lambda i: gap(yS, i), lambda i: gap(yR, i), len(yS), strata, REPS, SEED)
        lo, hi = M.ci(a - b)
        rows.append({"evaluation": "interaction: gap under source minus gap under released (same 1154 patients)",
                     "n": int(k.sum()), "events": int(yS.sum()), "contrast": f"{m1} - {m2}",
                     "auroc_1": gap(yS), "auroc_2": gap(yR), "delta": gap(yS) - gap(yR), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def training_effect_benchmark_folds(L, folds, scores):
    """POST-HOC. Patient-level uncertainty for the training-label effect in the released folds:
    models trained on source vs released labels, same held-out patients, paired bootstrap
    conditional on the fitted models (models are not refitted within the bootstrap)."""
    known = L.y_gdc_gt.notna().to_numpy()
    fold = L.fold.to_numpy()
    rows = []
    for kind in ["logistic", "hgb"]:
        s_rel = scores[kind][0]
        s_src, _ = oof_predict(L, L.y_gdc_gt.to_numpy(), folds, kind, MODELS[kind], train_mask=known)
        for label, y, mask in [("source-derived labels, ascertainable patients", L.y_gdc_gt.to_numpy(float), known),
                               ("released labels, all patients (benchmark lens)",
                                L.y_released_death.to_numpy(float), np.ones(len(L), bool))]:
            yy, f, a_s, b_s = y[mask], fold[mask], s_src[mask], s_rel[mask]
            strata = (f * 2 + (L.cohort[mask] == "CPTAC").to_numpy()).astype(int)
            a, b = M.paired_bootstrap(lambda i: fold_avg(M.auroc, yy, a_s, f, i),
                                      lambda i: fold_avg(M.auroc, yy, b_s, f, i), len(yy), strata, REPS, SEED)
            pa, pb = fold_avg(M.auroc, yy, a_s, f), fold_avg(M.auroc, yy, b_s, f)
            lo, hi = M.ci(a - b)
            rows.append({"model": kind, "evaluation": label, "n": int(mask.sum()), "events": int(np.nansum(yy)),
                         "auroc_trained_on_source": pa, "auroc_trained_on_released": pb, "delta": pa - pb,
                         "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def cohort_mechanism(L, folds, scores):
    """POST-HOC. Why does the pooled contrast differ from the within-cohort contrasts?
    Compares mean predicted risk by cohort with the event rate under each label set, and
    measures how well scores separate cohorts."""
    known = L.y_gdc_gt.notna().to_numpy()
    src_trained, _ = oof_predict(L, L.y_gdc_gt.to_numpy(), folds, "logistic", CLIN, train_mask=known)
    sets = {"logistic, trained on released": scores["logistic"][0], "gradient boosting, trained on released": scores["hgb"][0],
            "stage only, trained on released": scores["stage_only"][0], "logistic, trained on source": src_trained}
    cptac = (L.cohort == "CPTAC").to_numpy()
    rows = []
    for name, sc in sets.items():
        for cohort, m in [("TCGA", ~cptac & known), ("CPTAC", cptac & known)]:
            rows.append({"scores": name, "cohort": cohort, "n_ascertainable": int(m.sum()),
                         "mean_predicted_risk": float(sc[m].mean()),
                         "released_event_rate": float(L.y_released_death.to_numpy()[m].mean()),
                         "source_event_rate": float(L.y_gdc_gt.to_numpy()[m].mean()),
                         "auroc_for_cohort_cptac_vs_tcga": M.auroc(cptac[known].astype(int), sc[known])})
    return pd.DataFrame(rows)


def auroc_decomposition(L, scores, model="logistic"):
    """POST-HOC. Pooled AUROC is a pair-weighted average of case-control concordance. Splitting
    the pairs by the cohorts of the case and the control shows where a label set changes AUROC.
    Pooled over folds (fold-averaged AUROC cannot be decomposed this way)."""
    k = L.y_gdc_gt.notna().to_numpy()
    s_ = scores[model][0][k]
    coh = L.cohort.to_numpy()[k]
    rows = []
    for label, col in [("released", "y_released_death"), ("source", "y_gdc_gt")]:
        y = L[col].to_numpy(float)[k]
        total = (y == 1).sum() * (y == 0).sum()
        check = 0.0
        for a in ["TCGA", "CPTAC"]:
            for b in ["TCGA", "CPTAC"]:
                sc, sn = s_[(y == 1) & (coh == a)], s_[(y == 0) & (coh == b)]
                c = M.auroc(np.r_[np.ones(len(sc)), np.zeros(len(sn))], np.r_[sc, sn])
                share = len(sc) * len(sn) / total
                check += share * c
                rows.append({"model": model, "label_set": label, "case_cohort": a, "control_cohort": b,
                             "cases": len(sc), "controls": len(sn), "pair_share": share, "concordance": c,
                             "contribution": share * c})
        assert abs(check - M.auroc(y, s_)) < 1e-9, "decomposition must sum to pooled AUROC"
    return pd.DataFrame(rows)


def run(n_repeats_b: int = 50) -> dict:
    t0 = time.time()
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    L, folds = load()
    A, Ad, scores = experiment_a(L, folds)
    A.to_csv(TABLES / "exp_a_metrics.csv", index=False)
    Ad.to_csv(TABLES / "exp_a_paired_deltas.csv", index=False)
    model_ranking(L, scores).to_csv(TABLES / "exp_a_model_ranking.csv", index=False)
    training_effect_benchmark_folds(L, folds, scores).to_csv(TABLES / "exp_b_benchmark_folds.csv", index=False)
    cohort_mechanism(L, folds, scores).to_csv(TABLES / "exp_a_cohort_mechanism.csv", index=False)
    pd.concat([auroc_decomposition(L, scores, m) for m in ["logistic", "stage_only"]]).to_csv(
        TABLES / "exp_a_auroc_decomposition.csv", index=False)
    print(f"A done {time.time() - t0:.0f}s")
    A2 = experiment_a2(L)
    A2.to_csv(TABLES / "exp_a2_transport.csv", index=False)
    print(f"A2 done {time.time() - t0:.0f}s")
    Bper, Bsum, Bd = experiment_b(L, n_repeats_b)
    Bper.to_csv(TABLES / "exp_b_per_repeat.csv", index=False)
    Bsum.to_csv(TABLES / "exp_b_summary.csv", index=False)
    Bd.to_csv(TABLES / "exp_b_paired.csv", index=False)
    print(f"B done {time.time() - t0:.0f}s")
    C, Cdiag = experiment_c(L, folds, scores)
    C.to_csv(TABLES / "exp_c_censoring.csv", index=False)
    print(f"C done {time.time() - t0:.0f}s")
    D = experiment_d(L, folds)
    D.to_csv(TABLES / "exp_d_leakage.csv", index=False)
    S = sensitivity(L, folds, scores)
    S.to_csv(TABLES / "exp_sensitivity.csv", index=False)
    F = falsification(L, folds, scores)
    F.to_csv(TABLES / "falsification_checks.csv", index=False)
    # score distributions by label category (aggregate histogram, for figure/dashboard)
    s = scores["logistic"][0]
    cat = np.select(
        [(L.y_gdc_gt == 1) & (L.y_released_death == 0), (L.y_gdc_gt == 0) & (L.y_released_death == 1),
         (L.y_gdc_gt == 1) & (L.y_released_death == 1), (L.y_gdc_gt == 0) & (L.y_released_death == 0),
         L.y_gdc_gt.isna()],
        ["source early death, released survivor", "source survivor, released early death",
         "concordant early death", "concordant survivor", "not ascertainable"], "other")
    bins = np.linspace(0, 0.6, 25)
    hist = []
    for c in np.unique(cat):
        h, _ = np.histogram(np.clip(s[cat == c], 0, 0.599), bins=bins)
        hist += [{"category": c, "bin_lo": bins[i], "bin_hi": bins[i + 1], "n": int(h[i])} for i in range(len(h))]
    pd.DataFrame(hist).to_csv(TABLES / "score_distribution_by_label_category.csv", index=False)
    summ = pd.DataFrame([{"category": c, "n": int((cat == c).sum()), "median_score": float(np.median(s[cat == c])),
                          "q25": float(np.percentile(s[cat == c], 25)), "q75": float(np.percentile(s[cat == c], 75))}
                         for c in np.unique(cat)])
    summ.to_csv(TABLES / "score_summary_by_label_category.csv", index=False)

    prim = Ad[(Ad.model == "logistic") & (Ad.metric == "AUROC")].iloc[0].to_dict()
    results = {"protocol_version": CFG["protocol_version"], "primary_contrast": prim,
               "ipcw_diagnostics": Cdiag, "runtime_s": round(time.time() - t0, 1),
               "bootstrap_reps": REPS, "seed": SEED, "repeats_experiment_b": n_repeats_b}
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(json.dumps(run(n), indent=2, default=float))

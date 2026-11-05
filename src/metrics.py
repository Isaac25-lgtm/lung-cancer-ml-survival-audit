"""Evaluation metrics. Orientation everywhere: y = 1 means early death (<= horizon),
score = predicted probability of early death. Higher score = higher risk.
"""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.stats_utils import km_censoring


def _clean(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    ok = ~np.isnan(y) & ~np.isnan(s)
    return y[ok].astype(int), s[ok]


def auroc(y, s) -> float:
    """Mann-Whitney AUROC with mid-ranks for ties (identical to sklearn, much faster)."""
    y, s = _clean(y, s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, s) -> float:
    y, s = _clean(y, s)
    return float(np.mean((s - y) ** 2)) if len(y) else np.nan


def scaled_brier(y, s) -> float:
    """1 - Brier/Brier(prevalence model on the same labels); higher is better."""
    y, s = _clean(y, s)
    ref = brier_score_loss(y, np.full(len(y), y.mean()))
    return float(1 - brier_score_loss(y, s) / ref) if ref > 0 else np.nan


def auprc(y, s) -> float:
    y, s = _clean(y, s)
    return float(average_precision_score(y, s)) if y.sum() > 0 else np.nan


def balanced_accuracy(y, s, threshold: float) -> float:
    y, s = _clean(y, s)
    pred = (s >= threshold).astype(int)
    sens = np.mean(pred[y == 1] == 1) if (y == 1).any() else np.nan
    spec = np.mean(pred[y == 0] == 0) if (y == 0).any() else np.nan
    return float((sens + spec) / 2)


def calibration(y, s) -> tuple[float, float]:
    """Calibration-in-the-large (intercept with logit(p) offset) and slope."""
    y, s = _clean(y, s)
    if len(np.unique(y)) < 2:
        return (np.nan, np.nan)
    lp = logit(np.clip(s, 1e-6, 1 - 1e-6))
    try:
        cil = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    except Exception:  # separation / non-convergence
        return (np.nan, np.nan)
    return (float(cil), float(slope))


def all_metrics(y, s, threshold: float) -> dict:
    yc, _ = _clean(y, s)
    cil, slope = calibration(y, s)
    return {"n": int(len(yc)), "events": int(yc.sum()), "auroc": auroc(y, s), "brier": brier(y, s),
            "scaled_brier": scaled_brier(y, s), "auprc": auprc(y, s),
            "bacc": balanced_accuracy(y, s, threshold), "cal_intercept": cil, "cal_slope": slope}


# ---------------------------------------------------------------------------
# Paired bootstrap (conditional on fixed scores)
# ---------------------------------------------------------------------------

def paired_bootstrap(fn_a, fn_b, n: int, strata: np.ndarray | None, reps: int, seed: int):
    """Bootstrap patients (stratified), evaluate fn_a(idx) and fn_b(idx); return arrays of a, b."""
    rng = np.random.default_rng(seed)
    if strata is None:
        strata = np.zeros(n, dtype=int)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    a_out, b_out = np.empty(reps), np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        a_out[r] = fn_a(idx)
        b_out[r] = fn_b(idx)
    return a_out, b_out


def ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(x, 100 * alpha / 2)), float(np.percentile(x, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------------------
# Censoring-aware fixed-horizon metrics
# ---------------------------------------------------------------------------

def ipcw_weights(time, event, horizon: float, strata=None) -> tuple[np.ndarray, np.ndarray]:
    """IPCW weights for a fixed-horizon binary outcome.

    cases    (event, T <= h): w = 1 / G(T-)
    controls (T > h):          w = 1 / G(h)
    censored before h:        w = 0
    G is reverse Kaplan-Meier, optionally estimated separately within strata.
    Returns (y, w) where y is 1 for cases, 0 for controls, NaN otherwise.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    n = len(time)
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    y = np.full(n, np.nan)
    w = np.zeros(n)
    for g in np.unique(strata):
        m = (strata == g) & ~np.isnan(time)
        G = km_censoring(time[m], event[m])
        ti, ei = time[m], event[m]
        yi = np.full(m.sum(), np.nan)
        wi = np.zeros(m.sum())
        case = (ei == 1) & (ti <= horizon)
        ctrl = ti > horizon
        yi[case] = 1
        yi[ctrl] = 0
        if case.any():
            wi[case] = 1 / G(ti[case], left=True)
        if ctrl.any():
            wi[ctrl] = 1 / G(np.array([horizon]))[0]
        y[m], w[m] = yi, wi
    return y, w


def weighted_auroc(y, s, w) -> float:
    y, s, w = (np.asarray(a, float) for a in (y, s, w))
    ok = ~np.isnan(y) & ~np.isnan(s) & (w > 0)
    y, s, w = y[ok], s[ok], w[ok]
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return np.nan
    sp, wp = s[pos], w[pos]
    sn, wn = s[neg], w[neg]
    order = np.argsort(sn)
    sn, wn = sn[order], wn[order]
    cw = np.concatenate([[0], np.cumsum(wn)])
    lo = np.searchsorted(sn, sp, side="left")
    hi = np.searchsorted(sn, sp, side="right")
    conc = cw[lo] + 0.5 * (cw[hi] - cw[lo])
    return float(np.sum(wp * conc) / (wp.sum() * wn.sum()))


def weighted_brier(y, s, w) -> float:
    y, s, w = (np.asarray(a, float) for a in (y, s, w))
    ok = ~np.isnan(y) & ~np.isnan(s) & (w > 0)
    # IPCW Brier (Graf): average over ALL patients with weights; censored-before-h contribute 0
    n_all = np.sum(~np.isnan(s))
    return float(np.sum(w[ok] * (y[ok] - s[ok]) ** 2) / n_all)


def auroc_bounds(y, s) -> tuple[float, float]:
    """Sharp bounds on AUROC over all 0/1 completions of unknown (NaN) labels.

    For a fixed number k of unknowns completed as events, AUROC is minimised by
    giving the k lowest-scored unknowns the event label and maximised by giving
    it to the k highest-scored (pairwise label-swap argument). We scan k.
    """
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    known = ~np.isnan(y)
    su = np.sort(s[~known])
    nu = len(su)
    yk, sk = y[known], s[known]
    lo, hi = np.inf, -np.inf
    for k in range(nu + 1):
        # (events, non-events): lowest-k as events (minimiser), highest-k as events (maximiser)
        for events, non_events in ((su[:k], su[k:]), (su[nu - k:], su[:nu - k])):
            yy = np.concatenate([yk, np.ones(k), np.zeros(nu - k)])
            ss = np.concatenate([sk, events, non_events])
            if 0 < yy.sum() < len(yy):
                a = roc_auc_score(yy, ss)
                lo, hi = min(lo, a), max(hi, a)
    return float(lo), float(hi)


def harrell_c(time, event, risk) -> float:
    """Harrell's concordance for right-censored data (higher risk = earlier death)."""
    t, e, r = (np.asarray(a, float) for a in (time, event, risk))
    ok = ~np.isnan(t) & ~np.isnan(e) & ~np.isnan(r)
    t, e, r = t[ok], e[ok], r[ok]
    conc = ties = comp = 0.0
    for i in np.flatnonzero(e == 1):
        m = t > t[i]
        comp += m.sum()
        conc += np.sum(r[i] > r[m])
        ties += np.sum(r[i] == r[m])
    return float((conc + 0.5 * ties) / comp) if comp else np.nan

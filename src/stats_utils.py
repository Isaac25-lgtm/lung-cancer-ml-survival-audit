"""Small statistical helpers (no patient data)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def km_at(time: np.ndarray, event: np.ndarray, t: float) -> tuple[float, float, float]:
    """Kaplan-Meier S(t) with log(-log) 95% CI (Greenwood). Returns (S, lo, hi)."""
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    ok = ~np.isnan(time) & ~np.isnan(event)
    time, event = time[ok], event[ok]
    s, var_sum = 1.0, 0.0
    for u in np.unique(time[(event == 1) & (time <= t)]):
        n_risk = np.sum(time >= u)
        d = np.sum((time == u) & (event == 1))
        s *= 1 - d / n_risk
        if n_risk > d:
            var_sum += d / (n_risk * (n_risk - d))
    if s <= 0 or s >= 1 or var_sum == 0:
        return (s, np.nan, np.nan)
    se = np.sqrt(var_sum) / abs(np.log(s))
    z = 1.959963984540054
    lo = s ** np.exp(z * se)
    hi = s ** np.exp(-z * se)
    return (s, lo, hi)


def km_curve(time: np.ndarray, event: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """KM survival evaluated on a grid, plus number at risk on the grid."""
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    ok = ~np.isnan(time) & ~np.isnan(event)
    time, event = time[ok], event[ok]
    ev_times = np.unique(time[event == 1])
    surv_steps = []
    s = 1.0
    for u in ev_times:
        n_risk = np.sum(time >= u)
        d = np.sum((time == u) & (event == 1))
        s *= 1 - d / n_risk
        surv_steps.append(s)
    surv_steps = np.array(surv_steps)
    idx = np.searchsorted(ev_times, grid, side="right") - 1
    S = np.where(idx >= 0, surv_steps[np.clip(idx, 0, None)] if len(surv_steps) else 1.0, 1.0)
    at_risk = np.array([np.sum(time >= g) for g in grid])
    return S, at_risk


def km_censoring(time: np.ndarray, event: np.ndarray):
    """Return a function G(t) = P(C > t) estimated by reverse Kaplan-Meier (left-continuous lookup helper)."""
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    cens = 1 - event
    c_times = np.unique(time[cens == 1])
    steps = []
    g = 1.0
    for u in c_times:
        n_risk = np.sum(time >= u)
        c = np.sum((time == u) & (cens == 1))
        g *= 1 - c / n_risk
        steps.append(g)
    steps = np.array(steps)

    def G(t, left: bool = False):
        t = np.atleast_1d(np.asarray(t, float))
        side = "left" if left else "right"
        idx = np.searchsorted(c_times, t, side=side) - 1
        out = np.where(idx >= 0, steps[np.clip(idx, 0, None)] if len(steps) else 1.0, 1.0)
        return out

    return G

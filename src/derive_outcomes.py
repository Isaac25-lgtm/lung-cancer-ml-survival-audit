"""Fixed-horizon outcome derivation with explicit three-state status.

Coding used throughout the analysis:
    y_death = 1   death observed on/before the horizon (early death)
    y_death = 0   known alive beyond the horizon
    y_death = NaN not ascertainable (censored before horizon, missing/invalid times,
                  dead without a death date)

Two boundary conventions are supported:
    "gt": survival class requires survival time > horizon  (death on day 365 = early death)
    "ge": survival class requires survival time >= horizon (death on day 365 = survivor)

A patient censored (last known alive) at day c is known to survive beyond the
horizon when c >= horizon under both conventions, because T > c >= horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 365


def status_at_horizon(death_time, dead, last_contact, horizon: int = HORIZON,
                      convention: str = "gt") -> np.ndarray:
    """Vectorised three-state status.

    Parameters
    ----------
    death_time : array-like of float, days from index to death (NaN if none recorded)
    dead : array-like of {True, False, NaN}; True = recorded dead, False = not recorded dead
           (alive or vital status not reported). NaN treated as not dead.
    last_contact : array-like of float, days from index to last contact while alive
                   (NaN if none).
    """
    if convention not in {"gt", "ge"}:
        raise ValueError("convention must be 'gt' or 'ge'")
    dt = pd.to_numeric(pd.Series(death_time), errors="coerce").to_numpy(dtype=float)
    lc = pd.to_numeric(pd.Series(last_contact), errors="coerce").to_numpy(dtype=float)
    dead = pd.Series(dead).astype("boolean").fillna(False).to_numpy(dtype=bool)
    if not (len(dt) == len(lc) == len(dead)):
        raise ValueError("inputs must have equal length")

    out = np.full(len(dt), np.nan)
    valid_death = dead & ~np.isnan(dt) & (dt >= 0)
    if convention == "gt":
        early = valid_death & (dt <= horizon)
        late = valid_death & (dt > horizon)
    else:
        early = valid_death & (dt < horizon)
        late = valid_death & (dt >= horizon)
    # Alive at last contact (or vital status not reported) with contact at/after horizon.
    # Dead patients WITHOUT a valid death time are left unknown: their last contact may
    # pre-date death and we cannot confirm ordering conservatively.
    alive_known = (~dead) & ~np.isnan(lc) & (lc >= horizon)
    out[early] = 1.0
    out[late | alive_known] = 0.0
    return out


def state_label(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.where(np.isnan(y), "unknown", np.where(y == 1, "early death", "alive >365d"))


# ---------------------------------------------------------------------------
# Released-label generating rules (reverse-engineered, see docs/findings.md)
# ---------------------------------------------------------------------------

def released_rule_tcga(days_to_death) -> np.ndarray:
    """vital_status_12 = 1 if death after day 365 or no death recorded (reproduces TCGA 1026/1026)."""
    d = pd.to_numeric(pd.Series(days_to_death), errors="coerce")
    return np.where((d > 365) | d.isna(), 1, 0)


def released_rule_notebook(days_to_death) -> np.ndarray:
    """encode-csv.ipynb cell 14: 1 if death_days <= 365 or missing (reproduces CPTAC 339/339)."""
    d = pd.to_numeric(pd.Series(days_to_death), errors="coerce")
    return np.where((d <= 365) | d.isna(), 1, 0)


# ---------------------------------------------------------------------------
# Source-specific wrappers
# ---------------------------------------------------------------------------

def from_gdc(df: pd.DataFrame, convention: str = "gt", horizon: int = HORIZON) -> np.ndarray:
    return status_at_horizon(df["gdc_days_to_death"], df["gdc_vital_status"].eq("Dead"),
                             df["gdc_last_fu"], horizon, convention)


def from_cdr(df: pd.DataFrame, convention: str = "gt", horizon: int = HORIZON) -> np.ndarray:
    """TCGA-CDR: OS=1 death at OS.time; OS=0 censored at OS.time."""
    os_ = pd.to_numeric(df["cdr_OS"], errors="coerce")
    t = pd.to_numeric(df["cdr_OS.time"], errors="coerce")
    dead = os_.eq(1)
    death_t = t.where(dead)
    last = t.where(os_.eq(0))
    out = status_at_horizon(death_t, dead, last, horizon, convention)
    out[os_.isna().to_numpy()] = np.nan
    return out


def from_linkedomics(df: pd.DataFrame, convention: str = "gt", horizon: int = HORIZON,
                     strict: bool = True) -> np.ndarray:
    """LinkedOmics CPTAC: OS_event=1 death at OS_days; 0 censored at OS_days.

    strict=True requires a non-missing event indicator (the handover's exploratory rule
    accepted OS_days >= 365 with a missing event flag; strict=False reproduces that).
    """
    ev = pd.to_numeric(df["lo_OS_event"], errors="coerce")
    t = pd.to_numeric(df["lo_OS_days"], errors="coerce")
    dead = ev.eq(1)
    death_t = t.where(dead)
    last = t.where(ev.eq(0)) if strict else t.where(~dead)
    return status_at_horizon(death_t, dead, last, horizon, convention)


def from_mmist_fields(df: pd.DataFrame, convention: str = "gt", horizon: int = HORIZON) -> np.ndarray:
    """Only fields released in the benchmark (days_to_death, days_to_last_fup)."""
    d = df["demographic.days_to_death"]
    return status_at_horizon(d, d.notna(), df["days_to_last_fup"].where(d.isna()), horizon, convention)


def time_event_gdc(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Right-censored OS (days, event) from GDC; NaN time when not usable."""
    dead = df["gdc_vital_status"].eq("Dead").to_numpy()
    dt = df["gdc_days_to_death"].to_numpy(dtype=float)
    lc = df["gdc_last_fu"].to_numpy(dtype=float)
    t = np.where(dead, dt, lc)
    e = np.where(dead, 1, 0).astype(float)
    bad = np.isnan(t) | (t < 0)
    t = np.where(bad, np.nan, t)
    e = np.where(bad, np.nan, e)
    return t, e

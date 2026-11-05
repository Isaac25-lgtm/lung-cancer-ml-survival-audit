import numpy as np
import pandas as pd
import pytest

from src.derive_outcomes import (released_rule_notebook, released_rule_tcga, state_label,
                                 status_at_horizon)

NaN = np.nan

# (death_time, dead, last_contact, expected_gt, expected_ge, description)
CASES = [
    (364, True, NaN, 1, 1, "dead day 364"),
    (365, True, NaN, 1, 0, "dead exactly day 365: boundary-dependent"),
    (366, True, NaN, 0, 0, "dead day 366"),
    (0, True, NaN, 1, 1, "dead day 0"),
    (NaN, False, 364, NaN, NaN, "alive, last contact day 364 -> unknown"),
    (NaN, False, 365, 0, 0, "alive, last contact day 365 -> survived (T > 365)"),
    (NaN, False, 2000, 0, 0, "alive, long follow-up"),
    (NaN, False, NaN, NaN, NaN, "alive, no follow-up -> unknown (NOT survivor)"),
    (NaN, True, 500, NaN, NaN, "dead without date -> unknown (conservative)"),
    (-1, True, NaN, NaN, NaN, "negative death time -> unknown"),
    (800, True, 900, 0, 0, "late death; contact field ignored when dead"),
    (100, True, 900, 1, 1, "early death wins over contradictory contact"),
    (NaN, NaN, 400, 0, 0, "vital status missing treated as not dead, contact >= 365"),
]


@pytest.mark.parametrize("dt,dead,lc,egt,ege,desc", CASES, ids=[c[-1] for c in CASES])
def test_status_cases(dt, dead, lc, egt, ege, desc):
    for conv, exp in (("gt", egt), ("ge", ege)):
        got = status_at_horizon([dt], [dead], [lc], convention=conv)[0]
        if np.isnan(exp):
            assert np.isnan(got), f"{desc} [{conv}]"
        else:
            assert got == exp, f"{desc} [{conv}]"


def test_vectorised_matches_scalar():
    dt, dead, lc = zip(*[(c[0], c[1], c[2]) for c in CASES])
    vec = status_at_horizon(list(dt), list(dead), list(lc))
    for i, c in enumerate(CASES):
        s = status_at_horizon([c[0]], [c[1]], [c[2]])[0]
        assert (np.isnan(s) and np.isnan(vec[i])) or s == vec[i]


def test_row_order_invariance():
    rng = np.random.default_rng(0)
    n = 300
    dt = np.where(rng.random(n) < 0.4, rng.integers(-5, 1500, n), np.nan)
    dead = ~np.isnan(dt)
    lc = np.where(dead, np.nan, rng.integers(0, 1500, n))
    base = status_at_horizon(dt, dead, lc)
    perm = rng.permutation(n)
    out = status_at_horizon(dt[perm], dead[perm], lc[perm])
    np.testing.assert_array_equal(np.isnan(out), np.isnan(base[perm]))
    np.testing.assert_array_equal(np.nan_to_num(out, nan=-9), np.nan_to_num(base[perm], nan=-9))


def test_bad_convention():
    with pytest.raises(ValueError):
        status_at_horizon([1], [True], [NaN], convention="bogus")


def test_released_rules_opposite_polarity_for_deaths():
    d = pd.Series([100, 365, 366, NaN])
    assert released_rule_tcga(d).tolist() == [0, 0, 1, 1]
    assert released_rule_notebook(d).tolist() == [1, 1, 0, 1]
    # both rules label "no recorded death" as survivor regardless of follow-up
    assert released_rule_tcga([NaN])[0] == released_rule_notebook([NaN])[0] == 1


def test_state_label():
    assert state_label(np.array([1.0, 0.0, NaN])).tolist() == ["early death", "alive >365d", "unknown"]

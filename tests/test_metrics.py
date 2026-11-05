import itertools

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src import metrics as M
from src.stats_utils import km_at, wilson


def test_auroc_orientation():
    y = [0, 0, 1, 1]
    assert M.auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auroc_complement_on_fully_reversed_labels():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    assert M.auroc(y, s) == pytest.approx(1 - M.auroc(1 - y, s))


def test_nan_labels_dropped():
    assert M.auroc([0, 1, np.nan], [0.1, 0.9, 0.5]) == 1.0


def test_prevalence_model_auroc_half_and_brier():
    y = np.array([0] * 80 + [1] * 20)
    s = np.full(100, 0.2)
    assert M.auroc(y, s) == 0.5
    assert M.brier(y, s) == pytest.approx(0.16)
    assert M.scaled_brier(y, s) == pytest.approx(0.0)


def test_balanced_accuracy():
    y = [0, 0, 1, 1]
    assert M.balanced_accuracy(y, [0.1, 0.6, 0.7, 0.2], 0.5) == pytest.approx(0.5)


def test_weighted_auroc_equals_unweighted_with_unit_weights():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300).astype(float)
    s = np.round(rng.random(300), 2)  # include ties
    assert M.weighted_auroc(y, s, np.ones(300)) == pytest.approx(roc_auc_score(y, s))


def test_ipcw_no_censoring_reduces_to_complete_case():
    t = np.array([100, 200, 400, 500, 600.0])
    e = np.ones(5)
    y, w = M.ipcw_weights(t, e, 365)
    assert np.allclose(y, [1, 1, 0, 0, 0])
    assert np.allclose(w, 1)


def test_ipcw_censored_before_horizon_zero_weight():
    t = np.array([100, 200, 400, 500, 300.0])
    e = np.array([1, 1, 1, 0, 0.0])
    y, w = M.ipcw_weights(t, e, 365)
    assert np.isnan(y[4]) and w[4] == 0
    assert w[2] > 1  # controls up-weighted for censoring before the horizon


@pytest.mark.parametrize("seed", range(8))
def test_auroc_bounds_bruteforce(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 13).astype(float)
    s = rng.random(13)
    unk = sorted(rng.choice(13, 5, replace=False).tolist())
    y[unk] = np.nan
    lo, hi = M.auroc_bounds(y, s)
    vals = []
    for comb in itertools.product([0, 1], repeat=len(unk)):
        yy = y.copy()
        yy[unk] = comb
        if 0 < yy.sum() < len(yy):
            vals.append(roc_auc_score(yy, s))
    assert lo == pytest.approx(min(vals))
    assert hi == pytest.approx(max(vals))


def test_harrell_c_perfect():
    t = np.array([1, 2, 3, 4.0])
    e = np.ones(4)
    assert M.harrell_c(t, e, [4, 3, 2, 1]) == 1.0
    assert M.harrell_c(t, e, [1, 2, 3, 4]) == 0.0


def test_wilson_known_value():
    lo, hi = wilson(56, 177)
    assert lo == pytest.approx(0.2523, abs=1e-3)
    assert hi == pytest.approx(0.3882, abs=1e-3)


def test_km_no_censoring_equals_empirical():
    t = np.array([100, 200, 400, 500.0])
    S, _, _ = km_at(t, np.ones(4), 365)
    assert S == pytest.approx(0.5)


def test_calibration_perfect_model():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.6, 20000)
    y = rng.random(20000) < p
    cil, slope = M.calibration(y.astype(int), p)
    assert abs(cil) < 0.05 and abs(slope - 1) < 0.07


@pytest.mark.parametrize("seed", range(5))
def test_fast_auroc_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500), 2)  # many ties
    assert M.auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)

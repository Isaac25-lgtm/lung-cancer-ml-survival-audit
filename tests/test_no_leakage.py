import numpy as np
import pandas as pd
import pytest

from src.models import CLIN, FORBIDDEN, LeakageError, check_features, make_model, oof_predict


def test_clin_has_no_outcome_fields():
    assert not set(CLIN) & FORBIDDEN
    check_features(CLIN)  # must not raise
    assert len(CLIN) == 15


@pytest.mark.parametrize("col", sorted(FORBIDDEN) + ["y_gdc_gt", "gdc_days_to_death", "os_time_gdc", "flag_x"])
def test_forbidden_fields_rejected(col):
    with pytest.raises(LeakageError):
        make_model("logistic", CLIN + [col])


def test_leakage_requires_explicit_opt_in():
    make_model("logistic", CLIN + ["vital_status_dss"], allow_leakage=True)


def test_preprocessing_fit_on_training_fold_only():
    """A test-only extreme value must not shift the training-fold scaling (scores for
    other test rows are unchanged when one test row is perturbed)."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({c: rng.integers(0, 4, n).astype(float) for c in CLIN})
    y = rng.integers(0, 2, n).astype(float)
    folds = [(np.arange(0, 150), np.arange(150, 200))]
    s1, _ = oof_predict(df, y, folds, "logistic", CLIN)
    df2 = df.copy()
    df2.loc[199, "diagnoses.age_bin"] = 1e6
    s2, _ = oof_predict(df2, y, folds, "logistic", CLIN)
    np.testing.assert_allclose(s1[150:199], s2[150:199])


def test_oof_deterministic():
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({c: rng.integers(0, 4, n).astype(float) for c in CLIN})
    y = rng.integers(0, 2, n).astype(float)
    folds = [(np.arange(60, 120), np.arange(0, 60)), (np.arange(0, 60), np.arange(60, 120))]
    a, _ = oof_predict(df, y, folds, "hgb", CLIN)
    b, _ = oof_predict(df, y, folds, "hgb", CLIN)
    np.testing.assert_array_equal(a, b)


def test_folds_loader_rejects_disallowed_globals(tmp_path):
    import pickle

    from src.folds import load_folds

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    body = pickle.dumps(Evil(), protocol=2)
    header = "{'descr': '|O', 'fortran_order': False, 'shape': (1,), }".ljust(117) + "\n"
    p = tmp_path / "evil.npy"
    p.write_bytes(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header.encode() + body)
    with pytest.raises(pickle.UnpicklingError):
        load_folds(p)

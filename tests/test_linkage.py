"""Integration tests on the real (downloaded) data. Skipped on a clean checkout
until `python -m src.download_data` has been run."""
import numpy as np
import pandas as pd
import pytest

from src.paths import GDC, MMIST

pytestmark = pytest.mark.skipif(not (MMIST / "features" / "demographics.csv").exists()
                                or not (GDC / "cases_mmist.json").exists(),
                                reason="raw data not downloaded")


@pytest.fixture(scope="module")
def L():
    from src.link_sources import build
    return build()[0]


def test_one_row_per_patient(L):
    assert len(L) == 1365
    assert L["cases.submitter_id"].is_unique


def test_all_linked_to_gdc_and_cohort_verified(L):
    assert L.gdc_project.notna().all()
    assert (L.cohort == L.cohort_verified).all()
    assert (L.cohort == "TCGA").sum() == 1026 and (L.cohort == "CPTAC").sum() == 339


def test_released_rules_reproduce_exactly(L):
    t, c = L[L.cohort == "TCGA"], L[L.cohort == "CPTAC"]
    assert (t.y_rule_tcga == t.y_released_death).all()
    assert (c.y_rule_notebook == c.y_released_death).all()


def test_death_times_match_gdc_tcga(L):
    t = L[L.cohort == "TCGA"]
    both = t["demographic.days_to_death"].notna() & t.gdc_days_to_death.notna()
    assert both.sum() == 399
    assert (t.loc[both, "demographic.days_to_death"] == t.loc[both, "gdc_days_to_death"]).all()


def test_primary_counts_locked(L):
    """Regression lock on headline audit numbers (investigate, don't edit, if these change)."""
    k = L.y_gdc_gt.notna()
    assert k.sum() == 1154
    assert (L.y_gdc_gt == 1).sum() == 160
    disc = k & (L.y_gdc_gt != L.y_released_death)
    assert disc.sum() == 88
    assert (disc & (L.cohort == "TCGA")).sum() == 0
    assert (~k & (L.y_released_death == 0)).sum() == 211  # every unknown released as survivor


def test_no_unknown_assumed_survivor(L):
    """Patients alive with follow-up < 365 must be unknown, never survivors."""
    m = L.gdc_vital_status.ne("Dead") & (L.gdc_last_fu < 365)
    assert L.loc[m, "y_gdc_gt"].isna().all()


def test_linkedomics_join_one_to_one(L):
    assert L.lo_id.notna().sum() == 218
    assert L.lo_id.dropna().is_unique


def test_released_folds_reproduced_exactly(L):
    """folds.npy equals StratifiedKFold(seed 42) on released labels in the generation order."""
    from sklearn.model_selection import StratifiedKFold

    from src.folds import benchmark_order, load_folds
    from src.io import ID
    y = L.set_index(ID).loc[benchmark_order()].vital_status_12.to_numpy()
    regen = StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)
    for (_, te), (_, rte) in zip(load_folds(), regen):
        np.testing.assert_array_equal(np.sort(te), np.sort(rte))

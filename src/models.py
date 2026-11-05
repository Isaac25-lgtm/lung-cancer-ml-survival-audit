"""Model definitions. All preprocessing lives inside sklearn Pipelines so it is fit
on training folds only. A guard rejects any outcome or follow-up field unless the
caller explicitly opts into the quarantined leakage experiment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ORDINAL = ["diagnoses.age_bin", "exposures.pack_years_smoked", "exposures.tobacco_smoking_status",
           "exposures.tobacco_years_smoked", "diagnoses.ajcc_pathologic_m", "diagnoses.ajcc_pathologic_n",
           "diagnoses.ajcc_pathologic_stage", "diagnoses.ajcc_pathologic_t",
           "diagnoses.ajcc_staging_system_edition"]
NOMINAL = ["demographic.ethnicity", "demographic.gender", "demographic.race", "diagnoses.icd_10_code",
           "diagnoses.morphology", "diagnoses.prior_malignancy"]
CLIN = ORDINAL + NOMINAL  # the 15 released clinical features
STAGE = ["diagnoses.ajcc_pathologic_stage"]

FORBIDDEN = {"demographic.days_to_death", "days_to_last_fup", "vital_status_12", "vital_status_dss"}
FORBIDDEN_PREFIXES = ("y_", "gdc_", "cdr_", "lo_", "os_", "flag_")


class LeakageError(ValueError):
    pass


def check_features(cols: list[str], allow_leakage: bool = False) -> None:
    bad = [c for c in cols if c in FORBIDDEN or c.startswith(FORBIDDEN_PREFIXES)]
    if bad and not allow_leakage:
        raise LeakageError(f"Outcome/follow-up fields in predictor set: {bad}")


def _nominal_to_str(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(object).where(X.notna(), "missing").astype(str)


def make_model(kind: str, features: list[str], allow_leakage: bool = False, seed: int = 437) -> Pipeline:
    check_features(features, allow_leakage)
    if kind == "prevalence":
        return Pipeline([("clf", DummyClassifier(strategy="prior"))])
    if kind == "xgb":
        from xgboost import XGBClassifier
        # Used without a Pipeline wrapper: xgboost 2.1 is incompatible with sklearn 1.6 Pipeline tags.
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed,
                             n_jobs=2, eval_metric="logloss", verbosity=0)
    if kind == "hgb":
        return Pipeline([("clf", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, random_state=seed))])
    if kind in {"logistic", "stage_only", "mlp"}:
        ords = [c for c in features if c not in NOMINAL]
        noms = [c for c in features if c in NOMINAL]
        if kind == "stage_only":
            ords, noms = [], features  # stage treated as categorical (incl. missing)
        transformers = []
        if ords:
            transformers.append(("ord", Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("sc", StandardScaler())]), ords))
        if noms:
            transformers.append(("nom", Pipeline([
                ("str", FunctionTransformer(_nominal_to_str)),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), noms))
        if kind == "mlp":
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, early_stopping=True, max_iter=500,
                                random_state=seed)
        else:
            clf = LogisticRegression(C=1.0, max_iter=5000)
        return Pipeline([("pre", ColumnTransformer(transformers)), ("clf", clf)])
    raise ValueError(kind)


def oof_predict(df: pd.DataFrame, y_train: np.ndarray, folds, kind: str, features: list[str],
                train_mask: np.ndarray | None = None, allow_leakage: bool = False, seed: int = 437):
    """Out-of-fold P(early death). Training rows = fold-train ∩ train_mask ∩ non-missing label.

    Returns (scores, thresholds) where thresholds[i] is the training-fold event
    prevalence used for the test patient i (pre-specified decision threshold).
    """
    n = len(df)
    y_train = np.asarray(y_train, float)
    train_mask = np.ones(n, bool) if train_mask is None else np.asarray(train_mask, bool)
    scores = np.full(n, np.nan)
    thr = np.full(n, np.nan)
    X = df[features]
    for tr, te in folds:
        tr = tr[train_mask[tr] & ~np.isnan(y_train[tr])]
        assert len(np.intersect1d(tr, te)) == 0, "train/test overlap"
        mdl = make_model(kind, features, allow_leakage, seed)
        mdl.fit(X.iloc[tr], y_train[tr].astype(int))
        scores[te] = mdl.predict_proba(X.iloc[te])[:, 1]
        thr[te] = y_train[tr].mean()
    return scores, thr


def fit_predict(df_tr, y_tr, df_te, kind: str, features: list[str], seed: int = 437):
    y_tr = np.asarray(y_tr, float)
    ok = ~np.isnan(y_tr)
    mdl = make_model(kind, features, seed=seed).fit(df_tr[features][ok], y_tr[ok].astype(int))
    return mdl.predict_proba(df_te[features])[:, 1], float(y_tr[ok].mean())

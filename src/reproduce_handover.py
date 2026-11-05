"""Gate check: reproduce the exploratory numbers reported in the feasibility handover.

This replicates the handover's *exploratory* rules exactly (including choices we
later revise, e.g. accepting OS_days >= 365 with a missing event flag). Its only
purpose is to confirm that our inputs and code reproduce the earlier numbers
before anything is changed. Results are written to
outputs/tables/handover_reproduction.csv with expected vs observed values.

Usage: python -m src.reproduce_handover
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.io import load_linkedomics, load_mmist
from src.paths import TABLES

FEATS6 = ["diagnoses.age_bin", "demographic.gender", "diagnoses.ajcc_pathologic_stage",
          "diagnoses.ajcc_pathologic_t", "diagnoses.ajcc_pathologic_n", "diagnoses.ajcc_pathologic_m"]


def _cat(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].astype(object).where(df[cols].notna(), "Unknown").astype(str)


def _pipe(model) -> Pipeline:
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), slice(0, None))])
    return Pipeline([("pre", pre), ("clf", model)])


def run() -> pd.DataFrame:
    m = load_mmist()
    rows: list[tuple[str, float, float]] = []

    def rec(name, expected, observed):
        rows.append((name, expected, observed))

    dd, fu, y = m["demographic.days_to_death"], m["days_to_last_fup"], m["vital_status_12"]
    t = m.cohort == "TCGA"
    rec("TCGA deaths day 0-364 with released label 0", 133, int(((dd >= 0) & (dd < 365) & t & (y == 0)).sum()))
    rec("TCGA deaths day >=365 with released label 1", 265, int(((dd >= 365) & t & (y == 1)).sum()))
    rec("TCGA deaths exactly day 365 with label 0", 1, int(((dd == 365) & t & (y == 0)).sum()))
    c = m.cohort == "CPTAC"
    rec("CPTAC deaths day 0-364 with label 1", 27, int(((dd >= 0) & (dd < 365) & c & (y == 1)).sum()))
    rec("CPTAC deaths day >=365 with label 0", 62, int(((dd >= 365) & c & (y == 0)).sum()))
    known_b = ((dd >= 0) & (dd < 365)) | (dd >= 365) | (fu >= 365)
    rec("TCGA known by benchmark-field rule", 859, int((known_b & t).sum()))
    rec("CPTAC known by benchmark-field rule", 89, int((known_b & c).sum()))
    rec("Not established by benchmark-field rule", 417, int((~known_b).sum()))

    lo = load_linkedomics()
    L = m.merge(lo, left_on="cases.submitter_id", right_on="case_id", validate="one_to_one")
    rec("LinkedOmics rows matched to MMIST", 218, len(L))
    known = L["OS_days"].ge(365) | (L["OS_event"].eq(1) & L["OS_days"].ge(0) & L["OS_days"].lt(365))
    K = L[known].copy()
    K["src"] = K["OS_days"].ge(365).astype(int)
    rec("LinkedOmics known (exploratory rule)", 177, len(K))
    rec("  of which LUAD", 94, int((K["histology"] == "LUAD").sum()))
    rec("  of which LSCC", 83, int((K["histology"] == "LSCC").sum()))
    rec("Source early deaths (<365)", 13, int((K.src == 0).sum()))
    rec("Mismatched released label", 56, int((K.src != K.vital_status_12).sum()))
    rec("  LUAD mismatched", 25, int(((K.src != K.vital_status_12) & (K["histology"] == "LUAD")).sum()))
    rec("  LSCC mismatched", 31, int(((K.src != K.vital_status_12) & (K["histology"] == "LSCC")).sum()))
    both = K[K["OS_event"].eq(1) & K["demographic.days_to_death"].notna()]
    rec("Comparable deaths with equal times", 46, int((both["OS_days"] == both["demographic.days_to_death"]).sum()))
    rec("Source censored >=365 with released label 0", 10,
        int((K["OS_event"].eq(0) & (K.src == 1) & (K.vital_status_12 == 0)).sum()))
    rec("Unknown among 218", 41, int((~known).sum()))

    # Exploratory TCGA -> CPTAC model (handover section 4.3)
    tr = m[t & known_b].copy()
    tr["alive12"] = np.where((tr["demographic.days_to_death"] >= 0) & (tr["demographic.days_to_death"] < 365), 0, 1)
    rec("Training n (TCGA known)", 859, len(tr))
    Xtr, Xte = _cat(tr, FEATS6), _cat(K, FEATS6)
    exp = {"Logistic": (0.6977016886, 0.5803540437, 0.0757741870, 0.1942493156),
           "Random forest": (0.6845684803, 0.5400902464, 0.0732067762, 0.1905120908)}
    models = {"Logistic": LogisticRegression(max_iter=2000),
              "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=8,
                                                      random_state=437, n_jobs=2)}
    for name, mdl in models.items():
        p = _pipe(mdl).fit(Xtr, tr["alive12"]).predict_proba(Xte)[:, 1]
        e = exp[name]
        rec(f"{name} AUC source", e[0], roc_auc_score(K.src, p))
        rec(f"{name} AUC released", e[1], roc_auc_score(K.vital_status_12, p))
        rec(f"{name} Brier source", e[2], brier_score_loss(K.src, p))
        rec(f"{name} Brier released", e[3], brier_score_loss(K.vital_status_12, p))
        if name == "Logistic":
            rng = np.random.default_rng(437)
            diffs = []
            ys, yr = K.src.to_numpy(), K.vital_status_12.to_numpy()
            for _ in range(2000):
                i = rng.integers(0, len(p), len(p))
                if len(set(ys[i])) < 2 or len(set(yr[i])) < 2:
                    continue
                diffs.append(roc_auc_score(ys[i], p[i]) - roc_auc_score(yr[i], p[i]))
            lo_, hi_ = np.percentile(diffs, [2.5, 97.5])
            rec("Logistic bootstrap dAUC 2.5%", -0.064, lo_)
            rec("Logistic bootstrap dAUC 97.5%", 0.275, hi_)
            rng = np.random.default_rng(9)
            bd = []
            for _ in range(5000):
                i = rng.integers(0, len(p), len(p))
                bd.append(brier_score_loss(yr[i], p[i]) - brier_score_loss(ys[i], p[i]))
            rec("Logistic dBrier point", 0.1185, brier_score_loss(yr, p) - brier_score_loss(ys, p))
            rec("Logistic dBrier 2.5%", 0.065, np.percentile(bd, 2.5))
            rec("Logistic dBrier 97.5%", 0.174, np.percentile(bd, 97.5))

    # Leakage scenario (handover 4.4): 5-fold CV logistic on TCGA n=1026, released labels
    T = m[t].copy()
    skf = StratifiedKFold(5, shuffle=True, random_state=437)
    for label, cols, e in [("baseline 6 features", FEATS6, 0.6139),
                           ("+ vital_status_dss", FEATS6 + ["vital_status_dss"], 0.6742)]:
        oof = np.zeros(len(T))
        X = _cat(T, cols)
        for a, b in skf.split(X, T.vital_status_12):
            oof[b] = _pipe(LogisticRegression(max_iter=2000)).fit(X.iloc[a], T.vital_status_12.iloc[a]).predict_proba(X.iloc[b])[:, 1]
        rec(f"Leakage CV AUC {label}", e, roc_auc_score(T.vital_status_12, oof))

    out = pd.DataFrame(rows, columns=["quantity", "handover_value", "reproduced_value"])
    out["abs_diff"] = (out.reproduced_value - out.handover_value).abs()
    out["match"] = np.where(out.handover_value.abs() >= 2, out.abs_diff < 0.5, out.abs_diff < 0.006)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "handover_reproduction.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    r = run()
    print(r.to_string(index=False))
    print(f"\n{int(r.match.sum())}/{len(r)} handover quantities reproduced")

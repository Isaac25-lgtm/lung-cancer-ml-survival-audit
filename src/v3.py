"""Protocol v3 (docs/protocol_v3.md): relabelling effect with a consistent-multiplicity
patient bootstrap, on MMIST-LUNG in the released folds.

    python -m src.v3 [reps_logistic] [reps_hgb]

Output: outputs/tables/v3_training_label_bootstrap.csv, outputs/reports/v3_results.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from src import metrics as M
from src.paths import REPORTS, TABLES

SEED = 2026110504
VARIANTS = {"B1": ("y_released_death", False), "B2": ("y_released_death", True), "B3": ("y_gdc_gt", True)}


def _fold_avg_auroc(y, s, groups):
    vals = [M.auroc(y[g], s[g]) for g in groups]
    return float(np.nanmean(vals))


def one_replicate(X, ys, known, folds, make, mult):
    """Fit every variant in every fold using row multiplicities `mult` (one per patient)."""
    scores = {v: [] for v in VARIANTS}
    ytest, groups, offset = [], [], 0
    yS = ys["y_gdc_gt"]
    for tr, te in folds:
        tr_rep = np.repeat(tr, mult[tr])
        te_k = te[known[te]]
        te_rep = np.repeat(te_k, mult[te_k])
        for v, (col, known_only) in VARIANTS.items():
            rows = tr_rep[known[tr_rep]] if known_only else tr_rep
            y = ys[col][rows].astype(int)
            m = make().fit(X.iloc[rows], y)
            scores[v].append(m.predict_proba(X.iloc[te_rep])[:, 1])
        ytest.append(yS[te_rep])
        groups.append(np.arange(offset, offset + len(te_rep)))
        offset += len(te_rep)
    y_all = np.concatenate(ytest)
    auc = {v: _fold_avg_auroc(y_all, np.concatenate(s), groups) for v, s in scores.items()}
    return auc


def run(reps_logistic: int = 1000, reps_hgb: int = 300) -> pd.DataFrame:
    from src.experiments import load
    from src.models import CLIN, make_model
    L, folds = load()
    X = L[CLIN]
    ys = {c: L[c].to_numpy(float) for c in ["y_released_death", "y_gdc_gt"]}
    known = ~np.isnan(ys["y_gdc_gt"])
    rows, out = [], {}
    for kind, reps in [("logistic", reps_logistic), ("hgb", reps_hgb)]:
        make = lambda k=kind: make_model(k, CLIN)
        t0 = time.time()
        point = one_replicate(X, ys, known, folds, make, np.ones(len(L), int))
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(reps):
            mult = np.zeros(len(L), int)
            for _, te in folds:  # folds partition patients: each multiplicity drawn exactly once
                draw = rng.choice(te, len(te), replace=True)
                np.add.at(mult, draw, 1)
            boot.append(one_replicate(X, ys, known, folds, make, mult))
        b = pd.DataFrame(boot)
        for name, (a, c) in {"relabelling effect: B3 - B2 (same training patients)": ("B3", "B2"),
                             "training-strategy comparison: B3 - B1": ("B3", "B1")}.items():
            d = (b[a] - b[c]).to_numpy()
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            rows.append({"model": kind, "contrast": name, "auroc_a": point[a], "auroc_b": point[c],
                         "delta": point[a] - point[c], "ci_lo": float(lo), "ci_hi": float(hi),
                         "share_positive": float(np.mean(d > 0)), "reps": reps})
        out[kind] = {"point": point, "runtime_s": round(time.time() - t0, 1)}
        print(kind, json.dumps(rows[-2:], default=float), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "v3_training_label_bootstrap.csv", index=False)
    (REPORTS / "v3_results.json").write_text(json.dumps({"seed": SEED, "results": rows, "points": out}, indent=2, default=float))
    return res


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    run(*a)

"""What were the released folds stratified on?

The benchmark's utils.load_or_create_folds uses StratifiedKFold(5, shuffle=True,
random_state=42) on the dataset's labels in dataset order. We regenerate folds for
candidate labels and patient orderings and compare them with folds.npy.

Writes outputs/tables/fold_provenance.csv (aggregate only).
Usage: python -m src.fold_provenance
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.folds import benchmark_order, load_folds
from src.io import ID
from src.link_sources import load_linked
from src.paths import TABLES


def candidates(L: pd.DataFrame) -> dict[str, np.ndarray]:
    d = L["demographic.days_to_death"]
    return {
        "vital_status_12 (released)": L.vital_status_12.to_numpy(),
        "vital_status_dss (released)": L.vital_status_dss.to_numpy(),
        "TCGA rule applied to all patients": ((d > 365) | d.isna()).astype(int).to_numpy(),
        "notebook rule applied to all patients": ((d <= 365) | d.isna()).astype(int).to_numpy(),
        "death recorded (any time)": d.notna().astype(int).to_numpy(),
        "cohort": (L.cohort == "CPTAC").astype(int).to_numpy(),
    }


def run() -> pd.DataFrame:
    base = load_linked().sort_values(ID).reset_index(drop=True)
    released = [set(te.tolist()) for _, te in load_folds()]
    orders = {
        "ascending submitter_id (PatientDataset order)": base,
        "TCGA first, then CPTAC (both sorted)": pd.concat([base[base.cohort == "TCGA"], base[base.cohort == "CPTAC"]]),
        "sorted TCGA, then CPTAC in diagnosis.csv order": base.set_index(ID).loc[benchmark_order()].reset_index(),
    }
    rows = []
    for oname, df in orders.items():
        df = df.reset_index(drop=True)
        for lname, y in candidates(df).items():
            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            regen = [set(te.tolist()) for _, te in skf.split(np.zeros(len(y)), y)]
            exact = sum(any(r == g for g in regen) for r in released)
            best = np.mean([max(len(r & g) for g in regen) / len(r) for r in released])
            rows.append({"ordering": oname, "stratifying_label": lname, "released_folds_reproduced_exactly": exact,
                         "mean_best_overlap": round(float(best), 3)})
    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "fold_provenance.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(run().to_string(index=False))

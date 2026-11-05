"""Deterministic patient-level linkage of MMIST-LUNG to upstream source releases.

Keys: MMIST `cases.submitter_id` == GDC `submitter_id` == TCGA-CDR `bcr_patient_barcode`
== LinkedOmics `case_id`. Exact string match after whitespace stripping; no fuzzy
matching. Every join is validated one-to-one.

Outputs
  data/interim/linked.csv                 patient-level (gitignored; contains IDs)
  outputs/tables/linkage_summary.csv      aggregate join diagnostics
  outputs/tables/death_time_agreement.csv aggregate cross-release death-time agreement
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import derive_outcomes as do
from src.io import ID, load_cdr, load_gdc, load_linkedomics, load_mmist
from src.paths import INTERIM, TABLES


def build() -> tuple[pd.DataFrame, dict]:
    m = load_mmist()
    m[ID] = m[ID].str.strip()
    g, release = load_gdc()
    cdr = load_cdr()
    lo = load_linkedomics()

    L = m.merge(g, left_on=ID, right_on="id", how="left", validate="one_to_one", indicator="_gdc")
    cdr_cols = {"bcr_patient_barcode": "cdr_id", "OS": "cdr_OS", "OS.time": "cdr_OS.time",
                "DSS": "cdr_DSS", "DSS.time": "cdr_DSS.time", "vital_status": "cdr_vital_status",
                "cause_of_death": "cdr_cause_of_death"}
    L = L.merge(cdr[list(cdr_cols)].rename(columns=cdr_cols), left_on=ID, right_on="cdr_id",
                how="left", validate="one_to_one", indicator="_cdr")
    lo_cols = {"case_id": "lo_id", "OS_days": "lo_OS_days", "OS_event": "lo_OS_event", "histology": "lo_histology"}
    L = L.merge(lo[list(lo_cols)].rename(columns=lo_cols), left_on=ID, right_on="lo_id",
                how="left", validate="one_to_one", indicator="_lo")
    assert len(L) == len(m), "row count changed during linkage"

    # Verified cohort / histology from GDC project
    proj_cohort = L["gdc_project"].map({"TCGA-LUAD": "TCGA", "TCGA-LUSC": "TCGA", "CPTAC-3": "CPTAC"})
    L["cohort_verified"] = proj_cohort
    L["histology"] = np.select(
        [L.gdc_project.eq("TCGA-LUAD"), L.gdc_project.eq("TCGA-LUSC"), L.lo_histology.eq("LUAD"),
         L.lo_histology.eq("LSCC")], ["LUAD", "LUSC", "LUAD", "LUSC"], default="unknown")
    # CPTAC without LinkedOmics row: fall back to GDC disease-type independent morphology code
    # (MMIST encodes 8070-8083 squamous as codes 12-17; 8140-8550 adenocarcinoma as 0-11)
    morph = L["diagnoses.morphology"]
    fallback = np.where(morph >= 12, "LUSC", "LUAD")
    L.loc[L.histology.eq("unknown"), "histology"] = fallback[L.histology.eq("unknown").to_numpy()]

    # Outcomes
    L["y_released_death"] = 1 - L["vital_status_12"]
    for conv in ["gt", "ge"]:
        L[f"y_gdc_{conv}"] = do.from_gdc(L, conv)
        L[f"y_cdr_{conv}"] = np.where(L._cdr.eq("both"), do.from_cdr(L, conv), np.nan)
        L[f"y_lo_{conv}"] = np.where(L._lo.eq("both"), do.from_linkedomics(L, conv, strict=True), np.nan)
        L[f"y_mmistfields_{conv}"] = do.from_mmist_fields(L, conv)
        alt = np.where(L.cohort.eq("TCGA"), L[f"y_cdr_{conv}"], L[f"y_lo_{conv}"])
        L[f"y_alt_{conv}"] = alt
        # consensus: GDC and the alternative release agree; unknown if either unknown or they differ
        a, b = L[f"y_gdc_{conv}"].to_numpy(), alt
        L[f"y_consensus_{conv}"] = np.where(~np.isnan(a) & ~np.isnan(b) & (a == b), a, np.nan)
    L["y_rule_tcga"] = 1 - do.released_rule_tcga(L["demographic.days_to_death"])
    L["y_rule_notebook"] = 1 - do.released_rule_notebook(L["demographic.days_to_death"])
    t, e = do.time_event_gdc(L)
    L["os_time_gdc"], L["os_event_gdc"] = t, e

    # Version / provenance flags (aggregated in reports; patient IDs never published)
    mm_d, g_d = L["demographic.days_to_death"], L["gdc_days_to_death"]
    L["flag_death_removed_upstream"] = mm_d.notna() & g_d.isna()
    L["flag_death_added_upstream"] = mm_d.isna() & g_d.notna()
    L["flag_death_time_changed"] = mm_d.notna() & g_d.notna() & (mm_d != g_d)
    L["flag_negative_time"] = (mm_d < 0) | (g_d < 0)
    L["flag_non_diagnosis_index"] = L["gdc_index_date"].ne("Diagnosis")
    L["flag_dead_no_date"] = L["gdc_vital_status"].eq("Dead") & g_d.isna()

    diag = {
        "gdc_release": release,
        "n_mmist": len(m),
        "n_gdc_matched": int(L._gdc.eq("both").sum()),
        "n_cdr_matched": int(L._cdr.eq("both").sum()),
        "n_cdr_lung_rows": len(cdr),
        "n_linkedomics_matched": int(L._lo.eq("both").sum()),
        "n_linkedomics_rows": len(lo),
        "cohort_prefix_equals_gdc_project": int((L.cohort == L.cohort_verified).sum()),
    }
    return L, diag


def death_time_agreement(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, sub in L.groupby("cohort"):
        mm = sub["demographic.days_to_death"]
        pairs = {
            "GDC DR46": sub["gdc_days_to_death"],
            "TCGA-CDR": sub["cdr_OS.time"].where(sub["cdr_OS"].eq(1)),
            "LinkedOmics": sub["lo_OS_days"].where(sub["lo_OS_event"].eq(1)),
        }
        for src, s in pairs.items():
            both = mm.notna() & s.notna()
            if not (mm.notna() | s.notna()).any():
                continue
            rows.append({
                "cohort": cohort, "comparator": src,
                "mmist_deaths": int(mm.notna().sum()), "comparator_deaths": int(s.notna().sum()),
                "both_recorded": int(both.sum()),
                "identical_days": int((mm[both] == s[both]).sum()),
                "abs_diff_gt_30d": int(((mm[both] - s[both]).abs() > 30).sum()),
                "mmist_only": int((mm.notna() & s.isna() & (s.index.isin(sub.index))).sum()),
                "comparator_only": int((mm.isna() & s.notna()).sum()),
            })
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    L, diag = build()
    INTERIM.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    L.to_csv(INTERIM / "linked.csv", index=False)
    pd.DataFrame([diag]).T.rename(columns={0: "value"}).to_csv(TABLES / "linkage_summary.csv")
    death_time_agreement(L).to_csv(TABLES / "death_time_agreement.csv", index=False)
    return L


def load_linked() -> pd.DataFrame:
    p = INTERIM / "linked.csv"
    if not p.exists():
        return run()
    return pd.read_csv(p, low_memory=False)


if __name__ == "__main__":
    L = run()
    print(pd.read_csv(TABLES / "linkage_summary.csv").to_string(index=False))
    print(pd.read_csv(TABLES / "death_time_agreement.csv").to_string(index=False))

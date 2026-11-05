"""Loaders for raw inputs with schema and integrity checks.

All loaders return copies; raw files are never modified.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.paths import CPTAC, GDC, MMIST

ID = "cases.submitter_id"


class SchemaError(ValueError):
    pass


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}")


def _unique(df: pd.DataFrame, col: str, name: str) -> None:
    if df[col].isna().any():
        raise SchemaError(f"{name}: null IDs in {col}")
    if not df[col].is_unique:
        dups = df.loc[df[col].duplicated(), col].tolist()[:5]
        raise SchemaError(f"{name}: duplicate IDs in {col}, e.g. {dups}")


def load_mmist() -> pd.DataFrame:
    """Demographics + diagnosis joined one-to-one, with cohort from ID prefix.

    Cohort-by-prefix is verified against GDC project membership in
    src.link_sources (it matches 1365/1365 for the pinned commit).
    """
    d = pd.read_csv(MMIST / "features" / "demographics.csv")
    g = pd.read_csv(MMIST / "features" / "diagnosis.csv")
    _require(d, [ID, "demographic.days_to_death", "days_to_last_fup", "vital_status_12",
                 "vital_status_dss", "diagnoses.age_bin", "demographic.gender"], "demographics")
    _require(g, [ID, "diagnoses.ajcc_pathologic_stage", "diagnoses.ajcc_pathologic_t",
                 "diagnoses.ajcc_pathologic_n", "diagnoses.ajcc_pathologic_m"], "diagnosis")
    _unique(d, ID, "demographics")
    _unique(g, ID, "diagnosis")
    m = d.merge(g, on=ID, how="inner", validate="one_to_one")
    if len(m) != len(d) or len(m) != len(g):
        raise SchemaError("demographics/diagnosis ID sets differ")
    if not set(m["vital_status_12"].unique()) <= {0, 1}:
        raise SchemaError("vital_status_12 not binary")
    m["cohort"] = np.where(m[ID].str.startswith("TCGA"), "TCGA", "CPTAC")
    return m


def load_linkedomics() -> pd.DataFrame:
    parts = []
    for h in ["LUAD", "LSCC"]:
        s = pd.read_csv(CPTAC / h / f"{h}_survival.txt", sep="\t")
        _require(s, ["case_id", "OS_days", "OS_event", "PFS_days", "PFS_event"], f"LinkedOmics {h}")
        s["histology"] = h
        parts.append(s)
    s = pd.concat(parts, ignore_index=True)
    _unique(s, "case_id", "LinkedOmics LUAD+LSCC")
    bad_event = ~s["OS_event"].isin([0, 1]) & s["OS_event"].notna()
    if bad_event.any():
        raise SchemaError("LinkedOmics OS_event outside {0,1}")
    return s


def load_gdc() -> tuple[pd.DataFrame, str]:
    """Flatten GDC case records to one row per patient."""
    raw = json.loads((GDC / "cases_mmist.json").read_text())
    rows = []
    for h in raw["hits"]:
        dm = h.get("demographic") or {}
        dg = h.get("diagnoses") or []
        fu = h.get("follow_ups") or []
        dlf = [x["days_to_last_follow_up"] for x in dg if x.get("days_to_last_follow_up") is not None]
        dfu = [x["days_to_follow_up"] for x in fu if x.get("days_to_follow_up") is not None]
        rows.append({
            "id": h["submitter_id"],
            "gdc_project": h["project"]["project_id"],
            "gdc_index_date": h.get("index_date"),
            "gdc_vital_status": dm.get("vital_status"),
            "gdc_days_to_death": dm.get("days_to_death"),
            "gdc_cause_of_death": dm.get("cause_of_death"),
            "gdc_last_fu_diag": max(dlf) if dlf else np.nan,
            "gdc_last_fu_followups": max(dfu) if dfu else np.nan,
            "gdc_n_diagnoses": len(dg),
        })
    g = pd.DataFrame(rows)
    _unique(g, "id", "GDC cases")
    g["gdc_days_to_death"] = pd.to_numeric(g["gdc_days_to_death"], errors="coerce")
    g["gdc_last_fu"] = g[["gdc_last_fu_diag", "gdc_last_fu_followups"]].max(axis=1)
    return g, raw.get("data_release", "unknown")


def load_cdr() -> pd.DataFrame:
    x = pd.read_excel(GDC / "TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR")
    x = x[x["type"].isin(["LUAD", "LUSC"])].copy()
    _require(x, ["bcr_patient_barcode", "OS", "OS.time", "DSS", "DSS.time", "vital_status"], "TCGA-CDR")
    _unique(x, "bcr_patient_barcode", "TCGA-CDR")
    for c in ["OS", "OS.time", "DSS", "DSS.time", "death_days_to", "last_contact_days_to"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x

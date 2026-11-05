"""Dashboard data access: aggregate tables only, with integrity/staleness checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "outputs" / "reports"

REQUIRED = [
    "discordance_summary", "label_agreement", "rule_reproduction", "source_manifest", "linkage_summary",
    "death_time_agreement", "release_reconciliation", "km_365", "km_curves", "unknown_reasons",
    "fig2_death_time_histogram", "exp_a_metrics", "exp_a_paired_deltas", "exp_a2_transport",
    "exp_a_model_ranking", "exp_a_cohort_mechanism", "exp_b_benchmark_folds", "fold_provenance", "exp_b_summary", "exp_b_paired", "exp_sensitivity", "exp_c_censoring", "exp_d_leakage",
    "falsification_checks", "dss_audit", "i3lung_label_audit", "literature_register",
]


def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def load_results() -> dict:
    return json.loads((REPORTS / "results.json").read_text())


def artefact_status() -> dict:
    missing = [n for n in REQUIRED if not (TABLES / f"{n}.csv").exists()]
    if not (REPORTS / "results.json").exists():
        missing.append("results.json")
    files, stale = [], []
    hashes = {}
    hp = REPORTS / "table_hashes.json"
    if hp.exists():
        hashes = json.loads(hp.read_text())
    for n in REQUIRED:
        p = TABLES / f"{n}.csv"
        if not p.exists():
            continue
        h = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        ok = hashes.get(f"{n}.csv") == h if hashes else None
        if ok is False:
            stale.append(n)
        files.append({"table": f"{n}.csv", "sha256": h[:16] + "…",
                      "matches run manifest": {True: "yes", False: "NO (edited after run)", None: "no manifest"}[ok]})
    return {"missing": missing, "stale": stale, "files": files}


def fmt_ci(p, lo, hi) -> str:
    if pd.isna(p):
        return "n/a"
    return f"{p:.1f}% ({lo:.1f}–{hi:.1f})"

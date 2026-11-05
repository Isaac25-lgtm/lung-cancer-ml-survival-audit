"""Label audit: rule reproduction, released-vs-source agreement, unknown outcomes,
cross-release reconciliation, DSS-label check and Kaplan-Meier anchors.

All outputs are aggregate counts (no patient identifiers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.link_sources import load_linked
from src.paths import TABLES
from src.stats_utils import km_at, wilson

STATE = {np.nan: "unknown", 0.0: "alive >365d", 1.0: "early death"}


def _state(y: pd.Series) -> pd.Series:
    return y.map(lambda v: "unknown" if pd.isna(v) else ("early death" if v == 1 else "alive >365d"))


def rule_reproduction(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, s in L.groupby("cohort"):
        for rule, col in [("TCGA rule: survivor if death >365d or no death", "y_rule_tcga"),
                          ("Notebook rule (encode-csv cell 14): survivor if death <=365d or no death",
                           "y_rule_notebook")]:
            agree = int((s[col] == s["y_released_death"]).sum())
            rows.append({"cohort": cohort, "rule": rule, "n": len(s), "agree": agree,
                         "pct_agree": round(100 * agree / len(s), 2)})
    return pd.DataFrame(rows)


def agreement_table(L: pd.DataFrame, source: str) -> pd.DataFrame:
    """Released label x source-derived state, by cohort and histology."""
    t = L.assign(source_state=_state(L[f"y_{source}"]),
                 released=np.where(L.y_released_death == 1, "released early death", "released survivor"))
    out = (t.groupby(["cohort", "histology", "released", "source_state"]).size()
           .rename("n").reset_index())
    out.insert(0, "source", source)
    return out


def discordance_summary(L: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    rows = []
    groups = [("All", L)] + [(c, s) for c, s in L.groupby("cohort")] + \
             [(f"{c} {h}", s) for (c, h), s in L.groupby(["cohort", "histology"])]
    for src in sources:
        y = L[f"y_{src}"]
        for name, s in groups:
            ys = y.loc[s.index]
            known = ys.notna()
            r = s.y_released_death
            disc = known & (ys != r)
            k_n = int(known.sum())
            d_n = int(disc.sum())
            lo, hi = wilson(d_n, k_n)
            src_deaths = int((ys == 1).sum())
            rel_deaths = int((r == 1).sum())
            tp = int(((ys == 1) & (r == 1)).sum())
            rows.append({
                "source": src, "group": name, "n": len(s), "known": k_n,
                "unknown": int((~known).sum()),
                "unknown_released_as_survivor": int((~known & (r == 0)).sum()),
                "source_early_deaths": src_deaths, "released_early_deaths": rel_deaths,
                "released_early_deaths_among_known": int((known & (r == 1)).sum()),
                "discordant": d_n,
                "discordant_pct": round(100 * d_n / k_n, 2) if k_n else np.nan,
                "discordant_ci_lo": round(100 * lo, 2) if k_n else np.nan,
                "discordant_ci_hi": round(100 * hi, 2) if k_n else np.nan,
                "source_death_labelled_survivor": int(((ys == 1) & (r == 0)).sum()),
                "source_survivor_labelled_death": int(((ys == 0) & (r == 1)).sum()),
                "released_sensitivity_for_early_death": round(tp / src_deaths, 3) if src_deaths else np.nan,
                "released_ppv_for_early_death": round(tp / int((known & (r == 1)).sum()), 3)
                if int((known & (r == 1)).sum()) else np.nan,
            })
    return pd.DataFrame(rows)


def unknown_reasons(L: pd.DataFrame) -> pd.DataFrame:
    """Why the primary (GDC, gt) status is unknown."""
    u = L[L.y_gdc_gt.isna()].copy()
    reason = np.select(
        [u.flag_negative_time & u.gdc_vital_status.eq("Dead"),
         u.flag_dead_no_date,
         u.gdc_vital_status.ne("Dead") & u.gdc_last_fu.isna(),
         u.gdc_vital_status.ne("Dead") & (u.gdc_last_fu < 365)],
        ["negative death time", "dead, no death date", "no follow-up time recorded",
         "censored before 365 days"], default="other")
    u["reason"] = reason
    out = u.groupby(["cohort", "gdc_vital_status", "reason"], dropna=False).size().rename("n").reset_index()
    return out


def release_reconciliation(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    c = L[L.cohort == "CPTAC"]
    lo_cens_late_death = c.lo_OS_event.eq(0) & c["demographic.days_to_death"].notna()
    rows.append({"check": "CPTAC: LinkedOmics censored but MMIST/GDC record a later death",
                 "n": int(lo_cens_late_death.sum()),
                 "detail": f"in all {int(lo_cens_late_death.sum())}, LinkedOmics censoring day < MMIST death day: "
                           f"{bool((c.lo_OS_days[lo_cens_late_death] < c['demographic.days_to_death'][lo_cens_late_death]).all())}; "
                           f"LinkedOmics censoring day >= 365 in "
                           f"{int((c.lo_OS_days[lo_cens_late_death] >= 365).sum())}"})
    for flag, desc in [("flag_death_removed_upstream", "death in MMIST, none in GDC DR46 (now Alive)"),
                       ("flag_death_added_upstream", "no death in MMIST, death in GDC DR46"),
                       ("flag_death_time_changed", "death day differs MMIST vs GDC"),
                       ("flag_negative_time", "negative death time"),
                       ("flag_non_diagnosis_index", "GDC index date is not diagnosis"),
                       ("flag_dead_no_date", "GDC Dead without days_to_death")]:
        for cohort in ["TCGA", "CPTAC"]:
            s = L[L.cohort == cohort]
            n = int(s[flag].astype(bool).sum())
            if n:
                rows.append({"check": f"{cohort}: {desc}", "n": n, "detail": ""})
    t = L[L.cohort == "TCGA"]
    both = t["cdr_OS"].eq(1) & t["demographic.days_to_death"].notna()
    diff = (t["cdr_OS.time"] - t["demographic.days_to_death"]).abs()
    rows.append({"check": "TCGA: CDR vs MMIST death day differ", "n": int((both & (diff > 0)).sum()),
                 "detail": f">30 days: {int((both & (diff > 30)).sum())}; changes 365-day status: "
                           f"{int((t.y_gdc_gt.notna() & t.y_cdr_gt.notna() & (t.y_gdc_gt != t.y_cdr_gt)).sum())}"})
    return pd.DataFrame(rows)


def dss_audit(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    c = L[L.cohort == "CPTAC"]
    for cod, s in c.groupby(c.gdc_cause_of_death.fillna("(no death)")):
        rows.append({"cohort": "CPTAC", "category": f"GDC cause_of_death = {cod}", "n": len(s),
                     "mmist_vital_status_dss_0": int((s.vital_status_dss == 0).sum()),
                     "mmist_vital_status_dss_1": int((s.vital_status_dss == 1).sum())})
    t = L[L.cohort == "TCGA"]
    for v, s in t.groupby(t.cdr_DSS.fillna(-1)):
        rows.append({"cohort": "TCGA", "category": f"TCGA-CDR DSS = {'NA' if v == -1 else int(v)}", "n": len(s),
                     "mmist_vital_status_dss_0": int((s.vital_status_dss == 0).sum()),
                     "mmist_vital_status_dss_1": int((s.vital_status_dss == 1).sum())})
    return pd.DataFrame(rows)


def km_anchor(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("All", L)] + [(c, s) for c, s in L.groupby("cohort")] + \
             [(f"{c} {h}", s) for (c, h), s in L.groupby(["cohort", "histology"])]
    for name, s in groups:
        S, lo, hi = km_at(s.os_time_gdc.to_numpy(), s.os_event_gdc.to_numpy(), 365)
        usable = int(s.os_time_gdc.notna().sum())
        rows.append({"group": name, "n": len(s), "n_with_time": usable,
                     "km_survival_365": round(S, 4), "km_ci_lo": round(lo, 4), "km_ci_hi": round(hi, 4),
                     "released_survivor_fraction": round(float((s.vital_status_12 == 1).mean()), 4),
                     "known_only_survivor_fraction": round(float((s.y_gdc_gt == 0).sum() / s.y_gdc_gt.notna().sum()), 4)})
    return pd.DataFrame(rows)


def followup_distribution(L: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, -0.5, 182, 364, 365, 730, 1095, 1825, np.inf]
    labels = ["<0", "0-182", "183-364", "365", "366-730", "731-1095", "1096-1825", ">1825"]
    t = L.assign(band=pd.cut(L.os_time_gdc, bins=bins, labels=labels).astype(str).replace("nan", "missing"),
                 status=np.where(L.os_event_gdc == 1, "death", np.where(L.os_event_gdc == 0, "censored", "no usable time")))
    return t.groupby(["cohort", "status", "band"]).size().rename("n").reset_index()


def sample_flow(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, s in L.groupby("cohort"):
        rows += [
            {"cohort": cohort, "stage": "MMIST-LUNG patients", "n": len(s)},
            {"cohort": cohort, "stage": "linked to GDC DR46", "n": int(s.gdc_project.notna().sum())},
            {"cohort": cohort, "stage": "365-day status ascertainable (GDC)", "n": int(s.y_gdc_gt.notna().sum())},
            {"cohort": cohort, "stage": "  early death <=365d", "n": int((s.y_gdc_gt == 1).sum())},
            {"cohort": cohort, "stage": "  alive >365d", "n": int((s.y_gdc_gt == 0).sum())},
            {"cohort": cohort, "stage": "not ascertainable", "n": int(s.y_gdc_gt.isna().sum())},
            {"cohort": cohort, "stage": "released label consistent (known)",
             "n": int((s.y_gdc_gt.notna() & (s.y_gdc_gt == s.y_released_death)).sum())},
            {"cohort": cohort, "stage": "released label inconsistent (known)",
             "n": int((s.y_gdc_gt.notna() & (s.y_gdc_gt != s.y_released_death)).sum())},
        ]
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    L = load_linked()
    sources = ["gdc_gt", "gdc_ge", "alt_gt", "consensus_gt", "mmistfields_gt"]
    out = {
        "rule_reproduction": rule_reproduction(L),
        "label_agreement": pd.concat([agreement_table(L, s) for s in sources], ignore_index=True),
        "discordance_summary": discordance_summary(L, sources),
        "unknown_reasons": unknown_reasons(L),
        "release_reconciliation": release_reconciliation(L),
        "dss_audit": dss_audit(L),
        "km_365": km_anchor(L),
        "followup_distribution": followup_distribution(L),
        "sample_flow": sample_flow(L),
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_csv(TABLES / f"{k}.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    o = run()
    for k in ["rule_reproduction", "unknown_reasons", "release_reconciliation", "dss_audit", "km_365", "sample_flow"]:
        print(f"\n## {k}\n", o[k].to_string(index=False))
    d = o["discordance_summary"]
    print(d[d.group.isin(["All", "TCGA", "CPTAC", "CPTAC LUAD", "CPTAC LUSC"])].to_string(index=False))

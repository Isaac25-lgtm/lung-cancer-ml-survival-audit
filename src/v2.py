"""Protocol v2 (docs/protocol_v2.md): MMIST-ccRCC replication and confirmatory tests.

    python -m src.v2 download     # ccRCC CSV (pinned) + GDC records for its patients
    python -m src.v2 ccrcc        # C1, C3, S1-S3
    python -m src.v2 lung         # C2, S4 (refit bootstrap on MMIST-LUNG)
    python -m src.v2 all

Outputs: outputs/tables/v2_*.csv and outputs/reports/v2_results.json (aggregates only).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import derive_outcomes as do
from src import metrics as M
from src.download_data import fetch, load_manifest, save, write_manifest
from src.io import load_gdc
from src.paths import RAW, REPORTS, TABLES
from src.stats_utils import wilson

CC_REPO_COMMIT = "c3f1a066a63f90e539b8af1b2a7b8d29012673e4"
CC_URL = ("https://raw.githubusercontent.com/Multi-Modal-IST/multi-modal-ist.github.io/"
          f"{CC_REPO_COMMIT}/datasets/ccRCC/Code/clinical%2Bgenomic_split.csv")
CC_FILE = RAW / "mmist_ccrcc" / "clinical+genomic_split.csv"
SEED_C2, SEED_C3 = 2026110502, 2026110503


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def download() -> None:
    from src.download_api import GDC_FIELDS, gdc_cases
    manifest = load_manifest()
    r = fetch(CC_URL)
    save("mmist_ccrcc", "clinical+genomic_split.csv", CC_URL, r.content, r.status_code, manifest, False,
         {"pinned_commit": CC_REPO_COMMIT})
    ids = sorted(pd.read_csv(CC_FILE)["case_id"].astype(str).str.strip().unique())
    hits = gdc_cases(ids)
    status = fetch("https://api.gdc.cancer.gov/status").json().get("data_release")
    payload = json.dumps({"data_release": status, "fields": GDC_FIELDS, "hits": hits}, indent=1, sort_keys=True).encode()
    save("gdc", "cases_ccrcc.json", "https://api.gdc.cancer.gov/cases (POST, MMIST-ccRCC ids)", payload, 200,
         manifest, False, {"gdc_data_release": status, "n_hits": len(hits)})
    lo_url = "https://cptac-pancancer-data.s3.us-west-2.amazonaws.com/data_freeze_v1.2_reorganized/CCRCC/CCRCC_survival.txt"
    r = fetch(lo_url)
    save("linkedomics_cptac", "CCRCC/CCRCC_survival.txt", lo_url, r.content, r.status_code, manifest, False)
    write_manifest(manifest)
    print(f"ccRCC: {len(ids)} ids, {len(hits)} GDC hits ({status})")


def load_ccrcc() -> tuple[pd.DataFrame, dict]:
    c = pd.read_csv(CC_FILE)
    c["case_id"] = c["case_id"].astype(str).str.strip()
    if not c.case_id.is_unique:
        raise ValueError("duplicate case_id in ccRCC file")
    g, release = load_gdc("cases_ccrcc.json")
    L = c.merge(g, left_on="case_id", right_on="id", how="left", validate="one_to_one", indicator="_gdc")
    L["cohort"] = np.where(L.case_id.str.startswith("TCGA"), "TCGA", "CPTAC")
    v = pd.to_numeric(L.vital_status_12, errors="coerce")
    # Pre-specified orientation rule: the class with prevalence near 0.88 is "survivor".
    surv_is_one = abs(v.mean() - 0.88) < abs((1 - v.mean()) - 0.88)
    L["y_released_death"] = (1 - v) if surv_is_one else v
    L["y_gdc_gt"] = do.from_gdc(L, "gt")
    L["y_gdc_ge"] = do.from_gdc(L, "ge")
    L["y_rule_tcga"] = 1 - do.released_rule_tcga(L.gdc_days_to_death)
    L["y_rule_notebook"] = 1 - do.released_rule_notebook(L.gdc_days_to_death)
    info = {"gdc_release": release, "n": len(L), "gdc_linked": int(L._gdc.eq("both").sum()),
            "released_mean_vital_status_12": float(v.mean()), "orientation_1_is_survivor": bool(surv_is_one),
            "projects": L.gdc_project.value_counts(dropna=False).to_dict(),
            "cohort_prefix_matches_project": int(((L.cohort == "TCGA") == L.gdc_project.fillna("").str.startswith("TCGA")).sum())}
    return L, info


# ---------------------------------------------------------------------------
# Models for ccRCC (generic tabular pipeline; outcome fields excluded)
# ---------------------------------------------------------------------------

EXCLUDE = {"case_id", "vital_status_12", "Split"}


def cc_features(L: pd.DataFrame) -> list[str]:
    raw = pd.read_csv(CC_FILE, nrows=1).columns
    return [c for c in raw if c not in EXCLUDE]


def _as_str(X):
    return pd.DataFrame(X).astype(object).where(pd.DataFrame(X).notna(), "missing").astype(str)


def cc_model(kind: str, L: pd.DataFrame, feats: list[str]) -> Pipeline:
    if kind == "stage_only":
        feats = ["ajcc_path_tumor_stage"]
        pre = ColumnTransformer([("c", Pipeline([("s", FunctionTransformer(_as_str)),
                                                 ("o", OneHotEncoder(handle_unknown="ignore"))]), feats)])
        return Pipeline([("pre", pre), ("clf", LogisticRegression(C=1.0, max_iter=5000))])
    num = [c for c in feats if pd.api.types.is_numeric_dtype(L[c])]
    cat = [c for c in feats if c not in num]
    tr = []
    if num:
        tr.append(("n", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                                  ("s", StandardScaler())]), num))
    if cat:
        tr.append(("c", Pipeline([("s", FunctionTransformer(_as_str)),
                                  ("o", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return Pipeline([("pre", ColumnTransformer(tr)), ("clf", LogisticRegression(C=1.0, max_iter=5000))])


def _oof(L, y, folds, make, train_mask=None):
    y = np.asarray(y, float)
    tm = np.ones(len(L), bool) if train_mask is None else train_mask
    s = np.full(len(L), np.nan)
    for tr, te in folds:
        tr = tr[tm[tr] & ~np.isnan(y[tr])]
        s[te] = make().fit(L.iloc[tr], y[tr].astype(int)).predict_proba(L.iloc[te])[:, 1]
    return s


def _foldavg(y, s, fold, mask):
    vals = [M.auroc(y[(fold == k) & mask], s[(fold == k) & mask]) for k in np.unique(fold)]
    return float(np.nanmean(vals))


def refit_bootstrap_psi_b(L, make_fn, folds, y_src, y_rel, reps, seed, feats=None):
    """psi_B = AUROC(trained on source, Y) - AUROC(trained on released, Y), fold-averaged on
    ascertainable test patients. Each replicate resamples patients within every fold's training
    and test sets and REFITS both models, so intervals include model-fitting uncertainty."""
    y_src, y_rel = np.asarray(y_src, float), np.asarray(y_rel, float)
    known = ~np.isnan(y_src)
    if feats is not None:
        L = L[feats]

    def one(rng=None):
        vals = []
        for tr, te in folds:
            if rng is not None:
                tr = rng.choice(tr, len(tr), replace=True)
                te = rng.choice(te, len(te), replace=True)
            trs = tr[known[tr]]
            a = make_fn().fit(L.iloc[trs], y_src[trs].astype(int))
            b = make_fn().fit(L.iloc[tr], y_rel[tr].astype(int))
            tek = te[known[te]]
            yy = y_src[tek]
            vals.append(M.auroc(yy, a.predict_proba(L.iloc[tek])[:, 1]) - M.auroc(yy, b.predict_proba(L.iloc[tek])[:, 1]))
        return float(np.nanmean(vals))

    point = one()
    rng = np.random.default_rng(seed)
    boot = np.array([one(rng) for _ in range(reps)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_one_sided = float((np.sum(boot <= 0) + 1) / (len(boot) + 1))
    return {"psi_B": point, "ci_lo": float(lo), "ci_hi": float(hi), "p_one_sided": p_one_sided, "reps": reps}


# ---------------------------------------------------------------------------
# ccRCC analyses
# ---------------------------------------------------------------------------

def ccrcc(reps: int = 200) -> dict:
    L, info = load_ccrcc()
    out = {"info": info}
    rows = []
    for cohort, s in [("All", L)] + list(L.groupby("cohort")):
        y, r = s.y_gdc_gt, s.y_released_death
        k = y.notna()
        disc = int((k & (y != r)).sum())
        lo, hi = wilson(disc, int(k.sum()))
        dead = k & s.gdc_days_to_death.notna() & s.gdc_vital_status.eq("Dead")
        agree_dead = int((dead & (y == r)).sum())
        dl, dh = wilson(agree_dead, int(dead.sum()))
        rows.append({"cohort": cohort, "n": len(s), "ascertainable": int(k.sum()), "unknown": int((~k).sum()),
                     "unknown_released_survivor": int((~k & (r == 0)).sum()),
                     "source_early_deaths": int((y == 1).sum()), "released_early_deaths": int((r == 1).sum()),
                     "inconsistent": disc, "inconsistent_pct": 100 * disc / max(k.sum(), 1),
                     "inconsistent_ci_lo": 100 * lo, "inconsistent_ci_hi": 100 * hi,
                     "recorded_deaths_ascertainable": int(dead.sum()), "recorded_deaths_label_agrees": agree_dead,
                     "recorded_deaths_agree_pct": 100 * agree_dead / max(dead.sum(), 1),
                     "recorded_deaths_agree_ci_lo": 100 * dl, "recorded_deaths_agree_ci_hi": 100 * dh,
                     "rule_tcga_reproduces": int((s.y_rule_tcga == r).sum()),
                     "rule_notebook_reproduces": int((s.y_rule_notebook == r).sum())})
    c1 = pd.DataFrame(rows)
    c1.to_csv(TABLES / "v2_ccrcc_c1_label_audit.csv", index=False)
    t, c = c1.set_index("cohort").loc["TCGA"], c1.set_index("cohort").loc["CPTAC"]
    if c.recorded_deaths_agree_pct < 10 and t.recorded_deaths_agree_pct > 90:
        decision = "inversion replicated"
    elif c.recorded_deaths_agree_pct > 90 and t.recorded_deaths_agree_pct > 90:
        decision = "no inversion"
    else:
        decision = "mixed"
    out["C1_decision"] = decision
    agree = (L.groupby(["cohort", L.y_released_death.map({1: "released early death", 0: "released survivor"}),
                        L.y_gdc_gt.map({1.0: "early death", 0.0: "alive >365d"}).fillna("unknown")])
             .size().rename("n").reset_index())
    agree.columns = ["cohort", "released", "source_state", "n"]
    agree.to_csv(TABLES / "v2_ccrcc_label_agreement.csv", index=False)

    n_disc = int(c1.set_index("cohort").at["All", "inconsistent"])
    out["C3_applicable"] = n_disc >= 20
    feats = cc_features(L)
    state = np.where(L.y_gdc_gt.isna(), 2, L.y_gdc_gt.fillna(0)).astype(int)
    strat = (L.cohort == "CPTAC").to_numpy().astype(int) * 3 + state
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED_C3).split(np.zeros(len(L)), strat))
    fold = np.empty(len(L), int)
    for i, (_, te) in enumerate(folds):
        fold[te] = i
    yS, yR = L.y_gdc_gt.to_numpy(float), L.y_released_death.to_numpy(float)
    known = ~np.isnan(yS)
    if out["C3_applicable"]:
        out["C3"] = refit_bootstrap_psi_b(L, lambda: cc_model("logistic", L, feats), folds, yS, yR, reps, SEED_C3, feats)
        # S1-S3 (secondary, conditional on inversion)
        sc = {k: _oof(L, yR, folds, lambda k=k: cc_model(k, L, feats)) for k in ["logistic", "stage_only"]}
        s_src = _oof(L, yS, folds, lambda: cc_model("logistic", L, feats), train_mask=known)
        s1 = []
        for k, s in sc.items():
            a, b = _foldavg(yS, s, fold, known), _foldavg(yR, s, fold, known)
            boot = M.paired_bootstrap(lambda i: _foldavg(yS[known][i], s[known][i], fold[known][i], np.ones(len(i), bool)),
                                      lambda i: _foldavg(yR[known][i], s[known][i], fold[known][i], np.ones(len(i), bool)),
                                      int(known.sum()), (fold[known] * 2 + (L.cohort[known] == "CPTAC")).to_numpy().astype(int),
                                      2000, SEED_C3)
            d = boot[0] - boot[1]
            s1.append({"model": k, "auroc_source": a, "auroc_released": b, "psi_A": a - b,
                       "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))})
        pd.DataFrame(s1).to_csv(TABLES / "v2_ccrcc_s1_fixed_score.csv", index=False)
        out["S1"] = s1
        dec = []
        coh = L.cohort.to_numpy()
        for k, s in sc.items():
            parts = {}
            for lab, y in [("released", yR), ("source", yS)]:
                yk, sk, ck = y[known], s[known], coh[known]
                tot = (yk == 1).sum() * (yk == 0).sum()
                for a_ in ["TCGA", "CPTAC"]:
                    for b_ in ["TCGA", "CPTAC"]:
                        p1, p0 = sk[(yk == 1) & (ck == a_)], sk[(yk == 0) & (ck == b_)]
                        w = len(p1) * len(p0) / tot
                        cc = M.auroc(np.r_[np.ones(len(p1)), np.zeros(len(p0))], np.r_[p1, p0]) if w > 0 else np.nan
                        parts[(lab, a_, b_)] = (w, cc)
                        dec.append({"model": k, "label_set": lab, "case_cohort": a_, "control_cohort": b_,
                                    "cases": len(p1), "controls": len(p0), "pair_share": w, "concordance": cc})
            comp = sum((parts[("source", a_, b_)][0] - parts[("released", a_, b_)][0]) *
                       np.nan_to_num((parts[("source", a_, b_)][1] + parts[("released", a_, b_)][1]) / 2)
                       for a_ in ["TCGA", "CPTAC"] for b_ in ["TCGA", "CPTAC"])
            conc = sum(np.nan_to_num(parts[("source", a_, b_)][1] - parts[("released", a_, b_)][1]) *
                       (parts[("source", a_, b_)][0] + parts[("released", a_, b_)][0]) / 2
                       for a_ in ["TCGA", "CPTAC"] for b_ in ["TCGA", "CPTAC"])
            out.setdefault("S2", {})[k] = {"composition": float(comp), "concordance": float(conc),
                                           "composition_share": float(comp / (comp + conc)) if comp + conc else None}
        pd.DataFrame(dec).to_csv(TABLES / "v2_ccrcc_s2_decomposition.csv", index=False)
        s3 = []
        for name, s in [("logistic, trained on released", sc["logistic"]), ("logistic, trained on source", s_src)]:
            for cohort in ["TCGA", "CPTAC"]:
                m = known & (L.cohort == cohort).to_numpy()
                s3.append({"scores": name, "cohort": cohort, "n": int(m.sum()), "mean_predicted_risk": float(s[m].mean()),
                           "released_event_rate": float(yR[m].mean()), "source_event_rate": float(yS[m].mean())})
        pd.DataFrame(s3).to_csv(TABLES / "v2_ccrcc_s3_cohort_risk.csv", index=False)
        out["S3"] = s3
    return out


# ---------------------------------------------------------------------------
# MMIST-LUNG confirmatory (C2) and model families (S4)
# ---------------------------------------------------------------------------

def lung(reps_c2: int = 200, reps_s4: int = 100) -> dict:
    from src.experiments import load
    from src.models import CLIN, make_model
    L, folds = load()
    yS, yR = L.y_gdc_gt.to_numpy(float), L.y_released_death.to_numpy(float)
    res = {}
    if reps_c2:
        res["C2"] = refit_bootstrap_psi_b(L, lambda: make_model("logistic", CLIN), folds, yS, yR, reps_c2, SEED_C2, CLIN)
        print("C2", res["C2"], flush=True)
    for kind in ["xgb", "mlp"]:
        res[f"S4_{kind}"] = refit_bootstrap_psi_b(L, lambda k=kind: make_model(k, CLIN), folds, yS, yR, reps_s4,
                                                  SEED_C2 + 1, CLIN)
        print("S4", kind, res[f"S4_{kind}"], flush=True)
    return res


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def ccrcc_exploratory(reps: int = 2000) -> pd.DataFrame:
    """EXPLORATORY (not in protocol v2). How much do the unascertainable ccRCC labels move the
    benchmark's own metrics? Models trained on released labels in 5-fold CV (as a benchmark would);
    the same out-of-fold scores are scored on all 618 patients vs the 580 with known status."""
    L, _ = load_ccrcc()
    feats = cc_features(L)
    yR, yS = L.y_released_death.to_numpy(float), L.y_gdc_gt.to_numpy(float)
    known = ~np.isnan(yS)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED_C3).split(np.zeros(len(L)), yR))
    fold = np.empty(len(L), int)
    thr = np.empty(len(L))
    for i, (tr, te) in enumerate(folds):
        fold[te] = i
        thr[te] = yR[tr].mean()
    rows = []
    strata_all = (fold * 2 + (L.cohort == "CPTAC").to_numpy()).astype(int)
    for kind in ["logistic", "stage_only"]:
        s = _oof(L, yR, folds, lambda k=kind: cc_model(k, L, feats))
        def auc(y, sc, f, t):
            return _foldavg(y, sc, f, np.ones(len(y), bool))

        def bacc(y, sc, f, t):
            return float(np.mean([M.balanced_accuracy(y[f == k], sc[f == k], float(np.mean(t[f == k])))
                                  for k in np.unique(f)]))

        for metric, fn in [("AUROC", auc), ("Balanced accuracy", bacc)]:
            a = fn(yR, s, fold, thr)
            b = fn(yR[known], s[known], fold[known], thr[known])
            c = fn(yS[known], s[known], fold[known], thr[known])
            rng = np.random.default_rng(SEED_C3)
            groups = [np.flatnonzero(strata_all == g) for g in np.unique(strata_all)]
            d1 = []
            for _ in range(reps):
                idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
                kk = known[idx]
                d1.append(fn(yR[idx], s[idx], fold[idx], thr[idx]) -
                          fn(yR[idx][kk], s[idx][kk], fold[idx][kk], thr[idx][kk]))
            rows.append({"model": kind, "metric": metric, "released_all_618": a, "released_known_580": b,
                         "source_known_580": c, "all_minus_known": a - b,
                         "ci_lo": float(np.percentile(d1, 2.5)), "ci_hi": float(np.percentile(d1, 97.5))})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "v2_ccrcc_exploratory_metrics.csv", index=False)
    return out


def ccrcc_shortcut() -> pd.DataFrame:
    """EXPLORATORY. Released death labels in MMIST-ccRCC by verification status, by split, and
    whether pathological stage is coded missing (-1)."""
    L, _ = load_ccrcc()
    rows = []
    for split, s in [("all", L)] + list(L.groupby("Split")):
        d = s.y_released_death == 1
        miss = s.ajcc_path_tumor_stage < 0
        rows.append({"split": split, "patients": len(s), "released_deaths": int(d.sum()),
                     "deaths_confirmed": int((d & (s.y_gdc_gt == 1)).sum()),
                     "deaths_unverifiable": int((d & s.y_gdc_gt.isna()).sum()),
                     "deaths_contradicted": int((d & (s.y_gdc_gt == 0)).sum()),
                     "stage_missing": int(miss.sum()), "stage_missing_labelled_death": int((miss & d).sum()),
                     "deaths_with_stage_missing": int((d & miss).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "v2_ccrcc_shortcut.csv", index=False)
    return out


def main(argv):
    REPORTS.mkdir(parents=True, exist_ok=True)
    rp = REPORTS / "v2_results.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    what = argv[0] if argv else "all"
    if what in ("download", "all"):
        download()
    if what in ("ccrcc", "all"):
        res["ccrcc"] = ccrcc()
    if what in ("ccrcc", "all"):
        res["ccrcc_exploratory"] = ccrcc_exploratory().to_dict("records")
        res["ccrcc_shortcut"] = ccrcc_shortcut().to_dict("records")
    if what in ("lung", "all"):
        res["lung"] = lung()
    p = {}
    if "lung" in res:
        p["C2"] = res["lung"]["C2"]["p_one_sided"]
    if res.get("ccrcc", {}).get("C3_applicable"):
        p["C3"] = res["ccrcc"]["C3"]["p_one_sided"]
    if p:
        res["holm_adjusted_p"] = holm(p)
    rp.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main(sys.argv[1:])

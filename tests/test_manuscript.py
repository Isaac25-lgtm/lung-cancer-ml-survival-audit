"""Key manuscript numbers must match the generated result tables and protocol v2 results."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "outputs" / "reports"
MS = (ROOT / "docs" / "manuscript.md").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(not (TABLES / "v2_ccrcc_c1_label_audit.csv").exists()
                                or not (REPORTS / "v2_lung_results.json").exists(),
                                reason="protocol v2 outputs not generated")


def test_abstract_within_250_words():
    ab = MS[MS.index("**Objective.**"):MS.index("**Keywords:**")]
    assert len(re.findall(r"\w+", ab)) <= 250


def test_lung_audit_numbers():
    d = pd.read_csv(TABLES / "discordance_summary.csv")
    c = d[(d.source == "gdc_gt") & (d.group == "CPTAC")].iloc[0]
    a = d[(d.source == "gdc_gt") & (d.group == "All")].iloc[0]
    assert f"{int(c.discordant)} of {int(c.known)} ascertainable labels ({100 * c.discordant / c.known:.1f}%)" in MS
    assert f"{int(a.unknown)} patients with unknown status" in MS


def test_ccrcc_numbers():
    c1 = pd.read_csv(TABLES / "v2_ccrcc_c1_label_audit.csv").set_index("cohort")
    assert f"{int(c1.at['CPTAC', 'recorded_deaths_label_agrees'])} of {int(c1.at['CPTAC', 'recorded_deaths_ascertainable'])} CPTAC" in MS
    assert f"{int(c1.at['TCGA', 'recorded_deaths_label_agrees'])} of {int(c1.at['TCGA', 'recorded_deaths_ascertainable'])} TCGA" in MS
    assert f"Only {int(c1.at['All', 'inconsistent'])} of {int(c1.at['All', 'ascertainable'])} ascertainable labels" in MS
    e = pd.read_csv(TABLES / "v2_ccrcc_exploratory_metrics.csv")
    r = e[(e.model == "logistic") & (e.metric == "AUROC")].iloc[0]
    assert f"{r.released_known_580:.3f} to {r.released_all_618:.3f}" in MS
    assert f"{r.all_minus_known:.3f}; 95% CI {r.ci_lo:.3f}–{r.ci_hi:.3f}" in MS
    s = pd.read_csv(TABLES / "v2_ccrcc_shortcut.csv").set_index("split")
    assert int(s.at["all", "stage_missing_labelled_death"]) == int(s.at["all", "stage_missing"]) == 18


def test_confirmatory_numbers():
    r = json.loads((REPORTS / "v2_lung_results.json").read_text())
    c2 = r["C2"]
    assert f"ψ_B {c2['psi_B']:+.3f} (95% CI {c2['ci_lo']:+.3f} to {c2['ci_hi']:+.3f}" in MS
    assert f"one-sided p = {c2['p_one_sided']:.3f}" in MS
    for k in ["S4_xgb", "S4_mlp"]:
        x = r[k]
        assert f"ψ_B {x['psi_B']:+.3f} (95% CI {x['ci_lo']:+.3f}".replace("-", "−") in MS


def test_every_author_and_affiliation():
    for name, sup in [("Omoding Isaac", "¹"), ("Echodu Moses", "²"), ("Bob Tumushime", "³"),
                      ("Noah Okiror", "⁴"), ("Daniel Baguma", "⁵")]:
        assert f"{name}{sup}" in MS

"""The README's headline numbers must match the generated result tables, and every
image and local link it uses must exist."""
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
README = (ROOT / "README.md").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(not (TABLES / "exp_b_benchmark_folds.csv").exists(),
                                reason="analysis outputs not generated")


def _ci(lo, hi):
    return f"{lo:.3f} to {hi:.3f}"


def _minus(x, places=3):
    return f"{x:+.{places}f}".replace("-", "−")


def test_label_audit_numbers():
    d = pd.read_csv(TABLES / "discordance_summary.csv")
    d = d[d.source == "gdc_gt"].set_index("group")
    a = d.loc["All"]
    assert f"**{int(a.discordant)} of {int(a.known):,} checkable labels ({100 * a.discordant / a.known:.1f}%)" in README
    assert f"**{int(a.unknown)} patients ({100 * a.unknown / a.n:.1f}%)" in README
    c = d.loc["CPTAC"]
    assert f"**All {int(c.released_early_deaths)} patients labelled as dying within a year" in README
    assert f"**All {int(c.source_death_labelled_survivor)} patients who died within a year" in README
    assert c.released_ppv_for_early_death == 0 and c.released_sensitivity_for_early_death == 0
    assert d.loc["TCGA"].discordant == 0


def test_rule_reproduction_numbers():
    r = pd.read_csv(TABLES / "rule_reproduction.csv")
    best = r.groupby("cohort").agree.max()
    assert f"{best['TCGA']:,} of 1,026" in README
    assert f"{best['CPTAC']} of 339" in README


def test_kaplan_meier_numbers():
    k = pd.read_csv(TABLES / "km_365.csv").set_index("group").loc["All"]
    assert f"{100 * k.km_survival_365:.1f}% (95% CI {100 * k.km_ci_lo:.1f}–{100 * k.km_ci_hi:.1f})" in README


def test_primary_contrast_numbers():
    s = pd.read_csv(TABLES / "exp_sensitivity.csv")
    r = s[s.sensitivity.str.startswith("primary")].iloc[0]
    assert f"{_minus(r.delta)} (95% CI {_minus(r.ci_lo)} to {_minus(r.ci_hi)})" in README
    alt = s[s.sensitivity.str.startswith("folds: released")].iloc[0]
    assert _minus(alt.delta) in README


def test_training_and_interaction_numbers():
    b = pd.read_csv(TABLES / "exp_b_benchmark_folds.csv")
    r = b[(b.model == "logistic") & b.evaluation.str.startswith("source")].iloc[0]
    assert f"{r.delta:+.3f} (95% CI {_ci(r.ci_lo, r.ci_hi)})" in README
    m = pd.read_csv(TABLES / "exp_a_model_ranking.csv")
    i = m[m.evaluation.str.startswith("interaction") & (m.contrast == "stage_only - logistic")].iloc[0]
    assert f"{i.delta:+.3f} (95% CI {_ci(i.ci_lo, i.ci_hi)})" in README


def test_negative_control_numbers():
    a = pd.read_csv(TABLES / "i3lung_label_audit.csv")
    for lab, n in [("OS_6", "1,972"), ("OS_24", "1,699")]:
        v = a[(a.label == lab) & a.check.str.contains("1 = alive")].value.iloc[0]
        k, tot = v.split("/")
        assert k == tot and f"{int(k):,}" == n and f"{n} of {n}" in README


def test_local_links_and_images_exist():
    for target in re.findall(r"\]\((?!https?://)([^)#\s]+)", README):
        assert (ROOT / target).exists(), f"broken README link: {target}"


def test_images_have_alt_text():
    for alt, _ in re.findall(r"!\[([^\]]*)\]\((?!https?://)([^)]+)\)", README):
        assert len(alt) > 30, f"image needs descriptive alt text: {alt!r}"

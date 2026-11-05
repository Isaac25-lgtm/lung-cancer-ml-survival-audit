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
    for target in re.findall(r"\]\((?!https?://|mailto:)([^)#\s]+)", README):
        assert (ROOT / target).exists(), f"broken README link: {target}"


def test_images_have_alt_text():
    for alt, _ in re.findall(r"!\[([^\]]*)\]\((?!https?://)([^)]+)\)", README):
        assert len(alt) > 30, f"image needs descriptive alt text: {alt!r}"


def test_decomposition_numbers():
    d = pd.read_csv(TABLES / "exp_a_auroc_decomposition.csv")
    for model, gap_text, share_text in [("logistic", "\u22120.034", "78%"), ("stage_only", "+0.024", "94%")]:
        x = d[d.model == model]
        a, b = x[x.label_set == "released"].reset_index(drop=True), x[x.label_set == "source"].reset_index(drop=True)
        gap = (b.contribution - a.contribution).sum()
        comp = ((b.pair_share - a.pair_share) * (b.concordance + a.concordance) / 2).sum()
        conc = ((b.concordance - a.concordance) * (b.pair_share + a.pair_share) / 2).sum()
        assert abs(comp + conc - gap) < 1e-12
        assert gap_text == _minus(gap) and share_text in README
        share = comp / gap if model == "logistic" else conc / gap
        assert f"{share:.0%}" == share_text
    x = d[(d.model == "logistic") & (d.case_cohort == "CPTAC") & (d.control_cohort == "TCGA")].set_index("label_set")
    assert f"**{x.at['released', 'pair_share']:.1%}** | **{x.at['released', 'concordance']:.3f}**" in README
    assert f"**{x.at['source', 'pair_share']:.1%}** | **{x.at['source', 'concordance']:.3f}**" in README


def test_calibration_and_experiment_numbers():
    a = pd.read_csv(TABLES / "exp_a_metrics.csv")
    a = a[(a.cohort == "All") & a.evaluation.str.startswith("source")].set_index("model")
    assert f"calibration slope {a.at['logistic', 'cal_slope']:.2f}" in README
    assert f"scaled Brier score {_minus(a.at['logistic', 'scaled_brier_pooled'])}" in README
    assert f"slope {a.at['stage_only', 'cal_slope']:.2f}" in README
    assert f"scaled Brier score {_minus(a.at['stage_only', 'scaled_brier_pooled'])}" in README
    t = pd.read_csv(TABLES / "exp_a2_transport.csv")
    r = t[(t.model == "logistic") & t.test.str.contains("GDC")].iloc[0]
    assert f"{r.auroc_delta:+.3f} ({r.auroc_delta_ci_lo:+.3f} to {r.auroc_delta_ci_hi:+.3f})" in README
    c = pd.read_csv(TABLES / "exp_c_censoring.csv")
    c = c[c.scores == "logistic trained on source-derived labels"]
    b = c[c.method.str.startswith("sharp")].iloc[0]
    assert f"{b.auroc_lo:.3f}\u2013{b.auroc_hi:.3f}" in README
    assert f"{c[c.method.str.startswith('complete')].auroc.iloc[0]:.3f}" in README
    d = pd.read_csv(TABLES / "exp_d_leakage.csv")
    d = d[d.model == "logistic"]
    assert f"{d[d.features.str.contains('dss')].auroc_released_all_foldavg.iloc[0]:.3f}" in README
    assert f"{d[d.features.str.contains('last_fup')].auroc_released_all_foldavg.iloc[0]:.3f}" in README


def test_every_reference_is_cited_and_defined():
    cited = set(re.findall(r"\[\[(\d+)\]\]\(#ref-\1\)", README))
    defined = set(re.findall(r'<a id="ref-(\d+)"></a>', README))
    assert cited == defined, f"cited but undefined: {cited - defined}; defined but uncited: {defined - cited}"
    assert defined == {str(i) for i in range(1, len(defined) + 1)}


def test_author_details_present():
    assert "Omoding Isaac" in README and "omodingisaac111@gmail.com" in README
    assert "Bob Tumushime" in README and "Uganda Bureau of Statistics" in README
    order = ["Omoding Isaac", "Echodu Moses", "Bob Tumushime", "Noah Okiror", "Daniel Baguma"]
    assert [README.index(a) for a in order] == sorted(README.index(a) for a in order)
    assert "Isaac Omoding" not in README

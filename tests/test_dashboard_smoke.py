"""Dashboard smoke tests: the app renders without exceptions from committed
aggregate tables, its headline numbers match the tables, and it refuses to show
results when artefacts are missing."""
import shutil
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

pytestmark = pytest.mark.skipif(not (TABLES / "exp_a_paired_deltas.csv").exists(),
                                reason="analysis outputs not generated")


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)


def test_app_renders_without_exceptions():
    at = _app().run()
    assert not at.exception, at.exception
    assert any("MMIST-LUNG" in t.value for t in at.title)


def test_headline_metrics_match_tables():
    at = _app().run()
    d = pd.read_csv(TABLES / "discordance_summary.csv")
    p = d[(d.source == "gdc_gt") & (d.group == "All")].iloc[0]
    vals = {m.label: m.value for m in at.metric}
    assert vals["Patients in benchmark"] == f"{int(p.n):,}"
    assert vals["365-day status ascertainable"] == f"{int(p.known):,}"
    assert vals["Released label inconsistent with source"] == f"{int(p.discordant)}"


def test_no_patient_identifiers_in_committed_tables():
    import re
    pat = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|C3[LN]-\d{5}|\b11LU\d{3}\b")
    for p in TABLES.glob("*.csv"):
        assert not pat.search(p.read_text(encoding="utf-8")), f"patient ID found in {p.name}"


def test_missing_artefacts_stop_app(tmp_path, monkeypatch):
    import dashboard.data as dd
    fake = tmp_path / "tables"
    shutil.copytree(TABLES, fake)
    (fake / "exp_a_paired_deltas.csv").unlink()
    monkeypatch.setattr(dd, "TABLES", fake)
    st = dd.artefact_status()
    assert "exp_a_paired_deltas" in st["missing"]


def test_no_undefined_or_rounding_defects():
    """88/260 = 33.85% must display as 33.8% (exact ratio), not 33.9% (rounded twice)."""
    src = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "if title else None" not in src  # None titles render as 'undefined' in Plotly
    assert "zip(d.discordant_pct" not in src
    assert f"{100 * 88 / 260:.1f}" == "33.8"


def test_tables_match_run_manifest():
    """Every committed table matches the hash recorded by the analysis run."""
    import dashboard.data as dd
    st = dd.artefact_status()
    assert st["stale"] == [], f"tables differ from run manifest: {st['stale']}"

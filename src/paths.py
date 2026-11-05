"""Central path definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
REPORTS = OUTPUTS / "reports"
CONFIGS = ROOT / "configs"

MMIST = RAW / "mmist"
CPTAC = RAW / "linkedomics_cptac"
GDC = RAW / "gdc"
CBIO = RAW / "cbioportal"

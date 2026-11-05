"""Second case study: label-provenance audit of the public I3LUNG bundle.

Purpose: apply the same audit procedure (does a released fixed-horizon label
agree with the dataset's own time/event fields?) to an independent public
resource. This is NOT external validation of any MMIST-LUNG model: different
population (advanced NSCLC on immunotherapy), different endpoint (6-month OS),
different centres.

Source: Zenodo record 17535424 (CC BY-NC 4.0). Only I3LUNG_DATA.zip (~26 MB) is
fetched; the 2 GB results archive is not needed.

Usage: python -m src.i3lung
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd

from src.download_data import fetch, load_manifest, save, write_manifest
from src.paths import RAW, TABLES

RECORD = "https://zenodo.org/api/records/17535424"
DEST = RAW / "i3lung"


def download(force: bool = False) -> None:
    manifest = load_manifest()
    meta = fetch(RECORD)
    save("i3lung", "zenodo_record.json", RECORD, meta.content, meta.status_code, manifest, force)
    rec = meta.json()
    f = next(x for x in rec["files"] if x["key"] == "I3LUNG_DATA.zip")
    zpath = DEST / "I3LUNG_DATA.zip"
    if not zpath.exists() or force:
        url = f["links"]["self"]
        r = fetch(url, timeout=600)
        save("i3lung", "I3LUNG_DATA.zip", url, r.content, r.status_code, manifest, force,
             {"license": rec["metadata"].get("license", {}).get("id"), "zenodo_checksum": f.get("checksum")})
    write_manifest(manifest)
    with zipfile.ZipFile(zpath) as z:
        out = (DEST / "extracted").resolve()
        for name in z.namelist():
            target = (out / name).resolve()
            if not str(target).startswith(str(out)):  # zip-slip guard
                raise ValueError(f"unsafe path in zip: {name}")
            if name.endswith("/"):
                continue
            if not name.lower().endswith((".csv", ".json", ".txt", ".md")):
                continue  # only tabular/metadata files are extracted
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))


def _find(name: str):
    hits = list((DEST / "extracted").rglob(name))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def describe() -> dict:
    o = pd.read_csv(_find("outcomes.csv"))
    return {"outcomes_columns": list(o.columns), "n": len(o),
            "files": sorted(str(p.relative_to(DEST / "extracted")) for p in (DEST / "extracted").rglob("*"))[:50]}


def audit() -> pd.DataFrame:
    """Compare released OS_6 / OS_24 against 'OS MONTHS' and 'DEATH EVENT' in outcomes.csv."""
    o = pd.read_csv(_find("outcomes.csv"))
    norm = {c.lower().replace(" ", "_"): c for c in o.columns}
    tcol, ecol = norm.get("os_months"), norm.get("death_event")
    rows = [{"label": "-", "check": "rows / unique Subject",
             "value": f"{len(o)} / {o['Subject'].nunique()}"}]
    if tcol is None or ecol is None:
        rows.append({"label": "-", "check": "time/event fields", "value": f"NOT AVAILABLE: {list(o.columns)}"})
        return pd.DataFrame(rows)
    t = pd.to_numeric(o[tcol], errors="coerce").to_numpy(float)
    e = pd.to_numeric(o[ecol], errors="coerce").to_numpy(float)
    rows.append({"label": "-", "check": f"{ecol} values", "value": json.dumps(
        pd.Series(e).value_counts(dropna=False).rename(str).to_dict())})
    for lab, h in [("OS_6", 6.0), ("OS_24", 24.0)]:
        if lab not in o.columns:
            continue
        y = pd.to_numeric(o[lab], errors="coerce").to_numpy(float)
        derived = np.where((e == 1) & (t <= h), 1.0, np.where(t > h, 0.0, np.nan))
        derived[np.isnan(t) | np.isnan(e)] = np.nan
        rows.append({"label": lab, "check": "label values", "value": json.dumps(
            pd.Series(y).value_counts(dropna=False).rename(str).to_dict())})
        for orient, d in [("1 = death within horizon", derived), ("1 = alive at horizon", 1 - derived)]:
            k = ~np.isnan(d) & ~np.isnan(y)
            rows.append({"label": lab, "check": f"agreement with OS time/event if {orient}",
                         "value": f"{int((d[k] == y[k]).sum())}/{int(k.sum())}"})
        rows.append({"label": lab, "check": "label present but censored before horizon (not ascertainable)",
                     "value": int((~np.isnan(y) & np.isnan(derived) & ~np.isnan(t)).sum())})
        rows.append({"label": lab, "check": "label missing", "value": int(np.isnan(y).sum())})
        rows.append({"label": lab, "check": "label missing although ascertainable",
                     "value": int((np.isnan(y) & ~np.isnan(derived)).sum())})
        bd = (~np.isnan(y)) & (t == h)
        rows.append({"label": lab, "check": "exactly on horizon (boundary cases)", "value": int(bd.sum())})
    if "SET" in o.columns:
        rows.append({"label": "-", "check": "SET counts", "value": json.dumps(o["SET"].value_counts().to_dict())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    download()
    print(json.dumps(describe(), indent=1))
    a = audit()
    TABLES.mkdir(parents=True, exist_ok=True)
    a.to_csv(TABLES / "i3lung_label_audit.csv", index=False)
    print(a.to_string(index=False))

"""One-command reproduction.

    python -m src.run_analysis            # full run (downloads if needed)
    python -m src.run_analysis --quick    # 5 CV repeats instead of 50 for Experiment B

Steps: download -> handover reproduction gate -> linkage -> label audit ->
experiments -> I3LUNG audit -> literature register -> figures -> run manifest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess

import pandas as pd

from src.paths import REPORTS, ROOT, TABLES


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args(argv)

    if not a.skip_download:
        from src import download_data
        download_data.main([])
    from src import (audit_labels, experiments, fold_provenance, i3lung, link_sources, literature,
                     make_figures, make_story_figures, reproduce_handover)
    rep = reproduce_handover.run()
    n_ok = int(rep.match.sum())
    print(f"[gate] handover reproduction {n_ok}/{len(rep)}")
    if n_ok != len(rep):
        print("[gate] WARNING: inputs differ from the feasibility handover; see handover_reproduction.csv")
    link_sources.run()
    audit_labels.run()
    fold_provenance.run()
    res = experiments.run(5 if a.quick else 50)
    if not a.skip_download:
        i3lung.download()
    try:
        i3lung.audit().to_csv(TABLES / "i3lung_label_audit.csv", index=False)
    except FileNotFoundError:
        print("[i3lung] bundle not available; skipped")
    literature.run()
    make_figures.run()
    make_story_figures.run()

    res["run_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    res["git_commit"] = _git_commit()
    res["handover_reproduced"] = f"{n_ok}/{len(rep)}"
    res["gdc_release"] = pd.read_csv(TABLES / "linkage_summary.csv", index_col=0).loc["gdc_release", "value"]
    (REPORTS / "results.json").write_text(json.dumps(res, indent=2, default=float))
    # Line endings are normalised so hashes match across operating systems and git checkouts.
    hashes = {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for p in sorted(TABLES.glob("*.csv"))}
    (REPORTS / "table_hashes.json").write_text(json.dumps(hashes, indent=1))
    print(json.dumps({k: res[k] for k in ["primary_contrast", "handover_reproduced", "gdc_release"]},
                     indent=2, default=float))


if __name__ == "__main__":
    main()

"""API-based retrieval of upstream clinical sources (open access, no credentials).

* GDC (NCI Genomic Data Commons) /cases API: current clinical records for every
  MMIST-LUNG submitter_id (TCGA-LUAD, TCGA-LUSC, CPTAC-3). Saved as raw JSON
  together with the GDC /status payload (data-release identifier).
* TCGA Clinical Data Resource (TCGA-CDR; Liu et al., Cell 2018), Supplemental
  Table S1, from the GDC PanCanAtlas publication page.
* cBioPortal REST API: patient-level clinical attributes for the TCGA LUAD and
  LUSC PanCancer Atlas studies (OS in months), used only as a secondary
  cross-check because it is derived from the TCGA-CDR.

Responses are not deterministic over time (GDC releases change), so every pull
is hashed and its release identifier recorded in the manifest.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import requests

from src.download_data import USER_AGENT, fetch, save
from src.paths import MMIST

GDC = "https://api.gdc.cancer.gov"
CDR_UUID = "1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
CBIO = "https://www.cbioportal.org/api"
CBIO_STUDIES = ["luad_tcga_pan_can_atlas_2018", "lusc_tcga_pan_can_atlas_2018"]

GDC_FIELDS = [
    "submitter_id", "case_id", "project.project_id", "index_date", "disease_type", "primary_site",
    "demographic.vital_status", "demographic.days_to_death", "demographic.cause_of_death",
    "demographic.days_to_birth", "demographic.sex_at_birth", "demographic.gender",
    "demographic.updated_datetime",
    "diagnoses.diagnosis_id", "diagnoses.days_to_last_follow_up", "diagnoses.primary_diagnosis",
    "diagnoses.classification_of_tumor", "diagnoses.diagnosis_is_primary_disease",
    "diagnoses.days_to_diagnosis", "diagnoses.age_at_diagnosis", "diagnoses.ajcc_pathologic_stage",
    "diagnoses.tissue_or_organ_of_origin", "diagnoses.days_to_last_known_disease_status",
    "diagnoses.updated_datetime",
    "follow_ups.follow_up_id", "follow_ups.days_to_follow_up", "follow_ups.timepoint_category",
    "follow_ups.updated_datetime",
]


def gdc_cases(ids: list[str]) -> list[dict]:
    hits: list[dict] = []
    for i in range(0, len(ids), 200):
        chunk = ids[i:i + 200]
        body = {
            "filters": {"op": "in", "content": {"field": "submitter_id", "value": chunk}},
            "fields": ",".join(GDC_FIELDS),
            "format": "JSON",
            "size": 1000,
        }
        hits.extend(_post_with_retries(f"{GDC}/cases", body)["data"]["hits"])
    return hits


def _post_with_retries(url: str, body: dict, retries: int = 5, timeout: int = 60) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.post(url, json=body, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:  # network hiccup
            last = e
        time.sleep(5 * 2 ** i)
    raise RuntimeError(f"Failed to POST {url} after {retries} attempts: {last}")


def run(manifest: dict, force: bool = False) -> None:
    print("API pulls:")
    status = fetch(f"{GDC}/status")
    release = status.json().get("data_release")
    save("gdc", "status.json", f"{GDC}/status", status.content, status.status_code, manifest, force)
    print(f"  GDC {release}")

    ids = pd.read_csv(MMIST / "features" / "demographics.csv")["cases.submitter_id"].astype(str).tolist()
    hits = gdc_cases(sorted(set(ids)))
    payload = json.dumps({"data_release": release, "fields": GDC_FIELDS, "hits": hits},
                         indent=1, sort_keys=True).encode()
    save("gdc", "cases_mmist.json", f"{GDC}/cases (POST, submitter_id in MMIST list)", payload, 200,
         manifest, force, {"gdc_data_release": release, "n_hits": len(hits)})
    print(f"  GDC cases: {len(hits)} hits for {len(set(ids))} MMIST ids")

    url = f"{GDC}/data/{CDR_UUID}"
    r = fetch(url, timeout=300)
    save("gdc", "TCGA-CDR-SupplementalTableS1.xlsx", url, r.content, r.status_code, manifest, force,
         {"publication": "Liu et al. Cell 2018 doi:10.1016/j.cell.2018.02.052"})
    print("  TCGA-CDR table")

    for study in CBIO_STUDIES:
        url = f"{CBIO}/studies/{study}/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY&pageSize=100000"
        r = fetch(url, timeout=300)
        save("cbioportal", f"{study}_patient_clinical.json", url, r.content, r.status_code, manifest, force)
        print(f"  cBioPortal {study}: {len(r.json())} attribute rows")

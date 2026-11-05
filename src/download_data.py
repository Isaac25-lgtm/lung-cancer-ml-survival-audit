"""Download pinned public source files and record provenance.

Usage:  python -m src.download_data [--force]

Writes raw bytes to data/raw/<source>/<path> (gitignored) and a provenance
manifest (URL, HTTP status, bytes, SHA-256, UTC timestamp) to
data/raw/MANIFEST.json. A copy of the manifest without local paths is written to
outputs/tables/source_manifest.csv so that provenance can be committed even
though raw files cannot.

Existing files are never silently overwritten: if a file exists and --force is
not given, its hash is re-verified against the manifest and the download is
skipped. If --force is given and the new bytes differ, the old file is kept as
<name>.prev-<sha12> so that version changes are visible rather than lost.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.paths import RAW, ROOT, TABLES

USER_AGENT = "lung-cancer-ml-survival-audit/0.1 (research reproducibility; python-requests)"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, retries: int = 4, timeout: int = 120, **kw) -> requests.Response:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT}, **kw)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:  # network hiccup
            last = e
        time.sleep(2 ** i)
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def _safe_target(source: str, rel: str) -> Path:
    target = (RAW / source / rel).resolve()
    if not str(target).startswith(str((RAW / source).resolve())):
        raise ValueError(f"Unsafe path {rel}")
    return target


def save(source: str, rel: str, url: str, content: bytes, status: int, manifest: dict,
         force: bool, extra: dict | None = None) -> None:
    target = _safe_target(source, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    new_hash = sha256_bytes(content)
    if target.exists():
        old_hash = sha256_file(target)
        if old_hash != new_hash:
            backup = target.with_name(f"{target.name}.prev-{old_hash[:12]}")
            target.rename(backup)
            print(f"  ! content changed for {rel}; previous kept as {backup.name}")
    target.write_bytes(content)
    key = f"{source}/{rel}"
    manifest[key] = {
        "source": source,
        "path": rel,
        "url": url,
        "http_status": status,
        "bytes": len(content),
        "sha256": new_hash,
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }


def load_manifest() -> dict:
    mp = RAW / "MANIFEST.json"
    return json.loads(mp.read_text()) if mp.exists() else {}


def write_manifest(manifest: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in m.items() if k != "local"} for m in manifest.values()]
    pd.DataFrame(rows).sort_values(["source", "path"]).to_csv(TABLES / "source_manifest.csv", index=False)


def download_static(cfg: dict, manifest: dict, force: bool) -> None:
    for source, spec in cfg.items():
        base = spec["base_url"]
        commit = spec.get("commit")
        for rel in spec["files"]:
            key = f"{source}/{rel}"
            target = _safe_target(source, rel)
            if target.exists() and not force and key in manifest:
                if sha256_file(target) == manifest[key]["sha256"]:
                    print(f"  = {key} (verified, skipped)")
                    continue
                raise RuntimeError(f"Hash mismatch for existing {key}; rerun with --force to refetch")
            url = base.format(commit=commit, path=requests.utils.quote(rel))
            print(f"  get {key}")
            r = fetch(url)
            extra = {"pinned_commit": commit} if commit else {}
            if "last-modified" in r.headers:
                extra["last_modified_header"] = r.headers["last-modified"]
            if "etag" in r.headers:
                extra["etag_header"] = r.headers["etag"].strip('"')
            save(source, rel, url, r.content, r.status_code, manifest, force, extra)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="refetch even if verified local copy exists")
    ap.add_argument("--skip-gdc", action="store_true", help="skip GDC/cBioPortal API pulls")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((ROOT / "configs" / "sources.yaml").read_text())
    manifest = load_manifest()
    print("Static files:")
    download_static(cfg, manifest, args.force)
    write_manifest(manifest)
    if not args.skip_gdc:
        from src import download_api
        download_api.run(manifest, force=args.force)
        write_manifest(manifest)
    print(f"Manifest: {len(manifest)} files -> {RAW / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()

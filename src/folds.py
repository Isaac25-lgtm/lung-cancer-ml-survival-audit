"""Safe loader for the MMIST-LUNG `folds.npy` object array.

The file is a NumPy object array (pickled). Unpickling untrusted data can
execute arbitrary code, so we never call np.load(allow_pickle=True). Instead we
parse the .npy header ourselves and unpickle the payload with an allowlist that
only permits NumPy array reconstruction. Any other global raises.
"""
from __future__ import annotations

import ast
import io
import pickle

import numpy as np
import pandas as pd

from src.paths import MMIST

_ALLOWED = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"Disallowed global in folds.npy: {module}.{name}")
        if name == "_reconstruct":
            from numpy._core.multiarray import _reconstruct  # numpy>=2
            return _reconstruct
        return getattr(np, name)


def load_folds(path=None) -> list[tuple[np.ndarray, np.ndarray]]:
    path = path or (MMIST / "folds.npy")
    b = open(path, "rb").read()
    if b[:6] != b"\x93NUMPY":
        raise ValueError("not an .npy file")
    major = b[6]
    hl_bytes = 2 if major == 1 else 4
    hl = int.from_bytes(b[8:8 + hl_bytes], "little")
    header = ast.literal_eval(b[8 + hl_bytes:8 + hl_bytes + hl].decode("latin1"))
    body = b[8 + hl_bytes + hl:]
    if header["descr"] != "|O":
        raise ValueError(f"unexpected dtype {header['descr']}")
    arr = _RestrictedUnpickler(io.BytesIO(body)).load()
    arr = np.asarray(arr, dtype=object).reshape(header["shape"])
    folds = []
    for row in arr:
        tr, te = (np.asarray(x).astype(np.int64) for x in row)
        folds.append((tr, te))
    return folds


def benchmark_order() -> list[str]:
    """Patient order in which folds.npy was generated: sorted TCGA, then CPTAC in diagnosis.csv order.

    Under this order StratifiedKFold(5, shuffle=True, random_state=42) on the released
    vital_status_12 reproduces folds.npy exactly (1365/1365; see src/fold_provenance.py).
    The released PatientDataset.py instead sorts all patients by ID, so the released code and
    fold file are not mutually consistent; that alignment is run as a sensitivity analysis.
    """
    ids = pd.read_csv(MMIST / "features" / "demographics.csv")["cases.submitter_id"].astype(str)
    diag = pd.read_csv(MMIST / "features" / "diagnosis.csv")["cases.submitter_id"].astype(str)
    tcga = sorted(i for i in ids if i.startswith("TCGA"))
    cptac = [i for i in diag if not i.startswith("TCGA")]
    order = tcga + cptac
    if sorted(order) != sorted(ids):
        raise ValueError("benchmark order does not cover the demographics IDs")
    return order

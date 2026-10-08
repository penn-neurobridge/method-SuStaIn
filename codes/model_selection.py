"""How many subtypes? Lowest cross-validation information criterion (CVIC) wins."""
from __future__ import annotations

import numpy as np
import pandas as pd


def choose_n_subtypes(CVIC: np.ndarray) -> int:
    return int(np.argmin(CVIC)) + 1


def model_selection_table(CVIC: np.ndarray | None, loglike: np.ndarray | None, n_subtypes_max: int, chosen: int | None) -> pd.DataFrame:
    rows = []
    for s in range(1, n_subtypes_max + 1):
        row = {"n_subtypes": s}
        if CVIC is not None:
            row["cvic"] = float(CVIC[s - 1])
            for f in range(loglike.shape[0]):
                row[f"loglike_fold_{f}"] = float(loglike[f, s - 1])
            row["loglike_mean"] = float(loglike[:, s - 1].mean())
        row["chosen"] = bool(chosen == s)
        rows.append(row)
    return pd.DataFrame(rows)

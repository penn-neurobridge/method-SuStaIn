"""Read, validate and prepare the input table for SuStaIn's z-score model.

Input CSV contract (see README):
    subject_id            unique id (text or number)
    group                 "control" or "patient"
    biomarker_0 ...       one numeric column per biomarker, arbitrary units, no missing values
    true_subtype, true_stage   OPTIONAL ground truth (only exists for simulated data)
"""
from pathlib import Path

import numpy as np
import pandas as pd

BIOMARKER_PREFIX = "biomarker_"
GROUPS = ("control", "patient")


def load_input(path: Path) -> pd.DataFrame:
    """Read the CSV and fail early, with a readable message, if it does not follow the contract."""
    df = pd.read_csv(path)
    missing = [c for c in ("subject_id", "group") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}")
    bad = sorted(set(df["group"]) - set(GROUPS))
    if bad:
        raise ValueError(f"{path}: 'group' must be one of {GROUPS}, found {bad}")
    if df["subject_id"].duplicated().any():
        raise ValueError(f"{path}: subject_id values must be unique")
    cols = biomarker_columns(df)
    if df[cols].isna().any().any():
        raise ValueError(f"{path}: biomarker columns contain missing values (this model needs complete data)")
    if (df["group"] == "control").sum() < 20:
        raise ValueError(f"{path}: need at least 20 control rows to estimate the control mean/SD")
    return df


def biomarker_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(BIOMARKER_PREFIX)]
    if not cols:
        raise ValueError(f"no columns starting with '{BIOMARKER_PREFIX}' found")
    return sorted(cols, key=lambda c: int(c[len(BIOMARKER_PREFIX):]))


def zscore_against_controls(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    """z-score every biomarker against the controls (mean 0, SD 1 in controls), then flip biomarkers that
    DEcrease with disease (e.g. brain volume) so that every biomarker increases with disease, as SuStaIn requires.
    This is the same recipe as the official tutorial and the paper's w-score without the covariate adjustment."""
    X = df[cols].to_numpy(float)
    is_ctrl = (df["group"] == "control").to_numpy()
    mu, sd = X[is_ctrl].mean(0), X[is_ctrl].std(0)
    if (sd == 0).any():
        raise ValueError("a biomarker has zero variance in controls")
    Z = (X - mu) / sd
    flipped = Z[~is_ctrl].mean(0) < Z[is_ctrl].mean(0)
    Z[:, flipped] *= -1
    return Z, {"control_mean": mu.tolist(), "control_sd": sd.tolist(), "flipped": [c for c, f in zip(cols, flipped) if f]}


def build_z_levels(n_biomarkers: int, z_levels: list[float], z_max: float) -> tuple[np.ndarray, np.ndarray]:
    """SuStaIn's 'events' are biomarker x severity-level pairs. Same levels for every biomarker here."""
    if sorted(z_levels) != list(z_levels) or min(z_levels) <= 0:
        raise ValueError("z_levels must be positive and increasing")
    if z_max <= max(z_levels):
        raise ValueError("z_max must exceed the largest z level")
    return np.array([z_levels] * n_biomarkers, float), np.array([z_max] * n_biomarkers, float)


def parse_levels(text: str) -> list[float]:
    return [float(t) for t in text.split(",") if t.strip()]

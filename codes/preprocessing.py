"""
preprocessing.py — Load and prepare data for SuStaIn.

What this file does, in plain English:
  1. Reads the CSV file.
  2. Checks that all the expected columns are present.
  3. Separates controls from patients.
  4. Returns the biomarker values as a numeric matrix that SuStaIn can use.

SuStaIn expects z-scores — numbers that say how far a biomarker value is
from the average healthy person (in units of standard deviations).
In our synthetic data the z-scores are pre-computed, so we just read them.
For real data you would compute z-scores here using the control group mean
and standard deviation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


# The biomarker columns we expect in the CSV (in the order SuStaIn will use).
BIOMARKER_COLUMNS = [
    "hippocampus_zscore",
    "entorhinal_zscore",
    "amygdala_zscore",
    "fusiform_zscore",
    "precuneus_zscore",
    "posterior_cingulate_zscore",
]

# Human-readable labels for plots (same order as above).
BIOMARKER_LABELS = [
    "Hippocampus",
    "Entorhinal Ctx",
    "Amygdala",
    "Fusiform Gyrus",
    "Precuneus",
    "Post. Cingulate",
]


def load_and_preprocess(input_path: Path) -> dict:
    """
    Load the CSV and return everything SuStaIn needs.

    Parameters
    ----------
    input_path : Path
        Path to the input CSV file.

    Returns
    -------
    dict with keys:
        "data"              : np.ndarray, shape (N_subjects, N_biomarkers)
                              Z-score matrix — one row per subject, one column
                              per biomarker.
        "biomarker_labels"  : list of str, names for each biomarker column.
        "subject_ids"       : list of str, subject identifiers.
        "df"                : pd.DataFrame, the full raw dataframe (useful for
                              merging results later).
    """
    print(f"[preprocessing] Loading data from: {input_path}")

    # --- 1. Read the CSV ---
    df = pd.read_csv(input_path)
    print(f"[preprocessing] Found {len(df)} subjects, {len(df.columns)} columns.")

    # --- 2. Check required columns exist ---
    missing = [c for c in BIOMARKER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following expected columns are missing from your CSV:\n  {missing}\n"
            f"Columns present: {list(df.columns)}"
        )

    # --- 3. Extract the z-score matrix as a plain numpy array ---
    #   Shape: (number of subjects) × (number of biomarkers)
    data = df[BIOMARKER_COLUMNS].to_numpy(dtype=float)

    # --- 4. Basic sanity checks ---
    n_subjects, n_biomarkers = data.shape
    n_missing = np.isnan(data).sum()
    if n_missing > 0:
        print(
            f"[preprocessing] WARNING: {n_missing} missing values found. "
            "They will be replaced with 0 (population mean z-score)."
        )
        data = np.nan_to_num(data, nan=0.0)

    print(
        f"[preprocessing] Data matrix: {n_subjects} subjects × {n_biomarkers} biomarkers."
    )
    print(
        f"[preprocessing] Z-score range: "
        f"min={data.min():.2f}, max={data.max():.2f}, mean={data.mean():.2f}"
    )

    return {
        "data": data,
        "biomarker_labels": BIOMARKER_LABELS,
        "subject_ids": df["subject_id"].tolist() if "subject_id" in df.columns else list(range(n_subjects)),
        "df": df,
    }

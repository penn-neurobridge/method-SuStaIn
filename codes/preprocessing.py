"""
preprocessing.py — Load and prepare data for SuStaIn.

What this file does, in plain English:
  1. Reads the CSV file.
  2. Checks that all the expected columns are present.
  3. Returns the biomarker values as a numeric matrix that SuStaIn can use.

SuStaIn expects z-scores — Numbers that say how far a biomarker value is
from the average healthy person (in units of standard deviations).
In our synthetic data the z-scores are pre-computed, so we just read them.
For real data you would compute z-scores here using the control group mean
and standard deviation.

generate_synthetic_data()
--------------------------
Mimics the approach used in the official pySuStaIn examples
(see: https://github.com/ucl-pond/pySuStaIn).

Two subtypes are defined by their *biomarker ordering* — which region
deteriorates first. Subjects are assigned randomly to subtypes and
disease stages, then their z-scores are computed from the ordering,
with Gaussian noise added on top.

Change the ``noise_std`` argument to see how SuStaIn copes with noisier data!
Low noise (0.1) → clean separable subtypes.
High noise (1.5) → subtypes blur together.
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


def generate_synthetic_data(
    n_subjects: int = 300,
    n_subtypes: int = 2,
    noise_std: float = 0.5,
    seed: int = 42,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic SuStaIn data with a known ground-truth subtype structure.

    This follows the same approach as the official pySuStaIn simulated-data
    examples (ZscoreSuStaIn_example.py in the ucl-pond/pySuStaIn repo).

    Parameters
    ----------
    n_subjects : int
        Number of simulated subjects (patients).
    n_subtypes : int
        Number of disease subtypes to embed. The first two subtypes always use
        opposite biomarker orderings; extra subtypes use shuffled orderings.
    noise_std : float
        Standard deviation of Gaussian noise added to each z-score.
        **This is the "change one thing" parameter.**
        Try: 0.1 (very clean), 0.5 (default), 1.0 (moderate), 1.5 (very noisy).
    seed : int
        Random seed for reproducibility.
    output_path : Path or None
        If given, saves the dataframe as a CSV at this path.

    Returns
    -------
    pd.DataFrame  with columns:
        subject_id, group, true_subtype, true_stage,
        hippocampus_zscore, entorhinal_zscore, amygdala_zscore,
        fusiform_zscore, precuneus_zscore, posterior_cingulate_zscore
    """
    rng = np.random.default_rng(seed)
    n_biomarkers = len(BIOMARKER_COLUMNS)

    # ── Define subtype orderings ──────────────────────────────────────────────
    # Each ordering says "biomarker at rank 0 is the FIRST to become abnormal".
    # Subtype 1: anterior temporal → posterior cortex (hippocampal-first pattern)
    # Subtype 2: posterior cortex → anterior temporal (cortical-first pattern)
    anterior_to_posterior = list(range(n_biomarkers))          # [0,1,2,3,4,5]
    posterior_to_anterior = list(reversed(range(n_biomarkers))) # [5,4,3,2,1,0]

    subtype_orderings = [anterior_to_posterior, posterior_to_anterior]
    # If more subtypes are requested, add random orderings
    for _ in range(n_subtypes - 2):
        subtype_orderings.append(rng.permutation(n_biomarkers).tolist())
    subtype_orderings = subtype_orderings[:n_subtypes]

    # ── Z-score thresholds (matches what SuStaIn uses) ────────────────────────
    # Each biomarker can be normal (0), mildly abnormal (1), moderately (2),
    # or severely abnormal (3) standard deviations above the healthy mean.
    z_thresholds = [1.0, 2.0, 3.0]  # 3 stages per biomarker → 18 stages total
    n_stages_per_biomarker = len(z_thresholds)
    n_stages_total = n_biomarkers * n_stages_per_biomarker  # e.g. 18

    # ── Assign subjects to subtypes and stages ────────────────────────────────
    rows = []
    for i in range(n_subjects):
        subtype = int(rng.integers(0, n_subtypes))
        ordering = subtype_orderings[subtype]

        # Stage 0 = healthy; stage k means k biomarker-threshold transitions have occurred
        stage = int(rng.integers(0, n_stages_total + 1))

        # Compute expected z-scores from the subtype ordering and stage
        z_expected = np.zeros(n_biomarkers)
        stages_remaining = stage
        for biomarker_rank, biomarker_idx in enumerate(ordering):
            for z_val in z_thresholds:
                if stages_remaining > 0:
                    z_expected[biomarker_idx] = z_val
                    stages_remaining -= 1
                else:
                    break
            if stages_remaining == 0:
                break

        # Add Gaussian noise — **this is what --noise controls**
        z_noisy = z_expected + rng.normal(0.0, noise_std, n_biomarkers)

        row: dict = {
            "subject_id":  f"sub_{i:04d}",
            "group":       "patient",
            "true_subtype": subtype + 1,   # 1-indexed
            "true_stage":   stage,
        }
        for col, z in zip(BIOMARKER_COLUMNS, z_noisy):
            row[col] = round(float(z), 4)

        rows.append(row)

    df = pd.DataFrame(rows)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[preprocessing] Synthetic data saved → {output_path}")

    return df


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

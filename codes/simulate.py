"""Simulated cohorts with known subtypes and stages, for demos and tests.

`simulate_zscore_cohort` reproduces the generative model behind pySuStaIn's
`generate_data_Zscore_sustain` (sim/simfuncs.py) and then dresses the z-scores
up as raw-looking measurements with an age effect and one decreasing
biomarker, so the demo exercises every preprocessing step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The two mirror-image orderings used in the journal-club notebook:
#   subtype 1: A first, then B, C, D     subtype 2: D first, then C, B, A
# Expressed as (biomarker index, threshold level) pairs; levels are 1-based.
DEMO_SEQUENCES = [
    [(0, 1), (0, 2), (1, 1), (0, 3), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (2, 3), (3, 2), (3, 3)],
    [(3, 1), (3, 2), (2, 1), (3, 3), (2, 2), (1, 1), (2, 3), (1, 2), (0, 1), (1, 3), (0, 2), (0, 3)],
]


def event_index(biomarker: int, level: int, n_biomarkers: int) -> int:
    """pySuStaIn event numbering: column-major over (level, biomarker)."""
    return (level - 1) * n_biomarkers + biomarker


def stage_profiles(sequence: np.ndarray, Z_vals: np.ndarray, Z_max: np.ndarray) -> np.ndarray:
    """Expected z-score of every biomarker at every stage, shape (B, N_events + 2).

    Column k is stage k. Faithful port of the interpolation in pySuStaIn's
    simulator (including its extra final column), so simulated data sits
    exactly where the fitted model expects it.
    """
    Z_vals = np.asarray(Z_vals, dtype=float)
    B = Z_vals.shape[0]
    stage_z = Z_vals.T.flatten()
    bio_idx = np.tile(np.arange(B), Z_vals.shape[1])
    keep = stage_z > 0
    stage_z, bio_idx = stage_z[keep], bio_idx[keep]
    N = len(stage_z)
    S_inv = np.zeros(N, dtype=int)
    S_inv[np.asarray(sequence, dtype=int)] = np.arange(N)
    prof = np.zeros((B, N + 2))
    for b in range(B):
        loc = np.concatenate([[0], S_inv[bio_idx == b], [N]])
        val = np.concatenate([[0.0], stage_z[bio_idx == b], [float(Z_max[b])]])
        for j in range(len(loc) - 1):
            if j == 0:
                idx = np.arange(loc[j], loc[j + 1] + 2)
            else:
                idx = np.arange(loc[j] + 1, loc[j + 1] + 2)
            prof[b, idx] = np.linspace(val[j], val[j + 1], len(idx))
    return prof


@dataclass
class Cohort:
    data: pd.DataFrame   # what a user would upload
    truth: pd.DataFrame  # subject_id, true_subtype (1-based), true_stage
    sequences: np.ndarray  # (S, N_events) ground-truth event orderings
    Z_vals: np.ndarray
    Z_max: np.ndarray


def simulate_zscore_cohort(
    n_subjects: int = 400,
    n_controls: int = 80,
    biomarkers: tuple[str, ...] = ("A", "B", "C", "D"),
    z_vals: tuple[float, ...] = (1.0, 2.0, 3.0),
    z_max: float = 5.0,
    fractions: tuple[float, ...] = (0.6, 0.4),
    noise_sd: float = 1.0,
    seed: int = 0,
) -> Cohort:
    rng = np.random.default_rng(seed)
    B = len(biomarkers)
    Z_vals = np.array([list(z_vals)] * B, dtype=float)
    Z_max = np.array([z_max] * B, dtype=float)
    seqs = np.array([[event_index(b, lvl, B) for b, lvl in s] for s in DEMO_SEQUENCES], dtype=float)
    N = seqs.shape[1]
    S = len(fractions)
    profiles = np.stack([stage_profiles(seqs[s], Z_vals, Z_max) for s in range(S)])  # (S, B, N+2)

    n_pat = n_subjects - n_controls
    subtype = np.concatenate([np.zeros(n_controls, dtype=int), rng.choice(S, n_pat, p=fractions)])
    stage = np.concatenate([np.zeros(n_controls, dtype=int), rng.integers(1, N + 1, n_pat)])
    z_clean = profiles[subtype, :, stage]  # (M, B)
    z = z_clean + rng.standard_normal((n_subjects, B)) * noise_sd

    # Dress up as raw measurements: age effect, scale/offset, and make the last biomarker decreasing.
    age = rng.normal(60, 8, n_subjects).round(1)
    sign = np.ones(B)
    sign[-1] = -1
    raw = 100 + sign * 15 * z + 0.5 * (age - 60)[:, None] + rng.normal(0, 1, (n_subjects, B))

    ids = [f"sub-{i + 1:03d}" for i in range(n_subjects)]
    data = pd.DataFrame(raw.round(2), columns=list(biomarkers))
    data.insert(0, "age", age)
    data.insert(0, "group", np.where(stage == 0, "control", "patient"))
    data.insert(0, "subject_id", ids)
    truth = pd.DataFrame({"subject_id": ids, "true_subtype": np.where(stage == 0, 0, subtype + 1), "true_stage": stage})
    return Cohort(data, truth, seqs, Z_vals, Z_max)


def simulate_mixture_cohort(
    n_subjects: int = 300,
    n_biomarkers: int = 5,
    fractions: tuple[float, ...] = (0.5, 0.5),
    seed: int = 1,
) -> Cohort:
    """No group column: values are 'normal' N(0, .25) until the biomarker's event, then 'abnormal' N(1.5, .3)."""
    rng = np.random.default_rng(seed)
    B, S = n_biomarkers, len(fractions)
    seqs = np.stack([rng.permutation(B) for _ in range(S)]).astype(float)
    subtype = rng.choice(S, n_subjects, p=fractions)
    stage = rng.integers(0, B + 1, n_subjects)
    abnormal = np.zeros((n_subjects, B), dtype=bool)
    for i in range(n_subjects):
        abnormal[i, seqs[subtype[i], : stage[i]].astype(int)] = True
    sd_case = rng.uniform(0.25, 0.5, B)
    values = np.where(abnormal, rng.normal(1.5, sd_case, (n_subjects, B)), rng.normal(0, 0.25, (n_subjects, B)))
    ids = [f"sub-{i + 1:03d}" for i in range(n_subjects)]
    labels = [f"M{j + 1}" for j in range(B)]
    data = pd.DataFrame(values.round(3), columns=labels)
    data.insert(0, "subject_id", ids)
    truth = pd.DataFrame({"subject_id": ids, "true_subtype": np.where(stage == 0, 0, subtype + 1), "true_stage": stage})
    return Cohort(data, truth, seqs, np.ones((B, 1)), np.ones(B))


def match_subtype_labels(estimated: np.ndarray, truth: np.ndarray, n_subtypes: int) -> tuple[float, np.ndarray]:
    """Subtype labels are arbitrary; find the permutation that best matches the truth.

    Returns (accuracy, permutation) where permutation[k] is the truth label for estimated label k.
    """
    from itertools import permutations

    best, best_perm = -1.0, None
    for perm in permutations(range(n_subtypes)):
        mapped = np.asarray(perm)[estimated]
        acc = float(np.mean(mapped == truth))
        if acc > best:
            best, best_perm = acc, np.asarray(perm)
    return best, best_perm

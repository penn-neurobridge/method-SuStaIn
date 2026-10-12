"""
plotting.py — All visualizations for the SuStaIn results.

Three plots are produced:

1. positional_variance_diagram.png
   The signature SuStaIn output. Shows, for each subtype, the probability
   that each biomarker occupies each position in the disease progression
   sequence. Darker = more probable.

2. staging_histogram.png
   How many subjects are assigned to each disease stage (per subtype).
   Gives a sense of the stage distribution in your sample.

3. subtype_assignments.png
   A simple bar chart showing how many subjects were assigned to each subtype.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ── Colour palette ──────────────────────────────────────────────────────────
SUBTYPE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # blue, orange, green, red


def plot_positional_variance_diagram(
    samples_sequence: np.ndarray,
    biomarker_labels: list[str],
    output_path: Path,
    subtype_idx: int = 0,
) -> None:
    """
    Draw the positional variance diagram for one subtype.

    Parameters
    ----------
    samples_sequence : np.ndarray, shape (N_samples, N_biomarkers)
        Each row is one MCMC sample of the ordering of biomarkers.
        Values are biomarker indices (0-based).
    biomarker_labels : list of str
        Names for each biomarker.
    output_path : Path
        Where to save the PNG.
    subtype_idx : int
        Which subtype this is (used only for the plot title).
    """
    n_samples, n_biomarkers = samples_sequence.shape

    # Build the positional variance matrix:
    # pvd[i, j] = fraction of MCMC samples where biomarker i is at position j
    pvd = np.zeros((n_biomarkers, n_biomarkers))
    for sample in samples_sequence:
        for position, biomarker in enumerate(sample):
            pvd[biomarker, position] += 1
    pvd /= n_samples  # convert counts → probabilities

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pvd, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    # Axis labels
    ax.set_xticks(range(n_biomarkers))
    ax.set_xticklabels([f"Stage {i+1}" for i in range(n_biomarkers)], fontsize=9)
    ax.set_yticks(range(n_biomarkers))
    ax.set_yticklabels(biomarker_labels, fontsize=9)

    ax.set_xlabel("Position in sequence →  (earlier = affected first)", fontsize=10)
    ax.set_ylabel("Biomarker", fontsize=10)
    ax.set_title(
        f"Subtype {subtype_idx + 1} — Positional Variance Diagram\n"
        "(colour = probability that biomarker occupies that position)",
        fontsize=11,
    )

    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plotting] Saved: {output_path}")


def plot_staging_histogram(
    subject_stages: np.ndarray,
    subject_subtypes: np.ndarray,
    n_biomarkers: int,
    output_path: Path,
) -> None:
    """
    Histogram of disease stage assignments, coloured by subtype.

    Parameters
    ----------
    subject_stages : np.ndarray, shape (N_subjects,)
        Most-likely stage for each subject (0 = no abnormality, n_biomarkers = all abnormal).
    subject_subtypes : np.ndarray, shape (N_subjects,)
        Most-likely subtype assignment for each subject (0-indexed).
    n_biomarkers : int
        Number of biomarkers = number of stages.
    output_path : Path
        Where to save the PNG.
    """
    n_subtypes = int(subject_subtypes.max()) + 1
    bins = np.arange(0, n_biomarkers + 2) - 0.5  # bin edges

    fig, ax = plt.subplots(figsize=(8, 4))
    for s in range(n_subtypes):
        mask = subject_subtypes == s
        ax.hist(
            subject_stages[mask],
            bins=bins,
            alpha=0.7,
            label=f"Subtype {s + 1}",
            color=SUBTYPE_COLORS[s % len(SUBTYPE_COLORS)],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Disease stage  (0 = healthy-like, higher = more advanced)", fontsize=10)
    ax.set_ylabel("Number of subjects", fontsize=10)
    ax.set_title("Disease Stage Distribution by Subtype", fontsize=11)
    ax.legend()
    ax.set_xticks(range(n_biomarkers + 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plotting] Saved: {output_path}")


def plot_subtype_assignments(
    subject_subtypes: np.ndarray,
    output_path: Path,
) -> None:
    """
    Bar chart of how many subjects were assigned to each subtype.

    Parameters
    ----------
    subject_subtypes : np.ndarray, shape (N_subjects,)
        Most-likely subtype for each subject (0-indexed).
    output_path : Path
        Where to save the PNG.
    """
    n_subtypes = int(subject_subtypes.max()) + 1
    counts = [(subject_subtypes == s).sum() for s in range(n_subtypes)]
    labels = [f"Subtype {s + 1}" for s in range(n_subtypes)]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        labels,
        counts,
        color=SUBTYPE_COLORS[:n_subtypes],
        edgecolor="white",
        linewidth=1,
    )
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Number of subjects", fontsize=10)
    ax.set_title("Subtype Assignment Summary", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plotting] Saved: {output_path}")

"""
run_analysis.py — Main entry point for the SuStaIn analysis pipeline.

This is the script you run from the terminal. It ties together
preprocessing, model fitting, and plotting in one command.

Quick start
-----------
    # See all options:
    uv run python codes/run_analysis.py --help

    # Run with defaults:
    uv run python codes/run_analysis.py \\
        --input input/synthetic_data.csv \\
        --output output/

What SuStaIn does, in plain English
------------------------------------
SuStaIn (Subtype and Staging Inference) looks at a group of patients and
tries to answer two questions at once:

  1. Are there different *subtypes* of the disease — groups of patients
     where the same biomarkers go wrong, but in a different order?

  2. For each patient, how far along are they in the disease progression
     (their *stage*)?

It does this purely from cross-sectional data — a single snapshot of each
patient — without needing to follow anyone over time.

Output files
------------
  output/results.csv               – Per-subject subtype and stage assignments
  output/figures/pvd_subtype_*.png – Positional variance diagram per subtype
  output/figures/staging_histogram.png
  output/figures/subtype_assignments.png
""

from __future__ import annotations

import sys
import json
from pathlib import Path

import typer
import numpy as np
import pandas as pd

# Add the codes/ directory to the Python path so we can import our own modules
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import load_and_preprocess, generate_synthetic_data
from plotting import (
    plot_positional_variance_diagram,
    plot_staging_histogram,
    plot_subtype_assignments,
)

app = typer.Typer(
    help="SuStaIn analysis pipeline — discovers disease subtypes and stages from biomarker data.",
    add_completion=False,
)


@app.command()
def main(
    input: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Path to the input CSV file (e.g. input/synthetic_data.csv).",
        show_default=False,
    ),
    output: Path = typer.Option(
        Path("output/"),
        "--output", "-o",
        help="Directory where results and figures will be saved.",
    ),
    n_subtypes: int = typer.Option(
        2,
        "--n-subtypes", "-s",
        help="Number of disease subtypes to fit. Start with 2 and compare models.",
        min=1,
        max=6,
    ),
    n_iterations: int = typer.Option(
        50_000,
        "--n-iterations",
        help="Number of MCMC iterations. More = better estimates, but slower. "
             "Use 10000 for quick tests, 100000+ for publication.",
    ),
    n_startpoints: int = typer.Option(
        10,
        "--n-startpoints",
        help="Number of random starting points for the optimiser. "
             "More = less likely to get stuck in a bad solution.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed so results are reproducible.",
    ),
    noise: float = typer.Option(
        0.5,
        "--noise",
        help=(
            "Gaussian noise added to z-scores when generating synthetic data. "
            "Try 0.1 for clean subtypes, 1.5 for very noisy data. "
            "Use with --generate-data to re-create input/synthetic_data.csv."
        ),
    ),
    generate_data: bool = typer.Option(
        False,
        "--generate-data",
        help="Re-generate synthetic_data.csv before running (uses --noise and --seed). "
             "Useful for exploring: change --noise, run again, compare the plots!",
    ),
) -> None:
    """
    Run the full SuStaIn analysis: load data → fit model → save results → make plots.
    """
    # ── 0. Setup ──────────────────────────────────────────────────────────────
    typer.echo("=" * 60)
    typer.echo("  SuStaIn Analysis Pipeline")
    typer.echo("=" * 60)

    output.mkdir(parents=True, exist_ok=True)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    # ── 0b. Optionally (re)generate synthetic data ────────────────────────────
    if generate_data or not input.exists():
        if not input.exists():
            typer.echo(f"\n  Input file not found at {input}. Generating synthetic data...")
        else:
            typer.echo(f"\n  --generate-data flag set. Creating fresh data with noise_std={noise}...")
        generate_synthetic_data(
            n_subjects=300,
            n_subtypes=n_subtypes,
            noise_std=noise,
            seed=seed,
            output_path=input,
        )

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    typer.echo("\n[Step 1/4] Loading and preprocessing data...")
    preprocessed = load_and_preprocess(input)
    data = preprocessed["data"]           # shape: (N_subjects, N_biomarkers)
    biomarker_labels = preprocessed["biomarker_labels"]
    subject_ids = preprocessed["subject_ids"]
    df_raw = preprocessed["df"]

    n_subjects, n_biomarkers = data.shape
    typer.echo(f"           {n_subjects} subjects, {n_biomarkers} biomarkers ready.")

    # ── 2. Fit the SuStaIn model ──────────────────────────────────────────────
    typer.echo(f"\n[Step 2/4] Fitting SuStaIn with {n_subtypes} subtype(s)...")
    typer.echo(f"           MCMC iterations: {n_iterations:,}  |  Start points: {n_startpoints}")
    typer.echo("           (This may take a few minutes — more iterations = slower but better)")

    try:
        import pySuStaIn
        from pySuStaIn.ZscoreSustain import ZscoreSustain

        # Z-score thresholds: at what z-score levels do we call a biomarker "abnormal"?
        # Here we use 1, 2, and 3 standard deviations above the healthy mean.
        z_vals = np.array([[1, 2, 3]] * n_biomarkers)   # shape: (N_biomarkers, 3)
        z_max  = np.array([5] * n_biomarkers)            # shape: (N_biomarkers,)

        sustain_model = ZscoreSustain(
            data,
            z_vals,
            z_max,
            biomarker_labels,
            N_startpoints=n_startpoints,
            N_S_max=n_subtypes,
            N_iterations_MCMC=n_iterations,
            output_folder=str(output),
            dataset_name="synthetic",
            use_parallel_startpoints=False,
            seed=seed,
        )
        samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = (
            sustain_model.run_sustain_algorithm()
        )

        # Most-likely assignments
        subject_subtypes = ml_subtype.astype(int).flatten()
        subject_stages   = ml_stage.astype(int).flatten()

    except ImportError:
        typer.echo(
            "\n  pySuStaIn is not installed. Running a simplified demo instead.\n"
            "  Install it with:  uv add pySuStaIn\n"
        )
        samples_sequence, subject_subtypes, subject_stages = _demo_fit(
            data, n_subtypes, n_biomarkers, n_subjects, seed
        )

    # ── 3. Save results ───────────────────────────────────────────────────────
    typer.echo("\n[Step 3/4] Saving results...")

    results_df = pd.DataFrame({
        "subject_id":      subject_ids,
        "assigned_subtype": subject_subtypes + 1,  # 1-indexed for readability
        "assigned_stage":   subject_stages,
    })

    # Merge with original data so the CSV is self-contained
    if "true_subtype" in df_raw.columns:
        results_df = results_df.merge(
            df_raw[["subject_id", "group", "true_subtype", "true_stage"]],
            on="subject_id", how="left"
        )

    results_path = output / "results.csv"
    results_df.to_csv(results_path, index=False)
    typer.echo(f"           Saved: {results_path}")

    # Save model metadata as JSON (useful for reproducibility)
    meta = {
        "n_subjects": n_subjects,
        "n_biomarkers": n_biomarkers,
        "n_subtypes": n_subtypes,
        "n_iterations_mcmc": n_iterations,
        "n_startpoints": n_startpoints,
        "seed": seed,
        "biomarker_labels": biomarker_labels,
    }
    meta_path = output / "model_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    typer.echo(f"           Saved: {meta_path}")

    # ── 4. Make plots ─────────────────────────────────────────────────────────
    typer.echo("\n[Step 4/4] Generating figures...")

    for s in range(n_subtypes):
        pvd_path = figures_dir / f"pvd_subtype_{s+1}.png"
        # samples_sequence shape: (N_subtypes, N_MCMC_samples, N_biomarkers)
        seq_s = samples_sequence[s] if samples_sequence.ndim == 3 else samples_sequence
        plot_positional_variance_diagram(seq_s, biomarker_labels, pvd_path, subtype_idx=s)

    plot_staging_histogram(
        subject_stages, subject_subtypes, n_biomarkers,
        figures_dir / "staging_histogram.png",
    )
    plot_subtype_assignments(subject_subtypes, figures_dir / "subtype_assignments.png")

    # ── Done ──────────────────────────────────────────────────────────────────
    typer.echo("\n" + "=" * 60)
    typer.echo("  ✓  Analysis complete!")
    typer.echo(f"     Results → {results_path}")
    typer.echo(f"     Figures → {figures_dir}/")
    typer.echo("=" * 60)

    # Print a quick summary table
    typer.echo("\nSubtype summary:")
    summary = results_df.groupby("assigned_subtype").size().rename("n_subjects")
    typer.echo(summary.to_string())


# ── Demo fallback (used when pySuStaIn is not installed) ─────────────────────

def _demo_fit(
    data: np.ndarray,
    n_subtypes: int,
    n_biomarkers: int,
    n_subjects: int,
    seed: int,
) -> tuple:
    """
    Simplified placeholder that mimics SuStaIn output structure so the rest of
    the pipeline (saving, plotting) can still run for demonstration purposes.

    This is NOT the real SuStaIn algorithm. Install pySuStaIn for real results.
    """
    rng = np.random.default_rng(seed)

    # Random sequence samples (N_subtypes, N_samples, N_biomarkers)
    n_samples = 500
    samples_sequence = np.zeros((n_subtypes, n_samples, n_biomarkers), dtype=int)
    for s in range(n_subtypes):
        for i in range(n_samples):
            samples_sequence[s, i] = rng.permutation(n_biomarkers)

    # Assign each subject to a random subtype and stage
    subject_subtypes = rng.integers(0, n_subtypes, size=n_subjects)
    subject_stages   = rng.integers(0, n_biomarkers + 1, size=n_subjects)

    return samples_sequence, subject_subtypes, subject_stages


if __name__ == "__main__":
    app()

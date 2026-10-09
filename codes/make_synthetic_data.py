"""Generate the synthetic input table (and a small JSON file with the hidden ground truth).

    uv run python codes/make_synthetic_data.py --output input/synthetic_data.csv

Same generator as the official pySuStaIn tutorial ("SuStaIn tutorial using simulated data"). `--noise 1` reproduces the
tutorial. The z-score model assumes controls have SD 1, so more noise is simulated by shrinking the SIGNAL instead:
every severity level (and z_max) is divided by `--noise`. noise=2 puts events at z = 0.5, 1, 1.5 instead of 1, 2, 3:
same biology, half the signal-to-noise.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer


def _find_sim_dir() -> str:
    """pySuStaIn ships its simulator as a top-level package `sim` whose __init__ uses an old-style import, so
    `import sim` fails. The official tutorial puts the folder on sys.path; we locate it wherever pip/uv installed it."""
    cands = [os.environ.get("PYSUSTAIN_SIM_DIR")]
    spec = importlib.util.find_spec("sim")  # find_spec does not execute sim/__init__.py
    if spec and spec.submodule_search_locations:
        cands += list(spec.submodule_search_locations)
    for c in cands:
        if c and os.path.exists(os.path.join(c, "simfuncs.py")):
            return c
    raise ImportError("Could not find pySuStaIn's sim/simfuncs.py. Run `uv sync` or set PYSUSTAIN_SIM_DIR.")


sys.path.insert(0, _find_sim_dir())
from simfuncs import generate_data_Zscore_sustain, generate_random_Zscore_sustain_model  # noqa: E402


def simulate(n_biomarkers=5, n_subjects=500, n_controls=100, n_subtypes=2, noise=1.0, seed=0,
             z_levels=(1, 2, 3), z_max=5.0) -> tuple[pd.DataFrame, dict]:
    state = np.random.get_state()
    np.random.seed(seed)
    Z_vals = np.array([[lv / noise for lv in z_levels]] * n_biomarkers)
    Z_max = np.array([z_max / noise] * n_biomarkers)

    frac = [1 + 0.5 * i for i in range(n_subtypes)]          # tutorial: 2/3 vs 1/3 for two subtypes
    frac = [f / sum(frac) for f in frac][::-1]
    gt_sequence = generate_random_Zscore_sustain_model(Z_vals, n_subtypes)
    n_k = int(np.sum(Z_vals > 0) + 1)
    subtypes = np.random.choice(range(n_subtypes), n_subjects, replace=True, p=frac)
    stages = np.concatenate((np.zeros((n_controls, 1)),
                             np.ceil(np.random.rand(n_subjects - n_controls, 1) * n_k)), axis=0)
    data, _, _ = generate_data_Zscore_sustain(subtypes, stages, gt_sequence, Z_vals, Z_max)
    np.random.set_state(state)

    is_ctrl = np.arange(n_subjects) < n_controls
    df = pd.DataFrame({"subject_id": [f"S{i:04d}" for i in range(n_subjects)],
                       "group": np.where(is_ctrl, "control", "patient")})
    for b in range(n_biomarkers):
        df[f"biomarker_{b}"] = np.round(data[:, b], 4)
    df["true_subtype"] = pd.Series(subtypes, dtype="Int64").mask(is_ctrl)   # blank for controls
    df["true_stage"] = stages.flatten().astype(int)
    truth = {"noise": noise, "seed": seed, "z_levels": [float(v) for v in Z_vals[0]], "z_max": float(Z_max[0]),
             "n_biomarkers": n_biomarkers, "n_subtypes": n_subtypes,
             "subtype_fractions": frac, "gt_sequence": gt_sequence.astype(int).tolist(),
             "event_numbering": "event = level_index * n_biomarkers + biomarker_index"}
    return df, truth


def truth_path_for(csv_path: Path) -> Path:
    return csv_path.with_suffix(".truth.json")


def main(
    output: Annotated[Path, typer.Option(help="CSV to write; a <name>.truth.json is written next to it")] = Path("input/synthetic_data.csv"),
    noise: Annotated[float, typer.Option(help="1.0 = tutorial default; larger = noisier (smaller signal-to-noise)")] = 1.0,
    seed: Annotated[int, typer.Option(help="Random seed (same seed = same hidden routes, subtypes and stages)")] = 0,
    n_biomarkers: Annotated[int, typer.Option()] = 5,
    n_subjects: Annotated[int, typer.Option(help="Total rows, controls included")] = 500,
    n_controls: Annotated[int, typer.Option()] = 100,
    n_subtypes: Annotated[int, typer.Option(help="Number of true hidden subtypes")] = 2,
) -> None:
    """Write a synthetic SuStaIn input table."""
    df, truth = simulate(n_biomarkers, n_subjects, n_controls, n_subtypes, noise, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    truth_path_for(output).write_text(json.dumps(truth, indent=1))
    typer.echo(f"wrote {output} ({len(df)} rows) and {truth_path_for(output)}")


if __name__ == "__main__":
    typer.run(main)

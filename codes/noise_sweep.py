"""Change ONE thing (noise) and keep everything else identical (same seed = same hidden routes, subtypes, stages).

    uv run python codes/noise_sweep.py --input-dir input --output output/noise_sweep

For each noise level this writes input/synthetic_data_noise_<noise>.csv (if missing) and runs the same analysis as
run_analysis.py. Takes about 10 minutes at the default settings (one process per noise level).
Outputs: summary.csv, by_stage.csv, sweep.json (used by build_explorer.py) and output/figures/noise_sweep.png.
"""
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

import make_synthetic_data as msd
import plotting
import preprocessing as prep
import run_analysis as ra


def run_one(args: tuple) -> dict:
    noise, input_dir, workroot, seed, n_startpoints, n_mcmc, max_subtypes = args
    path = input_dir / f"synthetic_data_noise_{noise}.csv"
    if not path.exists():
        df, truth = msd.simulate(noise=noise, seed=seed)
        input_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        msd.truth_path_for(path).write_text(json.dumps(truth, indent=1))
    df = prep.load_input(path)
    tj = json.loads(msd.truth_path_for(path).read_text())
    gt = np.array(tj["gt_sequence"])
    r = ra.analyse(df, tj["z_levels"], tj["z_max"], workroot / f"noise_{noise}", tj["n_subtypes"], max_subtypes,
                   n_startpoints, n_mcmc, seed, gt)
    ev = r["truth_eval"]
    mapping = ev["mapping_true_to_model"]
    n_true = len(gt)
    order = [mapping[t] for t in range(n_true)]                        # model subtype matched to each true subtype
    inv = {m: t for t, m in mapping.items()}
    conf = np.array(ev["confusion_true_by_model"])[:, order]            # columns re-ordered so the diagonal = correct
    return {
        "summary": {"noise": noise, "subtype_accuracy": ev["subtype_accuracy"], "stage_mae": ev["stage_mae"],
                    **{f"sequence_rho_{t}": v for t, v in enumerate(ev["sequence_spearman"])},
                    "mean_prob_subtype": r["metrics"]["mean_prob_subtype_patients"],
                    "patients_at_stage_0": r["metrics"]["patients_at_stage_0"],
                    **{f"mean_loglik_{k}_subtypes": v for k, v in r["metrics"]["mean_mcmc_loglik"].items()}},
        "by_stage": [{"noise": noise, **row} for row in ev["subtype_accuracy_by_true_stage"]],
        "explorer": {   # shape expected by explorer_template.html; subtypes are re-labelled to the TRUE order
            "noise": noise, "gt_sequence": gt.tolist(),
            "gt_subtypes": df["true_subtype"].fillna(0).astype(int).tolist(), "gt_stages": df["true_stage"].tolist(),
            "eval": {"subtype_accuracy": ev["subtype_accuracy"], "stage_mae": ev["stage_mae"],
                     "sequence_spearman": ev["sequence_spearman"], "confusion": conf.tolist(),
                     "mean_prob_subtype": r["metrics"]["mean_prob_subtype_patients"],
                     "subtype_accuracy_by_stage": ev["subtype_accuracy_by_true_stage"]},
            "P_est": np.round(r["P"][order], 3).tolist(), "data": np.round(r["Z"], 2).tolist(),
            "ml_subtype": [inv.get(int(s), -1) for s in r["ml_subtype"]], "ml_stage": r["ml_stage"].tolist()},
    }


def main(
    input_dir: Annotated[Path, typer.Option(help="Folder for the per-noise input CSVs (created if missing)")] = Path("input"),
    output: Annotated[Path, typer.Option(help="Folder for summary.csv, by_stage.csv, sweep.json")] = Path("output/noise_sweep"),
    figures: Annotated[Path, typer.Option(help="Folder for noise_sweep.png")] = Path("output/figures"),
    noises: Annotated[str, typer.Option(help="Comma-separated noise levels (1 = tutorial default)")] = "0.5,1,1.5,2,3",
    seed: Annotated[int, typer.Option()] = 0,
    n_startpoints: Annotated[int, typer.Option()] = 10,
    mcmc_iterations: Annotated[int, typer.Option()] = 10_000,
    max_subtypes: Annotated[int, typer.Option()] = 3,
) -> None:
    """Run the same analysis at several noise levels and summarise how accuracy changes."""
    levels = sorted(float(x) for x in noises.split(","))
    jobs = [(n, input_dir, output / "_fit", seed, n_startpoints, mcmc_iterations, max_subtypes) for n in levels]
    with Pool(len(jobs)) as pool:
        out = pool.map(run_one, jobs)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([o["summary"] for o in out])
    by_stage = pd.DataFrame([r for o in out for r in o["by_stage"]])
    summary.round(4).to_csv(output / "summary.csv", index=False)
    by_stage.round(4).to_csv(output / "by_stage.csv", index=False)
    (output / "sweep.json").write_text(json.dumps([o["explorer"] for o in out], separators=(",", ":")))
    plotting.noise_sweep(summary, {n: g.to_dict("records") for n, g in by_stage.groupby("noise")}, figures / "noise_sweep.png")
    typer.echo(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    typer.run(main)

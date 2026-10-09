"""Terminal entry point: fit pySuStaIn's z-score model to an input table and write results, tables and figures.

    uv run python codes/run_analysis.py --help
    uv run python codes/run_analysis.py --input input/synthetic_data.csv --output output/

Steps: load + validate (preprocessing.py) -> z-score vs controls -> fit 1..max-subtypes models -> read the
n-subtypes model -> results.csv / sequences.csv / metrics.json -> figures (plotting.py).
"""
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

import plotting
import preprocessing as prep

DATASET_NAME = "sim"


# ----------------------------------------------------------------------------- fitting and reading SuStaIn output
def fit_sustain(Z, Z_vals, Z_max, workdir: Path, n_startpoints: int, n_mcmc: int, max_subtypes: int, seed: int) -> None:
    """Fit models with 1..max_subtypes subtypes. Results are pickled in workdir (SuStaIn reuses an existing folder
    silently, so we always start from an empty one)."""
    import pySuStaIn

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    np.random.seed(seed)
    labels = [f"biomarker_{i}" for i in range(Z.shape[1])]
    sustain = pySuStaIn.ZscoreSustain(Z, Z_vals, Z_max, labels, n_startpoints, max_subtypes, n_mcmc,
                                      str(workdir), DATASET_NAME, False)
    sustain.run_sustain_algorithm()


def load_fit(workdir: Path, n_subtypes: int) -> dict:
    with open(workdir / "pickle_files" / f"{DATASET_NAME}_subtype{n_subtypes - 1}.pickle", "rb") as f:
        return pickle.load(f)


def aligned(res: dict) -> tuple[np.ndarray, np.ndarray]:
    """GOTCHA: pySuStaIn labels PATIENTS with subtypes sorted by prevalence (subtype 0 = most common; see
    AbstractSustain.subtype_and_stage_individuals: ix = argsort(mean f)[::-1]) but stores `samples_sequence` in the
    unsorted internal order. Re-order the sequences so index i means the same thing in both."""
    ix = np.argsort(np.mean(res["samples_f"], axis=1))[::-1]
    return res["samples_sequence"][ix], np.asarray(res["samples_f"])[ix]


def position_matrix(samples_sequence: np.ndarray) -> np.ndarray:
    """(n_subtypes, n_events, n_samples) -> (n_subtypes, event, position): fraction of MCMC samples that place `event`
    at `position`. This is the positional variance diagram."""
    n_s, n_ev, _ = samples_sequence.shape
    P = np.zeros((n_s, n_ev, n_ev))
    for s in range(n_s):
        for pos in range(n_ev):
            ev = samples_sequence[s, pos, :].astype(int)
            for e in range(n_ev):
                P[s, e, pos] = np.mean(ev == e)
    return P


def truth_position_matrix(gt_sequence: np.ndarray) -> np.ndarray:
    n_s, n_ev = gt_sequence.shape
    P = np.zeros((n_s, n_ev, n_ev))
    for s in range(n_s):
        P[s, gt_sequence[s], np.arange(n_ev)] = 1
    return P


# ----------------------------------------------------------------------------- comparison with hidden truth
def evaluate(true_subtype: np.ndarray, true_stage: np.ndarray, is_patient: np.ndarray, ml_subtype: np.ndarray,
             ml_stage: np.ndarray, P_est: np.ndarray, gt_sequence: Optional[np.ndarray]) -> dict:
    """Only possible for simulated data. Cluster numbers are arbitrary, so match model subtypes to true subtypes first."""
    n_ev = P_est.shape[1]
    cases = is_patient & (true_stage > 0)
    n_true, n_model = int(true_subtype[cases].max()) + 1, P_est.shape[0]
    conf = np.zeros((n_true, n_model), int)
    for t, p in zip(true_subtype[cases], ml_subtype[cases]):
        conf[t, p] += 1
    r, c = linear_sum_assignment(-conf)
    mapping = {int(a): int(b) for a, b in zip(r, c)}                 # true subtype -> model subtype
    gs = np.minimum(true_stage, n_ev)
    out = {"subtype_accuracy": float(conf[r, c].sum() / conf.sum()),
           "stage_mae": float(np.abs(ml_stage[cases] - gs[cases]).mean()),
           "confusion_true_by_model": conf.tolist(), "mapping_true_to_model": mapping,
           "subtype_accuracy_by_true_stage": [
               {"stage": st, "n": int(np.sum(cases & (gs == st))),
                "acc": float(np.mean(ml_subtype[cases & (gs == st)] ==
                                     np.array([mapping.get(int(t), -1) for t in true_subtype[cases & (gs == st)]])))}
               for st in range(1, n_ev + 1) if np.any(cases & (gs == st))]}
    if gt_sequence is not None:
        est_pos = (P_est * np.arange(n_ev)[None, None, :]).sum(2)    # mean position of every event
        rho = []
        for t in range(n_true):
            true_pos = np.empty(n_ev)
            true_pos[gt_sequence[t]] = np.arange(n_ev)
            rho.append(float(spearmanr(true_pos, est_pos[mapping[t]])[0]) if t in mapping else float("nan"))
        out["sequence_spearman"] = rho
    return out


# ----------------------------------------------------------------------------- one full analysis
def analyse(df: pd.DataFrame, z_levels: list[float], z_max: float, workdir: Path, n_subtypes: int = 2,
            max_subtypes: int = 3, n_startpoints: int = 10, n_mcmc: int = 10_000, seed: int = 0,
            gt_sequence: Optional[np.ndarray] = None) -> dict:
    cols = prep.biomarker_columns(df)
    Z, zinfo = prep.zscore_against_controls(df, cols)
    Z_vals, Z_max = prep.build_z_levels(len(cols), z_levels, z_max)
    if not 1 <= n_subtypes <= max_subtypes:
        raise ValueError("need 1 <= n_subtypes <= max_subtypes")
    fit_sustain(Z, Z_vals, Z_max, workdir, n_startpoints, n_mcmc, max_subtypes, seed)

    res = load_fit(workdir, n_subtypes)
    seqs, f = aligned(res)
    P = position_matrix(seqs)
    n_ev = P.shape[1]
    ml_sub = np.asarray(res["ml_subtype"]).flatten().astype(int)
    ml_stage = np.asarray(res["ml_stage"]).flatten().astype(int)
    is_ctrl = (df["group"] == "control").to_numpy()

    table = pd.DataFrame({"subject_id": df["subject_id"], "group": df["group"],
                          "subtype": ml_sub, "prob_subtype": np.round(np.asarray(res["prob_ml_subtype"]).flatten(), 4),
                          "stage": ml_stage, "prob_stage": np.round(np.asarray(res["prob_ml_stage"]).flatten(), 4),
                          "subtypable": ml_stage > 0})
    metrics = {"n_subjects": len(df), "n_controls": int(is_ctrl.sum()), "n_events": int(n_ev),
               "n_subtypes_reported": n_subtypes, "max_subtypes_fitted": max_subtypes,
               "z_levels": z_levels, "z_max": z_max, "preprocessing": zinfo,
               "mean_prob_subtype_patients": float(table.loc[~is_ctrl, "prob_subtype"].mean()),
               "patients_at_stage_0": int(((ml_stage == 0) & ~is_ctrl).sum()),
               "subtype_fractions": [float(x) for x in np.mean(f, axis=1)],
               "mean_mcmc_loglik": {}}
    trace = {}
    for k in range(1, max_subtypes + 1):
        ll = np.asarray(load_fit(workdir, k)["samples_likelihood"]).flatten()
        trace[k] = ll
        metrics["mean_mcmc_loglik"][str(k)] = float(ll.mean())

    rows = []
    levels = np.array(z_levels)
    for s in range(P.shape[0]):
        for pos in range(n_ev):
            e = int(P[s, :, pos].argmax())
            rows.append({"subtype": s, "position": pos + 1, "event": e, "biomarker": cols[e % len(cols)],
                         "z_threshold": float(levels[e // len(cols)]), "probability": round(float(P[s, e, pos]), 4)})
    sequences = pd.DataFrame(rows)

    truth_eval = None
    if "true_stage" in df.columns and "true_subtype" in df.columns:
        ts = df["true_stage"].to_numpy(int)
        tsub = df["true_subtype"].fillna(-1).to_numpy(int)
        truth_eval = evaluate(tsub, ts, ~is_ctrl, ml_sub, ml_stage, P, gt_sequence)
        inv = {m: t for t, m in truth_eval["mapping_true_to_model"].items()}
        table["true_subtype"] = df["true_subtype"]
        table["true_stage"] = ts
        table["subtype_correct"] = [pd.NA if c else int(inv.get(int(p), -1) == t)
                                    for c, p, t in zip(is_ctrl, ml_sub, tsub)]
        metrics["vs_truth"] = truth_eval
    return {"table": table, "sequences": sequences, "metrics": metrics, "P": P, "Z": Z, "ml_subtype": ml_sub,
            "ml_stage": ml_stage, "is_control": is_ctrl, "trace": trace, "n_biomarkers": len(cols),
            "n_levels": len(z_levels), "truth_eval": truth_eval, "prob_subtype": np.asarray(res["prob_ml_subtype"]).flatten()}


def write_outputs(r: dict, output: Path, gt_sequence: Optional[np.ndarray]) -> None:
    fig = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    r["table"].to_csv(output / "results.csv", index=False)
    r["sequences"].to_csv(output / "sequences.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(r["metrics"], indent=1))
    nb, nl = r["n_biomarkers"], r["n_levels"]
    P_true = truth_position_matrix(gt_sequence) if gt_sequence is not None else None
    if P_true is not None and r["truth_eval"] is not None:       # put true subtypes in the same order as the model's
        order = [r["truth_eval"]["mapping_true_to_model"].get(t) for t in range(P_true.shape[0])]
        if all(o is not None for o in order) and len(order) == r["P"].shape[0]:
            P_true = P_true[np.argsort(order)]
        else:
            P_true = None
    plotting.positional_variance(r["P"], nb, nl, fig / "positional_variance.png", P_true=P_true)
    n_ev = nb * nl
    patients = ~r["is_control"]
    plotting.stage_distribution(r["ml_stage"][patients], r["ml_subtype"][patients], n_ev, fig / "stage_distribution.png")
    plotting.data_scatter(r["Z"], r["is_control"], r["ml_subtype"], fig / "data_scatter.png")
    plotting.mcmc_trace(r["trace"], fig / "mcmc_trace.png")
    t = r["table"]
    if "true_stage" in t:
        m = patients & (t["true_stage"].to_numpy() > 0)
        plotting.stage_scatter(t["true_stage"].to_numpy()[m], r["ml_stage"][m], t["true_subtype"].fillna(-1).astype(int).to_numpy()[m],
                               n_ev, fig / "stage_scatter.png")


# ----------------------------------------------------------------------------- command line
def main(
    input: Annotated[Path, typer.Option("--input", "-i", help="CSV with subject_id, group, biomarker_* (see README)")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Folder for results.csv, sequences.csv, metrics.json, figures/")] = Path("output"),
    truth: Annotated[Optional[Path], typer.Option(help="Optional <input>.truth.json from make_synthetic_data.py; adds ground-truth panels and scores. Auto-detected next to the input.")] = None,
    z_levels: Annotated[Optional[str], typer.Option(help="Comma-separated severity thresholds, e.g. '1,2,3'. Default: from the truth file if present, else 1,2,3")] = None,
    z_max: Annotated[Optional[float], typer.Option(help="z-score at the end of the disease. Default: from the truth file, else 5")] = None,
    n_subtypes: Annotated[int, typer.Option(help="Which model (number of subtypes) to report")] = 2,
    max_subtypes: Annotated[int, typer.Option(help="Fit models with 1..max-subtypes subtypes (for comparing log-likelihoods)")] = 3,
    n_startpoints: Annotated[int, typer.Option(help="Tutorial uses 10 for speed; 25 recommended for real work")] = 10,
    mcmc_iterations: Annotated[int, typer.Option(help="Tutorial uses 1e4; 1e5-1e6 recommended for real work")] = 10_000,
    seed: Annotated[int, typer.Option()] = 0,
    workdir: Annotated[Optional[Path], typer.Option(help="Where SuStaIn writes its pickles (deleted first). Default: <output>/_fit")] = None,
) -> None:
    """Fit pySuStaIn (z-score model) and write results.csv, sequences.csv, metrics.json and figures/."""
    df = prep.load_input(input)
    tpath = truth or input.with_suffix(".truth.json")
    tj = json.loads(tpath.read_text()) if tpath.exists() else None
    levels = prep.parse_levels(z_levels) if z_levels else (tj["z_levels"] if tj else [1.0, 2.0, 3.0])
    zmax = z_max if z_max is not None else (tj["z_max"] if tj else 5.0)
    gt = np.array(tj["gt_sequence"]) if tj else None
    typer.echo(f"{len(df)} subjects, {len(prep.biomarker_columns(df))} biomarkers, levels {levels}, z_max {zmax}"
               f"{', with ground truth' if tj else ''}")
    r = analyse(df, levels, zmax, workdir or output / "_fit", n_subtypes, max_subtypes, n_startpoints,
                mcmc_iterations, seed, gt)
    write_outputs(r, output, gt)
    typer.echo(f"mean log-likelihood by number of subtypes: {r['metrics']['mean_mcmc_loglik']}")
    if r["truth_eval"]:
        e = r["truth_eval"]
        typer.echo(f"vs truth: subtype accuracy {e['subtype_accuracy']:.1%}, mean stage error {e['stage_mae']:.2f}"
                   + (f", event-order Spearman {np.round(e['sequence_spearman'], 2).tolist()}" if "sequence_spearman" in e else ""))
    typer.echo(f"wrote results to {output}/")


if __name__ == "__main__":
    typer.run(main)

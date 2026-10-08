"""Assign subtype and stage to new subjects with a model fitted earlier by run_analysis.py.

    uv run python codes/stage_new_subjects.py --fit output/ --input new_people.csv --output output/new_people/

The preprocessing (covariate betas, control mean/SD, sign flips, clipping) is
re-applied from `preprocess.json`, so new people are scored exactly like the
training cohort.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import yaml

from config import ConfigError, config_from_dict
from data_io import RunPaths, read_table
from models import build_model
from preprocessing import ZscorePreprocessor
from reporting import subjects_table, write_json
from results import SubtypeModel, load_subtype_model


def stage_new_subjects(fit_dir: str | Path, csv_path: str | Path, out_dir: str | Path, n_subtypes: int | None = None, quiet: bool = False) -> pd.DataFrame:
    fit = RunPaths(fit_dir)
    if not fit.run_json.exists() or not fit.preprocess_json.exists():
        raise ConfigError(f"{fit.out} does not look like a run_analysis.py results folder (run.json / preprocess.json missing)")
    run = json.loads(fit.run_json.read_text())
    info = json.loads(fit.preprocess_json.read_text())
    cfg = config_from_dict(yaml.safe_load(fit.config.read_text()))
    n_subtypes = n_subtypes or run["chosen_n_subtypes"]
    pickle_path = fit.pickle(cfg.output.dataset_name, n_subtypes)
    if not pickle_path.exists():
        raise ConfigError(f"No fitted {n_subtypes}-subtype model in {fit.out} (looked for {pickle_path.name})")
    fitted = load_subtype_model(pickle_path)

    df = read_table(csv_path)
    id_col = cfg.data.id_column
    if id_col not in df.columns:
        raise ConfigError(f"id column {id_col!r} not in {csv_path}")
    missing = [c for c in info["biomarkers"] + info["covariates"] if c not in df.columns]
    if missing:
        raise ConfigError(f"New data is missing columns the model was fitted with: {missing}")

    # pySuStaIn's constructor insists on an output folder even though staging writes nothing there.
    scratch = Path(tempfile.mkdtemp(prefix="sustain_stage_"))
    try:
        res = _stage(info["model_type"], info, cfg, fit, df, fitted, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage = res
    new = SubtypeModel(
        n_subtypes=n_subtypes,
        samples_sequence=fitted.samples_sequence, samples_f=fitted.samples_f, samples_likelihood=fitted.samples_likelihood,
        ml_sequence_EM=fitted.ml_sequence_EM, ml_f_EM=fitted.ml_f_EM,
        ml_subtype=np.asarray(ml_subtype).reshape(-1), prob_ml_subtype=np.asarray(prob_ml_subtype).reshape(-1),
        ml_stage=np.asarray(ml_stage).reshape(-1), prob_ml_stage=np.asarray(prob_ml_stage).reshape(-1),
        prob_subtype=np.asarray(prob_subtype), prob_stage=np.asarray(prob_stage), prob_subtype_stage=np.asarray(prob_subtype_stage),
        reorder=fitted.reorder,
    )
    groups = df[cfg.data.group_column] if cfg.data.group_column and cfg.data.group_column in df.columns else None
    table = subjects_table(new, df[id_col], groups)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "results.csv", index=False)
    write_json(out / "run.json", {
        "command": " ".join(sys.argv),
        "fit_dir": str(fit.out.resolve()),
        "fit_fingerprint": run.get("fingerprint"),
        "input": str(Path(csv_path).resolve()),
        "n_subtypes": n_subtypes,
        "n_subjects": len(df),
        "n_unassigned": int(np.isnan(new.ml_subtype).sum()),
        "finished": datetime.now(timezone.utc).isoformat(),
    })
    if not quiet:
        counts = table.loc[table.subtypable].subtype.value_counts().sort_index()
        print(f"staged {len(df)} subjects with the {n_subtypes}-subtype model from {fit.out}")
        for s, n in counts.items():
            print(f"  subtype {s}: {n}")
        print(f"  stage 0 (no events): {int((table.stage == 0).sum())}")
        print(f"wrote {out / 'results.csv'}")
    return table


def _stage(model_type: str, info: dict, cfg, fit: RunPaths, df: pd.DataFrame, fitted: SubtypeModel, scratch: Path):
    """Preprocess the new rows exactly like the training data, then call pySuStaIn's newData stager."""
    n_samples = cfg.fit.n_posterior_samples
    if model_type == "zscore":
        pp = ZscorePreprocessor.from_dict(info["zscore"])
        Z_new = pp.transform(df)
        if np.isnan(Z_new).any() and cfg.data.missing != "model":
            raise ConfigError("New data has missing values; the fitted model cannot handle NaN (data.missing != model)")
        thr = info["thresholds"]
        Z_vals = np.zeros((len(info["biomarkers"]), max(len(v) for v in thr["z_vals"].values())))
        for i, b in enumerate(info["biomarkers"]):
            v = thr["z_vals"][b]
            Z_vals[i, : len(v)] = v
        Z_max = np.array([thr["z_max"][b] for b in info["biomarkers"]])
        model, _ = build_model("zscore", {"Z": Z_new}, info["biomarkers"], cfg.fit, scratch, cfg.output.dataset_name,
                               Z_vals=Z_vals, Z_max=Z_max, missing_data_model=(cfg.data.missing == "model"))
        return model.subtype_and_stage_individuals_newData(Z_new, fitted.samples_sequence, fitted.samples_f, n_samples)

    from features_mixture import MixtureFeatures

    feats = MixtureFeatures.load(fit.mixtures_pkl)
    L_yes, L_no = feats.transform(df)
    model, _ = build_model(model_type, {"L_yes": L_yes, "L_no": L_no}, info["biomarkers"], cfg.fit, scratch, cfg.output.dataset_name)
    return model.subtype_and_stage_individuals_newData(L_yes, L_no, fitted.samples_sequence, fitted.samples_f, n_samples)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Subtype and stage new subjects with a saved SuStaIn fit.")
    p.add_argument("--fit", type=Path, required=True, help="results folder written by run_analysis.py")
    p.add_argument("--input", type=Path, required=True, help="new subjects; same biomarker (and covariate) columns as the fit")
    p.add_argument("--output", type=Path, required=True, help="where to write results.csv for the new subjects")
    p.add_argument("--n-subtypes", type=int, default=None, help="which fitted model to use (default: the one chosen by CV)")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    try:
        stage_new_subjects(a.fit, a.input, a.output, n_subtypes=a.n_subtypes, quiet=a.quiet)
    except (ConfigError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

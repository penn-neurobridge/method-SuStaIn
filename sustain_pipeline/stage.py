"""Assign subtype and stage to new subjects with a model fitted earlier by `sustain run`.

The preprocessing (covariate betas, control mean/SD, sign flips, clipping) is
re-applied from `preprocess.json`, so new people are scored exactly like the
training cohort.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import ConfigError, config_from_dict
from .io import RunPaths, read_table
from .models import build_model
from .preprocess import ZscorePreprocessor
from .report import subjects_table, write_json
from .results import SubtypeModel, load_subtype_model


def stage_new_subjects(fit_dir: str | Path, csv_path: str | Path, out_dir: str | Path, n_subtypes: int | None = None, quiet: bool = False) -> pd.DataFrame:
    fit = RunPaths(fit_dir)
    if not fit.run_json.exists() or not fit.preprocess_json.exists():
        raise ConfigError(f"{fit.out} does not look like a `sustain run` results folder (run.json / preprocess.json missing)")
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

    model_type = info["model_type"]
    if model_type == "zscore":
        pp = ZscorePreprocessor.from_dict(info["zscore"])
        Z_new = pp.transform(df)
        if np.isnan(Z_new).any() and cfg.data.missing != "model":
            raise ConfigError("New data has missing values; the fitted model cannot handle NaN (data.missing != model)")
        Z_vals = np.zeros((len(info["biomarkers"]), max(len(v) for v in info["thresholds"]["z_vals"].values())))
        for i, b in enumerate(info["biomarkers"]):
            v = info["thresholds"]["z_vals"][b]
            Z_vals[i, : len(v)] = v
        Z_max = np.array([info["thresholds"]["z_max"][b] for b in info["biomarkers"]])
        model, spec = build_model("zscore", {"Z": Z_new}, info["biomarkers"], cfg.fit, _scratch_dir(out_dir), cfg.output.dataset_name,
                                  Z_vals=Z_vals, Z_max=Z_max, missing_data_model=(cfg.data.missing == "model"))
        res = model.subtype_and_stage_individuals_newData(Z_new, fitted.samples_sequence, fitted.samples_f, cfg.fit.n_posterior_samples)
    else:
        from .features_mixture import MixtureFeatures

        feats = MixtureFeatures.load(fit.mixtures_pkl)
        L_yes, L_no = feats.transform(df)
        model, spec = build_model(model_type, {"L_yes": L_yes, "L_no": L_no}, info["biomarkers"], cfg.fit, _scratch_dir(out_dir), cfg.output.dataset_name)
        res = model.subtype_and_stage_individuals_newData(L_yes, L_no, fitted.samples_sequence, fitted.samples_f, cfg.fit.n_posterior_samples)

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
    table.to_csv(out / "subjects.csv", index=False)
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
        print(f"wrote {out / 'subjects.csv'}")
    return table


def _scratch_dir(out_dir: str | Path) -> Path:
    """pySuStaIn's constructor wants an output folder even though staging writes nothing there."""
    p = Path(out_dir) / "sustain_tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p

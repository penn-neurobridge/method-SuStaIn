"""The whole thing, as one function: CSV + config -> results folder.

`run_pipeline` is what the CLI calls and what a web front-end would call.
"""
from __future__ import annotations

import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from config import __version__
from config import Config, ConfigError, effective_model_type, group_masks, resolve_columns
from fitting import cross_validate, fingerprint, fit_all, prepare_cache
from data_io import RunPaths, apply_missing_policy, read_table
from models import build_model
from preprocessing import ZscorePreprocessor
from reporting import subjects_table, write_config_yaml, write_json, write_model_outputs
from results import SubtypeModel
from model_selection import choose_n_subtypes, model_selection_table
from thresholds import make_thresholds

Logger = Callable[[str], None]


@dataclass
class RunSummary:
    out_dir: Path
    model_type: str
    n_subjects: int
    n_events: int
    chosen_n_subtypes: int
    selection: str
    cache: str
    models: list[SubtypeModel]
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


def _log_factory(quiet: bool) -> Logger:
    t0 = time.perf_counter()

    def log(msg: str) -> None:
        if not quiet:
            print(f"[sustain {time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    return log


def run_pipeline(
    cfg: Config,
    csv_path: str | Path,
    out_dir: str | Path,
    overwrite: bool = False,
    skip_cv: bool = False,
    quiet: bool = False,
    command: str | None = None,
) -> RunSummary:
    log = _log_factory(quiet)
    started = datetime.now(timezone.utc)
    timings: dict[str, float] = {}
    warnings: list[str] = []
    paths = RunPaths(out_dir)
    paths.ensure(plots=cfg.output.plots)

    # ---------------------------------------------------------------- 1. data
    t = time.perf_counter()
    df = read_table(csv_path)
    cols = resolve_columns(cfg, df)
    n_in = len(df)
    df, n_dropped = apply_missing_policy(df, cols.biomarkers + cols.covariates, cfg.data.missing)
    if n_dropped:
        warnings.append(f"Dropped {n_dropped} row(s) with missing values (data.missing: drop_rows)")
    is_control, is_case = group_masks(cfg, df)
    model_type = effective_model_type(cfg)
    log(f"{n_in} subjects read ({int(is_control.sum())} controls, {int(is_case.sum())} patients); "
        f"{len(cols.biomarkers)} biomarkers: {cols.biomarkers}; model: {model_type}")

    # ---------------------------------------------------------- 2. preprocess
    arrays: dict[str, np.ndarray] = {}
    preprocess_info: dict = {"model_type": model_type, "biomarkers": cols.biomarkers, "covariates": cols.covariates}
    Z_vals = Z_max = None
    if model_type == "zscore":
        pp = ZscorePreprocessor(cols.biomarkers, cols.covariates, cfg.preprocess.clip_z)
        pp.fit(df, is_control, is_case, cfg.preprocess.direction)
        Z = pp.transform(df)
        if cfg.data.missing != "model" and np.isnan(Z).any():
            raise ConfigError("NaN in z-scores after preprocessing; set data.missing to drop_rows or model")
        thr = make_thresholds(Z[is_case], cols.biomarkers, cfg.model.thresholds)
        warnings += thr.warnings
        for w in thr.warnings:
            log("WARNING " + w)
        Z_vals, Z_max = thr.Z_vals, thr.Z_max
        arrays["Z"] = Z
        preprocess_info.update(zscore=pp.to_dict(), thresholds=thr.to_dict(cols.biomarkers))
        shown = Z
        flips = [b for b in cols.biomarkers if pp.flip[b] < 0]
        log(f"z-scored against controls; sign-flipped (decreasing) biomarkers: {flips or 'none'}; "
            f"{thr.n_events} events from thresholds {thr.to_dict(cols.biomarkers)['z_vals']}")
    else:
        from features_mixture import MixtureFeatures

        feats = MixtureFeatures(cols.biomarkers, model_type, cfg.preprocess.direction)
        feats.fit(df, is_control, is_case)
        L_yes, L_no = feats.transform(df)
        arrays["L_yes"], arrays["L_no"] = L_yes, L_no
        feats.save(paths.mixtures_pkl)
        preprocess_info.update(mixture=feats.describe())
        shown = L_yes / (L_yes + L_no)
        log(f"fitted per-biomarker mixture models ({model_type}); {len(cols.biomarkers)} events")
    timings["preprocess"] = time.perf_counter() - t

    # ------------------------------------------------------------- 3. cache
    fp = fingerprint(arrays, cfg.fit, cfg.cv, extra={"model_type": model_type, "Z_vals": Z_vals, "Z_max": Z_max, "biomarkers": cols.biomarkers})
    cache = prepare_cache(paths, fp, overwrite)
    if cache == "reused":
        log("Reusing cached fit (fingerprint matches); pass --overwrite to refit")

    # Write the sidecars early so a crash mid-fit still leaves the inputs documented.
    cfg_dict = cfg.to_dict()
    write_config_yaml(paths.config, cfg_dict)
    pd.concat([df[[cols.id]].reset_index(drop=True), pd.DataFrame(shown, columns=cols.biomarkers)], axis=1).to_csv(paths.data_preprocessed, index=False)

    # --------------------------------------------------------------- 4. fit
    t = time.perf_counter()
    model, spec = build_model(model_type, arrays, cols.biomarkers, cfg.fit, paths.sustain, cfg.output.dataset_name,
                              Z_vals=Z_vals, Z_max=Z_max, missing_data_model=(cfg.data.missing == "model"))
    preprocess_info["events"] = [e.label for e in spec.events]
    preprocess_info["n_events"] = spec.n_events
    write_json(paths.preprocess_json, preprocess_info)
    log(f"fitting 1..{cfg.fit.n_subtypes_max} subtypes: {cfg.fit.n_startpoints} start points, "
        f"{cfg.fit.n_mcmc} MCMC iterations, seed {cfg.fit.seed}")
    models = fit_all(model, paths, cfg.output.dataset_name, cfg.fit.n_subtypes_max)
    timings["fit"] = time.perf_counter() - t
    log(f"fit done in {timings['fit']:.0f}s")

    # ---------------------------------------------------- 5. model selection
    CVIC = loglike = None
    if cfg.cv.enabled and not skip_cv:
        t = time.perf_counter()
        log(f"{cfg.cv.n_folds}-fold cross-validation for the number of subtypes")
        CVIC, loglike = cross_validate(model, is_control, cfg.cv)
        chosen, selection = choose_n_subtypes(CVIC), "cvic"
        timings["cv"] = time.perf_counter() - t
        log("CVIC per n_subtypes: " + ", ".join(f"{s + 1}: {c:.1f}" for s, c in enumerate(CVIC)) + f"  → {chosen} subtype(s)")
    else:
        chosen, selection = cfg.fit.n_subtypes_max, "none"
        log(f"cross-validation skipped; reporting the {chosen}-subtype model")
    model_selection_table(CVIC, loglike, cfg.fit.n_subtypes_max, chosen).to_csv(paths.model_selection, index=False)

    # ------------------------------------------------------------ 6. report
    t = time.perf_counter()
    ids = df[cols.id]
    groups = df[cols.group] if cols.group else None
    for m in models:
        write_model_outputs(m, spec.events, ids, groups, paths.model_dir(m.n_subtypes), title=f"{m.n_subtypes}-subtype model")
    chosen_dir = paths.model_dir(chosen)
    for name in ("results.csv", "sequences.csv", "sequences.md"):
        shutil.copyfile(chosen_dir / name, paths.out / name)
    if cfg.output.plots:
        import plotting as plots

        plots.biomarker_distributions(shown, is_control, spec, paths.figures / "biomarkers_after_preprocess.png")
        for m in models:
            plots.positional_variance(type(model), m, spec, len(df), paths.figures / f"pvd_S{m.n_subtypes}.png")
        if CVIC is not None:
            plots.cvic_plot(CVIC, loglike, chosen, paths.figures / "cvic.png")
        plots.mcmc_trace(models, paths.figures / "mcmc_trace.png")
    timings["report"] = time.perf_counter() - t

    chosen_model = models[chosen - 1]
    n_unassigned = int(np.isnan(chosen_model.ml_subtype).sum())
    if n_unassigned:
        warnings.append(f"{n_unassigned} subject(s) could not be assigned (posterior underflow); "
                        "consider lowering preprocess.clip_z or using thresholds.mode: auto")
    write_json(paths.run_json, {
        "command": command,
        "input": str(Path(csv_path).resolve()),
        "started": started.isoformat(),
        "finished": datetime.now(timezone.utc).isoformat(),
        "duration_s": {k: round(v, 1) for k, v in timings.items()},
        "versions": _versions(),
        "fingerprint": fp,
        "cache": cache,
        "n_subjects": len(df),
        "n_controls": int(is_control.sum()),
        "n_cases": int(is_case.sum()),
        "n_dropped_rows": n_dropped,
        "model_type": model_type,
        "n_events": spec.n_events,
        "n_subtypes_max": cfg.fit.n_subtypes_max,
        "chosen_n_subtypes": chosen,
        "selection": selection,
        "cvic": None if CVIC is None else [float(c) for c in CVIC],
        "subtype_fractions": [float(f) for f in chosen_model.fractions],
        "subtype_counts": [int(c) for c in chosen_model.counts()],
        "n_unassigned": n_unassigned,
        "warnings": warnings,
    })
    log(f"wrote {paths.out}  (results.csv, sequences.md, model_selection.csv, figures/)")
    return RunSummary(paths.out, model_type, len(df), spec.n_events, chosen, selection, cache, models, warnings, timings)


def _versions() -> dict:
    import importlib.metadata as md
    import json

    out = {"python": platform.python_version(), "method_sustain": __version__}
    for pkg in ("numpy", "pandas", "scikit-learn", "pySuStaIn", "kde_ebm"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    try:
        direct = json.loads(md.distribution("pySuStaIn").read_text("direct_url.json") or "{}")
        out["pySuStaIn_commit"] = direct.get("vcs_info", {}).get("commit_id")
    except Exception:  # noqa: BLE001 - purely informational
        pass
    return out

"""Running pySuStaIn's fit and cross-validation, with a cache that is honest.

pySuStaIn reloads any pickle it finds under `output_folder/pickle_files/`
without checking that the data or settings still match. We fingerprint the
inputs and refuse to reuse a fit unless the fingerprint matches, so a stale
result can never be mistaken for a fresh one.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from config import CVConfig, FitConfig
from data_io import RunPaths
from results import SubtypeModel, load_subtype_model


class CacheMismatch(RuntimeError):
    pass


def fingerprint(arrays: dict[str, np.ndarray], fit_cfg: FitConfig, cv_cfg: CVConfig, extra: dict | None = None) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key], dtype=float)
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    params = {
        "n_startpoints": fit_cfg.n_startpoints,
        "n_subtypes_max": fit_cfg.n_subtypes_max,
        "n_mcmc": fit_cfg.n_mcmc,
        "seed": fit_cfg.seed,
        "cv_n_folds": cv_cfg.n_folds,
        "cv_seed": cv_cfg.seed,
        **(extra or {}),
    }
    h.update(json.dumps(params, sort_keys=True, default=_jsonable).encode())
    return h.hexdigest()[:16]


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    raise TypeError(type(o))


def prepare_cache(paths: RunPaths, fp: str, overwrite: bool) -> str:
    """Return 'fresh' or 'reused'. Raise CacheMismatch if old pickles belong to a different fit."""
    if overwrite and paths.sustain.exists():
        shutil.rmtree(paths.sustain)
    if not paths.pickles.exists() or not any(paths.pickles.glob("*.pickle")):
        return "fresh"
    stored = None
    if paths.run_json.exists():
        try:
            stored = json.loads(paths.run_json.read_text()).get("fingerprint")
        except (OSError, ValueError):
            stored = None
    if stored == fp:
        return "reused"
    raise CacheMismatch(
        f"{paths.out} already holds a SuStaIn fit "
        f"({'from different data or settings' if stored else 'with no run.json to identify it'}). "
        "Pass --overwrite to discard it, or choose another --out folder."
    )


def fit_all(model, paths: RunPaths, dataset_name: str, n_subtypes_max: int) -> list[SubtypeModel]:
    """Fit 1..n_subtypes_max subtypes (pySuStaIn does this hierarchically) and load every model."""
    model.run_sustain_algorithm(plot=False)
    out = []
    for s in range(1, n_subtypes_max + 1):
        p = paths.pickle(dataset_name, s)
        if not p.exists():
            raise FileNotFoundError(f"pySuStaIn did not write {p}")
        out.append(load_subtype_model(p))
    return out


def make_folds(is_control: np.ndarray, cv_cfg: CVConfig) -> list[np.ndarray]:
    """Test-index arrays per fold, stratified by control/patient when there are both."""
    n = len(is_control)
    if is_control.any() and (~is_control).any():
        splitter = StratifiedKFold(n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed)
        return [test for _, test in splitter.split(np.zeros(n), is_control.astype(int))]
    splitter = KFold(n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed)
    return [test for _, test in splitter.split(np.zeros(n))]


def cross_validate(model, is_control: np.ndarray, cv_cfg: CVConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (CVIC per n_subtypes, mean test log-likelihood per fold x n_subtypes). Lower CVIC is better."""
    folds = make_folds(is_control, cv_cfg)
    test_idxs = np.empty(len(folds), dtype=object)
    for i, f in enumerate(folds):
        test_idxs[i] = f
    CVIC, loglike = model.cross_validate_sustain_model(test_idxs)
    return np.asarray(CVIC, dtype=float).reshape(-1), np.asarray(loglike, dtype=float)

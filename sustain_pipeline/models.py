"""Construct the right pySuStaIn model object from prepared arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import FitConfig
from .sequences import Event, events_for_mixture, events_for_zscore


@dataclass
class ModelSpec:
    type: str                      # zscore | mixture_unsupervised | mixture_gmm | mixture_kde
    biomarkers: list[str]
    events: list[Event]
    Z_vals: np.ndarray | None = None
    Z_max: np.ndarray | None = None

    @property
    def is_zscore(self) -> bool:
        return self.type == "zscore"

    @property
    def n_events(self) -> int:
        return len(self.events)


def build_model(
    model_type: str,
    arrays: dict[str, np.ndarray],
    biomarkers: list[str],
    fit_cfg: FitConfig,
    sustain_dir: Path,
    dataset_name: str,
    Z_vals: np.ndarray | None = None,
    Z_max: np.ndarray | None = None,
    missing_data_model: bool = False,
):
    """Return (pySuStaIn model, ModelSpec). Does not fit anything.

    pySuStaIn gotchas handled here: the data must be a plain ndarray, the MCMC
    count and seed must be Python ints, and `output_folder` must already exist
    (pySuStaIn calls a non-recursive os.mkdir on its pickle subfolder).
    """
    import pySuStaIn

    sustain_dir = Path(sustain_dir)
    sustain_dir.mkdir(parents=True, exist_ok=True)
    seed = None if fit_cfg.seed is None else int(fit_cfg.seed)
    common = (
        list(biomarkers),
        int(fit_cfg.n_startpoints),
        int(fit_cfg.n_subtypes_max),
        int(fit_cfg.n_mcmc),
        str(sustain_dir),
        str(dataset_name),
        bool(fit_cfg.parallel),
    )
    if model_type == "zscore":
        Z = np.ascontiguousarray(arrays["Z"], dtype=float)
        Z_vals = np.asarray(Z_vals, dtype=float)
        Z_max = np.asarray(Z_max, dtype=float)
        cls = pySuStaIn.ZscoreSustainMissingData if missing_data_model else pySuStaIn.ZscoreSustain
        model = cls(Z, Z_vals, Z_max, *common, seed=seed)
        spec = ModelSpec("zscore", list(biomarkers), events_for_zscore(list(biomarkers), Z_vals), Z_vals, Z_max)
    elif model_type.startswith("mixture"):
        L_yes = np.ascontiguousarray(arrays["L_yes"], dtype=float)
        L_no = np.ascontiguousarray(arrays["L_no"], dtype=float)
        model = pySuStaIn.MixtureSustain(L_yes, L_no, *common, seed=seed)
        spec = ModelSpec(model_type, list(biomarkers), events_for_mixture(list(biomarkers)))
    else:
        raise ValueError(f"Unknown model type {model_type!r}")
    return model, spec

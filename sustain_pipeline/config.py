"""YAML config -> validated dataclasses.

Everything the pipeline needs to know about the CSV and the fit lives here, so
switching from the demo to real data is an edit to the `data:` block only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import pandas as pd
import yaml

MODEL_TYPES = ("auto", "zscore", "mixture_unsupervised", "mixture_gmm", "mixture_kde")
MISSING_POLICIES = ("error", "drop_rows", "model")
THRESHOLD_MODES = ("fixed", "auto")
MAX_Z_LEVELS = 6  # pySuStaIn's positional-variance plot has 6 colours


class ConfigError(ValueError):
    """Raised for anything wrong in the YAML or in how it matches the CSV."""


# --------------------------------------------------------------------------- #
# Dataclasses (one per YAML block)
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    id_column: str = "subject_id"
    group_column: str | None = "group"
    control_label: Any = "control"
    biomarker_columns: str | list[str] = "auto"
    covariate_columns: list[str] = field(default_factory=list)
    missing: str = "error"


@dataclass
class PreprocessConfig:
    direction: str | dict[str, str] = "auto"
    clip_z: float | None = 10.0


@dataclass
class ThresholdConfig:
    mode: str = "fixed"
    z_vals: list[float] | dict[str, list[float]] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    z_max: float | dict[str, float] = 5.0
    n_levels: int = 3
    min_fraction_above_top: float = 0.05
    strict: bool = False


@dataclass
class ModelConfig:
    type: str = "auto"
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)


@dataclass
class FitConfig:
    n_startpoints: int = 3
    n_subtypes_max: int = 3
    n_mcmc: int = 5000
    seed: int | None = 1
    parallel: bool = False
    n_posterior_samples: int = 1000


@dataclass
class CVConfig:
    enabled: bool = True
    n_folds: int = 2
    seed: int = 0


@dataclass
class OutputConfig:
    dataset_name: str = "sustain"
    plots: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _build(cls, raw: Any, where: str):
    """Recursively build dataclass `cls` from a dict, rejecting unknown keys."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"`{where}` must be a mapping, got {type(raw).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ConfigError(f"Unknown key(s) under `{where}`: {sorted(unknown)}. Known: {sorted(known)}")
    hints = get_type_hints(cls)  # resolves the string annotations from `from __future__ import annotations`
    kwargs = {}
    for name in known:
        if name not in raw:
            continue
        value = raw[name]
        if is_dataclass(hints[name]):
            value = _build(hints[name], value, f"{where}.{name}")
        kwargs[name] = value
    return cls(**kwargs)


def _as_int(value: Any, where: str, minimum: int = 1) -> int:
    """PyYAML reads `1e4` as the *string* '1e4'; accept that, floats, and ints."""
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        raise ConfigError(f"`{where}` must be a number, got {value!r}") from None
    if out != float(value):
        raise ConfigError(f"`{where}` must be a whole number, got {value!r}")
    if out < minimum:
        raise ConfigError(f"`{where}` must be >= {minimum}, got {out}")
    return out


def coerce(cfg: Config) -> Config:
    """Cast numeric fields to plain Python types (pySuStaIn breaks on numpy ints / strings)."""
    f = cfg.fit
    f.n_startpoints = _as_int(f.n_startpoints, "fit.n_startpoints")
    f.n_subtypes_max = _as_int(f.n_subtypes_max, "fit.n_subtypes_max")
    f.n_mcmc = _as_int(f.n_mcmc, "fit.n_mcmc", minimum=10)
    f.n_posterior_samples = _as_int(f.n_posterior_samples, "fit.n_posterior_samples")
    f.seed = None if f.seed is None else _as_int(f.seed, "fit.seed", minimum=0)
    f.parallel = bool(f.parallel)
    cfg.cv.n_folds = _as_int(cfg.cv.n_folds, "cv.n_folds", minimum=2)
    cfg.cv.seed = _as_int(cfg.cv.seed, "cv.seed", minimum=0)
    cfg.cv.enabled = bool(cfg.cv.enabled)
    t = cfg.model.thresholds
    t.n_levels = _as_int(t.n_levels, "model.thresholds.n_levels")
    t.min_fraction_above_top = float(t.min_fraction_above_top)
    t.strict = bool(t.strict)
    if isinstance(t.z_vals, list):
        t.z_vals = [float(z) for z in t.z_vals]
    else:
        t.z_vals = {k: [float(z) for z in v] for k, v in t.z_vals.items()}
    t.z_max = float(t.z_max) if not isinstance(t.z_max, dict) else {k: float(v) for k, v in t.z_max.items()}
    if cfg.preprocess.clip_z is not None:
        cfg.preprocess.clip_z = float(cfg.preprocess.clip_z)
    if isinstance(cfg.data.covariate_columns, str):
        cfg.data.covariate_columns = [cfg.data.covariate_columns]
    cfg.data.covariate_columns = list(cfg.data.covariate_columns or [])
    return cfg


def validate_static(cfg: Config) -> None:
    """Checks that need only the config, not the data."""
    if cfg.model.type not in MODEL_TYPES:
        raise ConfigError(f"model.type must be one of {MODEL_TYPES}, got {cfg.model.type!r}")
    if cfg.data.missing not in MISSING_POLICIES:
        raise ConfigError(f"data.missing must be one of {MISSING_POLICIES}, got {cfg.data.missing!r}")
    if cfg.data.missing == "model" and cfg.model.type not in ("auto", "zscore"):
        raise ConfigError("data.missing: model (ZscoreSustainMissingData) only works with model.type: zscore")
    t = cfg.model.thresholds
    if t.mode not in THRESHOLD_MODES:
        raise ConfigError(f"model.thresholds.mode must be one of {THRESHOLD_MODES}, got {t.mode!r}")
    if t.mode == "fixed":
        for name, zv in (t.z_vals.items() if isinstance(t.z_vals, dict) else [("*", t.z_vals)]):
            if not zv or any(b <= a for a, b in zip(zv, zv[1:])) or zv[0] <= 0:
                raise ConfigError(f"z_vals for {name} must be positive and strictly increasing, got {zv}")
            if len(zv) > MAX_Z_LEVELS:
                raise ConfigError(f"At most {MAX_Z_LEVELS} z levels per biomarker (plotting limit); {name} has {len(zv)}")
            zmax = t.z_max.get(name, None) if isinstance(t.z_max, dict) else t.z_max
            if zmax is not None and zmax <= zv[-1]:
                raise ConfigError(f"z_max for {name} ({zmax}) must exceed the top z threshold ({zv[-1]})")
    else:
        if t.n_levels > MAX_Z_LEVELS:
            raise ConfigError(f"model.thresholds.n_levels must be <= {MAX_Z_LEVELS}")
        if not 0 < t.min_fraction_above_top < 0.5:
            raise ConfigError("model.thresholds.min_fraction_above_top must be in (0, 0.5)")
    d = cfg.preprocess.direction
    if isinstance(d, str):
        if d not in ("auto", "increasing", "decreasing"):
            raise ConfigError(f"preprocess.direction must be auto|increasing|decreasing|mapping, got {d!r}")
    elif isinstance(d, dict):
        bad = {k: v for k, v in d.items() if v not in ("increasing", "decreasing")}
        if bad:
            raise ConfigError(f"preprocess.direction entries must be increasing|decreasing: {bad}")
    else:
        raise ConfigError("preprocess.direction must be a string or a mapping biomarker -> direction")
    if cfg.data.group_column is None:
        if cfg.model.type in ("zscore", "mixture_gmm", "mixture_kde"):
            raise ConfigError(f"model.type: {cfg.model.type} needs a control group; set data.group_column")
        if d == "auto":
            raise ConfigError("preprocess.direction: auto needs a patient group; give explicit directions when data.group_column is null")
    if cfg.fit.n_posterior_samples > cfg.fit.n_mcmc:
        cfg.fit.n_posterior_samples = cfg.fit.n_mcmc


def load_config(path: str | Path) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _build(Config, raw, "config")
    coerce(cfg)
    validate_static(cfg)
    return cfg


def config_from_dict(raw: dict) -> Config:
    cfg = _build(Config, raw, "config")
    coerce(cfg)
    validate_static(cfg)
    return cfg


# --------------------------------------------------------------------------- #
# Matching the config against a DataFrame
# --------------------------------------------------------------------------- #
@dataclass
class Columns:
    id: str
    group: str | None
    biomarkers: list[str]
    covariates: list[str]


def resolve_columns(cfg: Config, df: pd.DataFrame) -> Columns:
    d = cfg.data
    missing = [c for c in [d.id_column, d.group_column, *d.covariate_columns] if c and c not in df.columns]
    if missing:
        raise ConfigError(f"Column(s) named in config not found in the CSV: {missing}. CSV has: {list(df.columns)}")
    reserved = {d.id_column, d.group_column, *d.covariate_columns} - {None}
    if d.biomarker_columns == "auto":
        bio = [c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]
        if not bio:
            raise ConfigError("biomarker_columns: auto found no numeric columns outside id/group/covariates")
    else:
        bio = list(d.biomarker_columns)
        absent = [c for c in bio if c not in df.columns]
        if absent:
            raise ConfigError(f"biomarker_columns not found in the CSV: {absent}")
        overlap = [c for c in bio if c in reserved]
        if overlap:
            raise ConfigError(f"biomarker_columns overlap id/group/covariate columns: {overlap}")
    non_numeric = [c for c in bio if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ConfigError(f"Biomarker columns must be numeric: {non_numeric}")
    non_numeric_cov = [c for c in d.covariate_columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_cov:
        raise ConfigError(f"Covariate columns must be numeric (encode categories as 0/1 first): {non_numeric_cov}")
    if d.group_column is not None:
        if not (df[d.group_column] == d.control_label).any():
            raise ConfigError(
                f"control_label {d.control_label!r} never appears in column {d.group_column!r}; "
                f"values seen: {sorted(map(str, df[d.group_column].unique()))[:10]}"
            )
    if isinstance(cfg.preprocess.direction, dict):
        unknown = set(cfg.preprocess.direction) - set(bio)
        if unknown:
            raise ConfigError(f"preprocess.direction names biomarkers not in the data: {sorted(unknown)}")
    t = cfg.model.thresholds
    for name, val in (("z_vals", t.z_vals), ("z_max", t.z_max)):
        if isinstance(val, dict):
            unknown = set(val) - set(bio)
            if unknown:
                raise ConfigError(f"model.thresholds.{name} names biomarkers not in the data: {sorted(unknown)}")
    return Columns(id=d.id_column, group=d.group_column, biomarkers=bio, covariates=list(d.covariate_columns))


def group_masks(cfg: Config, df: pd.DataFrame):
    """Boolean (is_control, is_case) arrays. Without a group column, nobody is a control."""
    import numpy as np

    if cfg.data.group_column is None:
        n = len(df)
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
    is_control = (df[cfg.data.group_column] == cfg.data.control_label).to_numpy()
    return is_control, ~is_control


def effective_model_type(cfg: Config) -> str:
    if cfg.model.type != "auto":
        return cfg.model.type
    return "zscore" if cfg.data.group_column is not None else "mixture_unsupervised"

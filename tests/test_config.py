import pytest

from config import ConfigError, config_from_dict, effective_model_type, resolve_columns


def test_yaml_scientific_notation_string_is_coerced():
    cfg = config_from_dict({"fit": {"n_mcmc": "1e4", "seed": 7}})
    assert cfg.fit.n_mcmc == 10000 and type(cfg.fit.n_mcmc) is int
    assert type(cfg.fit.seed) is int


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="n_mcmcc"):
        config_from_dict({"fit": {"n_mcmcc": 5}})


def test_z_max_must_exceed_top_threshold():
    with pytest.raises(ConfigError, match="z_max"):
        config_from_dict({"model": {"thresholds": {"z_vals": [1, 2, 3], "z_max": 3}}})


def test_no_groups_requires_explicit_direction_and_mixture():
    with pytest.raises(ConfigError, match="direction"):
        config_from_dict({"data": {"group_column": None}, "model": {"type": "mixture_unsupervised"}})
    with pytest.raises(ConfigError, match="control group"):
        config_from_dict({"data": {"group_column": None}, "preprocess": {"direction": "increasing"}, "model": {"type": "zscore"}})
    cfg = config_from_dict({"data": {"group_column": None}, "preprocess": {"direction": "increasing"}})
    assert effective_model_type(cfg) == "mixture_unsupervised"


def test_auto_biomarkers_exclude_id_group_covariates(frame):
    df, _ = frame
    cfg = config_from_dict({"data": {"covariate_columns": ["age"]}})
    cols = resolve_columns(cfg, df)
    assert cols.biomarkers == ["A", "D"] and cols.covariates == ["age"]


def test_missing_control_label_is_reported(frame):
    df, _ = frame
    cfg = config_from_dict({"data": {"control_label": "HC"}})
    with pytest.raises(ConfigError, match="never appears"):
        resolve_columns(cfg, df)

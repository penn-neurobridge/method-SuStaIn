"""Slow: actually calls pySuStaIn on the demo data with tiny settings (~1 min). Run with `pytest -m slow`."""
from pathlib import Path

import pandas as pd
import pytest

from config import load_config
from fitting import CacheMismatch
from pipeline import run_pipeline
from stage_new_subjects import stage_new_subjects

INPUT = Path(__file__).resolve().parents[1] / "input"
EXPECTED = ["config.yaml", "run.json", "preprocess.json", "data_preprocessed.csv", "model_selection.csv",
            "results.csv", "sequences.csv", "sequences.md", "models/S1/results.csv", "models/S2/posteriors.npz",
            "figures/biomarkers_after_preprocess.png", "figures/pvd_S2.png", "figures/cvic.png", "figures/mcmc_trace.png"]


def tiny(cfg):
    cfg.fit.n_startpoints, cfg.fit.n_subtypes_max, cfg.fit.n_mcmc, cfg.fit.n_posterior_samples = 1, 2, 200, 100
    cfg.cv.n_folds = 2
    return cfg


@pytest.mark.slow
def test_run_rerun_and_stage(tmp_path):
    cfg = tiny(load_config(INPUT / "config.yaml"))
    out = tmp_path / "res"
    s = run_pipeline(cfg, INPUT / "synthetic_data.csv", out, quiet=True)
    for rel in EXPECTED:
        assert (out / rel).exists(), rel
    subjects = pd.read_csv(out / "results.csv")
    assert len(subjects) == 400 and s.chosen_n_subtypes == 2 and s.cache == "fresh"
    assert subjects.subtypable.sum() > 250

    s2 = run_pipeline(cfg, INPUT / "synthetic_data.csv", out, quiet=True)
    assert s2.cache == "reused"

    cfg.fit.n_mcmc = 300
    with pytest.raises(CacheMismatch):
        run_pipeline(cfg, INPUT / "synthetic_data.csv", out, quiet=True)

    restaged = stage_new_subjects(out, INPUT / "synthetic_data.csv", tmp_path / "restaged", quiet=True)
    m = subjects.merge(restaged, on="subject_id", suffixes=("_a", "_b"))
    ok = m.subtypable_a & m.subtypable_b
    assert (m.subtype_a[ok] == m.subtype_b[ok]).mean() > 0.95


@pytest.mark.slow
def test_no_controls_mixture_path(tmp_path):
    cfg = tiny(load_config(INPUT / "config_nocontrols.yaml"))
    s = run_pipeline(cfg, INPUT / "synthetic_nocontrols.csv", tmp_path / "noc", quiet=True)
    assert s.model_type == "mixture_unsupervised" and s.n_events == 5
    assert (tmp_path / "noc" / "mixtures.pkl").exists()

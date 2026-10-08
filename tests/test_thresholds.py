import numpy as np
import pytest

from config import ConfigError, ThresholdConfig
from thresholds import make_thresholds


def _cases(spread):
    rng = np.random.default_rng(1)
    return rng.uniform(0, spread, size=(200, 2))


def test_fixed_mode_warns_when_too_few_exceed_top_threshold():
    t = make_thresholds(_cases(np.array([6.0, 2.0])), ["A", "B"], ThresholdConfig(mode="fixed"))
    assert t.n_events == 6
    assert t.fraction_above_top["A"] > 0.05 and t.fraction_above_top["B"] == 0
    assert len(t.warnings) == 1 and "'B'" in t.warnings[0]


def test_fixed_mode_strict_raises():
    with pytest.raises(ConfigError, match="exceed the top threshold"):
        make_thresholds(_cases(np.array([6.0, 2.0])), ["A", "B"], ThresholdConfig(mode="fixed", strict=True))


def test_fixed_mode_per_biomarker_mapping_and_padding():
    cfg = ThresholdConfig(mode="fixed", z_vals={"A": [1.0, 2.0, 3.0], "B": [1.0]}, z_max={"A": 5.0, "B": 2.0})
    t = make_thresholds(_cases(np.array([6.0, 3.0])), ["A", "B"], cfg)
    assert t.Z_vals.tolist() == [[1, 2, 3], [1, 0, 0]] and t.n_events == 4


def test_auto_mode_satisfies_the_rule_by_construction():
    t = make_thresholds(_cases(np.array([6.0, 2.0])), ["A", "B"], ThresholdConfig(mode="auto", n_levels=3))
    for b in ("A", "B"):
        assert abs(t.fraction_above_top[b] - 0.05) < 0.02
    assert (np.diff(t.Z_vals, axis=1) > 0).all()
    assert (t.Z_max > t.Z_vals.max(axis=1)).all()
    assert not t.warnings

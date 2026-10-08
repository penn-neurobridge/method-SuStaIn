import numpy as np
import pytest

from preprocessing import ZscorePreprocessor


def test_controls_become_standard_normal_and_decreasing_biomarker_is_flipped(frame):
    df, ctrl = frame
    pp = ZscorePreprocessor(["A", "D"], ["age"]).fit(df, ctrl, ~ctrl, "auto")
    Z = pp.transform(df)
    assert pp.flip == {"A": 1, "D": -1}
    assert np.allclose(Z[ctrl].mean(0), 0, atol=1e-9)
    assert np.allclose(Z[ctrl].std(0, ddof=1), 1, atol=1e-9)
    assert (Z[~ctrl].mean(0) > 2).all()  # patients abnormal in the positive direction for both


def test_covariate_is_regressed_out(frame):
    df, ctrl = frame
    pp = ZscorePreprocessor(["A"], ["age"]).fit(df, ctrl, ~ctrl, "auto")
    Z = pp.transform(df)
    r = np.corrcoef(Z[ctrl, 0], df.age[ctrl])[0, 1]
    assert abs(r) < 1e-6


def test_round_trip_through_dict_reproduces_transform_including_flip(frame):
    df, ctrl = frame
    pp = ZscorePreprocessor(["A", "D"], ["age"], clip_z=4).fit(df, ctrl, ~ctrl, {"D": "decreasing"})
    Z = pp.transform(df)
    Z2 = ZscorePreprocessor.from_dict(pp.to_dict()).transform(df)
    assert np.array_equal(Z, Z2)
    assert Z.max() <= 4 and Z.min() >= -4


def test_explicit_direction_overrides_auto(frame):
    df, ctrl = frame
    pp = ZscorePreprocessor(["A"], []).fit(df, ctrl, ~ctrl, "decreasing")
    assert pp.flip["A"] == -1 and pp.direction_source["A"] == "config"


def test_missing_column_on_transform_is_an_error(frame):
    df, ctrl = frame
    pp = ZscorePreprocessor(["A", "D"], ["age"]).fit(df, ctrl, ~ctrl, "auto")
    with pytest.raises(ValueError, match="missing"):
        pp.transform(df.drop(columns=["D"]))

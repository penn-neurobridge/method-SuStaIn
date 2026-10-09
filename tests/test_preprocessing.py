import numpy as np
import pandas as pd
import pytest

import preprocessing as prep


def make_df(n=60, decreasing=False):
    rng = np.random.default_rng(0)
    group = np.array(["control"] * 30 + ["patient"] * (n - 30))
    shift = np.where(group == "patient", -2.0 if decreasing else 2.0, 0.0)
    return pd.DataFrame({"subject_id": [f"s{i}" for i in range(n)], "group": group,
                         "biomarker_0": 10 + 3 * rng.standard_normal(n) + 3 * shift,
                         "biomarker_1": rng.standard_normal(n) + shift})


def test_controls_end_up_mean0_sd1():
    df = make_df()
    Z, info = prep.zscore_against_controls(df, prep.biomarker_columns(df))
    ctrl = (df["group"] == "control").to_numpy()
    assert np.allclose(Z[ctrl].mean(0), 0) and np.allclose(Z[ctrl].std(0), 1)
    assert info["flipped"] == []


def test_decreasing_biomarkers_are_flipped_to_increase_with_disease():
    df = make_df(decreasing=True)
    Z, info = prep.zscore_against_controls(df, prep.biomarker_columns(df))
    assert info["flipped"] == ["biomarker_0", "biomarker_1"]
    assert (Z[(df["group"] == "patient").to_numpy()].mean(0) > 0).all()


def test_biomarker_columns_sorted_numerically():
    df = pd.DataFrame(columns=["subject_id", "group", "biomarker_10", "biomarker_2", "biomarker_0"])
    assert prep.biomarker_columns(df) == ["biomarker_0", "biomarker_2", "biomarker_10"]


@pytest.mark.parametrize("break_it, message", [
    (lambda d: d.drop(columns="group"), "missing required"),
    (lambda d: d.assign(group="case"), "'group' must be"),
    (lambda d: d.assign(subject_id="x"), "unique"),
    (lambda d: d.assign(biomarker_0=np.nan), "missing values"),
    (lambda d: d.drop(columns=["biomarker_0", "biomarker_1"]), "biomarker_"),
])
def test_load_input_rejects_bad_files(tmp_path, break_it, message):
    df = make_df()
    path = tmp_path / "x.csv"
    break_it(df).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        prep.load_input(path)


def test_z_levels_validation():
    Z_vals, Z_max = prep.build_z_levels(3, [1, 2, 3], 5)
    assert Z_vals.shape == (3, 3) and (Z_max == 5).all()
    with pytest.raises(ValueError):
        prep.build_z_levels(3, [2, 1], 5)
    with pytest.raises(ValueError):
        prep.build_z_levels(3, [1, 2, 3], 3)

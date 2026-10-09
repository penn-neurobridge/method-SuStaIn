import numpy as np

import make_synthetic_data as msd


def test_default_table_follows_the_input_contract():
    df, truth = msd.simulate(seed=0)
    assert list(df.columns[:2]) == ["subject_id", "group"] and df.shape == (500, 9)
    assert (df["group"] == "control").sum() == 100
    assert df.loc[df["group"] == "control", "true_subtype"].isna().all()
    assert (df.loc[df["group"] == "control", "true_stage"] == 0).all()
    assert truth["z_levels"] == [1, 2, 3] and truth["z_max"] == 5


def test_noise_shrinks_the_signal_but_keeps_the_hidden_truth():
    a, ta = msd.simulate(noise=1.0, seed=0)
    b, tb = msd.simulate(noise=2.0, seed=0)
    assert np.allclose(tb["z_levels"], [.5, 1, 1.5])
    assert ta["gt_sequence"] == tb["gt_sequence"]                          # same biology
    assert a["true_stage"].tolist() == b["true_stage"].tolist()


def test_same_seed_same_data():
    a, _ = msd.simulate(seed=3)
    b, _ = msd.simulate(seed=3)
    assert a.equals(b)

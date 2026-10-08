import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def frame():
    """60 subjects, 20 controls; A increases with disease, D decreases; both depend on age."""
    rng = np.random.default_rng(0)
    n = 60
    age = rng.normal(60, 8, n)
    ctrl = np.r_[np.ones(20, bool), np.zeros(40, bool)]
    A = 10 + 0.5 * (age - 60) + np.where(ctrl, 0, 3) + rng.normal(0, 1, n)
    D = 50 - 2 * (age - 60) - np.where(ctrl, 0, 6) + rng.normal(0, 1, n)
    df = pd.DataFrame({"subject_id": [f"s{i}" for i in range(n)], "group": np.where(ctrl, "control", "patient"),
                       "age": age.round(1), "A": A, "D": D})
    return df, ctrl

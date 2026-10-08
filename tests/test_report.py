import numpy as np
import pandas as pd

from reporting import subjects_table
from helpers import fake_model


def test_subjects_table_contract():
    S, N, I, M = 2, 3, 4, 10
    seq = np.tile(np.arange(N)[:, None], (1, I))
    m = fake_model(np.stack([seq, seq[::-1]]), np.array([[0.6] * I, [0.4] * I]), ml_stage=np.r_[np.zeros(3), np.ones(7)])
    m.ml_subtype[0] = np.nan  # an unassignable subject
    ids = pd.Series([f"s{i}" for i in range(M)])
    t = subjects_table(m, ids, pd.Series(["control"] * 3 + ["patient"] * 7))
    assert len(t) == M and list(t.columns[:7]) == ["subject_id", "group", "subtype", "prob_subtype", "stage", "prob_stage", "subtypable"]
    assert t.subtype.isna().sum() == 1 and t.subtype.dropna().min() == 1  # 1-based, NaN preserved
    assert t.subtypable.sum() == 7 and not t.subtypable[:3].any()
    assert {"p_subtype_1", "p_subtype_2", "p_stage_0", "p_stage_3"} <= set(t.columns)

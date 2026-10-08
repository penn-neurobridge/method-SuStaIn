import numpy as np

from helpers import fake_model
from results import reorder_by_fraction
from sequences import describe_sequence, events_for_mixture, events_for_zscore, sequence_table
from simulate import DEMO_SEQUENCES, event_index


def test_event_labels_follow_pysustain_column_major_order():
    ev = events_for_zscore(["A", "B", "C", "D"], np.array([[1, 2, 3]] * 4))
    assert [e.label for e in ev[:5]] == ["A>1", "B>1", "C>1", "D>1", "A>2"]
    assert ev[event_index(1, 2, 4)].label == "B>2"
    seq = [event_index(b, l, 4) for b, l in DEMO_SEQUENCES[0]]
    assert describe_sequence(seq, ev).startswith("A>1 → A>2 → B>1 → A>3")
    assert [e.label for e in events_for_mixture(["x", "y"])] == ["x", "y"]


def test_zero_padded_thresholds_are_skipped():
    ev = events_for_zscore(["A", "B"], np.array([[1, 2], [1, 0]]))
    assert [e.label for e in ev] == ["A>1", "B>1", "A>2"]


def test_reorder_by_fraction_puts_largest_first():
    f = np.array([[0.3, 0.3], [0.7, 0.7]])
    assert reorder_by_fraction(f).tolist() == [1, 0]


def test_sequence_table_positions_and_mcmc_mean():
    I = 4
    s0 = np.tile(np.array([0, 1, 2])[:, None], (1, I))
    s1 = np.array([[2, 2, 1, 1], [1, 1, 2, 2], [0, 0, 0, 0]])  # events 1 and 2 swap half the time
    m = fake_model(np.stack([s0, s1]), np.array([[0.6] * I, [0.4] * I]))
    ev = events_for_mixture(["A", "B", "C"])
    t = sequence_table(m, ev)
    assert t[t.subtype == 1].event.tolist() == ["A", "B", "C"]
    assert t[t.subtype == 1].sd_position.eq(0).all()
    row = t[(t.subtype == 2) & (t.event == "B")].iloc[0]
    assert row.mean_position == 1.5 and row.sd_position == 0.5

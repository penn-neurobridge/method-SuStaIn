import numpy as np

import run_analysis as ra


def test_aligned_sorts_sequences_by_prevalence():
    # pySuStaIn numbers patient subtypes by descending prevalence but stores samples_sequence unsorted
    seq = np.stack([np.full((4, 3), 1), np.full((4, 3), 2)])               # (n_subtypes, n_events, n_samples)
    f = np.array([[.2, .2, .2], [.8, .8, .8]])                             # subtype index 1 is the more common one
    out, f_sorted = ra.aligned({"samples_sequence": seq, "samples_f": f})
    assert (out[0] == 2).all() and (out[1] == 1).all() and f_sorted[0, 0] == .8


def test_position_matrix_is_a_permutation_distribution():
    rng = np.random.default_rng(0)
    seqs = np.stack([np.stack([rng.permutation(6) for _ in range(50)], axis=1) for _ in range(2)])    # (2, 6, 50)
    P = ra.position_matrix(seqs)
    assert np.allclose(P.sum(1), 1) and np.allclose(P.sum(2), 1)


def test_evaluate_perfect_and_swapped_labels():
    gt = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    P_est = ra.truth_position_matrix(gt)[::-1]                              # model numbers the subtypes the other way round
    true_sub = np.array([-1, -1, 0, 0, 1, 1, 1])
    true_stage = np.array([0, 0, 1, 3, 2, 4, 4])
    is_patient = true_sub >= 0
    ml_sub = np.array([0, 1, 1, 1, 0, 0, 0])                               # all patients swapped, consistently
    out = ra.evaluate(true_sub, true_stage, is_patient, ml_sub, true_stage.copy(), P_est, gt)
    assert out["subtype_accuracy"] == 1.0 and out["stage_mae"] == 0.0
    assert out["mapping_true_to_model"] == {0: 1, 1: 0}
    assert np.allclose(out["sequence_spearman"], 1.0)

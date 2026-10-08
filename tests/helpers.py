import numpy as np

from results import SubtypeModel


def fake_model(samples_sequence, samples_f, ml_stage=None):
    S, N, I = samples_sequence.shape
    M = 10
    ml_stage = np.ones(M) if ml_stage is None else ml_stage
    return SubtypeModel(
        n_subtypes=S, samples_sequence=samples_sequence, samples_f=samples_f, samples_likelihood=np.zeros(I),
        ml_sequence_EM=samples_sequence[:, :, 0], ml_f_EM=samples_f.mean(1),
        ml_subtype=np.repeat(np.arange(S), M // S).astype(float)[:M], prob_ml_subtype=np.ones(M),
        ml_stage=ml_stage, prob_ml_stage=np.ones(M), prob_subtype=np.ones((M, S)) / S,
        prob_stage=np.ones((M, N + 1)) / (N + 1), prob_subtype_stage=np.ones((M, N + 1, S)), reorder=np.arange(S),
    )

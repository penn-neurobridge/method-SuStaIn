"""Loading a fitted pySuStaIn model from its pickle, in a consistent index order.

pySuStaIn's `subtype_and_stage_individuals` sorts subtypes by descending mean
fraction before assigning people (`ix = argsort(mean(samples_f))[::-1]`), so
`ml_subtype == k` refers to `samples_sequence[ix[k]]`, *not* `[k]`. We apply
that same permutation to the sequence arrays once, here, so every table and
plot downstream shares one numbering.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SubtypeModel:
    n_subtypes: int
    samples_sequence: np.ndarray  # (S, N_events, n_mcmc)
    samples_f: np.ndarray         # (S, n_mcmc)
    samples_likelihood: np.ndarray  # (n_mcmc,)
    ml_sequence_EM: np.ndarray    # (S, N_events)
    ml_f_EM: np.ndarray           # (S,)
    ml_subtype: np.ndarray        # (M,) float, NaN when unassignable, 0-based
    prob_ml_subtype: np.ndarray   # (M,)
    ml_stage: np.ndarray          # (M,) float, 0 = no events yet
    prob_ml_stage: np.ndarray     # (M,)
    prob_subtype: np.ndarray      # (M, S)
    prob_stage: np.ndarray        # (M, N_events + 1)
    prob_subtype_stage: np.ndarray  # (M, N_events + 1, S)
    reorder: np.ndarray           # the permutation applied to pySuStaIn's native order

    @property
    def n_events(self) -> int:
        return self.samples_sequence.shape[1]

    @property
    def n_subjects(self) -> int:
        return self.ml_subtype.shape[0]

    @property
    def fractions(self) -> np.ndarray:
        """Mean MCMC subtype fraction, in the reordered index space."""
        return self.samples_f.mean(axis=1)

    @property
    def ml_sequence(self) -> np.ndarray:
        """Maximum-likelihood ordering per subtype (from the EM step)."""
        return self.ml_sequence_EM

    def mean_position(self) -> np.ndarray:
        """(S, N_events) mean and (S, N_events) SD of each event's position across MCMC samples."""
        S, N, I = self.samples_sequence.shape
        pos = np.empty((S, N, I))
        for s in range(S):
            for i in range(I):
                pos[s, self.samples_sequence[s, :, i].astype(int), i] = np.arange(N)
        return pos.mean(axis=2), pos.std(axis=2)

    def counts(self) -> np.ndarray:
        """Number of subjects assigned to each subtype (stage > 0 only)."""
        sub = self.ml_subtype
        ok = ~np.isnan(sub) & (self.ml_stage > 0)
        return np.bincount(sub[ok].astype(int), minlength=self.n_subtypes)


def reorder_by_fraction(samples_f: np.ndarray) -> np.ndarray:
    """The permutation pySuStaIn applies before assigning subjects (AbstractSustain.py:537)."""
    return np.argsort(np.mean(samples_f, axis=1))[::-1]


def load_subtype_model(pickle_path: str | Path) -> SubtypeModel:
    with open(pickle_path, "rb") as fh:
        d = pickle.load(fh)
    samples_f = np.asarray(d["samples_f"])
    ix = reorder_by_fraction(samples_f)
    flat = lambda key: np.asarray(d[key]).reshape(-1)  # noqa: E731  (M,1) -> (M,)
    return SubtypeModel(
        n_subtypes=samples_f.shape[0],
        samples_sequence=np.asarray(d["samples_sequence"])[ix],
        samples_f=samples_f[ix],
        samples_likelihood=np.asarray(d["samples_likelihood"]).reshape(-1),
        ml_sequence_EM=np.asarray(d["ml_sequence_EM"])[ix],
        ml_f_EM=np.asarray(d["ml_f_EM"]).reshape(-1)[ix],
        ml_subtype=flat("ml_subtype"),
        prob_ml_subtype=flat("prob_ml_subtype"),
        ml_stage=flat("ml_stage"),
        prob_ml_stage=flat("prob_ml_stage"),
        prob_subtype=np.asarray(d["prob_subtype"]),
        prob_stage=np.asarray(d["prob_stage"]),
        prob_subtype_stage=np.asarray(d["prob_subtype_stage"]),
        reorder=ix,
    )

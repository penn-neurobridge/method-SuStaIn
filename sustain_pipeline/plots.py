"""Figures: biomarker distributions, positional variance diagrams, CVIC, MCMC trace."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .models import ModelSpec  # noqa: E402
from .results import SubtypeModel  # noqa: E402


def biomarker_distributions(values: np.ndarray, is_control: np.ndarray, spec: ModelSpec, path: Path) -> None:
    """Controls vs patients per biomarker after preprocessing, with threshold lines for z-score models."""
    B = len(spec.biomarkers)
    ncol = min(4, B)
    nrow = int(np.ceil(B / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
    has_controls = bool(is_control.any())
    for i, ax in enumerate(axes.flat):
        if i >= B:
            ax.axis("off")
            continue
        v = values[:, i]
        v = v[~np.isnan(v)] if not has_controls else v
        bins = np.linspace(np.nanmin(values[:, i]), np.nanmax(values[:, i]), 30)
        if has_controls:
            ax.hist(values[is_control, i], bins=bins, alpha=0.6, label="controls", color="#4c72b0")
            ax.hist(values[~is_control, i], bins=bins, alpha=0.6, label="patients", color="#dd8452")
        else:
            ax.hist(v, bins=bins, alpha=0.7, color="#dd8452", label="all subjects")
        if spec.is_zscore:
            for z in spec.Z_vals[i]:
                if z > 0:
                    ax.axvline(z, color="k", lw=0.8, ls="--")
            ax.axvline(spec.Z_max[i], color="r", lw=0.8, ls=":")
        ax.set_title(spec.biomarkers[i], fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7)
    xlabel = "z-score (positive = abnormal)" if spec.is_zscore else "P(abnormal)"
    fig.supxlabel(xlabel, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def positional_variance(sustain_cls, model: SubtypeModel, spec: ModelSpec, n_subjects: int, path: Path) -> None:
    """pySuStaIn's positional variance diagram for one model; subtypes already in our index order."""
    titles = [f"Subtype {s + 1}  (f = {model.fractions[s]:.2f}, n = {int(model.counts()[s])})" for s in range(model.n_subtypes)]
    kwargs = dict(
        biomarker_labels=list(spec.biomarkers),
        subtype_order=list(range(model.n_subtypes)),
        subtype_titles=titles,
        save_path=None,
    )
    if spec.is_zscore:
        figs, _ = sustain_cls.plot_positional_var(model.samples_sequence, model.samples_f, n_subjects, spec.Z_vals, **kwargs)
    else:
        figs, _ = sustain_cls.plot_positional_var(model.samples_sequence, model.samples_f, n_subjects, **kwargs)
    figs[0].savefig(path, dpi=150, bbox_inches="tight")
    for f in figs:
        plt.close(f)


def cvic_plot(CVIC: np.ndarray, loglike: np.ndarray, chosen: int, path: Path) -> None:
    S = np.arange(1, len(CVIC) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3))
    a1.plot(S, CVIC, "o-", color="#4c72b0")
    a1.plot(chosen, CVIC[chosen - 1], "o", ms=12, mfc="none", mec="r", mew=2)
    a1.set_xlabel("number of subtypes")
    a1.set_ylabel("CVIC  (lower is better)")
    a1.set_xticks(S)
    a2.boxplot([loglike[:, s - 1] for s in S], positions=S)
    a2.set_xlabel("number of subtypes")
    a2.set_ylabel("test log-likelihood per fold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mcmc_trace(models: list[SubtypeModel], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    for m in models:
        ax.plot(m.samples_likelihood, lw=0.6, label=f"{m.n_subtypes} subtype{'s' if m.n_subtypes > 1 else ''}")
    ax.set_xlabel("MCMC iteration")
    ax.set_ylabel("log-likelihood")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

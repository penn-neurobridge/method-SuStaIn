"""All figures. Pure functions: arrays in, PNG out. No SuStaIn imports here."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUBTYPE_COLORS = ["#2fc4b0", "#f59e4b", "#7a8cf5", "#d65db1"]
LEVEL_COLORS = np.array([[229, 72, 77], [214, 64, 159], [62, 99, 221], [60, 160, 90]]) / 255   # w=1 red, 2 magenta, 3 blue
BACKGROUND = 0.14


def _save(fig, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pv_image(P: np.ndarray, n_biomarkers: int, n_levels: int) -> np.ndarray:
    """P[event, position] -> RGB image, rows = biomarkers, columns = position in the sequence.
    event = level * n_biomarkers + biomarker (pySuStaIn's ordering); colour = severity level; opacity = probability."""
    n_events = n_biomarkers * n_levels
    img = np.full((n_biomarkers, n_events, 3), BACKGROUND)
    for e in range(n_events):
        level, b = divmod(e, n_biomarkers)
        for pos in range(n_events):
            p = P[e, pos]
            img[b, pos] = (1 - p) * img[b, pos] + p * LEVEL_COLORS[level]
    return img


def positional_variance(P_est: np.ndarray, n_biomarkers: int, n_levels: int, path: Path,
                        P_true: np.ndarray | None = None, labels: list[str] | None = None, title: str = "") -> None:
    """The paper's Fig. 1 style. One row per subtype; with ground truth, truth is the left column."""
    n_sub, n_events = P_est.shape[0], n_biomarkers * n_levels
    labels = labels or [f"B{i}" for i in range(n_biomarkers)]
    ncol = 2 if P_true is not None else 1
    fig, axes = plt.subplots(n_sub, ncol, figsize=(5.4 * ncol, 2.4 * n_sub), squeeze=False)
    for s in range(n_sub):
        panels = ([("ground truth", P_true[s])] if P_true is not None else []) + [("SuStaIn", P_est[s])]
        for c, (name, P) in enumerate(panels):
            ax = axes[s, c]
            ax.imshow(_pv_image(P, n_biomarkers, n_levels), aspect="auto", interpolation="nearest")
            ax.set_yticks(range(n_biomarkers), labels, fontsize=8)
            ax.set_xticks(range(0, n_events, 2), range(1, n_events + 1, 2), fontsize=8)
            ax.set_title(f"Subtype {s}: {name}", fontsize=10, color=SUBTYPE_COLORS[s % 4])
            if s == n_sub - 1:
                ax.set_xlabel("position in sequence (1 = first event)")
    fig.suptitle((title + "\n" if title else "") + "red = level 1, magenta = level 2, blue = level 3; faded = uncertain", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def stage_distribution(stage: np.ndarray, subtype: np.ndarray, n_events: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.arange(-0.5, n_events + 1.5)
    for s in np.unique(subtype):
        ax.hist(stage[subtype == s], bins=bins, alpha=0.65, color=SUBTYPE_COLORS[int(s) % 4], label=f"subtype {s}")
    ax.set_xlabel("SuStaIn stage (number of abnormal events so far)")
    ax.set_ylabel("patients")
    ax.legend()
    _save(fig, path)


def stage_scatter(true_stage: np.ndarray, est_stage: np.ndarray, true_subtype: np.ndarray, n_events: int, path: Path) -> None:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for s in np.unique(true_subtype):
        m = true_subtype == s
        ax.scatter(np.minimum(true_stage[m], n_events) + rng.uniform(-.2, .2, m.sum()), est_stage[m] + rng.uniform(-.2, .2, m.sum()),
                   s=10, alpha=.55, color=SUBTYPE_COLORS[int(s) % 4], label=f"true subtype {s}")
    ax.plot([0, n_events], [0, n_events], "k--", lw=1)
    ax.set_xlabel("true stage")
    ax.set_ylabel("SuStaIn stage")
    ax.legend(fontsize=8)
    _save(fig, path)


def data_scatter(Z: np.ndarray, is_control: np.ndarray, subtype: np.ndarray, path: Path, pairs=((0, 1), (2, 3))) -> None:
    pairs = [p for p in pairs if max(p) < Z.shape[1]] or [(0, 0)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.2), squeeze=False)
    for ax, (i, j) in zip(axes[0], pairs):
        ax.scatter(Z[is_control, i], Z[is_control, j], s=8, c="lightgray", label="controls")
        for s in np.unique(subtype[~is_control]):
            m = (~is_control) & (subtype == s)
            ax.scatter(Z[m, i], Z[m, j], s=10, alpha=.65, color=SUBTYPE_COLORS[int(s) % 4], label=f"assigned subtype {s}")
        ax.set_xlabel(f"biomarker_{i} (z-score vs controls)")
        ax.set_ylabel(f"biomarker_{j}")
    axes[0, 0].legend(fontsize=8)
    _save(fig, path)


def mcmc_trace(samples_likelihood: dict[int, np.ndarray], path: Path) -> None:
    """Should look like a stationary, noisy band. Long flat stretches mean the sampler is stuck."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for k, ll in samples_likelihood.items():
        ax.plot(ll, lw=.6, label=f"{k} subtype model")
    ax.set_xlabel("MCMC iteration")
    ax.set_ylabel("log likelihood")
    ax.legend(fontsize=8)
    _save(fig, path)


def noise_sweep(summary, by_stage: dict[float, list[dict]], path: Path) -> None:
    n = summary["noise"].to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    a = ax[0]
    a.plot(n, 100 * summary["subtype_accuracy"], "o-", c=SUBTYPE_COLORS[0])
    a.axhline(50, ls=":", c="gray")
    a.set_ylim(40, 100)
    a.set_xlabel("noise (x tutorial default)")
    a.set_ylabel("% patients, correct subtype", color=SUBTYPE_COLORS[0])
    b = a.twinx()
    b.plot(n, summary["stage_mae"], "s--", c=SUBTYPE_COLORS[1])
    b.set_ylabel("mean stage error", color=SUBTYPE_COLORS[1])
    a.set_title("Patient-level accuracy falls steadily")
    a = ax[1]
    for t in range(2):
        col = f"sequence_rho_{t}"
        if col in summary:
            a.plot(n, summary[col], "o-", c=SUBTYPE_COLORS[t], label=f"subtype {t}")
    a.set_ylim(0.5, 1.02)
    a.set_xlabel("noise (x tutorial default)")
    a.set_ylabel("event-order recovery (Spearman)")
    a.set_title("...but the event ORDER holds up longer")
    a.legend()
    a = ax[2]
    for (noise, rows), c in zip(sorted(by_stage.items()), plt.cm.viridis(np.linspace(0, .9, len(by_stage)))):
        a.plot([r["stage"] for r in rows], [100 * r["acc"] for r in rows], "o-", ms=3, c=c, label=f"noise {noise}")
    a.axhline(50, ls=":", c="gray")
    a.set_xlabel("true stage")
    a.set_ylabel("% correct subtype")
    a.set_title("Hardest at the first and last stages")
    a.legend(fontsize=8)
    fig.suptitle("pySuStaIn on simulated z-score data: changing only the noise", y=1.02)
    fig.tight_layout()
    _save(fig, path)

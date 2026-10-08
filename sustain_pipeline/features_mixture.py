"""Per-biomarker mixture models -> the (L_yes, L_no) likelihoods MixtureSustain needs.

Three flavours:
  mixture_unsupervised  no labels needed. A 2-component Gaussian mixture per
                        biomarker (scikit-learn); the component on the
                        "abnormal" side (per `direction`) is the event.
                        Experimental: with heavy overlap the components can
                        split on something other than disease.
  mixture_gmm / mixture_kde
                        kde_ebm's models (what pySuStaIn's own examples use).
                        These *require* control (0) and case (1) labels.
"""
from __future__ import annotations

from pathlib import Path

import dill
import numpy as np
import pandas as pd

from .config import ConfigError


class MixtureFeatures:
    def __init__(self, biomarkers: list[str], kind: str, direction="auto"):
        self.biomarkers = list(biomarkers)
        self.kind = kind
        self.direction = direction
        self.models: list = []
        self.info: list[dict] = []

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, is_control: np.ndarray, is_case: np.ndarray) -> "MixtureFeatures":
        X = df[self.biomarkers].to_numpy(dtype=float)
        if np.isnan(X).any():
            raise ConfigError("Mixture models cannot handle NaN; set data.missing: drop_rows")
        if self.kind == "mixture_unsupervised":
            self._fit_unsupervised(X)
        elif self.kind in ("mixture_gmm", "mixture_kde"):
            if not is_control.any() or not is_case.any():
                raise ConfigError(f"{self.kind} needs both controls and patients (kde_ebm fits on labels)")
            self._fit_kde_ebm(X, is_control, is_case)
        else:
            raise ValueError(self.kind)
        return self

    def _fit_unsupervised(self, X: np.ndarray) -> None:
        from sklearn.mixture import GaussianMixture

        for j, name in enumerate(self.biomarkers):
            d = self.direction.get(name, "auto") if isinstance(self.direction, dict) else self.direction
            if d == "auto":
                raise ConfigError(f"mixture_unsupervised needs an explicit direction for {name!r} (increasing|decreasing)")
            x = X[:, j].reshape(-1, 1)
            gm = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(x)
            means = gm.means_.ravel()
            abnormal = int(np.argmax(means)) if d == "increasing" else int(np.argmin(means))
            self.models.append(gm)
            self.info.append({"biomarker": name, "direction": d, "normal_mean": float(means[1 - abnormal]),
                              "abnormal_mean": float(means[abnormal]), "abnormal_component": abnormal,
                              "abnormal_weight": float(gm.weights_[abnormal])})

    def _fit_kde_ebm(self, X: np.ndarray, is_control: np.ndarray, is_case: np.ndarray) -> None:
        from kde_ebm.mixture_model import fit_all_gmm_models, fit_all_kde_models

        y = np.full(len(X), 2)  # 2 = unlabelled, ignored by kde_ebm
        y[is_control], y[is_case] = 0, 1
        fitter = fit_all_gmm_models if self.kind == "mixture_gmm" else fit_all_kde_models
        self.models = list(fitter(X, y))
        self.info = [{"biomarker": b, "kind": self.kind} for b in self.biomarkers]

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = [c for c in self.biomarkers if c not in df.columns]
        if missing:
            raise ValueError(f"Columns required by the fitted mixture models are missing: {missing}")
        X = df[self.biomarkers].to_numpy(dtype=float)
        L_yes = np.empty_like(X)
        L_no = np.empty_like(X)
        for j, m in enumerate(self.models):
            x = X[:, j]
            if self.kind == "mixture_unsupervised":
                dens = np.exp(m._estimate_weighted_log_prob(x.reshape(-1, 1)))  # (n, 2) weighted component pdfs
                k = self.info[j]["abnormal_component"]
                L_yes[:, j], L_no[:, j] = dens[:, k], dens[:, 1 - k]
            elif self.kind == "mixture_gmm":
                L_no[:, j], L_yes[:, j] = m.pdf(None, x)
            else:
                L_no[:, j], L_yes[:, j] = m.pdf(x.reshape(-1, 1))
        total = L_yes + L_no
        total[total == 0] = 1.0
        return L_yes / total, L_no / total  # normalised, as in pySuStaIn's mixed-data example

    # ---------------------------------------------------------- persistence
    def describe(self) -> dict:
        return {"kind": self.kind, "per_biomarker": self.info}

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            dill.dump(self, fh)

    @classmethod
    def load(cls, path: Path) -> "MixtureFeatures":
        with open(path, "rb") as fh:
            return dill.load(fh)

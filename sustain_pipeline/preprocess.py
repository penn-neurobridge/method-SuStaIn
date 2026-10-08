"""Raw biomarker columns -> the z-score matrix SuStaIn expects.

For every biomarker, in order:
  1. regress out covariates (fit on controls only, applied to everyone)
  2. z-score the residual against the controls (mean 0, SD 1)
  3. flip sign so that "more abnormal" is positive
  4. clip extreme values (very large z underflow pySuStaIn's likelihood to 0/0)

All four pieces are stored together in one object, so staging new subjects
later can never apply the z-scoring without the sign flip (the mistake that
bit the journal-club notebook).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ZscorePreprocessor:
    biomarkers: list[str]
    covariates: list[str] = field(default_factory=list)
    clip_z: float | None = 10.0
    # fitted state, one entry per biomarker
    beta: dict[str, list[float]] = field(default_factory=dict)  # [intercept, *covariate slopes]
    mean: dict[str, float] = field(default_factory=dict)
    sd: dict[str, float] = field(default_factory=dict)
    flip: dict[str, int] = field(default_factory=dict)  # +1 or -1
    direction_source: dict[str, str] = field(default_factory=dict)  # "auto" | "config"

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        df: pd.DataFrame,
        is_control: np.ndarray,
        is_case: np.ndarray,
        direction: str | dict[str, str] = "auto",
    ) -> "ZscorePreprocessor":
        if is_control.sum() < 3:
            raise ValueError(f"Need at least 3 controls to z-score against, found {int(is_control.sum())}")
        X = self._design(df)
        X_ctrl = X[is_control]
        for b in self.biomarkers:
            y = df[b].to_numpy(dtype=float)
            ok = ~np.isnan(y[is_control])
            beta, *_ = np.linalg.lstsq(X_ctrl[ok], y[is_control][ok], rcond=None)
            resid = y - X @ beta
            r_ctrl = resid[is_control][ok]
            sd = float(np.std(r_ctrl, ddof=1))
            if not np.isfinite(sd) or sd == 0:
                raise ValueError(f"Biomarker {b!r} has zero variance in controls; cannot z-score it")
            self.beta[b] = [float(v) for v in beta]
            self.mean[b] = float(np.mean(r_ctrl))
            self.sd[b] = sd
            z = (resid - self.mean[b]) / sd
            self.flip[b], self.direction_source[b] = self._choose_flip(b, z, is_case, direction)
        return self

    @staticmethod
    def _choose_flip(name: str, z: np.ndarray, is_case: np.ndarray, direction) -> tuple[int, str]:
        d = direction.get(name, "auto") if isinstance(direction, dict) else direction
        if d == "increasing":
            return 1, "config"
        if d == "decreasing":
            return -1, "config"
        if is_case.sum() == 0:
            raise ValueError(f"direction: auto for {name!r} needs patients to look at; give an explicit direction")
        return (-1 if np.nanmean(z[is_case]) < 0 else 1), "auto"

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return the (n_subjects, n_biomarkers) z-score matrix, NaN preserved."""
        missing = [c for c in [*self.biomarkers, *self.covariates] if c not in df.columns]
        if missing:
            raise ValueError(f"Columns required by the fitted preprocessor are missing: {missing}")
        X = self._design(df)
        Z = np.empty((len(df), len(self.biomarkers)), dtype=float)
        for j, b in enumerate(self.biomarkers):
            y = df[b].to_numpy(dtype=float)
            z = (y - X @ np.asarray(self.beta[b]) - self.mean[b]) / self.sd[b]
            z = z * self.flip[b]
            if self.clip_z is not None:
                z = np.clip(z, -self.clip_z, self.clip_z)
            Z[:, j] = z
        return Z

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        cols = [np.ones(len(df))] + [df[c].to_numpy(dtype=float) for c in self.covariates]
        return np.column_stack(cols)

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "biomarkers": list(self.biomarkers),
            "covariates": list(self.covariates),
            "clip_z": self.clip_z,
            "per_biomarker": {
                b: {
                    "beta": self.beta[b],
                    "mean": self.mean[b],
                    "sd": self.sd[b],
                    "flip": self.flip[b],
                    "direction_source": self.direction_source[b],
                }
                for b in self.biomarkers
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ZscorePreprocessor":
        self = cls(biomarkers=list(d["biomarkers"]), covariates=list(d["covariates"]), clip_z=d.get("clip_z"))
        for b, p in d["per_biomarker"].items():
            self.beta[b] = list(p["beta"])
            self.mean[b] = float(p["mean"])
            self.sd[b] = float(p["sd"])
            self.flip[b] = int(p["flip"])
            self.direction_source[b] = p.get("direction_source", "config")
        return self

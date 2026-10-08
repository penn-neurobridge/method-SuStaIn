"""Choosing the z-score thresholds (`Z_vals`) and ceiling (`Z_max`) per biomarker.

SuStaIn's z-score model needs, for every biomarker, the z levels that count
as "events" (e.g. 1, 2, 3) and the value reached at the final stage (Z_max).
Taylor et al. 2024 require that more than 5 % of patients exceed the top
threshold in every region; otherwise the last event is never observed and the
ordering becomes arbitrary. The journal-club noise experiment showed the
same thing from the other side: thresholds must track the patient spread.

`fixed` mode takes thresholds from the config and *checks* the rule;
`auto` mode derives them from patient quantiles so the rule holds by design.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ConfigError, ThresholdConfig


@dataclass
class Thresholds:
    Z_vals: np.ndarray  # (B, K), zero-padded
    Z_max: np.ndarray   # (B,)
    fraction_above_top: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return int((self.Z_vals > 0).sum())

    def to_dict(self, biomarkers: list[str]) -> dict:
        return {
            "z_vals": {b: [float(z) for z in self.Z_vals[i] if z > 0] for i, b in enumerate(biomarkers)},
            "z_max": {b: float(self.Z_max[i]) for i, b in enumerate(biomarkers)},
            "fraction_above_top": self.fraction_above_top,
        }


def _per_biomarker(value, biomarkers: list[str], what: str) -> list:
    if isinstance(value, dict):
        missing = [b for b in biomarkers if b not in value]
        if missing:
            raise ConfigError(f"model.thresholds.{what} is a mapping but has no entry for {missing}")
        return [value[b] for b in biomarkers]
    return [value] * len(biomarkers)


def make_thresholds(Z_case: np.ndarray, biomarkers: list[str], tcfg: ThresholdConfig) -> Thresholds:
    """`Z_case` is the (n_patients, B) z-score matrix of non-controls (NaN allowed)."""
    B = len(biomarkers)
    if tcfg.mode == "fixed":
        levels = _per_biomarker(tcfg.z_vals, biomarkers, "z_vals")
        zmax = _per_biomarker(tcfg.z_max, biomarkers, "z_max")
    else:
        if Z_case.shape[0] < 20:
            raise ConfigError("thresholds.mode: auto needs at least 20 patients to estimate quantiles")
        levels, zmax = [], []
        for i in range(B):
            z = Z_case[:, i]
            z = z[~np.isnan(z)]
            top = float(np.quantile(z, 1 - tcfg.min_fraction_above_top))
            if top <= 0:
                raise ConfigError(
                    f"Biomarker {biomarkers[i]!r}: fewer than {tcfg.min_fraction_above_top:.0%} of patients are above "
                    "the control mean, so no abnormal threshold can be placed. Check its direction or drop it."
                )
            lv = [round(top * k / tcfg.n_levels, 2) for k in range(1, tcfg.n_levels + 1)]
            levels.append(lv)
            zmax.append(round(max(float(np.quantile(z, 0.99)), top * (tcfg.n_levels + 1) / tcfg.n_levels), 2))

    K = max(len(lv) for lv in levels)
    Z_vals = np.zeros((B, K))
    for i, lv in enumerate(levels):
        Z_vals[i, : len(lv)] = lv
    Z_max = np.asarray(zmax, dtype=float)

    frac, warnings = {}, []
    for i, b in enumerate(biomarkers):
        z = Z_case[:, i]
        top = levels[i][-1]
        f = float(np.nanmean(z > top)) if len(z) else float("nan")
        frac[b] = round(f, 4)
        if tcfg.mode == "fixed" and f < tcfg.min_fraction_above_top:
            msg = (
                f"Only {f:.1%} of patients exceed the top threshold z={top:g} for {b!r} "
                f"(rule: > {tcfg.min_fraction_above_top:.0%}). Lower z_vals for it or use thresholds.mode: auto."
            )
            if tcfg.strict:
                raise ConfigError(msg)
            warnings.append(msg)
        if Z_max[i] <= top:
            raise ConfigError(f"z_max for {b!r} ({Z_max[i]}) must exceed its top threshold ({top})")
    return Thresholds(Z_vals, Z_max, frac, warnings)

"""Turning pySuStaIn's event indices into something a person can read.

pySuStaIn numbers events by walking the `Z_vals` matrix column by column
(threshold level 1 for every biomarker, then level 2, ...), skipping zero
entries. So with 4 biomarkers A–D and thresholds 1/2/3, event 5 is "B passes
z = 2". For MixtureSustain there is one event per biomarker, so event e is
simply biomarker e.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from results import SubtypeModel


@dataclass
class Event:
    index: int
    biomarker: str
    z: float | None  # None for mixture models
    label: str


def events_for_zscore(biomarker_labels: list[str], Z_vals: np.ndarray) -> list[Event]:
    Z_vals = np.asarray(Z_vals, dtype=float)
    events: list[Event] = []
    for k in range(Z_vals.shape[1]):
        for b, name in enumerate(biomarker_labels):
            z = Z_vals[b, k]
            if z > 0:
                events.append(Event(len(events), name, float(z), f"{name}>{z:g}"))
    return events


def events_for_mixture(biomarker_labels: list[str]) -> list[Event]:
    return [Event(i, name, None, name) for i, name in enumerate(biomarker_labels)]


def describe_sequence(sequence: np.ndarray, events: list[Event], sep: str = " → ") -> str:
    return sep.join(events[int(e)].label for e in sequence)


def sequence_table(model: SubtypeModel, events: list[Event]) -> pd.DataFrame:
    """Long-format table: one row per (subtype, position)."""
    mean_pos, sd_pos = model.mean_position()
    counts = model.counts()
    rows = []
    for s in range(model.n_subtypes):
        order = model.ml_sequence[s].astype(int)
        for position, e in enumerate(order):
            ev = events[e]
            rows.append(
                {
                    "subtype": s + 1,
                    "fraction": float(model.fractions[s]),
                    "n_subjects": int(counts[s]),
                    "position": position + 1,
                    "event": ev.label,
                    "biomarker": ev.biomarker,
                    "z_threshold": ev.z,
                    "mean_position": float(mean_pos[s, e]) + 1,
                    "sd_position": float(sd_pos[s, e]),
                }
            )
    return pd.DataFrame(rows)


def sequences_markdown(model: SubtypeModel, events: list[Event], title: str | None = None) -> str:
    table = sequence_table(model, events)
    out = []
    if title:
        out += [f"# {title}", ""]
    for s in range(model.n_subtypes):
        sub = table[table.subtype == s + 1]
        out.append(f"## Subtype {s + 1} — fraction {model.fractions[s]:.2f} (n = {int(model.counts()[s])})")
        out.append("")
        out.append(describe_sequence(model.ml_sequence[s], events))
        out.append("")
        out.append("| position | event | mean position (MCMC) | SD |")
        out.append("|---:|---|---:|---:|")
        for _, r in sub.iterrows():
            out.append(f"| {r.position} | {r.event} | {r.mean_position:.1f} | {r.sd_position:.1f} |")
        out.append("")
    return "\n".join(out)

"""Writing the human-facing outputs: subjects.csv, sequences.csv/.md, JSON sidecars."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .results import SubtypeModel
from .sequences import Event, sequence_table, sequences_markdown


def subjects_table(model: SubtypeModel, ids: pd.Series, groups: pd.Series | None) -> pd.DataFrame:
    """One row per input subject, same order. Subtypes are 1-based; stage 0 = no events yet."""
    sub = model.ml_subtype
    assigned = ~np.isnan(sub)
    df = pd.DataFrame({"subject_id": ids.to_numpy()})
    if groups is not None:
        df["group"] = groups.to_numpy()
    df["subtype"] = pd.array(np.where(assigned, sub + 1, np.nan), dtype="Int64")
    df["prob_subtype"] = model.prob_ml_subtype.round(4)
    df["stage"] = pd.array(np.where(np.isnan(model.ml_stage), np.nan, model.ml_stage), dtype="Int64")
    df["prob_stage"] = model.prob_ml_stage.round(4)
    df["subtypable"] = (model.ml_stage > 0) & assigned
    for s in range(model.n_subtypes):
        df[f"p_subtype_{s + 1}"] = model.prob_subtype[:, s].round(4)
    for k in range(model.prob_stage.shape[1]):
        df[f"p_stage_{k}"] = model.prob_stage[:, k].round(4)
    return df


def write_model_outputs(model: SubtypeModel, events: list[Event], ids: pd.Series, groups: pd.Series | None, out_dir: Path, title: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects_table(model, ids, groups).to_csv(out_dir / "subjects.csv", index=False)
    sequence_table(model, events).to_csv(out_dir / "sequences.csv", index=False)
    (out_dir / "sequences.md").write_text(sequences_markdown(model, events, title))
    np.savez_compressed(
        out_dir / "posteriors.npz",
        prob_subtype_stage=model.prob_subtype_stage,
        prob_subtype=model.prob_subtype,
        prob_stage=model.prob_stage,
        samples_f=model.samples_f,
        ml_sequence=model.ml_sequence,
        samples_likelihood=model.samples_likelihood,
    )


def write_json(path: Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n")


def write_config_yaml(path: Path, cfg_dict: dict) -> None:
    Path(path).write_text(yaml.safe_dump(cfg_dict, sort_keys=False))


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")

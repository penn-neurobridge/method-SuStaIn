"""Reading the input table and naming every output path in one place."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ConfigError


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Input file not found: {path}")
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)
    if df.empty:
        raise ConfigError(f"Input file has no rows: {path}")
    return df


def apply_missing_policy(df: pd.DataFrame, columns: list[str], policy: str) -> tuple[pd.DataFrame, int]:
    """Handle NaN in the biomarker/covariate columns according to `data.missing`.

    Returns the (possibly shortened) frame and the number of dropped rows.
    """
    has_nan = df[columns].isna().any(axis=1)
    n_bad = int(has_nan.sum())
    if n_bad == 0:
        return df, 0
    if policy == "error":
        cols = df[columns].columns[df[columns].isna().any()].tolist()
        raise ConfigError(
            f"{n_bad} row(s) have missing values in {cols}. "
            "Set data.missing to drop_rows (or model, z-score only) to proceed."
        )
    if policy == "drop_rows":
        return df.loc[~has_nan].reset_index(drop=True), n_bad
    return df, 0  # policy == "model": pySuStaIn's missing-data model handles NaN itself


@dataclass
class RunPaths:
    """Output-folder contract for `sustain run` (see README)."""

    out: Path

    def __post_init__(self):
        self.out = Path(self.out)

    # top level
    @property
    def config(self) -> Path: return self.out / "config.yaml"
    @property
    def run_json(self) -> Path: return self.out / "run.json"
    @property
    def preprocess_json(self) -> Path: return self.out / "preprocess.json"
    @property
    def mixtures_pkl(self) -> Path: return self.out / "mixtures.pkl"
    @property
    def data_preprocessed(self) -> Path: return self.out / "data_preprocessed.csv"
    @property
    def model_selection(self) -> Path: return self.out / "model_selection.csv"
    @property
    def subjects(self) -> Path: return self.out / "subjects.csv"
    @property
    def sequences_csv(self) -> Path: return self.out / "sequences.csv"
    @property
    def sequences_md(self) -> Path: return self.out / "sequences.md"

    # per-S models
    def model_dir(self, n_subtypes: int) -> Path:
        return self.out / "models" / f"S{n_subtypes}"

    # plots
    @property
    def plots(self) -> Path: return self.out / "plots"

    # pySuStaIn's own folder
    @property
    def sustain(self) -> Path: return self.out / "sustain"
    @property
    def pickles(self) -> Path: return self.sustain / "pickle_files"

    def pickle(self, dataset_name: str, n_subtypes: int) -> Path:
        # pySuStaIn numbers pickles from 0: `_subtype0` is the 1-subtype model.
        return self.pickles / f"{dataset_name}_subtype{n_subtypes - 1}.pickle"

    def ensure(self, plots: bool = True) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.sustain.mkdir(parents=True, exist_ok=True)  # pySuStaIn's os.mkdir is non-recursive
        if plots:
            self.plots.mkdir(parents=True, exist_ok=True)

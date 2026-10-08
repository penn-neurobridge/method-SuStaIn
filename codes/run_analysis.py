#!/usr/bin/env python
"""Terminal entry point: CSV in -> SuStaIn -> subtypes, stages and event orderings out.

    uv run python codes/run_analysis.py --help
    uv run python codes/run_analysis.py --input input/synthetic_data.csv --output output/

Everything about the CSV (which column is the subject id, which marks controls,
which are biomarkers, which are covariates) and about the fit lives in a YAML
config; the default `input/config.yaml` matches the synthetic example.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # before anything imports matplotlib

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_analysis.py",
        description="Fit SuStaIn (Subtype and Stage Inference) to a table of biomarkers and write a results folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="input table (.csv or .tsv), one row per subject")
    p.add_argument("--output", type=Path, required=True, help="results folder (created if needed)")
    p.add_argument("--config", type=Path, default=REPO / "input" / "config.yaml",
                   help="YAML describing the columns and the fit settings")
    p.add_argument("--overwrite", action="store_true", help="discard a previous fit found in --output")
    p.add_argument("--no-cv", action="store_true", help="skip cross-validation (no automatic choice of the number of subtypes)")
    p.add_argument("--quiet", action="store_true", help="only print errors")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from config import ConfigError, load_config
    from fitting import CacheMismatch
    from pipeline import run_pipeline

    try:
        cfg = load_config(args.config)
        summary = run_pipeline(cfg, args.input, args.output, overwrite=args.overwrite, skip_cv=args.no_cv,
                               quiet=args.quiet, command=" ".join(sys.argv))
    except (ConfigError, CacheMismatch, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not args.quiet:
        _print_summary(summary)
    return 0


def _print_summary(s) -> None:
    import json

    from sequences import Event, describe_sequence

    m = s.models[s.chosen_n_subtypes - 1]
    labels = json.loads((s.out_dir / "preprocess.json").read_text())["events"]
    events = [Event(i, "", None, lab) for i, lab in enumerate(labels)]
    print()
    print(f"=== {s.chosen_n_subtypes} subtype(s) {'chosen by cross-validation' if s.selection == 'cvic' else '(max requested; CV off)'} ===")
    for k in range(m.n_subtypes):
        print(f"Subtype {k + 1}: fraction {m.fractions[k]:.2f}, n = {int(m.counts()[k])}")
        print(f"    {describe_sequence(m.ml_sequence[k], events)}")
    for w in s.warnings:
        print(f"WARNING: {w}")
    print(f"\nResults: {s.out_dir}/results.csv, sequences.md, model_selection.csv, figures/")


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line: `sustain run`, `sustain stage`, `sustain make-demo`."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # before anything imports matplotlib


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sustain",
        description="CSV in -> SuStaIn (Subtype and Stage Inference) -> subtypes, stages and event orderings out.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="fit SuStaIn to a CSV and write a results folder")
    r.add_argument("csv", type=Path, help="input table (.csv or .tsv), one row per subject")
    r.add_argument("--config", type=Path, required=True, help="YAML config (see demo/config_quick.yaml)")
    r.add_argument("--out", type=Path, required=True, help="results folder (created if needed)")
    r.add_argument("--overwrite", action="store_true", help="discard a previous fit in --out")
    r.add_argument("--no-cv", action="store_true", help="skip cross-validation (no automatic choice of n_subtypes)")
    r.add_argument("--quiet", action="store_true")

    s = sub.add_parser("stage", help="assign subtype + stage to new subjects using a saved fit")
    s.add_argument("csv", type=Path, help="new subjects, same biomarker (and covariate) columns as the fit")
    s.add_argument("--fit", type=Path, required=True, help="results folder written by `sustain run`")
    s.add_argument("--out", type=Path, required=True, help="where to write subjects.csv for the new people")
    s.add_argument("--n-subtypes", type=int, default=None, help="which fitted model to use (default: the chosen one)")
    s.add_argument("--quiet", action="store_true")

    d = sub.add_parser("make-demo", help="regenerate the simulated demo CSVs")
    d.add_argument("--out", type=Path, default=Path("demo"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from .config import ConfigError
    from .fit import CacheMismatch

    try:
        if args.cmd == "run":
            from .config import load_config
            from .pipeline import run_pipeline

            cfg = load_config(args.config)
            summary = run_pipeline(cfg, args.csv, args.out, overwrite=args.overwrite, skip_cv=args.no_cv,
                                   quiet=args.quiet, command=" ".join(sys.argv))
            if not args.quiet:
                _print_summary(summary)
            return 0
        if args.cmd == "stage":
            from .stage import stage_new_subjects

            stage_new_subjects(args.fit, args.csv, args.out, n_subtypes=args.n_subtypes, quiet=args.quiet)
            return 0
        if args.cmd == "make-demo":
            from .simulate import simulate_mixture_cohort, simulate_zscore_cohort

            args.out.mkdir(parents=True, exist_ok=True)
            zc, mc = simulate_zscore_cohort(seed=0), simulate_mixture_cohort(seed=1)
            zc.data.to_csv(args.out / "demo_zscore.csv", index=False)
            zc.truth.to_csv(args.out / "demo_zscore_truth.csv", index=False)
            mc.data.to_csv(args.out / "demo_nocontrols.csv", index=False)
            mc.truth.to_csv(args.out / "demo_nocontrols_truth.csv", index=False)
            print(f"wrote demo CSVs to {args.out}/")
            return 0
    except (ConfigError, CacheMismatch, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 1


def _print_summary(s) -> None:
    from .sequences import describe_sequence

    m = s.models[s.chosen_n_subtypes - 1]
    print()
    print(f"=== {s.chosen_n_subtypes} subtype(s) {'chosen by CVIC' if s.selection == 'cvic' else '(max requested; CV off)'} ===")
    md = (s.out_dir / "preprocess.json")
    events = None
    if md.exists():
        import json

        from .sequences import Event

        labels = json.loads(md.read_text())["events"]
        events = [Event(i, "", None, lab) for i, lab in enumerate(labels)]
    for k in range(m.n_subtypes):
        order = describe_sequence(m.ml_sequence[k], events) if events else " ".join(map(str, m.ml_sequence[k].astype(int)))
        print(f"Subtype {k + 1}: fraction {m.fractions[k]:.2f}, n = {int(m.counts()[k])}")
        print(f"    {order}")
    for w in s.warnings:
        print(f"WARNING: {w}")
    print(f"\nResults: {s.out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())

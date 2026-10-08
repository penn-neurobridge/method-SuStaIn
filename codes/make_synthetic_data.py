#!/usr/bin/env python
"""Regenerate the synthetic CSVs in input/ (deterministic), or score a results folder against the truth.

    uv run python codes/make_synthetic_data.py
    uv run python codes/make_synthetic_data.py --check output/ [--truth input/synthetic_data_truth.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from simulate import match_subtype_labels, simulate_mixture_cohort, simulate_zscore_cohort

INPUT = Path(__file__).resolve().parents[1] / "input"


def write_demo() -> None:
    zc = simulate_zscore_cohort(seed=0)
    zc.data.to_csv(INPUT / "synthetic_data.csv", index=False)
    zc.truth.to_csv(INPUT / "synthetic_data_truth.csv", index=False)
    mc = simulate_mixture_cohort(seed=1)
    mc.data.to_csv(INPUT / "synthetic_nocontrols.csv", index=False)
    mc.truth.to_csv(INPUT / "synthetic_nocontrols_truth.csv", index=False)
    for c, name in ((zc, "synthetic_data"), (mc, "synthetic_nocontrols")):
        print(f"{name}: {len(c.data)} subjects, truth sequences:")
        for s in c.sequences:
            print("   ", " ".join(str(int(e)) for e in s))


def check(results: Path, truth_path: Path) -> int:
    subjects = pd.read_csv(results / "results.csv")
    truth = pd.read_csv(truth_path)
    m = subjects.merge(truth, on="subject_id", how="inner")
    if len(m) != len(truth):
        print(f"WARNING: {len(truth) - len(m)} truth rows not found in results.csv")
    sub_ok = m.subtype.notna() & (m.true_stage > 0) & (m.stage > 0)
    est = m.loc[sub_ok, "subtype"].astype(int).to_numpy() - 1
    tru = m.loc[sub_ok, "true_subtype"].astype(int).to_numpy() - 1
    n_sub = int(subjects.subtype.dropna().max()) if subjects.subtype.notna().any() else 1
    acc, perm = match_subtype_labels(est, tru, max(n_sub, tru.max() + 1))
    stage_err = np.abs(m.stage - m.true_stage)
    print(f"subjects compared        : {len(m)} (subtypable in both: {int(sub_ok.sum())})")
    print(f"subtype accuracy         : {acc:.3f}  (label map est->true: {[int(p) + 1 for p in perm]})")
    print(f"mean |stage error|       : {stage_err.mean():.2f}  (controls: {stage_err[m.true_stage == 0].mean():.2f})")
    print(f"controls placed at stage 0: {(m.loc[m.true_stage == 0, 'stage'] == 0).mean():.2%}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", metavar="RESULTS_DIR", type=Path, help="compare a results folder to the truth")
    ap.add_argument("--truth", type=Path, default=INPUT / "synthetic_data_truth.csv")
    args = ap.parse_args()
    if args.check:
        sys.exit(check(args.check, args.truth))
    write_demo()

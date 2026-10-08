#!/usr/bin/env python
"""Regenerate the demo CSVs (deterministic), or check a results folder against the truth.

    python demo/make_demo_data.py                 # writes demo/demo_*.csv
    python demo/make_demo_data.py --check results/demo [--truth demo/demo_zscore_truth.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sustain_pipeline.simulate import match_subtype_labels, simulate_mixture_cohort, simulate_zscore_cohort  # noqa: E402

HERE = Path(__file__).resolve().parent


def write_demo() -> None:
    zc = simulate_zscore_cohort(seed=0)
    zc.data.to_csv(HERE / "demo_zscore.csv", index=False)
    zc.truth.to_csv(HERE / "demo_zscore_truth.csv", index=False)
    mc = simulate_mixture_cohort(seed=1)
    mc.data.to_csv(HERE / "demo_nocontrols.csv", index=False)
    mc.truth.to_csv(HERE / "demo_nocontrols_truth.csv", index=False)
    for c, name in ((zc, "demo_zscore"), (mc, "demo_nocontrols")):
        print(f"{name}: {len(c.data)} subjects, truth sequences:")
        for s in c.sequences:
            print("   ", " ".join(str(int(e)) for e in s))


def check(results: Path, truth_path: Path) -> int:
    subjects = pd.read_csv(results / "subjects.csv")
    truth = pd.read_csv(truth_path)
    m = subjects.merge(truth, on="subject_id", how="inner")
    if len(m) != len(truth):
        print(f"WARNING: {len(truth) - len(m)} truth rows not found in subjects.csv")
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
    ap.add_argument("--truth", type=Path, default=HERE / "demo_zscore_truth.csv")
    args = ap.parse_args()
    if args.check:
        sys.exit(check(args.check, args.truth))
    write_demo()

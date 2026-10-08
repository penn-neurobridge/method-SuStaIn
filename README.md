# method-SuStaIn

**CSV in → SuStaIn → subtypes, stages and event orderings out.**

SuStaIn (Subtype and Stage Inference; Young et al. 2018) takes one snapshot per
person of several biomarkers and recovers, from the cross-sectional spread alone,
(1) a small number of *subtypes*, each with its own ordering in which the
biomarkers become abnormal, and (2) a *stage* for every person along their
subtype's ordering. This repo wraps the reference implementation
([ucl-pond/pySuStaIn](https://github.com/ucl-pond/pySuStaIn)) so that running it
on a new table is a config edit, not a new notebook. The method and the
preprocessing follow Taylor et al. 2024 (*Brain* 148:955–968).

## Run it

```bash
uv sync
uv run python codes/run_analysis.py --help
uv run python codes/run_analysis.py --input input/synthetic_data.csv --output output/
```

Needs Python 3.10–3.12 (uv fetches it), a C++ compiler for one small dependency
(`xcode-select --install` on macOS), and ~2–3 minutes for the synthetic example.
The results in `output/` were produced by exactly that command; rerunning it reuses
the saved fit in seconds (pass `--overwrite` to refit).

## What the example is

`input/synthetic_data.csv` is 400 simulated people: 80 healthy controls and 320
patients drawn from **two** hidden subtypes (60 % / 40 %) with mirror-image
progressions over four regional measures A–D:

| subtype | true ordering (threshold 1 of each region, then 2, then 3, interleaved) |
|---|---|
| 1 | A → A → B → A → B → C → B → C → D → C → D → D |
| 2 | D → D → C → D → C → B → C → B → A → B → A → A |

The generator (`codes/simulate.py`) uses pySuStaIn's own z-score model, then
dresses the values up as raw measurements: an age effect, a scale/offset, and
region D made to *decrease* with disease, so the pipeline has to regress out age,
z-score against controls and flip D's sign before anything else works. The
truth is in `input/synthetic_data_truth.csv`; `uv run python
codes/make_synthetic_data.py --check output/` scores a results folder against it.

### Input columns

| column | type | meaning (synthetic units are arbitrary) |
|---|---|---|
| `subject_id` | string | unique id, carried through to the results |
| `group` | `control` / `patient` | who defines "normal" |
| `age` | years | covariate, regressed out (fitted on controls) |
| `A`, `B`, `C`, `D` | float | regional measures; A–C increase with disease, D decreases (e.g. a volume) |

Any table with an id column, a group column and numeric biomarker columns works;
`input/config.yaml` says which is which.

## Expected outputs (`output/`)

```
results.csv            one row per input subject: subtype (1-based), stage (0 = no events yet),
                       posterior probabilities, and the full p(subtype), p(stage) columns
sequences.md / .csv    each subtype's ordering of events ("A>1 → A>2 → B>1 …"), its fraction,
                       and each event's mean ± SD position over the MCMC samples
model_selection.csv    CVIC per number of subtypes (lower is better); `chosen` marks the winner
figures/
  pvd_S{1,2,3}.png     positional variance diagrams, one per fitted model
  cvic.png             CVIC and per-fold log-likelihood vs number of subtypes
  biomarkers_after_preprocess.png   controls vs patients per region, with the z thresholds
  mcmc_trace.png       log-likelihood trace per model
models/S1..S3/         the same results/sequences files for every fitted model, plus posteriors.npz
preprocess.json        everything needed to score new people: covariate betas, control mean/SD,
                       sign flips, thresholds, event labels
run.json               command, timings, versions, input fingerprint, warnings
data_preprocessed.csv  the z-score matrix that was actually fed to SuStaIn (first thing to look at
                       when a real-data run looks wrong)
sustain/               pySuStaIn's own pickles (git-ignored; regenerate by running the command)
```

### What the synthetic run found

| | found | truth |
|---|---|---|
| number of subtypes | **2** (CVIC 6448.7 / **5842.2** / 5848.4 for 1 / 2 / 3 subtypes) | 2 |
| subtype fractions | 0.56 / 0.44 | 0.60 / 0.40 |
| subtype 1 ordering | B>1 → A>1 → A>2 → A>3 → B>2 → C>1 → B>3 → C>2 → C>3 → D>1 → D>2 → D>3 | A>1 → A>2 → B>1 → A>3 → B>2 → C>1 → B>3 → C>2 → D>1 → C>3 → D>2 → D>3 |
| subtype 2 ordering | D>1 → D>2 → C>1 → D>3 → B>1 → C>2 → C>3 → B>2 → B>3 → A>1 → A>2 → A>3 | D>1 → D>2 → C>1 → D>3 → C>2 → B>1 → C>3 → B>2 → A>1 → B>3 → A>2 → A>3 |
| per-person subtype accuracy | 96.1 % of the 308 people who are past stage 0 in both truth and fit | |
| mean stage error | 0.89 of 12 stages | |
| wall time | 131 s (51 s fit + 79 s cross-validation) | |

The orderings differ from the truth only by adjacent swaps, which the positional
variance diagram (`figures/pvd_S2.png`) shows as blurred cells. `A>1` reads
"region A passes z = 1". Three start points and 5 000 MCMC iterations is a quick
profile; the paper settings (`input/config_full.yaml`) sharpen the diagrams but
take hours.

Subtype labels are arbitrary between runs (label switching), so "subtype 1" in
one run may be "subtype 2" in another. People at stage 0 have no events yet and
cannot be assigned a subtype; `results.csv` marks them `subtypable = False`.

## From the example to real data

Copy `input/config.yaml`, edit the `data:` block, and point `--config` at it:

```yaml
data:
  id_column: subject_id
  group_column: group          # null -> no groups (see "no controls" below)
  control_label: control
  biomarker_columns: auto      # or an explicit list
  covariate_columns: [age]     # e.g. [age, sex, icv]; must be numeric (0/1 for categories)
  missing: error               # error | drop_rows | model
preprocess:
  direction: auto              # or {hippocampus: decreasing, ...}
model:
  thresholds: {mode: fixed, z_vals: [1, 2, 3], z_max: 5}   # or mode: auto
fit:
  n_startpoints: 3             # papers use 25
  n_mcmc: 5000                 # papers use 100000 -> see input/config_full.yaml (hours)
cv:
  n_folds: 2                   # papers use 10
```

Things the pipeline checks for you, because they bit us in the toy runs:

- **Thresholds must match the patient spread.** Taylor et al. require > 5 % of
  patients above the top threshold in every region. `fixed` mode warns when that
  fails (`strict: true` makes it an error); `auto` mode derives thresholds from
  patient quantiles so it holds by construction.
- **Sign flips travel with the model.** `preprocess.json` stores covariate betas,
  control mean/SD and the flip together, so `stage_new_subjects.py` can never
  z-score new people without flipping them.
- **Stale results can't masquerade as fresh ones.** pySuStaIn silently reloads
  any pickle it finds; this pipeline fingerprints the data and settings and
  refuses to reuse a fit from different inputs unless you pass `--overwrite`.

### No control group

```bash
uv run python codes/run_analysis.py --input input/synthetic_nocontrols.csv \
    --config input/config_nocontrols.yaml --output output/nocontrols_example/
```

With no controls there is nothing to z-score against. pySuStaIn's
`MixtureSustain` is then driven by a 2-component Gaussian mixture per biomarker
fitted *without* labels (`model.type: mixture_unsupervised`, experimental;
requires an explicit `direction` per biomarker). Note that kde_ebm's own mixture
models (`mixture_gmm`, `mixture_kde`) still need control/patient labels — they
are an alternative to z-thresholds, not to controls.

### Scoring new people with a saved fit

```bash
uv run python codes/stage_new_subjects.py --fit output/ --input new_people.csv --output output/new_people/
```

## Code map (`codes/`)

| file | role |
|---|---|
| `run_analysis.py` | **terminal entry point** (argparse) |
| `pipeline.py` | the whole run as one function, `run_pipeline()` — what a web UI would call |
| `config.py` | YAML → validated settings; column resolution |
| `preprocessing.py` | covariate regression, z-scoring vs controls, sign flips, clipping (round-trips to JSON) |
| `thresholds.py` | `Z_vals` / `Z_max` per biomarker, fixed or data-driven, with the 5 % check |
| `features_mixture.py` | per-biomarker mixture models → `L_yes`/`L_no` for the no-controls path |
| `models.py`, `fitting.py` | build the pySuStaIn object; fit, cross-validate, cache by fingerprint |
| `results.py`, `sequences.py` | load pickles in a consistent subtype order; event labels and orderings |
| `model_selection.py`, `reporting.py`, `plotting.py` | CVIC choice; CSV/MD/JSON writers; figures |
| `stage_new_subjects.py` | score new people with a saved fit |
| `simulate.py`, `make_synthetic_data.py` | the synthetic cohorts and the truth checker |

Tests: `uv run pytest` (fast, ~1 s) and `uv run pytest -m slow` (end-to-end on the
synthetic data with tiny MCMC settings, ~1 min).

## Runtime

Measured on an M-series MacBook, 400 subjects × 4 biomarkers, 3 subtypes max:

| profile | start points | MCMC | CV folds | wall time |
|---|---|---|---|---|
| `input/config.yaml` (this example) | 3 | 5 000 | 2 | 2 min 11 s |
| journal-club notebook | 5 | 10 000 | 2 | ~11 min |
| `input/config_full.yaml` (paper settings) | 25 | 100 000 | 10 | hours — run overnight |

Fit time scales roughly with start points × MCMC iterations × number of subtypes,
and cross-validation multiplies that by the number of folds.

## References

- Young AL et al. (2018). Uncovering the heterogeneity and temporal complexity of
  neurodegenerative diseases with Subtype and Stage Inference. *Nature
  Communications* 9:4273.
- Taylor NL et al. (2024). Data-driven neuroanatomical subtypes of primary
  progressive aphasia. *Brain* 148:955–968.
- Aksman LM et al. (2021). pySuStaIn: a Python implementation of the Subtype and
  Stage Inference algorithm. *SoftwareX* 15:100811.

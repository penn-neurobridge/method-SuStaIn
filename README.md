# method-SuStaIn

A reusable, beginner-friendly implementation of **SuStaIn** (Subtype and Staging Inference) for the Penn NeuroBridge Lab.

SuStaIn is a statistical method that looks at a group of patients and simultaneously asks:
1. **Are there distinct subtypes?** — groups of patients where the same biomarkers go wrong, but in a different order?
2. **How far along is each patient?** — their *disease stage* within their subtype.

It does this from a single cross-sectional snapshot (no need to follow people over time).

---

## What's in this example

We simulate a toy neurodegenerative dataset with **200 subjects** and **6 brain biomarkers** (z-scores relative to healthy controls). Two hidden subtypes are embedded in the data:

| Subtype | Progression order |
|---------|-------------------|
| **Temporal-first** | Hippocampus → Entorhinal Ctx → Amygdala → Fusiform → Precuneus → Post. Cingulate |
| **Posterior-first** | Precuneus → Post. Cingulate → Fusiform → Entorhinal Ctx → Amygdala → Hippocampus |

The goal is to see whether SuStaIn can recover these hidden patterns.

---

## Input data format

**File:** `input/synthetic_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | string | Unique ID for each person |
| `group` | string | `control` or `patient` |
| `true_subtype` | int | Ground-truth subtype (0=control, 1 or 2=patient subtype) |
| `true_stage` | int | Ground-truth disease stage (0–6) |
| `hippocampus_zscore` | float | Z-score vs. healthy mean (unitless) |
| `entorhinal_zscore` | float | Z-score vs. healthy mean (unitless) |
| `amygdala_zscore` | float | Z-score vs. healthy mean (unitless) |
| `fusiform_zscore` | float | Z-score vs. healthy mean (unitless) |
| `precuneus_zscore` | float | Z-score vs. healthy mean (unitless) |
| `posterior_cingulate_zscore` | float | Z-score vs. healthy mean (unitless) |

> **What is a z-score?** It measures how far a value is from the healthy average, in units of standard deviation. A z-score of 0 = perfectly average; z-score of 2 = two standard deviations above healthy average (i.e., quite abnormal for an atrophy biomarker).

---

## Expected outputs

After running the analysis you will find:

```
output/
├── results.csv                      ← Per-subject subtype & stage assignments
├── model_metadata.json              ← Model settings (for reproducibility)
└── figures/
    ├── pvd_subtype_1.png            ← Positional variance diagram, Subtype 1
    ├── pvd_subtype_2.png            ← Positional variance diagram, Subtype 2
    ├── staging_histogram.png        ← Stage distribution per subtype
    └── subtype_assignments.png      ← How many subjects per subtype
```

**results.csv columns:**

| Column | Meaning |
|--------|---------|
| `subject_id` | Subject identifier |
| `assigned_subtype` | Model's subtype assignment (1 or 2) |
| `assigned_stage` | Model's stage assignment (0 = healthy-like) |

---

## Setup

### Step 1 — Install `uv` (once, on your machine)

`uv` is a fast Python package manager. Install it from your terminal:

```bash
# Mac/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal afterwards.

### Step 2 — Clone this repository

```bash
git clone https://github.com/penn-neurobridge/method-SuStaIn.git
cd method-SuStaIn
git checkout dev-mukki
```

### Step 3 — Install dependencies

This installs everything listed in `pyproject.toml` into an isolated environment:

```bash
uv sync
```

### Step 4 — Install pySuStaIn (the SuStaIn model itself)

pySuStaIn is not on PyPI, so install it separately from GitHub:

```bash
pip install "git+https://github.com/ucl-pond/pySuStaIn.git"
```

> **Note:** Without pySuStaIn the pipeline still runs in **demo mode** — it produces all the same output files, but with random (non-meaningful) subtype/stage assignments. This is useful for checking the pipeline works on your machine before installing pySuStaIn.

---

## Running the analysis

### See all options

```bash
uv run python codes/run_analysis.py --help
```

### Run with defaults (recommended first run)

```bash
uv run python codes/run_analysis.py \
    --input input/synthetic_data.csv \
    --output output/
```

### Full example with all options specified

```bash
uv run python codes/run_analysis.py \
    --input  input/synthetic_data.csv \
    --output output/ \
    --n-subtypes    2 \
    --n-iterations  100000 \
    --n-startpoints 25 \
    --seed          42
```

| Flag | What it does | Default |
|------|-------------|---------|
| `--input` | Path to your CSV | *(required)* |
| `--output` | Where to save results | `output/` |
| `--n-subtypes` | How many subtypes to fit | `2` |
| `--n-iterations` | MCMC iterations (more = slower but better) | `50000` |
| `--n-startpoints` | Optimiser restarts (more = more robust) | `10` |
| `--seed` | Random seed (keep fixed for reproducibility) | `42` |

---

## Project structure

```
method-SuStaIn/
├── README.md                  ← You are here
├── pyproject.toml             ← Dependency list (managed by uv)
├── uv.lock                    ← Exact pinned versions (committed for reproducibility)
│
├── input/
│   └── synthetic_data.csv     ← Toy dataset (200 subjects, 6 biomarkers, 2 hidden subtypes)
│
├── codes/
│   ├── run_analysis.py        ← 🚀 ENTRY POINT — start here
│   ├── preprocessing.py       ← Loads and validates the CSV
│   └── plotting.py            ← Makes all the figures
│
└── output/                    ← Generated by running the pipeline
    ├── results.csv
    ├── model_metadata.json
    └── figures/
```

---

## How to read the outputs

### Positional Variance Diagram (PVD)

The PVD is the main SuStaIn output. Each row is a biomarker; each column is a position in the disease progression sequence. The colour shows the *probability* that a biomarker occupies that position.

- A dark block in the top-left means: "this biomarker is almost always the first to become abnormal in this subtype."
- A diagonal pattern means SuStaIn is confident about the ordering.
- A diffuse/noisy pattern means SuStaIn is uncertain — you may need more data or fewer subtypes.

### Staging histogram

Shows how many subjects are at each disease stage. Stage 0 = no abnormality detected; Stage 6 = all 6 biomarkers abnormal.

---

## Adding your own data

Replace `input/synthetic_data.csv` with your own CSV. The only requirement is that it has these columns (one per biomarker, as z-scores):

```
subject_id, hippocampus_zscore, entorhinal_zscore, amygdala_zscore,
fusiform_zscore, precuneus_zscore, posterior_cingulate_zscore
```

To add or remove biomarkers, edit the `BIOMARKER_COLUMNS` list at the top of `codes/preprocessing.py`.

---

## Troubleshooting

**"command not found: uv"** — restart your terminal after installing uv.

**"ModuleNotFoundError: No module named 'typer'"** — run `uv sync` first.

**Pipeline runs but figures look random** — pySuStaIn is not installed. Follow Step 4 above.

**Results differ between runs** — make sure `--seed` is set to the same value.

---

*Penn NeuroBridge Lab · method-SuStaIn · branch: dev-mukki*

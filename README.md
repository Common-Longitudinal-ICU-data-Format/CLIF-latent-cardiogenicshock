# CLIF shock cohort — site run

## What the script does

Where do shock patients sit, and do the published subphenotypes correspond to
anything measurable? The cohort is **undifferentiated shock**, selected on
physiology alone, with the coded mechanism attached afterwards as a labelling
rather than a filter — so the question of whether septic, cardiogenic and
hypovolaemic shock separate stays answerable.

Each site builds its own cohort from its own records using the identical
definition. Raw CLIF records never move.

The script reads your CLIF-2.1 tables and writes summary tables covering:

- **patient flow** — how many hospitalizations survive each inclusion criterion
- **cohort composition** — the Zweck, Seymour and Bhavani phenotype
  breakdowns, and the coded shock types
- **baseline characteristics** — median physiology and shares by coded
  mechanism
- **trial and guideline fit** — eighteen randomised trials across the
  cardiogenic, septic and hypovolaemic literatures, scored for eligibility,
  position against each published interquartile range, whether the recommended
  treatment was delivered, which single guideline covers each patient, and how
  many trials cover them at all
- **charting practice** — per-token prevalence, which is how we tell a real
  population difference from a documentation difference

---

## What you need

| | |
|---|---|
| Your CLIF-2.1 extract | 14 parquet tables, listed below |
| Python | 3.11 or newer |
| Disk | roughly 40× the size of your cohort's raw tables, as working space |
| Time | a few hours of CPU for a large cohort |

The 14 tables the script reads:

```
clif_hospitalization   clif_patient              clif_adt
clif_labs              clif_vitals               clif_respiratory_support
clif_medication_admin_continuous                 clif_patient_assessments
clif_medication_admin_intermittent               clif_patient_procedures
clif_position          clif_code_status          clif_crrt_therapy
clif_hospital_diagnosis
```

Four of them change what gets reported rather than stopping the run if they are
present but empty — `clif_hospital_diagnosis`, `clif_patient_procedures`,
`clif_crrt_therapy`, `clif_code_status`. `--check` names them if they are
missing, and says what each one costs.

---

## Setup

### 1. Get the code

```bash
git clone <repo>
cd CLIF-latent-cardiogenicshock
```

### 2. Find a Python 3.11 or newer

```bash
python3 --version          # Windows: py --version
```

**It has to be 3.11 or newer.** On a Mac, `python3` is usually Apple's 3.9,
which is too old even when a newer Python is installed alongside it. Check for
one by version number:

```bash
python3.12 --version       # also try python3.11, python3.13. Windows: py -3.12 --version
```

If none is installed, get one from [python.org](https://www.python.org/downloads/).
Without admin rights, [uv](https://docs.astral.sh/uv/) can do step 3 for you and
fetch the Python itself: `uv venv --seed --python 3.12 ~/venvs/clif-cs`.

### 3. Create a virtual environment and install

Build the environment with the interpreter you found in step 2. An environment
keeps the Python it was created with, so this is the step that fixes the
version.

macOS / Linux:

```bash
python3.12 -m venv ~/venvs/clif-cs     # the interpreter from step 2
source ~/venvs/clif-cs/bin/activate
python --version                       # must say 3.11 or newer
pip install -r requirements-site.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv $HOME\venvs\clif-cs    # the version from step 2
$HOME\venvs\clif-cs\Scripts\Activate.ps1
python --version
pip install -r requirements-site.txt
```

### 4. Point it at your data

Create `code/config/paths.yaml` containing four lines:

```yaml
role: site
sites:
  YOURSITE:
    data_dir: /the/path/to/your/CLIF_2_1_directory
```

`data_dir` is the folder that directly holds the `clif_*.parquet` files.
`YOURSITE` is the label that appears in every table and in the name of the
directory you send, so use something recognisable. On Windows, write the path
with forward slashes: `C:/data/CLIF_2_1`.

**That is the whole file for most sites.** Everything else is optional.
[`code/config/paths.example.yaml`](code/config/paths.example.yaml) explains each
setting if you want to copy one across:

| setting | what it is | if you leave it out |
|---|---|---|
| `output_home` | where results are written | `./output` |
| `scratch_home` | working space, large and deletable afterwards. Point it at a fast local disk, never a synced drive (Box, Dropbox, OneDrive) | `./scratch` |
| `tokenizer_yaml` | the tokenizer the model was trained with. Ask the coordinating site for it; one you build locally is not a substitute | `tokenize`, `winnow` and `tokencounts` are skipped and the bundle carries no token tables. Everything else is unaffected |

`paths.yaml` is gitignored and never leaves your machine.

### 5. Check before you run

```bash
python code/01_site_tables.py --check
```

This reads your configuration, confirms each path resolves, and lists which of
the 14 tables it found. **It reads table schemas only, never rows**, takes about
a second, and is safe to run anywhere. Fix anything it marks `FAIL` before going
further. It also warns about timestamp columns stored without a timezone, which
are read as UTC: if yours hold local time, convert them to UTC first.

---

## Run it

```bash
python code/01_site_tables.py
```

Ten stages, in order. Each one announces itself when it starts and reports how
long it took, so a long run is readable as it goes:

```
=== [1/10] cohort: identifying the shock cohort and attaching labels
=== [1/10] cohort finished in 94s
=== [2/10] subset: cutting the CLIF tables down to the cohort
...
```

| stage | |
|---|---|
| `cohort` | identify shock, print the filter ledger, attach outcomes, coded shock type and the three phenotype systems |
| `subset` | cut your CLIF tables down to the cohort |
| `derive` | respiratory-support waterfall, dose conversion, SOFA — **slow, and the most memory-hungry** |
| `collate` | assemble one timeline per patient — **slow** |
| `tokenize` | encode each timeline |
| `winnow` | cut each timeline at the 24h and onset anchors |
| `tokencounts` | per-token and per-family prevalence |
| `baseline` | median physiology and shares by coded mechanism |
| `trials` | eligibility, position against published IQRs, guideline fit |
| `export` | assemble the directory to send |

Everything is written as it goes, so nothing is lost if the run stops. Each
stage writes its summary tables straight into `output/upload_to_box_<SITE>/`,
already suppressed, so they exist in one place only; `output/cohort/` keeps the
patient-level parquet files.

## If a stage fails

The run stops at the stage that failed and prints a block naming it:

```
======================================================================
STAGE FAILED: collate -- assembling one timeline per patient (slow)

  FileNotFoundError: [Errno 2] No such file or directory: '...'
  raised at   code/clifcs/clif.py:214 in load_table()
  look at     also memory-hungry, and it reads what `derive` wrote: if
              `derive` half-finished, rerun it with --stage derive --force first.

  resume from this stage once it is fixed:
      python code/01_site_tables.py --from-stage collate

  earlier stages are not redone unless you pass --force.
======================================================================
```

`raised at` is the line in this repository that failed, not a line inside a
library. `look at` is where the *cause* usually is, which is often somewhere
else. The full traceback follows the block.

**You do not need to start over.** Fix the problem and resume:

```bash
python code/01_site_tables.py --from-stage collate   # continue from here
python code/01_site_tables.py --stage derive --force # redo just one stage
```

Everything is also written to `output/logs/01_site_tables.log`. If you get
stuck, send us that file — it has the stage timings and the full traceback.

**If the process disappears without an error**, it ran out of memory. That
happens in `derive` or `collate` on a large cohort. Give it more memory, or run
one site at a time with `--site YOURSITE`.

---

## What to send

**`output/upload_to_box_<SITE>/`** — that one directory, nothing else.

```
output/
  upload_to_box_<SITE>/        <-- SEND THIS
    manifest.json                site, code version, minimum cell size
    token_levels.csv             the token legend (only with tokenizer_yaml)
    counts/
      filter_ledger.csv          patient flow and attrition
      phenotype_summary.csv      Zweck phenotype breakdown
      shock_type_summary.csv     septic / cardiogenic / hypovolaemic / mixed
      seymour_summary.csv        Seymour sepsis phenotypes
      bhavani_summary.csv        Bhavani sepsis phenotypes
      seymour_availability.csv   which features your site charts
      baseline_by_shock_type.csv medians and shares by coded mechanism
      phenotype_summary_cs_phenotype.csv  the Zweck breakdown under its variant name
      seymour_centroids.csv      Seymour cluster profiles, when refit at this site
      bhavani_centroids.csv      Bhavani cluster profiles, when refit at this site
      seymour_agreement.csv      refit vs published Seymour labels, when refit here
      token_prevalence_anchor-*  per-token prevalence (only with tokenizer_yaml)
      token_families_anchor-*    the same, rolled up to family (only with tokenizer_yaml)
    models/
      seymour_model.json         the Seymour model fitted here, when refit at this site
      bhavani_model.json         the Bhavani model fitted here, when refit at this site
    trials/
      trial_criteria.csv         the criteria as encoded. No patient data
      trial_iqr_position_*       your medians against the published quartiles
      trial_outcomes_*           per trial: who qualifies and how they did
      trial_guideline_assignment_*  which guideline covers each patient
      trial_coverage_*           how many trials cover each patient

  patient_level_data_<SITE>/   LOCAL ONLY -- never shared
  cohort/                      patient-level parquet files. LOCAL ONLY
  logs/  scratch/              working files. Not sent
```

The directories are named so that what may leave is obvious without reading any
code. Every column permitted to cross a site boundary is listed, with the
reason, in
[`code/clifcs/export/contract.py`](code/clifcs/export/contract.py); anything not
listed is refused at both ends. Print it:

```bash
python code/clifcs/export/contract.py
```

Suppression is applied twice — once where each table is built, then re-checked
at export, so a table assembled by any other route cannot skip it. The
threshold is 11 patients. If your institution requires a larger one:

```bash
python code/01_site_tables.py --min-cell 20
```

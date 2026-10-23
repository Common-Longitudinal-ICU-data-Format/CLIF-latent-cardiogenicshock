# CLIF shock cohort — site run

> ## What we are asking for
>
> Check out this repository, run **one script**, and send back **one
> directory**.
>
> ```bash
> python code/01_site_tables.py --check    # verify your setup, runs nothing
> python code/01_site_tables.py            # the run
> ```
>
> Then send `output/upload_to_box_<SITE>/`.
>
> It runs as a plain Python program on one machine — laptop, workstation or
> cluster, Windows, macOS or Linux. Everything in the directory you send is a
> count, a share or a median over at least 11 patients; no patient-level data
> leaves your site.

---

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

### 2. Check your Python

```bash
python3 --version          # Windows: py --version
```

**It has to be 3.11 or newer.**

### 3. Create a virtual environment and install

macOS / Linux:

```bash
python3 -m venv ~/venvs/clif-cs
source ~/venvs/clif-cs/bin/activate
python --version                       # confirm 3.11+ -- see the note below
pip install -r requirements-site.txt
```

Windows (PowerShell):

```powershell
py -m venv $HOME\venvs\clif-cs
$HOME\venvs\clif-cs\Scripts\Activate.ps1
python --version
pip install -r requirements-site.txt
```

### 4. Point it at your data

```bash
cp code/config/paths.example.yaml code/config/paths.yaml
```

Edit `code/config/paths.yaml`. **One setting is required.** The shortest
working file is four lines:

```yaml
role: site
sites:
  YOURSITE:
    data_dir: /the/path/to/your/CLIF_2_1_directory
```

`YOURSITE` is the label that appears in every table and in the name of the
directory you send, so use something recognisable. On Windows, write the path
with forward slashes: `C:/data/CLIF_2_1`.

Two optional settings, both defaulting to the directory you run from:

| | |
|---|---|
| `output_home` | where results are written. Defaults to `./output` |
| `scratch_home` | working space. Defaults to `./scratch`. Large and deletable afterwards, so point it at a fast local disk if you have one |

`paths.yaml` is gitignored and never leaves your machine.

### 5. Check before you run

```bash
python code/01_site_tables.py --check
```

This reads your configuration, confirms each path resolves, and lists which of
the 14 tables it found. **It opens no data file**, takes about a second, and is
safe to run anywhere. Fix anything it marks `FAIL` before going further.

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

Everything is written as it goes, so nothing is lost if the run stops.

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
    token_levels.csv             the token legend
    counts/
      filter_ledger.csv          patient flow and attrition
      phenotype_summary.csv      Zweck phenotype breakdown
      shock_type_summary.csv     septic / cardiogenic / hypovolaemic / mixed
      seymour_summary.csv        Seymour sepsis phenotypes
      bhavani_summary.csv        Bhavani sepsis phenotypes
      seymour_availability.csv   which features your site charts
      baseline_by_shock_type.csv medians and shares by coded mechanism
      token_prevalence_anchor-*  per-token prevalence
      token_families_anchor-*    the same, rolled up to family
    trials/
      trial_criteria.csv         the criteria as encoded. No patient data
      trial_iqr_position_*       your medians against the published quartiles
      trial_outcomes_*           per trial: who qualifies and how they did
      trial_guideline_assignment_*  which guideline covers each patient
      trial_coverage_*           how many trials cover each patient

  patient_level_data_<SITE>/   LOCAL ONLY -- never shared
  cohort/  logs/  scratch/     working files. Not sent
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

# CLIF latent space — shock

> ## 📣 Run request — one script
>
> **Run [`code/01_site_tables.py`](code/01_site_tables.py). Send back
> `output/upload_to_box_<SITE>/`.** That is the whole ask.
>
> ```bash
> pip install -r requirements-site.txt
> cp code/config/paths.example.yaml code/config/paths.yaml   # then edit it
> python code/01_site_tables.py
> ```
>
> **No GPU. No model weights. No patient-level data leaves your site.**
> Everything in the bundle is a count, a share or a median over at least 11
> patients. Budget a few hours of CPU on a large cohort, and about 200 MB of
> install.
>
> Nothing else in this repository is being asked of contributing sites right
> now. Scripts `02`–`13` are the patient-level and coordinating-site stages;
> they are documented in
> [docs/TIER_2_AND_POOLED.md](docs/TIER_2_AND_POOLED.md) and are not part of
> this request.
>
> Questions, or anything in `filter_ledger.csv` that looks wrong: raise it with
> the coordinating site **before** sending.

---

## What this is

Where do shock patients sit in the embedding space of a frozen CLIF foundation
model, and do the published subphenotypes correspond to anything the model sees?

The cohort is **undifferentiated shock**, selected on physiology alone, with the
coded mechanism attached afterwards as a labelling rather than a filter — so the
question of whether septic, cardiogenic and hypovolaemic shock separate in the
latent space stays answerable.

Runs **federated**: raw CLIF records never move. **Each site builds its own
cohort** from its own records using the identical definition; nothing is pooled
until the coordinating site combines the bundles that arrive.

Script `01` answers a real question on its own, with no embedding involved: how
differently do the sites chart, how does attrition differ, and do the published
trial populations and subphenotypes look the same at each hospital. It is also
the input the site effect is decomposed with — and site, at 97% neighbour
purity, is the largest single finding here so far.

**The whole trial and guideline analysis is in tier 1.** Eighteen trials across
the cardiogenic, septic and hypovolaemic literatures — including IABP-SHOCK II,
ECLS-SHOCK and DanGer Shock — scored for eligibility, position against each
published interquartile range, treatment concordance, which single guideline
covers each patient, and how many trials cover them at all. None of it reads an
embedding, so none of it needs a patient-level agreement. Every table is broken
down by Zweck phenotype, both sepsis phenotype systems, and coded shock type as
well as overall.

Sites are configured in `code/config/paths.yaml`; adding one needs no code
change.

---

## The run request, in full

### 1. Install — tier 1 only

Python **3.11 or newer**. One install, no torch, no CUDA, no GPU:

```bash
git clone <repo> && cd CLIF-latent-cardiogenicshock
python3 --version                     # must say 3.11+; see below if it does not
python3 -m venv ~/venvs/clif-cs       # NOT on Box, Dropbox or OneDrive
source ~/venvs/clif-cs/bin/activate   # Windows: ~/venvs/clif-cs/Scripts/activate
python --version                      # confirm 3.11+ before installing
pip install -r requirements-site.txt  # ~200 MB
```

Do **not** install `requirements.txt` or `requirements-embed.txt`. Those pull
torch, transformers, lightgbm, optuna, xgboost and wandb — about 15 GB — and
`01` never loads a model.

### 2. Configure

```bash
cp code/config/paths.example.yaml code/config/paths.yaml
```

Four keys, and nothing else is needed for this run:

| key | what to put |
|---|---|
| `sites.<NAME>.data_dir` | your CLIF 2.1 parquet directory. `<NAME>` is the label that appears in every table and in the bundle directory name |
| `tokenizer_yaml` | `GEM-weights-FL-manuscript/tokenizer.yaml` — **required**, see below |
| `scratch_home` | fast local scratch. Intermediates are large and fully regenerable; never a synced drive |
| `output_home` | where the bundle is written. `./output` is fine |

`models.<key>.home_glob` is **tier 2 only** and may point at nothing — `01`
reads the tokenizer, which is a different file from the weights. `paths.yaml` is
gitignored and never travels.

**The tokenizer has to be the one the model was trained with.** The token
prevalence tables in the bundle are token ids, and the numeric bin edges are
baked into the token names, so a locally fitted tokenizer maps the same clinical
event to a different id and the tables stop being comparable across sites.

Check the configuration before running anything. This opens no CLIF table and is
the cheapest way to catch a mistake:

```bash
bash hpc/00_preflight.sh
```

It verifies that every path resolves and that each site has the 14 tables the
pipeline reads. An unresolved `model` line is expected here and does not fail it.

### 3. Run

```bash
python code/01_site_tables.py
```

No flags needed. The three phenotype models in [`models/`](models/) are applied
by default, which is what a contributing site wants: they were fitted once on
the derivation cohort and transferred, so "phenotype II", "beta" and "group A"
mean the same thing everywhere. Refitting is still possible but has to be asked
for (`--refit-phenotypes`), and is only correct at the derivation site.

The run is staged and **resumable**. If it dies, restart from where it stopped
rather than from the beginning:

```bash
python code/01_site_tables.py --from-stage collate     # resume
python code/01_site_tables.py --stage cohort --force   # redo one stage
```

Stages, in order: `cohort subset derive collate tokenize winnow tokencounts
baseline trials export`. The slow ones are `derive` (clifpy) and `collate`
(cocoa) — both CPU, both memory-hungry. Budget roughly 40× the size of the
cohort's raw tables on scratch.

### 4. Read `filter_ledger.csv` before you send

```bash
cat output/cohort/filter_ledger.csv
```

Two things go wrong here, and both are far cheaper to catch now than after
pooling:

- **An ICU definition that does not match.** Sites label units differently.
  Compare the ICU row against total hospitalizations and ask whether it is
  plausible.
- **A table that is present but not populated.** This bit us already: UCMC's
  `clif_ecmo_mcs` names 82% of all hospitalizations and is a concept table, not
  a record of MCS. Check each criterion's sub-count for anything implausibly
  large or zero.

### 5. Send the bundle

Upload **`output/upload_to_box_<SITE>/`** — one directory, nothing else.

```
output/
  upload_to_box_<SITE>/        <-- THE ONE DIRECTORY YOU SEND
    manifest.json                site, code commit, tokenizer hash, min cell size
    token_levels.csv             the token legend, from the tokenizer alone
    counts/
      filter_ledger.csv          patient flow and attrition
      phenotype_summary.csv      Zweck phenotype breakdown
      shock_type_summary.csv     septic / cardiogenic / hypovolaemic / mixed
      seymour_summary.csv        Seymour sepsis phenotypes
      bhavani_summary.csv        Bhavani sepsis phenotypes
      seymour_availability.csv   which Seymour features this site charts
      baseline_by_shock_type.csv medians and shares by coded mechanism
      token_prevalence_anchor-*  per-token prevalence, per anchor
      token_families_anchor-*    the same, rolled up to token family
    trials/
      trial_criteria.csv         the criteria as encoded. No patient data
      trial_iqr_position_*       where this site sits against published IQRs,
                                 per trial per variable
      trial_outcomes_*           per trial: who qualifies, how they did, whether
                                 the recommended treatment was delivered, and the
                                 share inside every published IQR
      trial_guideline_assignment_*  the single most relevant guideline per
                                 patient, and whether they were also inside the
                                 population it was tested on
      trial_coverage_*           how many trials' criteria each patient meets,
                                 against outcome

  patient_level_data_<SITE>/   LOCAL ONLY -- unredacted. Never shared.
  cohort/  logs/               local working output. Not sent.
```

The two output directories are named so that what may leave is obvious without
reading any code.

Every column allowed to cross a site boundary is named in
[`code/clifcs/export/contract.py`](code/clifcs/export/contract.py) with the
argument for why; anything not named is refused at both ends. Print it:

```bash
python code/clifcs/export/contract.py
```

Suppression is applied twice — once where each table is built, then re-checked
at export, so a table assembled by some other route cannot skip it. The
threshold is 11 patients, the CMS threshold; `--min-cell` raises it if your
institution requires more.

---

## If `python3` is too old

`python` on a cluster is often 3.6, and `python -m venv` will build a 3.6 venv
without complaint — then the install fails on `cocoa-tokenizer`
(`requires_python >=3.11`) and leaves you an empty venv whose error says nothing
about versions. Get a newer interpreter without root using
[uv](https://astral.sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
PY=$(uv python find 3.11)             # an absolute path to a 3.11 interpreter
"$PY" -m venv .venv
.venv/bin/python --version            # must say 3.11+ -- stop here if it does not
.venv/bin/pip install -r requirements-site.txt
```

## If you have Slurm

Optional. [`hpc/`](hpc/) holds job scripts wrapping the same commands, useful
for a large cohort. They are written against one cluster's partition names and
memory-per-CPU rules, so treat them as a worked example rather than something
that runs unedited:

```bash
bash   hpc/00_preflight.sh     # login node: checks paths.yaml resolves
sbatch hpc/01_cohort.sbatch    # then READ output/cohort/filter_ledger.csv
sbatch hpc/02_prepare.sbatch   # the long one: derive and collate
sbatch hpc/03_tier1.sbatch     # finishes tier 1 and writes the bundle
```

Jobs `01`–`03` are exactly this request and ask for no GPU. Jobs `04`–`06` are
not part of it. See [docs/RUNNING_ON_HPC.md](docs/RUNNING_ON_HPC.md) for
partition and storage notes, including the `--mem` trap.

---

## Background

- **Model** — `GEM*-all`, a 9-layer Llama-3.2 architecture (hidden size 1024,
  1344-token vocabulary, 76.9M parameters) trained on MIMIC + UCMC + NU ICU
  data. Frozen, eval mode, under `torch.inference_mode()`. **Not loaded by this
  request** — only by tier 2.
- **Representation** — the final hidden state at the last token before the
  anchor. One 1024-d vector per patient per anchor (`24h`, `onset`).
- **Sites** — UCMC, NU and Rush have contributed so far.

## Documentation

| | |
|---|---|
| [docs/ADDING_A_SITE.md](docs/ADDING_A_SITE.md) | **Start here if you are running this.** The contributing-site procedure in detail |
| [docs/COHORT.md](docs/COHORT.md) | How the cohort is built, what the labels mean, plotting conventions |
| [docs/FEDERATION.md](docs/FEDERATION.md) | The export contract, suppression rules, what the split costs, what is still open |
| [docs/FINDINGS.md](docs/FINDINGS.md) | **What this found**, with its caveats — read the site effect first |
| [docs/TIER_2_AND_POOLED.md](docs/TIER_2_AND_POOLED.md) | Everything *not* in this request: scripts `02`–`13` |
| [docs/COORDINATING_SITE.md](docs/COORDINATING_SITE.md) | Procedure at the coordinating site |
| [docs/RUNNING_ON_HPC.md](docs/RUNNING_ON_HPC.md) | Cluster setup, the job scripts, storage, troubleshooting |

## Provenance

Dataset processing follows
[bbj-lab/coreopsis@950c2478](https://github.com/bbj-lab/coreopsis/blob/950c24782670103d130328822d06b39cb4724d7a/RUNME.sh#L14-L54).
`collation.yaml`, `tokenization.yaml`, `extraction.yaml` and `winnowing_24h.yaml`
are copied from it verbatim; `clifpy_units.yaml` is extracted from its recipes by
[code/config/_regen_clifpy_units.py](code/config/_regen_clifpy_units.py).

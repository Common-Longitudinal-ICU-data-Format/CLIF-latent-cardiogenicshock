# Running a site contribution on an HPC cluster

Written for: the analyst running this pipeline at Rush on the HPCC, under
SLURM, with GPU nodes and network access from compute nodes.

This is the cluster-specific companion to
[ADDING_A_SITE.md](ADDING_A_SITE.md), which is the authority on *what* a site
runs and *why*. Read that first. This covers only what changes when the work
is scheduled rather than run at a terminal: where to put things so a purge
does not destroy them, how to split the job so the GPU is held for the twenty
minutes that need it rather than the four hours that do not, and where the
pipeline expects a human to stop and look.

Job scripts referenced here are in [`hpc/`](../hpc/).

---

## Only one stage needs a GPU

`01_site_aggregate.py` runs ten stages. Exactly one of them, `embed`, touches
the GPU:

| stage | resource | what it does |
|---|---|---|
| `cohort` | CPU | reads raw CLIF, applies the shock definition, applies the phenotype models |
| `subset` | CPU, I/O | writes cohort-restricted CLIF to scratch |
| `derive` | CPU, **RAM** | clifpy: respiratory waterfall, dose conversion, SOFA. pandas-based, so the memory high-water mark of the whole pipeline |
| `collate` | CPU, I/O | cocoa collate |
| `tokenize` | CPU | cocoa tokenize under the frozen tokenizer |
| `winnow` | CPU | cut each timeline at both anchors |
| **`embed`** | **GPU** | the frozen model |
| `purity` | CPU | neighbour purity in GEM space |
| `trials` | CPU | trial eligibility — reads raw CLIF again |
| `export` | CPU | assemble the bundle |

Running the whole thing in one GPU allocation works and wastes most of it.
The split in `hpc/` is four jobs: CPU up to `winnow`, GPU for `embed`, CPU
from `purity` on, and a last tiny CPU job for tier 2.

`02_site_embeddings.py` needs **no GPU and almost no time** — it imports no
torch, and on the 34,142-patient UCMC+NU cohort it completed in three
seconds. It reads the tokenized sequences the `winnow` stage already wrote.

---

## What to settle before submitting anything

### Three files have to reach Rush out of band

The phenotype models are in git, at [`models/`](../models/) — cloning the repo
is enough. These are not, and must be copied in by whatever route the
consortium uses:

- the checkpoint directory `mdl-cotorra/` (`config.json`, `model.safetensors`)
- `tokenizer.yaml` — **the one the model was trained with**

Shipping a locally-fitted tokenizer is the classic failure. The embedding
table has one row per token id and the bin edges are baked into the token
names, so the wrong tokenizer produces embeddings that are noise rather than
an error. `02` refuses to run on a vocabulary-size mismatch, but that only
catches the gross case.

### Rush must be on the current commit

Bundle manifests record the repository commit and the coordinating site checks
them. A mismatch warns rather than failing, which means a stale checkout
degrades quietly: the trial set grew from 13 to 18 in commit `c7d1108`, so a
Rush bundle built before that carries no `elig_` column for SMART, APROCCHSS,
the 65 trial, CLASSIC or CLOVERS, and pooled `11` will drop those five trials
from the run with a warning rather than an error.

```bash
git -C /path/to/CLIF-latent-cardiogenicshock log --oneline -1
# should match the coordinating site
```

### Decide the tier before writing paths.yaml, not after

Tier 2 is **not de-identified** and needs IRB approval and a DUA covering
patient-level transfer in hand before `02` is run. See ADDING_A_SITE.md. Tier 1
alone is a complete contribution.

---

## Storage: the part HPC gets wrong

Three locations, and conflating them is the mistake that costs a rerun.

| | where | why |
|---|---|---|
| `scratch_home` | fast parallel scratch | ~40× the raw table size, fully regenerable, **purgeable** |
| `output_home` | project/persistent storage | the bundle, the logs, and the salt. **Never on purgeable scratch** |
| the venv | home or project | not on scratch — a purge mid-project is a silent breakage |

### The salt is the one unrecoverable file

`output/patient_level_data/.id_salt` maps the shared pseudonym back to a Rush
record number. It is never sent, and if it is lost every future export carries
pseudonyms that cannot be matched to the ones already sent — the contribution
cannot be updated, only replaced.

On a cluster this is a real risk and not a theoretical one, because the
obvious place to point `output_home` is scratch. Put it on persistent storage
and back the salt up somewhere else as well:

```bash
cp output/patient_level_data/.id_salt ~/clif-cs-salt-backup-$(date +%F)
chmod 400 ~/clif-cs-salt-backup-*
```

### Check the purge policy against the walltime

If scratch is purged on a 14-day unaccessed rule and the GPU queue is a week
deep, the `winnow` output can vanish between job 2 and job 3. Either run them
close together or touch the tree.

---

## Setup

### First: find out what this cluster calls things

Every module and partition name in these scripts is a **placeholder**. They
are wrong on any given cluster until checked, and all three fail loudly
rather than silently, which is the one good thing about them:

```bash
sinfo -o "%20P %5a %10l %6D %N"          # partitions; the one with a * is default
sinfo -o "%20P %10G %N" | grep -i gpu    # which partitions actually have GPUs
module spider python                      # every python the cluster has
module spider cuda
module avail 2>&1 | head -50
```

Then set them once, in [`hpc/_common.sh`](../hpc/_common.sh), and in the
`#SBATCH --partition=` line of each job script.

**You may not need a python module at all.**  is empty by
default and on Rush CRI it should stay empty: both  and
 are hierarchical modules sitting behind a toolchain
prerequisite, and Lmod refuses them with *"exist but cannot be loaded as
requested"*. None of that matters, because the venv is built on a standalone
CPython that  installed and activating the venv is what puts the right
interpreter on PATH. Set  only where the system python is too
old to create the venv in the first place.

There is no CUDA module either, and none is needed -- the pip  torch
wheels bundle their own CUDA runtime and the GPU nodes carry the driver.

There may be no GPU partition at all, or it may be reachable only through an
account or QoS flag. If `sinfo` shows no GRES anywhere, the pipeline still
runs -- `embed` falls back to CPU at roughly twenty times the wall time --
and the realistic plan is to split the embed job by site or anchor rather
than to request one very long allocation.

### Then build the venv

```bash
sbatch hpc/00_setup_env.sbatch
```

Run once. It builds the venv, installs the pinned stack in the order that
matters, and verifies CUDA from inside the allocation.

**It requests a GPU to run a pip install, and that is the point.** The last
thing it does is check `torch.cuda.is_available()`, and that check can only
mean anything on a node with a card. On a login node it returns False even
when everything is correct, so a setup verified there tells you nothing.

### Interactive instead, the first time

Watching pip fail in real time is often worth an allocation:

```bash
srun --partition=gpuq --gres=gpu:1 --cpus-per-task=4 \
     --mem=32G --time=01:00:00 --pty bash
```

`srun --pty bash` asks for an interactive terminal. It is an **alternative**
to `sbatch`, never a line inside a batch script -- a batch job has no
terminal to attach to and is already on a compute node by the time it runs.

Do not paste the whole setup block into a terminal in one go. If the `srun`
fails -- a wrong partition name is enough -- every following line executes
**on the login node** instead, and the first of them builds a venv against
whatever `python` happens to be on the default PATH. Delete and rebuild if
that happens; `python --version` inside the venv is the tell.

### Install order is load-bearing

`cotorra` depends on `torch`, `torchvision` and `torchaudio` without version
pins, so installing `requirements.txt` after the CUDA wheels silently pulls
CPU-only builds over the top of them. Requirements first, CUDA torch second:

```bash
pip install -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1     --index-url https://download.pytorch.org/whl/cu121
```

The failure is quiet: a CPU-only torch runs `embed` perfectly correctly and
about twenty times slower, with nothing to say so beyond `torch.__version__`
lacking a `+cu` suffix. The setup script fails rather than warns on it.

Match the CUDA wheel to the module. If the cluster has CUDA 12.4 and no
12.1, `cu121` wheels generally still run -- the driver is
backward-compatible within a major version -- but prefer the matching index
(`cu124`) when one exists.

`statsmodels` is in `requirements.txt` for `12_trial_fit_outcomes.py`, a
pooled-side script. A site running only `01` and `02` does not need it.

### Configure

```bash
cp code/config/paths.example.yaml code/config/paths.yaml
```

Filled in for Rush:

```yaml
role: site

sites:
  RUSH:
    data_dir: /gpfs/data/healthcare-allocate/CLIF_RUSH_2_1

models:
  GEM-all:
    home_glob: /gpfs/data/pearson-lab/PERSONAL/hhieromnimon/GEM-weights-FL-manuscript/GEM?-all/mdl-cotorra
default_model: GEM-all

tokenizer_yaml: /gpfs/data/pearson-lab/PERSONAL/hhieromnimon/GEM-weights-FL-manuscript/tokenizer.yaml

output_home:  /gpfs/data/pearson-lab/PERSONAL/hhieromnimon/clif-cs/output
scratch_home: /gpfs/scratch/$USER/clif-cs          # CONFIRM THIS ONE
```

`data_dir` is only ever read. The pipeline writes its cohort-restricted
copies, the `_processed` respiratory waterfall and the `_converted` dose
tables to `scratch_home`, never back into the CLIF directory, so a read-only
shared allocation is fine.

`scratch_home` is the one line left to confirm — it is the only path above
not yet pinned down, and it needs roughly **40x the size of the raw tables**:

```bash
du -sh /gpfs/data/healthcare-allocate/CLIF_RUSH_2_1    # then multiply by 40
ls -d /gpfs/scratch/$USER 2>/dev/null || echo "no /gpfs/scratch -- ask what the scratch tier is"
```

If there is no scratch tier, a subdirectory of the lab allocation works, but
check the quota against that 40x figure before starting rather than after
`derive` fails half way.

### The U+25A1 in the weights directory name

The model directories are **not** named `GEM-all`. They are named `GEM□-all`,
with a literal U+25A1 WHITE SQUARE between `GEM` and the dash:

```
GEM-weights-FL-manuscript/
├── GEM-mimic-icu/            <- plain
├── GEM-ucmc-icu/             <- plain
├── GEM□-all/                 <- U+25A1, and the only "all" checkpoint
│   └── mdl-cotorra/
│       ├── config.json
│       └── model.safetensors        293 MiB
├── GEM□-mimic-icu/
├── GEM□-ucmc-icu/
└── tokenizer.yaml
```

That is what the `?` in `GEM?-all` is matching: a single-character wildcard,
not a typo and not a shell artifact. `GEM*-all` would be ambiguous for the
other two datasets, which have both a plain and a square-prefixed directory.

This is fragile across a copy. If the transfer to GPFS dropped the character,
replaced it with an underscore, or expanded it into an escape sequence, the
glob matches nothing (or two things) and the failure surfaces later as an
unresolved path. Any *single* replacement character is fine — `?` matches
whatever it became. Check with:

```bash
ls -b /gpfs/data/pearson-lab/PERSONAL/hhieromnimon/GEM-weights-FL-manuscript/
```

Only `config.json` and `model.safetensors` are needed for inference. The
checkpoints also carry `optimizer.pt` at 587 MiB, which can be left out of
the copy.

### Preflight before queueing anything

```bash
bash hpc/00_preflight.sh
```

Resolves every path, counts the `clif_*` tables it found, checks the
checkpoint and tokenizer, and probes that `output_home` and `scratch_home`
are writable. Safe on a login node: it opens no CLIF table. If the model glob
fails it lists what is actually on disk with the codepoints spelled out, which
is the fastest way to see a mangled square.

Expected output:

```
sites
  [ok  ] RUSH             /gpfs/data/healthcare-allocate/CLIF_RUSH_2_1
         19 clif_* tables
model
  [ok  ] checkpoint       .../GEM□-all/mdl-cotorra
  ...
PASS
```

The check is the **required set by name, not a count**. The pipeline opens
14 tables; UCMC and NU each hold 19 and they are not the same 19, because
each site ships extras this pipeline never reads (`bal`,
`clinical_trials`, `intermittent_hemodialysis`, the microbiology tables,
and `ecmo_mcs`, which is deliberately unused -- it is a concept table and
names 82% of UCMC hospitalizations). A site can hold 16 tables and be
complete, or 20 and be missing something load-bearing.

Missing tables degrade quietly rather than erroring, which is why the
preflight names each one and what it costs:

| absent | what silently goes wrong |
|---|---|
| `clif_hospital_diagnosis` | every diagnosis-based trial criterion reads as not met; eligibility and shock type are understated |
| `clif_patient_procedures` | IABP, ECMO and microaxial flags all read false, so the device trials' concordance is wrong rather than absent |
| `clif_crrt_therapy` | renal support is invisible to the tokenizer |
| `clif_code_status` | limitation-of-treatment context is lost |

---

## The run, in four jobs

### Job 1 — cohort, then STOP

```bash
bash hpc/00_preflight.sh && sbatch hpc/01_cohort.sbatch
```

This is deliberately its own job because **a human has to read the output
before anything expensive runs**. Open `output/cohort/filter_ledger.csv` and
check:

- **the ICU definition.** Sites label units differently. Compare the ICU row
  against total hospitalizations and ask whether the fraction is plausible for
  Rush.
- **tables that are present but not populated.** This has already bitten this
  project once: UCMC's `clif_ecmo_mcs` names 82% of all hospitalizations
  because it is a concept table, not a record of MCS. Any criterion sub-count
  that is implausibly large or implausibly zero is the signal.

Also confirm the phenotype counts in the log are the *applied* models and not a
refit — the three `--*-model` flags are in the job script and are required. If
Rush refits its own clusters, "phenotype II" stops meaning the same thing at
each site and every pooled breakdown becomes uninterpretable.

Do not submit job 2 until the ledger looks right.

### Job 2 — subset through winnow (CPU, high memory)

```bash
sbatch hpc/02_prepare.sbatch
```

The `derive` stage is the memory peak — clifpy is pandas-based. If it is
OOM-killed, raise `--mem` before anything else; the stages are resumable, so
`--from-stage derive` picks up where it stopped.

### Job 3 — embed (GPU)

```bash
sbatch hpc/03_embed.sbatch
```

**Raise the batch size.** The default `--batch-size 8` is sized for an 8 GB
card. On a 40 GB A100 start at 32 and drop if you see CUDA OOM —
`output_hidden_states` keeps every layer, so memory scales with batch and
sequence length together.

Timing, measured on the UCMC+NU run (34,142 patients, both anchors, batch size
8, GPU model unrecorded): **1 h 58 m wall**, which works out at roughly
**0.11–0.17 s per patient per anchor**. Two anchors run, so budget about
`0.3 s × n_patients` and double it for headroom. A modern datacentre card with
a larger batch should come in well under that.

Sequence lengths from the same run, for context on truncation: median 234
tokens, p90 920, max 18,711, with 1.1% of patients exceeding the 4,096 limit
and truncated keeping the most recent tokens.

### Job 4 — purity, trials, export (CPU)

```bash
sbatch hpc/04_export.sbatch
```

Produces `output/upload_to_box_RUSH/`. Aggregate, small-cell suppressed at
`--min-cell 11`. **This is the tier 1 deliverable — send this directory.** If
Rush is also sending tier 2, job 5 adds `patient/` to this same directory
rather than making a second one.

### Job 5 — tier 2, only if the DUA is in place

```bash
sbatch --export=ALL,I_HAVE_CONFIRMED_THE_DUA=yes hpc/05_embeddings.sbatch
```

Adds `patient/` to `output/upload_to_box_RUSH/`, roughly 45 MB per 10,000
patients. **That one directory is then the whole deliverable** -- tier 1 and
tier 2 together, nothing else leaves.

Not de-identified. Move it under the same approval you would use to share
patient-level clinical data.

---

## When the bundle lands at the coordinating site

```bash
python code/03_embedding_analysis.py --role pooled
python code/11_trial_alignment.py   --role pooled
python code/12_trial_fit_outcomes.py                 # fit-score outcomes
```

Two things to check the moment Rush is in, both of which can invalidate the
pooled picture rather than merely complicate it:

**Site lift.** `output/tables/neighbourhood_purity_anchor-*.csv`, written every
run. With UCMC and NU alone, site purity is 97.1% against 60.8% by chance — a
1.60× lift, higher than any clinical label. If adding Rush keeps site lift
high, the pooled UMAP is a picture of three hospitals' charting conventions and
the analysis that answers the clinical question is a within-site or
site-adjusted one, not the pooled scatter.

**Three sites is the hard cap** for a single scatter panel: only three palette
slots pass the all-pairs colourblind gate, and a fourth raises rather than
silently picking an unsafe hue. Rush is the third. A fourth site means
faceting.

Also note that PCA and UMAP are **refit on the pooled set**, so every
coordinate shifts when Rush arrives. Figures are not comparable across runs
with different site membership — regenerate them all together.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `torch.cuda.is_available()` is False in the job | CUDA module not loaded inside the job, or torch was reinstalled by a later `pip install` and pulled CPU wheels. Reinstall the cu121 stack last. |
| CUDA OOM in `embed` | drop `--batch-size`; a long-sequence batch is the usual trigger |
| OOM-killed in `derive` | raise `--mem`; clifpy is pandas-based and this is the memory peak |
| `paths.site_data` raises | `role: pooled` in paths.yaml. A site needs `role: site`. |
| vocabulary size mismatch | the tokenizer is not the one the model was trained with |
| pooled `11` warns that trials are missing | Rush's bundle predates the trial set it is being pooled against — check the commit |
| scratch files gone between jobs | purge policy; rerun from the last completed stage with `--from-stage` |

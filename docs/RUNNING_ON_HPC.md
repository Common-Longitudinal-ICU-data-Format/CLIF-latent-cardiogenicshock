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

## The GPU is tier 2 only

**Tier 1 never loads the GEM weights**, and does not need them on disk.
`01_site_tables.py` runs ten stages, all CPU. Three of them -- `tokenize`,
`winnow`, `tokencounts` -- load the frozen *tokenizer*, which is a different
file from the weights; the other seven read CLIF and the published literature
and see no token at all.

Both are resolved on access rather than at config load, so **neither can stop a
tier-1 run**:

| `paths.yaml` key | absent means |
|---|---|
| `models.<key>.home_glob` | `02_site_embeddings.py` and `08_token_space.py` refuse to start. `01` is unaffected |
| `tokenizer_yaml` | `01` skips the three token stages with a warning and exports everything else. The bundle carries no token tables and a null tokenizer hash |

Only `02_site_embeddings.py` touches the GPU, and only a site with a
patient-level agreement runs it at all.

| script | stage | resource | what it does |
|---|---|---|---|
| 01 | `cohort` | CPU | reads raw CLIF, applies the shock definition and the phenotype models |
| 01 | `subset` | CPU, I/O | writes cohort-restricted CLIF to scratch |
| 01 | `derive` | CPU, **RAM** | clifpy: respiratory waterfall, dose conversion, SOFA. pandas-based, so the memory high-water mark of the whole pipeline |
| 01 | `collate` | CPU, I/O | cocoa collate |
| 01 | `tokenize` | CPU | cocoa tokenize under the frozen tokenizer |
| 01 | `winnow` | CPU | cut each timeline at both anchors |
| 01 | `tokencounts` | CPU | per-token and per-family prevalence — the tornado's data |
| 01 | `trials` | CPU | eligibility, IQR fit, concordance — reads raw CLIF again |
| 01 | `export` | CPU | assemble the tier-1 bundle |
| 02 | **`embed`** | **GPU** | the frozen model |
| 02 | `purity` | CPU | neighbour purity in GEM space |
| 02 | `export` | CPU | redact to the contract, add `patient/` and `purity/` |

The split in `hpc/` follows that line exactly: jobs 01–03 are tier 1 and ask for
no GPU, jobs 04–05 are tier 2. A site that stops after job 03 has made a
complete contribution.

---

## What to settle before submitting anything

### What has to reach a new site out of band

The phenotype models are in git, at [`models/`](../models/) — cloning the repo
is enough. These are not, and must be copied in by whatever route the
consortium uses:

- `tokenizer.yaml` — **the one the model was trained with**. Needed by the
  three token stages and by tier 2, and **optional**: without it `01` skips
  `tokenize`, `winnow` and `tokencounts` and exports everything else
- the checkpoint directory `mdl-cotorra/` (`config.json`, `model.safetensors`)
  — **tier 2 only**. A site asked for the aggregate tables alone does not need
  it, and tier 1 runs without it

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

Four locations, and conflating them is the mistake that costs a rerun.

| | where | why |
|---|---|---|
| `scratch_home` | fast parallel scratch | ~40× the raw table size, fully regenerable, **purgeable** |
| `output_home` | project/persistent storage | the bundle, the logs, and the salt. **Never on purgeable scratch** |
| `VENV_HOME` | project storage — **not `$HOME`** | 10–15 GB; see below. Not scratch either: a purge mid-project is a silent breakage |
| `PIP_CACHE_DIR` | beside the venv | several GB of wheels, and `~/.cache/pip` is the other half of the home-quota problem |

### The venv does not fit in a home directory -- unless you only need tier 1

**Install the tier you are running**, which for most sites is tier 1:

```bash
sbatch hpc/00_setup_env.sbatch                              # TIER=site, ~200 MB
sbatch --export=ALL,TIER=embed hpc/00_setup_env.sbatch      # ~10-15 GB, needs a GPU
```

Tier 1 is CPU-only and about **200 MB** — it will fit anywhere, including a
home directory. The size problem belongs entirely to tier 2: one line of
`cotorra` in `requirements-embed.txt` pulls torch, transformers, datasets,
lightgbm, optuna, xgboost and wandb, and the `cu121` line then installs a
second CUDA stack over the top of the first. Peak usage is **10–15 GB**, plus
several more in pip's wheel cache. A typical home
quota is smaller than that, and the failure is nasty rather than clean: pip
gets most of the way through, dies with `[Errno 122] Disk quota exceeded`, and
leaves a venv that activates fine and imports about half of what it needs. A
later run then reports `Requirement already satisfied` for broken packages.

`VENV_HOME` and `PIP_CACHE_DIR` both default to project storage in
[`hpc/_common.sh`](../hpc/_common.sh). If you move the venv, **delete the
failed one first** — reusing a half-populated venv is how the breakage
survives a rebuild:

```bash
rm -rf ~/venvs/clif-cs ~/.cache/pip
quota -s                      # or: lfs quota -h $HOME
```

The HuggingFace `datasets` cache is already redirected to `scratch_home` by
[`clifcs/config.py`](../code/clifcs/config.py), so that one is not a third
place to remember.

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

**The system python is 3.6.8 on Rush CRI and far too old.** `cotorra` and
`cocoa-tokenizer` both declare `requires_python >=3.11`. Of the two modules
`module spider python` lists, `python/3.10.5` is still too old and
**`python/3.12.3` works**:

```bash
module load python/3.12.3
python3 --version            # 3.12.3
```

`PYTHON_MODULE` in [`hpc/_common.sh`](../hpc/_common.sh) and
[`hpc/00_setup_env.sbatch`](../hpc/00_setup_env.sbatch) therefore defaults to
`python/3.12.3`. It has to be right in **both**: the setup job builds the venv
from it, and every later job does `module purge` and then activates that venv.

> An earlier version of this section said Lmod refused both pythons here --
> *"exist but cannot be loaded as requested"* -- and that a `uv`-installed
> standalone CPython was the only route. That is no longer true on this
> cluster. `module spider python` is cheaper than believing either version of
> this paragraph.

### The venv is bound to the interpreter that built it

**Load the module before creating the venv, not after.** `bin/python` is a
symlink to the creating interpreter and `pyvenv.cfg` records its absolute path,
so a venv built against 3.6.8 stays 3.6.8 however many modules are loaded
afterwards, and it cannot be upgraded in place.

The symptom, inside an activated venv:

```
$ python3 --version
Python 3.12.3          <- the module
$ python --version
Python 3.6.8           <- the venv. This is the one pip uses.
```

They disagree because the module supplies `python3` and the venv supplies
`python`. `00_setup_env.sbatch` asserts the *venv's* version after activating
it, not only `BOOTSTRAP_PYTHON`'s before creating it, so the reuse branch
cannot quietly install into a stale 3.6 venv. To recover:

```bash
deactivate
module load python/3.12.3
rm -rf "$VENV_HOME"
sbatch hpc/00_setup_env.sbatch
```

**Where no usable module exists**, build from a standalone interpreter instead
and leave `PYTHON_MODULE` empty:

```bash
uv python install 3.11
sbatch --export=ALL,PYTHON_MODULE=,BOOTSTRAP_PYTHON=$(uv python find 3.11) hpc/00_setup_env.sbatch
```

There is no CUDA module either, and none is needed: the pip `cu121` torch
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

Run once. It builds the venv and installs the pinned stack in the order that
matters. **CPU partition, no GPU.**

An earlier version asked for `--gres=gpu:1` so it could finish with
`torch.cuda.is_available()`, on the argument that spending a GPU on a pip
install was the only way to make that check mean anything. Both halves of that
were wrong. The check that catches the real failure -- cotorra pulling CPU
torch over the cu121 wheels -- is the `+cu121` string on the installed wheel,
which is a property of what pip resolved and reads the same on any node. And
`torch.cuda.is_available()` is already asserted at the top of
[`hpc/04_embed.sbatch`](../hpc/04_embed.sbatch), the only job that needs a
card, so checking it here gated the whole venv on a six-node partition to
learn something the job that needs it re-learns anyway.

Tier 1 is CPU throughout -- jobs 01-03 never load the GEM weights -- so a site
with no GPU runs its entire contribution, and has to be able to build the venv
to do it.

### Interactive instead, the first time

Watching pip fail in real time is often worth an allocation:

```bash
srun --partition=tier1q --cpus-per-task=4 \
     --mem=16000M --time=01:00:00 --pty bash
```

Note `--mem=16000M`, not `16G`. Memory is capped per CPU here and Slurm's
`G` is 1024 MB, so `--mem=16G` on 4 CPUs asks 4096 MB/CPU and is refused by
96 MB against a 4000 MB cap. That reads as *"Requested node configuration is
not available"*, which sounds like a shortage and is not.

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
pins, so installing `requirements-embed.txt` after the CUDA wheels silently pulls
CPU-only builds over the top of them. Requirements first, CUDA torch second:

```bash
pip install -r requirements-embed.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

This applies to tier 2 only. `requirements-site.txt` contains no torch,
so the ordering question does not arise for a site running tier 1.

The failure is quiet: a CPU-only torch runs `embed` perfectly correctly and
about twenty times slower, with nothing to say so beyond `torch.__version__`
lacking a `+cu` suffix. The setup script fails rather than warns on it.

Match the CUDA wheel to the module. If the cluster has CUDA 12.4 and no
12.1, `cu121` wheels generally still run -- the driver is
backward-compatible within a major version -- but prefer the matching index
(`cu124`) when one exists.

`statsmodels` is in `requirements-pooled.txt` for `12_trial_fit_outcomes.py`.
A site running only `01` and `02` never installs it.

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

Resolves every path, checks the required `clif_*` tables by name, checks the
checkpoint and tokenizer, and probes that `output_home` and `scratch_home`
are writable. Safe on a login node: it opens no CLIF table. If the model glob
fails it lists what is actually on disk with the codepoints spelled out, which
is the fastest way to see a mangled square.

**The checkpoint line cannot fail the preflight** -- it is reported as `[--  ]`
when unresolved, because tier 1 does not read it. The tokenizer line can fail
it, and should.

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

## The run, in five jobs plus one

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
refit. 01 applies the shared models in `models/` by default and will only refit
when asked with `--refit-phenotypes`, which logs a warning — so look for that
warning rather than assuming. A site that refits its own clusters makes
"phenotype II" stop meaning the same thing at each site, and every pooled
breakdown becomes uninterpretable.

Do not submit job 2 until the ledger looks right.

### Job 2 — subset through winnow (CPU, high memory)

```bash
sbatch hpc/02_prepare.sbatch
```

The `derive` stage is the memory peak — clifpy is pandas-based. If it is
OOM-killed, raise `--mem` before anything else; the stages are resumable, so
`--from-stage derive` picks up where it stopped.

### Job 3 — tier 1: token counts, trials, export (CPU)

```bash
sbatch hpc/03_tier1.sbatch
```

Produces `output/upload_to_box_RUSH/`. Aggregate, small-cell suppressed at
`--min-cell 11`. **This is the tier 1 deliverable — send this directory.**
No GPU was used and the GEM weights were never loaded.

The `tokencounts` stage is new: it counts how often each token appears here,
which is what `04_site_effect.py` draws its tornado from. That comparison used
to need the token sequences and so existed only for sites that shared them.

If Rush is also sending tier 2, jobs 4 and 5 add `purity/` and `patient/` to
this same directory rather than making a second one.

### Job 4 — embed (GPU, TIER 2)

```bash
sbatch hpc/04_embed.sbatch
```

**This is the first point at which the GEM weights are loaded.** Do not submit
it unless the IRB approval and DUA cover patient-level transfer.

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

### Job 5 — purity and export (CPU, TIER 2)

```bash
sbatch --export=ALL,I_HAVE_CONFIRMED_THE_DUA=yes hpc/05_tier2.sbatch
```

Adds `purity/` and `patient/` to `output/upload_to_box_RUSH/`, roughly 45 MB
per 10,000 patients. **That one directory is then the whole deliverable** --
tier 1 and tier 2 together, nothing else leaves.

`purity` is the phenotype-coherence measure. It used to sit in tier 1 and was
the reason tier 1 needed a GPU; it is here now, which is why a tier-1-only site
does not report it.

Not de-identified. Move it under the same approval you would use to share
patient-level clinical data.

### Job 6 — trajectories (GPU), only where sites are co-located

```bash
python code/05_site_correction.py --anchor onset --save-map
sbatch hpc/06_trajectories.sbatch
```

**Not part of the federated tiers.** This runs only where several sites' CLIF
extracts sit on the same filesystem; it pools them directly and sends nothing,
so no export contract applies to it.

It embeds every patient at 26 grid points around shock onset -- roughly 26x the
work of job 4, which is why the walltime is 24 h and why it is resumable: the
extract stage skips grid points whose output already exists, so a job that runs
out of time can simply be resubmitted. The grid embeddings are the largest thing
this pipeline writes and go to `scratch_home`, not to `output/`.

05 must run with `--save-map` first. 09 applies that map to every grid point
*before* projecting -- the map is affine on the 1024-d embedding, so correcting
PaCMAP coordinates instead would be meaningless -- and it deliberately does not
fit its own, because refitting on the grid would weight the fit by whose grid is
complete.

The figures it exists for are the two readings of cardiogenic shock side by
side, with the PaCMAP scatter coloured by Zweck phenotype:

```
output/figures/<model>/trajectory/corrected-leace/
  trajectory_flow_cardiogenic_coded.png              ICD R57.0
  trajectory_flow_cardiogenic_cardiac_origin.png     inotrope / MCS / troponin
  trajectory_cardiogenic_coded_phenotype.png
  trajectory_cardiogenic_cardiac_origin_phenotype.png
```

Both panels use identical bins and the same low-count mask, so they are
comparable; the log reports how far the two definitions overlap, because they
are two readings of one cohort rather than two independent findings. Add
`--export=ALL,CORRECTION=none` for the uncorrected field, which lands in a
different directory and does not overwrite it.

---

## When the bundle lands at the coordinating site

```bash
python code/03_embedding_analysis.py --role pooled
python code/04_site_effect.py        --role pooled   # the tornado, from counts/
python code/05_site_correction.py    --role pooled --save-map
python code/03_embedding_analysis.py --role pooled --corrected leace
python code/11_trial_alignment.py    --role pooled
python code/12_trial_fit_outcomes.py                 # fit-score outcomes
```

No GPU and no re-embedding: sites ship the vectors. Running 03 twice is
deliberate — the second pass redraws the identical panel set from 05's affine
output into `figures/<model>/anchor-<a>/corrected-leace/`, so every colouring
can be read raw and corrected side by side.

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

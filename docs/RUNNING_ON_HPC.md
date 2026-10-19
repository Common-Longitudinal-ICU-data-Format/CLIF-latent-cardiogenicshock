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

`02_site_tokens.py` needs **no GPU and almost no time** — it imports no
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

Build the venv on a compute node, not the login node — the login node usually
has no GPU and often a different CPU generation.

```bash
git clone <repo> CLIF-latent-cardiogenicshock
cd CLIF-latent-cardiogenicshock

srun --partition=<gpu-partition> --gres=gpu:1 --cpus-per-task=4 \
     --mem=32G --time=01:00:00 --pty bash

module load python/3.11
module load cuda/12.1          # must match the cu121 wheels below

python -m venv ~/venvs/clif-cs
source ~/venvs/clif-cs/bin/activate
pip install --upgrade pip

pip install -r requirements.txt
# CUDA torch LAST: cotorra depends on torch/torchvision/torchaudio unpinned,
# so a plain install silently replaces a CUDA build with CPU wheels
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.5.1+cu121 True   -- and it must say True HERE, inside the job
```

`statsmodels` is in `requirements.txt` for `12_trial_fit_outcomes.py`, which is
a pooled-side script. A site running only `01` and `02` does not need it and
can ignore it if it gives trouble.

### Configure

```bash
cp code/config/paths.example.yaml code/config/paths.yaml
```

```yaml
role: site

sites:
  RUSH:
    data_dir: /project/.../CLIF_RUSH_2_1

models:
  GEM-all:
    home_glob: /project/.../GEM-weights-FL-manuscript/GEM?-all/mdl-cotorra
default_model: GEM-all

tokenizer_yaml: /project/.../GEM-weights-FL-manuscript/tokenizer.yaml

output_home:  /project/.../clif-cs/output      # persistent
scratch_home: /scratch/$USER/clif-cs           # fast, purgeable
```

Note the `GEM?-` wildcard: it is a single-character match that selects the
"star" run and would be ambiguous as `GEM*-`.

Config-only sanity check, reads no data:

```bash
python -c "import sys; sys.path.insert(0,'code'); from clifcs import load_paths; p=load_paths(); print(p.sites, p.model_home, p.tokenizer_yaml)"
```

---

## The run, in four jobs

### Job 1 — cohort, then STOP

```bash
sbatch hpc/01_cohort.sbatch
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
`--min-cell 11`. **This is the tier 1 deliverable — send this directory.**

### Job 5 — tier 2, only if the DUA is in place

```bash
sbatch hpc/05_tokens.sbatch     # seconds, CPU
```

Produces `output/token_level_data_RUSH/`, roughly 10 MB per 10,000 patients.
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

# CLIF latent space — shock

Where do shock patients sit in the embedding space of a frozen CLIF foundation
model, and do the published subphenotypes correspond to anything the model sees?

The cohort is **undifferentiated shock**, selected on physiology alone, with the
coded mechanism attached afterwards as a labelling rather than a filter — so the
question of whether septic, cardiogenic and hypovolaemic shock separate in the
latent space stays answerable.

Runs **federated**: raw CLIF records never move. Each site produces a bundle and
the coordinating site pools whichever bundles arrive.

- **Model** — `GEM*-all`, a 9-layer Llama-3.2 architecture (hidden size 1024,
  1344-token vocabulary, 76.9M parameters) trained on MIMIC + UCMC + NU ICU
  data. Frozen, eval mode, under `torch.inference_mode()`.
- **Representation** — the final hidden state at the last token before the
  anchor. One 1024-d vector per patient per anchor (`24h`, `onset`).
- **Sites** — three contribute: UCMC, NU and Rush. **Each builds its own
  cohort** from its own records using the identical definition; nothing is
  pooled until the coordinating site combines the bundles. Configured in
  `code/config/paths.yaml`; adding one needs no code change, see
  [docs/ADDING_A_SITE.md](docs/ADDING_A_SITE.md).

## The three analyses

| | runs | needs | produces |
|---|---|---|---|
| **1 — site tables** | every site | CPU, tokenizer only | counts, phenotype breakdowns, trial comparison, token prevalence |
| **2 — embeddings** | sites with a patient-level DUA | **GPU** + GEM weights | the 1024-d vectors, plus phenotype coherence |
| **3 — pooled** | coordinating site | the received bundles | projections, site correction, trial alignment, the report |

Tier 1 never loads the model. A site with no GPU or no agreement runs it and
stops, and that is a complete contribution.

Two output directories, named so what may leave is obvious without reading code:

```
output/
  patient_level_data_<SITE>/   LOCAL ONLY -- unredacted. Never shared.
  upload_to_box_<SITE>/        THE ONE DIRECTORY A SITE SENDS
    counts/ trials/              tier 1 -- aggregate, small-cell suppressed
    purity/ patient/             tier 2 -- needs a DUA
```

Every column allowed to cross a site boundary is named in
[`code/clifcs/export/contract.py`](code/clifcs/export/contract.py) with the
argument for why; anything not named is refused at both ends. Print it with:

```bash
python code/clifcs/export/contract.py
```

## Running it

No cluster required. Tier 1 is a plain Python program on one machine — install
`requirements-site.txt`, copy `code/config/paths.example.yaml` to `paths.yaml`,
fill in this machine's data directories and scratch path, and run it.

**At a contributing site** — this is the whole of it:

```bash
# tier 1 -- everyone. No GPU.
python code/01_site_tables.py \
    --phenotype-model models/phenotype_model.json \
    --seymour-model   models/seymour_model.json \
    --bhavani-model   models/bhavani_model.json

# tier 2 -- ONLY with a patient-level DUA in place. Needs a GPU.
python code/02_site_embeddings.py --batch-size 8
```

The three `--*-model` flags are **required**: without them the site refits its
own clusters and "phenotype II" stops meaning the same thing everywhere. Both
scripts are staged and resumable (`--stage`, `--from-stage`, `--force`); the
slow stages are `derive` and `collate`.

Then send `output/upload_to_box_<SITE>/` — one directory, whichever tier ran.

**At the coordinating site**, drop the received bundles in `output/received/`:

```bash
python code/03_embedding_analysis.py --role pooled
python code/04_site_effect.py        --role pooled
python code/05_site_correction.py    --role pooled --save-map
python code/03_embedding_analysis.py --role pooled --corrected leace
python code/11_trial_alignment.py    --role pooled
python code/12_trial_fit_outcomes.py
python code/07_report.py
```

No GPU: sites ship the vectors, so nothing is re-embedded. Running `03` twice is
deliberate — the second pass redraws every panel from the site-corrected
embeddings into `corrected-leace/`, so each colouring reads raw and corrected
side by side.

Under `--role pooled`, `paths.site_data` **raises**, so a pooled run cannot read
raw records even by mistake.

### If you have Slurm

`hpc/` holds job scripts that wrap the same commands — useful for a large
cohort, and **optional**. They are written against one cluster's partition
names and memory-per-CPU rules, so treat them as a worked example rather than
something that runs unedited:

```bash
bash   hpc/00_preflight.sh         # login node: checks paths.yaml resolves
sbatch hpc/01_cohort.sbatch        # then READ output/cohort/filter_ledger.csv
sbatch hpc/02_prepare.sbatch       # the long one: derive and collate
sbatch hpc/03_tier1.sbatch         # finishes tier 1
```

Jobs 01–03 are tier 1 and ask for no GPU; 04–05 are tier 2. See
[docs/RUNNING_ON_HPC.md](docs/RUNNING_ON_HPC.md) for the partition and storage
notes, including the `--mem` trap.

`00_preflight.sh` is worth running even without Slurm: it checks that every
path in `paths.yaml` resolves and that each site has the 14 tables the
pipeline reads, which is the cheapest way to catch a misconfiguration.

### Trajectories

[`code/09_trajectories.py`](code/09_trajectories.py) embeds each patient at 26
points around shock onset and draws where they move. It is **not federated** —
26 vectors per patient is far too large a disclosure surface — and runs only
where several contributing sites share a filesystem:

```bash
python code/05_site_correction.py --anchor onset --save-map
sbatch hpc/06_trajectories.sbatch
```

It applies the affine map 05 fitted at the anchor to every grid point *before*
projecting, then draws the flow for both readings of cardiogenic shock — the ICD
code and the measurement-based `cardiac_origin` flag — over a scatter coloured
by Zweck phenotype.

## Outputs

```
output/
  cohort/                     cohort.parquet, filter_ledger.csv, token_prevalence_*
  patient_level_data_<SITE>/  LOCAL ONLY -- embeddings_anchor-{24h,onset}.parquet
  upload_to_box_<SITE>/       the bundle to send
  figures/<model>/anchor-*/   projections coloured by site, phenotype, shock type,
                              mortality, ICU LOS, ventilation, pressors, MAP, SOFA,
                              trial eligibility, trial fit, concordance
                 corrected-*/ the identical set, site effect removed
                 trajectory/  09's flow fields
  tables/<model>/             purity, site-effect decomposition, trial tables
  logs/
```

Intermediates go to `scratch_home` — large, regenerable, and never on a synced
drive.

## Setup

Python 3.11 or newer. **Install only the tier you are running** — this is the
difference between a 200 MB install and a 15 GB one:

| | install | size | needs a GPU |
|---|---|---|---|
| tier 1, every site | `pip install -r requirements-site.txt` | ~200 MB | no |
| tier 2, with a DUA | `pip install -r requirements-embed.txt` | ~10–15 GB | yes |
| coordinating site | `pip install -r requirements-pooled.txt` | ~500 MB | no |
| all three roles | `pip install -r requirements.txt` | ~15 GB | yes |

A contributing site running tier 1 needs **no torch, no CUDA and no GPU**. One
line of `cotorra` pulls torch, transformers, datasets, lightgbm, optuna,
xgboost and wandb, which is where the 15 GB comes from — and tier 1 never loads
the model, so it is in `requirements-embed.txt` and nowhere else.

**Check your interpreter first.** `python` on a cluster is often 3.6, and
`python -m venv` will build a 3.6 venv without complaint — then the install
fails on `cocoa-tokenizer` (`requires_python >=3.11`) and leaves you an empty
venv whose error says nothing about versions:

```bash
python3 --version                     # need 3.11 or newer
```

If it is too old, get one without root using [uv](https://astral.sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
PY=$(uv python find 3.11)             # an absolute path to a 3.11 interpreter
```

Then build the venv **from that interpreter** and check it before installing:

```bash
"${PY:-python3}" -m venv .venv        # NOT in $HOME on a cluster if you need tier 2
.venv/bin/python --version            # must say 3.11+ -- stop here if it does not
.venv/bin/pip install -r requirements-site.txt
```

For tier 2, install the CUDA wheels **last** — `cotorra` leaves
`torch`/`torchvision`/`torchaudio` unpinned, so installing them first lets it
resolve CPU-only or mismatched builds over the top:

```bash
.venv/bin/pip install -r requirements-embed.txt
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1     --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.5.1+cu121 True   <- a bare 2.5.1 means the CPU wheels won; embedding will
#                       then be ~20x slower and say nothing
```

The tokenizer at `GEM-weights-FL-manuscript/tokenizer.yaml` is **required** for
tier 1 as well as tier 2, and must be the training-time one: the embedding
table is only meaningful under that exact token-to-id mapping, and the decile
bin edges have to be the ones the values were binned with during training. The
model *weights* are needed only for tier 2.

## Documentation

| | |
|---|---|
| [docs/FINDINGS.md](docs/FINDINGS.md) | **What this found**, with its caveats — read the site effect first |
| [docs/COHORT.md](docs/COHORT.md) | How the cohort is built, what the labels mean, plotting conventions |
| [docs/FEDERATION.md](docs/FEDERATION.md) | The export contract, suppression rules, what the split costs, what is still open |
| [docs/ADDING_A_SITE.md](docs/ADDING_A_SITE.md) | Procedure at a new contributing site |
| [docs/COORDINATING_SITE.md](docs/COORDINATING_SITE.md) | Procedure at the coordinating site |
| [docs/RUNNING_ON_HPC.md](docs/RUNNING_ON_HPC.md) | Cluster setup, the job scripts, storage, troubleshooting |

## Provenance

Dataset processing follows
[bbj-lab/coreopsis@950c2478](https://github.com/bbj-lab/coreopsis/blob/950c24782670103d130328822d06b39cb4724d7a/RUNME.sh#L14-L54).
`collation.yaml`, `tokenization.yaml`, `extraction.yaml` and `winnowing_24h.yaml`
are copied from it verbatim; `clifpy_units.yaml` is extracted from its recipes by
[code/config/_regen_clifpy_units.py](code/config/_regen_clifpy_units.py).

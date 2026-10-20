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
- **Sites** — configured in `code/config/paths.yaml`. Adding one needs no code
  change; see [docs/ADDING_A_SITE.md](docs/ADDING_A_SITE.md).

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

Set up the environment, then copy `code/config/paths.example.yaml` to
`paths.yaml` and fill in this machine's data directories, model and scratch
paths.

**At a contributing site:**

```bash
# tier 1 -- everyone. No GPU.
python code/01_site_tables.py     --phenotype-model models/phenotype_model.json     --seymour-model   models/seymour_model.json     --bhavani-model   models/bhavani_model.json

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

**On an HPC cluster**, use the job scripts instead — jobs 01–03 are tier 1 and
ask for no GPU, 04–05 are tier 2:

```bash
sbatch hpc/00_setup_env.sbatch     # once
bash   hpc/00_preflight.sh         # login node: checks paths.yaml
sbatch hpc/01_cohort.sbatch        # then READ output/cohort/filter_ledger.csv
sbatch hpc/02_prepare.sbatch
sbatch hpc/03_tier1.sbatch
```

See [docs/RUNNING_ON_HPC.md](docs/RUNNING_ON_HPC.md) — storage layout, the
memory-per-CPU trap, and what to check at each gate.

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

Python 3.11. The pinned combination matters: `cotorra` pulls unpinned
`torch`/`torchvision`/`torchaudio`, which will silently replace a CUDA build
with CPU wheels.

```bash
python -m venv <venv>            # NOT in $HOME: this needs 10-15 GB
<venv>/bin/python -m pip install -r requirements.txt
<venv>/bin/python -m pip install torch==2.5.1 torchvision==0.20.1     torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the CUDA torch **last**, then verify it reports `2.5.1+cu121` and
`True`:

```bash
<venv>/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The tokenizer at `GEM-weights-FL-manuscript/tokenizer.yaml` is **required** and
must be the training-time one: the embedding table is only meaningful under that
exact token-to-id mapping, and the decile bin edges have to be the ones the
values were binned with during training.

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

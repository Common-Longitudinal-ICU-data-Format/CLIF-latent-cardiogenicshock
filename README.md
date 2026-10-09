# CLIF latent space — cardiogenic shock

Exploring where cardiogenic shock patients sit in the embedding space of a
frozen CLIF foundation model, using data from two contributing sites.

- **Model** — `GEM*-all`, the coreopsis `cxxx-all` run: a 9-layer Llama-3.2
  architecture derived from `meta-llama/Llama-3.2-1B` (hidden size **1024**, 8
  heads, 1344-token vocabulary, 76.9M parameters) trained over 5 passes of the
  combined MIMIC + UCMC + NU ICU datasets. Used strictly frozen, in eval mode, under `torch.inference_mode()`.
- **Sites** — configured in `code/config/paths.yaml`; currently UCMC and NU,
  with Rush planned. Adding one needs no code change — see
  [docs/ADDING_A_SITE.md](docs/ADDING_A_SITE.md).
- **Representation** — the final hidden state at the last token before the
  anchor, i.e. the model's state after reading everything known about the
  patient up to that point. One 1024-d vector per patient per anchor.

## Pipeline

| Script | Does |
|---|---|
| [code/01_data_processing.py](code/01_data_processing.py) | Cardiogenic shock cohort, filter ledger, outcomes, phenotypes; then CLIF subset → clifpy derived tables → cocoa collate/tokenize/winnow |
| [code/02_model_embedding.py](code/02_model_embedding.py) | Reports model structure and embedding dimension; runs `cotorra.Extractor`; writes `output/patient_level_data/` |
| [code/03_embedding_analysis.py](code/03_embedding_analysis.py) | PCA + UMAP, and the scatterplots |

```bash
python code/01_data_processing.py          # all six stages
python code/02_model_embedding.py
python code/03_embedding_analysis.py
```

`01` is staged and resumable — `--stage {cohort,subset,derive,collate,tokenize,winnow}`,
or `--from-stage <name>`; add `--force` to redo work whose outputs already exist.
The two slow stages are `derive` (clifpy) and `collate` (cocoa).

## Cohort definition

Configured in [code/config/cohort.yaml](code/config/cohort.yaml).

**Base cohort** mirrors coreopsis `recipes/preprocessing.py` — first
hospitalization per patient, ≥24 h long, with an ICU stay in the first 24 h — so
the patients embedded here sit inside the distribution the model was trained on.

**(1) Hemodynamic criterion** — SBP < 90 mmHg sustained for > 30 min (a maximal
run of consecutive sub-threshold readings spanning more than 30 min), **or** a
supportive intervention to maintain SBP > 90: a continuous vasoactive/inotrope
infusion at a non-zero dose.

**(2) End-organ damage**, within −6 h/+24 h of when (1) is first met — altered
mental status (GCS < 15, or a ≥2-point drop from the patient's own best GCS),
lactate ≥ 2.0 mmol/L, or creatinine-based KDIGO AKI.

`cs_onset_dttm` is the time both criteria hold.

### Two things to know about this definition

**Urine output is not available.** The specified end-organ criterion is "AMS, UO
< 30 mL/h". Neither site's CLIF 2.1 extract has a `patient_intake_output` table,
and urine output appears in neither `clif_vitals.vital_category` nor
`clif_patient_assessments.assessment_category` at either site. Creatinine-based
KDIGO AKI stands in for it, and lactate is added as a second hypoperfusion
marker.

**The definition has no cardiac criterion**, so it selects *undifferentiated*
shock — septic, hypovolaemic and haemorrhagic shock are all included, which is
why the cohort runs to tens of thousands rather than the hundreds a cardiogenic
shock registry would hold. A `cardiac_origin` flag is computed and reported for
every patient but **not applied**; set `cardiac_origin.enabled: true` in the
config to restrict to it. It is satisfied by an **inotrope infusion**
(dobutamine, milrinone) or a **mechanical circulatory support procedure** (ECMO,
IABP, percutaneous VAD, in CPT and ICD-10-PCS). Diagnosis codes are deliberately
not used: a diagnosis code asserts a condition, whereas an infusion and a
procedure record a therapy actually delivered — the stronger evidence that the
treating team judged the shock to be cardiac.

**`clif_ecmo_mcs` is not read anywhere.** It is a concept table at both sites
and is not reliably populated — at UCMC it names 136,565 of 166,814
hospitalizations (82%) and carries no device or flow columns. MCS is therefore
taken from `clif_patient_procedures` instead, and MCS is not part of
criterion (1): MCS without either sustained hypotension or a vasoactive infusion
occurred in 3 of 34,142 patients, so nothing is lost by leaving it out.

## Embedding anchors

Each patient is embedded twice, at two cut points:

- **`24h`** — 24 h after admission. Matches the winnowing config the model was
  trained and evaluated under (`duration_s: 86400`), so the representation is
  in-distribution.
- **`onset`** — at `cs_onset_dttm`, so representations are aligned to the
  clinical event rather than to admission. cocoa's winnower cannot express a
  per-patient cut point, so
  [code/clifcs/onset_winnower.py](code/clifcs/onset_winnower.py) overrides its
  thresholding; everything downstream is cocoa's unchanged.

## Phenotypes

Zweck et al., *Phenotyping Cardiogenic Shock*, J Am Heart Assoc 2021;10:e020085
([doi:10.1161/JAHA.120.020085](https://doi.org/10.1161/JAHA.120.020085)) —
phenotypes **I** non-congested, **II** cardiorenal, **III** cardiometabolic.

This is a **refit, not a transfer**: the paper names the clustering variables
and each phenotype's direction but does not publish the centroids, so k-means is
refit on the eight labs it names and the clusters are then labelled
*deterministically from their centroid profiles* rather than by cluster index.
`clifcs.phenotype.summarize` checks the recovered labels against the paper's
mortality ordering (I < II < III) and warns if it does not hold.

## Outputs

```
output/
  cohort/              cohort.parquet, filter_ledger.csv, phenotype_*.csv, cs_onset.parquet
  patient_level_data/  embeddings_anchor-{24h,onset}_{SITE}.parquet  <- 1024-d vector per patient
                       projections_anchor-{24h,onset}.parquet        <- PCA/UMAP coordinates
  figures/anchor-*/    {pca,umap}_{site,phenotype,mortality,icu_los}.png
  tables/              model_summary.json, embedding_summary.csv
  logs/
```

Intermediates (cohort-restricted CLIF tables, token streams, raw feature shards)
go to `scratch_home` in [code/config/paths.yaml](code/config/paths.yaml), off the
Box-synced tree — they are large and regenerable.

## Multiple sites

Each site runs 01 and 02 against its own data and produces one embedding file
per anchor; 03 globs them, so pooling is a file copy rather than a merge step.
Raw CLIF data never moves. The full procedure, including what the transfer needs
by way of IRB/DUA cover, is in [docs/ADDING_A_SITE.md](docs/ADDING_A_SITE.md).

Phenotypes are fit once on the derivation cohort and saved to
`output/cohort/phenotype_model.json`; contributing sites **apply** it via
`--phenotype-model` rather than refitting, so the label means the same thing
everywhere.

## Read the site plot first

The embedding geometry at the 24 h anchor is **dominated by contributing site**,
not by physiology. Measured as the share of each patient's 30 nearest
neighbours (50-d PCA space) carrying the same label:

| label | purity | by chance | lift |
|---|---|---|---|
| **site** | **97.1%** | 60.8% | **1.60x** |
| cs_phenotype | 71.5% | 61.6% | 1.16x |
| in-hospital mortality | 80.5% | 72.3% | 1.11x |

94% of a UCMC patient's neighbours are UCMC; 98% of an NU patient's are NU. A
model trained on event streams reads documentation practice -- which units
exist, which assessment scales get charted, how often labs are drawn -- as
readily as it reads physiology, and all of that is site-specific.

So structure in the phenotype and mortality plots is substantially site
composition showing through. Compare **within** site before reading anything
clinical into the pooled cloud. `03` recomputes this table every run
(`output/tables/neighbourhood_purity_anchor-*.csv`) and warns when site lift
exceeds 1.3x; check it before each new site is added, not after.

## Plot conventions

dpi 500; the legend sits in its own axes outside the scatter; the scatter keeps a
1:1 data aspect ratio. Colours come from the dataviz reference palette's
all-pairs–validated slots —
[code/clifcs/palette.py](code/clifcs/palette.py) carries the validator and
self-tests itself against the published reference figures:

```bash
python code/clifcs/palette.py
```

## Setup

Python 3.11. The pinned combination matters: `cotorra` pulls unpinned
`torch`/`torchvision`/`torchaudio`, which will silently replace a CUDA build with
CPU wheels, and it accepts polars 2.x, which the reference implementation was not
written against.

```bash
python -m venv <venv>                     # keep this OFF the Box-synced tree
<venv>/Scripts/python -m pip install -r requirements.txt
<venv>/Scripts/python -m pip install torch==2.5.1 torchvision==0.20.1 \
    torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the CUDA torch **last** and verify:

```bash
<venv>/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.5.1+cu121 True
```

## Provenance

Dataset processing follows
[bbj-lab/coreopsis@950c2478](https://github.com/bbj-lab/coreopsis/blob/950c24782670103d130328822d06b39cb4724d7a/RUNME.sh#L14-L54).
`collation.yaml`, `tokenization.yaml`, `extraction.yaml` and `winnowing_24h.yaml`
are copied from it verbatim; `clifpy_units.yaml` is extracted from its recipes by
[code/config/_regen_clifpy_units.py](code/config/_regen_clifpy_units.py).

The tokenizer at `GEM-weights-FL-manuscript/tokenizer.yaml` is **required** and
must be the training-time one — the model's 1344-row embedding table is only
meaningful under that exact token→id mapping, and the decile bin edges have to be
the ones the numeric values were binned with during training. `02` asserts that
the model's `vocab_size` matches the tokenizer's lookup before extracting.

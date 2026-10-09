# Adding a contributing site

Written for: the analyst running this pipeline at a contributing site, and the
coordinating analyst pooling the results.

Raw CLIF data never leaves the site. The site runs steps 01 and 02 against its
own data and its own copy of the frozen model, and sends back **two parquet
files per anchor** — one row per patient, holding the 1024-d embedding plus the
handful of covariates the plots need. Step 03 pools whatever files are present.

---

## Before anything else: this transfer needs cover

The files a site sends back are **patient-level data**, not aggregate
statistics. `--share-safe` removes direct identifiers and raw timestamps and
caps age at 90, but:

> A 1024-d embedding is a learned function of the patient's entire recorded
> history. No one has shown that such representations resist re-identification
> attack, and they should not be treated as de-identified.

Treat the export as a **limited data set at best**, and move it under the same
IRB approval and data use agreement you would use to share patient-level
clinical data. Confirm this is in place before anyone runs step 02. If your DUA
does not permit patient-level transfer, the alternative is to ship the *model*
to each site and pool only aggregate statistics — a different design from this
one, and a larger change.

---

## What the site needs

| | |
|---|---|
| CLIF 2.1 tables | `clif_hospitalization, patient, adt, labs, vitals, medication_admin_continuous, medication_admin_intermittent, patient_assessments, respiratory_support, patient_procedures, position, code_status, crrt_therapy` |
| Model checkpoint | the `mdl-cotorra/` directory: `config.json`, `model.safetensors` |
| **Tokenizer** | `tokenizer.yaml` — the one the model was **trained** with |
| Phenotype model | `cohort/phenotype_model.json` from the coordinating site |
| Hardware | a GPU with ≥8 GB is plenty; CPU works but is ~20× slower |
| Disk | roughly 40× the size of the cohort's raw tables, on scratch |

The tokenizer is the part people get wrong. The model's embedding table has one
row per token id and the numeric bin edges are baked into the token names, so a
tokenizer learned locally maps the same clinical event to a different id and the
embeddings become noise. Ship `tokenizer.yaml` with the weights. Step 02 refuses
to run if its vocabulary size does not match the model's.

---

## Site steps

### 1. Environment

```bash
git clone <repo> && cd CLIF-latent-cardiogenicshock
python -m venv ~/venvs/clif-cs              # NOT on a synced or networked drive
source ~/venvs/clif-cs/bin/activate         # Windows: ~/venvs/clif-cs/Scripts/activate

pip install -r requirements.txt
# install CUDA torch LAST: cotorra depends on torch/torchvision/torchaudio
# unpinned, so a plain install silently replaces a CUDA build with CPU wheels
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.5.1+cu121 True
```

On an HPC login node there is usually no GPU — check `torch.cuda.is_available()`
from inside the *job*, not the login shell, and load the CUDA module your
cluster provides before installing.

### 2. Configure

```bash
cp code/config/paths.example.yaml code/config/paths.yaml
```

Fill in one entry per site you hold, the model glob, the tokenizer, and a
`scratch_home` on fast local scratch. `paths.yaml` is gitignored.

Sanity check — this reads nothing but config:

```bash
python -c "from sys import path; path.insert(0,'code'); from clifcs import load_paths; p=load_paths(); print(p.sites); print(p.model_home); print(p.tokenizer_yaml)"
```

### 3. Cohort — and check the numbers before going further

```bash
python code/01_data_processing.py --stage cohort \
    --phenotype-model /path/to/phenotype_model.json
```

`--phenotype-model` is **required at a contributing site**. Without it the site
refits k-means on its own patients and "phenotype II" stops meaning the same
thing at each site, which makes the pooled phenotype plot uninterpretable.

Read `output/cohort/filter_ledger.csv` before running anything expensive. The
two things that go wrong:

- **An ICU definition that does not match.** Sites label units differently;
  compare the ICU row against total hospitalizations and ask whether it is
  plausible.
- **A table that is present but not populated.** This bit us already: UCMC's
  `clif_ecmo_mcs` names 82% of all hospitalizations and is a concept table, not
  a record of MCS. Check each criterion's sub-count in the ledger for anything
  implausibly large or zero.

### 4. Everything else

```bash
python code/01_data_processing.py --from-stage subset   # ~20 min / 25k patients
python code/02_model_embedding.py --batch-size 8 --share-safe
```

`--batch-size 8` fits an 8 GB card; raise it if you have more. Drop it to 2–4 if
you hit CUDA OOM — `output_hidden_states` keeps every layer.

### 5. Send back

```
output/patient_level_data/embeddings_anchor-24h_<SITE>_share-safe.parquet
output/patient_level_data/embeddings_anchor-onset_<SITE>_share-safe.parquet
```

Roughly 70 MB per 10,000 patients. Also send `output/cohort/filter_ledger.csv`
(counts only, no patient data) so the pooled paper can report attrition.

Keep `output/patient_level_data/.id_salt` — it is what lets you map a shared
pseudonym back to a local record number, and it must not be sent. (Pseudonyms
are SHA-256 over salt + record number, truncated to 128 bits.)

To re-export after a policy change, without re-running the model:

```bash
python code/02_model_embedding.py --share-safe-only
```

---

## Coordinating site steps

Drop the received files into `output/patient_level_data/` and rerun step 03:

```bash
python code/03_embedding_analysis.py
```

That is the whole merge. Step 03 globs `embeddings_anchor-<anchor>_*.parquet`,
so a new site is picked up with no code change and no merge step. It takes the
union of columns, so a redacted file sits alongside a local unredacted one; it
insists only on `site, cs_phenotype, icu_los_days, in_hospital_mortality`.

Two things to know:

- **PCA and UMAP are refit on the pooled set**, so coordinates shift when a site
  is added. That is correct — the projection should describe the whole cohort —
  but figures are not comparable across runs with different site membership.
  Regenerate all of them together.
- **Three sites is the cap** for one scatter panel. Colours come from the
  palette slots that pass the all-pairs colourblind gate, and only three do.
  A fourth site raises an error rather than silently picking an unsafe hue;
  facet into small multiples at that point.

### Reading the site plot honestly

If the sites separate cleanly in UMAP, the leading explanation is **not** that
their patients differ. It is that their *data* differ — documentation habits,
lab panel composition, how a unit is labelled, which assessment scales get
charted. The model reads charting practice as readily as physiology.

This is not hypothetical here. With UCMC and NU alone, site already explains
almost all of the geometry:

| label | purity (k=30) | by chance | lift |
|---|---|---|---|
| **site** | **97.1%** | 60.8% | **1.60x** |
| cs_phenotype | 71.5% | 61.6% | 1.16x |
| in-hospital mortality | 80.5% | 72.3% | 1.11x |

Step 03 writes this table every run to
`output/tables/neighbourhood_purity_anchor-*.csv` and warns when site lift
exceeds 1.3x. **Check it when Rush lands.** If site lift stays high, the pooled
UMAP is a picture of three hospitals' charting conventions, and the analysis
that answers the clinical question is a within-site one — or one that explicitly
adjusts for site — rather than the pooled scatter.

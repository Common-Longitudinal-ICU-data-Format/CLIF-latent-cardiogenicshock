# Adding a contributing site

Written for: the analyst running this pipeline at a contributing site, and the
coordinating analyst pooling the results.

Raw CLIF data never leaves the site. The site runs `01_site_aggregate.py`
against its own data and its own copy of the frozen model, and sends back a
**bundle** of aggregate tables. If its agreement permits patient-level
transfer, it also runs `02_site_embeddings.py`, which adds the embeddings to
that same bundle. The coordinating site pools whichever bundles arrive.


---

## Before anything else: which tier are you sending?

The pipeline splits what a site sends in two, and the split is the whole point.

**Tier 1 (`01_site_aggregate.py`) is aggregate.** Patient flow, cohort sizes,
the three phenotype breakdowns, and how coherent each phenotype is in this
site's embedding space. Every number is a count or an average over at least
`--min-cell` patients, so it travels under a consortium's ordinary terms. Every
site runs it.

**Tier 2 (`02_site_embeddings.py`) is patient-level, and is not
de-identified.** It adds `patient/` to the tier-1 bundle, so a site sends
one directory either way. Record numbers are replaced by a salted hash, timestamps are
dropped and age is reduced to five-year bands, which narrows what travels — it
does not make the file anonymous:

> A 1024-d embedding is a learned function of the patient's entire recorded
> history. No one has shown that such representations resist re-identification
> attack, and they should not be treated as de-identified.

Treat the tier-2 export as a **limited data set at best**, and move it under the
same IRB approval and data use agreement you would use to share patient-level
clinical data. Confirm that is in place before running it.

If your DUA does not permit patient-level transfer, **run tier 1 and stop**.
That is a complete contribution, not a degraded one: the consortium gets your
attrition, your phenotype distributions, and your answer to whether the
phenotypes correspond to anything the model sees. The pooled analysis takes
tier-1-only sites in its count tables and its phenotype-coherence comparison,
and leaves them out of the projections.

The full contract — every column permitted to cross a site boundary, with the
argument for each — is in [FEDERATION.md](FEDERATION.md), or print it:

```bash
python code/clifcs/export/contract.py
```

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
python code/01_site_aggregate.py --stage cohort     --phenotype-model /path/to/phenotype_model.json     --seymour-model   /path/to/seymour_model.json     --bhavani-model   /path/to/bhavani_model.json
```

The three `--*-model` flags are **required at a contributing site**. Without
them the site refits its own clusters and "phenotype II", "beta" and "group A"
stop meaning the same thing at each site, which makes every pooled breakdown
uninterpretable.

Read `output/cohort/filter_ledger.csv` before running anything expensive. The
two things that go wrong:

- **An ICU definition that does not match.** Sites label units differently;
  compare the ICU row against total hospitalizations and ask whether it is
  plausible.
- **A table that is present but not populated.** This bit us already: UCMC's
  `clif_ecmo_mcs` names 82% of all hospitalizations and is a concept table, not
  a record of MCS. Check each criterion's sub-count in the ledger for anything
  implausibly large or zero.

### 4. Tier 1 — the rest of what every site sends

```bash
python code/01_site_aggregate.py --from-stage subset --batch-size 8
```

`--batch-size 8` fits an 8 GB card; raise it if you have more. Drop it to 2–4 if
you hit CUDA OOM — `output_hidden_states` keeps every layer.

This runs through to `output/upload_to_box_<SITE>/`, holding counts, the three
phenotype breakdowns, and phenotype coherence in the embedding space. All of it
is aggregate and small-cell suppressed. **Send this directory.**

### 5. Tier 2 — only with a patient-level agreement

```bash
python code/02_site_embeddings.py
```

Adds `output/upload_to_box_<SITE>/patient/`: one row per patient per anchor,
holding the 1024-d embedding and the labels needed to colour and stratify it.
Roughly 45 MB per 10,000 patients.

**It adds to the tier-1 bundle rather than making a second directory**, so the
site still sends exactly one. The unredacted embeddings stay in
`output/patient_level_data_<SITE>/`.

**Do not run this without the IRB approval and DUA in place.** Tier 1 is a
complete contribution on its own, and the pooled analysis degrades rather than
failing: a tier-1-only site appears in every count table and in the
phenotype-coherence comparison, and in no projection.

### 6. Keep the salt

`output/patient_level_data/.id_salt` maps a shared pseudonym back to a local
record number. It must not be sent, and it must be **backed up** — if it is
lost, every future export carries new pseudonyms that cannot be matched to the
ones already sent. (Pseudonyms are SHA-256 over salt + record number, truncated
to 128 bits.)

---


## Coordinating site steps

Drop each received bundle into `output/received/` and run the pooled
scripts with `--role pooled`:

```bash
python code/03_embedding_analysis.py --role pooled
python code/11_trial_alignment.py   --role pooled
```

That is the whole merge. Bundles are discovered by their manifests, so a new
site is picked up with no code change — and the manifests are checked against
each other on the way in: a mismatch in model weights or tokenizer is fatal,
since embeddings from different weights are not in the same space, and a
mismatch in the repository commit warns, since the cohort definition may differ.

Under `--role pooled`, `paths.site_data` raises. The coordinating machine
cannot read raw records even by mistake, which is the mechanism that makes any
of this more than a promise.


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

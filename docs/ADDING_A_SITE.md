# Adding a contributing site

Written for: the analyst running this pipeline at a contributing site.

> ## What is being asked for
>
> **Run `01_site_tables.py` against your own data. Send back
> `output/upload_to_box_<SITE>/`.**
>
> No GPU, no model weights, and raw CLIF data never leaves the site. Everything
> in the bundle is a count, a share or a median over at least 11 patients, so it
> travels under a consortium's ordinary terms.
>
> `02_site_embeddings.py` is patient-level and is **not part of this request**.
> If you want to know what it does and why it is gated, see
> [TIER_2_AND_POOLED.md](TIER_2_AND_POOLED.md). Do not run it without an IRB
> approval and DUA in place.

---

## What tier 1 contributes

Patient flow, cohort sizes, the phenotype breakdowns, baseline characteristics
by coded mechanism, the whole trial and guideline analysis, and per-token
prevalence — how differently this site charts.

The trial half is the substantive one. Eighteen randomised trials across the
cardiogenic, septic and hypovolaemic literatures — SHOCK, IABP-SHOCK II,
CULPRIT-SHOCK, ECMO-CS, ECLS-SHOCK, DanGer Shock, OptimaCC, SOAP II, and the
sepsis and haemorrhage trials — are scored four ways:

| table | what it answers |
|---|---|
| `trial_outcomes_*` | who qualifies, how they did, whether the recommended treatment was delivered, and what share sit inside every published IQR |
| `trial_iqr_position_*` | per trial per variable: our median against the published median and quartiles, and the share below Q1 / inside / above Q3 |
| `trial_guideline_assignment_*` | the single most relevant guideline per patient, and whether they were also inside the population it was tested on |
| `trial_coverage_*` | how many trials' criteria each patient meets, against outcome |

**None of it reads an embedding.** The eligibility predicates and the baseline
variables read raw CLIF, which is why it runs site-side; everything that leaves
is a share or a median. The guideline assignment and the coverage table used to
be computed at the coordinating site, which meant a site sharing nothing
patient-level reported neither, for no reason other than where the code lived.

**It needs no GPU and never loads the GEM weights.** If your site has no GPU, or
no patient-level agreement, tier 1 is the whole of your contribution and it is a
complete one: the consortium gets your attrition, your phenotype distributions,
your trial comparison, and the token prevalences the site effect is decomposed
with. What you will not contribute is the phenotype-coherence measure, which
needs the embeddings and so sits in tier 2.

The full contract — every column permitted to cross a site boundary, with the
argument for each — is in [FEDERATION.md](FEDERATION.md), or print it:

```bash
python code/clifcs/export/contract.py
```

---

## What the site needs

| | |
|---|---|
| CLIF 2.1 tables | `clif_adt, clif_code_status, clif_crrt_therapy, clif_hospital_diagnosis, clif_hospitalization, clif_labs, clif_medication_admin_continuous, clif_medication_admin_intermittent, clif_patient, clif_patient_assessments, clif_patient_procedures, clif_position, clif_respiratory_support, clif_vitals` |
| **Tokenizer** | `tokenizer.yaml` — the one the model was **trained** with. Required |
| Phenotype models | shipped in [`models/`](../models/). Nothing to fetch |
| Hardware | CPU only. No GPU, and the GEM weights are not needed |
| Python | 3.11 or newer |
| Disk | roughly 40× the size of the cohort's raw tables, on scratch |

The tokenizer is the part people get wrong. The numeric bin edges are baked into
the token names, so a tokenizer learned locally maps the same clinical event to
a different id and the token-prevalence tables stop being comparable across
sites. Ship `tokenizer.yaml` with the weights even if you only run tier 1.

Four of the CLIF tables **degrade the run quietly rather than stopping it** if
they are present but unpopulated, so check them rather than assuming:

| missing table | what goes wrong |
|---|---|
| `clif_hospital_diagnosis` | every diagnosis-based trial criterion reads as not met; eligibility and shock type are understated, not errored |
| `clif_patient_procedures` | IABP, ECMO and microaxial flags all read false, so the device trials' concordance is silently wrong rather than absent |
| `clif_crrt_therapy` | renal support is invisible to the tokenizer |
| `clif_code_status` | limitation-of-treatment context is lost |

---

## Site steps

### 1. Environment

```bash
git clone <repo> && cd CLIF-latent-cardiogenicshock
# Python 3.11+ is required. `python` is 3.6 on many clusters, and `venv` will
# build a 3.6 environment silently -- then the install fails on
# cocoa-tokenizer (requires_python >=3.11) with an error that says nothing
# about versions. Check first; if it is too old, `uv python install 3.11`
# gives you one without root.
python3 --version
python3 -m venv ~/venvs/clif-cs             # NOT on a synced or networked drive
source ~/venvs/clif-cs/bin/activate         # Windows: ~/venvs/clif-cs/Scripts/activate
python --version                            # must say 3.11 or newer

# About 200 MB. No torch, no CUDA, no GPU. This is the whole install.
pip install -r requirements-site.txt
```

Do not install `requirements.txt` or `requirements-embed.txt` for this run.
They pull torch, transformers, lightgbm, optuna, xgboost and wandb — about
15 GB — and `01` never loads a model.

### 2. Configure

```bash
cp code/config/paths.example.yaml code/config/paths.yaml
```

Fill in one entry per site you hold, the tokenizer, and a `scratch_home` on fast
local scratch. `paths.yaml` is gitignored.

`models.<key>.home_glob` is tier 2 only and **may point at nothing** — the glob
is resolved on access, not at load, so tier 1 runs without the weights on disk.

Sanity check — this reads config and lists directories, and opens no CLIF table:

```bash
bash hpc/00_preflight.sh
```

It verifies that every path resolves and that each site has the 14 required
tables. The `model` line is reported but cannot fail the preflight; an
unresolved checkpoint is expected if you are running tier 1 only. The
`tokenizer` line **can** fail it, and should.

Worth running even without Slurm — it is the cheapest way to catch a
misconfiguration.

### 3. Cohort — and check the numbers before going further

```bash
python code/01_site_tables.py --stage cohort
```

No flags needed. The three phenotype models in [`models/`](../models/) are
applied by default, which is what a contributing site wants. They were fitted
once on the derivation cohort and transferred: without them the site refits its
own clusters and "phenotype II", "beta" and "group A" stop meaning the same
thing at each site, which makes every pooled breakdown uninterpretable.

Refitting is possible but has to be asked for explicitly with
`--refit-phenotypes`, and is only correct at the derivation site. It used to be
the silent default when the three `--*-model` flags were absent, and that bit
us: at Rush a shell ate the flags, the cohort stage refit its own clusters, and
the only sign was the word "refit" in one log line among warnings that looked
like findings about the data.

Read `output/cohort/filter_ledger.csv` before running anything expensive. The
two things that go wrong:

- **An ICU definition that does not match.** Sites label units differently;
  compare the ICU row against total hospitalizations and ask whether it is
  plausible.
- **A table that is present but not populated.** This bit us already: UCMC's
  `clif_ecmo_mcs` names 82% of all hospitalizations and is a concept table, not
  a record of MCS. Check each criterion's sub-count in the ledger for anything
  implausibly large or zero.

### 4. The rest of tier 1

```bash
python code/01_site_tables.py --from-stage subset
```

No GPU and no batch size: tier 1 loads the *tokenizer*, which is a different
file from the weights. The slow stages are `derive` (clifpy) and `collate`
(cocoa), both CPU and both memory-hungry. Every stage is resumable — pass
`--from-stage <name>` to restart where it stopped, or `--stage <name> --force`
to redo one.

This runs through to `output/upload_to_box_<SITE>/`, holding counts, the
phenotype breakdowns, baseline characteristics by shock type, the four trial
tables and the token-prevalence tables. All of it is aggregate and small-cell
suppressed.

The log for the `trials` stage is worth reading. It prints the guideline
assignment for your cohort, and a line comparing the trial **specificity order**
derived from your eligible shares against the committed one in
`code/config/trials.yaml`. Large drift there is expected and is not an error --
the committed order is the one that was applied, which is what makes your
assignment comparable with every other site's -- but it is a useful signal about
how your population differs.

### 5. Send it

**Send `output/upload_to_box_<SITE>/` — one directory, nothing else.**

`output/patient_level_data_<SITE>/` is local only and unredacted. The two
directories are named so that what may leave is obvious without reading any
code. `output/cohort/` and `output/logs/` are working output and are not sent.

The README has the file-by-file listing of what the bundle contains.

---

## What happens to it

The coordinating site drops each received bundle into `output/received/` and
runs the pooled scripts with `--role pooled`. Bundles are discovered by their
manifests, so a new site is picked up with no code change — and the manifests
are checked against each other on the way in: a mismatch in tokenizer is fatal,
and a mismatch in the repository commit warns, since the cohort definition may
differ. Full procedure in [COORDINATING_SITE.md](COORDINATING_SITE.md).

Two things to know about how your numbers will be read:

- **PCA and UMAP are refit on the pooled set**, so coordinates shift when a site
  is added. That is correct — the projection should describe the whole cohort —
  but figures are not comparable across runs with different site membership.
- **Three sites is the cap** for one scatter panel. Colours come from the
  palette slots that pass the all-pairs colourblind gate, and only three do. A
  fourth site raises an error rather than silently picking an unsafe hue;
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
exceeds 1.3x. If site lift stays high as more sites land, the pooled UMAP is a
picture of several hospitals' charting conventions, and the analysis that
answers the clinical question is a within-site one — or one that explicitly
adjusts for site — rather than the pooled scatter.

**This is why tier 1 matters on its own.** The token-prevalence tables are what
the site effect is decomposed with, and they come from the tokenizer alone. A
site that shares nothing patient-level still contributes to the largest single
finding in this pipeline.

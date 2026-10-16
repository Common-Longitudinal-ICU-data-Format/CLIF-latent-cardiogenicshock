# Running as the coordinating site

Written for: the analyst at the coordinating institution, where the consortium
permits token transfer.

You are doing two jobs on one machine. UChicago holds UCMC's data, so you are a
**contributing site** producing your own bundles, and you are the
**coordinating site** pooling everyone's. The `--role` flag is how one machine
does both; this is not a single linear run.

---

## 0. Before anything: push the repository

The GitHub remote is currently **empty** -- `git ls-remote origin` returns no
refs. No contributing site can clone anything yet.

```bash
git push -u origin federated          # or merge to main first, then push
```

Decide which branch sites clone from and say so when you circulate the link. A
site on a different commit enrols a different cohort; the manifest check warns
about it but cannot fix it.

---

## 1. Set up this machine

```bash
git clone https://github.com/hmhiero/CLIF-latent-cardiogenicshock.git
cd CLIF-latent-cardiogenicshock

python -m venv ~/venvs/clif-cs          # NOT on a synced drive
source ~/venvs/clif-cs/Scripts/activate

pip install -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.5.1+cu121 True
```

Install the CUDA torch **last**. `cotorra` depends on torch unpinned, so a plain
install silently replaces a CUDA build with CPU wheels.

```bash
cp code/config/paths.example.yaml code/config/paths.yaml
```

Fill in your CLIF directory, the model glob, the tokenizer, and a `scratch_home`
on fast local disk. Leave `role: site` -- you override it with `--role pooled`
for the pooled half.

Sanity check, which reads nothing but config:

```bash
python -c "from sys import path; path.insert(0,'code'); from clifcs import load_paths; \
p=load_paths(); print(p.sites, p.model_home, p.tokenizer_yaml)"
```

**You need a GPU** -- not only for your own embedding run, but because the
coordinating role re-embeds every received token bundle.

---

## 2. Produce your own bundles

Tier 1, which you would run even if nothing else were permitted:

```bash
python code/01_site_aggregate.py --batch-size 8 \
    --phenotype-model models/phenotype_model.json \
    --seymour-model   models/seymour_model.json \
    --bhavani-model   models/bhavani_model.json
```

Nine stages: `cohort subset derive collate tokenize winnow embed purity export`.
`derive`, `collate` and `embed` are the slow ones. Resume with
`--from-stage embed` rather than starting over.

**Read `output/cohort/filter_ledger.csv` before running anything expensive.** The
two failures it catches are an ICU definition that does not match your unit
labels, and a table that is present but not populated.

Then tier 2, because your consortium permits tokens:

```bash
python code/02_site_tokens.py
```

You now have three directories, and the names are the policy:

| directory | goes where |
|---|---|
| `output/patient_level_data_UCMC/` | **nowhere.** Local embeddings. |
| `output/upload_to_box_UCMC/` | Box. Summary tables, ~19 KB. |
| `output/token_level_data_UCMC/` | Box, under the DUA. Sequences, ~25 MB. |

Back up `output/patient_level_data/.id_salt`. It is gitignored and lives under
`output/`, so clearing that directory gives every future export new pseudonyms
that cannot be matched to what you already sent.

---

## 3. Tell contributing sites what to run

Send them the repo URL, the branch, and this:

```bash
# tier 1 -- everyone
python code/01_site_aggregate.py --batch-size 8 \
    --phenotype-model models/phenotype_model.json \
    --seymour-model   models/seymour_model.json \
    --bhavani-model   models/bhavani_model.json

# tier 2 -- ONLY with the patient-level DUA executed
python code/02_site_tokens.py
```

The three `--*-model` flags are **not optional at a contributing site**. Without
them the site refits its own clusters and "phenotype II", "beta" and "group A"
stop meaning the same thing anywhere, which makes every pooled breakdown
uninterpretable. The models ship in `models/`, so they arrive with the clone.

Ask for `upload_to_box_<SITE>/` always, and `token_level_data_<SITE>/` only once
the agreement is executed. **A site that sends tier 1 only is a full
contributor** -- it appears in every count table and in the phenotype-coherence
comparison, and in no projection.

---

## 4. Receive

Drop what arrives into `output/received/`, keeping the directory names:

```
output/received/
  upload_to_box_UCMC/          your own, copied in
  token_level_data_UCMC/
  upload_to_box_NU/
  token_level_data_NU/
  upload_to_box_RUSH/          tier 1 only -- RUSH has no DUA yet
```

Bundles are found by their manifests, not by directory name, so a rename in
transit is harmless. On ingest the manifests are checked against each other:

| check | on mismatch |
|---|---|
| schema version, model key, weights, tokenizer | **fatal** -- embeddings from different weights are not in the same space |
| repository commit | warning -- the cohort definition may differ |
| dirty working tree | warning -- the SHA does not describe the code |
| manifest row count vs file | **fatal** -- truncated copy |
| duplicate `row_key` | **fatal** -- a bundle was ingested twice |

---

## 5. Run the pooled analysis

```bash
python code/03_embedding_analysis.py --role pooled --anchor 24h
```

`--role pooled` does two things. It makes `paths.site_data` **raise**, so this
run cannot read raw records even by mistake; and it points the loader at
`output/received/` instead of your local files.

The first pooled run **re-embeds every received token sequence** with the frozen
model. That is the point of tier 2: you receive the model's input and derive its
output yourself. Roughly fifteen minutes per anchor for 34,000 patients, then
cached to `output/patient_level_data/reembedded_anchor-<anchor>_<SITE>.parquet`,
so every later script is immediate.

This is exact, not approximate. The model is frozen in eval mode under
`torch.inference_mode()`, so it is a deterministic function of token ids and
elapsed times. Checked against embeddings computed directly at the site over 512
patients: **maximum absolute difference 0.0**, bit for bit.

Then the rest, in any order:

```bash
python code/04_site_effect.py      --role pooled
python code/05_site_correction.py  --role pooled
python code/06_downstream_tasks.py --role pooled
python code/08_token_space.py      --role pooled
python code/10_site_features.py    --role pooled
python code/11_trial_alignment.py  --role pooled
python code/07_report.py                        # assembles whatever exists
```

---

## 6. Read these three first

**`output/received/upload_to_box_*/purity/phenotype_purity.csv`** -- whether the
phenotypes correspond to anything the model sees, per phenotype, per site. The
one result that needs no patient-level transfer at all, so it is available from
every contributor including tier-1-only ones. Zweck III cardiometabolic runs
2.41x at UCMC and 3.34x at NU; Seymour gamma is the weakest of its four, which is
what the missing CRP and ESR predict.

**The site plot.** Site explains ~94% of neighbour purity against ~61% by
chance. If sites separate cleanly the leading explanation is not that their
patients differ -- it is that their *data* differ, and the model reads charting
practice as readily as physiology. Compare within site before reading anything
clinical into the pooled cloud.

**`filter_ledger.csv` across sites.** Attrition that differs sharply between
sites usually means an ICU definition or a table-population problem, not a
population difference.

---

## The trial comparison comes from tier 1, not from tokens

Everything the trial analysis reports is a share or a median over a site's own
patients, so all of it is computed site-side and arrives in
`upload_to_box_<SITE>/trials/`:

| table | holds |
|---|---|
| `trial_outcomes_anchor-*.csv` | per trial and stratum: n eligible, share eligible, mortality, median ICU LOS, share treated in line with the guideline, mortality split by concordance, share inside every published IQR with its null |
| `trial_iqr_position_anchor-*.csv` | per trial and variable: our median against theirs, and the share below Q1 / inside / above Q3 |
| `trial_criteria.csv` | each trial's published criteria and whether this pipeline can apply it |

Both are broken down **overall and by Zweck, Seymour and Bhavani phenotype and
by coded shock type**. All 13 trials are included -- 8 cardiogenic, 3 septic
(VASST, SEPSISPAM, ADRENAL) and 2 hypovolaemic (CRASH-2, Villanueva) -- because
the septic-coded patients are 18% of this cohort and reporting only the
cardiogenic trials would leave them uncovered by construction.

**A site that sends tier 1 only contributes the whole trial analysis.** That is
worth saying to a site still negotiating its DUA: they are not holding up the
trial work.

Read `trial_iqr_position` first. The blood-pressure gap replicates
independently at both current sites -- OptimaCC systolic 77.2% below the
published first quartile at UCMC and 78.3% at NU, against a 25% null -- and
`share_concordant` carries its own finding: only 29% of ADRENAL-eligible septic
shock patients received hydrocortisone, at both sites.

---

## What a pooled run does not give you

**Trial regions in embedding space.** Fitting a region from a trial's eligible
patients needs the pooled embeddings, so that half of 11 stays at the
coordinating site. The *comparison* half -- eligibility, IQR position,
concordance, outcomes -- moved to tier 1 and no longer needs you at all.

**06's clinical baseline.** The embedding-vs-site-correction comparison
survives, including the cross-site transfer that is the point of it. The
physiology baseline it is scored against does not: `lab_*` and `vtl_*` are
measured values.

**Trajectories (09).** Runs site-side by consortium decision; pool the
`gradient_alignment` and `position_encodes` scalars and the per-site flow
figures, not the inputs.

---

## Things that will bite

**`MAX_PLOTTED_SITES = 3`.** Scatter panels raise rather than mis-colouring,
which is right, but a consortium is more than three sites. Facet into small
multiples before a fourth token bundle arrives.

**The projection moves under every site's feet.** PaCMAP is refit on the pooled
set, so every figure changes when a site lands, and PaCMAP has no clean
`transform`. If sites arrive on different schedules, pin a reference projection
or regenerate every figure together and never mix runs.

**Phenotype scalers come from the derivation site**, not from pooled moments. A
one-round exchange of per-site `n`, `sum(x)`, `sum(x^2)` would make them
genuinely pooled. Not implemented.

**The sepsis denominator is a diagnosis code** used 1.3x more often at NU than
UCMC. Seymour and Bhavani shares are reported per site and must never be pooled
into one percentage. See
[FEDERATION.md](FEDERATION.md#the-sepsis-phenotypes-and-what-is-wrong-with-them).

**A token sequence is not de-identified.** It is a re-serialisation of the
record -- which labs were drawn, in what order, how often, against what clock.
More disclosive than the embedding derived from it, because an embedding is a
lossy projection and this is not. Hold `token_level_data_*` to the same standard
as patient-level clinical data, and say so to anyone who asks whether tokens are
"just summaries".

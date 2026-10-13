# CLIF latent space — shock

Exploring where shock patients sit in the embedding space of a frozen CLIF
foundation model, using data from two contributing sites.

The cohort is **undifferentiated shock**, selected on physiology alone, with the
coded mechanism attached afterwards as a labelling. The question is whether the
model's latent space separates septic from cardiogenic from hypovolaemic shock —
which filtering to one of them would make unanswerable.

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
| [code/03_embedding_analysis.py](code/03_embedding_analysis.py) | PCA + UMAP + PaCMAP, and the scatterplots |
| [code/11_trial_alignment.py](code/11_trial_alignment.py) | Where the guideline trials' populations sit, who falls inside them, and who falls inside none |

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
shock registry would hold. That is the design, not a limitation: see
[Shock type](#shock-type) below.

A `cardiac_origin` flag is computed and reported for every patient but **not
applied** (`cardiac_origin.enabled: false`). Applying it would cut the cohort to
2,219 UCMC and 5,104 NU patients. It is satisfied by any of:

| arm | UCMC | NU |
|---|---|---|
| inotrope infusion (dobutamine, milrinone) | 1,335 | 3,027 |
| MCS procedure (ECMO, IABP, pVAD) | 846 | 1,168 |
| peak troponin ≥500 ng/L near onset | 813 | 1,909 |
| **any** | **2,219 (24.3%)** | **5,104 (20.4%)** |

Diagnosis codes are deliberately not used: a diagnosis code asserts a condition,
whereas an infusion, a procedure and a troponin record something that was
actually done or measured.

The troponin threshold is high on purpose. UCMC runs troponin-T and NU
troponin-I — different assays, different reference limits — so one absolute cut
is not equivalent across them. At the ~14 ng/L URL the sites diverge (44.1% vs
29.2%) and the criterion catches the sepsis and PE patients the flag exists to
exclude; at 500 ng/L they behave alike (8.9% vs 7.3%).

**`clif_patient_procedures` at UCMC is only half usable.** Its CPT rows carry
1.72M distinct ids, none of which key on `hospitalization_id`; only the
ICD-10-PCS rows join (98.2% of the cohort). UCMC's MCS ascertainment is
therefore ICD-10-PCS-only, and its echo/cath/PCI are unavailable, while NU's CPT
joins at 99.8%. `01` checks per-format coverage and warns rather than letting an
unjoinable code system contribute zero silently.

**`clif_ecmo_mcs` is not read anywhere.** It is a concept table at both sites
and is not reliably populated — at UCMC it names 136,565 of 166,814
hospitalizations (82%) and carries no device or flow columns. MCS is therefore
taken from `clif_patient_procedures` instead, and MCS is not part of
criterion (1): MCS without either sustained hypotension or a vasoactive infusion
occurred in 3 of 34,142 patients, so nothing is lost by leaving it out.

## Shock type

What the coders said the mechanism was, from `clif_hospital_diagnosis`, attached
to every enrolled patient by
[code/clifcs/shock_type.py](code/clifcs/shock_type.py). **A labelling, never a
filter.**

Codes overlap, so a patient carrying more than one of septic, cardiogenic or
hypovolaemic is `mixed`; `other shock` is R57.8/R57.9 plus the rare mechanisms.
Matching is by code prefix across ICD-10-CM and ICD-9-CM without consulting
`diagnosis_code_format` — ICD-9 shock codes are numeric and ICD-10's begin with a
letter, so the prefixes cannot collide, and the match does not then depend on a
column NU populates with two values and UCMC with one.

| shock type | n | share | mortality | median ICU LOS |
|---|---|---|---|---|
| septic (R65.21) | 6,189 | 18.1% | 29.6% | 4.3 d |
| cardiogenic (R57.0) | 2,560 | 7.5% | 25.2% | 5.2 d |
| hypovolemic (R57.1) | 1,314 | 3.8% | 12.6% | 3.2 d |
| mixed (≥2 of the above) | 2,508 | 7.3% | 43.3% | 7.3 d |
| other shock (R57.8/R57.9, …) | 2,945 | 8.6% | 26.8% | 3.8 d |
| **none coded** | **18,626** | **54.6%** | **6.2%** | 2.5 d |

### Read the uncoded group before anything else

**Over half the cohort carries no shock code at all, and that group is barely
sick** — 6.2% in-hospital mortality against 25–43% in every coded group, and a
median ICU stay of 2.5 days.

The most likely reading is that criteria 1 and 2 are over-inclusive. A single
run of SBP < 90 lasting half an hour, or a short phenylephrine infusion around a
procedure, satisfies the hemodynamic criterion without anyone treating the
patient as being in shock; a lactate of 2.0 or a 0.3 mg/dL creatinine rise then
satisfies the end-organ criterion. So the physiological definition is doing what
it was specified to do and catching a large population the clinical teams did
not consider shocked.

Treat the coded groups as the shock cohort and the uncoded remainder as the cost
of a purely physiological definition. It is also the reason the `shock_type`
panels are worth more than the phenotype panels here: they are the only thing in
the pipeline that distinguishes the two.

### Coding practice differs between the sites

Septic shock is the label on 20.3% of NU's cohort and 12.3% of UCMC's, while
cardiogenic shock runs the other way (5.3% NU, 13.5% UCMC). `summarize` warns
whenever the septic share differs by more than 1.5x between sites, which it
currently does at 1.7x. Compare composition **within** a site before reading
anything across them.

### Why this reads diagnosis codes at all

It is a deliberate exception to the rule applied everywhere else in this
pipeline. The cohort is built from things that were done or measured — an
infusion, a procedure, a laboratory value — on the argument that a code asserts
a condition rather than recording an event. That argument still holds. But the
mechanism of shock is a clinical judgement with no corresponding measurement, so
the coded diagnosis is the only evidence available; it only ever describes
patients already enrolled on physiology, and never selects one.

Two further caveats: `clif_hospital_diagnosis` is hospitalization-level, so a
code entered on day 20 is indistinguishable from one entered on admission; and
`poa_present` would narrow that but is populated for every UCMC row and only 61%
of NU's, so it is reported rather than required
(`shock_type.require_present_on_arrival`).

## Guideline fit

Every cardiogenic shock recommendation rests on a trial that enrolled a narrow
slice of the people who present in shock.
[code/11_trial_alignment.py](code/11_trial_alignment.py) asks where those slices
sit in the latent space, which of our patients fall inside them, and who falls
inside none. The eight trials and the populations they enrolled are in
[code/config/trials.yaml](code/config/trials.yaml); the eligibility logic is in
[code/clifcs/trials.py](code/clifcs/trials.py), one commented predicate per
criterion.

Two questions, kept apart because they are easy to conflate:

- **Eligibility** — would the trial have enrolled this patient? A boolean from
  the published criteria.
- **Proximity** — do they resemble the patients it did enrol? The share of the
  patient's 30 nearest neighbours (50-d PCA space) that the trial would have
  taken, against the trial's cohort-wide base rate. Same neighbourhood measure
  `03` uses for site purity, so the numbers are comparable.

At the 24 h anchor:

| trial | eligible | mortality | inside, not eligible | centroid separation |
|---|---|---|---|---|
| SHOCK 1999 | 4,397 (12.9%) | 28.7% | 1,072 | 0.17 |
| SOAP II 2010 | 30,497 (89.3%) | 17.4% | 45 | 0.03 |
| IABP-SHOCK II 2012 | 4,397 (12.9%) | 28.7% | 1,072 | 0.17 |
| CULPRIT-SHOCK 2017 | 4,397 (12.9%) | 28.7% | 1,072 | 0.17 |
| OptimaCC 2018 | 3,990 (11.7%) | 30.2% | 1,405 | 0.18 |
| ECMO-CS 2022 | 7,756 (22.7%) | 31.6% | 1,769 | 0.19 |
| ECLS-SHOCK 2023 | 1,163 (3.4%) | 43.4% | 1,204 | 0.32 |
| DanGer Shock 2024 | 199 (0.6%) | 37.7% | 192 | 0.27 |

### Eligible is not the same as represented

Enrolment criteria are thresholds, so they admit a much wider range than a trial
ends up containing. The published baseline tables in
[code/config/trials.yaml](code/config/trials.yaml) are what the result was
actually observed in. Seven of the eight are transcribed in full (SHOCK published no
interquartile ranges at all, so it contributes only an approximate one for age); the `criteria` block alongside them records each
trial's published inclusion and exclusion criteria and whether this pipeline can
apply it — 19 of 39 in full, 14 partly, 6 not at all.

**The discrepancy is blood pressure, and essentially nothing else.** Three
independent trials give the same answer:

| trial | systolic, ours vs theirs | below their Q1 (null 25%) |
|---|---|---|
| OptimaCC | 78 vs 104 (94–120.5) | **77.9%** |
| ECLS-SHOCK | 73 vs 96 (80–120) | **66.8%** |
| CULPRIT-SHOCK | 79 vs 100 (84–125) | **62.6%** |
| ECMO-CS | 75 vs 85 (80–100) | **58.9%** |
| DanGer Shock | 76 vs 83 (72–91) | 37.9% |

### The gap tracks how selectively the trial enrolled

Mean arterial pressure separates the trials cleanly, and SOAP II is the control
case:

| trial | their MAP | ours | below their Q1 (null 25%) | enrolled |
|---|---|---|---|---|
| OptimaCC | 72 | 58 | **79.1%** | post-PCI AMI shock |
| CULPRIT-SHOCK | 76 | 58 | **67.2%** | AMI + multivessel disease |
| ECMO-CS | 63 | 56 | 48.1% | severe or deteriorating CS |
| DanGer Shock | 64 | 56 | 44.6% | STEMI, SBP < 100 |
| SOAP II | 58 | 61 | **16.9%** | any shock on a vasopressor |

SOAP II enrolled any shock requiring a vasopressor, of any cause, and its own
population sat at a MAP of 58 — genuinely hypotensive, like ours. The
specialised cardiogenic shock trials enrolled at 63 to 76. **The more narrowly a
trial selected, the further its population sits from routine ICU shock**, and the
one pragmatic all-comers trial is the only one whose patients ours resemble
(16.9% is *below* the 25% null).

That has an uncomfortable corollary. The single recommendation whose trial
population matches these patients — norepinephrine rather than dopamine — is
also the one resting on the thinnest evidence for cardiogenic shock
specifically: a 280-patient subgroup of a 1,679-patient trial that was 62%
septic, and whose baseline characteristics were never published. The SOAP II
table transcribed here is the whole trial, not that subgroup, and is the only
one in this file describing a different disease from the patients compared
against it.

**And it is not that our patients are less supported.** ECMO-CS is the one trial
that published which drugs were running at randomisation:

| agent | ours | ECMO-CS |
|---|---|---|
| norepinephrine | 86.4% | 85.5% |
| vasopressin | 48.3% | 35.0% |
| epinephrine | 29.6% | 3.4% |
| dobutamine | 10.5% | **54.7%** |
| milrinone | 6.8% | **32.5%** |

Vasopressor use matches or exceeds theirs; what differs is **inotropes**, by a
factor of five. So the pressure gap is not lighter vasopressor treatment. The
readings left are genuinely deeper shock, a cardiology-led trial population
managed with far more inotropic support than an undifferentiated ICU shock
cohort receives, or both.

Age, weight, height,
body-mass index, lactate, arterial pH and heart rate show **no net shift** —
each sits within a few points of the 25% a representative sample would put below
a first quartile.

The reason is visible in the trials' own numbers. CULPRIT-SHOCK enrolled at a
median systolic of 100 mmHg and ECLS-SHOCK at 96, despite both requiring
"SBP < 90 for > 30 min **or** catecholamines": their populations were largely
supported on catecholamines at a near-normal pressure. DanGer Shock, whose
threshold is SBP < 100, enrolled at 83 and is correspondingly the least shifted.
Our criteria-matching patients sit at 73–79. A recommendation tested there is
being read across to patients in considerably deeper shock.

**Do not read "inside every published IQR" as representativeness.** An
interquartile range holds half the trial's own population by construction, so
even a perfectly representative cohort would have only 0.5^k inside all k of
them — 50% at one variable, 6.2% at four, 0.8% at seven. Observed over null:

| trial | k | inside all | null | ratio |
|---|---|---|---|---|
| SHOCK | 1 | 36.1% | 50.0% | 0.72 |
| IABP-SHOCK II | 4 | 7.8% | 6.2% | 1.25 |
| CULPRIT-SHOCK | 7 | 2.4% | 0.8% | 3.06 |
| ECLS-SHOCK | 7 | 0.6% | 0.8% | 0.77 |
| DanGer Shock | 5 | 5.0% | 3.1% | 1.61 |

Every ratio is between 0.7 and 3.1, so on that metric the eligible patients are
about as concentrated as a representative sample. The column falls with every
variable added whether or not anything differs, which is why it is reported with
its null beside it and why the per-variable directional shifts are the result.

Distance is measured in interquartile ranges because that is all a paper
supports: it reports a median and an IQR, not a covariance, so there is no way
to weight variables against one another or to account for their correlation.

**Only `measured` variables count toward the judgement.** A trial's
medical-history rows come from a case-report form and ours from ICD codes, and
they do not measure the same thing in either direction. Restricting history
flags to diagnoses present on arrival helped and did not fix it:

| | ours | published | why it is not comparable |
|---|---|---|---|
| heart failure | 54.3% | 9.3% | POA-flagged `I50` includes the index event's acute failure |
| prior MI | 1.0% | 16.1% | `poa_present` is seldom set on a history code |
| prior PCI | 0.3% | 19.2% | same |

`VARIABLE_CONFIDENCE` in [code/clifcs/trials.py](code/clifcs/trials.py) marks
these `ascertainment`; they are reported and never used to decide whether a
patient is inside the tested population. Age, the pressures, heart rate,
lactate, pH, creatinine, weight, height, BMI and sex are what drive it.

Left out rather than compared: ejection fraction and coronary anatomy
(unavailable at both sites), oliguria (no intake/output table), and peripheral
artery disease for ECLS-SHOCK — severe PVD is one of its exclusions, so our
eligible set is 0% by construction and the comparison would measure the filter
rather than the patients.

### Most of the cohort is covered by nothing

**62.6% of patients meet no specific trial's criteria and have no
trial-enriched neighbourhood.** They run 8.7% in-hospital mortality against
27.2% for the covered, and **72.9% of them also carry no shock diagnosis code** —
two independent measures of "not the population the evidence is about" agreeing
on largely the same 13,574 patients. Even among the cardiogenic-coded patients,
**26.3% are covered by nothing**.

Read with [Shock type](#shock-type), the conclusion is the same from both
directions: criteria 1 and 2 enrol a large, mild population that neither the
coders nor the trialists would recognise as cardiogenic shock.

### Three trials collapse into one

Once the criteria we cannot evaluate are dropped, SHOCK, IABP-SHOCK II and
CULPRIT-SHOCK all become "acute MI plus shock" and select the identical 4,397
patients. The figure labels the duplicates rather than implying three
independent pieces of evidence.

What cannot be applied, and so makes every count a **superset**:

| trial | criterion dropped | why |
|---|---|---|
| SHOCK | shock due to LV failure | no echo table at either site |
| DanGer Shock | LVEF < 45% | same |
| CULPRIT-SHOCK | multivessel disease | no coronary anatomy in CLIF 2.1 |
| IABP-SHOCK II | planned early revascularisation | not recorded |
| OptimaCC | after successful PCI | not recorded |
| ECMO-CS | cardiac index / SvO2 severity | not recorded |
| ECLS-SHOCK | CPR > 45 min | arrest is coded, its duration is not |

### Why proximity is a neighbourhood rate and not a distance

The first two versions measured each patient's distance to the centroid of a
trial's eligible patients — Mahalanobis under each region's own covariance, then
Euclidean under one shared whitening. Both returned a median distance of 7.1 and
a 95th-percentile radius of **8.5 for all eight trials**, which is the chi
quantile for 50 dimensions and says nothing about any trial. Distances
concentrate as dimension grows, and the trial centroids sit far closer to the
global centroid than the patients sit to any of them.

`centroid_separation` now reports that directly, because it is a finding rather
than a nuisance: at 0.03–0.32 within-group spreads, **the trial populations are
not distinct regions of the latent space at all.** Local neighbourhood structure
survives in high dimensions where centroid distance does not, which is why the
measure is a neighbour rate — and it is also what `03` already uses.

The "inside a region" threshold is the point at which a neighbourhood holds more
trial-eligible patients than chance would put there (upper binomial tail at the
base rate, α = 0.01). A fixed percentile was tried first and degenerates for the
rare trials: DanGer Shock covers 0.6%, so almost every ineligible patient has
zero eligible neighbours, the 95th percentile sits at one neighbour in thirty,
and 4,381 patients were called "inside the DanGer Shock region" on the strength
of a single neighbour. The binomial rule raises the bar to three as prevalence
falls.

### A circularity worth naming

A trial region is fitted from **our** eligible patients, never from the trial's —
there are no embeddings for trial participants and never will be. So "inside the
SHOCK region" means "resembles the patients here who would have qualified for
SHOCK", not "resembles the patients SHOCK enrolled". What links the two is the
representativeness table (`trial_representativeness_*.csv`), which puts our
eligible subset beside the enrolled population on the few variables trials
report. Where they diverge — our ECLS-SHOCK-eligible patients are 37.9% women
against the trial's 18.7% — the region is ours and the trial's name on it is a
label of convenience.

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

## Projections: PCA, UMAP and PaCMAP

`03` computes all three every run and saves all three — coordinates to
`projections_anchor-*.parquet`, figures to `figures/<model>/anchor-*/`. **The
report shows PaCMAP**; UMAP is kept beside it.

PaCMAP ([Wang et al. 2021](https://github.com/YingfanWang/PaCMAP)) is built to
preserve global structure alongside local neighbourhoods, where UMAP optimises
the local graph and will tear apart or fuse regions that are genuinely adjacent.
That matters most in `09`, whose whole subject is *motion* across the layout: a
projection that invents distance invents displacement with it.

Both are fitted on the same 50-d PCA scores with the same neighbour count
(`N_NEIGHBORS = 30`), so the pair is a comparison of the two algorithms rather
than of two neighbourhood scales. PaCMAP is told `apply_pca=False` because it is
already receiving PCA scores. Where the two disagree, neither is the truth — it
is a sign the layout carries less information than it appears to, which is what
the PCA panel is there to settle.

`09` takes `--projection {pacmap,umap,pca}` and defaults to PaCMAP. It names its
flow figure after whichever projection it drew (`trajectory_flow_pacmap.png`), so
a field cannot be shown under the wrong axis labels.

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

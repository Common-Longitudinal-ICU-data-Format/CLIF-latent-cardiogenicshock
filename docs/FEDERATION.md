# Running this pipeline federated

Written for: the consortium's data committee, and the analyst at each
contributing site.

Raw records never move. Each site runs the pipeline on its own data and sends a
**bundle**: aggregate count tables, plus — where its agreement permits — latent
embeddings and a minimal set of ordered token labels. The coordinating site
holds nothing else and, by construction, cannot read anything else.

---

## The two roles

Every script takes `--role`, defaulting to `role:` in `paths.yaml` and then to
`site`.

| | `site` | `pooled` |
|---|---|---|
| holds raw CLIF tables | yes | **no** |
| `paths.site_data` | the data directories | **raises `RoleViolation`** |
| reads | its own records | `output/received/*/` only |
| runs | 01, 02 | 03, 04, 05, 06, 08, 09, 10, 11 |

Verified under the pooled role: **03, 04, 05, 06, 10, 11**. **07** assembles the
report from whatever tables the others wrote and is role-agnostic. **08** and
**09** need the sequences, which tier 2 supplies, so they run at the
coordinating site too.

`paths.site_data` raising is the enforcement. It is not a convention or a note
in a README — a pooled-role run that tries to reach records stops. It raises
rather than returning an empty mapping deliberately: an empty mapping would make
every per-site loop produce nothing, which looks exactly like "that site
contributed no patients" and is the most plausible way for this design to be
quietly wrong.

---

## The export contract

[`code/clifcs/export/contract.py`](../code/clifcs/export/contract.py) is the
machine-readable form of the policy. Print it:

```bash
python code/clifcs/export/contract.py
```

Every column permitted to cross a site boundary is named there with the argument
for why. **Anything not named is refused**, at both ends:

- at the **sending** site, before a file is written, so a column added upstream
  cannot leak by being picked up automatically;
- at the **coordinating** site, on read, so a bundle carrying more than the
  contract allows is refused rather than quietly analysed.

Validating only on write would be trusting every site to run the current code.
Validating only on read would mean the data had already left. Both check, which
is the only arrangement where a disagreement is visible to someone.

It is an allowlist, not a denylist, because the failure being guarded against is
somebody adding a useful column to the cohort frame and it travelling because
nothing said it could not. A denylist cannot see a column nobody thought of.

### What travels

```
output/
  patient_level_data_<SITE>/   LOCAL ONLY -- embeddings. Never shared.
  upload_to_box_<SITE>/        TIER 1 -- summary tables only.
    manifest.json
    counts/                    attrition, phenotype breakdowns, availability
    purity/                    per-phenotype coherence in GEM space
    token_levels.csv           the rank -> value-range legend
    patient/                   TIER 2 -- embeddings. Needs a DUA.
      embeddings_anchor-{24h,onset}.parquet
```

Tier 2 adds `patient/` to the tier-1 bundle rather than making a second
directory, so a site has exactly one thing to upload whichever tier it is
sending. Whether that one directory carries patient-level data is recorded in
its manifest as `shares_patient_level`, and is visible from the presence of
`patient/`.

The directory names carry the policy. Someone about to upload a folder can see
which one it is without reading any code, which is the point.

## The two site tiers

**Tier 1 — [`01_site_aggregate.py`](../code/01_site_aggregate.py).** Every site
runs it. Cohort → tokenize → embed → purity → export, as resumable stages.
Everything it writes to `upload_to_box_<SITE>/` is a count or an average over at
least `--min-cell` patients. This is the classic CLIF federated contribution.

It still needs the GPU, because the interesting part of tier 1 is the
phenotype-coherence measure and that needs the embeddings. They are computed
into `patient_level_data_<SITE>/` and **stay there**.

**Tier 2 — [`02_site_embeddings.py`](../code/02_site_embeddings.py).** Only a
site whose agreement permits patient-level transfer runs it. It adds
`patient/` to the bundle tier 1 already wrote: one row per patient per anchor,
holding the 1024-d embedding plus the labels needed to colour and stratify it
-- the three phenotypes, the coded shock type, the outcomes, trial
eligibility, token-family counts, ten last-pre-anchor token bins, and age as a
five-year band.

The copy in `patient/` is a redaction. The embeddings the model actually
produced stay in `patient_level_data_<SITE>/`, because those carry the record
number, the patient id and the onset timestamp.

Two scripts rather than one with a flag, because a flag invites somebody to pass
it without the question having been asked.

## What tier 2 actually contains

The model's entire input for each patient. Not the record:

- **No measured values.** A token is a decile bin fitted at training time, so
  the stream says `LAB-RES//creatinine_Q7` and never `2.3 mg/dL`.
- **No wall-clock times.** `times`, the timestamp of every event, sits in the
  same source file and is not exported. `s_elapsed_past` carries seconds since
  the start of the record instead — relative, and the same class of quantity as
  `hours_admission_to_cs_onset`.
- **No record numbers, exact age, race or ethnicity**, all of which also sit in
  that file.

What it *is*, stated plainly: a re-serialisation of the record. Which labs were
drawn, in what order, how often, against what clock, is all still there. **It is
the most disclosive artefact this pipeline produces — more so than the embedding
derived from it**, because an embedding is a lossy projection and a sequence is
not. It must not be described as de-identified, and it moves under a
patient-level DUA or not at all.

## Why sequences rather than embeddings

Because the coordinating site can reproduce the embeddings from the sequences
exactly, and cannot go the other way.

The model is frozen, in eval mode, under `torch.inference_mode()` — a
deterministic function of token ids and elapsed times. Re-embedding a received
bundle was checked against the embeddings the site computed directly, over 512
patients: **maximum absolute difference 0.0**, bit for bit.

So shipping sequences costs nothing and buys re-anchoring at a different cut
point, reading a different layer, different pooling, and re-running everything
after a model update without asking anyone to re-export. A fixed 1024-d vector
answers one question; the sequence answers the ones nobody has asked yet.

It is also smaller. UCMC's two token files are 25 MB against 45 MB of
embeddings, because 783 median token ids compress better than 1,024 float32s.

Re-embedding happens automatically on the first pooled run and is cached to
`patient_level_data/reembedded_anchor-<anchor>_<SITE>.parquet`; it is a GPU job,
so the coordinating site needs one.

## Small-cell suppression

`counts/` tables pass through
[`clifcs/export/suppress.py`](../code/clifcs/export/suppress.py) at
`--min-cell` (default 11, the CMS cell-size threshold). Three things happen, and
doing only the first is the usual mistake:

1. **Primary** — blank any cell holding 1 to `min_cell - 1`.
2. **Complementary** — if exactly one cell in a row is blanked and the row total
   is published, subtraction returns it. So a second cell goes too: the
   next-smallest, which loses the least.
3. **Derived statistics inherit it** — a mortality of 1.000 on n = 3 names three
   patients' outcomes whether or not the 3 is printed.

Suppressed cells are **null, not zero**. A zero claims nobody was in the cell,
which is a different and sometimes very interesting statement — "no patient at
this site was assigned Bhavani D" is a data-quality finding, and conflating it
with "between one and ten were" would hide it.

---

## Site procedure

```bash
# once
cp code/config/paths.example.yaml code/config/paths.yaml   # set role: site

# tier 1 -- every site runs this
python code/01_site_aggregate.py --batch-size 8 \
    --phenotype-model <...>/phenotype_model.json \
    --seymour-model   <...>/seymour_model.json \
    --bhavani-model   <...>/bhavani_model.json

# tier 2 -- ONLY with a patient-level DUA in place
python code/02_site_embeddings.py
```

Then send `output/upload_to_box_<SITE>/` -- one directory, whichever tier
if tier 2 was run. Tier 1 alone is a complete, usable
contribution: counts, the three phenotype breakdowns, and phenotype
coherence in GEM space.

**The three `--*-model` flags are required at a contributing site.** Without
them the site refits its own clusters and "phenotype II", "beta" and "group A"
stop meaning the same thing at each site, which makes every pooled breakdown
uninterpretable. The model files carry the feature set, the transform, the
derivation cohort's moments and the centroids — all four, because a site that
standardised against its own moments would re-centre the space and move patients
between phenotypes without anything clinical changing.

If your DUA does not permit patient-level transfer, do not run tier 2. The
bundle then carries `counts/` and `purity/` only, and the pooled side degrades
rather than failing: that site contributes to every count table, to the
phenotype-coherence comparison, and to no projection.

### Back up `.id_salt`

`output/patient_level_data/.id_salt` is created on first export and **must be
backed up**. It is gitignored and lives under `output/`, so a site that clears
its output directory and re-exports produces an entirely new set of `row_key`s
for the same patients — which the coordinating site cannot distinguish from a
new cohort, and would double-count. The manifest carries a fingerprint of the
salt (a hash of a distinct string, so it cannot seed a dictionary attack on the
pseudonyms) so a change is at least detectable.

---

## Coordinating site procedure

```bash
# paths.yaml: role: pooled, an empty `sites:` block, model + tokenizer set
mkdir -p output/received
cp -r <received>/upload_to_box_UCMC    output/received/
cp -r <received>/token_level_data_UCMC output/received/   # if they sent one

python code/03_embedding_analysis.py --role pooled   # re-embeds, then analyses
python code/11_trial_alignment.py   --role pooled
```

The first pooled run **re-embeds the received sequences**, which needs a GPU and
takes roughly fifteen minutes per anchor for 34,000 patients; the result is
cached, so later runs are immediate.

The coordinating machine needs the model and tokenizer for two reasons now: to
re-embed at all, and to check that the bundles were produced under the weights
it is about to interpret
them against.

Bundles are checked against each other on ingest:

| check | on failure |
|---|---|
| schema, model key, weights, tokenizer | **fatal** — embeddings from different weights are not in the same space |
| git SHA | warning — the cohort definition may differ; re-run one site |
| dirty working tree | warning — the SHA does not describe the code |
| manifest row count vs file | **fatal** — the copy is truncated |
| duplicate `row_key` | **fatal** — a bundle was ingested twice |
| two bundles claiming one site | **fatal** |

---

## What the federated split costs

Three things, named honestly.

**The published-baseline comparison is NOT lost.** An earlier version of this
document said it was, and that was wrong. What cannot travel is the measured
*value*; every quantity the comparison needs is a share or a median over values,
which is a statistic rather than a value. So it is computed site-side by
[`clifcs/trial_aggregate.py`](../code/clifcs/trial_aggregate.py) and shipped in
tier 1:

- meets the inclusion criteria — count and share, per trial;
- position against each published interquartile range — share below Q1, inside,
  above Q3, with our median beside theirs;
- treated in line with the guideline — share, with mortality split by whether
  the recommended treatment was delivered;
- mortality and median ICU length of stay among the eligible.

All of it broken down by **Zweck, Seymour and Bhavani phenotype and by coded
shock type**, because a trial's evidence is about a population and the question
"is this trial's population like ours" has a different answer inside each
phenotype.

This means a site that shares no patient-level data at all still contributes the
entire trial analysis. Verified: with NU sending tier 1 only, the
blood-pressure gap replicates independently at both sites — OptimaCC systolic
77.2% below Q1 at UCMC and 78.3% at NU, against the 77.9% the patient-level
route gave pooled.

What genuinely does not survive is **fitting trial regions in embedding space**,
which needs the pooled embeddings and so belongs to the coordinating site. That
is the half of 11 that stays pooled; the comparison half moved to tier 1.

**Token-space geometry (08).** Needs the sequences — which, under tier 2, now
travel. It runs at the coordinating site over the received bundles like
everything else. 04's *per-token* tornado goes with
it — the family-level comparison survives on `fam_n_*`, the per-token ranking
does not, and 04 skips that figure rather than drawing a degraded version that
looks like the real one.

**06's clinical baseline.** The embedding-vs-site-correction comparison
survives, including the cross-site transfer that is the point of it. The
physiology baseline it is scored against does not: `lab_*` and `vtl_*` are
measured values. 06 reports the embedding arms alone under the pooled role.

**Trajectories (09).** Runs site-side by consortium decision; only the
`gradient_alignment` and `position_encodes` scalars and the per-site flow
figures are pooled. Exporting 26 grid-point embeddings per patient plus a
per-timepoint binned-value table is a far larger disclosure surface than the
anchor export, for one figure.

The site effect survives: `fam_n_*` is what `04` regresses on, and at 97%
neighbour purity that decomposition is the largest single finding here.

---

## Still open

- **`MAX_PLOTTED_SITES = 3`** raises rather than mis-colouring, which is right,
  but a consortium is more than three sites. Scatter panels need faceting into
  small multiples before a fourth bundle arrives.
- **The projection moves under every site's feet.** PaCMAP is refit on the
  pooled set, so every figure changes when a site lands, and PaCMAP has no clean
  `transform`. If sites arrive on different schedules, pin a reference
  projection or regenerate the whole figure set each time.
- **Federated moments.** Phenotype scalers currently come from the derivation
  site rather than from pooled statistics. A one-round exchange of per-site
  `n`, `Σx`, `Σx²` would make them genuinely pooled; it is not implemented.
- **Sepsis denominator.** See below.

---

## The sepsis phenotypes, and what is wrong with them

[`config/phenotypes_sepsis.yaml`](../code/config/phenotypes_sepsis.yaml) holds
the literature values; [`clifcs/sepsis_phenotype.py`](../code/clifcs/sepsis_phenotype.py)
holds the logic.

**Read this before reading any of their numbers.** Both schemes were derived in
sepsis cohorts, and this pipeline has no sepsis definition — there is no
microbiology or antibiotic logic anywhere in it, so Sepsis-3 cannot be
evaluated. The denominator is the **coded** septic-shock flag (ICD-10 R65.21),
which fails in both directions and, worse, fails *differently at each site*:
septic shock is coded on 26.7% of NU's shock cohort and 20.4% of UCMC's. So
phenotype shares are reported **per site and never pooled into one percentage**,
and they are not comparable to the published distributions.

### Seymour — four clinical phenotypes

Seymour CW et al. *JAMA* 2019;321(20):2003-2017. **Note the year**: this is
commonly asked for as "Seymour 2016", which is the Sepsis-3 criteria paper
(*JAMA* 2016;315:762-74) — a definition, not a phenotyping. The α/β/γ/δ
phenotypes are the 2019 SENECA paper.

27 of the 29 published variables exist in CLIF 2.1 here. Omitted: **bands** (not
a `lab_category`; `neutrophils_percent` is a different measurement),
**Elixhauser** (no comorbidity mapping, and the diagnosis table is undated so the
index event cannot be excluded), and **troponin** — which *is* available but is
excluded because UCMC runs troponin-T and NU troponin-I, different assays with
different reference limits, so including it would cluster on which hospital ran
the test.

Two further variables are dropped at runtime for sparsity, and this one hurts:
**CRP is measured in 9% of these patients and ESR in 4%** — 9% and 5% even at a
72-hour lookback. They are not ordered in undifferentiated shock. They are also
two of the three markers Table 2 separates **γ** by, so γ here is recovered from
WBC and oxygenation alone and is the weakest of the four labels.

This is a **refit**, not a transfer: the paper publishes per-phenotype means but
not the cluster centroids, not the dispersion needed to put those means on a
common scale, and no classifier. A transfer from Table 2 is computed alongside
as a cross-check and the agreement reported — it currently runs **41%**, which
says the two are not the same partition and `seymour_phenotype` should be read
as this cohort's four classes rather than as the paper's.

Recovered against published:

| | n | share | published | mortality | published |
|---|---|---|---|---|---|
| α | 3,416 | 40.1% | 33% | 34.3% | 2% |
| β | 1,451 | 17.0% | 27% | 34.3% | 5% |
| γ | 2,781 | 32.6% | 27% | 28.8% | 15% |
| δ | 880 | 10.3% | 13% | 46.1% | 32% |

Shares are close. **Mortality does not reproduce the published ordering** and
`summarize` warns about it. The absolute level is expected — every patient here
met shock criteria 1 and 2, whereas Seymour's α included mild sepsis at 2%
mortality — but α ≈ β and γ < α is a genuine signal that the clusters do not
correspond. δ being clearly worst is the one part that does reproduce.

### Bhavani — four vital-sign trajectory groups

Bhavani SV et al. *Intensive Care Med* 2022;48(11):1582-1592.

The paper publishes **no trajectory coefficients, no classifier and no code** —
only four qualitative shapes. There is nothing to transfer, so this is a refit
and labelled from those descriptions.

It is also **k-means over the standardised 8 × 5 trajectory matrix, not
group-based trajectory modelling**, and that difference is not cosmetic. GBTM is
a finite mixture fitted by maximum likelihood, which is what lets the paper
handle missing hours through the likelihood and report posterior membership
probabilities. k-means has neither, so empty hours are filled (last observation
carried forward, then cohort median) and there is no membership probability to
threshold on.

The paper required three *complete* sets of vitals in eight hours. Applied
literally that admits 49% of patients, for a reason about charting cadence
rather than data quality: temperature appears in a mean of 3.4 of the eight
hourly blocks while heart rate, respiratory rate and both pressures appear in
7.2. The paper's window starts at hospital presentation, where vitals are taken
as a set by one person; ours starts at shock onset in an ICU, where they are
not. So the rule is split — core vitals must reach three blocks, temperature
must appear at least once and is carried forward — which admits 90%.
`require_complete_blocks: true` restores the literal rule.

| | n | share | published | mortality | published |
|---|---|---|---|---|---|
| A | 2,562 | 30.0% | 32% | 35.1% | 4.7% |
| B | 1,207 | 14.2% | 11% | 35.2% | 3.8% |
| C | 588 | 6.9% | **22%** | **55.1%** | **2.3%** |
| D | 3,360 | 39.4% | 36% | 26.8% | 5.2% |
| unassigned | 811 | 9.5% | — | | |

A, B and D match the published shares well. **C does not reproduce** — a quarter
of its published size, with the *highest* mortality where it should have the
lowest. The low-vitals cluster it captures is profoundly bradycardic and
hypothermic, which in a shock cohort is peri-arrest; that is a real clinical
group but it is not Bhavani's C.

The labelling rule was deliberately **not** tuned until the ordering matched.
Doing so would fit the labels to the outcome, which is exactly what
deterministic profile-labelling exists to prevent. The rule claims B first (the
only group the paper describes as hypertensive, so the only one separated by the
sign of a variable rather than its degree), then A as the most hyperthermic of
the rest, then splits the final two on pressure.

The likeliest reason both schemes under-reproduce is the denominator. Bhavani
derived these in patients with suspected infection at hospital presentation, at
2–5% mortality; ours are coded septic shock meeting shock criteria, at 27–55%.
Trajectory groups that separate a presentation cohort need not separate a shock
cohort. **Implementing Sepsis-3 is the single highest-value next step** for both
schemes.

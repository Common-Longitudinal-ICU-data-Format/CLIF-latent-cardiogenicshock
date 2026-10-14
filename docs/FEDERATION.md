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
| reads | its own records | `output/export/*/` only |
| runs | 01, 02, 08, 09, 12 | 03, 04, 05, 06, 10, 11 |

Verified under the pooled role: **03, 04, 05, 06, 10, 11**. **08** raises by
design — the token-space geometry needs the sequences. **07** assembles the
report from whatever tables the others wrote and is role-agnostic.

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
export/<SITE>/
  manifest.json          schema version, code/model/tokenizer hashes, salt fingerprint
  token_levels.csv       the rank -> value-range legend (no patient data)
  counts/                aggregate tables, small-cell suppressed
  patient/               embeddings + ordered token labels   <- needs a DUA
```

**`counts/`** is aggregate and suppressed. Attrition ledger, the three phenotype
breakdowns, shock type, token-family totals, per-variable availability.

**`patient/`** is one row per patient per anchor:

| | |
|---|---|
| `emb_0000..emb_1023` | the latent embedding |
| `tok_*` | last pre-anchor decile **rank** per allowlisted variable |
| `fam_n_*` | per-family token counts (~23 integers) |
| `elig_*` | per-trial eligibility, computed site-side |
| labels | the three phenotype systems, shock type, sepsis flag |
| outcomes | mortality, ICU LOS, `age_band`, sex |
| `row_key` | salted SHA-256 of the local record number |

### What does not travel, and cannot

Record numbers, every timestamp, exact age, race, ethnicity, peak troponin, raw
measured values, and **token sequences**.

That last one matters more than it looks. A full `tokens_past` sequence is a
reconstruction of the record — it is the *most* disclosive object in this
pipeline, more so than the embedding derived from it. So "minimal tokenized
elements" is defined narrowly: the last bin per variable, and family counts.
`clifcs.tokens.load_pre_anchor_tokens` raises under the pooled role.

### Why ranks and not values

The tokenizer bins every numeric into deciles and bakes the bin into the token
name — `VTL//map_Q3`. The model never saw 71.4 mmHg; it saw `map_Q3`. So a rank
is not a coarsening of what the embedding was built from, it **is** what the
embedding was built from, and a panel coloured by rank shows the model's own
resolution rather than an approximation of it.

It also solves cross-site comparability for free. The bin edges were fitted once
at training time and ship inside `tokenizer.yaml`, which every site must already
hold for its embeddings to mean anything. Rank 3 is the same range of millimetres
of mercury everywhere, with nobody exchanging a quantile.

And because the edges are model metadata rather than patient data, the
coordinating site renders the legend — "Q3 = 71–75 mmHg" — from the tokenizer
alone. That is the labelling the consortium asked for, obtained without any site
sending a single value.

Two caveats. Ranks are **not always contiguous**: `spo2` has no Q3 or Q8 and
`gcs_total` has only Q0, Q1, Q2 and Q9, because most charted GCS totals are 15.
Build ramps from `token_levels.csv`, not from `range(10)`. And rank is
**ordinal, not linear** — Q0→Q1 is a different number of mmHg from Q8→Q9, so
ranks may be ordered and binned but never averaged.

---

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

python code/01_data_processing.py --phenotype-model <...>/phenotype_model.json \
    --seymour-model <...>/seymour_model.json \
    --bhavani-model <...>/bhavani_model.json
python code/02_model_embedding.py --batch-size 8
python code/12_export.py
```

Then send `output/export/<SITE>/`.

**The three `--*-model` flags are required at a contributing site.** Without
them the site refits its own clusters and "phenotype II", "beta" and "group A"
stop meaning the same thing at each site, which makes every pooled breakdown
uninterpretable. The model files carry the feature set, the transform, the
derivation cohort's moments and the centroids — all four, because a site that
standardised against its own moments would re-centre the space and move patients
between phenotypes without anything clinical changing.

If your DUA does not permit patient-level transfer:

```bash
python code/12_export.py --counts-only
```

The bundle then carries `counts/` only. The pooled side degrades rather than
failing: that site contributes to every count table and to no projection.

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
cp -r <received>/UCMC output/export/
python code/03_embedding_analysis.py --role pooled
python code/11_trial_alignment.py --role pooled
```

The coordinating machine still needs the model and tokenizer, because it has to
check that the bundles were produced under the weights it is about to interpret
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

**The published-baseline comparison.** This is the real loss, and it falls on
the strongest result in the README — the blood-pressure gap against OptimaCC,
ECLS-SHOCK and CULPRIT-SHOCK. `trial_table1_*`,
`trial_representativeness_*` and `trial_inside_tested_*` measure how far each
eligible patient sits from a trial's published medians **in interquartile
ranges**, which needs the measured value. A decile rank cannot be placed in a
published IQR: the bin edges are our cohort's quantiles, not the trial's, so
"below their Q1" is not recoverable from "in our decile 2". These tables must be
computed site-side and exported as aggregates if they are to survive.

**Token-space geometry (08).** Needs the sequences, which do not travel. Runs
site-side; pool its results, not its inputs. 04's *per-token* tornado goes with
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

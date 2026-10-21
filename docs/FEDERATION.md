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
| runs | 01, 02 | 03, 04, 05, 06, 08, 10, 11 |

Verified under the pooled role: **03, 04, 05, 06, 10, 11**. **07** assembles the
report from whatever tables the others wrote and is role-agnostic. **08** reads
the model's own embedding table and needs no patient data at all. **09** is not
federated: see [Trajectories](#trajectories-are-not-federated).

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
    counts/                    attrition, phenotype breakdowns, availability,
                               token_prevalence_anchor-*.csv (the tornado)
                               token_families_anchor-*.csv
    trials/                    eligibility, IQR position, concordance
    token_levels.csv           the rank -> value-range legend
    purity/                    TIER 2 -- phenotype coherence in GEM space.
      phenotype_purity.csv
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

**Tier 1 — [`01_site_tables.py`](../code/01_site_tables.py).** Every site runs
it, and it is the only thing currently being asked of contributing sites.
Cohort → subset → derive → collate → tokenize → winnow → tokencounts →
baseline → trials → export, as resumable stages. Everything it writes to
`upload_to_box_<SITE>/` is a count or an average over at least `--min-cell`
patients. This is the classic CLIF federated contribution.

**It needs no GPU and never loads the GEM weights** — they need not be on the
machine at all, since `models.<key>.home_glob` is resolved on access rather than
at config load. It loads the *tokenizer*, which is a different file, because the
token-prevalence table is built from the token stream. A site with no GPU, or
with no agreement permitting patient-level transfer, runs this and stops.

A bundle produced without the weights carries a null model hash in its manifest,
which is correct: nothing was loaded, so there is nothing for the coordinator to
check. Its tokenizer hash still travels, and that is the hash the token counts
actually depend on. The pooled agreement check skips nulls rather than reading
them as disagreement.

That is a change. Tier 1 used to embed, because the phenotype-coherence measure
needs the embeddings and that was the one substantive thing tier 1 reported
beyond counts. Purity has moved to tier 2, and a tier-1-only site no longer
reports it. What tier 1 gained in exchange is the **token-prevalence table** —
per-token and per-family counts, suppressed — which is what 04 draws its
tornado from. That comparison previously required the token sequences and so
existed only for sites that shared them; it now covers every site. Since the
site effect is the largest single finding here, being able to decompose it into
"because they chart differently" should not have been contingent on a DUA.

**Tier 2 — [`02_site_embeddings.py`](../code/02_site_embeddings.py).** Only a
site whose agreement permits patient-level transfer runs it. This is where the
GEM weights are loaded. It adds `purity/` and `patient/` to the bundle tier 1
already wrote: one row per patient per anchor, holding the 1024-d embedding
plus the labels needed to colour and stratify it -- the three phenotypes, the
coded shock type, the outcomes, per-trial eligibility, fit against the
published interquartile ranges, treatment concordance, token-family counts, ten
last-pre-anchor token bins, and age as a five-year band.

The copy in `patient/` is a redaction. The embeddings the model actually
produced stay in `patient_level_data_<SITE>/`, because those carry the record
number, the patient id and the onset timestamp.

Two scripts rather than one with a flag, because a flag invites somebody to pass
it without the question having been asked.

## What tier 2 actually contains

A 1024-d vector per patient per anchor, plus the labels needed to colour it.
Not the record, and no longer the model's input either:

- **No measured values.** Where a physiological variable travels at all it is
  the ordinal rank of the token bin the model read -- `tok_map` is "bin 3 of
  10", never 58 mmHg. The bin edges are the tokenizer's, so the ranks are
  identical at every site by construction, which is also what makes them
  comparable without anyone agreeing on anything.
- **No timestamps.** `hours_admission_to_cs_onset` is relative and is the only
  time-like quantity on the allowlist.
- **No record numbers, exact age, race or ethnicity.** `row_key` is a salted
  SHA-256 of the record number truncated to 128 bits, and age is a five-year
  band topped at 90+.
- **No token sequences.** Withdrawn in schema 3.0 — see below.

What it *is*, stated plainly: a learned function of the patient's entire
recorded history. Nobody has shown that such representations resist
re-identification attack. It must not be described as de-identified, and it
moves under a patient-level DUA or not at all.

## Why embeddings rather than sequences

**This reverses the earlier decision, and the earlier argument was not wrong.**

Schema 2.0 shipped `tokens_past` and `s_elapsed_past` — the ordered token ids
the model read and the seconds elapsed at each — so the coordinating site could
re-embed. It reproduced the sites' own vectors exactly: eval mode under
`torch.inference_mode()` is a deterministic function of token ids and elapsed
times, and re-embedding was checked over 512 patients at a **maximum absolute
difference of 0.0**, bit for bit. Shipping the input rather than the output
bought re-anchoring at a different cut point, reading a different layer,
changing the pooling, and re-running after a model update without asking anyone
to re-export. It was also smaller: 25 MB of tokens against 45 MB of embeddings
for UCMC.

All of that was true and all of it has been given up, for one reason. A token
stream is a **re-serialisation of the record** — which labs were drawn, in what
order, how often, against what clock. It was the most disclosive artefact this
pipeline produced, *more so than the embedding derived from it*, because an
embedding is a lossy projection and cannot be inverted back to the stream while
the stream can always be turned into the embedding. Tier 2 now ships the answer
rather than the input, and the two sequence columns sit in
`FORBIDDEN_COLUMNS` so the old path cannot be revived by accident.

**What this costs, so nobody discovers it later.** Re-anchoring, layer probing
and pooling changes now require sites to re-export, and a model update means
asking every contributing site to run 02 again. If the consortium expects to
change the anchor or the layer, that is an argument for revisiting this
decision — not for quietly keeping a copy of the sequences.

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

# tier 1 -- every site runs this. No GPU; the GEM weights are never loaded,
# and need not even be present. The shared phenotype models in models/ are
# applied by default, so there are no flags to get wrong.
python code/01_site_tables.py

# tier 2 -- ONLY with a patient-level DUA in place. Needs a GPU.
python code/02_site_embeddings.py --batch-size 8
```

Then send `output/upload_to_box_<SITE>/` -- one directory, whichever tier was
run. Tier 1 alone is a complete, usable contribution: counts, the three
phenotype breakdowns, the trial comparison, and the token-prevalence tables the
site-effect decomposition reads.

**A contributing site must not refit the phenotype models.** Its own clusters
make "phenotype II", "beta" and "group A" mean something different there than at
every other site, and every pooled breakdown becomes uninterpretable. 01 applies
the shared models in `models/` by default and refuses to refit unless asked with
`--refit-phenotypes`, which is only correct at the derivation site. The model
files carry the feature set, the transform, the
derivation cohort's moments and the centroids — all four, because a site that
standardised against its own moments would re-centre the space and move patients
between phenotypes without anything clinical changing.

If your DUA does not permit patient-level transfer, do not run tier 2. The
bundle then carries `counts/`, `trials/` and the token legend, and the pooled
side degrades rather than failing: that site contributes to every count table,
to the whole trial analysis, to the token-prevalence comparison 04 decomposes
the site effect with, and to no projection.

What it does **not** then contribute is `purity/`. Phenotype coherence needs
the embeddings, so it is gated on the same agreement as the embeddings
themselves.

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
cp -r <received>/upload_to_box_UCMC output/received/
cp -r <received>/upload_to_box_NU   output/received/

python code/03_embedding_analysis.py --role pooled
python code/04_site_effect.py       --role pooled   # the tornado, from counts/
python code/05_site_correction.py   --role pooled --save-map
python code/03_embedding_analysis.py --role pooled --corrected leace
python code/11_trial_alignment.py   --role pooled
```

**No GPU, and no re-embedding step.** Sites ship the embeddings under schema
3.0, so the pooled side reads them straight through. The cached
`reembedded_anchor-*.parquet` files schema 2.0 produced are no longer written
or read.

The coordinating machine still needs the tokenizer, to check that the bundles
were produced under the token mapping it is about to interpret the exported
ranks against, and to render the `token_levels.csv` legend.

Running 03 twice is the point: the second pass with `--corrected leace` redraws
the identical panel set from 05's affine output into
`figures/<model>/anchor-<a>/corrected-leace/`, so every colouring can be read
raw and corrected side by side.

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

**Token-space geometry (08).** Unaffected: it is a UMAP of the model's own
embedding table, one point per vocabulary entry, and uses no patient data at
all. It weights tokens by cohort frequency, which now comes from the tier-1
`token_prevalence` tables rather than from the streams.

**04's per-token tornado now survives, and for every site.** This is the one
thing the restructure improved rather than cost. The figure ranks tokens by the
difference in per-patient *rate* between sites, which needs marginal counts and
not sequences — so the counts are computed site-side by the `tokencounts` stage
and shipped in tier 1. What still needs the streams, and is therefore
site-side only, is the per-token **log-odds** ranking (a per-patient quantity)
and the UNK burden.

**06's clinical baseline.** The embedding-vs-site-correction comparison
survives, including the cross-site transfer that is the point of it. The
physiology baseline it is scored against does not: `lab_*` and `vtl_*` are
measured values. 06 reports the embedding arms alone under the pooled role.

**Trajectories (09) are not federated at all.** See below.

The site effect survives: `fam_n_*` is what `04` regresses on, and at 97%
neighbour purity that decomposition is the largest single finding here.

---

## Trajectories are not federated

[`09_trajectories.py`](../code/09_trajectories.py) is deliberately outside the
tier structure. It embeds every patient at 26 grid points around shock onset,
and exporting 26 vectors per patient is a far larger disclosure surface than the
anchor export for one figure.

It runs instead where several sites' CLIF extracts sit on the **same
filesystem**, pools them directly, and sends nothing — so no export contract
applies to it, because nothing crosses a boundary. A consortium member who does
not host other sites' data cannot run it and is not expected to.

Three options for doing it genuinely federated were considered and are recorded
here in case the question returns:

| | what crosses | verdict |
|---|---|---|
| binned field only | per-site 26×26 `(count, Σdx, Σdy)`, suppressed | unsound — each site bins in its own projection, so the cells do not correspond |
| shared basis, then binned field | the PCA basis out, the binned field back | works, aggregate-only, but needs two rounds and only the **PCA** field pools: PaCMAP has no clean `transform` |
| projected grid coordinates | 50-d PCA scores × 26 points per patient | keeps per-patient trajectories pooled, ~20× smaller than raw vectors, but needs a new contract prefix and a heavier DUA |

The pipeline as written does none of these. It does the co-located analysis,
and the figure it exists for is described in
[RUNNING_ON_HPC.md](RUNNING_ON_HPC.md).

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

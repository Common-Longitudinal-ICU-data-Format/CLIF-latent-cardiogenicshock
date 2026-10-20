# What this pipeline found

Written for: the analyst or reviewer reading the output, and anyone deciding
what to believe from it.

The substantive results, with their caveats. Numbers are from the UCMC + NU
two-site run (34,142 patients) unless stated otherwise; Rush was not yet
included. How the cohort and labels are defined is in [COHORT.md](COHORT.md),
how to run it in the [README](../README.md).

**Read the site effect first.** It conditions everything below it.

---

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

---

## Phenotype coherence in the embedding space

For each phenotype, the share of a patient's 30 nearest neighbours (in that
site's own 50-d PCA space) carrying the same phenotype, against the share chance
would give. Computed per site, reported **per phenotype**, and sent as scalars —
`output/upload_to_box_<SITE>/purity/phenotype_purity.csv`.

**This is a tier-2 output.** It is an aggregate — a handful of scalars per
scheme — but it needs the embeddings to compute, so it is gated on the same
agreement they are. It used to sit in tier 1 and was the reason tier 1 required
a GPU; moving it is what makes tier 1 runnable anywhere, at the price that a
tier-1-only site does not report it.

Zweck's phenotypes are genuinely regions of the space, and the ordering
replicates independently at both sites:

| Zweck phenotype | UCMC lift | NU lift |
|---|---|---|
| III cardiometabolic | **2.41x** | **3.34x** |
| II cardiorenal | 1.78x | 2.07x |
| I non-congested | 1.13x | 1.09x |

I non-congested is 71% of the cohort, so it has little room above chance by
construction; III is a tenth of it and still holds three to five times the
expected share of like neighbours.

The sepsis phenotypes are weaker and sit in a narrow band (Seymour 1.14–1.36,
Bhavani 1.24–1.47 at UCMC). **Seymour γ is the lowest of its four**, which is
what the missing CRP and ESR predict — they are two of the three markers Table 2
separates γ by. The coded mechanism beats both schemes: cardiogenic shock runs
2.23x at UCMC and 2.86x at NU.

Each site fits its own PCA, so the 50 dimensions differ between sites and only
`lift` is comparable — it is already normalised by local phenotype prevalence.
Raw purity is not. See [code/clifcs/purity.py](../code/clifcs/purity.py).

---

## Probing the undifferentiated group

[code/13_undifferentiated.py](../code/13_undifferentiated.py) takes the
**54.6% of the cohort carrying no coded shock mechanism** (see
[Shock type](COHORT.md#shock-type)) as its subject: what those 18,626 patients
resemble in the model's representation, whether a linear probe trained on the
coded patients will assign them a mechanism, and whether the treatment they
received matches the assignment. Settings and the ICD groups live in
[code/config/undifferentiated.yaml](../code/config/undifferentiated.yaml).

**Everything is computed within site.** The two hospitals leave a similar share
uncoded (NU 56.6%, UCMC 48.9%) but reach for different mechanisms when they do
code one — the 2.5x cardiogenic and 1.7x septic gaps in
[Shock type](COHORT.md#shock-type). The coded patients
define every centroid and train every probe here, so a mechanism mix differing
that much means neither the centroids nor the class priors transport.

### The centroids do not separate the mechanisms, but a linear probe does

Coded patients sit a mean cosine distance of **0.456** from their own
mechanism's centroid against **0.494** from the other two — a gap of 0.038,
against a within-group spread an order of magnitude larger. By that measure the
mechanisms are not distinct regions, which is the same answer the trial regions
gave (centroid separation 0.03–0.32).

A multinomial logistic regression on the same 50 components nonetheless reaches
**AUROC 0.90 for cardiogenic, 0.87 for septic and 0.79 for hypovolemic**,
cross-validated with patients as the grouping unit. Cross-site scoring —
train on one hospital, test on the other — gives 0.89 / 0.85 / 0.75, so very
little of the within-site number was documentation habit.

The information is there and it is linearly decodable; it simply does not lie
along the direction joining the class means. **Distance to a centroid is the
wrong instrument for this space**, and the nearest-centroid columns are kept
mainly to show why.

### The separation is not the model reading the treatment back out

This was the main thing that could have invalidated the probe. Medication
administrations are part of the token vocabulary, so an embedding taken after
the clinician starts treating has seen the antibiotics and the inotrope that
the working diagnosis produced.

Re-running the identical probe on 09's grid at 0 h, +6 h and +24 h from onset:

| class | 0 h | +6 h | +24 h |
|---|---:|---:|---:|
| cardiogenic | 0.899 | 0.917 | 0.919 |
| septic | 0.869 | 0.877 | 0.887 |
| hypovolemic | 0.791 | 0.811 | 0.824 |

A full day of post-onset record buys about **two AUROC points**. Essentially all
of the separation is present before the shock is treated — and the rise is not
attributable to treatment alone in any case, since the patient is also
deteriorating over the same hours. No re-embedding was needed for this: 09
already writes all three offsets for all 34,142 patients.

### One assignment survives an external check; the other two do not

The only evidence available against a probe assignment is non-shock coding and
microbiology. It supports the septic class: among uncoded patients the probe
confidently calls septic, **51.6% have a positive culture at NU and 73.4% at
UCMC**, against 32.2% and 58.1% for the ones it is unsure about. Mortality
agrees — confidently septic is the deadliest assigned group at both sites
(15.2% and 24.5%).

The cardiogenic class does not replicate it. Patients confidently called
cardiogenic carry a cardiogenic diagnosis code 39.0% (NU) and 47.5% (UCMC) of
the time against base rates of 29.1% and 26.1%, and they **die at 1.1% and
3.4%** against 25.2% among patients the coders actually called cardiogenic.
Whatever that class selects, it is not patients in cardiogenic shock.

The likeliest reading of the section as a whole is that the probe sorts the
uncoded by severity and by how septic they look, not by mechanism. A three-way
probe forced to choose will put a mild patient somewhere, and the class it picks
carries no claim that any mechanism was present.

### D62 had to be withdrawn from the evidence flag

Whether a diagnosis arrives *later* is not answerable here — the diagnosis table
has no timestamp and an uncoded patient has no shock code at any point by
definition. What is reported instead is whether a mechanism was recorded by some
other means anywhere in the admission.

The first version of that flag counted `D62`, acute posthaemorrhagic anaemia, as
evidence of a haemorrhagic mechanism. It is coded on **7,916 of NU's 14,163
uncoded patients (55.9%)** against 0.7–1.3% for each specific gastrointestinal
haemorrhage code, so it made 57% of the undifferentiated look like evidence of
bleeding and was carrying the whole flag by itself. It is a post-procedural
anaemia code far more often than evidence of hypovolemic shock, and at that
prevalence it could not have distinguished one patient from another whichever it
meant. Removed; the flag now reads 21.1% in NU's coded hypovolemic patients
against 2.7% in its uncoded.

### What cannot be assessed

Three of the treatment comparisons have no table behind them at either site:
fluid volume (no intake/output), transfusion (no blood product table) and
**echocardiography or POCUS (no echo table)**. The third is the serious one —
echo is the main act of differentiating shock, and whether a study was *done* is
as unavailable as what it showed, so this analysis cannot distinguish a patient
nobody managed to diagnose from one nobody tried to. Counting crystalloid
administration events as a fluid proxy was considered and rejected: an event
without a volume cannot separate a 30 mL/kg resuscitation from a carrier line.

Treatment concordance is therefore scoreable for two of the probe's three
classes. Among uncoded patients called cardiogenic, those who received an
inotrope or mechanical support have a mortality log-odds 0.71 lower unadjusted
and 0.48 lower adjusted for non-cardiovascular SOFA, age and site (p = 0.016);
for the septic class, antibiotics within six hours give −0.26 and −0.16
(p = 0.069). **Confounding by indication is the whole of this** — the clinician
read the same record the probe reads, so concordance measures agreement between
two readings of one patient, and adjusting for severity leaves the judgement
itself unmeasured.

### Outputs

`undiff_groups_*.csv`, `undiff_evidence_*.csv`, `undiff_reference_*.csv`,
`undiff_neighbours_*.csv`, `undiff_probe_metrics_*.csv`,
`undiff_probe_calibration_*.csv`, `undiff_probe_vs_evidence_*.csv`,
`undiff_snapshots_*.csv`, `undiff_treatments_*.csv`,
`undiff_first_agents_*.csv`, `undiff_concordance_*.csv`,
`undiff_concordance_adjusted_*.csv` and `undiff_not_assessable.csv`; the
per-patient `undiff_proximity_*`, `undiff_probe_predictions_*` and
`undiff_snapshot_assignments_*` parquets alongside the other patient-level
files; five figures under `figures/<model>/anchor-<anchor>/`; and a written
summary at `output/undifferentiated_summary_<anchor>.md`.

All of it renders into the report as **10 · Undifferentiated shock**, which is
omitted along with its contents entry when 13 has not run. The snapshot stage
needs 09's grid and degrades to the offsets that exist; `--skip-snapshots`
turns it off.

---

## Guideline fit

Every cardiogenic shock recommendation rests on a trial that enrolled a narrow
slice of the people who present in shock.
[code/11_trial_alignment.py](../code/11_trial_alignment.py) asks where those slices
sit in the latent space, which of our patients fall inside them, and who falls
inside none. The eight trials and the populations they enrolled are in
[code/config/trials.yaml](../code/config/trials.yaml); the eligibility logic is in
[code/clifcs/trials.py](../code/clifcs/trials.py), one commented predicate per
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
[code/config/trials.yaml](../code/config/trials.yaml) are what the result was
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

Measured over a comparable window — six hours either side of onset, which is
what "at randomisation" means:

| agent | ours | ECMO-CS |
|---|---|---|
| norepinephrine | 78.2% | 85.5% |
| vasopressin | 29.9% | 35.0% |
| epinephrine | 19.6% | 3.4% |
| dobutamine | **5.6%** | **54.7%** |
| milrinone | **3.9%** | **32.5%** |

Our patients are on slightly *less* norepinephrine and vasopressin, more
epinephrine, and **tenfold fewer inotropes**. A seven-point norepinephrine
deficit cannot account for a 10 mmHg difference in achieved systolic pressure,
so the readings that remain are genuinely deeper shock, a cardiology-led trial
population carrying far more inotropic support than an undifferentiated ICU
cohort receives, or both.

**On timing.** These flags are anchored at `cs_onset_dttm`; an earlier version
built them over the whole admission and reported norepinephrine at 86.4% and
vasopressin at 48.4%, both *above* the trial. Anchoring reverses the direction
of both. Physiology (pressures, lactate, pH, temperature) is anchored at onset
throughout; **diagnosis codes are hospitalization-level with no timestamp at
all**, and the procedure flags carry no time filter — right for the three
negative trials, where concordance means the device was never used, and lenient
for DanGer Shock, where it means the pump was used, so that 14.1% is if anything
an overestimate of timely use.

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

`VARIABLE_CONFIDENCE` in [code/clifcs/trials.py](../code/clifcs/trials.py) marks
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

Read with [Shock type](COHORT.md#shock-type), the conclusion is the same from both
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

---

## Does fitting the trial track with doing better?

[code/12_trial_fit_outcomes.py](../code/12_trial_fit_outcomes.py) takes the gap 11
measures and asks what it is worth. Three outcomes — in-hospital mortality,
**ventilator-free days** and **vasopressor-free days** to day 28 — against two
readings of the same gap, each unadjusted and adjusted for non-cardiovascular
SOFA, age and site.

The two free-day outcomes are counted separately, not as one composite. They
come off on different clocks and the correlation between them among survivors
is only 0.65: 6,103 patients are ventilator-free for the full 28 days while
still needing a vasopressor, and 2,718 the reverse. A composite would be
dominated by whichever has the longer tail while reading as a statement about
both. Both are derived in [code/clifcs/freedays.py](../code/clifcs/freedays.py),
which counts a day as supported if any record falls in it — coarse, in the same
unit as the outcome, and biased downward for intermittent support, so a
free-day count is a floor.

### Distance has no direction, and the direction is the finding

`t1_absdist_{trial}` is how far outside the published interquartile ranges a
patient sits, averaged over the variables they have. `t1_severity_gap_{trial}`
is **which way**, signed by `VARIABLE_SEVERITY_SIGN` so that positive means
sicker or older than the trial's median. Heart rate and temperature are
deliberately unsigned: both are U-shaped in shock, and a wrong sign is worse
than a missing one.

The split matters immediately. Among ECLS-SHOCK-eligible patients the 934 who
are sicker than the trial run **46.6% mortality against 30.6%** for the 229 who
are healthier — a contrast the old binary inside/outside could not express,
because both groups are "outside".

### The curve turns

Entering the score as a restricted cubic spline was not decoration. Mortality
against the severity gap is **U-shaped** for SHOCK, IABP-SHOCK II,
CULPRIT-SHOCK and ECLS-SHOCK: patients *healthier* than the trial on the
measured axes die more than those sitting at its median. 12 classifies every
fitted curve and names the turning ones in its summary rather than leaving it
to the eye.

A U-shape is a warning about the score, not only about the patients. It says
the severity gap stops tracking severity at the healthy end — a patient with a
normal lactate and a decent pressure who is nevertheless in shock is not mild,
they are atypical, and the measured axes do not see why.

### Where adjustment is collinearity rather than correction

The exposure and the confounder are built from overlapping measurements, and
for some trials from the *same* measurement. SHOCK publishes exactly one
usable quartile — age — so its severity gap is a rescaling of age, and
adjusting it for age regresses a variable on itself. That fit returns a
72-point mortality difference which is collinearity.

`trial_fit_adjustment_overlap.csv` gives the overlap **per trial and per
score**, because the two scores are built from different variable sets:

| trial | score | overlap with the adjustment set |
|---|---|---|
| SHOCK | `absdist`, `severity_gap` | 100% |
| IABP-SHOCK II | `severity_gap` | 100% |
| SOAP II | `severity_gap` | 50% |
| ECLS-SHOCK | `absdist` | 12% |

Rows at 100% are reported and labelled `collinear` rather than dropped —
suppressing them would leave the unadjusted estimate looking unopposed.

### Secondary: concordant care by fit

Whether the fit score's slope differs between patients who got the recommended
treatment and those who did not. **Confounding by indication is severe and is
not adjustable away**, and it does not point the same way for every trial, so
each row carries what concordance actually means there — read off the
concordance definition itself, not the trial's description:

- the negative device trials (IABP-SHOCK II, ECMO-CS, ECLS-SHOCK) invert it:
  "concordant" means the device was **withheld**, which happens most in
  patients too well or too sick to cannulate
- SOAP II and OptimaCC are an agent-choice contrast — one drug used, another
  avoided — confounded by whatever drove the choice
- SEPSISPAM and the 65 trial test an **achieved physiological value**, which is
  partly an outcome of severity rather than a decision

### Outputs

`trial_fit_outcomes_*.csv` (per trial × outcome × score × adjustment),
`trial_fit_curves_*.csv` (fitted curves for plotting),
`trial_fit_interaction_*.csv`, `trial_fit_adjustment_overlap.csv`,
`trial_fit_skipped_*.csv`, the figure
`figures/<model>/anchor-<anchor>/trial_fit_outcomes.png`, and a written
summary with limitations at `output/trial_fit_summary_<anchor>.md`.

All of it renders into the report as **4.1 &middot; Fit and outcome**, a
subsection of the guideline-fit section rather than a section of its own: 11
produces the score and 12 says what it is worth, so separating them would put
the two under different headings. The subsection is omitted when 12 has not
run, and its contents entry with it.

Needs `statsmodels`, which nothing in 01–11 imports — a site running only the
site-side stages does not need it.

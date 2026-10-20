# Cohort, labels and conventions

Written for: anyone reading a number this pipeline produced and needing to know
how the patients were selected and what the labels mean.

How the cohort is built, what the shock-type and phenotype labels are, where the
embeddings are anchored, and the plotting conventions. The *results* are in
[FINDINGS.md](FINDINGS.md); how to run any of it is in the
[README](../README.md).

---

## Cohort definition

Configured in [code/config/cohort.yaml](../code/config/cohort.yaml).

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

---

## Shock type

What the coders said the mechanism was, from `clif_hospital_diagnosis`, attached
to every enrolled patient by
[code/clifcs/shock_type.py](../code/clifcs/shock_type.py). **A labelling, never a
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

---

## Phenotypes

Three systems are attached: Zweck's cardiogenic shock phenotypes over the whole
cohort, and Seymour's and Bhavani's sepsis phenotypes over the coded-septic
subset. The two sepsis systems, what they reproduce and what they do not, and
why their denominator is the weakest part of them, are in
[docs/FEDERATION.md](FEDERATION.md#the-sepsis-phenotypes-and-what-is-wrong-with-them).
In short: **Seymour's shares are close and its mortality ordering is not;
Bhavani's group C does not reproduce at all.** Both are reported per site and
never pooled into one percentage.

Zweck et al., *Phenotyping Cardiogenic Shock*, J Am Heart Assoc 2021;10:e020085
([doi:10.1161/JAHA.120.020085](https://doi.org/10.1161/JAHA.120.020085)) —
phenotypes **I** non-congested, **II** cardiorenal, **III** cardiometabolic.

This is a **refit, not a transfer**: the paper names the clustering variables
and each phenotype's direction but does not publish the centroids, so k-means is
refit on the eight labs it names and the clusters are then labelled
*deterministically from their centroid profiles* rather than by cluster index.
`clifcs.phenotype.summarize` checks the recovered labels against the paper's
mortality ordering (I < II < III) and warns if it does not hold.

---

## Embedding anchors

Each patient is embedded twice, at two cut points:

- **`24h`** — 24 h after admission. Matches the winnowing config the model was
  trained and evaluated under (`duration_s: 86400`), so the representation is
  in-distribution.
- **`onset`** — at `cs_onset_dttm`, so representations are aligned to the
  clinical event rather than to admission. cocoa's winnower cannot express a
  per-patient cut point, so
  [code/clifcs/onset_winnower.py](../code/clifcs/onset_winnower.py) overrides its
  thresholding; everything downstream is cocoa's unchanged.

---

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

`09` also takes `--correction {none,leace,site-standardized}`, defaulting to
**leace**, and applies the map 05 fitted at the anchor to every grid point
*before* projecting. Site dominates this geometry at 97% neighbour purity, so an
uncorrected flow field is substantially a picture of three hospitals charting
differently. The map is loaded rather than refitted: each patient appears at up
to 26 grid points, so refitting would weight the fit by whose grid is complete.
Run `05 --save-map` first.

---

## Plot conventions

dpi 500; the legend sits in its own axes outside the scatter; the scatter keeps a
1:1 data aspect ratio. Colours come from the dataviz reference palette's
all-pairs–validated slots —
[code/clifcs/palette.py](../code/clifcs/palette.py) carries the validator and
self-tests itself against the published reference figures:

```bash
python code/clifcs/palette.py
```

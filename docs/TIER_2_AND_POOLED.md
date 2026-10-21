# Tier 2 and the pooled analysis

Written for: anyone who wants to know what the rest of this pipeline does, and
the coordinating analyst who actually runs it.

**None of this is part of the current run request.** Contributing sites are
being asked for `01_site_tables.py` only — see the
[README](../README.md). This file exists so that the material that used to sit
in the README is still written down, and so a site reading the repository can
see what it is contributing to.

---

## The three tiers

| | runs | needs | produces |
|---|---|---|---|
| **1 — site tables** | **every site. The current request** | CPU, tokenizer only | counts, phenotype breakdowns, baseline by shock type, trial comparison, token prevalence |
| **2 — embeddings** | sites with a patient-level DUA | **GPU** + GEM weights | the 1024-d vectors, plus phenotype coherence |
| **3 — pooled** | coordinating site | the received bundles | projections, site correction, trial alignment, the report |

Tier 1 never loads the model. A site with no GPU or no agreement runs it and
stops, and that is a complete contribution — not a degraded one. The pooled
analysis takes tier-1-only sites in its count tables, in the whole trial
analysis, in the token-prevalence comparison the site effect is decomposed with,
and leaves them out of the projections only.

A site sends **one directory** either way. Tier 2 adds to the tier-1 bundle
rather than making a second one:

```
output/upload_to_box_<SITE>/
  counts/ trials/      tier 1 -- aggregate, small-cell suppressed
  purity/ patient/     tier 2 -- needs a DUA
```

---

## Tier 2 — `02_site_embeddings.py`

Only with a patient-level agreement in place.

```bash
pip install -r requirements-embed.txt     # ~10-15 GB, pulls torch
python code/02_site_embeddings.py --batch-size 8
```

This is the GPU job, and the first point at which the GEM weights are loaded —
so `models.<key>.home_glob` in `paths.yaml` has to resolve here, which it need
not for tier 1. `--batch-size 8` fits an 8 GB card; raise it if you have more,
drop it to 2–4 on CUDA OOM, since `output_hidden_states` keeps every layer.

Install the CUDA wheels **last**: `cotorra` leaves `torch`, `torchvision` and
`torchaudio` unpinned, so installing them first lets it resolve CPU-only or
mismatched builds over the top.

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.5.1+cu121 True   <- a bare 2.5.1 means the CPU wheels won; embedding will
#                       then be ~20x slower and say nothing
```

It adds two things to the bundle:

- `patient/` — one row per patient per anchor, holding the 1024-d embedding and
  the labels needed to colour and stratify it, including per-trial eligibility,
  fit against the published interquartile ranges, and treatment concordance.
  Roughly 45 MB per 10,000 patients.
- `purity/` — the phenotype-coherence scalars. Aggregate, but it needs the
  embeddings to compute, so it is gated on the same agreement.

### Why it is gated

**The tier-2 export is patient-level and is not de-identified.** Record numbers
are replaced by a salted hash, timestamps are dropped and age is reduced to
five-year bands, which narrows what travels — it does not make the file
anonymous:

> A 1024-d embedding is a learned function of the patient's entire recorded
> history. No one has shown that such representations resist re-identification
> attack, and they should not be treated as de-identified.

Treat it as a **limited data set at best**, and move it under the same IRB
approval and data use agreement you would use to share patient-level clinical
data. If your DUA does not permit patient-level transfer, run tier 1 and stop.

### Keep the salt

`output/patient_level_data/.id_salt` maps a shared pseudonym back to a local
record number. It must not be sent, and it must be **backed up** — if it is
lost, every future export carries new pseudonyms that cannot be matched to the
ones already sent. (Pseudonyms are SHA-256 over salt + record number, truncated
to 128 bits.)

---

## Tier 3 — the pooled analysis

At the coordinating site, drop the received bundles in `output/received/` and
run:

```bash
pip install -r requirements-pooled.txt     # ~500 MB, no GPU
python code/03_embedding_analysis.py --role pooled
python code/04_site_effect.py        --role pooled
python code/05_site_correction.py    --role pooled --save-map
python code/03_embedding_analysis.py --role pooled --corrected leace
python code/11_trial_alignment.py    --role pooled
python code/12_trial_fit_outcomes.py
python code/07_report.py
```

No GPU: sites ship the vectors, so nothing is re-embedded. Running `03` twice is
deliberate — the second pass redraws every panel from the site-corrected
embeddings into `corrected-leace/`, so each colouring reads raw and corrected
side by side.

Under `--role pooled`, `paths.site_data` **raises**, so a pooled run cannot read
raw records even by mistake. That is the mechanism that makes any of this more
than a promise.

Bundles are discovered by their manifests, so a new site is picked up with no
code change — and the manifests are checked against each other on the way in: a
mismatch in model weights or tokenizer is fatal, since embeddings from different
weights are not in the same space, and a mismatch in the repository commit
warns, since the cohort definition may differ.

A tier-1 bundle produced on a machine holding no GEM weights carries a null
model hash, which is correct: tier 1 never loaded them, so there is nothing to
compare. Its tokenizer hash still travels, and that is the hash the token counts
depend on. Null hashes are skipped by the agreement check rather than read as
disagreement.

See [COORDINATING_SITE.md](COORDINATING_SITE.md) for the full procedure, what to
read first, and the three-site cap on a scatter panel.

---

## Trajectories — `09_trajectories.py`

[`code/09_trajectories.py`](../code/09_trajectories.py) embeds each patient at 26
points around shock onset and draws where they move. It is **not federated** —
26 vectors per patient is far too large a disclosure surface — and runs only
where several contributing sites share a filesystem:

```bash
python code/05_site_correction.py --anchor onset --save-map
sbatch hpc/06_trajectories.sbatch
```

It applies the affine map `05` fitted at the anchor to every grid point *before*
projecting, then draws the flow for both readings of cardiogenic shock — the ICD
code and the measurement-based `cardiac_origin` flag — over a scatter coloured
by Zweck phenotype.

---

## What the pooled run produces

```
output/
  figures/<model>/anchor-*/   projections coloured by site, phenotype, shock type,
                              mortality, ICU LOS, ventilation, pressors, MAP, SOFA,
                              trial eligibility, trial fit, concordance
                 corrected-*/ the identical set, site effect removed
                 trajectory/  09's flow fields
  tables/<model>/             purity, site-effect decomposition, trial tables
```

## Installing more than one tier

| | install | size | needs a GPU |
|---|---|---|---|
| tier 1, every site — **the current request** | `pip install -r requirements-site.txt` | ~200 MB | no |
| tier 2, with a DUA | `pip install -r requirements-embed.txt` | ~10–15 GB | yes |
| coordinating site | `pip install -r requirements-pooled.txt` | ~500 MB | no |
| all three roles | `pip install -r requirements.txt` | ~15 GB | yes |

One line of `cotorra` pulls torch, transformers, datasets, lightgbm, optuna,
xgboost and wandb, which is where the 15 GB comes from — and tier 1 never loads
the model, so it is in `requirements-embed.txt` and nowhere else.

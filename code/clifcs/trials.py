#!/usr/bin/env python3

"""
how close each patient is to the populations the guidelines were tested on

Every cardiogenic shock recommendation rests on a trial that enrolled a narrow
slice of the people who present in shock. This module asks, for each patient,
two separate questions that are easy to conflate:

  ELIGIBILITY   would this trial have enrolled them? A boolean, from the
                published inclusion and exclusion criteria.
  PROXIMITY     do they resemble the patients it did enrol? A Mahalanobis
                distance to the trial's region of the embedding space.

They come apart constantly, and that is the interesting part. A patient can be
ineligible on a formality (81 years old, where ECLS-SHOCK capped at 80) and sit
in the middle of the trial's region; another can satisfy every criterion and sit
far outside it. The first is someone a guideline arguably should cover; the
second is someone it nominally covers but was never tested on.

A NOTE ON CIRCULARITY, because it limits what the distances mean. A trial region
is fitted from *our* eligible patients, not from the trial's -- we have no
embeddings for trial participants and never will. So "inside the SHOCK region"
means "resembles the patients here who would have qualified for SHOCK", not
"resembles the patients SHOCK enrolled". The link between the two is the
representativeness check in `compare_to_published`: if our eligible subset
matches the trial on the few variables trials actually report, the substitution
is reasonable; where it does not, the region is ours and the trial's name on it
is a label of convenience. 11 prints that comparison next to every region.

Eligibility logic lives here rather than in the config because each criterion
needs a comment explaining which CLIF column stands in for it and what is being
given up. The literature values live in code/config/trials.yaml.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import polars as pl

from clifcs.clif import has_table, scan

# --------------------------------------------------------------- ICD code sets
#
# Matched by prefix over ICD-10-CM and ICD-9-CM together. The two systems'
# codes for these conditions cannot collide -- ICD-9's are numeric and
# ICD-10's begin with a letter -- so `diagnosis_code_format` is not consulted,
# and the match does not depend on a column NU populates with two values and
# UCMC with one.
DX_CODES: dict[str, tuple[str, ...]] = {
    # acute myocardial infarction of any type, plus subsequent MI
    "ami": ("I21", "I22", "410"),
    # ST-elevation MI: I21.0-I21.3 are by wall, I21.9 is unspecified and is
    # deliberately NOT included -- it would roughly double the count with
    # patients whose ECG pattern is unknown, and DanGer Shock enrolled on a
    # positive STEMI diagnosis
    "stemi": ("I210", "I211", "I212", "I213"),
    "nstemi": ("I214", "I21A"),
    "cardiac_arrest": ("I46", "4275"),
    # severe peripheral arterial disease, which excludes femoral cannulation
    "severe_pvd": ("I702", "I703", "I704", "I705", "I706", "I707", "I739", "4402"),
    # severe aortic stenosis or regurgitation, a DanGer Shock exclusion
    "aortic_valve": ("I350", "I351", "I352", "I06"),
    "esrd": ("N186", "Z992"),
}


def _normalized_code() -> pl.Expr:
    return (
        pl.col("diagnosis_code")
        .cast(pl.String)
        .str.replace_all(r"[.\s]", "")
        .str.to_uppercase()
    )


def attach_features(data_dir, cohort: pl.DataFrame, cfg, logger) -> pl.DataFrame:
    """
    add the columns the eligibility criteria need and the cohort does not carry

    Two groups: diagnosis flags, and the systolic pressure at shock onset.
    Everything else the criteria read (age, lactate, GCS, vasoactive use,
    inotrope, MCS) is already on the cohort table from 01.
    """
    ids = cohort.select("hospitalization_id")

    # ---------------------------------------------------------- diagnosis flags
    if has_table(data_dir, "clif_hospital_diagnosis"):
        code = _normalized_code()
        flags = (
            scan(data_dir, "clif_hospital_diagnosis")
            .join(ids.lazy(), on="hospitalization_id", how="inner")
            .select("hospitalization_id", c=code)
            .group_by("hospitalization_id")
            .agg(
                **{
                    f"dx_{name}": pl.any_horizontal(
                        *(pl.col("c").str.starts_with(p) for p in prefixes)
                    ).any()
                    for name, prefixes in DX_CODES.items()
                }
            )
            .collect()
        )
        cohort = cohort.join(flags, on="hospitalization_id", how="left", validate="1:1")
    else:
        logger.warning(
            "clif_hospital_diagnosis absent; every diagnosis-based criterion "
            "will read as not met, so trial eligibility will be understated"
        )
    cohort = cohort.with_columns(
        **{
            f"dx_{name}": pl.col(f"dx_{name}").fill_null(False)
            if f"dx_{name}" in cohort.columns
            else pl.lit(False)
            for name in DX_CODES
        }
    )

    # ------------------------------------------------- systolic pressure at onset
    # The lowest plausible SBP in a window around onset. Trials state their
    # threshold as a pressure at presentation ("SBP < 90", "< 100"), and the
    # lowest reading in the window is the one a clinician would have acted on.
    window_hours = 6.0
    lo, hi = 20.0, 300.0
    window = cohort.lazy().select(
        "hospitalization_id",
        win_start=pl.col("cs_onset_dttm") - pl.duration(hours=window_hours),
        win_end=pl.col("cs_onset_dttm") + pl.duration(hours=window_hours),
    )
    sbp = (
        scan(data_dir, "clif_vitals", dttm_cols=("recorded_dttm",))
        .filter(
            pl.col("vital_category") == "sbp",
            pl.col("vital_value").is_between(lo, hi),
        )
        .join(window, on="hospitalization_id", how="inner")
        .filter(pl.col("recorded_dttm").is_between(pl.col("win_start"), pl.col("win_end")))
        .group_by("hospitalization_id")
        .agg(sbp_at_onset=pl.col("vital_value").min())
        .collect()
    )
    cohort = cohort.join(sbp, on="hospitalization_id", how="left", validate="1:1")

    logger.info(
        f"  trial features: AMI {cohort['dx_ami'].mean():.1%}, "
        f"STEMI {cohort['dx_stemi'].mean():.1%}, "
        f"arrest {cohort['dx_cardiac_arrest'].mean():.1%}, "
        f"SBP at onset available for {cohort['sbp_at_onset'].is_not_null().mean():.1%}"
    )
    return cohort


# ----------------------------------------------------------- eligibility rules
#
# Each returns a polars boolean expression over the cohort frame. `shock` below
# is the criterion every one of these trials shares -- hypotension or
# vasoactive support -- which 01 has already established for the whole cohort,
# so it is not re-tested; what differs between trials is the cause, the
# severity thresholds and the exclusions.


def _pressor_or_hypotension() -> pl.Expr:
    return pl.col("met_vasoactive_support") | pl.col("met_sustained_hypotension")


def _elig_shock() -> pl.Expr:
    # AMI with shock. "Shock due to LV failure" is the published criterion and
    # the LV part is not computable, so this is a superset.
    return pl.col("dx_ami") & _pressor_or_hypotension() & (pl.col("age_at_admission") >= 18)


def _elig_iabp_shock_ii() -> pl.Expr:
    # AMI, shock, and planned early revascularisation. The last is not
    # recorded; an MI code with shock at these sites almost always went to the
    # catheter laboratory, so it is left unapplied rather than guessed at.
    return pl.col("dx_ami") & _pressor_or_hypotension()


def _elig_culprit_shock() -> pl.Expr:
    # AMI with shock AND multivessel disease. Coronary anatomy is unavailable,
    # so the multivessel requirement cannot be applied -- a superset, and the
    # one that overstates most, since multivessel disease was the whole point.
    return pl.col("dx_ami") & _pressor_or_hypotension()


def _elig_soap_ii() -> pl.Expr:
    # Shock of any cause requiring a vasopressor. The broadest of the set, and
    # the only one that does not require an infarction.
    return pl.col("met_vasoactive_support")


def _elig_optimacc() -> pl.Expr:
    # AMI-related shock after successful PCI. PCI is not ascertainable, so the
    # "after PCI" qualifier is dropped.
    return pl.col("dx_ami") & pl.col("met_vasoactive_support")


def _elig_ecmo_cs() -> pl.Expr:
    # Rapidly deteriorating or severe shock, any cause. The trial defined this
    # haemodynamically (cardiac index, SvO2) which CLIF does not carry; a
    # vasoactive infusion plus lactate above 3 stands in for "severe".
    return pl.col("met_vasoactive_support") & (pl.col("lactate") > 3.0)


def _elig_ecls_shock() -> pl.Expr:
    # The most specific criteria of the set, and the most of them computable:
    # AMI, shock, lactate > 3, age <= 80, and exclusions for severe peripheral
    # vascular disease. Prolonged CPR (> 45 min) is an exclusion we cannot
    # time, so arrest is left in.
    return (
        pl.col("dx_ami")
        & _pressor_or_hypotension()
        & (pl.col("lactate") > 3.0)
        & (pl.col("age_at_admission") <= 80)
        & ~pl.col("dx_severe_pvd")
    )


def _elig_danger_shock() -> pl.Expr:
    # STEMI, shock with lactate > 2.5, excluding comatose post-arrest patients,
    # severe aortic valve disease and severe peripheral vascular disease.
    # LVEF < 45% is not computable, so this is a superset on the ventricular
    # criterion; everything else is applied.
    comatose_after_arrest = pl.col("dx_cardiac_arrest") & (
        pl.col("gcs_min_in_window") < 8
    )
    return (
        pl.col("dx_stemi")
        & _pressor_or_hypotension()
        & (pl.col("lactate") > 2.5)
        & ~comatose_after_arrest.fill_null(False)
        & ~pl.col("dx_aortic_valve")
        & ~pl.col("dx_severe_pvd")
    )


@dataclasses.dataclass(frozen=True)
class Trial:
    """one trial: what it tested, whom it enrolled, and who here qualifies"""

    key: str
    label: str
    year: int
    intervention: str
    guideline: str
    citation: str
    published: dict
    eligible: Callable[[], pl.Expr]
    #: criteria we cannot evaluate, so the eligible set is a superset
    not_implementable: tuple[str, ...]


#: in chronological order, which is also roughly the order the guidelines
#: accumulated
TRIALS: tuple[Trial, ...] = ()


def load_trials(cfg) -> tuple[Trial, ...]:
    """join the literature values in trials.yaml to the rules defined above"""
    rules = {
        "shock": (_elig_shock, ("shock due to LV failure (no echo)",)),
        "iabp_shock_ii": (
            _elig_iabp_shock_ii,
            ("planned early revascularisation (not recorded)",),
        ),
        "culprit_shock": (
            _elig_culprit_shock,
            ("multivessel disease (no coronary anatomy)",),
        ),
        "soap_ii": (_elig_soap_ii, ()),
        "optimacc": (_elig_optimacc, ("after successful PCI (not recorded)",)),
        "ecmo_cs": (
            _elig_ecmo_cs,
            ("cardiac index / SvO2 severity definition (not recorded)",),
        ),
        "ecls_shock": (_elig_ecls_shock, ("CPR duration > 45 min (not timed)",)),
        "danger_shock": (_elig_danger_shock, ("LVEF < 45% (no echo)",)),
    }
    out = []
    for key, spec in cfg.trials.items():
        if key not in rules:
            raise KeyError(f"trials.yaml defines {key!r} with no eligibility rule")
        rule, gaps = rules[key]
        out.append(
            Trial(
                key=key,
                label=str(spec.label),
                year=int(spec.year),
                intervention=str(spec.intervention),
                guideline=" ".join(str(spec.guideline).split()),
                citation=str(spec.citation),
                published={k: spec.published[k] for k in spec.published},
                eligible=rule,
                not_implementable=gaps,
            )
        )
    return tuple(sorted(out, key=lambda t: t.year))


def add_eligibility(frame: pl.DataFrame, trials) -> pl.DataFrame:
    """one boolean column per trial"""
    return frame.with_columns(
        **{f"elig_{t.key}": t.eligible().fill_null(False) for t in trials}
    )


# --------------------------------------------------------------- trial regions
#
# WHY THIS IS A NEIGHBOURHOOD MEASURE AND NOT A DISTANCE TO A CENTROID
#
# The first two versions of this measured each patient's distance to the
# centroid of a trial's eligible patients -- first Mahalanobis under each
# region's own covariance, then Euclidean under one whitening shared by all of
# them. Both failed, and in the same way: every region came out with a median
# distance of 7.1 and a 95th-percentile radius of 8.5.
#
# That is 50 dimensions talking, not the trials. Distances concentrate as
# dimension grows -- the distance from a centroid to its own members piles up
# at sqrt(d), which is 7.07 for d = 50 -- and the trial centroids sit much
# closer to the global centroid than the patients sit to any of them. So the
# distance to every region was, to two significant figures, the same number,
# and "nearest trial" was noise.
#
# `centroid_separation` below reports that directly, because it is a finding
# about the embedding rather than a nuisance: if trial populations were
# distinct clusters, their centroids would be separated by more than the
# within-group spread, and they are not.
#
# What survives in high dimensions is the *local* structure, which is what 03
# already uses for neighbourhood purity. For each patient, the affinity to a
# trial is the share of their k nearest neighbours that the trial would have
# enrolled. It needs no covariance, does not concentrate, and reads directly:
# 0.8 means four in five of the patients most similar to this one would have
# qualified.


@dataclasses.dataclass
class Region:
    """a trial's neighbourhood of the embedding space"""

    key: str
    label: str
    n_eligible: int
    fitted: bool
    prevalence: float | None = None  # share of the cohort eligible
    #: the affinity a non-eligible patient has to reach to count as
    #: resembling the trial's population -- see `enrichment_threshold`
    threshold: float | None = None
    #: how many of the k neighbours that corresponds to
    threshold_neighbours: int | None = None
    centroid_separation: float | None = None


def neighbour_affinity(
    coords: np.ndarray, masks: dict[str, np.ndarray], k: int, logger
) -> dict[str, np.ndarray]:
    """
    for each patient and trial, the share of their k nearest neighbours the
    trial would have enrolled

    One neighbour graph serves every trial, which is what makes this cheap:
    the graph is built once over the cohort and then each trial's eligibility
    is averaged over it.

    A patient is excluded from their own neighbourhood. Otherwise an eligible
    patient starts with 1/k of the answer already in hand, which inflates
    affinity for exactly the patients whose affinity is least interesting.
    """
    from sklearn.neighbors import NearestNeighbors

    logger.info(f"  neighbour graph: k={k} over {coords.shape[0]:,} patients")
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, index = neighbours.kneighbors(coords)
    index = index[:, 1:]  # drop self

    return {key: mask[index].mean(axis=1) for key, mask in masks.items()}


def centroid_separation(coords: np.ndarray, mask: np.ndarray) -> float:
    """
    how far a trial's centroid sits from the cohort's, in within-group spreads

    A standardised effect size: < 0.2 means the trial's patients are, on
    average, where everyone else is, and no centroid-based region could
    separate them however it was fitted.
    """
    if mask.sum() < 2:
        return float("nan")
    inside = coords[mask]
    delta = np.linalg.norm(inside.mean(axis=0) - coords.mean(axis=0))
    spread = np.sqrt(inside.var(axis=0).sum())
    return float(delta / spread) if spread > 0 else float("nan")


def enrichment_threshold(prevalence: float, k: int, alpha: float) -> tuple[float, int]:
    """
    the affinity at which a neighbourhood is enriched beyond chance

    Returns the smallest affinity -- and the neighbour count it corresponds to
    -- whose upper binomial tail under the trial's base rate falls below
    `alpha`. A patient at or above it has more trial-eligible neighbours than
    sampling the cohort at random would give.

    This replaces a percentile of the ineligible patients' affinities, which
    degenerates exactly where it matters most. DanGer Shock covers 0.6% of this
    cohort, so almost every ineligible patient has an affinity of zero, the 95th
    percentile of that distribution sits at one neighbour out of thirty, and
    4,381 patients were being called "inside the DanGer Shock region" on the
    strength of a single eligible neighbour. Under the binomial rule the bar for
    a trial that rare rises to two or three neighbours, because one is what
    chance produces.
    """
    from scipy.stats import binom

    for count in range(1, k + 1):
        if binom.sf(count - 1, k, prevalence) < alpha:
            return count / k, count
    # no attainable count is enriched at this alpha: the trial is too common
    # for its neighbourhood to be distinguishable, so demand everything
    return 1.0, k


def fit_regions(frame: pl.DataFrame, coords: np.ndarray, trials, cfg, logger):
    """
    characterise each trial's neighbourhood, and how distinct it is at all

    `threshold` comes from `enrichment_threshold`: the point at which a
    patient's neighbourhood holds more trial-eligible patients than chance
    would put there. It scales with the trial's prevalence, which a fixed
    percentile or a fixed lift cannot do -- the rare trials need a higher bar
    precisely because one eligible neighbour out of thirty is unremarkable when
    the base rate is 0.6%.
    """
    min_eligible = int(cfg.regions.min_eligible)
    alpha = float(cfg.regions.enrichment_alpha)
    k = int(cfg.regions.neighbours)

    masks = {
        t.key: frame[f"elig_{t.key}"].to_numpy()
        for t in trials
        if int(frame[f"elig_{t.key}"].sum()) >= min_eligible
    }
    affinity = neighbour_affinity(coords, masks, k, logger) if masks else {}

    regions = []
    for trial in trials:
        mask = frame[f"elig_{trial.key}"].to_numpy()
        n = int(mask.sum())
        if trial.key not in masks:
            logger.warning(
                f"  {trial.label}: only {n} eligible patients "
                f"(need {min_eligible}); no region fitted"
            )
            regions.append(Region(trial.key, trial.label, n_eligible=n, fitted=False))
            continue

        values = affinity[trial.key]
        prevalence = float(mask.mean())
        threshold, threshold_k = enrichment_threshold(prevalence, k, alpha)
        separation = centroid_separation(coords, mask)

        logger.info(
            f"  {trial.label}: {n:,} eligible ({prevalence:.1%}), "
            f"affinity among eligible {np.median(values[mask]):.2f} vs "
            f"{np.median(values[~mask]):.2f} among the rest; "
            f"enriched at >={threshold_k}/{k} neighbours; "
            f"centroid separation {separation:.2f} spreads"
        )
        regions.append(
            Region(
                trial.key,
                trial.label,
                n_eligible=n,
                fitted=True,
                prevalence=prevalence,
                threshold=threshold,
                threshold_neighbours=threshold_k,
                centroid_separation=separation,
            )
        )
    # stash the affinities for add_distances, which the driver calls next
    for region in regions:
        if region.fitted:
            region._affinity = affinity[region.key]  # noqa: SLF001
    return regions


def add_distances(frame: pl.DataFrame, coords: np.ndarray, regions) -> pl.DataFrame:
    """
    per trial: neighbourhood affinity, its lift over the base rate, and whether
    the patient resembles the trial's population

    `lift` is what to compare between trials, not `affinity`. SOAP II would
    have enrolled most of this cohort, so an affinity of 0.9 to it is
    unremarkable, while an affinity of 0.3 to DanGer Shock -- which covers under
    1% -- is extraordinary. Dividing by prevalence makes the two comparable.
    """
    columns = {}
    for region in regions:
        if not region.fitted:
            nan = np.full(frame.height, np.nan)
            columns[f"affinity_{region.key}"] = pl.Series(nan)
            columns[f"lift_{region.key}"] = pl.Series(nan)
            columns[f"inside_{region.key}"] = pl.Series(
                np.zeros(frame.height, dtype=bool)
            )
            continue
        values = region._affinity  # noqa: SLF001
        columns[f"affinity_{region.key}"] = pl.Series(values)
        columns[f"lift_{region.key}"] = pl.Series(values / max(region.prevalence, 1e-9))
        # eligible patients are inside their own trial's region by definition;
        # everyone else has to clear the threshold calibrated on the ineligible
        columns[f"inside_{region.key}"] = pl.Series(
            frame[f"elig_{region.key}"].to_numpy() | (values >= region.threshold)
        )
    return frame.with_columns(**columns)


def rank_and_orphans(frame: pl.DataFrame, regions) -> pl.DataFrame:
    """
    per patient: how many trials cover them, the closest one, and whether none do

    Four summaries, because "does a guideline apply here?" has several
    different honest answers:

      `n_eligible_trials`  how many trials would have enrolled them, counting
                           SOAP II. The formal, criteria-based answer.
      `n_specific_trials`  the same, excluding SOAP II, which asked only for a
                           vasopressor and so covers most of this cohort.
      `nearest_trial`      the trial whose population they most resemble, by
                           neighbourhood lift, whether or not they were
                           eligible for it.
      `evidence_orphan`    no specific trial's criteria AND resembling no
                           trial's population.
    """
    fitted = [r for r in regions if r.fitted]
    elig_cols = [f"elig_{r.key}" for r in regions]
    inside_cols = [f"inside_{r.key}" for r in fitted]
    specific_cols = [c for c in elig_cols if c != "elig_soap_ii"]
    # ranking is over the specific trials too: SOAP II would otherwise be the
    # nearest trial for most of the cohort by sheer prevalence
    rank_regions = [r for r in fitted if r.key != "soap_ii"]
    lift_cols = [f"lift_{r.key}" for r in rank_regions]

    frame = frame.with_columns(
        n_eligible_trials=pl.sum_horizontal(
            *(pl.col(c).cast(pl.Int32) for c in elig_cols)
        ),
        n_specific_trials=pl.sum_horizontal(
            *(pl.col(c).cast(pl.Int32) for c in specific_cols)
        ),
        n_regions_inside=pl.sum_horizontal(
            *(pl.col(c).cast(pl.Int32) for c in inside_cols)
        ),
    )

    if lift_cols:
        labels = [r.label for r in rank_regions]
        frame = frame.with_columns(
            nearest_lift=pl.max_horizontal(*lift_cols),
            nearest_trial=pl.concat_list(*lift_cols)
            .list.arg_max()
            .replace_strict(
                range(len(labels)), labels, default=None, return_dtype=pl.String
            ),
        )
    else:
        frame = frame.with_columns(
            nearest_lift=pl.lit(None, dtype=pl.Float64),
            nearest_trial=pl.lit(None, dtype=pl.String),
        )

    specific_inside = [
        f"inside_{r.key}" for r in fitted if r.key != "soap_ii"
    ]
    return frame.with_columns(
        n_specific_inside=pl.sum_horizontal(
            *(pl.col(c).cast(pl.Int32) for c in specific_inside)
        )
    ).with_columns(
        evidence_orphan=(pl.col("n_specific_trials") == 0)
        & (pl.col("n_specific_inside") == 0),
        criteria_orphan=pl.col("n_specific_trials") == 0,
    )


# ------------------------------------------------------ representativeness check


def compare_to_published(frame: pl.DataFrame, trials, logger) -> pl.DataFrame:
    """
    our eligible subset against the population the trial actually enrolled

    This is what licenses calling a region by a trial's name. Trials report very
    few variables, so the comparison is thin -- age, sex, lactate, systolic
    pressure, and nothing else is consistently available -- but a decade of age
    or a halving of lactate is visible in it, and either would mean the region
    is ours rather than theirs.
    """
    rows = []
    for trial in trials:
        part = frame.filter(pl.col(f"elig_{trial.key}"))
        published = trial.published
        ours = {
            "age": part["age_at_admission"].median() if part.height else None,
            "pct_female": (part["sex_category"] == "Female").mean()
            if part.height
            else None,
            "lactate": part["lactate"].median() if part.height else None,
            "sbp": part["sbp_at_onset"].median()
            if part.height and "sbp_at_onset" in part.columns
            else None,
        }
        theirs = {
            "age": published.get("age_median") or published.get("age_mean"),
            "pct_female": published.get("pct_female"),
            "lactate": published.get("lactate_median"),
            "sbp": published.get("sbp_median"),
        }
        rows.append(
            {
                "trial": trial.label,
                "year": trial.year,
                "n_published": published.get("n"),
                "n_eligible_here": part.height,
                **{f"{k}_published": theirs[k] for k in ours},
                **{f"{k}_here": ours[k] for k in ours},
            }
        )

    out = pl.DataFrame(rows)
    logger.info("our eligible subset vs the enrolled population:")
    for row in out.iter_rows(named=True):
        bits = []
        for name, unit in (("age", "y"), ("lactate", " mmol/L"), ("sbp", " mmHg")):
            them, us = row[f"{name}_published"], row[f"{name}_here"]
            if them is not None and us is not None:
                bits.append(f"{name} {us:.1f}{unit} vs {them:.1f}{unit}")
            elif us is not None:
                bits.append(f"{name} {us:.1f}{unit} vs not reported")
        female = (
            f"women {row['pct_female_here']:.1%} vs {row['pct_female_published']:.1%}"
            if row["pct_female_published"] is not None
            and row["pct_female_here"] is not None
            else None
        )
        if female:
            bits.append(female)
        logger.info(
            f"  {row['trial']:<22} n {row['n_eligible_here']:>6,} here vs "
            f"{row['n_published']:>5,} enrolled   " + "; ".join(bits)
        )
    return out

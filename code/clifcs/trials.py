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
    # the medical-history rows of a baseline table
    "hypertension": ("I10", "I11", "I12", "I13", "I15", "I16", "401", "402",
                     "403", "404", "405"),
    "diabetes": ("E08", "E09", "E10", "E11", "E13", "250"),
    # *prior* infarction, which is what a medical-history row means -- the
    # index event is dx_ami. I25.2 is old MI, Z86.71 a personal history of one.
    "prior_mi": ("I252", "Z8671", "412"),
    "heart_failure": ("I50", "I110", "I130", "I132", "428"),
    "chronic_kidney_disease": ("N18", "585"),
    # anterior wall STEMI, reported separately by DanGer Shock
    "anterior_mi": ("I210", "I2109"),
    # upper gastrointestinal bleeding, for the one hypovolaemic-shock trial
    # whose population resembles this cohort's
    "gi_bleed": (
        "K920", "K921", "K922",  # haematemesis, melaena, unspecified GI bleed
        "K25", "K26", "K27", "K28",  # peptic ulcer with haemorrhage
        "K2211", "K226", "I8501", "I8511",  # oesophageal / varices bleeding
        "5789", "5780", "5781", "4560",
    ),
    "stroke": ("I63", "I64", "Z8673", "434", "436"),
    "prior_pci": ("Z9861", "Z955", "V4582"),
    "prior_cabg": ("Z951", "Z955", "V4581"),
}


#: the flags that stand for a *medical history* row of a baseline table, and
#: so must be restricted to diagnoses present on arrival
#:
#: Without that restriction the comparison is meaningless rather than merely
#: noisy. clif_hospital_diagnosis is hospitalization-level, so I50 catches the
#: acute heart failure of the index event as readily as pre-existing disease,
#: and the DanGer Shock comparison came out at 66.8% against a published 9.3%
#: -- a sevenfold gap that is ascertainment, not case mix.
HISTORY_FLAGS = (
    "hypertension",
    "diabetes",
    "prior_mi",
    "heart_failure",
    "chronic_kidney_disease",
    "stroke",
    "prior_pci",
    "prior_cabg",
)


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
        diagnoses = (
            scan(data_dir, "clif_hospital_diagnosis")
            .join(ids.lazy(), on="hospitalization_id", how="inner")
            .select("hospitalization_id", c=code, poa=pl.col("poa_present"))
        )
        # Event flags use every diagnosis; history flags only those present on
        # arrival, because that is what a baseline table's medical-history row
        # means. poa_present is populated for every UCMC row and 61% of NU's,
        # so a history flag is understated at NU rather than overstated -- the
        # safer direction, and reported in the comparison table.
        present_on_arrival = diagnoses.filter(pl.col("poa") == 1)
        flags = (
            diagnoses.group_by("hospitalization_id")
            .agg(
                **{
                    f"dx_{name}": pl.any_horizontal(
                        *(pl.col("c").str.starts_with(p) for p in prefixes)
                    ).any()
                    for name, prefixes in DX_CODES.items()
                    if name not in HISTORY_FLAGS
                }
            )
            .join(
                present_on_arrival.group_by("hospitalization_id").agg(
                    **{
                        f"dx_{name}": pl.any_horizontal(
                            *(pl.col("c").str.starts_with(p) for p in DX_CODES[name])
                        ).any()
                        for name in HISTORY_FLAGS
                    }
                ),
                on="hospitalization_id",
                how="full",
                coalesce=True,
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

    # ----------------------------------------- MAP and heart rate at onset
    # MAP is derived from SBP/DBP wherever a measured value is missing or
    # implausible. It has to be: `map` is charted for 99.9% of NU
    # hospitalizations and only 15.1% of UCMC's, so a comparison against a
    # trial's published MAP would otherwise be a comparison between sites. The
    # substitution is validated -- at NU, where both exist, DBP + (SBP-DBP)/3
    # reproduces the charted value at r = 0.995 with a median difference of
    # 0.00, because that is how NU computes it.
    paired = (
        scan(data_dir, "clif_vitals", dttm_cols=("recorded_dttm",))
        .filter(
            pl.col("vital_category").is_in(
                ["sbp", "dbp", "map", "heart_rate", "temp_c"]
            )
        )
        .join(window, on="hospitalization_id", how="inner")
        .filter(pl.col("recorded_dttm").is_between(pl.col("win_start"), pl.col("win_end")))
        .select("hospitalization_id", "recorded_dttm", "vital_category", "vital_value")
        .collect()
    )
    if paired.height:
        wide = paired.pivot(
            on="vital_category",
            index=["hospitalization_id", "recorded_dttm"],
            values="vital_value",
            aggregate_function="median",
        )
        for name in ("sbp", "dbp", "map", "heart_rate", "temp_c"):
            if name not in wide.columns:
                wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
        wide = wide.with_columns(
            map_best=pl.coalesce(
                pl.when(pl.col("map").is_between(20.0, 200.0)).then(pl.col("map")),
                pl.when(
                    pl.col("sbp").is_between(20.0, 300.0)
                    & pl.col("dbp").is_between(5.0, 200.0)
                    & (pl.col("sbp") > pl.col("dbp"))
                ).then(pl.col("dbp") + (pl.col("sbp") - pl.col("dbp")) / 3.0),
            ),
            heart_rate_ok=pl.when(
                pl.col("heart_rate").is_between(20.0, 250.0)
            ).then(pl.col("heart_rate")),
            # 30-43 C: temp_c carries unconverted Fahrenheit readings at these
            # sites, the same artefact the phenotype module guards against
            temp_ok=pl.when(pl.col("temp_c").is_between(30.0, 43.0)).then(
                pl.col("temp_c")
            ),
        )
        # the lowest MAP and the heart rate at that same instant, which is what
        # a baseline table records: the state the clinician was looking at
        # The median MAP over the 24 h *after* onset, which is what a
        # MAP-target recommendation is about: the pressure achieved and held,
        # not the nadir that defined the shock. Using the nadir put
        # SEPSISPAM concordance at 11.9%, which only said that a shock
        # patient's worst pressure is below 65 -- true of nearly all of them
        # and not a statement about treatment.
        achieved = (
            wide.drop_nulls("map_best")
            .join(
                cohort.select("hospitalization_id", "cs_onset_dttm"),
                on="hospitalization_id",
                how="inner",
            )
            .filter(
                pl.col("recorded_dttm").is_between(
                    pl.col("cs_onset_dttm"),
                    pl.col("cs_onset_dttm") + pl.duration(hours=24),
                )
            )
            .group_by("hospitalization_id")
            .agg(map_achieved=pl.col("map_best").median())
        )
        at_onset = (
            wide.drop_nulls("map_best")
            .sort(["hospitalization_id", "map_best"])
            .group_by("hospitalization_id")
            .agg(
                map_at_onset=pl.col("map_best").first(),
                heart_rate_at_onset=pl.col("heart_rate_ok").first(),
                # the diastolic reading at that same instant, which ECLS-SHOCK
                # reports alongside the systolic
                dbp_at_onset=pl.col("dbp").first(),
                temp_at_onset=pl.col("temp_ok").first(),
            )
        )
        cohort = cohort.join(
            at_onset, on="hospitalization_id", how="left", validate="1:1"
        ).join(achieved, on="hospitalization_id", how="left", validate="1:1")
    for name in (
        "map_at_onset",
        "heart_rate_at_onset",
        "dbp_at_onset",
        "temp_at_onset",
        "map_achieved",
    ):
        if name not in cohort.columns:
            cohort = cohort.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))

    # ------------------------------------------- which vasoactive agents
    # ECMO-CS reports "therapy at randomisation" -- which drugs were running --
    # and that is a *measured* comparison: a medication administration record
    # against a trial's own drug chart. It also bears directly on the pressure
    # gap. If our patients sit 60 points lower on systolic than the trials'
    # while receiving the same agents, the gap is severity; if they are less
    # supported, the gap is partly treatment intensity.
    #
    # Read from the cohort's `vasoactives_used`, which 01 builds as a sorted
    # comma-joined list of med_categories seen at or after onset.
    agents = (
        "norepinephrine",
        "epinephrine",
        "dobutamine",
        "milrinone",
        "vasopressin",
        "phenylephrine",
        "dopamine",
    )

    # Peri-onset, read straight from the medication table rather than from the
    # cohort's `vasoactives_used` string.
    #
    # This exists because the string is built over the whole admission, and a
    # trial's "therapy at randomisation" is not. Using the admission-wide
    # version put norepinephrine at 86.4% of ECMO-CS-eligible patients against
    # a published 85.5% and vasopressin at 48.4% against 35.0%, which read as
    # "our patients are supported at least as heavily". On the comparable
    # window they are 78.2% and 29.9% -- slightly *less* on both. The
    # admission-wide flags are kept as `ever_<agent>`, since for the negative
    # device trials "never received it" is the right reading.
    window = cohort.lazy().select(
        "hospitalization_id",
        win_start=pl.col("cs_onset_dttm") - pl.duration(hours=6),
        win_end=pl.col("cs_onset_dttm") + pl.duration(hours=6),
    )
    peri = (
        scan(data_dir, "clif_medication_admin_continuous", dttm_cols=("admin_dttm",))
        .join(ids.lazy(), on="hospitalization_id", how="inner")
        .filter(pl.col("med_category").is_in(list(agents)), pl.col("med_dose") > 0)
        .join(window, on="hospitalization_id", how="inner")
        .filter(
            pl.col("admin_dttm").is_between(pl.col("win_start"), pl.col("win_end"))
        )
        .group_by("hospitalization_id")
        .agg(
            **{
                f"on_{agent}": (pl.col("med_category") == agent).any()
                for agent in agents
            }
        )
        .collect()
    )

    if "vasoactives_used" in cohort.columns:
        # Split and match exactly, never substring: "norepinephrine" contains
        # "epinephrine", so a contains() test put 87.9% of ECMO-CS-eligible
        # patients on epinephrine against a published 3.4% -- it was really
        # reporting norepinephrine again.
        used = pl.col("vasoactives_used").fill_null("").str.split(",")
        cohort = cohort.with_columns(
            **{f"ever_{agent}": used.list.contains(agent) for agent in agents}
        )
    else:
        # Loudly, not silently: every per-agent comparison reads 0% without
        # this column, which looks like a finding rather than a missing join.
        logger.warning(
            "vasoactives_used is not on the frame, so no vasoactive-agent "
            "flag can be computed; every 'running' comparison will read 0%"
        )
        cohort = cohort.with_columns(
            **{f"ever_{agent}": pl.lit(None, dtype=pl.Boolean) for agent in agents}
        )

    cohort = cohort.join(peri, on="hospitalization_id", how="left", validate="1:1")
    cohort = cohort.with_columns(
        **{
            f"on_{agent}": pl.col(f"on_{agent}").fill_null(False)
            if f"on_{agent}" in cohort.columns
            else pl.lit(False)
            for agent in agents
        }
    )

    # ------------------------------------------------- corticosteroids
    # ADRENAL and APROCCHSS turn on hydrocortisone, and the steroid med_group
    # is populated at both sites. Restricted to the shock window so that a
    # patient's inhaled or oncological steroid weeks earlier is not counted as
    # treatment for their shock.
    steroid_cats = ("hydrocortisone", "fludrocortisone", "methylprednisolone")
    steroids = (
        scan(
            data_dir, "clif_medication_admin_intermittent", dttm_cols=("admin_dttm",)
        )
        .join(ids.lazy(), on="hospitalization_id", how="inner")
        .filter(pl.col("med_category").is_in(list(steroid_cats)))
        .join(
            cohort.lazy().select(
                "hospitalization_id",
                win_start=pl.col("cs_onset_dttm") - pl.duration(hours=24),
                win_end=pl.col("cs_onset_dttm") + pl.duration(hours=72),
            ),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            pl.col("admin_dttm").is_between(pl.col("win_start"), pl.col("win_end"))
        )
        .group_by("hospitalization_id")
        .agg(
            on_hydrocortisone=(pl.col("med_category") == "hydrocortisone").any(),
            # APROCCHSS tested hydrocortisone PLUS fludrocortisone and was
            # positive where ADRENAL, testing hydrocortisone alone, was not.
            # Both drugs are med_categories at both sites, so this is the one
            # place in the file where two trials' interventions can actually
            # be told apart.
            on_fludrocortisone=(pl.col("med_category") == "fludrocortisone").any(),
            on_any_steroid=pl.lit(True),
        )
        .collect()
    )
    cohort = cohort.join(steroids, on="hospitalization_id", how="left", validate="1:1")
    for name in ("on_hydrocortisone", "on_fludrocortisone", "on_any_steroid"):
        if name not in cohort.columns:
            cohort = cohort.with_columns(pl.lit(False).alias(name))
    cohort = cohort.with_columns(
        on_hydrocortisone=pl.col("on_hydrocortisone").fill_null(False),
        on_fludrocortisone=pl.col("on_fludrocortisone").fill_null(False),
        on_any_steroid=pl.col("on_any_steroid").fill_null(False),
    )

    # ------------------------------------------- weight, height and BMI
    # Reported by IABP-SHOCK II and ECLS-SHOCK, and directly comparable: a
    # scale is a scale. Charted for 96-100% of this cohort's weights and
    # 93-94% of its heights, so the comparison is not thinned by missingness.
    #
    # The plausible ranges are load-bearing. height_cm reaches 36 at UCMC and
    # 91 at NU, which are transcription errors rather than patients, and a
    # single one of those produces a BMI in the hundreds.
    anthropometry = (
        scan(data_dir, "clif_vitals")
        .join(ids.lazy(), on="hospitalization_id", how="inner")
        .filter(
            pl.col("vital_category").is_in(["weight_kg", "height_cm"]),
            pl.col("vital_value").is_not_null(),
        )
        .group_by("hospitalization_id", "vital_category")
        .agg(value=pl.col("vital_value").median())
        .collect()
    )
    if anthropometry.height:
        wide = anthropometry.pivot(
            on="vital_category", index="hospitalization_id", values="value"
        )
        for name in ("weight_kg", "height_cm"):
            if name not in wide.columns:
                wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
        wide = wide.select(
            "hospitalization_id",
            weight_kg=pl.when(pl.col("weight_kg").is_between(25.0, 350.0)).then(
                pl.col("weight_kg")
            ),
            height_cm=pl.when(pl.col("height_cm").is_between(120.0, 230.0)).then(
                pl.col("height_cm")
            ),
        ).with_columns(
            bmi=pl.col("weight_kg") / (pl.col("height_cm") / 100.0) ** 2
        )
        cohort = cohort.join(
            wide, on="hospitalization_id", how="left", validate="1:1"
        )
    for name in ("weight_kg", "height_cm", "bmi"):
        if name not in cohort.columns:
            cohort = cohort.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))

    # ------------------------------------------------- arterial pH at onset
    # ECLS-SHOCK reports admission pH, and ph_arterial is a lab_category at
    # both sites. Taken as the lowest value in the window, matching how the
    # other onset values are taken: the worst reading the clinician saw.
    ph = (
        scan(data_dir, "clif_labs", dttm_cols=("lab_result_dttm",))
        .filter(
            pl.col("lab_category") == "ph_arterial",
            pl.col("lab_value_numeric").is_between(6.5, 8.0),
        )
        .join(
            cohort.lazy().select(
                "hospitalization_id",
                win_start=pl.col("cs_onset_dttm") - pl.duration(hours=window_hours),
                win_end=pl.col("cs_onset_dttm") + pl.duration(hours=window_hours),
            ),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            pl.col("lab_result_dttm").is_between(pl.col("win_start"), pl.col("win_end"))
        )
        .group_by("hospitalization_id")
        .agg(ph_at_onset=pl.col("lab_value_numeric").min())
        .collect()
    )
    cohort = cohort.join(ph, on="hospitalization_id", how="left", validate="1:1")
    if "ph_at_onset" not in cohort.columns:
        cohort = cohort.with_columns(
            ph_at_onset=pl.lit(None, dtype=pl.Float64)
        )

    # altered mental status, as a baseline table reports it
    cohort = cohort.with_columns(
        ams_gcs_lt15=(pl.col("gcs_min_in_window") < 15).fill_null(False)
    )

    # --------------------------------------------- ventilated before onset
    # A baseline table records "intubation before randomisation", so this is
    # invasive ventilation at or before shock onset, not at any point in the
    # stay. The difference is not cosmetic: ventilated-ever ran at 70.9% of
    # DanGer Shock-eligible patients against a published 17.8%, almost all of
    # which was ventilation that started after the shock did.
    if has_table(data_dir, "clif_respiratory_support"):
        onset = cohort.lazy().select("hospitalization_id", "cs_onset_dttm")
        imv = (
            scan(data_dir, "clif_respiratory_support", dttm_cols=("recorded_dttm",))
            .join(ids.lazy(), on="hospitalization_id", how="inner")
            .filter(
                pl.col("device_category").cast(pl.String).str.to_uppercase() == "IMV"
            )
            .join(onset, on="hospitalization_id", how="inner")
            .filter(pl.col("recorded_dttm") <= pl.col("cs_onset_dttm"))
            .select("hospitalization_id")
            .unique()
            .with_columns(imv_before_onset=True)
            .collect()
        )
        cohort = cohort.join(imv, on="hospitalization_id", how="left", validate="1:1")
    if "imv_before_onset" not in cohort.columns:
        cohort = cohort.with_columns(imv_before_onset=pl.lit(False))
    cohort = cohort.with_columns(
        imv_before_onset=pl.col("imv_before_onset").fill_null(False),
        male=pl.col("sex_category") == "Male",
    )

    logger.info(
        f"  trial features: AMI {cohort['dx_ami'].mean():.1%}, "
        f"STEMI {cohort['dx_stemi'].mean():.1%}, "
        f"arrest {cohort['dx_cardiac_arrest'].mean():.1%}, "
        f"SBP {cohort['sbp_at_onset'].is_not_null().mean():.0%} / "
        f"MAP {cohort['map_at_onset'].is_not_null().mean():.0%} / "
        f"HR {cohort['heart_rate_at_onset'].is_not_null().mean():.0%} available"
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


# ------------------------------------- septic and hypovolaemic shock
#
# These gate on the ICD-coded shock type rather than on a derived definition.
# A Sepsis-3 derivation exists in the sibling CLIF-latent-septicshock
# repository and reproducing a worse one here would be the wrong trade; the
# consequence is that these eligible sets inherit every weakness of
# diagnosis-code ascertainment, including the 1.7x septic-shock coding gap
# between the two sites.


def _elig_sepsispam() -> pl.Expr:
    # septic shock on a vasopressor. The fluid-resuscitation prerequisite is
    # not computable -- no intake/output table at either site.
    return pl.col("shock_septic") & pl.col("met_vasoactive_support")


def _elig_vasst() -> pl.Expr:
    # septic shock on norepinephrine. The trial's ">= 5 mcg/min for 6 h"
    # threshold needs a dose-time series this pipeline does not assemble.
    return pl.col("shock_septic") & pl.col("on_norepinephrine")


def _elig_adrenal() -> pl.Expr:
    # septic shock, ventilated, on a vasopressor. The trial also required four
    # hours of vasopressor, which is not applied.
    return (
        pl.col("shock_septic")
        & pl.col("met_vasoactive_support")
        & pl.col("imv_before_onset")
    )


# The five trials added for the Surviving Sepsis recommendations the
# cardiogenic set did not reach. Four of their five predicates reduce to
# "coded septic shock on a vasopressor", which is the same collapse the
# cardiogenic trials show and has the same cause: what separates these trials
# is a fluid volume, a crystalloid brand or a vasopressor duration, and none
# of those three is in a CLIF-2.1 extract. The trials stay listed separately
# because their *populations* differ even where our reconstruction of their
# criteria cannot, and the population comparison is the point.


def _elig_smart() -> pl.Expr:
    # SMART enrolled all adult ICU patients receiving crystalloid, which would
    # be very nearly this whole cohort. It is narrowed here to the population
    # the Surviving Sepsis recommendation addresses, so the eligible set is a
    # sicker slice than the trial's own -- a mismatch the baseline comparison
    # should surface rather than one hidden in the predicate.
    return pl.col("shock_septic") & pl.col("met_vasoactive_support")


def _elig_aprocchss() -> pl.Expr:
    # Septic shock on a vasopressor. Unlike ADRENAL this trial did not require
    # mechanical ventilation, so the predicate is deliberately broader than
    # _elig_adrenal; the two-organ-failure requirement is not applied.
    return pl.col("shock_septic") & pl.col("met_vasoactive_support")


def _elig_sixtyfive() -> pl.Expr:
    # The most faithfully reproducible of the five: age and vasopressor use
    # are both directly available, and only "vasodilatory shock" has to be
    # approximated, by the coded septic type -- so vasodilatory shock that was
    # not coded septic is missed.
    return (
        pl.col("shock_septic")
        & pl.col("met_vasoactive_support")
        & (pl.col("age_at_admission") >= 65)
    )


def _elig_classic() -> pl.Expr:
    # Septic shock already given at least a litre. The volume prerequisite is
    # not computable, so this is a superset on exactly the criterion that
    # defined the trial's starting point.
    return pl.col("shock_septic") & pl.col("met_vasoactive_support")


def _elig_clovers() -> pl.Expr:
    # Sepsis-induced hypotension refractory to initial fluid. Hypotension is
    # admitted as well as vasopressor use, because CLOVERS enrolled before the
    # vasopressor decision rather than after it -- the one place among these
    # five where that distinction changes who qualifies.
    return pl.col("shock_septic") & _pressor_or_hypotension()


def _elig_villanueva() -> pl.Expr:
    # hypovolaemic shock with coded upper gastrointestinal bleeding. The
    # transfusion threshold that defines the trial's arms is invisible here.
    return pl.col("shock_hypovolemic") & pl.col("dx_gi_bleed")


def _elig_crash_2() -> pl.Expr:
    # Deliberately never satisfied. Trauma is not reliably identifiable in this
    # cohort, time from injury is not recorded, and tranexamic acid is not a
    # med_category at either site -- so neither the population nor the
    # intervention can be ascertained. Returning a constant false keeps the
    # trial visible in every table, with zero eligible and its criteria showing
    # why, rather than quietly dropping it.
    return pl.lit(False)


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
    #: the paper's baseline characteristics table, keyed by cohort column name
    table1: dict
    #: which coded shock type this trial's evidence is about
    applies_to: str = "cardiogenic"


#: in chronological order, which is also roughly the order the guidelines
#: accumulated
TRIALS: tuple[Trial, ...] = ()


def _usable_table1(spec, key: str, logger=None) -> dict:
    """
    the baseline table entries that carry a real published value

    trials.yaml holds placeholder entries -- `{median: null, iqr: null,
    verify: true}` -- for every trial whose paper has not been transcribed
    yet. They exist so the gap is itemised and appears in the verification
    checklist rather than being invisible, and they must never reach
    `table1_position`, where a null median would either crash or, worse, be
    silently skipped and inflate `n_compared`.

    Two ways an entry is dropped, and they guard opposite mistakes:

      `verify: true` still set    the value was not reviewed against the paper
      median/pct is null          the value was never filled in

    An entry that is filled but still flagged is the dangerous case -- someone
    transcribed the numbers and forgot to clear the flag, and the trial would
    go on contributing nothing with no sign of it. That one is warned about by
    name rather than dropped quietly.
    """
    published = getattr(spec, "table1", None)
    if not published:
        return {}
    out = {}
    for name in published:
        entry = dict(published[name])
        has_value = entry.get("median") is not None or entry.get("pct") is not None
        flagged = bool(entry.get("verify", False))
        if flagged and has_value and logger is not None:
            logger.warning(
                f"  {key}.table1.{name} has a value but is still marked "
                "`verify: true`, so it is being ignored. Delete the flag in "
                "trials.yaml to let it count."
            )
        if flagged or not has_value:
            continue
        entry.pop("verify", None)
        out[name] = entry
    return out


def table1_to_verify(cfg) -> pl.DataFrame:
    """
    every baseline value still waiting to be read off a paper

    The checklist for filling in trials.yaml by hand. One row per trial and
    variable, carrying the citation to look it up in, the cohort column it
    will be compared against, and whether that column exists at all -- because
    a placeholder for a variable this pipeline cannot compute is wasted
    transcription effort and should be deleted rather than filled.

    `status` is `placeholder` for an unfilled entry and `flagged_but_filled`
    for the inconsistent case `_usable_table1` warns about.
    """
    rows = []
    for key in cfg.trials:
        spec = cfg.trials[key]
        published = spec.get("table1")
        if not published:
            continue
        for name in published:
            entry = dict(published[name])
            has_value = (
                entry.get("median") is not None or entry.get("pct") is not None
            )
            flagged = bool(entry.get("verify", False))
            if not flagged:
                continue
            rows.append(
                {
                    "trial": str(spec.label),
                    "trial_key": key,
                    "citation": str(spec.citation),
                    "citation_verified": bool(spec.get("citation_verified", True)),
                    "variable": name,
                    "label": TABLE1_LABELS.get(name, name),
                    "cohort_column_known": name in TABLE1_LABELS,
                    "confidence": VARIABLE_CONFIDENCE.get(name, "unknown"),
                    "needs": "pct" if "pct" in entry else "median + iqr",
                    "status": "flagged_but_filled" if has_value else "placeholder",
                }
            )
    return pl.DataFrame(rows)


def load_trials(cfg, logger=None) -> tuple[Trial, ...]:
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
        "sepsispam": (
            _elig_sepsispam,
            ("adequate fluid resuscitation (no intake/output table)",),
        ),
        "vasst": (
            _elig_vasst,
            ("norepinephrine >= 5 mcg/min for 6 h (no dose-time series)",),
        ),
        "adrenal": (
            _elig_adrenal,
            ("vasopressor for >= 4 h (not timed)",),
        ),
        "smart": (
            _elig_smart,
            (
                "all-comers ICU population (narrowed to septic shock here)",
                "balanced crystalloid vs saline (no balanced crystalloid "
                "med_category at either site)",
            ),
        ),
        "aprocchss": (
            _elig_aprocchss,
            (
                "vasopressor-dependent for >= 6 h (not timed)",
                "at least two organ failures by SOFA (not used to gate)",
            ),
        ),
        "sixtyfive": (
            _elig_sixtyfive,
            (
                "vasodilatory shock (approximated by the coded septic type)",
                "< 6 h of vasopressor before enrolment (not timed)",
            ),
        ),
        "classic": (
            _elig_classic,
            ("at least 1 L of fluid already given (no intake/output table)",),
        ),
        "clovers": (
            _elig_clovers,
            (
                "refractory to 1-3 L of initial fluid (no intake/output table)",
                "within 4 h of meeting the hypotension criterion (not timed)",
            ),
        ),
        "villanueva": (
            _elig_villanueva,
            ("transfusion threshold (no blood product table at either site)",),
        ),
        "crash_2": (
            _elig_crash_2,
            (
                "trauma population (not identifiable)",
                "time from injury (not recorded)",
                "tranexamic acid (not a med_category at either site)",
            ),
        ),
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
                table1=_usable_table1(spec, key, logger),
                applies_to=str(spec.get("applies_to", "cardiogenic")),
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
    #: the coded shock type this trial's evidence is about, carried over from
    #: the Trial so `rank_and_orphans` can restrict a comparison to the trials
    #: that could bear on a given patient at all
    applies_to: str = "cardiogenic"


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
            regions.append(
                Region(
                    trial.key,
                    trial.label,
                    n_eligible=n,
                    fitted=False,
                    applies_to=trial.applies_to,
                )
            )
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
                applies_to=trial.applies_to,
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

    With **unfitted** regions -- which is how 01 calls it, having no embeddings
    site-side -- there is no population side: `n_regions_inside` and
    `n_specific_inside` are 0, `nearest_trial` is null, and `evidence_orphan`
    collapses to `criteria_orphan`. That is the criteria-only reading, and it is
    the honest one for a tier-1 bundle rather than a degraded one. 11 recomputes
    all of it against fitted regions once the embeddings are pooled.
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
        # literal 0 when nothing was fitted, because `sum_horizontal` over an
        # empty column list raises rather than returning zero -- it cannot know
        # how many rows to produce. That is the tier-1 path: 01 passes unfitted
        # regions so the criteria-only columns can be computed site-side, and
        # before this guard that call died on an empty fold.
        n_regions_inside=pl.sum_horizontal(
            *(pl.col(c).cast(pl.Int32) for c in inside_cols)
        )
        if inside_cols
        else pl.lit(0, dtype=pl.Int32),
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
    frame = frame.with_columns(
        n_specific_inside=pl.sum_horizontal(
            *(pl.col(c).cast(pl.Int32) for c in specific_inside)
        )
        if specific_inside
        else pl.lit(0, dtype=pl.Int32)
    ).with_columns(
        evidence_orphan=(pl.col("n_specific_trials") == 0)
        & (pl.col("n_specific_inside") == 0),
        criteria_orphan=pl.col("n_specific_trials") == 0,
    )
    return _add_in_type(frame, regions)


#: the cohort boolean that says a patient's coded shock is of the type a
#: trial's evidence is about. A patient with several codes matches several,
#: which is right: a mixed septic-cardiogenic patient is genuinely in scope
#: for both literatures.
SHOCK_TYPE_COLUMN: dict[str, str] = {
    "cardiogenic": "shock_cardiogenic",
    "septic": "shock_septic",
    "hypovolemic": "shock_hypovolemic",
}


def _add_in_type(frame: pl.DataFrame, regions) -> pl.DataFrame:
    """
    the same coverage summaries, restricted to the trials that could bear on
    this patient at all

    WHY THE POOLED COLUMNS ARE NOT ENOUGH. Until the sepsis trials were added
    the pool was nearly all cardiogenic and the distinction did not arise.
    With eighteen trials across three literatures it does: `nearest_trial`
    ranks by neighbourhood lift over every fitted region, so a cardiogenic
    shock patient can come back with CLOVERS as their nearest trial. That is
    not false -- they really do sit near those patients in the embedding --
    but it is not an answer to "which guideline covers this person", and read
    as one it is worse than no answer.

    The eligibility predicates already gate on coded shock type, so the
    *criteria* counts barely move. What moves is the population side, which
    is fitted in the embedding and has no such gate.

    `n_trials_in_type` is the denominator and is reported rather than assumed:
    a patient whose shock is coded `none coded` or `other shock` matches no
    trial's `applies_to` at all, and the honest summary for them is "no trial
    in this literature was even applicable", not "no trial fitted".
    """
    specific = [r for r in regions if r.key != "soap_ii"]
    missing = [
        c
        for c in {SHOCK_TYPE_COLUMN[r.applies_to] for r in specific
                  if r.applies_to in SHOCK_TYPE_COLUMN}
        if c not in frame.columns
    ]
    if missing:
        # Without the coded type flags there is no in-type question to ask.
        # Nulls rather than zeros, so a downstream reader cannot mistake
        # "not computed here" for "nothing applied".
        return frame.with_columns(
            n_trials_in_type=pl.lit(None, dtype=pl.Int32),
            n_specific_trials_in_type=pl.lit(None, dtype=pl.Int32),
            n_specific_inside_in_type=pl.lit(None, dtype=pl.Int32),
            nearest_trial_in_type=pl.lit(None, dtype=pl.String),
            nearest_lift_in_type=pl.lit(None, dtype=pl.Float64),
            type_evidence_orphan=pl.lit(None, dtype=pl.Boolean),
        )

    def in_type(region) -> pl.Expr:
        column = SHOCK_TYPE_COLUMN.get(region.applies_to)
        if column is None:
            return pl.lit(False)
        return pl.col(column).fill_null(False)

    frame = frame.with_columns(
        n_trials_in_type=pl.sum_horizontal(
            *(in_type(r).cast(pl.Int32) for r in specific)
        ),
        n_specific_trials_in_type=pl.sum_horizontal(
            *(
                (in_type(r) & pl.col(f"elig_{r.key}")).cast(pl.Int32)
                for r in specific
            )
        ),
        n_specific_inside_in_type=pl.sum_horizontal(
            *(
                (in_type(r) & pl.col(f"inside_{r.key}")).cast(pl.Int32)
                for r in specific
                if r.fitted
            )
        )
        if any(r.fitted for r in specific)
        else pl.lit(0, dtype=pl.Int32),
    )

    # nearest trial among those applicable to this patient's coded type.
    # Out-of-type regions are pushed to -inf rather than null, because
    # arg_max over a list containing nulls does not do the obvious thing.
    rankable = [r for r in specific if r.fitted]
    if rankable:
        masked = [
            pl.when(in_type(r))
            .then(pl.col(f"lift_{r.key}"))
            .otherwise(float("-inf"))
            .fill_null(float("-inf"))
            .alias(f"_lift_in_type_{r.key}")
            for r in rankable
        ]
        frame = frame.with_columns(masked)
        labels = [r.label for r in rankable]
        cols = [f"_lift_in_type_{r.key}" for r in rankable]
        best = pl.max_horizontal(*cols)
        frame = frame.with_columns(
            nearest_lift_in_type=pl.when(best > float("-inf"))
            .then(best)
            .otherwise(None),
            nearest_trial_in_type=pl.when(best > float("-inf"))
            .then(
                pl.concat_list(*cols)
                .list.arg_max()
                .replace_strict(
                    range(len(labels)), labels, default=None, return_dtype=pl.String
                )
            )
            .otherwise(None),
        ).drop(cols)
    else:
        frame = frame.with_columns(
            nearest_lift_in_type=pl.lit(None, dtype=pl.Float64),
            nearest_trial_in_type=pl.lit(None, dtype=pl.String),
        )

    return frame.with_columns(
        # only meaningful where some trial in the patient's literature was
        # applicable; null otherwise, which is the `none coded` case
        type_evidence_orphan=pl.when(pl.col("n_trials_in_type") > 0)
        .then(
            (pl.col("n_specific_trials_in_type") == 0)
            & (pl.col("n_specific_inside_in_type") == 0)
        )
        .otherwise(None),
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
        # `n` is null for a trial whose paper has not been transcribed yet,
        # which is most of the sepsis set. Printed as "?" rather than crashing
        # or quietly reading as zero enrolled.
        enrolled = (
            format(row["n_published"], ">5,")
            if row["n_published"] is not None
            else format("?", ">5")
        )
        logger.info(
            f"  {row['trial']:<22} n {row['n_eligible_here']:>6,} here vs "
            f"{enrolled} enrolled   " + "; ".join(bits)
        )
    return out


# ------------------------------------------ inside the population as published


#: human-readable names for the baseline-table variables
TABLE1_LABELS: dict[str, str] = {
    "age_at_admission": "age (years)",
    "sbp_at_onset": "systolic BP (mmHg)",
    "map_at_onset": "mean arterial BP (mmHg)",
    "heart_rate_at_onset": "heart rate (bpm)",
    "lactate": "arterial lactate (mmol/L)",
    "dbp_at_onset": "diastolic BP (mmHg)",
    "ph_at_onset": "arterial pH",
    "temp_at_onset": "temperature (C)",
    "creatinine": "creatinine (mg/dL)",
    "weight_kg": "weight (kg)",
    "height_cm": "height (cm)",
    "bmi": "body-mass index",
    "on_norepinephrine": "norepinephrine running at onset",
    "on_epinephrine": "epinephrine running at onset",
    "on_dobutamine": "dobutamine running at onset",
    "on_milrinone": "milrinone running at onset",
    "on_vasopressin": "vasopressin running at onset",
    "dx_prior_pci": "prior PCI",
    "dx_prior_cabg": "prior bypass surgery",
    "dx_stemi": "ST-elevation infarction",
    "dx_stroke": "prior stroke",
    "dx_severe_pvd": "peripheral artery disease",
    "ams_gcs_lt15": "altered mental status",
    "sofa_total": "SOFA score",
    "male": "male sex",
    "dx_hypertension": "hypertension",
    "dx_diabetes": "diabetes",
    "dx_prior_mi": "prior myocardial infarction",
    "dx_heart_failure": "heart failure",
    "dx_chronic_kidney_disease": "chronic kidney disease",
    "dx_cardiac_arrest": "resuscitated before enrolment",
    "imv_before_onset": "intubated before onset",
    "dx_anterior_mi": "anterior infarction",
}


#: how far a published variable can be trusted against our version of it
#:
#: `measured` variables are the same kind of observation in both: a trial nurse
#: and a CLIF extract both record a systolic pressure from a monitor, so the
#: comparison is meaningful and these are the only variables that drive the
#: inside/outside determination.
#:
#: `ascertainment` variables are a trial case-report form's medical history
#: against ICD codes, and the two do not measure the same thing in either
#: direction. Restricting to diagnoses present on arrival helped but did not
#: fix it: for DanGer Shock, heart failure still reads 54.3% against a
#: published 9.3% because POA-flagged I50 includes the acute failure of the
#: index event, while prior myocardial infarction collapses to 1.0% against
#: 16.1% because POA is seldom set on a history code. Reported, never used to
#: judge whether a patient is inside the tested population.
VARIABLE_CONFIDENCE: dict[str, str] = {
    "age_at_admission": "measured",
    "sbp_at_onset": "measured",
    "map_at_onset": "measured",
    "heart_rate_at_onset": "measured",
    "lactate": "measured",
    "dbp_at_onset": "measured",
    "ph_at_onset": "measured",
    "temp_at_onset": "measured",
    "creatinine": "measured",
    "weight_kg": "measured",
    "height_cm": "measured",
    "bmi": "measured",
    # a medication administration record against a trial's drug chart: the
    # same kind of observation on both sides, unlike the ICD-derived history
    "on_norepinephrine": "measured",
    "on_epinephrine": "measured",
    "on_dobutamine": "measured",
    "on_milrinone": "measured",
    "on_vasopressin": "measured",
    "imv_before_onset": "measured",
    "male": "measured",
    # A score computed from chart values on both sides, which is why this is
    # `measured` and not `ascertainment`. It is the weakest member of that
    # class all the same: a trial computes SOFA from protocol-mandated daily
    # labs and ours comes from whatever was drawn, so a missing component
    # scores zero and biases ours DOWNWARD. Read a SOFA gap as a floor on the
    # severity difference, not an estimate of it. APACHE II and SAPS II, which
    # several of these papers report instead, are not computed here at all.
    "sofa_total": "measured",
    "dx_hypertension": "ascertainment",
    "dx_diabetes": "ascertainment",
    "dx_prior_mi": "ascertainment",
    "dx_heart_failure": "ascertainment",
    "dx_chronic_kidney_disease": "ascertainment",
    "dx_cardiac_arrest": "ascertainment",
    "dx_anterior_mi": "ascertainment",
    "dx_stemi": "ascertainment",
    "dx_stroke": "ascertainment",
    "dx_severe_pvd": "ascertainment",
    "dx_prior_pci": "ascertainment",
    "dx_prior_cabg": "ascertainment",
    # GCS is measured, but a trial's "altered mental status" is a clinician's
    # judgement and ours is a threshold on a chart value that cannot separate
    # organ dysfunction from sedation. Not comparable.
    "ams_gcs_lt15": "ascertainment",
}


#: which direction is "sicker" for each baseline variable
#:
#: The distance scores below say how FAR a patient sits from the population a
#: trial reported. On its own that conflates the two cases the question is
#: really about: a patient outside the interquartile range because they are
#: older, more acidotic and on more support than anyone the trial enrolled,
#: and a patient outside it because they are healthier than the floor the
#: trial recruited from. Those have opposite implications for whether the
#: result transfers, and averaging them cancels them out.
#:
#: +1  a higher value is sicker or older
#: -1  a lower value is sicker
#: variables absent from this registry have no severity orientation and are
#: excluded from the directional scores while still counting toward distance
#:
#: Heart rate and temperature are deliberately absent rather than guessed.
#: Both are U-shaped in shock -- bradycardia and tachycardia are each bad,
#: hypothermia and fever are each bad -- so a sign would be wrong for half the
#: cohort, and a wrong sign is worse here than a missing one.
VARIABLE_SEVERITY_SIGN: dict[str, int] = {
    "age_at_admission": +1,
    "lactate": +1,
    "creatinine": +1,
    "sofa_total": +1,
    "map_at_onset": -1,
    "sbp_at_onset": -1,
    "dbp_at_onset": -1,
    "ph_at_onset": -1,
}


def criteria_frame(cfg) -> pl.DataFrame:
    """
    every trial's published criteria, with whether this pipeline applies it

    Kept as data rather than rendered in the report, so the same table can be
    read from a notebook and so a reader can sort it by what is missing.
    """
    rows = []
    labels = {k: str(cfg.trials[k].label) for k in cfg.trials}
    for key in cfg.get("criteria", {}) or {}:
        for row in cfg.criteria[key]:
            rows.append(
                {
                    "trial": labels.get(key, key),
                    "kind": str(row.kind),
                    "criterion": str(row.criterion),
                    "applied_here": str(row.applied),
                    "note": str(row.note),
                }
            )
    return pl.DataFrame(rows)


def table1_variables(trial) -> dict:
    """the published baseline variables for a trial, or an empty dict"""
    published = getattr(trial, "table1", None) or {}
    return {str(k): published[k] for k in published}


def table1_position(frame: pl.DataFrame, trial, logger) -> pl.DataFrame:
    """
    for each patient, where they sit relative to a trial's published baseline

    Enrolment criteria are thresholds and admit a wide range of patients; the
    baseline table is what the trial's result was actually observed in. A
    DanGer Shock-eligible patient with a systolic pressure of 60 and a heart
    rate of 70 clears every criterion and sits below both published
    interquartile ranges, so nothing much like them was tested.

    Distance is measured in interquartile ranges, which is the only scale the
    published summaries support -- a paper gives a median and an IQR, not a
    covariance, so there is no way to weight variables against each other or
    to account for their correlation. Treating the variables as independent
    and equally weighted is therefore a constraint of the source material
    rather than a modelling choice, and it is the main reason these scores
    rank patients better than they measure them.

    Seven columns per trial, in three groups.

    COVERAGE -- how much of a comparison there was at all:

      t1_n_published_{key}    continuous variables the paper reported that
                              this cohort could in principle compute
      t1_n_compared_{key}     how many of those this patient actually has
      t1_frac_compared_{key}  the ratio, which is the missingness measure

    MAGNITUDE -- how far outside, 0 when inside:

      t1_outside_{key}   count of variables outside their IQR
      t1_maxdist_{key}   the worst single variable, in IQRs beyond the nearer
                         edge. Kept unchanged because the README quotes it
      t1_absdist_{key}   the MEAN over compared variables, which is the
                         continuous fit score. Averaged rather than summed so
                         that a trial reporting nine variables and one
                         reporting four land on the same scale, and so that a
                         patient missing a variable is not scored as though
                         they were comfortably inside its range

    DIRECTION -- which way, and the reason magnitude alone will not do:

      t1_severity_gap_{key}  mean signed position in IQR units over the
                             variables carrying a severity orientation.
                             Positive means sicker or older than the trial's
                             median, negative means healthier.

    Two things about the severity gap are deliberate. It is measured from the
    MEDIAN rather than from the edge of the interquartile range, because a
    direction that only comes into existence once a patient leaves the IQR is
    a direction with a hole in the middle of it. And it averages only the
    variables in VARIABLE_SEVERITY_SIGN, so a trial whose published table is
    all heart rate and temperature gets a null here rather than a number
    built on variables with no agreed sicker end.

    A patient with no comparable variable gets nulls throughout rather than
    zeros. Zero is the score of someone sitting inside every published range,
    which is the opposite of knowing nothing about them.
    """
    spec = table1_variables(trial)
    key = trial.key
    null_columns = {
        f"t1_outside_{key}": pl.lit(None, dtype=pl.Int32),
        f"t1_maxdist_{key}": pl.lit(None, dtype=pl.Float64),
        f"t1_absdist_{key}": pl.lit(None, dtype=pl.Float64),
        f"t1_severity_gap_{key}": pl.lit(None, dtype=pl.Float64),
        f"t1_n_compared_{key}": pl.lit(0, dtype=pl.Int32),
        f"t1_n_published_{key}": pl.lit(0, dtype=pl.Int32),
        f"t1_frac_compared_{key}": pl.lit(None, dtype=pl.Float64),
    }

    available = [
        name
        for name in spec
        if name in frame.columns and frame[name].is_not_null().any()
    ]
    # Only `measured` variables. An ascertainment variable has no IQR anyway,
    # but the filter is explicit so that adding one to trials.yaml later cannot
    # quietly start driving the inside/outside determination.
    continuous = [
        n
        for n in available
        if spec[n].get("iqr") is not None
        and spec[n].get("median") is not None
        and VARIABLE_CONFIDENCE.get(n) == "measured"
    ]
    proportions = [n for n in available if spec[n].get("pct") is not None]

    if not continuous:
        logger.info(
            f"  {trial.label}: no published IQRs usable here "
            f"({len(proportions)} proportion(s) only); no per-patient position"
        )
        return frame.with_columns(**null_columns)

    oriented = [n for n in continuous if n in VARIABLE_SEVERITY_SIGN]

    helpers = []
    for name in continuous:
        q1, q3 = (float(x) for x in spec[name]["iqr"])
        width = max(q3 - q1, 1e-9)
        value = pl.col(name).cast(pl.Float64)
        # 0 inside the IQR, otherwise how many IQRs beyond the nearer edge.
        # Null in, null out -- filling happens below, against a denominator
        # that excludes the nulls.
        helpers.append(
            pl.max_horizontal(
                (q1 - value) / width, (value - q3) / width, pl.lit(0.0)
            ).alias(f"_t1d_{name}")
        )
        if name in VARIABLE_SEVERITY_SIGN:
            median = float(spec[name]["median"])
            sign = float(VARIABLE_SEVERITY_SIGN[name])
            helpers.append((sign * (value - median) / width).alias(f"_t1s_{name}"))

    frame = frame.with_columns(helpers)

    distance_cols = [f"_t1d_{n}" for n in continuous]
    severity_cols = [f"_t1s_{n}" for n in oriented]
    n_published = len(continuous)

    n_compared = pl.sum_horizontal(
        *(pl.col(c).is_not_null().cast(pl.Int32) for c in distance_cols)
    )
    # The mean over the variables this patient actually has: an explicit sum
    # of the non-null distances over an explicit count of them, rather than
    # mean_horizontal, so the denominator cannot be misread.
    absdist = pl.sum_horizontal(
        *(pl.col(c).fill_null(0.0) for c in distance_cols)
    ) / pl.when(n_compared > 0).then(n_compared).otherwise(None)

    if severity_cols:
        n_oriented = pl.sum_horizontal(
            *(pl.col(c).is_not_null().cast(pl.Int32) for c in severity_cols)
        )
        severity_gap = pl.sum_horizontal(
            *(pl.col(c).fill_null(0.0) for c in severity_cols)
        ) / pl.when(n_oriented > 0).then(n_oriented).otherwise(None)
    else:
        severity_gap = pl.lit(None, dtype=pl.Float64)

    outside = [(pl.col(c) > 0).cast(pl.Int32).fill_null(0) for c in distance_cols]

    frame = frame.with_columns(
        **{
            f"t1_outside_{key}": pl.sum_horizontal(*outside),
            f"t1_maxdist_{key}": pl.max_horizontal(*(pl.col(c) for c in distance_cols)),
            f"t1_absdist_{key}": absdist,
            f"t1_severity_gap_{key}": severity_gap,
            f"t1_n_compared_{key}": n_compared,
            f"t1_n_published_{key}": pl.lit(n_published, dtype=pl.Int32),
            f"t1_frac_compared_{key}": n_compared / float(n_published),
        }
    ).drop(distance_cols + severity_cols)

    eligible = frame.filter(pl.col(f"elig_{key}"))
    if eligible.height:
        covered = float(eligible[f"t1_frac_compared_{key}"].mean())
        gap = eligible[f"t1_severity_gap_{key}"].median()
        shown = f"{gap:+.2f} IQR from their median" if gap is not None else "n/a"
        logger.info(
            f"  {trial.label:<22} {n_published} comparable variable(s), "
            f"{len(oriented)} oriented; {covered:.0%} present per eligible "
            f"patient; median severity gap {shown}"
        )
    return frame


def score_variables(trial, score: str = "absdist") -> list[str]:
    """
    the cohort columns a given fit score for this trial is actually built from

    Needed because the two scores are built from different variable sets and
    the difference decides whether an adjusted model means anything.
    `absdist` averages every measured continuous variable the paper
    published; `severity_gap` averages only the subset carrying a severity
    orientation, which for the older trials can be a single variable.

    SHOCK is the worst case and it is not hypothetical: the only quartile it
    publishes is age, so its severity gap is a rescaling of age alone. Adjust
    that model for age and the spline and the adjuster are the same variable.
    """
    spec = table1_variables(trial)
    continuous = [
        name
        for name, published in spec.items()
        if published.get("iqr") is not None
        and published.get("median") is not None
        and VARIABLE_CONFIDENCE.get(name) == "measured"
    ]
    if score == "severity_gap":
        return [n for n in continuous if n in VARIABLE_SEVERITY_SIGN]
    return continuous


def inside_all_null(k: int) -> float:
    """
    the share of a *representative* cohort that would fall inside all k IQRs

    0.5 ** k, because each published interquartile range contains half the
    trial's own population. This is here because the raw "inside all k" share
    is not comparable between trials and is easy to over-read: ECLS-SHOCK's
    0.6% across seven variables and IABP-SHOCK II's 7.8% across four look very
    different and are both approximately the null. Report the ratio, and read
    the per-variable directional shifts for the actual signal.
    """
    return 0.5**k


def table1_comparison(frame: pl.DataFrame, trial, logger) -> pl.DataFrame:
    """
    per variable: the trial's published value, ours among the eligible, and the
    share of our eligible patients falling outside the published IQR

    This is the table that quantifies the gap. A row where 40% of our eligible
    patients sit below the published first quartile is a row where the
    recommendation is being applied to people the trial barely saw.
    """
    spec = table1_variables(trial)
    eligible = frame.filter(pl.col(f"elig_{trial.key}"))
    rows = []
    for name, published in spec.items():
        if name not in eligible.columns or eligible.height == 0:
            continue
        column = eligible[name]
        if "pct" in published:
            if column.null_count() == column.len():
                continue
            rows.append(
                {
                    "trial": trial.label,
                    "variable": TABLE1_LABELS.get(name, name),
                    "column": name,
                    "kind": "proportion",
                    "confidence": VARIABLE_CONFIDENCE.get(name, "unknown"),
                    "severity_sign": VARIABLE_SEVERITY_SIGN.get(name),
                    "n_eligible": int(eligible.height),
                    "n_present": int(column.len() - column.null_count()),
                    "pct_missing": float(column.null_count() / column.len()),
                    "published": float(published["pct"]),
                    "published_q1": None,
                    "published_q3": None,
                    "ours": float(column.cast(pl.Float64).mean()),
                    "pct_below_q1": None,
                    "pct_above_q3": None,
                    "pct_outside_iqr": None,
                    "approximate": bool(published.get("approximate", False)),
                }
            )
            continue
        values = column.cast(pl.Float64).drop_nulls()
        if values.len() == 0:
            continue
        row = {
            "trial": trial.label,
            "variable": TABLE1_LABELS.get(name, name),
            "column": name,
            "kind": "continuous",
            "confidence": VARIABLE_CONFIDENCE.get(name, "unknown"),
            "severity_sign": VARIABLE_SEVERITY_SIGN.get(name),
            "n_eligible": int(eligible.height),
            "n_present": int(values.len()),
            # Missingness is reported per variable and not only per patient,
            # because the two fail differently: a variable missing for most of
            # the cohort weakens every patient's score a little, while a
            # patient missing most variables has a score that should not be
            # read at all. `t1_frac_compared_{key}` is the second; this is the
            # first.
            "pct_missing": float(1.0 - values.len() / max(eligible.height, 1)),
            "published": float(published["median"]),
            "published_q1": None,
            "published_q3": None,
            "ours": float(values.median()),
            "pct_below_q1": None,
            "pct_above_q3": None,
            "pct_outside_iqr": None,
            "approximate": bool(published.get("approximate", False)),
        }
        if "iqr" in published:
            q1, q3 = (float(x) for x in published["iqr"])
            below = float((values < q1).mean())
            above = float((values > q3).mean())
            # An interquartile range contains half the trial's own population
            # by construction, so a perfectly representative cohort would put
            # 25% below the first quartile and 25% above the third. The excess
            # over that null is the signal; the raw share is not. 66.8% below
            # is a shift of 42 points; 26% below is nothing.
            row.update(
                published_q1=q1,
                published_q3=q3,
                pct_below_q1=below,
                pct_above_q3=above,
                pct_outside_iqr=below + above,
                excess_below_q1=below - 0.25,
                excess_above_q3=above - 0.25,
                # positive means shifted low, negative means shifted high
                net_shift_low=below - above,
            )
            # The same shift restated in the direction that matters
            # clinically. `net_shift_low` is about the number line;
            # `net_shift_sicker` is about the patient, and the two have
            # opposite signs for a variable like MAP where lower is worse.
            # Null where the variable has no agreed sicker end.
            sign = VARIABLE_SEVERITY_SIGN.get(name)
            row["net_shift_sicker"] = (
                None if sign is None else float(sign * (above - below))
            )
        rows.append(row)

    out = pl.DataFrame(rows) if rows else pl.DataFrame()
    if out.height:
        logger.info(f"  {trial.label}: eligible patients vs its baseline table")
        for row in out.iter_rows(named=True):
            if row["kind"] == "proportion":
                mark = " (ascertainment, not comparable)" if row[
                    "confidence"
                ] == "ascertainment" else ""
                logger.info(
                    f"      {row['variable']:<30} ours {row['ours']:>6.1%} vs "
                    f"published {row['published']:>6.1%}{mark}"
                )
            elif row["pct_outside_iqr"] is None:
                logger.info(
                    f"      {row['variable']:<30} ours {row['ours']:>6.1f} vs "
                    f"published {row['published']:>6.1f} (no IQR published)"
                )
            else:
                shift = row["net_shift_low"]
                direction = (
                    f"shifted LOW by {shift:+.0%}"
                    if abs(shift) >= 0.10
                    else "no net shift"
                )
                logger.info(
                    f"      {row['variable']:<30} ours {row['ours']:>7.1f} vs "
                    f"{row['published']:>6.1f} "
                    f"[{row['published_q1']:g}-{row['published_q3']:g}]  "
                    f"below {row['pct_below_q1']:>5.1%} (null 25%), "
                    f"above {row['pct_above_q3']:>5.1%}  -> {direction}"
                )
    return out


# ------------------------------------------------- assigning one guideline


def specificity_order(cfg, logger=None) -> list[str] | None:
    """
    the committed most-specific-first trial ordering from trials.yaml

    Returns None when the key is absent, which makes `assign_guideline` fall
    back to ranking on the local cohort and say so. None rather than an empty
    list, so "not configured" cannot be confused with "configured as empty".
    """
    order = cfg.get("specificity_order")
    if not order:
        if logger is not None:
            logger.warning(
                "trials.yaml has no `specificity_order`, so the guideline "
                "assignment will be ranked on this cohort. Add the committed "
                "ordering if this is a contributing site."
            )
        return None
    return [str(k) for k in order]


def specificity_from_frame(frame: pl.DataFrame, trials) -> list[str]:
    """
    the trial keys ordered most-specific-first by eligible share of `frame`

    Only correct on the derivation cohort. `assign_guideline` takes the
    committed ordering instead -- see `order` there for why.
    """
    return [
        t.key
        for t in sorted(trials, key=lambda t: float(frame[f"elig_{t.key}"].mean()))
    ]


def assign_guideline(
    frame: pl.DataFrame, trials, logger, order: list[str] | None = None
) -> pl.DataFrame:
    """
    the single guideline most relevant to each patient, and how well it fits

    Two columns. `assigned_guideline` names a trial or is null; `guideline_fit`
    says on what basis:

      criteria and population  eligible, and inside every published IQR that
                               could be checked -- the strongest claim available
      criteria only            eligible, but outside at least one published IQR:
                               nominally covered, not actually tested on
      no guideline fits        meets no specific trial's criteria

    Where a patient qualifies for several trials the most *specific* one wins,
    specificity being the trial that enrolled the smallest share of a shock
    cohort. SOAP II would otherwise take almost everyone by covering 89% of
    them, and a recommendation that applies to nearly all shock is the least
    informative thing one could say about any particular patient.

    `order` is that ranking, most specific first, and it is **transferred, not
    recomputed** -- `specificity_order` in trials.yaml, on the same argument as
    the phenotype models in models/. Deriving it from `frame` was fine while
    this only ran once on the pooled cohort; run site-side it is not, because
    the eligible shares reorder between sites and the same patient profile
    would then be assigned a different guideline depending on who scored it.
    Omitting it falls back to the local sort with a warning.

    Reads only `elig_*` and the `t1_*` IQR positions, so it needs no embeddings
    and belongs in tier 1.
    """
    if order:
        rank = {key: i for i, key in enumerate(order)}
        unknown = [t.key for t in trials if t.key not in rank]
        if unknown:
            raise ValueError(
                f"specificity_order does not name {unknown}. Every trial has to "
                "appear in it, or the assignment silently depends on the order "
                "trials happen to be defined in. Add them to "
                "`specificity_order` in trials.yaml, most specific first."
            )
        specificity = sorted(
            (t for t in trials if t.key != "soap_ii"), key=lambda t: rank[t.key]
        )
    else:
        logger.warning(
            "no specificity_order given, so the guideline assignment is being "
            "ranked on THIS cohort's eligible shares. Only correct at the "
            "derivation site -- elsewhere the assignment is not comparable "
            "with any other site's."
        )
        specificity = sorted(
            (t for t in trials if t.key != "soap_ii"),
            key=lambda t: float(frame[f"elig_{t.key}"].mean()),
        )

    # inside every checkable IQR: zero variables outside, and at least one
    # variable was actually comparable.
    #
    # "At least one comparable" is read from `t1_n_compared_` when it is there
    # and `t1_frac_compared_` otherwise, because the two live in different
    # places: the raw count exists only in the site-side frame, while the
    # contract exports the fraction and not the count. The old code tested
    # `t1_outside_` for presence and then used `t1_n_compared_` regardless, so
    # it raised ColumnNotFound on any frame that had been through the export --
    # which is every frame the coordinating site ever sees.
    for trial in trials:
        outside = f"t1_outside_{trial.key}"
        count = f"t1_n_compared_{trial.key}"
        frac = f"t1_frac_compared_{trial.key}"
        comparable = (
            pl.col(count) > 0
            if count in frame.columns
            else pl.col(frac) > 0
            if frac in frame.columns
            else None
        )
        if outside in frame.columns and comparable is not None:
            frame = frame.with_columns(
                **{
                    f"inside_t1_{trial.key}": (
                        (pl.col(outside) == 0) & comparable
                    ).fill_null(False)
                }
            )
        else:
            frame = frame.with_columns(**{f"inside_t1_{trial.key}": pl.lit(False)})

    assigned = pl.lit(None, dtype=pl.String)
    fit = pl.lit("no guideline fits", dtype=pl.String)
    # most specific first, so a later (broader) trial cannot override it
    for trial in reversed(specificity):
        eligible = pl.col(f"elig_{trial.key}")
        inside = eligible & pl.col(f"inside_t1_{trial.key}")
        assigned = pl.when(eligible).then(pl.lit(trial.label)).otherwise(assigned)
        fit = pl.when(inside).then(pl.lit("criteria and population")).otherwise(
            pl.when(eligible).then(pl.lit("criteria only")).otherwise(fit)
        )
    # re-apply the strongest basis in specificity order, so "criteria and
    # population" on a specific trial beats "criteria only" on a broader one
    for trial in reversed(specificity):
        inside = pl.col(f"elig_{trial.key}") & pl.col(f"inside_t1_{trial.key}")
        assigned = pl.when(inside).then(pl.lit(trial.label)).otherwise(assigned)

    out = frame.with_columns(assigned_guideline=assigned, guideline_fit=fit)

    summary = (
        out.group_by("assigned_guideline", "guideline_fit")
        .agg(n=pl.len(), mortality=pl.col("in_hospital_mortality").mean())
        .sort("n", descending=True)
    )
    logger.info("most relevant guideline per patient:")
    for row in summary.iter_rows(named=True):
        name = row["assigned_guideline"] or "(none)"
        logger.info(
            f"  {name:<22} {row['guideline_fit']:<24} n={row['n']:>6,} "
            f"({row['n'] / out.height:>5.1%})  mortality {row['mortality']:>5.1%}"
        )
    return out

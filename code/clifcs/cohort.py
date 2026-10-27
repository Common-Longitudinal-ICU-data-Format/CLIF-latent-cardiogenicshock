#!/usr/bin/env python3

"""
cardiogenic shock cohort construction

Definition applied (see code/config/cohort.yaml for the parameters):

  (1) hemodynamic criterion -- either
        SBP < 90 mmHg sustained for > 30 min, or
        a supportive intervention to maintain SBP > 90 (continuous vasoactive
        infusion at a non-zero dose, or mechanical circulatory support)
  (2) evidence of end-organ damage, within a window around the time (1) is
      first met -- any of altered mental status, lactate >= 2 mmol/L, or
      creatinine-based KDIGO AKI

The base cohort mirrors coreopsis recipes/preprocessing.py (first
hospitalization per patient, >= 24 h long, ICU within the first 24 h) so that
the patients embedded here sit inside the distribution the frozen model saw.
"""

from __future__ import annotations

import dataclasses
import datetime

import polars as pl

from clifcs.clif import has_table, scan


@dataclasses.dataclass
class FilterLedger:
    """running tally of cohort attrition, one row per filter step per site"""

    rows: list[dict] = dataclasses.field(default_factory=list)

    def add(self, site: str, step: str, frame: pl.DataFrame, note: str = "") -> None:
        self.rows.append(
            {
                "site": site,
                "step": step,
                "n_hospitalizations": frame.height,
                "n_patients": frame["patient_id"].n_unique(),
                "note": note,
            }
        )

    def add_raw(
        self, site: str, step: str, n_hosp: int, n_pt: int, note: str = ""
    ) -> None:
        self.rows.append(
            {
                "site": site,
                "step": step,
                "n_hospitalizations": n_hosp,
                "n_patients": n_pt,
                "note": note,
            }
        )

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            self.rows,
            schema={
                "site": pl.String,
                "step": pl.String,
                "n_hospitalizations": pl.Int64,
                "n_patients": pl.Int64,
                "note": pl.String,
            },
        )


def _hours(x: float) -> datetime.timedelta:
    return datetime.timedelta(hours=float(x))


def _minutes(x: float) -> datetime.timedelta:
    return datetime.timedelta(minutes=float(x))


# --------------------------------------------------------------- base cohort


def base_cohort(site: str, data_dir, cfg, ledger: FilterLedger, logger) -> pl.DataFrame:
    """first adult hospitalization per patient, >= 24 h long, with ICU in the first 24 h"""
    bc = cfg.base_cohort
    hosp = scan(
        data_dir,
        "clif_hospitalization",
        dttm_cols=("admission_dttm", "discharge_dttm"),
    ).select(
        "patient_id",
        "hospitalization_id",
        "admission_dttm",
        "discharge_dttm",
        "age_at_admission",
        "admission_type_category",
        "discharge_category",
    )

    df = hosp.collect()
    ledger.add(site, "all hospitalizations", df)

    if bc.get("min_age_years") is not None:
        # before the first-hospitalization pick, so a patient first seen as a
        # child enters on their first adult stay; a missing age is not adult
        df = df.filter(pl.col("age_at_admission") >= float(bc.min_age_years))
        ledger.add(site, f">= {float(bc.min_age_years):g} years at admission", df)

    if bc.first_hospitalization_only:
        # deterministic: earliest admission, ties broken on hospitalization_id,
        # so the selected index hospitalization does not depend on row order
        df = df.sort(["patient_id", "admission_dttm", "hospitalization_id"]).unique(
            subset="patient_id", keep="first", maintain_order=True
        )
        ledger.add(site, "first hospitalization per patient", df)

    if bc.min_hospitalization_duration_days:
        df = df.filter(
            (pl.col("discharge_dttm") - pl.col("admission_dttm"))
            >= pl.duration(days=float(bc.min_hospitalization_duration_days))
        )
        ledger.add(
            site,
            f">= {float(bc.min_hospitalization_duration_days):g}d hospitalization",
            df,
        )

    if bc.require_icu:
        icu = (
            scan(data_dir, "clif_adt", dttm_cols=("in_dttm",))
            .filter(
                pl.col("location_category").str.to_lowercase()
                == cfg.outcomes.icu_location_category
            )
            .group_by("hospitalization_id")
            .agg(first_icu_admit=pl.col("in_dttm").min())
            .collect()
        )
        df = df.join(icu, on="hospitalization_id", how="inner", validate="1:1")
        ledger.add(site, "ICU stay during hospitalization", df)

        df = df.filter(
            pl.col("first_icu_admit")
            <= pl.col("admission_dttm")
            + pl.duration(days=float(bc.icu_within_days_of_admission))
        )
        ledger.add(
            site,
            f"ICU within {float(bc.icu_within_days_of_admission):g}d of admission"
            " [BASE COHORT]",
            df,
        )

    logger.info(
        f"[{site}] base cohort: {df.height} hospitalizations / "
        f"{df['patient_id'].n_unique()} patients"
    )
    return df


def attach_demographics(data_dir, cohort: pl.DataFrame) -> pl.DataFrame:
    """
    join the patient-level columns the rest of the pipeline needs

    `sex_category` feeds the CKD-EPI eGFR equation; the others are carried for
    cohort description. clif_patient is one row per patient at both sites, so
    this is validated as many-to-one.
    """
    patient = (
        scan(data_dir, "clif_patient")
        .select(
            "patient_id",
            "sex_category",
            "race_category",
            "ethnicity_category",
        )
        .collect()
    )
    return cohort.join(patient, on="patient_id", how="left", validate="m:1")


# ------------------------------------------------- (1) hemodynamic criterion


def _sustained_hypotension(data_dir, base: pl.DataFrame, cfg) -> pl.DataFrame:
    """
    first time at which more than `sbp_min_duration_min` of uninterrupted
    SBP < threshold has been observed

    A run is a maximal block of consecutive SBP observations all below
    threshold; any reading at or above threshold breaks it. The reported onset is
    the first observation inside such a run that is more than the required
    duration after the run started, so it is always an actual measurement time.
    """
    hemo = cfg.hemodynamics
    lo, hi = (float(x) for x in hemo.sbp_plausible_range)
    threshold = float(hemo.sbp_threshold_mmhg)
    min_duration = _minutes(hemo.sbp_min_duration_min)

    window = base.lazy().select("hospitalization_id", "admission_dttm", "discharge_dttm")

    sbp = (
        scan(data_dir, "clif_vitals", dttm_cols=("recorded_dttm",))
        .filter(
            pl.col("vital_category") == "sbp",
            pl.col("vital_value").is_between(lo, hi),
        )
        .join(window, on="hospitalization_id", how="inner")
        .filter(
            pl.col("recorded_dttm").is_between(
                pl.col("admission_dttm"), pl.col("discharge_dttm")
            )
        )
        # about 10% of SBP timestamps carry more than one reading (e.g. cuff and
        # arterial line); collapse to the median so a single instant cannot both
        # open and break a hypotensive run
        .group_by("hospitalization_id", "recorded_dttm")
        .agg(sbp=pl.col("vital_value").median())
        .sort("hospitalization_id", "recorded_dttm")
        .with_columns(low=pl.col("sbp") < threshold)
        .with_columns(run_id=pl.col("low").rle_id().over("hospitalization_id"))
        .filter(pl.col("low"))
        .with_columns(
            run_start=pl.col("recorded_dttm").min().over("hospitalization_id", "run_id")
        )
        .filter((pl.col("recorded_dttm") - pl.col("run_start")) > min_duration)
        .group_by("hospitalization_id")
        .agg(sbp_onset_dttm=pl.col("recorded_dttm").min())
    )
    return sbp.collect()


def _vasoactive_support(data_dir, base: pl.DataFrame, cfg) -> pl.DataFrame:
    """first non-zero continuous vasoactive / inotrope infusion"""
    hemo = cfg.hemodynamics
    window = base.lazy().select("hospitalization_id", "admission_dttm", "discharge_dttm")

    meds = scan(
        data_dir, "clif_medication_admin_continuous", dttm_cols=("admin_dttm",)
    ).filter(pl.col("med_category").is_in(list(hemo.vasoactive_categories)))
    if hemo.require_nonzero_dose:
        meds = meds.filter(pl.col("med_dose") > 0)

    return (
        meds.join(window, on="hospitalization_id", how="inner")
        .filter(
            pl.col("admin_dttm").is_between(
                pl.col("admission_dttm"), pl.col("discharge_dttm")
            )
        )
        .group_by("hospitalization_id")
        .agg(
            vasoactive_onset_dttm=pl.col("admin_dttm").min(),
            vasoactives_used=pl.col("med_category").unique().sort().str.join(","),
        )
        .collect()
    )


def hemodynamic_criterion(
    site: str, data_dir, base: pl.DataFrame, cfg, ledger: FilterLedger, logger
) -> pl.DataFrame:
    """apply criterion (1); records the time it is first met as `hemo_onset_dttm`"""
    sbp = _sustained_hypotension(data_dir, base, cfg)
    vaso = _vasoactive_support(data_dir, base, cfg)

    df = (
        base.join(sbp, on="hospitalization_id", how="left")
        .join(vaso, on="hospitalization_id", how="left")
        .with_columns(
            met_sustained_hypotension=pl.col("sbp_onset_dttm").is_not_null(),
            met_vasoactive_support=pl.col("vasoactive_onset_dttm").is_not_null(),
        )
        .with_columns(
            hemo_onset_dttm=pl.min_horizontal(
                "sbp_onset_dttm", "vasoactive_onset_dttm"
            )
        )
    )

    duration = float(cfg.hemodynamics.sbp_min_duration_min)
    threshold = float(cfg.hemodynamics.sbp_threshold_mmhg)
    for label, mask in (
        (f"SBP < {threshold:g} for > {duration:g} min", "met_sustained_hypotension"),
        ("vasoactive / inotrope support", "met_vasoactive_support"),
    ):
        ledger.add(site, f"  [sub] {label}", df.filter(pl.col(mask)))

    df = df.filter(pl.col("hemo_onset_dttm").is_not_null())
    ledger.add(
        site,
        f"SBP < {threshold:g} for > {duration:g} min OR supportive intervention"
        " [CRITERION 1]",
        df,
    )
    logger.info(
        f"[{site}] criterion 1 (hemodynamic): {df.height} hospitalizations / "
        f"{df['patient_id'].n_unique()} patients"
    )
    return df


# --------------------------------------------- (2) end-organ damage criterion


def _ams_evidence(data_dir, hemo: pl.DataFrame, cfg) -> pl.DataFrame:
    """
    altered mental status from GCS

    Qualifies on GCS total below `gcs_total_below`, or a drop of at least
    `gcs_drop_from_baseline` points from the patient's own best GCS during the
    hospitalization. Note this cannot distinguish organ dysfunction from
    sedation, which is a known limitation in ICU cohorts.
    """
    ams = cfg.end_organ.ams
    lo, hi = (float(x) for x in ams.gcs_total_valid_range)
    window = hemo.lazy().select("hospitalization_id", "win_start", "win_end")

    return (
        scan(data_dir, "clif_patient_assessments", dttm_cols=("recorded_dttm",))
        .filter(
            pl.col("assessment_category") == "gcs_total",
            pl.col("numerical_value").is_between(lo, hi),
        )
        .join(window, on="hospitalization_id", how="inner")
        .select(
            "hospitalization_id",
            "recorded_dttm",
            "numerical_value",
            "win_start",
            "win_end",
        )
        .with_columns(
            gcs_baseline=pl.col("numerical_value").max().over("hospitalization_id")
        )
        .filter(
            (pl.col("numerical_value") < float(ams.gcs_total_below))
            | (
                pl.col("numerical_value")
                <= pl.col("gcs_baseline") - float(ams.gcs_drop_from_baseline)
            )
        )
        .filter(pl.col("recorded_dttm").is_between(pl.col("win_start"), pl.col("win_end")))
        .group_by("hospitalization_id")
        .agg(
            ams_dttm=pl.col("recorded_dttm").min(),
            gcs_min_in_window=pl.col("numerical_value").min(),
        )
        .collect()
    )


def _lactate_evidence(data_dir, hemo: pl.DataFrame, cfg) -> pl.DataFrame:
    """lactate at or above threshold (reported in mmol/L at both sites)"""
    lac = cfg.end_organ.lactate
    window = hemo.lazy().select("hospitalization_id", "win_start", "win_end")
    return (
        scan(data_dir, "clif_labs", dttm_cols=("lab_result_dttm",))
        .filter(
            pl.col("lab_category") == "lactate",
            pl.col("lab_value_numeric") >= float(lac.threshold_mmol_l),
        )
        .join(window, on="hospitalization_id", how="inner")
        .filter(
            pl.col("lab_result_dttm").is_between(pl.col("win_start"), pl.col("win_end"))
        )
        .group_by("hospitalization_id")
        .agg(
            lactate_dttm=pl.col("lab_result_dttm").min(),
            lactate_max_in_window=pl.col("lab_value_numeric").max(),
        )
        .collect()
    )


def _aki_evidence(data_dir, hemo: pl.DataFrame, cfg) -> pl.DataFrame:
    """
    creatinine-based KDIGO AKI, standing in for the unavailable urine-output
    criterion

    Qualifies on either a rise of at least `absolute_rise_mg_dl` over the lowest
    creatinine in the preceding `rise_window_hours`, or a value at or above
    `relative_rise_factor` times the lowest creatinine from admission to onset.
    """
    aki = cfg.end_organ.aki
    empty = pl.DataFrame(
        schema={
            "hospitalization_id": pl.String,
            "aki_dttm": pl.Datetime("us"),
            "creatinine_baseline": pl.Float64,
        }
    )
    window = hemo.lazy().select(
        "hospitalization_id",
        "admission_dttm",
        "discharge_dttm",
        "hemo_onset_dttm",
        "win_start",
        "win_end",
    )

    creat = (
        scan(data_dir, "clif_labs", dttm_cols=("lab_result_dttm",))
        .filter(
            pl.col("lab_category") == "creatinine",
            pl.col("lab_value_numeric").is_not_null(),
            pl.col("lab_value_numeric") > 0,
        )
        .join(window, on="hospitalization_id", how="inner")
        .filter(
            pl.col("lab_result_dttm").is_between(
                pl.col("admission_dttm"), pl.col("discharge_dttm")
            )
        )
        # one value per instant, so the rolling-min join below stays 1:1
        .group_by("hospitalization_id", "lab_result_dttm")
        .agg(
            creatinine=pl.col("lab_value_numeric").median(),
            hemo_onset_dttm=pl.col("hemo_onset_dttm").first(),
            win_start=pl.col("win_start").first(),
            win_end=pl.col("win_end").first(),
        )
        .sort("hospitalization_id", "lab_result_dttm")
        .collect()
    )

    if creat.height == 0:
        return empty

    # lowest creatinine in the trailing rise window, inclusive of the current value
    trailing = creat.rolling(
        index_column="lab_result_dttm",
        period=f"{int(float(aki.rise_window_hours) * 60)}m",
        group_by="hospitalization_id",
    ).agg(creatinine_trailing_min=pl.col("creatinine").min())

    baseline = (
        creat.filter(pl.col("lab_result_dttm") <= pl.col("hemo_onset_dttm"))
        .group_by("hospitalization_id")
        .agg(creatinine_baseline=pl.col("creatinine").min())
    )

    return (
        creat.join(
            trailing,
            on=["hospitalization_id", "lab_result_dttm"],
            how="left",
            validate="1:1",
        )
        .join(baseline, on="hospitalization_id", how="left")
        .filter(
            pl.col("lab_result_dttm").is_between(pl.col("win_start"), pl.col("win_end"))
        )
        .filter(
            (
                pl.col("creatinine") - pl.col("creatinine_trailing_min")
                >= float(aki.absolute_rise_mg_dl)
            )
            | (
                pl.col("creatinine")
                >= float(aki.relative_rise_factor) * pl.col("creatinine_baseline")
            )
        )
        .group_by("hospitalization_id")
        .agg(
            aki_dttm=pl.col("lab_result_dttm").min(),
            creatinine_baseline=pl.col("creatinine_baseline").first(),
        )
    )


def end_organ_criterion(
    site: str, data_dir, hemo: pl.DataFrame, cfg, ledger: FilterLedger, logger
) -> pl.DataFrame:
    """apply criterion (2); sets `cs_onset_dttm` to the time both criteria hold"""
    eo = cfg.end_organ
    df = hemo.with_columns(
        win_start=pl.col("hemo_onset_dttm") - _hours(eo.window_lookback_hours),
        win_end=pl.col("hemo_onset_dttm") + _hours(eo.window_lookahead_hours),
    )

    pieces: dict[str, pl.DataFrame] = {}
    if eo.ams.enabled:
        pieces["ams"] = _ams_evidence(data_dir, df, cfg)
    if eo.lactate.enabled:
        pieces["lactate"] = _lactate_evidence(data_dir, df, cfg)
    if eo.aki.enabled:
        pieces["aki"] = _aki_evidence(data_dir, df, cfg)

    for piece in pieces.values():
        df = df.join(piece, on="hospitalization_id", how="left")

    df = df.with_columns(
        **{f"met_{k}": pl.col(f"{k}_dttm").is_not_null() for k in pieces}
    ).with_columns(
        end_organ_dttm=pl.min_horizontal(*(f"{k}_dttm" for k in pieces))
    )

    labels = {
        "ams": f"AMS (GCS < {float(eo.ams.gcs_total_below):g})",
        "lactate": f"lactate >= {float(eo.lactate.threshold_mmol_l):g} mmol/L",
        "aki": "creatinine-based AKI (urine-output surrogate)",
    }
    for key in pieces:
        ledger.add(site, f"  [sub] {labels[key]}", df.filter(pl.col(f"met_{key}")))

    df = df.filter(pl.col("end_organ_dttm").is_not_null())

    # cardiogenic shock is established once both criteria hold
    df = df.with_columns(
        cs_onset_dttm=pl.max_horizontal("hemo_onset_dttm", "end_organ_dttm")
    )
    ledger.add(site, "evidence of end-organ damage [CRITERION 2]", df)
    logger.info(
        f"[{site}] criterion 2 (end-organ): {df.height} hospitalizations / "
        f"{df['patient_id'].n_unique()} patients"
    )
    return df.drop("win_start", "win_end")


# ------------------------------------------------- cardiac origin (optional)


def _warn_unjoinable_formats(data_dir, shock: pl.DataFrame, logger) -> None:
    """
    flag a procedure code system whose ids do not key on hospitalization_id

    UCMC's clif_patient_procedures carries 1.72M distinct ids under
    procedure_code_format "cpt" and *none* of them match a hospitalization;
    only its ICD-10-PCS rows use the real key. A code-set query over that table
    therefore returns zero from the CPT arm and a plausible-looking number from
    the PCS arm, with nothing to indicate half the search silently found
    nothing. Checking coverage per format turns that into a visible warning.
    """
    cohort_ids = set(shock["hospitalization_id"].to_list())
    frame = scan(data_dir, "clif_patient_procedures").select(
        "hospitalization_id", "procedure_code_format"
    )
    formats = (
        frame.select("procedure_code_format").unique().collect().to_series().to_list()
    )
    for fmt in formats:
        ids = set(
            frame.filter(pl.col("procedure_code_format") == fmt)
            .select("hospitalization_id")
            .unique()
            .collect()
            .to_series()
            .to_list()
        )
        covered = len(cohort_ids & ids) / max(len(cohort_ids), 1)
        if covered < 0.01:
            logger.warning(
                f"clif_patient_procedures format {fmt!r} has {len(ids)} ids but "
                f"matches {covered:.1%} of the cohort: it is keyed on something "
                "other than hospitalization_id, so procedures recorded under it "
                "cannot be used here"
            )


def _troponin_evidence(data_dir, shock: pl.DataFrame, cfg) -> pl.DataFrame:
    """
    peak troponin above threshold in a window around cardiogenic shock onset

    Myocardial injury is the most directly available marker of cardiac origin
    in CLIF and, unlike procedures, it comes from clif_labs, which keys on
    hospitalization_id at both sites.
    """
    tc = cfg.cardiac_origin.troponin
    window = shock.lazy().select(
        "hospitalization_id",
        win_start=pl.col("cs_onset_dttm")
        - pl.duration(hours=float(tc.window_lookback_hours)),
        win_end=pl.col("cs_onset_dttm")
        + pl.duration(hours=float(tc.window_lookahead_hours)),
    )
    return (
        scan(data_dir, "clif_labs", dttm_cols=("lab_result_dttm",))
        .filter(
            pl.col("lab_category").is_in(list(tc.categories)),
            pl.col("lab_value_numeric") >= float(tc.threshold_ng_l),
        )
        .join(window, on="hospitalization_id", how="inner")
        .filter(
            pl.col("lab_result_dttm").is_between(pl.col("win_start"), pl.col("win_end"))
        )
        .group_by("hospitalization_id")
        .agg(
            cardiac_troponin=pl.lit(True),
            peak_troponin_ng_l=pl.col("lab_value_numeric").max(),
        )
        .collect()
    )


def _mcs_procedures(data_dir, shock: pl.DataFrame, cfg, logger) -> pl.DataFrame:
    """
    hospitalizations with a mechanical circulatory support procedure

    Read from clif_patient_procedures, not clif_ecmo_mcs: the latter is a
    concept table at both sites and is not reliably populated (it names 82% of
    all UCMC hospitalizations). The code sets cover ECMO, intra-aortic balloon
    pump and percutaneous VAD across both CPT and ICD-10-PCS, because the two
    sites code these differently -- UCMC records them almost entirely in
    ICD-10-PCS while NU uses both.

    These are *procedure* codes: unlike a diagnosis code, which asserts a
    condition, they record a therapy that was actually delivered.
    """
    empty = pl.DataFrame(schema={"hospitalization_id": pl.String, "cardiac_mcs": pl.Boolean})
    if not has_table(data_dir, "clif_patient_procedures"):
        return empty
    _warn_unjoinable_formats(data_dir, shock, logger)

    spec = cfg.cardiac_origin.mcs_procedures
    ecmo = spec.cpt.ecmo
    cpt_codes = [str(c) for c in range(int(ecmo.start), int(ecmo.end) + 1)]
    cpt_codes += [str(c) for c in spec.cpt.codes]
    pcs_prefixes = [str(x) for x in spec.icd10pcs_prefixes]

    code = pl.col("procedure_code").str.replace_all(r"\.", "")
    # matched exactly, not by substring: NU also files HCPCS codes, and
    # "hcpcs" contains "pcs"
    fmt = pl.col("procedure_code_format").str.to_lowercase().str.replace_all(
        r"[^a-z0-9]", ""
    )

    return (
        scan(data_dir, "clif_patient_procedures")
        .join(shock.lazy().select("hospitalization_id"), on="hospitalization_id", how="inner")
        .filter(
            ((fmt == "cpt") & code.is_in(cpt_codes))
            | (
                (fmt == "icd10pcs")
                & pl.any_horizontal(*(code.str.starts_with(p) for p in pcs_prefixes))
            )
        )
        .select("hospitalization_id")
        .unique()
        .with_columns(cardiac_mcs=True)
        .collect()
    )


def cardiac_origin(
    site: str, data_dir, shock: pl.DataFrame, cfg, ledger: FilterLedger, logger
) -> pl.DataFrame:
    """
    flag (and optionally require) evidence that the shock is cardiac in origin

    The study's definition has no cardiac criterion, so it selects
    undifferentiated shock. The flag is always computed and always reported, so
    the size of a cardiac-origin cohort is visible without rerunning; it is only
    applied as a filter when `cardiac_origin.enabled` is set.
    """
    co = cfg.cardiac_origin
    required = set(co.require_any_of)
    ids = shock.lazy().select("hospitalization_id")

    flags: list[pl.Expr] = []

    if "inotrope" in required:
        inotropes = (
            scan(data_dir, "clif_medication_admin_continuous")
            .join(ids, on="hospitalization_id", how="inner")
            .filter(
                pl.col("med_category").is_in(list(co.inotrope_categories)),
                pl.col("med_dose") > 0,
            )
            .select("hospitalization_id")
            .unique()
            .with_columns(cardiac_inotrope=True)
            .collect()
        )
        shock = shock.join(inotropes, on="hospitalization_id", how="left")
        flags.append(pl.col("cardiac_inotrope").fill_null(False))

    if "mcs_procedure" in required:
        shock = shock.join(
            _mcs_procedures(data_dir, shock, cfg, logger),
            on="hospitalization_id",
            how="left",
        )
        flags.append(pl.col("cardiac_mcs").fill_null(False))

    if "troponin" in required:
        shock = shock.join(
            _troponin_evidence(data_dir, shock, cfg), on="hospitalization_id", how="left"
        )
        flags.append(pl.col("cardiac_troponin").fill_null(False))

    shock = shock.with_columns(
        cardiac_origin=pl.any_horizontal(*flags) if flags else pl.lit(False)
    )

    for arm, column in (
        ("inotrope infusion", "cardiac_inotrope"),
        ("MCS procedure", "cardiac_mcs"),
        ("troponin above threshold", "cardiac_troponin"),
    ):
        if column in shock.columns:
            ledger.add(
                site,
                f"    [arm] {arm}",
                shock.filter(pl.col(column).fill_null(False)),
            )

    n_cardiac = int(shock["cardiac_origin"].sum())
    ledger.add(
        site,
        "  [sub] evidence of cardiac origin",
        shock.filter(pl.col("cardiac_origin")),
        note="reported only" if not co.enabled else "applied as filter",
    )
    logger.info(
        f"[{site}] cardiac origin: {n_cardiac}/{shock.height} "
        f"({n_cardiac / max(shock.height, 1):.1%})"
        + ("" if co.enabled else " -- flagged but not applied")
    )

    if co.enabled:
        shock = shock.filter(pl.col("cardiac_origin"))
        ledger.add(site, "cardiac origin required [CRITERION 3]", shock)
    return shock

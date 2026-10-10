#!/usr/bin/env python3

"""
downstream prediction labels, anchored at the embedding time

Each label asks what happens in the `horizon_hours` after the moment the model
read the patient. Labels are computed here from CLIF rather than taken from
cocoa's `*_future` flags, for two reasons: those flags mean "ever after the
anchor" with no horizon, and adding one would mean re-winnowing, which changes
the row order that the extracted features are positionally joined on.

Every task carries an eligibility rule. Predicting intubation among patients
already intubated, or escalation among patients on no vasopressors, measures
nothing; those rows are dropped rather than scored as negatives.
"""

from __future__ import annotations

import datetime

import numpy as np
import polars as pl

from clifcs.clif import scan

#: vasopressor doses converted to norepinephrine equivalents (mcg/kg/min).
#: `factor` multiplies a dose already expressed in `unit`.
NOREPINEPHRINE_EQUIVALENTS = {
    "norepinephrine": {"mcg/kg/min": 1.0},
    "epinephrine": {"mcg/kg/min": 1.0},
    "phenylephrine": {"mcg/kg/min": 0.1},
    "dopamine": {"mcg/kg/min": 1.0 / 150.0},
    "angiotensin": {"ng/kg/min": 0.01, "mcg/kg/min": 10.0},
    # vasopressin is dosed as an absolute rate: 0.04 u/min is conventionally
    # taken as ~0.1 mcg/kg/min of norepinephrine, hence 2.5 per u/min
    "vasopressin": {"u/min": 2.5, "u/hr": 2.5 / 60.0},
}

#: weight-independent units, converted with the patient's weight
PER_MINUTE_UNITS = {"mcg/min": "mcg/kg/min", "u/hr": "u/hr"}

#: mar_action_category values meaning the infusion is not running
STOPPED_ACTIONS = ("stop",)

DEFAULT_WEIGHT_KG = 80.0


def anchor_times(paths, site: str, anchor: str, cohort: pl.DataFrame) -> pl.DataFrame:
    """
    the wall-clock moment each patient's embedding was taken

    Recovered from the winnowed stream rather than assumed: `s_elapsed_past`
    counts seconds from the patient's first token, and the embedding sits at
    the last token before the anchor, which is at or just before the nominal
    cut. Using the nominal time instead would misdate every patient by up to
    one inter-event gap.
    """
    from clifcs.tokens import load_pre_anchor_tokens

    tokens = load_pre_anchor_tokens(paths, site, anchor)
    elapsed = pl.Series(
        "anchor_offset_s",
        [float(e[-1]) if e is not None and len(e) else 0.0 for e in tokens["s_elapsed_past"].to_list()],
    )
    return (
        tokens.select("hospitalization_id")
        .with_columns(elapsed)
        .join(
            cohort.select("hospitalization_id", "admission_dttm", "discharge_dttm",
                          "discharge_category", "site"),
            on="hospitalization_id",
            how="inner",
        )
        .with_columns(
            anchor_dttm=pl.col("admission_dttm")
            + pl.duration(seconds=pl.col("anchor_offset_s"))
        )
    )


def _window(anchors: pl.DataFrame, horizon_hours: float) -> pl.LazyFrame:
    return anchors.lazy().select(
        "hospitalization_id",
        "anchor_dttm",
        horizon_end=pl.col("anchor_dttm")
        + pl.duration(hours=float(horizon_hours)),
    )


def label_mortality(
    data_dir, anchors: pl.DataFrame, horizon_hours: float, cfg
) -> pl.DataFrame:
    """
    in-hospital death within the horizon

    Death time is the discharge timestamp of a hospitalization whose discharge
    category is a death. Patients discharged alive inside the horizon are
    negatives, not censored: the question is in-hospital death, and they left
    the hospital alive.
    """
    expired = list(cfg.outcomes.mortality_discharge_categories)
    return anchors.with_columns(
        label=(
            pl.col("discharge_category").is_in(expired)
            & (
                pl.col("discharge_dttm")
                <= pl.col("anchor_dttm") + pl.duration(hours=float(horizon_hours))
            )
        ),
        eligible=pl.lit(True),
    ).select("hospitalization_id", "label", "eligible")


def label_new_imv(data_dir, anchors: pl.DataFrame, horizon_hours: float, cfg) -> pl.DataFrame:
    """
    first invasive mechanical ventilation within the horizon

    Patients already on IMV at the anchor are ineligible: there is no new
    intubation to predict for them.
    """
    window = _window(anchors, horizon_hours)
    resp = (
        scan(data_dir, "clif_respiratory_support_processed", dttm_cols=("recorded_dttm",))
        .filter(pl.col("device_category").str.to_lowercase() == "imv")
        .join(window, on="hospitalization_id", how="inner")
    )
    before = (
        resp.filter(pl.col("recorded_dttm") <= pl.col("anchor_dttm"))
        .select("hospitalization_id")
        .unique()
        .with_columns(imv_before=True)
        .collect()
    )
    during = (
        resp.filter(
            pl.col("recorded_dttm") > pl.col("anchor_dttm"),
            pl.col("recorded_dttm") <= pl.col("horizon_end"),
        )
        .select("hospitalization_id")
        .unique()
        .with_columns(imv_during=True)
        .collect()
    )
    return (
        anchors.select("hospitalization_id")
        .join(before, on="hospitalization_id", how="left")
        .join(during, on="hospitalization_id", how="left")
        .with_columns(
            eligible=~pl.col("imv_before").fill_null(False),
            label=pl.col("imv_during").fill_null(False),
        )
        .select("hospitalization_id", "label", "eligible")
    )


def _weights(data_dir, anchors: pl.DataFrame) -> pl.DataFrame:
    """median recorded weight per hospitalization, for weight-independent doses"""
    return (
        scan(data_dir, "clif_vitals")
        .filter(pl.col("vital_category") == "weight_kg", pl.col("vital_value") > 20)
        .group_by("hospitalization_id")
        .agg(weight_kg=pl.col("vital_value").median())
        .collect()
    )


def norepinephrine_equivalents(data_dir, anchors: pl.DataFrame) -> pl.DataFrame:
    """
    long table of (hospitalization, time, norepinephrine-equivalent rate)

    One row per administration record, with every agent put on a single
    mcg/kg/min scale. Weight-independent orders (mcg/min, u/hr) are divided by
    the patient's median recorded weight; where no weight exists at all a
    default is used, which matters only for the handful of patients with no
    weight documented anywhere.
    """
    weights = _weights(data_dir, anchors)
    meds = (
        scan(
            data_dir,
            "clif_medication_admin_continuous_converted",
            dttm_cols=("admin_dttm",),
        )
        .filter(pl.col("med_category").is_in(list(NOREPINEPHRINE_EQUIVALENTS)))
        .join(anchors.lazy().select("hospitalization_id"), on="hospitalization_id", how="inner")
        .join(weights.lazy(), on="hospitalization_id", how="left")
        .with_columns(
            weight_kg=pl.col("weight_kg").fill_null(DEFAULT_WEIGHT_KG),
            dose=pl.col("med_dose_converted").fill_null(pl.col("med_dose")),
            unit=pl.col("med_dose_unit_converted").fill_null(pl.col("med_dose_unit")),
        )
        # an explicit stop means the rate goes to zero, not that the last rate
        # persists; without this a stopped pressor keeps contributing forever
        .with_columns(
            dose=pl.when(pl.col("mar_action_category").is_in(list(STOPPED_ACTIONS)))
            .then(0.0)
            .otherwise(pl.col("dose"))
        )
        .with_columns(
            dose=pl.when(pl.col("unit") == "mcg/min")
            .then(pl.col("dose") / pl.col("weight_kg"))
            .otherwise(pl.col("dose")),
            unit=pl.when(pl.col("unit") == "mcg/min")
            .then(pl.lit("mcg/kg/min"))
            .otherwise(pl.col("unit")),
        )
    )

    factor = pl.lit(0.0)
    for category, units in NOREPINEPHRINE_EQUIVALENTS.items():
        for unit, value in units.items():
            factor = (
                pl.when((pl.col("med_category") == category) & (pl.col("unit") == unit))
                .then(pl.lit(value))
                .otherwise(factor)
            )

    return (
        meds.with_columns(nee=(pl.col("dose").fill_null(0.0) * factor).clip(0.0, 100.0))
        .select("hospitalization_id", "med_category", "admin_dttm", "nee")
        .collect()
    )


def label_pressor_escalation(
    data_dir, anchors: pl.DataFrame, horizon_hours: float, cfg, step_hours: float = 1.0
) -> pl.DataFrame:
    """
    a rise of at least `threshold` in total norepinephrine-equivalent dose

    Total NEE is reconstructed on an hourly grid: each agent's last known rate
    is carried forward to each grid point and the agents are summed, so
    concurrent infusions add and a stopped one drops out. Baseline is the total
    at the anchor; the label is whether the peak over the horizon exceeds it by
    the threshold.

    Eligible only if the patient was on some vasopressor at the anchor --
    starting a pressor from nothing is a different event from escalating one,
    and mixing them would make the label two things at once.
    """
    threshold = float(cfg.downstream.pressor_escalation.threshold_nee)
    doses = norepinephrine_equivalents(data_dir, anchors)
    if doses.height == 0:
        return anchors.select("hospitalization_id").with_columns(
            label=pl.lit(False), eligible=pl.lit(False)
        )

    steps = int(round(float(horizon_hours) / step_hours))
    grid = (
        anchors.select("hospitalization_id", "anchor_dttm")
        .join(
            pl.DataFrame({"step": np.arange(0, steps + 1, dtype=np.int64)}),
            how="cross",
        )
        .with_columns(
            grid_dttm=pl.col("anchor_dttm")
            + pl.duration(seconds=(pl.col("step") * step_hours * 3600).cast(pl.Int64))
        )
        .sort("grid_dttm")
    )

    # last known rate of each agent at each grid point
    carried = (
        doses.sort("admin_dttm")
        .join_asof(
            grid.sort("grid_dttm"),
            left_on="admin_dttm",
            right_on="grid_dttm",
            by="hospitalization_id",
            strategy="forward",
        )
        .drop_nulls("step")
    )
    totals = (
        grid.join(
            carried.group_by("hospitalization_id", "med_category", "step")
            .agg(nee=pl.col("nee").last())
            .group_by("hospitalization_id", "step")
            .agg(total=pl.col("nee").sum()),
            on=["hospitalization_id", "step"],
            how="left",
        )
        .with_columns(total=pl.col("total").fill_null(0.0))
    )

    baseline = totals.filter(pl.col("step") == 0).select(
        "hospitalization_id", baseline=pl.col("total")
    )
    peak = (
        totals.filter(pl.col("step") > 0)
        .group_by("hospitalization_id")
        .agg(peak=pl.col("total").max())
    )
    return (
        anchors.select("hospitalization_id")
        .join(baseline, on="hospitalization_id", how="left")
        .join(peak, on="hospitalization_id", how="left")
        .with_columns(
            baseline=pl.col("baseline").fill_null(0.0),
            peak=pl.col("peak").fill_null(0.0),
        )
        .with_columns(
            eligible=pl.col("baseline") > 0,
            label=(pl.col("peak") - pl.col("baseline")) >= threshold,
        )
        .select("hospitalization_id", "label", "eligible")
    )


TASKS = {
    "mortality_24h": label_mortality,
    "new_imv_24h": label_new_imv,
    "pressor_escalation_24h": label_pressor_escalation,
}


#: leak-free clinical baseline measured strictly before the anchor
_BASELINE_LABS = ("lactate", "creatinine", "bicarbonate", "bun", "platelet_count")
_BASELINE_VITALS = ("sbp", "map", "heart_rate", "spo2", "respiratory_rate")


def baseline_features(data_dir, anchors: pl.DataFrame) -> pl.DataFrame:
    """
    a simple clinical predictor set, using only what was known at the anchor

    Exists so the embedding AUCs mean something: "0.78" is uninterpretable
    without knowing what age, a lactate and a blood pressure already buy.
    Every value is the last one recorded at or before the anchor, so nothing
    from the prediction window leaks in. The phenotype labs already on the
    cohort table are deliberately not reused -- those are drawn around
    cardiogenic shock onset, which for the 24h anchor can be after it.
    """
    window = anchors.lazy().select("hospitalization_id", "anchor_dttm")

    def last_before(frame, category_col, time_col, value_col, names, prefix):
        return (
            frame.filter(
                pl.col(category_col).is_in(list(names)),
                pl.col(value_col).is_not_null(),
            )
            .join(window, on="hospitalization_id", how="inner")
            .filter(pl.col(time_col) <= pl.col("anchor_dttm"))
            .sort(time_col)
            .group_by("hospitalization_id", category_col)
            .agg(value=pl.col(value_col).last())
            .collect()
            .pivot(on=category_col, index="hospitalization_id", values="value")
            .rename(lambda c: c if c == "hospitalization_id" else f"{prefix}_{c}")
        )

    labs = last_before(
        scan(data_dir, "clif_labs", dttm_cols=("lab_result_dttm",)),
        "lab_category",
        "lab_result_dttm",
        "lab_value_numeric",
        _BASELINE_LABS,
        "lab",
    )
    vitals = last_before(
        scan(data_dir, "clif_vitals", dttm_cols=("recorded_dttm",)),
        "vital_category",
        "recorded_dttm",
        "vital_value",
        _BASELINE_VITALS,
        "vtl",
    )

    # vasopressor burden at the anchor, summed across concurrent agents
    doses = norepinephrine_equivalents(data_dir, anchors)
    if doses.height:
        nee = (
            doses.join(window.collect(), on="hospitalization_id", how="inner")
            .filter(pl.col("admin_dttm") <= pl.col("anchor_dttm"))
            .sort("admin_dttm")
            .group_by("hospitalization_id", "med_category")
            .agg(nee=pl.col("nee").last())
            .group_by("hospitalization_id")
            .agg(
                nee_at_anchor=pl.col("nee").sum(),
                n_vasopressors=(pl.col("nee") > 0).sum(),
            )
        )
    else:
        nee = pl.DataFrame(
            schema={
                "hospitalization_id": pl.String,
                "nee_at_anchor": pl.Float64,
                "n_vasopressors": pl.Int64,
            }
        )

    return (
        anchors.select("hospitalization_id")
        .join(labs, on="hospitalization_id", how="left")
        .join(vitals, on="hospitalization_id", how="left")
        .join(nee, on="hospitalization_id", how="left")
        .with_columns(
            nee_at_anchor=pl.col("nee_at_anchor").fill_null(0.0),
            n_vasopressors=pl.col("n_vasopressors").fill_null(0),
        )
    )

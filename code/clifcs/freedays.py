#!/usr/bin/env python3

"""
ventilator-free and vasopressor-free days, counted separately

The trial-fit analysis needs an outcome that is not mortality. Mortality in
this cohort runs 4.7% among the evidence orphans and 26.5% among everyone
else, which is a large enough gap that almost any exposure correlates with it
through severity alone; an organ-support outcome at least moves for reasons
other than dying.

The two are counted INDEPENDENTLY rather than as one composite. They are not
the same organ, they do not come off on the same clock -- vasopressors in
hours and ventilators in days -- and a composite would be dominated by
whichever has the longer tail while looking like a statement about both. A
patient can be free of one and not the other for reasons that have nothing to
do with how well they fit a trial.

HOW A DAY IS COUNTED

Day d is the 24 hours from `d` days after shock onset to `d + 1` days after
it, for d in 0..27. A day counts as supported if any qualifying record falls
inside it. Free days are then

    0                                   if the patient died within 28 days
    28 - (number of supported days)     otherwise

WHY DAY BINS RATHER THAN RECONSTRUCTED INTERVALS

The alternative is to rebuild each infusion and each ventilator run as an
interval and integrate. That needs a rule for how long a charted state
persists when nothing follows it, and the rule does the work: an infusion
charted once and never stopped either runs to discharge or is truncated at
some invented cap, and the free-day count moves by days depending on which.
Counting days with any record makes no such assumption. It is coarser, and
the coarseness is in the same unit as the outcome.

What it costs: a patient ventilated for one hour on each of three calendar
days scores three supported days, the same as one ventilated continuously for
72 hours. This biases free days DOWNWARD for intermittent support, and the
bias is larger for vasopressors than for ventilation because pressors are
more often started and stopped. Read a free-day count as a floor.

WHAT ANCHORS DAY ZERO

Shock onset, not admission and not ICU arrival. A trial counts from
randomisation and shock onset is this cohort's nearest equivalent -- it is
what every other anchor in the pipeline uses, and counting from admission
would hand a patient who deteriorated on day 10 of a long admission ten free
days they spent in hospital before the question arose.

TWO ASSUMPTIONS WORTH NAMING

A patient discharged alive before day 28 is counted as free for every
remaining day. This is the standard convention and it is wrong for anyone
readmitted to support at another hospital, which this extract cannot see.

Death after day 28 does not zero the count: the 28-day window closes first,
which is what the free-day definition means. A patient who dies on day 40
having never been ventilated scores 28 ventilator-free days.
"""

from __future__ import annotations

import polars as pl

from clifcs.clif import has_table, scan
from clifcs.labels import norepinephrine_equivalents

#: the window every free-day count is measured over
HORIZON_DAYS = 28


def _day_bins(
    events: pl.LazyFrame,
    window: pl.LazyFrame,
    time_col: str,
    horizon_days: int,
) -> pl.DataFrame:
    """
    per patient, how many of the first `horizon_days` days carry an event

    `events` is any long table with a hospitalization_id and a timestamp;
    `window` carries the onset each patient's day zero starts at. Days are
    counted distinctly, so two records in the same day count once.
    """
    return (
        events.join(window, on="hospitalization_id", how="inner")
        .with_columns(
            day=(
                (pl.col(time_col) - pl.col("cs_onset_dttm")).dt.total_seconds()
                / 86400.0
            ).floor()
        )
        .filter(pl.col("day").is_between(0, horizon_days - 1))
        .group_by("hospitalization_id")
        .agg(n_days=pl.col("day").n_unique())
        .collect()
    )


def _free_days(
    cohort: pl.DataFrame,
    supported: pl.DataFrame,
    name: str,
    horizon_days: int,
) -> pl.DataFrame:
    """
    turn a supported-day count into a free-day count, applying the death rule

    A patient with no row in `supported` was never on that support, which is a
    count of zero rather than a missing value -- the absence of a ventilator
    record is informative in a way the absence of a lactate is not.
    """
    days_col = f"{name}_days"
    free_col = f"{name}_free_days"
    return (
        cohort.join(
            supported.rename({"n_days": days_col}),
            on="hospitalization_id",
            how="left",
        )
        .with_columns(pl.col(days_col).fill_null(0).cast(pl.Int32))
        .with_columns(
            **{
                free_col: pl.when(pl.col("died_within_horizon"))
                .then(pl.lit(0, dtype=pl.Int32))
                .otherwise(
                    (horizon_days - pl.col(days_col))
                    .clip(0, horizon_days)
                    .cast(pl.Int32)
                )
            }
        )
    )


def add_free_days(
    data_dir,
    cohort: pl.DataFrame,
    logger,
    horizon_days: int = HORIZON_DAYS,
) -> pl.DataFrame:
    """
    attach ventilator-free and vasopressor-free days to one site's cohort

    Seven columns: `vent_days` / `vent_free_days`, `pressor_days` /
    `pressor_free_days`, `died_within_horizon`, and `free_days_ascertained`.

    The last one matters. A site whose extract has no respiratory support
    table would otherwise report every patient as never ventilated and so as
    having a full 28 ventilator-free days, which is the best possible outcome
    rather than an absent measurement. Where a source table is missing the
    corresponding columns are null and `free_days_ascertained` says so.
    """
    window = cohort.lazy().select("hospitalization_id", "cs_onset_dttm")

    cohort = cohort.with_columns(
        died_within_horizon=(
            pl.col("in_hospital_mortality")
            & (
                (pl.col("discharge_dttm") - pl.col("cs_onset_dttm")).dt.total_seconds()
                <= horizon_days * 86400.0
            )
        ).fill_null(False)
    )
    deaths = int(cohort["died_within_horizon"].sum())
    logger.info(
        f"  free days: {deaths:,} of {cohort.height:,} "
        f"({deaths / max(cohort.height, 1):.1%}) died within {horizon_days} "
        "days of onset and score zero on both"
    )

    ascertained = []

    # ------------------------------------------------- invasive ventilation
    if has_table(data_dir, "clif_respiratory_support_processed"):
        imv = (
            scan(
                data_dir,
                "clif_respiratory_support_processed",
                dttm_cols=("recorded_dttm",),
            )
            .filter(pl.col("device_category").str.to_lowercase() == "imv")
            .select("hospitalization_id", "recorded_dttm")
        )
        cohort = _free_days(
            cohort,
            _day_bins(imv, window, "recorded_dttm", horizon_days),
            "vent",
            horizon_days,
        )
        ascertained.append("vent")
    else:
        logger.warning(
            "clif_respiratory_support_processed absent; ventilator-free days "
            "are null rather than a full 28, which is what counting no "
            "ventilator records as no ventilation would produce"
        )
        cohort = cohort.with_columns(
            vent_days=pl.lit(None, dtype=pl.Int32),
            vent_free_days=pl.lit(None, dtype=pl.Int32),
        )

    # ------------------------------------------------------- vasopressors
    #
    # Read off the norepinephrine-equivalent table rather than the raw
    # medication rows, because that is where an explicit stop has already been
    # turned into a rate of zero. A stop record is still a record, and
    # counting it as a supported day would keep a patient "on vasopressors"
    # on the day they came off them.
    doses = norepinephrine_equivalents(data_dir, cohort.select("hospitalization_id"))
    if doses.height:
        running = (
            doses.lazy()
            .filter(pl.col("nee") > 0)
            .select("hospitalization_id", "admin_dttm")
        )
        cohort = _free_days(
            cohort,
            _day_bins(running, window, "admin_dttm", horizon_days),
            "pressor",
            horizon_days,
        )
        ascertained.append("pressor")
    else:
        logger.warning(
            "no vasoactive infusions found; vasopressor-free days are null "
            "rather than a full 28"
        )
        cohort = cohort.with_columns(
            pressor_days=pl.lit(None, dtype=pl.Int32),
            pressor_free_days=pl.lit(None, dtype=pl.Int32),
        )

    cohort = cohort.with_columns(
        free_days_ascertained=pl.lit("+".join(ascertained) or "none")
    )

    for name in ("vent", "pressor"):
        column = f"{name}_free_days"
        if cohort[column].null_count() == cohort.height:
            continue
        alive = cohort.filter(~pl.col("died_within_horizon"))
        logger.info(
            f"  {name}-free days: median {cohort[column].median():.0f} overall, "
            f"{alive[column].median():.0f} among {alive.height:,} survivors; "
            f"{float((cohort[column] == 0).mean()):.1%} score zero, "
            f"median {cohort[f'{name}_days'].median():.0f} supported day(s)"
        )

    return cohort

#!/usr/bin/env python3

"""
when the record ends, and how

Phase 1 found that a third of decision steps were undecidable, rising from
16.5% on day 0 to 53.3% on day 6 -- and that this was the record ENDING, not
sparse charting: inside the first 48 hours it is 18.6%. An hourly grid run out
to seven days asks about hours that do not exist for most patients, and without
a discharge time a discharged patient and an unmonitored one are the same row.

This reads `clif_hospitalization`, which carries `discharge_dttm` and
`discharge_category`, so every series can be bounded at the hour the record
stops and the stop can be classified.

HOSPICE IS A SEPARATE DISPOSITION FROM DEATH and is present at all three sites
(250 / 1,028 / 350 patients). `clif_code_status` carries only DNAR/DNI, DNAR
and Full -- no comfort-care order -- so a discharge to hospice is the closest
thing in this data to a goals-of-care outcome. It is a DISPOSITION, fixed at
the end of the stay, not a timestamped in-stay transition, so it can stand as
an absorbing state but cannot date the decision that led to it.

DISCHARGE ALIVE IS NOT RECOVERY. A patient discharged to a long-term care
hospital still on support has ended their record without recovering, and
nothing here should be read as a recovery event -- that definition lives in
phase 3 and requires six sustained hours off pressors with an adequate MAP.
"""

from __future__ import annotations

import polars as pl

from clifcs.clif import scan

#: discharge_category values meaning the patient died in hospital
DEATH = ("Expired",)
#: discharge to hospice: a goals-of-care outcome, kept apart from death
HOSPICE = ("Hospice",)


def dispositions(data_dir, anchors, logger):
    """
    one row per patient: when the record ends, how, and hours from onset

    `anchors` needs hospitalization_id and onset. `hours_to_discharge` is
    measured from onset, so it is directly comparable to the `offset_hours`
    every hourly series is keyed on.
    """
    frame = (
        scan(data_dir, "clif_hospitalization", dttm_cols=("discharge_dttm",))
        .join(anchors.lazy().select("hospitalization_id"),
              on="hospitalization_id", how="inner")
        .select("hospitalization_id", "discharge_dttm", "discharge_category")
        .collect()
        .join(anchors.select("hospitalization_id", "onset"),
              on="hospitalization_id", how="inner")
        .with_columns(
            hours_to_discharge=(
                pl.col("discharge_dttm").cast(pl.Datetime("us"))
                - pl.col("onset")
            ).dt.total_seconds() / 3600.0,
            died=pl.col("discharge_category").is_in(list(DEATH)),
            to_hospice=pl.col("discharge_category").is_in(list(HOSPICE)),
        )
        .drop("onset")
    )
    # A discharge BEFORE onset is a data error, not a zero-length stay: onset
    # was derived from events inside the encounter, so the encounter cannot
    # have ended first. Reported rather than clipped, because clipping would
    # hide a cohort-construction problem as a very short stay.
    backwards = frame.filter(pl.col("hours_to_discharge") < 0)
    if backwards.height:
        logger.warning(
            f"  {backwards.height:,} patients discharge before their shock "
            "onset; their records cannot be bounded and they are flagged"
        )
    logger.info(
        f"  dispositions: {frame.height:,} patients, "
        f"{frame['died'].sum():,} died ({frame['died'].mean():.1%}), "
        f"{frame['to_hospice'].sum():,} to hospice "
        f"({frame['to_hospice'].mean():.1%}), median stay after onset "
        f"{frame['hours_to_discharge'].median():.0f}h"
    )
    return frame


def bound_to_record(hourly, disposition, logger, grace_hours: float = 0.0):
    """
    drop grid hours past the end of the record

    Without this the state distribution is partly a distribution over hours
    that never happened: an hourly grid to seven days covers a median stay of
    well under that, and every hour past discharge arrives with no MAP and no
    dose, which the encoder can only call undecidable.

    `grace_hours` extends the bound, for the case where a measurement is
    charted a little after the recorded discharge time. Zero by default --
    anything after discharge is outside the record unless there is a reason to
    think otherwise.
    """
    before = hourly.height
    bounded = (
        hourly.join(
            disposition.select("hospitalization_id", "hours_to_discharge"),
            on="hospitalization_id", how="left",
        )
        .filter(
            pl.col("hours_to_discharge").is_null()
            | (pl.col("offset_hours") <= pl.col("hours_to_discharge")
               + grace_hours)
        )
    )
    logger.info(
        f"  bounded to the record: {before - bounded.height:,} of "
        f"{before:,} patient-hours dropped as past discharge "
        f"({(before - bounded.height) / max(before, 1):.1%})"
    )
    return bounded

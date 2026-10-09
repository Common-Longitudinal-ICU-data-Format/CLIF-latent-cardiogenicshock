#!/usr/bin/env python3

"""
outcomes used to colour the embedding space: ICU length of stay and in-hospital
mortality
"""

from __future__ import annotations

import polars as pl

from clifcs.clif import scan


def icu_length_of_stay(data_dir, cohort: pl.DataFrame, cfg) -> pl.DataFrame:
    """
    total ICU time per hospitalization, in days

    Summing ADT rows directly would double-count: a patient can have
    back-to-back or overlapping ICU rows (unit moves within the same ICU, or
    rows reopened after a procedure). Overlapping and abutting intervals are
    merged first, then the merged durations are summed. An ICU row with a null
    out_dttm is closed at discharge.

    Intervals are also clipped to the hospitalization window. Without it 723 of
    34,139 patients come out with more ICU time than hospital time, because some
    ADT rows run past discharge_dttm -- an artifact of how units close out their
    records, not real ICU time.
    """
    window = cohort.lazy().select(
        "hospitalization_id", "admission_dttm", "discharge_dttm"
    )

    icu = (
        scan(data_dir, "clif_adt", dttm_cols=("in_dttm", "out_dttm"))
        .filter(
            pl.col("location_category").str.to_lowercase()
            == cfg.outcomes.icu_location_category
        )
        .join(window, on="hospitalization_id", how="inner")
        .with_columns(out_dttm=pl.col("out_dttm").fill_null(pl.col("discharge_dttm")))
        .with_columns(
            in_dttm=pl.max_horizontal("in_dttm", "admission_dttm"),
            out_dttm=pl.min_horizontal("out_dttm", "discharge_dttm"),
        )
        .filter(pl.col("out_dttm") >= pl.col("in_dttm"))
        .select("hospitalization_id", "in_dttm", "out_dttm")
        .sort("hospitalization_id", "in_dttm", "out_dttm")
        # a row starts a new block only if it begins after every preceding row
        # in the same hospitalization has already ended
        .with_columns(
            prior_end=pl.col("out_dttm")
            .cum_max()
            .shift(1)
            .over("hospitalization_id")
        )
        .with_columns(
            starts_block=pl.col("prior_end").is_null()
            | (pl.col("in_dttm") > pl.col("prior_end"))
        )
        .with_columns(
            block=pl.col("starts_block").cum_sum().over("hospitalization_id")
        )
        .group_by("hospitalization_id", "block")
        .agg(block_start=pl.col("in_dttm").min(), block_end=pl.col("out_dttm").max())
        .group_by("hospitalization_id")
        .agg(
            icu_los_days=(pl.col("block_end") - pl.col("block_start"))
            .dt.total_seconds()
            .sum()
            / 86400.0,
            n_icu_blocks=pl.len(),
        )
    )
    return icu.collect()


def add_outcomes(data_dir, cohort: pl.DataFrame, cfg) -> pl.DataFrame:
    """attach ICU LOS, hospital LOS, and in-hospital mortality to the cohort"""
    expired = list(cfg.outcomes.mortality_discharge_categories)
    return (
        cohort.join(
            icu_length_of_stay(data_dir, cohort, cfg),
            on="hospitalization_id",
            how="left",
        )
        .with_columns(
            in_hospital_mortality=pl.col("discharge_category").is_in(expired),
            hospital_los_days=(
                pl.col("discharge_dttm") - pl.col("admission_dttm")
            ).dt.total_seconds()
            / 86400.0,
            hours_admission_to_cs_onset=(
                pl.col("cs_onset_dttm") - pl.col("admission_dttm")
            ).dt.total_seconds()
            / 3600.0,
        )
        .with_columns(icu_los_days=pl.col("icu_los_days").fill_null(0.0))
    )

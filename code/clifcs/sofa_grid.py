#!/usr/bin/env python3

"""
SOFA at every point on a trajectory grid, via clifpy

WHY THIS EXISTS. `stage_derive` already computes SOFA with clifpy, but it
collapses the result to one row per encounter stamped at the LAST event time of
the stay -- the worst score over the whole admission. `clifcs.state` then
as-of-joins that against the grid, which can only match a grid point after the
stay has ended, so `sofa_total` came back 100% null at every offset. The column
was advertised in VARIABLES, silently dropped from the flow panels, and
unavailable as a severity covariate.

Two different quantities were being conflated. The whole-stay worst is a
summary; it is also contaminated by the outcome, since deterioration caused by
the trajectory is inside it. What a baseline covariate needs is the worst score
in a window ENDING at the grid point, which is the usual clinical reading of
"SOFA at time t".

clifpy supports exactly this: `compute_sofa_scores(cohort_df=...)` takes
per-patient [start_time, end_time] windows. Building the wide dataset costs
about a minute per site and each windowed scoring pass about a second, so a
13-point grid is roughly 80 seconds per site rather than anything that needs
its own schedule.
"""

from __future__ import annotations

import time

import polars as pl


def sofa_on_grid(paths, site, offsets, logger, window_hours: float = 24.0):
    """
    worst SOFA in the `window_hours` ending at each grid point, per patient

    Returns a long frame: hospitalization_id, offset_hours, sofa_total and the
    six component scores. A patient with no scoreable data in a window gets no
    row rather than a zero -- clifpy's `fill_na_scores_with_zero` fills missing
    COMPONENTS within a scored window, which is its own documented behaviour,
    but a window with nothing in it at all is absence of measurement and not a
    SOFA of zero.
    """
    import clifpy as cpy
    from omegaconf import OmegaConf

    from clifcs.config import CONFIG_HOME

    units = OmegaConf.load(CONFIG_HOME / "clifpy_units.yaml")
    data_dir = str(paths.raw_subset(site))

    t0 = time.perf_counter()
    orchestrator = cpy.ClifOrchestrator(
        data_directory=data_dir, filetype="parquet", timezone="UTC",
        output_directory=str(paths.log_home / f"sofa-{site.lower()}"),
    )
    orchestrator.convert_dose_units_for_continuous_meds(
        preferred_units=OmegaConf.to_container(units.sofa_continuous),
        override=True,
    )
    # the first call is what builds the wide dataset; every later one reuses it
    orchestrator.compute_sofa_scores(id_name="hospitalization_id")
    wide = orchestrator.wide_df
    logger.info(
        f"[{site}] wide dataset {len(wide):,} rows in "
        f"{time.perf_counter() - t0:.0f}s"
    )

    onset = (
        pl.read_parquet(paths.cohort_home / "cs_onset.parquet")
        .select(
            hospitalization_id=pl.col("hospitalization_id").cast(pl.String),
            onset=pl.col("cs_onset_dttm").cast(pl.Datetime("us")),
        )
        .unique(subset="hospitalization_id")
    )
    known = set(wide["hospitalization_id"].astype(str).unique())
    onset = onset.filter(pl.col("hospitalization_id").is_in(list(known)))
    if onset.is_empty():
        logger.warning(f"[{site}] no onset times match the wide dataset")
        return pl.DataFrame()

    parts = []
    for offset in offsets:
        window = (
            onset.with_columns(
                end_time=pl.col("onset") + pl.duration(hours=float(offset)),
            )
            .with_columns(
                start_time=pl.col("end_time") - pl.duration(hours=window_hours)
            )
            .select("hospitalization_id", "start_time", "end_time")
        )
        cohort_df = window.to_pandas()
        cohort_df["hospitalization_id"] = cohort_df["hospitalization_id"].astype(str)
        scored = orchestrator.compute_sofa_scores(
            wide_df=wide, cohort_df=cohort_df, id_name="hospitalization_id",
            create_new_wide_df=False,
        )
        frame = pl.from_pandas(scored.reset_index(drop=True)).with_columns(
            hospitalization_id=pl.col("hospitalization_id").cast(pl.String),
            offset_hours=pl.lit(float(offset)),
        )
        parts.append(frame)

    out = pl.concat(parts, how="diagonal_relaxed")
    logger.info(
        f"[{site}] SOFA on {len(offsets)} grid points: {out.height:,} rows, "
        f"median {out['sofa_total'].median()}, "
        f"{out['sofa_total'].null_count():,} null "
        f"(total {time.perf_counter() - t0:.0f}s)"
    )
    return out

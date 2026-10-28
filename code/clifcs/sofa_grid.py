#!/usr/bin/env python3

"""
SOFA from clifpy: one wide build, two artifacts

WHY THIS EXISTS. `stage_derive` computes SOFA and collapses it to one row per
encounter stamped at the LAST event time of the stay -- the worst score over
the whole admission. `clifcs.state` then as-of-joins that against the grid,
which can only match a grid point after the stay has ended, so `sofa_total`
came back 100% null at every offset. The column was advertised in VARIABLES,
silently dropped from the flow panels, and unavailable as a severity covariate.

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

ONE WIDE BUILD SERVES BOTH. `build_sofa_wide` is the single place the wide
frame is constructed, which is what makes the whole-stay score and the
windowed scores provably consistent -- they are extremal reductions of the
same rows -- and it is the only place that has to get the loading right.
"""

from __future__ import annotations

import time

import polars as pl

# The CLIF column that holds the category for each table. Taken from the CLIF
# 2.1 schema, which is frozen, rather than from clifpy -- whereas the category
# VALUES below come from clifpy, because those change whenever the SOFA
# definition does.
_CATEGORY_COLUMN = {
    "labs": "lab_category",
    "vitals": "vital_category",
    "patient_assessments": "assessment_category",
    "medication_admin_continuous": "med_category",
}

# Needed by the dose conversion rather than by SOFA itself, and the reason
# vitals has to be loaded here rather than left to clifpy; see `preload`.
_WEIGHT = "weight_kg"

#: the trajectory grid, matching 09_trajectories.py defaults
#: (--start-hours -24, --end-hours 48, --step-hours 6). Thirteen points, and
#: they have to be the same thirteen: `state.py` as-of-joins SOFA against this
#: grid by `offset_hours`, so a point missing here is a null column there.
SOFA_GRID_OFFSETS = tuple(float(h) for h in range(-24, 49, 6))

#: the offset whose window is the baseline severity adjuster. Zero means
#: [onset - 24h, onset]: no post-onset data, the conventional 24 h
#: accumulation period, and ending AT onset rather than before it so the
#: sickest-on-arrival patients -- who are not missing at random -- keep a
#: score. 12_covariate_balance.py and 13_adjusted_mortality.py already define
#: the baseline covariate in exactly those words.
BASELINE_OFFSET = 0.0


def sofa_requirements() -> dict[str, list[str]]:
    """
    which categories of which tables SOFA reads, according to clifpy

    Read from clifpy rather than written out here so that a version bump which
    adds a category cannot silently reintroduce the starvation `preload`
    exists to prevent: an unknown table shows up as a KeyError there instead of
    as a column quietly missing from the wide frame.
    """
    from clifpy.utils.sofa import REQUIRED_SOFA_CATEGORIES_BY_TABLE

    return {table: list(cats) for table, cats in REQUIRED_SOFA_CATEGORIES_BY_TABLE.items()}


def preload(orchestrator, logger, site: str = "") -> None:
    """
    load the SOFA tables with the rows and categories SOFA actually needs

    THE BUG THIS FIXES. `convert_dose_units_for_continuous_meds` loads vitals
    itself, filtered to `vital_category=['weight_kg']`, and leaves the table
    object on the orchestrator. `create_wide_dataset` then skips loading vitals
    -- "load optional tables only if not already loaded" -- so the wide frame
    pivots a vitals table with no `map` and no `spo2` in it. Measured at Rush:
    0 of 3.24M wide rows carried a MAP. With `fill_na_scores_with_zero=True`
    the cardiovascular component then rested on vasopressor dose alone, scoring
    0 for every hypotensive patient not on a pressor at that moment: 78% of
    patients had `sofa_cv_97 > 0` instead of 99%, and mean `sofa_total` came
    out 11.11 instead of 11.32.

    Loading vitals HERE, with the SOFA categories and `weight_kg` together,
    fixes it in one place for both callers -- the conversion filters to
    `weight_kg` itself in SQL, so giving it more categories than it asked for
    costs nothing.

    It is also the largest memory reduction available: `clif_labs` is the
    biggest table at every site and SOFA needs four of its categories.
    """
    prefix = f"[{site}] " if site else ""
    for table, categories in sofa_requirements().items():
        column = _CATEGORY_COLUMN.get(table)
        if column is None:
            # respiratory_support, whose "categories" are columns rather than
            # category values -- nothing to filter on, so clifpy loads it.
            continue
        wanted = sorted(set(categories) | ({_WEIGHT} if table == "vitals" else set()))
        orchestrator.load_table(table, filters={column: wanted})
        rows = len(getattr(orchestrator, table).df)
        logger.info(f"{prefix}  {table}: {rows:,} rows over {len(wanted)} categories")


def build_sofa_wide(paths, site, logger):
    """
    the wide dataset SOFA is scored from, built once

    Returns `(orchestrator, wide)`. The orchestrator is returned because the
    scoring calls hang off it and because `wide` alone cannot be re-scored.
    """
    import clifpy as cpy
    from omegaconf import OmegaConf

    from clifcs.config import CONFIG_HOME

    units = OmegaConf.load(CONFIG_HOME / "clifpy_units.yaml")

    t0 = time.perf_counter()
    orchestrator = cpy.ClifOrchestrator(
        data_directory=str(paths.raw_subset(site)),
        filetype="parquet",
        timezone="UTC",
        output_directory=str(paths.log_home / f"clifpy-{site.lower()}"),
    )
    preload(orchestrator, logger, site)
    orchestrator.convert_dose_units_for_continuous_meds(
        preferred_units=OmegaConf.to_container(units.sofa_continuous),
        override=True,
    )
    # Built directly rather than as a side effect of a throwaway scoring call,
    # which is how `compute_sofa_scores` does it internally -- the scoring pass
    # that call would also run is an extremal reduction over millions of rows,
    # and neither caller wants its answer.
    requirements = sofa_requirements()
    wide = orchestrator.create_wide_dataset(
        tables_to_load=list(requirements),
        category_filters=requirements,
        return_dataframe=True,
    )
    wide = orchestrator.wide_df if wide is None else wide
    charted = (
        float(wide["map"].notna().mean()) if "map" in wide.columns else 0.0
    )
    logger.info(
        f"[{site}] wide dataset {len(wide):,} rows in "
        f"{time.perf_counter() - t0:.0f}s; MAP charted on {charted:.1%} of rows"
    )
    if charted == 0.0:
        # The direct evidence for the bug above, checked at every site rather
        # than trusted: a wide frame with no MAP in it scores the
        # cardiovascular component from vasopressors alone.
        logger.warning(
            f"[{site}] no MAP in the wide dataset -- the cardiovascular SOFA "
            "component will rest on vasopressor dose alone. Check that the "
            "vitals preload ran before the dose conversion."
        )
    return orchestrator, wide


# The whole-stay collapse lives in `01_site_tables.sofa_by_hospitalization`,
# not here. It is a pure function of two frames -- the stamped wide frame and
# clifpy's score table -- so it needs no orchestrator and is tested directly;
# `build_sofa_wide` below supplies the frames and `_derive_sofa` joins them.
#
# That split matters because of what it fixed. `compute_sofa_scores` returns
# `encounter_block` as a COLUMN under a fresh RangeIndex, while grouping the
# wide frame leaves it as the INDEX with the real block ids, and
# `DataFrame.join(other, on=...)` matches the caller's key against the OTHER
# FRAME'S INDEX. Every encounter therefore carried its neighbour's score and
# the highest block was dropped, with `validate="1:1"` satisfied throughout.


def sofa_on_grid(
    paths, site, offsets, logger, window_hours: float = 24.0, wide=None,
    orchestrator=None,
):
    """
    worst SOFA in the `window_hours` ending at each grid point, per patient

    Returns a long frame: hospitalization_id, offset_hours, sofa_total and the
    six component scores. A patient with no scoreable data in a window gets no
    row rather than a zero -- clifpy's `fill_na_scores_with_zero` fills missing
    COMPONENTS within a scored window, which is its own documented behaviour,
    but a window with nothing in it at all is absence of measurement and not a
    SOFA of zero.

    `wide` and `orchestrator` may be passed in by a caller that has already
    built them, which is what makes this cost seconds rather than minutes when
    it runs beside the whole-stay score.
    """
    t0 = time.perf_counter()
    if wide is None or orchestrator is None:
        orchestrator, wide = build_sofa_wide(paths, site, logger)

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

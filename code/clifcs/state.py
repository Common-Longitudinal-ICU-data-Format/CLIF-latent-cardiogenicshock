#!/usr/bin/env python3

"""
physiological state at each point on a trajectory grid

Gives every (patient, timepoint) the clinical picture as it stood then: the
last value recorded at or before that moment, never after. That makes the
variables usable as an underlay for the flow field, where the question is what
is changing as patients move through the latent space.

Everything is a backward as-of join against the grid, so a value persists
until it is superseded -- which is how a clinician reads a chart, and how the
model saw it too.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from clifcs.clif import has_table, scan
from clifcs.labels import DEFAULT_WEIGHT_KG, norepinephrine_equivalents

#: vitals carried forward onto the grid, with the range each is kept to.
#: Needed because the raw tables carry artefacts a colour scale cannot survive
#: -- MAP reached 1,040 mmHg before this filter, which put every real patient
#: into one bin of the ramp.
_VITALS = {
    "map": (20.0, 200.0),
    "sbp": (20.0, 300.0),
    "heart_rate": (10.0, 300.0),
    "spo2": (30.0, 100.0),
    "respiratory_rate": (2.0, 80.0),
}

#: labs carried forward onto the grid, likewise bounded
_LABS = {
    "lactate": (0.1, 30.0),
    "creatinine": (0.1, 25.0),
    "bicarbonate": (2.0, 60.0),
    "bun": (1.0, 250.0),
    "platelet_count": (1.0, 2000.0),
}

#: how each variable is read clinically, for axis labels and colour direction.
#: `worse_is_high` drives nothing automatically -- it is recorded so a reader
#: of the flow figures knows which way deterioration points.
VARIABLES = {
    "map": ("MAP (mmHg)", False),
    "shock_index": ("shock index (HR / SBP)", True),
    "heart_rate": ("heart rate (bpm)", True),
    "nee": ("norepinephrine equivalents (mcg/kg/min)", True),
    "n_vasopressors": ("vasoactive agents running", True),
    "lactate": ("lactate (mmol/L)", True),
    "bicarbonate": ("bicarbonate (mmol/L)", False),
    "creatinine": ("creatinine (mg/dL)", True),
    "gcs_total": ("GCS total", False),
    "sofa_total": ("SOFA total", True),
    "on_imv": ("invasive ventilation", True),
}


def _asof(grid: pl.DataFrame, events: pl.LazyFrame, time_col: str, value_cols) -> pl.DataFrame:
    """
    last observation at or before each grid point, per patient

    join_asof needs both sides sorted on the join key; polars will not sort
    them silently, and an unsorted input gives wrong answers rather than an
    error, so it is done explicitly here.
    """
    events = events.sort(time_col).collect()
    if events.height == 0:
        return grid.select("hospitalization_id", "offset_hours").with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in value_cols]
        )
    return (
        grid.sort("grid_dttm")
        .join_asof(
            events,
            left_on="grid_dttm",
            right_on=time_col,
            by="hospitalization_id",
            strategy="backward",
        )
        .select("hospitalization_id", "offset_hours", *value_cols)
    )


def trajectory_state(data_dir, grid: pl.DataFrame) -> pl.DataFrame:
    """
    one row per (patient, grid point) with the physiology as it stood

    `grid` must carry hospitalization_id, offset_hours and grid_dttm.
    """
    keys = grid.select("hospitalization_id", "offset_hours")
    out = keys

    vitals = (
        scan(data_dir, "clif_vitals", dttm_cols=("recorded_dttm",))
        .filter(pl.col("vital_category").is_in(list(_VITALS)),
                pl.col("vital_value").is_not_null())
        .join(grid.lazy().select("hospitalization_id").unique(),
              on="hospitalization_id", how="inner")
    )
    for name, (low, high) in _VITALS.items():
        part = vitals.filter(
            pl.col("vital_category") == name,
            pl.col("vital_value").is_between(low, high),
        ).select(
            "hospitalization_id", "recorded_dttm", **{name: pl.col("vital_value")}
        )
        out = out.join(
            _asof(grid, part, "recorded_dttm", [name]),
            on=["hospitalization_id", "offset_hours"], how="left",
        )

    labs = (
        scan(data_dir, "clif_labs", dttm_cols=("lab_result_dttm",))
        .filter(pl.col("lab_category").is_in(list(_LABS)),
                pl.col("lab_value_numeric").is_not_null())
        .join(grid.lazy().select("hospitalization_id").unique(),
              on="hospitalization_id", how="inner")
    )
    for name, (low, high) in _LABS.items():
        part = labs.filter(
            pl.col("lab_category") == name,
            pl.col("lab_value_numeric").is_between(low, high),
        ).select(
            "hospitalization_id", "lab_result_dttm",
            **{name: pl.col("lab_value_numeric")},
        )
        out = out.join(
            _asof(grid, part, "lab_result_dttm", [name]),
            on=["hospitalization_id", "offset_hours"], how="left",
        )

    gcs = (
        scan(data_dir, "clif_patient_assessments", dttm_cols=("recorded_dttm",))
        .filter(pl.col("assessment_category") == "gcs_total",
                pl.col("numerical_value").is_between(3, 15))
        .join(grid.lazy().select("hospitalization_id").unique(),
              on="hospitalization_id", how="inner")
        .select("hospitalization_id", "recorded_dttm",
                gcs_total=pl.col("numerical_value"))
    )
    out = out.join(_asof(grid, gcs, "recorded_dttm", ["gcs_total"]),
                   on=["hospitalization_id", "offset_hours"], how="left")

    if has_table(data_dir, "clif_sofa"):
        sofa = (
            scan(data_dir, "clif_sofa", dttm_cols=("event_time",))
            .join(grid.lazy().select("hospitalization_id").unique(),
                  on="hospitalization_id", how="inner")
            .select("hospitalization_id", "event_time",
                    sofa_total=pl.col("sofa_total").cast(pl.Float64))
        )
        out = out.join(_asof(grid, sofa, "event_time", ["sofa_total"]),
                       on=["hospitalization_id", "offset_hours"], how="left")

    # invasive ventilation: a state, not a measurement, so carried forward from
    # the last respiratory-support row of any kind rather than only IMV rows --
    # otherwise a patient extubated hours ago still reads as ventilated
    if has_table(data_dir, "clif_respiratory_support_processed"):
        resp = (
            scan(data_dir, "clif_respiratory_support_processed",
                 dttm_cols=("recorded_dttm",))
            .join(grid.lazy().select("hospitalization_id").unique(),
                  on="hospitalization_id", how="inner")
            .filter(pl.col("device_category").is_not_null())
            .select(
                "hospitalization_id", "recorded_dttm",
                on_imv=(pl.col("device_category").str.to_lowercase() == "imv").cast(
                    pl.Float64
                ),
            )
        )
        out = out.join(_asof(grid, resp, "recorded_dttm", ["on_imv"]),
                       on=["hospitalization_id", "offset_hours"], how="left")

    # vasoactive burden: total norepinephrine-equivalent rate across every
    # agent running at that moment
    doses = norepinephrine_equivalents(data_dir, grid.select("hospitalization_id").unique())
    if doses.height:
        # one as-of join per agent, then summed: a single join keyed only on
        # the patient would return whichever agent happened to be charted last
        # rather than everything running concurrently
        agents = doses["med_category"].unique().to_list()
        totals = keys.with_columns(
            nee=pl.lit(0.0), n_vasopressors=pl.lit(0, dtype=pl.Int64)
        )
        for agent in agents:
            part = doses.filter(pl.col("med_category") == agent).select(
                "hospitalization_id", "admin_dttm", **{f"nee_{agent}": pl.col("nee")}
            )
            joined = _asof(grid, part.lazy(), "admin_dttm", [f"nee_{agent}"])
            totals = totals.join(
                joined, on=["hospitalization_id", "offset_hours"], how="left"
            ).with_columns(
                nee=pl.col("nee") + pl.col(f"nee_{agent}").fill_null(0.0),
                n_vasopressors=pl.col("n_vasopressors")
                + (pl.col(f"nee_{agent}").fill_null(0.0) > 0).cast(pl.Int64),
            ).drop(f"nee_{agent}")
        out = out.join(totals, on=["hospitalization_id", "offset_hours"], how="left")
    else:
        out = out.with_columns(nee=pl.lit(0.0), n_vasopressors=pl.lit(0))

    return out.with_columns(
        shock_index=pl.when(pl.col("sbp") > 0)
        .then(pl.col("heart_rate") / pl.col("sbp"))
        .otherwise(None)
    )


def gradient_alignment(
    px, py, vx, vy, values, bins: int = 22, min_count: int = 5
) -> tuple[float, int]:
    """
    does the flow point uphill or downhill in this variable?

    Bins the plane, takes the mean displacement and the mean value per cell,
    estimates the spatial gradient of the value field, and returns the mean
    cosine between displacement and gradient. +1 means patients move toward
    higher values of the variable, -1 toward lower, 0 means the flow is
    unrelated to it.

    This is the quantitative form of "what changes as patients move": a figure
    shows the colours and the arrows, this says whether they agree.
    """
    finite = np.isfinite(values) & np.isfinite(px) & np.isfinite(py)
    px, py, vx, vy, values = (a[finite] for a in (px, py, vx, vy, values))
    if len(px) < 50:
        return float("nan"), 0

    edges_x = np.linspace(*np.percentile(px, [1, 99]), bins + 1)
    edges_y = np.linspace(*np.percentile(py, [1, 99]), bins + 1)
    counts, _, _ = np.histogram2d(px, py, [edges_x, edges_y])
    sum_v, _, _ = np.histogram2d(px, py, [edges_x, edges_y], weights=values)
    sum_x, _, _ = np.histogram2d(px, py, [edges_x, edges_y], weights=vx)
    sum_y, _, _ = np.histogram2d(px, py, [edges_x, edges_y], weights=vy)

    ok = counts >= min_count
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_v = np.where(ok, sum_v / counts, np.nan)
        mean_x = np.where(ok, sum_x / counts, np.nan)
        mean_y = np.where(ok, sum_y / counts, np.nan)

    grad_x, grad_y = np.gradient(np.nan_to_num(mean_v, nan=np.nanmean(mean_v)))
    valid = ok & np.isfinite(mean_x) & np.isfinite(mean_y)
    gx, gy = grad_x[valid], grad_y[valid]
    dx, dy = mean_x[valid], mean_y[valid]

    norm = np.hypot(gx, gy) * np.hypot(dx, dy)
    usable = norm > 1e-12
    if usable.sum() < 10:
        return float("nan"), int(usable.sum())
    cosine = (gx[usable] * dx[usable] + gy[usable] * dy[usable]) / norm[usable]
    return float(np.mean(cosine)), int(usable.sum())


def position_encodes(px, py, values, seed: int = 42) -> float:
    """
    how much of a variable the position in the plane accounts for

    Out-of-sample R^2 of a 30-nearest-neighbour regression predicting the
    variable from the 2-d coordinates, fitted on half the points and scored on
    the other half. This is a different question from gradient alignment and
    the two can disagree completely: a space can be strongly stratified by MAP
    -- low-MAP patients here, high-MAP patients there -- while patients
    nonetheless fail to travel along that gradient over six hours. The first
    is about where patients sit, the second about where they go.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor

    finite = np.isfinite(values) & np.isfinite(px) & np.isfinite(py)
    if finite.sum() < 500:
        return float("nan")
    coords = np.column_stack([px[finite], py[finite]])
    target = values[finite]
    if np.std(target) < 1e-9:
        return float("nan")
    a, b, ya, yb = train_test_split(coords, target, test_size=0.5, random_state=seed)
    model = KNeighborsRegressor(n_neighbors=30, n_jobs=-1).fit(a, ya)
    return float(model.score(b, yb))

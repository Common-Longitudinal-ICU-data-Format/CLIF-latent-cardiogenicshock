"""
baseline characteristics by coded shock type

The table a reader wants before any latent-space claim: who are these patients,
split by the mechanism the coders recorded. Median and interquartile range for
the continuous variables, count and share for the binary ones.

Everything is measured at cs_onset_dttm rather than at admission or over the
whole stay. The shock types are being compared on how they presented, and a
median over an entire admission answers a different question -- a patient who
spends a week in the ICU contributes a week of readings to it.

WHAT TRAVELS. This is an aggregate: a median is a statistic over values, not a
value, which is the same argument trial_aggregate makes. Cells below min_cell
patients are suppressed, so a shock type with few patients at a site reports
nothing rather than reporting one person physiology.
"""

from __future__ import annotations

import polars as pl

from clifcs.clif import scan
from clifcs.state import trajectory_state

#: continuous variables: median [Q1-Q3], with the decimals to round to.
#:
#: nee is norepinephrine equivalents, the only way to compare vasopressor
#: intensity across agents -- a dose of phenylephrine and a dose of
#: norepinephrine are not the same quantity, and reporting "on a pressor" for
#: both discards exactly the difference a cardiogenic/septic comparison is
#: about.
CONTINUOUS: tuple[tuple[str, str, int], ...] = (
    ("age_at_admission", "Age (years)", 0),
    ("map", "MAP at onset (mmHg)", 0),
    ("sbp", "Systolic at onset (mmHg)", 0),
    ("heart_rate", "Heart rate at onset (bpm)", 0),
    ("lactate", "Lactate (mmol/L)", 1),
    ("creatinine", "Creatinine (mg/dL)", 2),
    ("platelet_count", "Platelets (x10^9/L)", 0),
    ("bicarbonate", "Bicarbonate (mmol/L)", 0),
    ("sofa_total", "SOFA total", 0),
    ("sofa_non_cv", "SOFA non-cardiovascular", 0),
    ("nee", "Norepinephrine equivalents (mcg/kg/min)", 3),
    ("n_vasopressors", "Vasoactive agents running", 0),
    ("icu_los_days", "ICU length of stay (days)", 1),
)

#: binary variables: n (share).
BINARY: tuple[tuple[str, str], ...] = (
    ("on_imv", "Invasive mechanical ventilation"),
    ("on_rrt", "Renal replacement therapy"),
    ("cardiac_inotrope", "Inotrope (dobutamine or milrinone)"),
    ("cardiac_mcs", "Mechanical circulatory support (ECMO, IABP, pVAD)"),
    ("cardiac_troponin", "Peak troponin above threshold"),
    ("in_hospital_mortality", "Died in hospital"),
)


def _rrt_flag(data_dir, ids: pl.Series) -> pl.DataFrame:
    """
    whether the patient ever received renal replacement therapy

    clif_crrt_therapy is read for presence only, over the whole admission
    rather than at onset: CRRT is started as a consequence of shock far more
    often than before it, so an at-onset flag would read as "almost nobody",
    and the question here is which shock types end up needing it.

    A site without the table reports null rather than false. "No row" and "no
    table" are different claims, and a false here would assert that a site
    cardiogenic patients never needed dialysis.
    """
    keys = ids.cast(pl.String)
    try:
        frame = (
            scan(data_dir, "clif_crrt_therapy")
            .select("hospitalization_id")
            .unique()
            .collect()
        )
    except Exception:
        return pl.DataFrame({
            "hospitalization_id": keys,
            "on_rrt": pl.Series([None] * len(keys), dtype=pl.Boolean),
        })
    seen = set(frame["hospitalization_id"].cast(pl.String).to_list())
    return pl.DataFrame({
        "hospitalization_id": keys,
        "on_rrt": [k in seen for k in keys.to_list()],
    })


def collect(data_dir, cohort: pl.DataFrame, logger) -> pl.DataFrame:
    """
    per-patient baseline values at shock onset, for one site

    The physiology comes from clifcs.state.trajectory_state with a one-point
    grid at onset, which is the same code 09 uses for its 6-hourly grid.
    Reusing it is deliberate: a second implementation of "MAP at time t" would
    drift from the one the trajectory figures are drawn with, and the two would
    disagree in a way nobody would think to check.
    """
    grid = cohort.select(
        "hospitalization_id",
        pl.lit(0.0).alias("offset_hours"),
        pl.col("cs_onset_dttm").alias("grid_dttm"),
    )
    state = trajectory_state(data_dir, grid).drop("offset_hours")
    logger.info(f"  baseline: physiology at onset for {state.height} patients")

    return cohort.join(state, on="hospitalization_id", how="left").join(
        _rrt_flag(data_dir, cohort["hospitalization_id"]),
        on="hospitalization_id",
        how="left",
    )


def summarise(frame: pl.DataFrame, site: str, min_cell: int) -> pl.DataFrame:
    """
    median [IQR] and n (%) per coded shock type, long-form and suppressed

    Long rather than wide -- one row per (variable, shock type) -- because a
    wide table has to be rebuilt whenever a shock type is added or a site
    reports a different set, and the report renders either shape equally well.
    """
    rows: list[dict] = []
    groups = [
        g for g in frame["shock_type"].unique().sort().to_list() if g is not None
    ]
    for group in groups:
        part = frame.filter(pl.col("shock_type") == group)
        n_group = part.height
        suppressed = 0 < n_group < min_cell

        for key, label, places in CONTINUOUS:
            if key not in part.columns:
                continue
            values = part[key].cast(pl.Float64, strict=False).drop_nulls()
            ok = (not suppressed) and len(values) >= min_cell
            rows.append({
                "site": site,
                "shock_type": group,
                "variable": label,
                "kind": "continuous",
                "n": n_group if not suppressed else None,
                "n_measured": len(values) if ok else None,
                "median": round(values.median(), places) if ok else None,
                "q1": round(values.quantile(0.25), places) if ok else None,
                "q3": round(values.quantile(0.75), places) if ok else None,
                "share": None,
            })

        for key, label in BINARY:
            if key not in part.columns:
                continue
            measured = part[key].cast(pl.Boolean, strict=False).drop_nulls()
            positives = int(measured.sum()) if len(measured) else 0
            # the count is suppressed when it names few patients, and so is the
            # share, which would otherwise hand it straight back
            ok = (
                not suppressed
                and len(measured) >= min_cell
                and not (0 < positives < min_cell)
            )
            rows.append({
                "site": site,
                "shock_type": group,
                "variable": label,
                "kind": "binary",
                "n": n_group if not suppressed else None,
                "n_measured": positives if ok else None,
                "median": None,
                "q1": None,
                "q3": None,
                "share": round(positives / len(measured), 4) if ok else None,
            })

    return pl.DataFrame(rows) if rows else pl.DataFrame()

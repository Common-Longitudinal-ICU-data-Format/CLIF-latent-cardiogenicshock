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

#: binary variables: key, label, and whether a null means "no".
#:
#: THE THIRD FIELD MATTERS AND IS NOT COSMETIC. The cardiac_* arms are stored
#: as True-or-null, never False: a patient who got no inotrope has a null, not
#: a false. Dropping nulls before taking a share therefore collapses the
#: denominator onto the positives and every one of those rows reads 100%, which
#: is what the first version of this table published for all six shock types.
#:
#: `on_rrt` is the opposite case and must NOT be filled: a null there means the
#: site has no clif_crrt_therapy table, and turning that into "no dialysis"
#: would assert something about the patients from the absence of a file.
#: `on_imv` likewise -- a null is no respiratory record at onset.
BINARY: tuple[tuple[str, str, bool], ...] = (
    ("on_imv", "Invasive mechanical ventilation", False),
    ("on_rrt", "Renal replacement therapy", False),
    ("cardiac_inotrope", "Inotrope (dobutamine or milrinone)", True),
    ("cardiac_mcs", "Mechanical circulatory support (ECMO, IABP, pVAD)", True),
    ("cardiac_troponin", "Peak troponin above threshold", True),
    ("in_hospital_mortality", "Died in hospital", True),
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


def _sofa(data_dir, ids: pl.Series) -> pl.DataFrame:
    """
    SOFA total and the non-cardiovascular part, from the derived clif_sofa

    One row per patient: clif_sofa is per encounter block, so the first is
    taken. The non-cardiovascular total is the sum of the components other than
    the cardiovascular one -- the cardiovascular component is largely a
    restatement of the vasopressor dose, which is what defines shock here, so
    adjusting on the full score adjusts partly on the exposure.
    """
    from clifcs.concordance import NON_CV_SOFA_COMPONENTS

    try:
        frame = scan(data_dir, "clif_sofa").collect()
    except Exception:
        return pl.DataFrame()
    have = [c for c in NON_CV_SOFA_COMPONENTS if c in frame.columns]
    if "hospitalization_id" not in frame.columns or not have:
        return pl.DataFrame()
    keep = ["hospitalization_id", *(["sofa_total"] if "sofa_total" in frame.columns else []), *have]
    out = (
        frame.select(keep)
        .with_columns(pl.col("hospitalization_id").cast(pl.String))
        .unique(subset=["hospitalization_id"], keep="first")
        .with_columns(
            sofa_non_cv=pl.sum_horizontal(
                *(pl.col(c).cast(pl.Int32) for c in have)
            )
        )
        .drop([c for c in have if c != "sofa_non_cv"])
    )
    return out.filter(pl.col("hospitalization_id").is_in(ids.cast(pl.String).implode()))


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

    # SOFA separately. trajectory_state lists sofa_total in its VARIABLES but
    # never populates it -- the grid frame carries no sofa column either, which
    # is also why 09 draws no sofa panel. It comes from clif_sofa, the parquet
    # the derive stage writes, which is where attach_severity reads it.
    sofa = _sofa(data_dir, cohort["hospitalization_id"])
    if sofa.height:
        # Drop whatever trajectory_state already called sofa_* before joining.
        # It emits sofa_total as part of its variable set and never fills it,
        # so a plain join put the real column in sofa_total_right and this
        # table read the empty one -- a blank row that looked like "this site
        # does not compute SOFA" rather than a name collision.
        collision = [c for c in state.columns if c.startswith("sofa_")]
        if collision:
            logger.info(
                f"  baseline: dropping {collision} from the state frame; "
                "clif_sofa is the populated source"
            )
            state = state.drop(collision)
        state = state.join(sofa, on="hospitalization_id", how="left")

    return _join_state(cohort, state).join(
        _rrt_flag(data_dir, cohort["hospitalization_id"]),
        on="hospitalization_id",
        how="left",
    )


def _join_state(cohort: pl.DataFrame, state: pl.DataFrame) -> pl.DataFrame:
    """
    the cohort with the at-onset state attached, the state winning shared names

    cohort.parquet already carries lactate, creatinine, bicarbonate, bun and
    platelet_count -- the nearest value in the phenotype window, -24 h to +6 h.
    A plain join kept those and parked the state's at-onset values in
    `*_right`, so the table read a different window from the one documented
    in `collect`, with numbers plausible enough that nothing flagged it.
    """
    shared = [c for c in state.columns if c in cohort.columns and c != "hospitalization_id"]
    return cohort.drop(shared).join(state, on="hospitalization_id", how="left")


def summarise(frame: pl.DataFrame, site: str) -> pl.DataFrame:
    """
    median [IQR] and n (%) per coded shock type, long-form

    Long rather than wide -- one row per (variable, shock type) -- because a
    wide table has to be rebuilt whenever a shock type is added or a site
    reports a different set, and the report renders either shape equally well.

    EMITS TRUE VALUES. Suppression is applied afterwards by
    `clifcs.export.suppress.suppress_counts`, which is the only thing that
    decides what may be published. This function used to implement its own
    policy inline, and the policy it implemented was incomplete in a way that
    is hard to see: on a binary row it put `positives` -- the NUMERATOR -- in
    `n_measured`, left the real denominator unexported, and checked only that
    the positive arm was not small. The negative arm, `len(measured) -
    positives`, was never checked, and `n_measured / share` gave the
    denominator back anyway.

    So binary rows now carry `n_measured` (the denominator, as the name says)
    and `n_positive` (the numerator), and the proportion is `share_positive`
    rather than `share`. The rename matters: `share` means "over `n_total`"
    everywhere else in the bundle, and the registry permits a column name only
    one denominator.
    """
    rows: list[dict] = []
    groups = [
        g for g in frame["shock_type"].unique().sort().to_list() if g is not None
    ]
    for group in groups:
        part = frame.filter(pl.col("shock_type") == group)
        n_group = part.height

        for key, label, places in CONTINUOUS:
            if key not in part.columns:
                continue
            values = part[key].cast(pl.Float64, strict=False).drop_nulls()
            measured = len(values)
            rows.append({
                "site": site,
                "shock_type": group,
                "variable": label,
                "kind": "continuous",
                "n": n_group,
                "n_measured": measured,
                "n_positive": None,
                "median": round(values.median(), places) if measured else None,
                "q1": round(values.quantile(0.25), places) if measured else None,
                "q3": round(values.quantile(0.75), places) if measured else None,
                "share_positive": None,
            })

        for key, label, null_is_false in BINARY:
            if key not in part.columns:
                continue
            series = part[key].cast(pl.Boolean, strict=False)
            if null_is_false:
                series = series.fill_null(False)
            measured = series.drop_nulls()
            positives = int(measured.sum()) if len(measured) else 0
            rows.append({
                "site": site,
                "shock_type": group,
                "variable": label,
                "kind": "binary",
                "n": n_group,
                "n_measured": len(measured),
                "n_positive": positives,
                "median": None,
                "q1": None,
                "q3": None,
                # NOT rounded. `publishable` recovers the numerator as
                # share * n_measured, and a value rounded to four places lands
                # between integers at these denominators, which turns the
                # numerator test into a guess.
                "share_positive": (
                    positives / len(measured) if len(measured) else None
                ),
            })

    return pl.DataFrame(rows) if rows else pl.DataFrame()

#!/usr/bin/env python3

"""
12 -- does fitting the trial's population track with doing better?

11 asks which trials would have enrolled each patient and how far outside the
enrolled population they sit. This asks what that distance is worth: whether
mortality, ventilator-free days and vasopressor-free days move with it, and
whether they still move once severity is accounted for.

THE PROBLEM THIS SCRIPT CANNOT SOLVE, STATED FIRST

A patient sits outside a trial's published interquartile ranges largely
because they are older, more acidotic, more hypotensive or on more support
than the people it enrolled. Those are also the things that kill them. So the
exposure and the confounder are built from overlapping measurements, and in
several trials they are built from the SAME measurement -- adjusting for age
when age is one of the variables the fit score is computed from removes part
of the exposure along with the confounding.

That overlap is not hidden here. `trial_fit_adjustment_overlap.csv` lists,
per trial, exactly which fit-score variables also appear in the adjustment
set, and the adjusted column of every results table should be read against
it. Where the overlap is total the adjusted estimate is close to meaningless
and is marked so rather than dropped, because dropping it would leave the
unadjusted estimate looking unopposed.

WHAT IS FITTED

For each trial with a usable published baseline, among its eligible patients:

  mortality            logistic, in-hospital death
  vent_free_days       least squares, ventilator-free days to day 28
  pressor_free_days    least squares, vasopressor-free days to day 28

against two different readings of the same gap:

  absdist        how FAR outside the published ranges, unsigned
  severity_gap   WHICH WAY, signed so positive is sicker or older than the
                 trial's median

each unadjusted and adjusted for non-cardiovascular SOFA, age and site, and
each with the score entered as a restricted cubic spline so a non-monotone
relation can show itself. A linear model is fitted alongside and the two are
compared by likelihood ratio, which is the nonlinearity test reported.

The severity_gap models are the ones that answer the question the binary
inside/outside could not: a patient healthier than the trial's population and
a patient sicker than it are both "outside", and if only one of them does
badly then the distance score was measuring severity all along.

SECONDARY -- does concordant care pay off differently by fit?

For each trial whose recommended treatment is ascertainable, the same outcome
is regressed on fit score, concordance and their interaction. CONFOUNDING BY
INDICATION IS SEVERE HERE and is not adjustable away: a patient gets the
recommended treatment partly because a clinician judged they would benefit,
and the negative device trials invert it -- "concordant" there means the
device was withheld, which happens most for patients too well or too sick to
cannulate. Every interaction row carries a `confounding` flag naming which of
the two patterns applies.

Least squares is used for the free-day outcomes because a trial reports a
mean or median difference in days and that is the quantity people compare
against. It is a poor distributional fit: both outcomes pile up at 0 and at
28. The fitted values are conditional means and should be read as such, not
as predictions for an individual.

Run after 11. Usage:

    python code/12_trial_fit_outcomes.py
    python code/12_trial_fit_outcomes.py --anchor 24h
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from omegaconf import OmegaConf
from patsy import dmatrix

from clifcs import ANCHORS, get_logger, load_paths
from clifcs.figures import save_figure, style
from clifcs.concordance import concordance_expressions
from clifcs.freedays import HORIZON_DAYS, add_free_days
from clifcs.palette import CATEGORICAL, CONTEXT, GRID, INK_MUTED, INK_SECONDARY
from clifcs.trials import TABLE1_LABELS, load_trials, score_variables

#: degrees of freedom for the restricted cubic spline on the fit score.
#: Four places three interior knots at quantiles of the score, which is the
#: usual default and is about as much flexibility as this many patients will
#: support for the smaller trials.
SPLINE_DF = 4

#: the severity adjustment. Non-cardiovascular SOFA rather than total,
#: following clifcs.concordance: the cardiovascular component is the shock
#: itself, and adjusting for it would adjust away the thing being studied.
ADJUSTERS = ("sofa_non_cv", "age_at_admission")

#: the cohort columns the adjustment set is built from, for the overlap
#: report. `sofa_non_cv` is listed against `sofa_total` because a trial that
#: publishes a SOFA is publishing the total, and the two are not independent.
ADJUSTER_SOURCES = {
    "sofa_non_cv": "sofa_total",
    "age_at_admission": "age_at_admission",
}

#: below this many eligible patients with a complete record, no model is
#: fitted. The spline alone costs four parameters before any adjuster.
MIN_ROWS = 200

#: and below this many deaths, no logistic model is fitted -- roughly ten
#: events per parameter for the adjusted spline specification
MIN_EVENTS = 70

OUTCOMES = (
    ("in_hospital_mortality", "in-hospital mortality", "logistic"),
    ("vent_free_days", f"ventilator-free days to {HORIZON_DAYS}", "ols"),
    ("pressor_free_days", f"vasopressor-free days to {HORIZON_DAYS}", "ols"),
)

SCORES = (
    ("absdist", "distance outside the published IQRs (IQR units)"),
    ("severity_gap", "severity gap: + is sicker/older than the trial"),
)


# --------------------------------------------------------------------- data


def load_alignment(paths, anchor: str, logger) -> pl.DataFrame:
    """the per-patient output of 11, which carries the fit scores"""
    path = paths.patient_level_home / f"patient_trial_alignment_{anchor}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run 11_trial_alignment.py --anchor {anchor} first."
        )
    frame = pl.read_parquet(path)
    logger.info(f"[{anchor}] {frame.height:,} patients from {path.name}")
    return frame


def attach_free_days(paths, frame: pl.DataFrame, logger) -> pl.DataFrame:
    """
    join ventilator-free and vasopressor-free days, derived per site

    The source tables live in the per-site scratch subset rather than in the
    site's own CLIF directory, because they are the `_processed` and
    `_converted` variants the earlier stages wrote -- the same place
    clifcs.concordance.attach_severity reads clif_sofa from.
    """
    cohort = pl.read_parquet(
        paths.cohort_home / "cohort.parquet",
        columns=[
            "hospitalization_id", "site", "cs_onset_dttm", "discharge_dttm",
            "in_hospital_mortality",
        ],
    )
    parts = []
    for site in paths.sites:
        part = cohort.filter(pl.col("site") == site)
        if part.height == 0:
            continue
        logger.info(f"[{site}] deriving free days")
        parts.append(add_free_days(paths.raw_subset(site), part, logger))
    if not parts:
        raise RuntimeError("no site produced free days")

    free = pl.concat(parts, how="diagonal_relaxed").select(
        "hospitalization_id", "site", "vent_days", "vent_free_days",
        "pressor_days", "pressor_free_days", "died_within_horizon",
        "free_days_ascertained",
    )
    return frame.join(free, on=["hospitalization_id", "site"], how="left")


# ------------------------------------------------------------------ fitting


def _winsorise(values: np.ndarray, lower: float = 0.01, upper: float = 0.99):
    """
    clip a score to its 1st and 99th percentile before splining it

    Not cosmetic. patsy places the spline's interior knots at quantiles but
    the boundary knots at the extremes, so a single patient fifteen IQRs out
    stretches the last basis function across a region holding nobody and the
    fitted curve there is drawn by one observation.
    """
    if values.size == 0:
        return values, (np.nan, np.nan)
    lo, hi = np.nanquantile(values, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return values, (lo, hi)
    return np.clip(values, lo, hi), (lo, hi)


def _design(score: np.ndarray, df: int):
    """
    spline basis for the score, and the state needed to predict from it

    `constraints="center"` is load-bearing, not a refinement. An unconstrained
    `cr()` basis is a partition of unity -- its columns sum to 1 at every x --
    so stacking it against an intercept column gives a design of rank k where
    k + 1 was intended. statsmodels fits that through a pseudo-inverse without
    complaining, returns coefficients that are one of infinitely many
    solutions, and then produces negative prediction variances; the symptom is
    a run full of "invalid value encountered in sqrt" and confidence bands
    that are silently NaN. Centring the basis against the intercept makes the
    design full rank and the standard errors real.
    """
    return dmatrix(
        f"cr(x, df={df}, constraints='center') - 1",
        {"x": score},
        return_type="dataframe",
    )


def _fit(kind: str, y: np.ndarray, X: np.ndarray):
    """one model, with the convergence noise suppressed and reported instead"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "logistic":
            model = sm.GLM(y, X, family=sm.families.Binomial())
        else:
            model = sm.GLM(y, X, family=sm.families.Gaussian())
        try:
            return model.fit()
        except Exception:  # noqa: BLE001 -- separation, singularity, anything
            return None


def _curve_shape(x: np.ndarray, y: np.ndarray, tolerance: float = 0.05) -> str:
    """
    classify a fitted curve as monotone or turning, and which way

    The point of splining the score was to let a non-monotone relation show
    itself, so whether one did is a result and not a thing to leave to the
    reader's eye. The turning point has to be genuinely interior -- inside the
    middle 80% of the range -- and the turn has to be worth at least
    `tolerance` of the curve's total span, so that a flat curve with a
    numerically lowest point somewhere in the middle is not called U-shaped.

    A U in mortality against the severity gap is the interesting case: it says
    patients HEALTHIER than the trial on the measured axes do worse than those
    sitting at its median, which no inside/outside split could show and which
    usually means the score is not measuring what its name suggests for that
    part of the range.
    """
    if len(x) < 5:
        return "unknown"
    span = float(np.nanmax(y) - np.nanmin(y))
    if span <= 0 or not np.isfinite(span):
        return "flat"
    lo, hi = np.nanquantile(x, [0.1, 0.9])
    interior = (x >= lo) & (x <= hi)
    argmin, argmax = int(np.nanargmin(y)), int(np.nanargmax(y))

    def turns(index, extreme):
        if not interior[index]:
            return False
        depth = min(abs(y[0] - extreme), abs(y[-1] - extreme))
        return depth / span >= tolerance

    if turns(argmin, y[argmin]):
        return "u-shaped"
    if turns(argmax, y[argmax]):
        return "inverted-u"
    return "increasing" if y[-1] >= y[0] else "decreasing"


def _lr_test(full, reduced) -> float | None:
    """likelihood-ratio p-value for full against nested reduced"""
    if full is None or reduced is None:
        return None
    from scipy import stats

    statistic = 2.0 * (full.llf - reduced.llf)
    df = int(full.df_model - reduced.df_model)
    if df <= 0 or not np.isfinite(statistic) or statistic < 0:
        return None
    return float(stats.chi2.sf(statistic, df))


def fit_one(
    data: pl.DataFrame,
    score_col: str,
    outcome: str,
    kind: str,
    adjusted: bool,
    logger,
):
    """
    fit the spline, the linear comparison and the null, and summarise them

    Returns a dict of results and a curve over the score's range, or None if
    the data could not support a model.
    """
    columns = [score_col, outcome, *(ADJUSTERS if adjusted else ())]
    if adjusted:
        columns.append("site")
    usable = data.drop_nulls([c for c in columns if c in data.columns])
    if usable.height < MIN_ROWS:
        return None, None

    y = usable[outcome].cast(pl.Float64).to_numpy()
    if kind == "logistic":
        events = int(y.sum())
        if min(events, int(len(y) - events)) < MIN_EVENTS:
            return None, None

    raw = usable[score_col].cast(pl.Float64).to_numpy()
    score, (clip_lo, clip_hi) = _winsorise(raw)

    basis = _design(score, SPLINE_DF)
    spline_cols = list(basis.columns)

    extra = []
    extra_names = []
    if adjusted:
        for name in ADJUSTERS:
            extra.append(usable[name].cast(pl.Float64).to_numpy())
            extra_names.append(name)
        # site as a set of indicators, so a between-site difference in
        # case mix or coding does not load onto the fit score
        sites = usable["site"].to_list()
        levels = sorted(set(sites))[1:]
        for level in levels:
            extra.append(np.array([1.0 if s == level else 0.0 for s in sites]))
            extra_names.append(f"site[{level}]")

    intercept = np.ones((usable.height, 1))
    base = np.column_stack([intercept, *extra]) if extra else intercept

    X_spline = np.column_stack([base, basis.to_numpy()])
    X_linear = np.column_stack([base, score.reshape(-1, 1)])
    X_null = base

    spline = _fit(kind, y, X_spline)
    linear = _fit(kind, y, X_linear)
    null = _fit(kind, y, X_null)
    if spline is None:
        return None, None

    grid = np.linspace(np.nanmin(score), np.nanmax(score), 60)
    grid_basis = dmatrix(
        basis.design_info, {"x": grid}, return_type="dataframe"
    ).to_numpy()
    # hold the adjusters at their median, so the curve is a like-for-like
    # comparison across the score rather than a mixture of case mixes
    if extra:
        held = np.array([np.nanmedian(column) for column in extra])
        grid_base = np.column_stack(
            [np.ones((len(grid), 1)), np.tile(held, (len(grid), 1))]
        )
    else:
        grid_base = np.ones((len(grid), 1))
    X_grid = np.column_stack([grid_base, grid_basis])

    prediction = spline.get_prediction(X_grid)
    mean = prediction.predicted_mean
    confidence = prediction.conf_int()
    if kind == "logistic":
        # GLM with a Binomial family already returns the response scale here
        mean = np.clip(mean, 0.0, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)

    quantiles = np.nanquantile(score, [0.1, 0.5, 0.9])
    at = dmatrix(basis.design_info, {"x": quantiles}, return_type="dataframe").to_numpy()
    if extra:
        at_base = np.column_stack(
            [np.ones((3, 1)), np.tile(held, (3, 1))]
        )
    else:
        at_base = np.ones((3, 1))
    at_pred = spline.get_prediction(np.column_stack([at_base, at])).predicted_mean

    result = {
        "outcome": outcome,
        "score": score_col,
        "adjusted": adjusted,
        "n": int(usable.height),
        "n_events": int(y.sum()) if kind == "logistic" else None,
        "score_p10": float(quantiles[0]),
        "score_p50": float(quantiles[1]),
        "score_p90": float(quantiles[2]),
        "predicted_at_p10": float(at_pred[0]),
        "predicted_at_p50": float(at_pred[1]),
        "predicted_at_p90": float(at_pred[2]),
        "difference_p90_minus_p10": float(at_pred[2] - at_pred[0]),
        "p_overall": _lr_test(spline, null),
        "p_nonlinearity": _lr_test(spline, linear),
        "curve_shape": _curve_shape(grid, np.asarray(mean, dtype=float)),
        "clip_low": float(clip_lo),
        "clip_high": float(clip_hi),
    }
    curve = pl.DataFrame(
        {
            "outcome": [outcome] * len(grid),
            "score": [score_col] * len(grid),
            "adjusted": [adjusted] * len(grid),
            "x": grid,
            "fit": mean,
            "lower": confidence[:, 0],
            "upper": confidence[:, 1],
        }
    )
    return result, curve


def fit_interaction(
    data: pl.DataFrame, score_col: str, outcome: str, kind: str, concordant: str
):
    """
    is the fit score's association with outcome different in patients who got
    the recommended treatment?

    A single interaction term on the LINEAR score rather than on the spline.
    Interacting four spline bases with a binary exposure costs four more
    parameters and buys a test nobody can interpret; the linear interaction
    answers "does the slope differ", which is the question.
    """
    usable = data.drop_nulls([score_col, outcome, concordant, *ADJUSTERS])
    if usable.height < MIN_ROWS:
        return None
    flag = usable[concordant].cast(pl.Float64).to_numpy()
    if min(flag.sum(), len(flag) - flag.sum()) < MIN_ROWS / 4:
        return None

    y = usable[outcome].cast(pl.Float64).to_numpy()
    if kind == "logistic":
        events = int(y.sum())
        if min(events, int(len(y) - events)) < MIN_EVENTS:
            return None

    score, _ = _winsorise(usable[score_col].cast(pl.Float64).to_numpy())
    adjusters = [usable[name].cast(pl.Float64).to_numpy() for name in ADJUSTERS]
    base = np.column_stack([np.ones(len(y)), *adjusters, score, flag])
    full = np.column_stack([base, score * flag])

    with_interaction = _fit(kind, y, full)
    without = _fit(kind, y, base)
    if with_interaction is None or without is None:
        return None

    return {
        "outcome": outcome,
        "score": score_col,
        "n": int(usable.height),
        "n_concordant": int(flag.sum()),
        "slope_discordant": float(with_interaction.params[-3]),
        "interaction": float(with_interaction.params[-1]),
        "p_interaction": _lr_test(with_interaction, without),
    }


# ---------------------------------------------------------------- reporting


def adjustment_overlap(trial, score: str) -> dict:
    """
    which of this score's variables are also in the adjustment set

    The central limitation of the adjusted models, made explicit per trial
    AND per score, because the two scores are built from different variables
    and the difference decides whether adjustment means anything.

    SHOCK is the case that forced this to be score-specific. The only
    quartile it publishes is age, so its severity gap is a rescaling of age;
    adjusting that for age leaves the spline and the adjuster carrying the
    same information, and the fit returns a 72-point mortality difference
    that is collinearity rather than signal. Computed over all published
    variables the overlap reads 50% and looks tolerable; computed over the
    variables the score actually uses it reads 100%, which is the truth.

    `interpretable` is False at total overlap. Those rows are kept and
    labelled rather than dropped, because dropping them would leave the
    unadjusted estimate standing unopposed.
    """
    variables = set(score_variables(trial, score))
    shared = sorted(variables & set(ADJUSTER_SOURCES.values()))
    share = len(shared) / max(len(variables), 1)
    return {
        "trial": trial.label,
        "trial_key": trial.key,
        "score": score,
        "fit_score_variables": "; ".join(
            TABLE1_LABELS.get(v, v) for v in sorted(variables)
        )
        or "none",
        "also_adjusted_for": "; ".join(TABLE1_LABELS.get(v, v) for v in shared)
        or "none",
        "n_shared": len(shared),
        "n_score_variables": len(variables),
        "overlap_share": share,
        "adjusted_interpretable": share < 1.0,
    }


def plot_curves(curves: pl.DataFrame, out_path: pathlib.Path, anchor: str) -> None:
    """one row per outcome, one column per trial: fitted curve against score"""
    if curves.height == 0:
        return
    use = curves.filter(pl.col("score") == "severity_gap")
    if use.height == 0:
        use = curves
    trials = list(dict.fromkeys(use["trial"].to_list()))
    outcomes = [o for o, _, _ in OUTCOMES]
    labels = {o: label for o, label, _ in OUTCOMES}

    figure, axes = plt.subplots(
        len(outcomes),
        len(trials),
        figsize=(2.3 * len(trials), 2.2 * len(outcomes)),
        squeeze=False,
        sharex="col",
    )
    for row, outcome in enumerate(outcomes):
        for column, trial in enumerate(trials):
            axis = axes[row][column]
            axis.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
            axis.set_axisbelow(True)
            for adjusted, colour, name in (
                (False, CATEGORICAL[0], "unadjusted"),
                (True, CATEGORICAL[1], "adjusted"),
            ):
                part = use.filter(
                    (pl.col("trial") == trial)
                    & (pl.col("outcome") == outcome)
                    & (pl.col("adjusted") == adjusted)
                ).sort("x")
                if part.height == 0:
                    continue
                x = part["x"].to_numpy()
                axis.fill_between(
                    x, part["lower"].to_numpy(), part["upper"].to_numpy(),
                    color=colour, alpha=0.15, linewidth=0,
                )
                axis.plot(x, part["fit"].to_numpy(), color=colour,
                          linewidth=1.4, label=name)
            axis.axvline(0.0, color=INK_MUTED, linewidth=0.7, linestyle=":")
            if row == 0:
                axis.set_title(trial, fontsize=8)
            if column == 0:
                axis.set_ylabel(labels[outcome], fontsize=7)
            if row == len(outcomes) - 1:
                axis.set_xlabel("severity gap (IQR units)", fontsize=7)
    handles, names = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, names, loc="lower center", ncol=2,
                      bbox_to_anchor=(0.5, -0.02), fontsize=8)
    figure.suptitle(
        "Outcome against how far a patient sits from the trial's population, "
        f"by direction (anchor {anchor})",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.01, 1, 0.97))
    save_figure(figure, out_path)
    plt.close(figure)


def write_summary(
    path: pathlib.Path,
    results: pl.DataFrame,
    interactions: pl.DataFrame,
    overlap: pl.DataFrame,
    pending: int,
    anchor: str,
    logger,
) -> None:
    """the markdown write-up, limitations included rather than appended"""
    lines: list[str] = []
    add = lines.append

    def show_p(value) -> str:
        """p-values here run to 1e-300; rounding them all to 0.0000 hides
        the difference between strong and overwhelming"""
        if value is None:
            return "n/a"
        if value < 1e-4:
            return "<0.0001"
        return f"{value:.4f}"

    add(f"# Trial fit and outcome (anchor `{anchor}`)")
    add("")
    add(
        "Whether sitting outside the population a trial enrolled tracks with "
        "doing worse, and whether it still does once severity is accounted "
        "for."
    )
    add("")

    if pending:
        add(
            f"> **{pending} published baseline values are still placeholders.** "
            "The trials carrying them contribute eligibility but no fit score "
            "and so appear nowhere below. See `trial_table1_to_verify.csv`."
        )
        add("")

    add("## What was fitted")
    add("")
    add(
        f"Restricted cubic splines, {SPLINE_DF} degrees of freedom, on two "
        "readings of the same gap: `absdist`, how far outside the published "
        "interquartile ranges a patient sits, and `severity_gap`, which way "
        "-- positive meaning sicker or older than the trial's median. Each "
        "unadjusted and adjusted for non-cardiovascular SOFA, age and site."
    )
    add("")
    add(
        f"Mortality is logistic. Ventilator-free and vasopressor-free days to "
        f"day {HORIZON_DAYS} are least squares and are counted separately, "
        "not as a composite: they come off on different clocks, and a "
        "composite would be dominated by whichever has the longer tail."
    )
    add("")

    if results.height:
        add("## Results")
        add("")
        for outcome, label, _ in OUTCOMES:
            part = results.filter(
                (pl.col("outcome") == outcome) & (pl.col("score") == "severity_gap")
            ).sort("trial")
            if part.height == 0:
                continue
            add(f"### {label}, against the severity gap")
            add("")
            add(
                "| trial | adjusted | n | at p10 | at p90 | difference "
                "| shape | p (overall) | p (nonlinear) | note |"
            )
            add("|---|---|---:|---:|---:|---:|---|---:|---:|---|")
            for row in part.iter_rows(named=True):
                def show(value, digits=3):
                    return "n/a" if value is None else f"{value:.{digits}f}"
                note = (
                    ""
                    if row.get("interpretable", True)
                    else "collinear: score and adjuster are the same variable"
                )
                add(
                    f"| {row['trial']} | {'yes' if row['adjusted'] else 'no'} "
                    f"| {row['n']:,} | {show(row['predicted_at_p10'])} "
                    f"| {show(row['predicted_at_p90'])} "
                    f"| {show(row['difference_p90_minus_p10'])} "
                    f"| {row['curve_shape']} "
                    f"| {show_p(row['p_overall'])} "
                    f"| {show_p(row['p_nonlinearity'])} | {note} |"
                )
            add("")
            turning = part.filter(
                pl.col("curve_shape").is_in(["u-shaped", "inverted-u"])
                & pl.col("interpretable")
            )
            if turning.height:
                named = sorted(
                    {
                        f"{r['trial']} ({r['curve_shape']}, "
                        f"{'adjusted' if r['adjusted'] else 'unadjusted'})"
                        for r in turning.iter_rows(named=True)
                    }
                )
                add(
                    "**The curve turns** for " + ", ".join(named) + ". A "
                    "U-shape here means patients healthier than the trial on "
                    "the measured axes do worse than those at its median, "
                    "which an inside/outside split cannot show and which "
                    "usually means the score stops tracking severity in that "
                    "part of its range."
                )
                add("")
            if part.filter(~pl.col("interpretable")).height:
                add(
                    "Rows marked collinear are reported because suppressing "
                    "them would leave the unadjusted estimate unopposed, not "
                    "because they can be read as bias-corrected."
                )
                add("")

    if interactions.height:
        add("## Concordant care, by fit")
        add("")
        add(
            "Whether the fit score's slope differs between patients who got "
            "the recommended treatment and those who did not."
        )
        add("")
        add(
            "| trial | score | outcome | n | n concordant | interaction | p "
            "| what concordance means here |"
        )
        add("|---|---|---|---:|---:|---:|---:|---|")
        for row in interactions.sort(["trial", "score", "outcome"]).iter_rows(
            named=True
        ):
            add(
                f"| {row['trial']} | `{row['score']}` | {row['outcome']} "
                f"| {row['n']:,} | {row['n_concordant']:,} "
                f"| {row['interaction']:.4f} "
                f"| {show_p(row['p_interaction'])} "
                f"| {row['concordance_means']} |"
            )
        add("")
        add("Confounding pattern per trial:")
        add("")
        for row in (
            interactions.select("trial", "confounding")
            .unique()
            .sort("trial")
            .iter_rows(named=True)
        ):
            add(f"- **{row['trial']}** -- {row['confounding']}")
        add("")

    add("## Limitations")
    add("")
    add(
        "**The exposure and the confounder are partly the same measurement.** "
        "A patient is outside a trial's ranges largely because they are older "
        "or sicker, which is also why they die. Adjusting for age and SOFA "
        "therefore removes part of the exposure along with the confounding. "
        "The table below gives the overlap per trial; where it is high the "
        "adjusted estimate should not be read as a bias-corrected version of "
        "the unadjusted one."
    )
    add("")
    if overlap.height:
        add("| trial | score | built from | also adjusted for | overlap |")
        add("|---|---|---|---|---:|")
        for row in overlap.sort(
            ["overlap_share", "trial"], descending=[True, False]
        ).iter_rows(named=True):
            add(
                f"| {row['trial']} | `{row['score']}` "
                f"| {row['fit_score_variables']} "
                f"| {row['also_adjusted_for']} | {row['overlap_share']:.0%} |"
            )
        add("")
        total = overlap.filter(~pl.col("adjusted_interpretable"))
        if total.height:
            names = sorted(
                {f"{r['trial']} (`{r['score']}`)" for r in total.iter_rows(named=True)}
            )
            add(
                "At 100% overlap the adjusted model regresses a variable on "
                "itself. This affects " + ", ".join(names) + ". For those the "
                "unadjusted estimate is the only one worth reading, and the "
                "adjusted figure in the tables above is collinearity."
            )
            add("")

    add(
        "**Half the interquartile range is outside the interquartile range.** "
        "An IQR holds half the trial's own population by construction, so "
        "roughly half of a perfectly representative cohort would fall outside "
        "any given one. The distance and direction scores are built to be "
        "read against that null; a raw count of patients outside a range is "
        "not evidence of anything on its own."
    )
    add("")
    add(
        "**Free days are a floor, not a measurement.** A day counts as "
        "supported if any record falls in it, so intermittent support is "
        "scored as though it were continuous. This biases free days downward, "
        "and more so for vasopressors than for ventilation."
    )
    add("")
    add(
        "**Confounding by indication in the concordance analysis is severe "
        "and is not adjusted away.** Treatment is given partly because a "
        "clinician expected benefit. For the negative device trials the "
        "direction inverts: concordance there means the device was withheld, "
        "which happens most in patients too well or too sick to cannulate."
    )
    add("")
    add(
        "**Least squares on a bimodal bounded outcome.** Free days pile up at "
        "0 and at the horizon. The fitted values are conditional means, which "
        "is what trials report, but they are not predictions for a patient."
    )
    add("")
    add(
        "**Eligibility is a superset for most trials.** Criteria needing an "
        "echocardiogram, coronary anatomy, fluid volumes or a timed infusion "
        "duration cannot be applied here. `trial_criteria.csv` lists every "
        "one, per trial."
    )
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"summary -> {path}")


# --------------------------------------------------------------------- main


def analyse(paths, anchor: str, trial_cfg, args, logger) -> None:
    trials = load_trials(trial_cfg, logger)
    frame = load_alignment(paths, anchor, logger)
    frame = attach_free_days(paths, frame, logger)
    frame = attach_assigned(frame, trials, logger)

    pending = sum(
        1
        for key in trial_cfg.trials
        for name in (trial_cfg.trials[key].get("table1") or {})
        if (trial_cfg.trials[key].table1[name].get("verify", False))
    )

    results: list[dict] = []
    curves: list[pl.DataFrame] = []
    interactions: list[dict] = []
    overlaps: list[dict] = []
    skipped: list[dict] = []

    for trial in trials:
        key = trial.key
        elig = f"elig_{key}"
        if elig not in frame.columns:
            continue
        eligible = frame.filter(pl.col(elig))
        score_columns = {
            "absdist": f"t1_absdist_{key}",
            "severity_gap": f"t1_severity_gap_{key}",
        }
        usable_scores = {
            name: column
            for name, column in score_columns.items()
            if column in eligible.columns
            and eligible[column].null_count() < eligible.height
        }
        if not usable_scores:
            skipped.append(
                {
                    "trial": trial.label,
                    "reason": "no published baseline transcribed, so no fit score",
                    "n_eligible": eligible.height,
                }
            )
            continue

        overlap_by_score = {
            name: adjustment_overlap(trial, name) for name in usable_scores
        }
        overlaps.extend(overlap_by_score.values())
        logger.info(f"=== {trial.label}: {eligible.height:,} eligible")
        for name, row in overlap_by_score.items():
            if not row["adjusted_interpretable"]:
                logger.warning(
                    f"  {trial.label}: every variable behind `{name}` "
                    f"({row['fit_score_variables']}) is also in the "
                    "adjustment set; its adjusted models are reported but "
                    "marked uninterpretable"
                )

        for score_name, score_column in usable_scores.items():
            for outcome, _, kind in OUTCOMES:
                if outcome not in eligible.columns:
                    continue
                for adjusted in (False, True):
                    result, curve = fit_one(
                        eligible, score_column, outcome, kind, adjusted, logger
                    )
                    if result is None:
                        continue
                    result["trial"] = trial.label
                    result["trial_key"] = key
                    result["score"] = score_name
                    overlap_row = overlap_by_score[score_name]
                    result["adjustment_overlap"] = overlap_row["overlap_share"]
                    # an adjusted fit whose score and adjuster are the same
                    # variable is collinear, not confounding-corrected
                    result["interpretable"] = (
                        True if not adjusted
                        else bool(overlap_row["adjusted_interpretable"])
                    )
                    results.append(result)
                    curve = curve.with_columns(
                        trial=pl.lit(trial.label), score=pl.lit(score_name)
                    )
                    curves.append(curve)

        # ------------------------------------------------- concordance arm
        concordant = f"concordant_{key}"
        if (
            concordant in eligible.columns
            and eligible[concordant].null_count() < eligible.height
        ):
            # Read the direction off the concordance definition itself
            # rather than off the trial's description. An earlier version
            # keyed on words in `intervention`, which labelled OptimaCC
            # ("norepinephrine rather than epinephrine" -- a drug given) as a
            # withholding and ECMO-CS (genuinely `~had_ecmo`) as a treatment
            # given. The note beside each expression says what the flag
            # actually tests, so that is what gets classified.
            note = concordance_expressions().get(key, (None, ""))[1]
            lowered = note.lower()
            if "withheld" in lowered:
                flag = (
                    "withholding is 'concordant' (negative trial) -- the "
                    "sickest and the healthiest are both least likely to be "
                    "treated, so this inverts the usual direction"
                )
            elif "never given" in lowered:
                flag = (
                    "one agent used and another avoided -- confounded by "
                    "whichever indication drove the choice of agent"
                )
            elif "at least" in lowered or "at or below" in lowered:
                flag = (
                    "an achieved physiological value, not a treatment -- it "
                    "is partly an outcome of severity rather than a decision"
                )
            else:
                flag = (
                    "treatment given to those a clinician expected to "
                    "benefit -- the usual confounding by indication"
                )
            for score_name, score_column in usable_scores.items():
                for outcome, _, kind in OUTCOMES:
                    if outcome not in eligible.columns:
                        continue
                    row = fit_interaction(
                        eligible, score_column, outcome, kind, concordant
                    )
                    if row is None:
                        continue
                    row["trial"] = trial.label
                    row["score"] = score_name
                    row["concordance_means"] = note
                    row["confounding"] = flag
                    interactions.append(row)

    # ------------------------------------------------------------- outputs
    paths.table_home.mkdir(parents=True, exist_ok=True)
    results_frame = pl.DataFrame(results) if results else pl.DataFrame()
    curves_frame = (
        pl.concat(curves, how="diagonal_relaxed") if curves else pl.DataFrame()
    )
    interactions_frame = (
        pl.DataFrame(interactions) if interactions else pl.DataFrame()
    )
    overlap_frame = pl.DataFrame(overlaps) if overlaps else pl.DataFrame()

    for name, table in (
        (f"trial_fit_outcomes_{anchor}", results_frame),
        (f"trial_fit_curves_{anchor}", curves_frame),
        (f"trial_fit_interaction_{anchor}", interactions_frame),
        ("trial_fit_adjustment_overlap", overlap_frame),
        (f"trial_fit_skipped_{anchor}", pl.DataFrame(skipped) if skipped else pl.DataFrame()),
    ):
        if table.height:
            table.write_csv(paths.table_home / f"{name}.csv")
            logger.info(f"  {name}.csv  ({table.height} rows)")

    # ------------------------------------------- the four-cell comparison
    cells = four_cell_table(frame, trials, logger)
    if cells.height:
        cells.write_csv(paths.table_home / f"trial_fit_cells_{anchor}.csv")
        logger.info(f"  trial_fit_cells_{anchor}.csv ({cells.height} rows)")

    style()
    figure_home = paths.figure_home / f"anchor-{anchor}"
    figure_home.mkdir(parents=True, exist_ok=True)
    if curves_frame.height:
        plot_curves(
            curves_frame, figure_home / "trial_fit_outcomes.png", anchor
        )
    if cells.height:
        plot_four_cells(
            frame, cells, trials, figure_home / "trial_fit_cells.png", anchor
        )
    plot_space(frame, figure_home / "trial_fit_space.png", anchor, logger)
    logger.info(f"  figures -> {figure_home}")

    write_summary(
        paths.output_home / f"trial_fit_summary_{anchor}.md",
        results_frame,
        interactions_frame,
        overlap_frame,
        pending,
        anchor,
        logger,
    )

    if skipped:
        logger.warning(
            f"{len(skipped)} trial(s) had no fit score and were skipped:"
        )
        for row in skipped:
            logger.warning(
                f"      {row['trial']:<22} {row['n_eligible']:>7,} eligible "
                f"-- {row['reason']}"
            )


# ------------------------------------------------- the four-cell comparison

#: the 2x2 the outcome tables could not show, in reading order.
#:
#: (inside the published IQRs, treated per the guideline) -> label. The first
#: cell is the only one the trial's own result actually speaks to: a patient
#: who resembles the enrolled population *and* got what the trial recommends.
#: The other three are each one step away from it, and keeping them in a fixed
#: order means the same cell sits in the same place in every panel.
CELLS = (
    (True, True, "resembles\ntreated"),
    (True, False, "resembles\nuntreated"),
    (False, True, "differs\ntreated"),
    (False, False, "differs\nuntreated"),
)

#: below this a cell gets its count printed and no estimate.
#:
#: The hollow marker already says "under 50, do not read the shape", but
#: ECLS-SHOCK has one patient who resembles its population and was not
#: treated, and a Wilson interval on n=1 drew a point at 100% mortality
#: spanning the whole axis. At that size the estimate is not weak evidence,
#: it is no evidence, and drawing it invites a reader to see a pattern in
#: the position of a single patient.
MIN_CELL_TO_DRAW = 10

#: blue for the cells the trial tested, orange for the ones it did not
CELL_COLOURS = (CATEGORICAL[0], CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[1])


def _wilson(successes: int, total: int, z: float = 1.96):
    """Wilson score interval, which behaves at the small counts here"""
    if total == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = (
        z
        * np.sqrt(p * (1.0 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return p, max(0.0, centre - spread), min(1.0, centre + spread)


def four_cell_table(frame: pl.DataFrame, trials, logger) -> pl.DataFrame:
    """
    one row per trial x outcome x cell: for the figure, and to check it against

    A figure a reader cannot check against numbers is an assertion, so the same
    cells are written out as a table. `reliable` carries the same 50-patient
    floor the inside/outside tables use: ECLS-SHOCK puts six concordant
    patients inside its published ranges, and a violin of six points is a
    drawing rather than a distribution.
    """
    rows = []
    for trial in trials:
        key = trial.key
        outside_col, concordant_col = f"t1_outside_{key}", f"concordant_{key}"
        compared = f"t1_n_compared_{key}"
        if outside_col not in frame.columns or concordant_col not in frame.columns:
            continue
        part = frame.filter(
            pl.col(f"elig_{key}")
            & (pl.col(compared) > 0)
            & pl.col(outside_col).is_not_null()
            & pl.col(concordant_col).is_not_null()
        )
        if part.height < 50:
            continue
        part = part.with_columns(resembles=pl.col(outside_col) == 0)
        for resembles, concordant, label in CELLS:
            cell = part.filter(
                (pl.col("resembles") == resembles)
                & (pl.col(concordant_col) == concordant)
            )
            for outcome, pretty, kind in OUTCOMES:
                if outcome not in cell.columns:
                    continue
                values = cell[outcome].drop_nulls()
                if not values.len():
                    continue
                if kind == "logistic":
                    point, low, high = _wilson(
                        int(values.sum()), int(values.len())
                    )
                else:
                    point = float(values.mean())
                    spread = float(values.std() or 0.0) / max(
                        np.sqrt(values.len()), 1.0
                    )
                    low, high = point - 1.96 * spread, point + 1.96 * spread
                rows.append(
                    {
                        "trial": trial.label,
                        "outcome": outcome,
                        "resembles_trial": resembles,
                        "treated_per_guideline": concordant,
                        "cell": label.replace(chr(10), " "),
                        "n": int(values.len()),
                        "value": point,
                        "ci_low": low,
                        "ci_high": high,
                        "reliable": int(values.len()) >= 50,
                    }
                )
    table = pl.DataFrame(rows) if rows else pl.DataFrame()
    if table.height:
        thin = table.filter(~pl.col("reliable"))["cell"].len()
        logger.info(
            f"  four-cell: {table.height} cells, {thin} below the 50-patient floor"
        )
    return table


def attach_assigned(frame: pl.DataFrame, trials, logger) -> pl.DataFrame:
    """
    collapse the per-trial columns onto the one guideline each patient was given

    11 assigns every patient their single most relevant guideline and writes
    `assigned_guideline`, but leaves the fit score and the concordance flag in
    eighteen per-trial columns. The space figure needs one column per quantity,
    not eighteen, and picking the assigned trial's is the only collapse that
    does not mix patients judged against different trials into one scale.

    Adds `assigned_absdist`, `assigned_severity_gap` and
    `concordant_assigned`, each null where no guideline was assigned -- which
    is 18,779 of 34,142 patients and shows up as the grey context layer.
    """
    if "assigned_guideline" not in frame.columns:
        return frame
    label_to_key = {t.label: t.key for t in trials}
    for target, prefix in (
        ("assigned_absdist", "t1_absdist_"),
        ("assigned_severity_gap", "t1_severity_gap_"),
        ("concordant_assigned", "concordant_"),
    ):
        branches = [
            (pl.col("assigned_guideline") == label, pl.col(f"{prefix}{key}"))
            for label, key in label_to_key.items()
            if f"{prefix}{key}" in frame.columns
        ]
        if not branches:
            continue
        expression = pl.when(branches[0][0]).then(branches[0][1])
        for condition, value in branches[1:]:
            expression = expression.when(condition).then(value)
        frame = frame.with_columns(expression.otherwise(None).alias(target))
    covered = int(frame["assigned_absdist"].is_not_null().sum()) if (
        "assigned_absdist" in frame.columns
    ) else 0
    logger.info(
        f"  assigned-guideline fit score present for {covered:,} of "
        f"{frame.height:,} patients"
    )
    return frame


def plot_four_cells(
    frame: pl.DataFrame, cells: pl.DataFrame, trials, out_path: pathlib.Path,
    anchor: str,
) -> None:
    """
    the four cells drawn, one row per outcome and one column per trial

    Replaces three wide tables that nobody could read across. The question is
    whether resembling the enrolled population and being treated as the trial
    recommends move the outcome, and in a table that comparison is four numbers
    in different rows of different tables.

    The mark differs by row because the outcomes do. Mortality is binary, and a
    violin of a binary variable is two lumps at 0 and 1 that say only what the
    proportion already says, so it gets a proportion with a Wilson interval.
    The free-day outcomes are bounded and bimodal -- piled at 0 and at the
    28-day horizon -- which is exactly where a violin earns its place and a mean
    with an error bar misleads, so they get violins with the cell mean marked.

    Cells below 50 patients are drawn hollow. ECLS-SHOCK has six patients in
    its resembles-and-treated cell, and at that size the shape of a violin is
    an artefact of the kernel rather than a property of the patients.
    """
    if cells.height == 0:
        return
    trial_labels = [t.label for t in trials if t.label in set(cells["trial"])]
    if not trial_labels:
        return
    outcomes = [o for o, _, _ in OUTCOMES]
    pretty = {o: label for o, label, _ in OUTCOMES}
    kinds = {o: kind for o, _, kind in OUTCOMES}
    key_of = {t.label: t.key for t in trials}

    # sharey per row: the comparison a reader makes is across trials within an
    # outcome, and seven independent mortality axes made a 10-point difference
    # at one trial look like a 30-point difference at another
    figure, axes = plt.subplots(
        len(outcomes),
        len(trial_labels),
        figsize=(3.0 * len(trial_labels), 2.6 * len(outcomes)),
        squeeze=False,
        sharey="row",
    )
    positions = np.arange(len(CELLS))

    for row, outcome in enumerate(outcomes):
        for column, label in enumerate(trial_labels):
            axis = axes[row][column]
            axis.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)
            axis.set_axisbelow(True)
            key = key_of[label]
            outside_col, concordant_col = f"t1_outside_{key}", f"concordant_{key}"
            part = frame.filter(
                pl.col(f"elig_{key}")
                & pl.col(outside_col).is_not_null()
                & pl.col(concordant_col).is_not_null()
            ).with_columns(resembles=pl.col(outside_col) == 0)

            for index, (resembles, concordant, _) in enumerate(CELLS):
                cell_rows = cells.filter(
                    (pl.col("trial") == label)
                    & (pl.col("outcome") == outcome)
                    & (pl.col("resembles_trial") == resembles)
                    & (pl.col("treated_per_guideline") == concordant)
                )
                if not cell_rows.height:
                    continue
                stats = cell_rows.row(0, named=True)
                colour = CELL_COLOURS[index]
                solid = bool(stats["reliable"])
                if stats["n"] < MIN_CELL_TO_DRAW:
                    axis.annotate(
                        f"n={stats['n']}",
                        (positions[index], 0.5), xycoords=("data", "axes fraction"),
                        ha="center", va="center", fontsize=6, color=INK_MUTED,
                    )
                    continue

                if kinds[outcome] == "logistic":
                    axis.errorbar(
                        positions[index], stats["value"],
                        yerr=[[stats["value"] - stats["ci_low"]],
                              [stats["ci_high"] - stats["value"]]],
                        fmt="o", markersize=6, color=colour,
                        markerfacecolor=colour if solid else "white",
                        capsize=3, linewidth=1.2,
                    )
                else:
                    values = (
                        part.filter(
                            (pl.col("resembles") == resembles)
                            & (pl.col(concordant_col) == concordant)
                        )[outcome]
                        .drop_nulls()
                        .to_numpy()
                    )
                    if values.size >= 5:
                        body = axis.violinplot(
                            [values], positions=[positions[index]],
                            widths=0.72, showextrema=False,
                        )
                        for patch in body["bodies"]:
                            patch.set_facecolor(colour)
                            patch.set_alpha(0.55 if solid else 0.22)
                    axis.plot(
                        positions[index], stats["value"], "o", markersize=4.5,
                        color=colour,
                        markerfacecolor=colour if solid else "white",
                    )
                axis.annotate(
                    f"{stats['n']:,}",
                    (positions[index], 0.02), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6, color=INK_MUTED,
                )

            if row == 0:
                axis.set_title(label, fontsize=9)
            if column == 0:
                axis.set_ylabel(pretty[outcome], fontsize=8)
            axis.set_xticks(positions)
            if row == len(outcomes) - 1:
                axis.set_xticklabels([c[2] for c in CELLS], fontsize=6.5)
            else:
                axis.set_xticklabels([])
            axis.tick_params(labelsize=7)

    figure.suptitle(
        "Outcome by whether the patient resembles the trial population and "
        f"was treated as it recommends ({anchor} anchor)",
        fontsize=10.5,
    )
    figure.supxlabel(
        "blue = cells the trial's result speaks to; hollow = fewer than 50 "
        "patients, shape not interpretable; numbers are cell sizes",
        fontsize=7, color=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    save_figure(figure, out_path)
    plt.close(figure)


def plot_space(frame: pl.DataFrame, out_path: pathlib.Path, anchor: str,
               logger) -> None:
    """
    the cohort in 11's projection, coloured by fit and by outcome side by side

    The question this answers and the tables cannot: are the patients who sit
    far from every trial population the *same* patients who do badly, or a
    different region of the space. A table of means per trial cannot separate
    "poor fit and death coincide" from "poor fit and death are both common",
    and putting both on one layout can.

    EVERY NUMERIC PANEL IS A BINNED LOCAL MEAN, NOT A SCATTER. The first
    version drew one point per patient and was unreadable for exactly the
    reason this cohort is interesting: mortality is 16.6%, so 5,600 coloured
    points over 28,500 grey ones read as "most of them died" wherever the
    coloured ones happened to land on top. A hexagonal bin coloured by the mean
    of its members shows the local rate, which is the quantity the eye was
    being asked to estimate anyway, and makes the panels comparable to one
    another instead of comparable to whichever was drawn last. Bins under
    `mincnt` patients are left empty rather than shown at a rate computed from
    three people.

    Read the fit panels against each other before reading either against the
    outcomes. If fit and coverage disagree about which region is poorly served,
    the eligibility criteria and the baseline tables are picking out different
    patients -- which is section 3's finding, restated geometrically.
    """
    if any(c not in frame.columns for c in ("pacmap_1", "pacmap_2")):
        logger.warning(
            "  alignment file carries no pacmap coordinates; space figure "
            "skipped (re-run 11 after 03)"
        )
        return

    panels: list[tuple[str, str, str, str]] = []
    if "assigned_absdist" in frame.columns:
        panels.append(("assigned_absdist",
                       "distance outside the assigned trial's IQRs",
                       "mean", "IQRs"))
    if "guideline_fit" in frame.columns:
        panels.append(("guideline_fit", "how the assigned guideline fits",
                       "categorical", ""))
    if "n_regions_inside" in frame.columns:
        panels.append(("n_regions_inside", "trial regions the patient is inside",
                       "mean", "regions"))
    if "in_hospital_mortality" in frame.columns:
        panels.append(("in_hospital_mortality", "in-hospital mortality",
                       "rate", "share who died"))
    for column, label in (
        ("vent_free_days", f"ventilator-free days to {HORIZON_DAYS}"),
        ("pressor_free_days", f"vasopressor-free days to {HORIZON_DAYS}"),
    ):
        if column in frame.columns:
            panels.append((column, label, "mean", "days"))
    if "concordant_assigned" in frame.columns:
        panels.append(("concordant_assigned", "assigned guideline followed",
                       "rate", "share treated per guideline"))
    if not panels:
        return

    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.6 * columns, 3.4 * rows), squeeze=False
    )
    x = frame["pacmap_1"].to_numpy().astype(float)
    y = frame["pacmap_2"].to_numpy().astype(float)
    mincnt = 10

    for position, (column, label, kind, units) in enumerate(panels):
        axis = axes[position // columns][position % columns]

        if kind == "categorical":
            raw = frame[column].to_list()
            levels = sorted({v for v in raw if v is not None})
            present = np.array([v is not None for v in raw])
            axis.scatter(
                x[~present], y[~present], s=1.0, c=CONTEXT, linewidths=0,
                rasterized=True,
            )
            # commonest level first, so the 337 patients matching a trial on
            # both counts are not buried under the 18,779 matching none
            ordered = sorted(levels, key=lambda v: -sum(1 for r in raw if r == v))
            for index, level in enumerate(ordered):
                mask = np.array([r == level for r in raw])
                axis.scatter(
                    x[mask], y[mask], s=1.6 + 1.4 * index,
                    c=CATEGORICAL[index % len(CATEGORICAL)],
                    linewidths=0, rasterized=True, zorder=2 + index,
                    label=f"{level} ({int(mask.sum()):,})",
                )
            axis.legend(
                fontsize=5.5, frameon=False, loc="lower center",
                bbox_to_anchor=(0.5, -0.17), markerscale=4,
                handletextpad=0.2, labelspacing=0.25,
            )
            axis.set_title(label, fontsize=8.5, loc="left")
        else:
            values = frame[column].cast(pl.Float64).to_numpy()
            known = ~np.isnan(values)
            axis.scatter(
                x[~known], y[~known], s=0.8, c=CONTEXT, linewidths=0,
                rasterized=True,
            )
            if int(known.sum()) < mincnt:
                axis.set_title(f"{label}\nnot measured", fontsize=8.5,
                               loc="left")
            else:
                top = float(np.nanpercentile(values[known], 98))
                bottom = float(np.nanmin(values[known]))
                mesh = axis.hexbin(
                    x[known], y[known], C=values[known],
                    reduce_C_function=np.mean,
                    gridsize=46, mincnt=mincnt, cmap="viridis",
                    # rate panels: floor at 0 but let the top come from the
                    # bins. Pinning it to 1.0 is defensible and unreadable --
                    # local mortality tops out near 0.5, so half the ramp went
                    # unused and every bin sat in the dark third of it.
                    vmin=0.0 if kind == "rate" else bottom,
                    vmax=None if kind == "rate" else max(top, bottom + 1e-9),
                    linewidths=0,
                )
                bar = figure.colorbar(mesh, ax=axis, fraction=0.042, pad=0.02)
                bar.ax.tick_params(labelsize=6, labelcolor=INK_SECONDARY)
                bar.outline.set_visible(False)
                bar.set_label(units, fontsize=6, color=INK_SECONDARY)
                overall = float(np.nanmean(values[known]))
                summary = (
                    f"{overall:.1%} overall" if kind == "rate"
                    else f"mean {overall:.1f}"
                )
                axis.set_title(
                    f"{label}\n{summary}, {int(known.sum()):,} patients",
                    fontsize=8.5, loc="left",
                )
        axis.set_xticks([])
        axis.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            axis.spines[side].set_visible(False)

    for position in range(len(panels), rows * columns):
        axes[position // columns][position % columns].axis("off")

    figure.suptitle(
        f"Guideline fit and outcome on one layout ({anchor} anchor, PaCMAP)",
        fontsize=11, x=0.01, ha="left",
    )
    figure.supxlabel(
        f"hexagons are local means over at least {mincnt} patients; pale grey "
        "is not measured for that panel; mean scales clipped at p98",
        fontsize=7, color=INK_SECONDARY, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_figure(figure, out_path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--anchor", choices=ANCHORS, help="one anchor only")
    parser.add_argument("--model", help="which checkpoint's embeddings")
    args = parser.parse_args()

    paths = load_paths(model=args.model) if args.model else load_paths()
    paths.mkdirs()
    logger = get_logger(
        "12_trial_fit_outcomes", paths.log_home / "12_trial_fit_outcomes.log"
    )
    trial_cfg = OmegaConf.load(
        pathlib.Path(__file__).resolve().parent / "config" / "trials.yaml"
    )

    for anchor in ([args.anchor] if args.anchor else ANCHORS):
        analyse(paths, anchor, trial_cfg, args, logger)
    logger.info("12 done")


if __name__ == "__main__":
    main()

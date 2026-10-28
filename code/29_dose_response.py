#!/usr/bin/env python3

"""
29_dose_response.py -- what the pressure does in the hour after the dose moves

One point per patient-hour where the norepinephrine dose changed: the change in
dose on x, the change in MAP over the FOLLOWING hour on y, coloured by hours
since shock onset, panelled by what else was running.

    python code/29_dose_response.py --subset cardiogenic

THE UNIT IS THE PATIENT-HOUR, NOT THE PATIENT. The earlier sensitivity fit
asked whether patients differ from one another in dose responsiveness and found
that they do not -- among patients with ten or more titrations the spread in
response rate was smaller than binomial noise around a single shared rate. That
result says nothing about whether individual HOURS differ, which is a different
question and the one asked here.

DOSES ARE IN NATIVE UNITS, not norepinephrine equivalents. The x-axis is the
change in norepinephrine in mcg/kg/min, because that is the number on the pump.
A composite would make the axis unreadable and would confound turning the
norepinephrine up with starting a second drug.

WHAT THE SCATTER CANNOT SETTLE, and why a null is drawn beside it: the shape
everyone expects to see -- a dense wedge of points with small dose changes and
large pressure swings -- is exactly what measurement noise produces. MAP here
carries an observation sd near 6 mmHg, so any hour with a small dose change has
a y-value dominated by noise, and dividing one by the other manufactures
"sensitivity" wherever the denominator is small. Every ratio-based statement
below is therefore reported against a permutation null in which the same dose
changes are paired with pressure changes drawn from other hours.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.action_response import hourly_map
from clifcs.config import CONFIG_HOME
from clifcs.nee import agent_rates, hourly_by_agent

END_HOURS = 168.0

#: panel -> the agent that must ALSO be running. None means norepinephrine
#: alone. Panels are defined by what accompanies the norepinephrine, so a
#: patient on three drugs appears in more than one -- stated rather than
#: silently deduplicated, because forcing exact-regimen panels would empty
#: them: combinations are the norm in this cohort, not the exception.
PANELS = (
    ("norepinephrine alone", None),
    ("norepinephrine + vasopressin", "vasopressin"),
    ("norepinephrine + epinephrine", "epinephrine"),
    ("norepinephrine + phenylephrine", "phenylephrine"),
)
COMPANIONS = ("vasopressin", "epinephrine", "phenylephrine")


def build_events(hourly, cfg, logger):
    """
    one row per hour in which the norepinephrine dose moved

    `d_ne` is the change from the previous hour to this one; `d_map` is the
    change from this hour to the next. The response window therefore starts
    where the dose change finished, never overlapping it -- a MAP read before
    the dose moved cannot be a response to it.
    """
    absolute = float(cfg.action.absolute_minimum)
    ordered = hourly.sort(["hospitalization_id", "offset_hours"])
    frame = ordered.with_columns(
        d_ne=pl.col("dose_norepinephrine").diff().over("hospitalization_id"),
        d_map=(pl.col("map").shift(-1) - pl.col("map"))
        .over("hospitalization_id"),
        gap_back=pl.col("offset_hours").diff().over("hospitalization_id"),
        gap_forward=(pl.col("offset_hours").shift(-1) - pl.col("offset_hours"))
        .over("hospitalization_id"),
    ).filter(
        # both the dose change and the response must span exactly one hour,
        # or the x and y axes mean different things from row to row
        (pl.col("gap_back") == 1.0) & (pl.col("gap_forward") == 1.0)
        & pl.col("d_ne").is_not_null() & pl.col("d_map").is_not_null()
    )
    moved = frame.filter(pl.col("d_ne").abs() >= absolute)
    logger.info(
        f"  {frame.height:,} hours with a complete one-hour window either "
        f"side; {moved.height:,} of them changed the norepinephrine by at "
        f"least {absolute} mcg/kg/min"
    )
    for agent in COMPANIONS:
        moved = moved.with_columns(
            **{f"on_{agent}": pl.col(f"dose_{agent}") > 0}
        )
    moved = moved.with_columns(
        n_companions=sum(
            pl.col(f"on_{agent}").cast(pl.Int64) for agent in COMPANIONS
        )
    )
    return moved


def panel_rows(events, label, companion):
    """the subset of events belonging to one panel"""
    if companion is None:
        return events.filter(pl.col("n_companions") == 0)
    return events.filter(pl.col(f"on_{companion}"))


def ratio_structure(events, cfg, logger, seed=42, rounds=200):
    """
    is there a 'more sensitive' and a 'less sensitive' population of hours?

    The sensitivity of an hour is |d_map| / |d_ne|, which is unusable near
    d_ne = 0 and is why the dose floor exists. Two things are asked of it:

      SPREAD AGAINST A NULL. If hours genuinely differ in responsiveness the
      observed spread of log-ratio should exceed what pairing the same dose
      changes with pressure changes from other hours produces. If it does not,
      the spread is noise divided by a small number.

      BIMODALITY. A 'more sensitive' and a 'less sensitive' cluster would show
      as two modes. Dip-test-style bimodality is not run; instead the observed
      log-ratio density is compared with the null's at the same quantiles,
      which answers the weaker but better-posed question of whether the tails
      are heavier than noise alone would make them.
    """
    floor = float(cfg.action.absolute_minimum)
    usable = events.filter(
        (pl.col("d_ne").abs() >= floor) & (pl.col("d_map").abs() > 0)
    )
    if usable.height < 500:
        logger.warning("too few usable hours for the ratio analysis")
        return pl.DataFrame()
    d_ne = usable["d_ne"].to_numpy()
    d_map = usable["d_map"].to_numpy()
    ratio = np.log10(np.abs(d_map) / np.abs(d_ne))

    rng = np.random.default_rng(seed)
    spreads = np.empty(rounds)
    for i in range(rounds):
        shuffled = rng.permutation(d_map)
        spreads[i] = np.std(np.log10(np.abs(shuffled) / np.abs(d_ne)))
    observed = float(np.std(ratio))
    p = float((spreads >= observed).mean())
    logger.info(
        f"  log10 |dMAP/dNE|: observed sd {observed:.4f}, "
        f"null {spreads.mean():.4f} +/- {spreads.std():.4f}, p={p:.3f}"
    )
    return pl.DataFrame([{
        "hours": usable.height,
        "median_log_ratio": float(np.median(ratio)),
        "observed_sd": observed,
        "null_sd_mean": float(spreads.mean()),
        "null_sd_sd": float(spreads.std()),
        "p_spread_exceeds_null": p,
        "iqr_log_ratio": float(
            np.percentile(ratio, 75) - np.percentile(ratio, 25)
        ),
        "null_iqr": float(np.percentile(
            np.log10(np.abs(rng.permutation(d_map)) / np.abs(d_ne)), 75
        ) - np.percentile(
            np.log10(np.abs(rng.permutation(d_map)) / np.abs(d_ne)), 25
        )),
    }])


def plot_panels(events, paths, subset, logger, max_points=25000, seed=42):
    """
    four scatterplots, one per regimen, coloured by hours since onset

    POINTS ARE SUBSAMPLED AND DRAWN SMALL. With tens of thousands of hours per
    panel a full-opacity scatter is a solid block whose shape is set by draw
    order, so the densest region -- which is the whole question -- is the least
    readable part. A fixed subsample with low alpha keeps the density visible;
    the panel titles carry the true counts.

    A REFERENCE SLOPE IS NOT DRAWN. Any line through this cloud implies a gain
    that the data does not support, and a reader will take a drawn slope as the
    answer. The quadrant guides are there instead: they say only where the
    pressure moved the way the dose was pushed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from clifcs.figures import save_figure

    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 11.0), sharex=True,
                                sharey=True)
    scatter = None
    for panel, (label, companion) in zip(axes.ravel(), PANELS):
        part = panel_rows(events, label, companion)
        if part.is_empty():
            panel.set_title(f"{label}\n(no hours)", fontsize=10)
            continue
        if part.height > max_points:
            keep = rng.choice(part.height, size=max_points, replace=False)
            shown = part[keep]
        else:
            shown = part
        scatter = panel.scatter(
            shown["d_ne"], shown["d_map"],
            c=shown["offset_hours"], cmap="viridis", s=5, alpha=0.35,
            linewidths=0, rasterized=True, vmin=0, vmax=END_HOURS,
        )
        panel.axhline(0, color="#888", lw=0.8, zorder=0)
        panel.axvline(0, color="#888", lw=0.8, zorder=0)
        same_way = float(
            ((part["d_ne"] > 0) == (part["d_map"] > 0)).mean()
        )
        panel.set_title(label, fontsize=11)
        # The counts go INSIDE the axes. As a second title line they ran
        # wider than the panel and collided with the neighbouring title,
        # which is what sharing axes does to long titles.
        up_mean = part.filter(pl.col("d_ne") > 0)["d_map"].mean()
        down_mean = part.filter(pl.col("d_ne") < 0)["d_map"].mean()
        panel.text(
            0.02, 0.975,
            f"{part.height:,} hours\n"
            f"{same_way:.0%} moved the way the dose was pushed\n"
            f"median |dMAP| {part['d_map'].abs().median():.1f} mmHg\n"
            f"mean dMAP  up {up_mean:+.2f}   down {down_mean:+.2f}",
            transform=panel.transAxes, va="top", ha="left", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78,
                  "edgecolor": "none", "pad": 3.0},
        )
    for panel in axes[-1]:
        panel.set_xlabel("change in norepinephrine over the previous hour\n"
                         "(mcg/kg/min)")
    for panel in axes[:, 0]:
        panel.set_ylabel("change in MAP over the NEXT hour (mmHg)")
    # The bulk of dose changes are tiny; without clipping, a handful of large
    # ones set the scale and the informative region collapses to a line.
    limit = float(np.nanpercentile(np.abs(events["d_ne"].to_numpy()), 99))
    map_limit = float(np.nanpercentile(np.abs(events["d_map"].to_numpy()), 99))
    for panel in axes.ravel():
        panel.set_xlim(-limit, limit)
        panel.set_ylim(-map_limit, map_limit)
    if scatter is not None:
        bar = figure.colorbar(
            scatter, ax=axes, fraction=0.025, pad=0.02
        )
        bar.set_label("hours since shock onset")
        bar.solids.set(alpha=1.0)
    figure.suptitle(
        f"{subset} shock: norepinephrine change against the MAP response in "
        f"the following hour\naxes clipped at the 99th percentile of |change|",
        fontsize=12,
    )
    out = (
        paths.output_home / "figures" / paths.model_key
        / f"dose_response_scatter_{subset}.png"
    )
    save_figure(figure, out, manuscript=True)
    logger.info(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--end-hours", type=float, default=END_HOURS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger("29_dose_response", paths.log_home / "29_dose_response.log")
    cfg = OmegaConf.load(args.config or CONFIG_HOME / "dynamics.yaml")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "longitudinal",
        pathlib.Path(__file__).resolve().parent / "11_longitudinal_clusters.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cohort = module.cohort_ids(paths, args.subset, logger)
    onset = pl.read_parquet(paths.cohort_home / "cs_onset.parquet").select(
        "hospitalization_id",
        onset=pl.col("cs_onset_dttm").cast(pl.Datetime("us")),
    )
    from clifcs.disposition import bound_to_record, dispositions

    parts = []
    for site in paths.sites:
        ids = cohort.filter(
            pl.col("site") == site
        )["hospitalization_id"].to_list()
        if not ids:
            continue
        anchors = onset.filter(pl.col("hospitalization_id").is_in(ids))
        logger.info(f"[{site}] {anchors.height:,} patients")
        rates = agent_rates(paths.raw_subset(site), anchors, cfg, logger)
        per_agent = hourly_by_agent(
            rates, anchors, cfg, args.end_hours, logger
        )
        pressure = hourly_map(
            paths.raw_subset(site), anchors, cfg, args.end_hours, logger
        )
        merged = per_agent.join(
            pressure.select("hospitalization_id", "offset_hours", "map"),
            on=["hospitalization_id", "offset_hours"], how="left",
        ).with_columns(site=pl.lit(site))
        merged = bound_to_record(
            merged, dispositions(paths.raw_subset(site), anchors, logger),
            logger,
        )
        parts.append(merged)
    if not parts:
        logger.error("no site produced an hourly series")
        return
    hourly = pl.concat(parts, how="diagonal_relaxed")

    # The FULL grid, before the dose-change filter. 31 needs hold hours --
    # hours in which nothing was changed -- and they are exactly what
    # `build_events` removes, so writing only the event table left the hourly
    # state space unable to represent a patient sitting still.
    hourly.write_parquet(
        paths.patient_level_home / f"hourly_agents_{args.subset}.parquet"
    )
    logger.info(
        f"wrote the full per-agent grid: {hourly.height:,} patient-hours"
    )

    events = build_events(hourly, cfg, logger)
    if events.is_empty():
        logger.error("no dose-change events")
        return
    events.write_parquet(
        paths.patient_level_home / f"dose_response_events_{args.subset}.parquet"
    )

    logger.info("per-panel summary:")
    rows = []
    for label, companion in PANELS:
        part = panel_rows(events, label, companion)
        if part.is_empty():
            continue
        up = part.filter(pl.col("d_ne") > 0)
        down = part.filter(pl.col("d_ne") < 0)
        row = {
            "panel": label,
            "hours": part.height,
            "patients": part["hospitalization_id"].n_unique(),
            "median_abs_d_ne": float(part["d_ne"].abs().median()),
            "median_abs_d_map": float(part["d_map"].abs().median()),
            "moved_the_same_way": float(
                ((part["d_ne"] > 0) == (part["d_map"] > 0)).mean()
            ),
            "mean_d_map_after_increase": float(up["d_map"].mean())
            if up.height else None,
            "mean_d_map_after_decrease": float(down["d_map"].mean())
            if down.height else None,
        }
        rows.append(row)
        logger.info(
            f"    {label:<32} {part.height:>7,} hours  "
            f"same way {row['moved_the_same_way']:>5.1%}  "
            f"dMAP after up {row['mean_d_map_after_increase']:>+5.2f} / "
            f"down {row['mean_d_map_after_decrease']:>+5.2f}"
        )
    pl.DataFrame(rows).write_csv(
        tables / f"dose_response_panels_{args.subset}.csv"
    )

    logger.info("is there a more-sensitive and a less-sensitive set of hours?")
    structure = ratio_structure(events, cfg, logger, seed=args.seed)
    if not structure.is_empty():
        structure.write_csv(
            tables / f"dose_response_structure_{args.subset}.csv"
        )

    plot_panels(events, paths, args.subset, logger, seed=args.seed)
    logger.info("29 done")


if __name__ == "__main__":
    main()

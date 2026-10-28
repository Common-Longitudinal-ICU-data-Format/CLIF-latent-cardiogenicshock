#!/usr/bin/env python3

"""
30_response_regions.py -- the dose-response plane read as clinical regions,
and whether patients differ

Three questions, in order:

  1. Does the MAP response scale with the SIZE of the dose change? Answered by
     binning |dNE| rather than dividing by it.
  2. What does each region of the plane mean, and how much of the cohort's
     time sits in each?
  3. Do PATIENTS differ in their region composition by more than the noise of
     having been observed for a finite number of hours?

    python code/30_response_regions.py --subset cardiogenic

Runs after 29, which writes the dose-change events this reads.
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
from clifcs.config import CONFIG_HOME
from clifcs.regions import (  # noqa: F401
    REGION_ORDER, between_patient_variance, by_starting_map, dose_bins,
    label_regions, patient_composition, persistence_test,
    transition_matrix, transitions,
)


def plot_regions(labelled, binned, paths, subset, logger, seed=42,
                 max_points=30000):
    """the plane with its regions marked, and the binned response beside it"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from clifcs.figures import save_figure
    from clifcs.palette import husl_colors

    colors = husl_colors(list(REGION_ORDER))
    rng = np.random.default_rng(seed)
    figure, (left, right) = plt.subplots(1, 2, figsize=(15.0, 6.6))

    shown = labelled
    if shown.height > max_points:
        keep = rng.choice(shown.height, size=max_points, replace=False)
        shown = shown[keep]
    for region in REGION_ORDER:
        part = shown.filter(pl.col("region") == region)
        if part.is_empty():
            continue
        share = (
            labelled.filter(pl.col("region") == region).height
            / max(labelled.height, 1)
        )
        left.scatter(
            part["d_ne"], part["d_map"], s=5, alpha=0.4, linewidths=0,
            color=colors[region], rasterized=True,
            label=f"{region} ({share:.0%})",
        )
    left.axhline(0, color="#555", lw=0.8)
    left.axvline(0, color="#555", lw=0.8)
    limit = float(np.nanpercentile(np.abs(labelled["d_ne"].to_numpy()), 99))
    span = float(np.nanpercentile(np.abs(labelled["d_map"].to_numpy()), 99))
    left.set_xlim(-limit, limit)
    left.set_ylim(-span, span)
    left.set_xlabel("change in norepinephrine over the previous hour "
                    "(mcg/kg/min)")
    left.set_ylabel("change in MAP over the NEXT hour (mmHg)")
    left.set_title("Regions of the plane", fontsize=11)
    left.legend(loc="lower right", fontsize=7.5, frameon=True, markerscale=2.5)

    for direction, marker in (("increase", "o"), ("decrease", "s")):
        part = binned.filter(pl.col("direction") == direction).sort(
            "dose_change_from"
        )
        if part.is_empty():
            continue
        centres = part["dose_change_from"].to_numpy()
        right.errorbar(
            centres, part["mean_d_map"],
            yerr=[
                part["mean_d_map"] - part["ci_low"],
                part["ci_high"] - part["mean_d_map"],
            ],
            marker=marker, capsize=3, lw=1.6,
            label=f"dose {direction}",
        )
    right.axhline(0, color="#555", lw=0.8)
    right.set_xscale("log")
    right.set_xlabel("size of the norepinephrine change (mcg/kg/min, bin floor)")
    right.set_ylabel("mean change in MAP over the next hour (mmHg)")
    right.set_title(
        "Does the response scale with the size of the change?\n"
        "binned instead of divided, so no denominator blows up",
        fontsize=11,
    )
    right.legend(fontsize=9, frameon=False)
    figure.suptitle(
        f"{subset} shock: dose-change regions and the binned MAP response",
        fontsize=12,
    )
    figure.tight_layout()
    out = (
        paths.output_home / "figures" / paths.model_key
        / f"response_regions_{subset}.png"
    )
    save_figure(figure, out, manuscript=True)
    logger.info(f"wrote {out}")


def plot_transitions(matrix, by_day, paths, subset, logger):
    """the transition matrix as a heatmap, and the region mix over the week"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from clifcs.figures import save_figure
    from clifcs.palette import husl_colors

    figure, (left, right) = plt.subplots(1, 2, figsize=(15.5, 6.4))
    grid = np.array([
        [row[target] for target in REGION_ORDER]
        for row in matrix.iter_rows(named=True)
    ])
    image = left.imshow(grid, cmap="magma_r", vmin=0, vmax=float(grid.max()))
    left.set_xticks(range(len(REGION_ORDER)))
    left.set_xticklabels(REGION_ORDER, rotation=40, ha="right", fontsize=8)
    left.set_yticks(range(len(REGION_ORDER)))
    left.set_yticklabels(
        [f"{r}  (n={n:,})"
         for r, n in zip(REGION_ORDER, matrix["n"].to_list())], fontsize=8
    )
    for i in range(len(REGION_ORDER)):
        for j in range(len(REGION_ORDER)):
            left.text(
                j, i, f"{grid[i, j]:.0%}", ha="center", va="center",
                fontsize=7,
                color="white" if grid[i, j] > grid.max() * 0.55 else "#222",
            )
    # the diagonal is the question: a box around it says where to look
    for i in range(len(REGION_ORDER)):
        left.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#1b9e77", lw=1.8
        ))
    left.set_xlabel("region at the NEXT titration")
    left.set_ylabel("region at this titration")
    left.set_title(
        "Where the next titration lands\nrows sum to 1; the outlined "
        "diagonal is staying put", fontsize=10,
    )
    figure.colorbar(image, ax=left, fraction=0.046, pad=0.03)

    colors = husl_colors(list(REGION_ORDER))
    days = sorted(by_day["day"].unique().to_list())
    bottom = np.zeros(len(days))
    for region in REGION_ORDER:
        shares = []
        for day in days:
            cell = by_day.filter(
                (pl.col("day") == day) & (pl.col("region") == region)
            )
            shares.append(float(cell["share"][0]) if cell.height else 0.0)
        shares = np.array(shares)
        right.bar(
            days, shares, bottom=bottom, width=0.84,
            color=colors[region], label=region,
        )
        bottom += shares
    right.set_xlabel("day since shock onset")
    right.set_ylabel("share of that day's titration-hours")
    right.set_ylim(0, 1)
    right.set_title(
        "Does the mix change over the week?", fontsize=10
    )
    right.legend(
        fontsize=7.5, frameon=False, ncol=2, loc="lower center"
    )
    figure.suptitle(
        f"{subset} shock: movement between dose-response regions", fontsize=12
    )
    figure.tight_layout()
    out = (
        paths.output_home / "figures" / paths.model_key
        / f"response_transitions_{subset}.png"
    )
    save_figure(figure, out, manuscript=True)
    logger.info(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "30_response_regions", paths.log_home / "30_response_regions.log"
    )
    cfg = OmegaConf.load(args.config or CONFIG_HOME / "dynamics.yaml")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    events_file = (
        paths.patient_level_home / f"dose_response_events_{args.subset}.parquet"
    )
    if not events_file.is_file():
        logger.error(f"no {events_file.name}; run 29 first")
        return
    events = pl.read_parquet(events_file)
    logger.info(
        f"{events.height:,} dose-change hours, "
        f"{events['hospitalization_id'].n_unique():,} patients"
    )

    # ---- 1. does the response scale with the size of the change
    binned = dose_bins(events, cfg)
    binned.write_csv(tables / f"response_dose_bins_{args.subset}.csv")
    logger.info("mean MAP response by size of the dose change:")
    for row in binned.iter_rows(named=True):
        top = (
            f"{row['dose_change_to']:.2f}" if row["dose_change_to"]
            else "+"
        )
        logger.info(
            f"    {row['direction']:<9} {row['dose_change_from']:.2f}-{top:<5} "
            f"{row['hours']:>7,} hours  mean dMAP "
            f"{row['mean_d_map']:>+6.2f} [{row['ci_low']:>+6.2f}, "
            f"{row['ci_high']:>+6.2f}]"
        )

    starting = by_starting_map(events, cfg)
    if not starting.is_empty():
        starting.write_csv(
            tables / f"response_by_starting_map_{args.subset}.csv"
        )
        logger.info("mean MAP response by the MAP the hour started at:")
        for row in starting.iter_rows(named=True):
            top = (
                f"{row['starting_map_to']:.0f}" if row["starting_map_to"]
                else "+"
            )
            logger.info(
                f"    MAP {row['starting_map_from']:.0f}-{top:<4} "
                f"{row['direction']:<9} {row['hours']:>7,} hours  "
                f"mean dMAP {row['mean_d_map']:>+6.2f} "
                f"[{row['ci_low']:>+6.2f}, {row['ci_high']:>+6.2f}]"
            )

    # ---- 2. the regions
    labelled = label_regions(events, cfg)
    labelled.write_parquet(
        paths.patient_level_home / f"response_regions_{args.subset}.parquet"
    )
    rows = []
    logger.info("share of dose-change hours in each region:")
    for region in REGION_ORDER:
        part = labelled.filter(pl.col("region") == region)
        rows.append({
            "region": region,
            "hours": part.height,
            "share": part.height / max(labelled.height, 1),
            "patients": part["hospitalization_id"].n_unique(),
            "mean_d_ne": float(part["d_ne"].mean()) if part.height else None,
            "mean_d_map": float(part["d_map"].mean()) if part.height else None,
        })
        logger.info(
            f"    {region:<20} {part.height:>7,} hours "
            f"{part.height / max(labelled.height, 1):>6.1%}   mean dNE "
            f"{rows[-1]['mean_d_ne'] or float('nan'):>+7.4f}  mean dMAP "
            f"{rows[-1]['mean_d_map'] or float('nan'):>+6.2f}"
        )
    pl.DataFrame(rows).write_csv(
        tables / f"response_region_shares_{args.subset}.csv"
    )

    # ---- 3. do patients differ
    shares = patient_composition(labelled, cfg, logger)
    if shares.height < 200:
        logger.warning("too few patients clear the hour floor")
        return
    shares.write_parquet(
        paths.patient_level_home
        / f"response_region_shares_{args.subset}.parquet"
    )
    variance = between_patient_variance(shares, logger, seed=args.seed)
    variance.write_csv(
        tables / f"response_between_patient_{args.subset}.csv"
    )
    any_real = variance.filter(pl.col("p_exceeds_null") < 0.05)
    logger.info(
        f"  {any_real.height} of {variance.height} regions show "
        "between-patient variation beyond the null"
        + (
            ": " + ", ".join(any_real["region"].to_list())
            if any_real.height else " -- none"
        )
    )

    # ---- 4. do patients move between regions
    logger.info("transitions between regions:")
    pairs = transitions(labelled, logger)
    matrix = transition_matrix(pairs)
    matrix.write_csv(tables / f"response_transitions_{args.subset}.csv")
    logger.info("  row-normalised transition matrix (Dirichlet alpha=1):")
    for row in matrix.iter_rows(named=True):
        best = max(REGION_ORDER, key=lambda r: row[r])
        logger.info(
            f"    from {row['from_region']:<20} n={row['n']:>7,}  "
            f"stays {row[row['from_region']]:>5.1%}  "
            f"most likely next: {best} ({row[best]:.1%})"
        )

    observed, null_mean, null_sd, p = persistence_test(
        pairs, logger, seed=args.seed
    )
    pl.DataFrame([{
        "pairs": pairs.height,
        "observed_stay_rate": observed,
        "within_patient_shuffle_mean": null_mean,
        "within_patient_shuffle_sd": null_sd,
        "excess": observed - null_mean,
        "p_exceeds_null": p,
    }]).write_csv(tables / f"response_persistence_{args.subset}.csv")

    # how much of the plane each patient actually visits
    visited = (
        labelled.group_by("hospitalization_id")
        .agg(regions=pl.col("region").n_unique(), hours=pl.len())
        .filter(pl.col("hours") >= int(cfg.regions.min_hours_per_patient))
    )
    logger.info(
        f"  distinct regions visited, among patients with at least "
        f"{int(cfg.regions.min_hours_per_patient)} hours: median "
        f"{visited['regions'].median():.0f} of {len(REGION_ORDER)}, "
        f"{(visited['regions'] <= 2).mean():.1%} visit two or fewer, "
        f"{(visited['regions'] >= 6).mean():.1%} visit six or more"
    )

    # ---- 5. does the mix change over the week
    by_day = (
        labelled.with_columns(day=(pl.col("offset_hours") // 24).cast(pl.Int64))
        .group_by("day", "region").len()
        .sort("day")
    )
    totals = by_day.group_by("day").agg(total=pl.col("len").sum())
    by_day = by_day.join(totals, on="day").with_columns(
        share=pl.col("len") / pl.col("total")
    )
    by_day.write_csv(tables / f"response_regions_by_day_{args.subset}.csv")
    logger.info("region share by day since onset:")
    for day in sorted(by_day["day"].unique().to_list())[:7]:
        part = by_day.filter(pl.col("day") == day)
        pieces = ", ".join(
            f"{r['region'].split()[0]} {r['share']:.0%}"
            for r in part.sort("share", descending=True).head(4).iter_rows(
                named=True
            )
        )
        logger.info(
            f"    day {day}: {int(part['total'][0]):>6,} hours   {pieces}"
        )
    plot_transitions(matrix, by_day, paths, args.subset, logger)

    plot_regions(labelled, binned, paths, args.subset, logger, seed=args.seed)
    logger.info("30 done")


if __name__ == "__main__":
    main()

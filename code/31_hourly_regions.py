#!/usr/bin/env python3

"""
31_hourly_regions.py -- the dose-response plane as HOURLY states, and
transitions that cannot share a measurement

Supersedes the transition analysis in 30, which was built on dose-change
events and was wrong for two reasons this script fixes:

  SPACING. An "event" was an hour in which the norepinephrine moved, so 85% of
  hours were dropped and a transition spanned whatever gap separated two
  titrations -- 46% at one hour, 10% more than six hours apart. Here every hour
  carries a state, including holds and hours off pressors, so a transition
  always means the same thing.

  OVERLAP. A state at hour t is built from map(t) and map(t+1); a state at t+1
  from map(t+1) and map(t+2). Sharing map(t+1) makes one low reading look like
  deterioration followed by recovery. The measured correlation between
  consecutive dMAP values was -0.401 at the adjacent lag against -0.011 at
  non-adjacent ones. Transitions here are taken at lag 2, where the windows are
  disjoint on both axes.

    python code/31_hourly_regions.py --subset cardiogenic

Run after 29, which writes the hourly per-agent dose and MAP grid.
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
from clifcs.regions import (
    HOURLY_REGION_ORDER, label_hourly_regions, lagged_transitions,
)

#: regimen -> colour key, used instead of a time gradient. Hours since onset
#: showed no gradient in the first version of this figure, so the colormap was
#: spending the reader's attention on a variable that carried nothing.
REGIMENS = (
    ("norepinephrine alone", None),
    ("norepinephrine + vasopressin", "vasopressin"),
    ("norepinephrine + epinephrine", "epinephrine"),
    ("norepinephrine + phenylephrine", "phenylephrine"),
)


def regimen_of(frame, companion):
    if companion is None:
        return frame.filter(
            ~pl.col("on_vasopressin_h") & ~pl.col("on_epinephrine_h")
            & ~pl.col("on_phenylephrine_h")
        )
    return frame.filter(pl.col(f"on_{companion}_h"))


def region_boxes(axes, cfg, limit, span, order):
    """
    the region definitions drawn as transparent blocks behind the points

    The boundaries are read from the config rather than hard-coded, so a block
    cannot drift out of step with the labels: if `big_mmhg` changes, the block
    moves with the rule.

    Blocks are drawn with `zorder` below the scatter and no edge, so they read
    as background. The diagonal "highly responsive" cell is the one that is not
    a simple rectangle -- it spans the whole left half plus the sliver of the
    right half below `minimal_increase` -- so it is drawn as two blocks sharing
    one colour and one legend entry.
    """
    import matplotlib.patches as patches

    from clifcs.palette import husl_colors

    colors = husl_colors(list(order))
    deadband = float(cfg.response.deadband_mmhg)
    big = float(cfg.regions.big_mmhg)
    minimal = float(cfg.regions.minimal_increase)
    floor = float(cfg.action.absolute_minimum)

    #: (region, x0, x1, y0, y1) in data coordinates
    blocks = [
        ("highly responsive", -limit, minimal, big, span),
        ("responsive", floor, limit, deadband, big),
        ("nonresponsive", floor, limit, -deadband, deadband),
        ("deteriorating", floor, limit, -span, -deadband),
        ("wean held", -limit, -floor, -deadband, big),
        ("wean with a dip", -limit, -floor, -big, -deadband),
        ("failed wean", -limit, -floor, -span, -big),
        ("stable hold", -floor, floor, -deadband, span),
        ("drifting hold", -floor, floor, -span, -deadband),
    ]
    drawn = set()
    for region, x0, x1, y0, y1 in blocks:
        if region not in colors:
            continue
        axes.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=colors[region], alpha=0.17, edgecolor="none",
            zorder=0, label=region if region not in drawn else None,
        ))
        drawn.add(region)
    # the second half of the highly-responsive cell: a big rise bought with a
    # dose DECREASE is the same clinical reading as one bought with no increase
    axes.add_patch(patches.Rectangle(
        (-limit, big), limit - (-limit), span - big,
        facecolor=colors["highly responsive"], alpha=0.17, edgecolor="none",
        zorder=0,
    ))
    for y in (-big, -deadband, deadband, big):
        axes.axhline(y, color="#9a9a9a", lw=0.6, ls=":", zorder=1)
    for x in (-floor, floor, minimal):
        axes.axvline(x, color="#9a9a9a", lw=0.6, ls=":", zorder=1)
    return colors


def plot_plane(labelled, cfg, paths, subset, logger, seed=42,
               max_points=20000):
    """
    four panels, one per regimen, over the region blocks

    POINTS ARE COLOURED BY REGIMEN, one colour per panel, replacing the
    hours-since-onset gradient. That gradient showed nothing -- no visible
    structure in either cohort -- so it was spending the reader's attention on
    a variable carrying no signal while the region boundaries, which do the
    actual work, were invisible.

    Only hours in which the dose MOVED are plotted. Holds sit at dNE = 0 by
    definition and would pile into a single vertical line through the middle of
    every panel, hiding the blocks behind it; their share is given in the panel
    text instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from clifcs.figures import save_figure
    from clifcs.palette import husl_colors

    regimen_colors = husl_colors([label for label, _ in REGIMENS])
    rng = np.random.default_rng(seed)
    moved = labelled.filter(
        pl.col("d_ne").abs() >= float(cfg.action.absolute_minimum)
    )
    limit = float(np.nanpercentile(np.abs(moved["d_ne"].to_numpy()), 99))
    span = float(np.nanpercentile(np.abs(moved["d_map"].to_numpy()), 99))

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 12.0), sharex=True,
                                sharey=True)
    for panel, (label, companion) in zip(axes.ravel(), REGIMENS):
        part = regimen_of(moved, companion)
        colors = region_boxes(panel, cfg, limit, span, HOURLY_REGION_ORDER)
        if part.is_empty():
            panel.set_title(f"{label}  (no hours)", fontsize=10)
            continue
        shown = part
        if shown.height > max_points:
            keep = rng.choice(shown.height, size=max_points, replace=False)
            shown = shown[keep]
        panel.scatter(
            shown["d_ne"], shown["d_map"], s=4.5, alpha=0.40, linewidths=0,
            color=regimen_colors[label], rasterized=True, zorder=3,
        )
        held = regimen_of(labelled, companion).filter(
            pl.col("region").is_in(["stable hold", "drifting hold"])
        ).height
        total = regimen_of(labelled, companion).height
        panel.set_title(label, fontsize=11,
                        color=regimen_colors[label])
        panel.text(
            0.015, 0.985,
            f"{part.height:,} hours with a dose change\n"
            f"{held:,} hold hours not plotted "
            f"({held / max(total, 1):.0%} of this regimen)",
            transform=panel.transAxes, va="top", ha="left", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none",
                  "pad": 3.0}, zorder=5,
        )
        panel.set_xlim(-limit, limit)
        panel.set_ylim(-span, span)
    for panel in axes[-1]:
        panel.set_xlabel("change in norepinephrine over the previous hour "
                         "(mcg/kg/min)")
    for panel in axes[:, 0]:
        panel.set_ylabel("change in MAP over the next hour (mmHg)")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels_, loc="lower center", ncol=5, fontsize=9,
        frameon=False, title="region (shaded background)",
    )
    figure.suptitle(
        f"{subset} shock: hourly dose-response states\n"
        "shaded blocks are the region definitions; points coloured by regimen",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    out = (
        paths.output_home / "figures" / paths.model_key
        / f"hourly_regions_plane_{subset}.png"
    )
    save_figure(figure, out, manuscript=True)
    logger.info(f"wrote {out}")


def plot_state_hexmaps(labelled, paths, subset, logger, gridsize=34,
                       min_count=10):
    """
    the projection tiled by which hourly state owns each hexagon

    JOINED EXACTLY, hour to grid point. The embedding grid runs -24h to +48h in
    six-hour steps, so only the nine non-negative grid points can carry a state
    and each patient contributes at most nine of their ~170 hours. Taking a
    plurality over the surrounding six hours would use more data, but it would
    also answer a different question -- "what mostly happened near here" rather
    than "what was happening at the moment this embedding was taken".

    TWO SCOPES, because "off pressors" is 78.5% of hours and wins the plurality
    map outright. The on-support variant drops it from the competition so the
    question becomes which SUPPORTED state owns a region, the same move the BPE
    word maps needed for the same reason.
    """
    import importlib.util

    from clifcs.palette import husl_colors

    spec = importlib.util.spec_from_file_location(
        "traj", pathlib.Path(__file__).resolve().parent / "09_trajectories.py"
    )
    traj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(traj)

    candidates = sorted(paths.patient_level_home.glob(
        f"trajectory_projections_{subset}_corrected-*.parquet"
    ))
    if not candidates:
        logger.warning(f"no corrected projection for {subset}; hexmaps skipped")
        return
    grid = pl.read_parquet(candidates[0]).select(
        "hospitalization_id", "offset_hours", "pacmap_1", "pacmap_2"
    )
    joined = grid.join(
        labelled.select("hospitalization_id", "offset_hours", "region"),
        on=["hospitalization_id", "offset_hours"], how="inner",
    )
    if joined.is_empty():
        logger.warning("no grid point matched an hourly state")
        return
    logger.info(
        f"  {joined.height:,} grid points carry an hourly state "
        f"({joined.height / max(grid.height, 1):.0%} of the projection grid, "
        f"{joined['hospitalization_id'].n_unique():,} patients)"
    )
    for row in joined["region"].value_counts(sort=True).iter_rows(named=True):
        logger.info(
            f"    {row['region']:<20} {row['count']:>7,} "
            f"{row['count'] / joined.height:>6.1%}"
        )

    colors = husl_colors(list(HOURLY_REGION_ORDER))
    on_support = [r for r in HOURLY_REGION_ORDER if r != "off pressors"]
    figures = paths.output_home / "figures" / paths.model_key
    note_base = (
        "hourly MAP/pressor response states at the nine grid points from "
        "onset to +48h"
    )
    dropped = (
        "  |  'off pressors' (78% of hours) dropped, so each hexagon is "
        "coloured by which SUPPORTED state owns it"
    )

    # FOUR FIGURES PER SCOPE: hexagons and small multiples, each with and
    # without the displacement field. The flow variants answer a different
    # question from the static ones -- not "which state is here" but "where do
    # patients in this region go next" -- and the arrows cost legibility, so
    # both are drawn rather than one being chosen for the reader.
    for scope, levels in (("", list(HOURLY_REGION_ORDER)),
                          ("_on_support", on_support)):
        note = note_base + (dropped if scope else "")
        scoped = (
            joined if not scope
            else joined.filter(pl.col("region") != "off pressors")
        )
        for mode in ("plurality", "enrichment"):
            for flow, tag in ((False, ""), (True, "_flow")):
                traj.plot_state_hexbin(
                    joined, "pacmap_1", "pacmap_2", "region", colors, levels,
                    figures
                    / f"hourly_state_hex_{mode}{scope}{tag}_{subset}.png",
                    f"{subset} shock: hourly response state by hexagon, "
                    f"{mode}" + (", on support only" if scope else "")
                    + (" (with flow)" if flow else ""),
                    note, logger, mode=mode, gridsize=gridsize,
                    min_count=min_count, flow=flow,
                    step_hours=6.0 if flow else None,
                )
        for flow, tag in ((None, ""), (6.0, "_flow")):
            traj.plot_state_small_multiples(
                scoped, "pacmap_1", "pacmap_2", "region", colors, levels,
                figures / f"hourly_state_panels{scope}{tag}_{subset}.png",
                f"{subset} shock: each hourly response state on its own"
                + (", on support only" if scope else "")
                + (" (with flow)" if flow else ""),
                logger, step_hours=flow,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--lag-hours", type=int, default=2,
                        help="2 is the minimum that avoids a shared MAP reading")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "31_hourly_regions", paths.log_home / "31_hourly_regions.log"
    )
    cfg = OmegaConf.load(args.config or CONFIG_HOME / "dynamics.yaml")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    grid_file = (
        paths.patient_level_home / f"hourly_agents_{args.subset}.parquet"
    )
    if not grid_file.is_file():
        logger.error(
            f"no {grid_file.name}; re-run 29, which now persists the full "
            "per-agent grid rather than only the dose-change hours"
        )
        return
    # THE FULL GRID, not 29's event table. The event table is already filtered
    # to hours in which the dose moved, so labelling it could never produce a
    # hold state -- 85% of hours would be missing and the chain could not
    # represent a patient sitting still.
    grid = pl.read_parquet(grid_file)
    agent_columns = [c for c in grid.columns if c.startswith("dose_")]
    logger.info(
        f"{grid.height:,} patient-hours on the full grid; per-agent columns: "
        f"{', '.join(agent_columns)}"
    )

    labelled = label_hourly_regions(grid, cfg).with_columns([
        (pl.col(f"dose_{agent}") > 0).alias(f"on_{agent}_h")
        for agent in ("vasopressin", "epinephrine", "phenylephrine")
    ])
    logger.info(
        f"{labelled.height:,} labelled patient-hours, "
        f"{labelled['hospitalization_id'].n_unique():,} patients"
    )
    labelled.write_parquet(
        paths.patient_level_home / f"hourly_regions_{args.subset}.parquet"
    )

    rows = []
    for region in HOURLY_REGION_ORDER:
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
            f"    {region:<20} {part.height:>8,} hours "
            f"{part.height / max(labelled.height, 1):>6.1%}"
        )
    pl.DataFrame(rows).write_csv(
        tables / f"hourly_region_shares_{args.subset}.csv"
    )

    logger.info(f"transitions at lag {args.lag_hours}h:")
    pairs = lagged_transitions(labelled, logger, lag_hours=args.lag_hours)
    matrix_rows = []
    for source in HOURLY_REGION_ORDER:
        part = pairs.filter(pl.col("region") == source)
        total = part.height
        row = {"from_region": source, "n": total}
        for target in HOURLY_REGION_ORDER:
            row[target] = (
                part.filter(pl.col("next_region") == target).height + 1.0
            ) / (total + len(HOURLY_REGION_ORDER))
        matrix_rows.append(row)
    matrix = pl.DataFrame(matrix_rows)
    matrix.write_csv(tables / f"hourly_transitions_{args.subset}.csv")
    logger.info("  most likely next state, and whether it stays:")
    for row in matrix.iter_rows(named=True):
        best = max(HOURLY_REGION_ORDER, key=lambda r: row[r])
        logger.info(
            f"    from {row['from_region']:<20} n={row['n']:>8,}  "
            f"stays {row[row['from_region']]:>5.1%}  "
            f"-> {best} ({row[best]:.1%})"
        )

    # persistence against a within-patient shuffle, on the hourly encoding
    rng = np.random.default_rng(args.seed)
    observed = float((pairs["region"] == pairs["next_region"]).mean())
    paths_list = (
        labelled.sort(["hospitalization_id", "offset_hours"])
        .group_by("hospitalization_id", maintain_order=True)
        .agg(path=pl.col("region"))
    )["path"].to_list()
    null = np.empty(300)
    for draw in range(300):
        same = total = 0
        for path in paths_list:
            if len(path) <= args.lag_hours:
                continue
            shuffled = rng.permutation(np.asarray(path, dtype=object))
            same += int(
                (shuffled[:-args.lag_hours] == shuffled[args.lag_hours:]).sum()
            )
            total += len(shuffled) - args.lag_hours
        null[draw] = same / max(total, 1)
    p = float((null >= observed).mean())
    logger.info(
        f"  stays in the same state at lag {args.lag_hours}h: observed "
        f"{observed:.1%}, within-patient shuffle null {null.mean():.1%} "
        f"+/- {null.std():.1%}, p={p:.4f}"
    )
    pl.DataFrame([{
        "lag_hours": args.lag_hours, "pairs": pairs.height,
        "observed_stay_rate": observed,
        "shuffle_null_mean": float(null.mean()),
        "shuffle_null_sd": float(null.std()),
        "excess": observed - float(null.mean()),
        "p_exceeds_null": p,
    }]).write_csv(tables / f"hourly_persistence_{args.subset}.csv")

    plot_plane(labelled, cfg, paths, args.subset, logger, seed=args.seed)
    logger.info("hourly state hexagon maps:")
    plot_state_hexmaps(labelled, paths, args.subset, logger)
    logger.info("31 done")


if __name__ == "__main__":
    main()

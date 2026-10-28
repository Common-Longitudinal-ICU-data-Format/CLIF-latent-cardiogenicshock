#!/usr/bin/env python3

"""
38_layer_hexmaps.py -- both state layers on the projection

Treatment (what the clinician is doing) and response (what the patient is
doing), each tiled into hexagons on the same projection, so the two can be read
against one another.

    python code/38_layer_hexmaps.py --subset cardiogenic

THE PALETTES DIFFER ON PURPOSE AND THE DIFFERENCE IS NOT DECORATIVE. Response
states are NOMINAL -- "deteriorating" is not more or less than "wean held", it
is a different thing -- so they take the evenly spaced husl hues this project
uses for states throughout. Treatment cells are ORDINAL: none < low < mid <
high, with agents layered on. An ordinal variable given a categorical palette
invites a reader to see sixteen unrelated groups where there is one intensity
axis, so treatment takes a sequential ramp ordered by intensity and colour
carries "how much support" directly.

THE GRID IS WHATEVER WAS EXTRACTED. At six-hour spacing a patient contributes
seven points over 0-48h; at two-hour spacing, twenty-five. Every filename and
subtitle carries the spacing, because a hexagon's count means something
different in each.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.regions import HOURLY_REGION_ORDER
from clifcs.treatment import (
    TREATMENT_BANDS, TREATMENT_SYMBOLS, add_treatment_symbol,
    treatment_label,
)


def treatment_palette():
    """
    hue for WHICH AGENTS, shade for HOW MUCH norepinephrine

    The treatment cell is a factorial variable -- four norepinephrine bands
    crossed with the presence of vasopressin and of any other vasoactive -- and
    a one-dimensional ramp cannot show a two-dimensional thing. The previous
    version ordered all sixteen cells on a single `rocket_r` ramp by intensity,
    which made "NE mid" and "NE low +vaso +other" neighbouring shades of the
    same colour: a reader could see how much support a region was on and could
    not see what it was made of.

    So the encoding is split to match the variable. The AGENT SET picks the hue
    family -- norepinephrine alone, plus vasopressin, plus another agent, or
    all three -- and the NOREPINEPHRINE BAND picks the shade within it, dark to
    light from high dose to none. `tab20c` is built for exactly this: five hue
    families of four graded shades each, in that order. Four of the five are
    used and the grey family is left for "under the count floor" elsewhere.

    The ordinal reading survives WITHIN a hue and is gone ACROSS hues, which is
    correct -- adding vasopressin is not a step on the norepinephrine scale.
    """
    import matplotlib.pyplot as plt

    # tab20c runs blues, oranges, greens, purples, greys; each family is four
    # shades ordered dark to light.
    families = plt.get_cmap("tab20c").colors
    by_agents = {
        (False, False): families[0:4],     # norepinephrine alone
        (True, False): families[4:8],      # + vasopressin
        (False, True): families[8:12],     # + another agent
        (True, True): families[12:16],     # + both
    }
    colors, order = {}, []
    for agents in ((False, False), (True, False), (False, True), (True, True)):
        vaso, other = agents
        shades = by_agents[agents]
        # TREATMENT_BANDS is none, low, mid, high and the shades run dark to
        # light, so the band list is reversed: high dose takes the darkest.
        for band, rgb in zip(reversed(TREATMENT_BANDS), shades):
            label = (
                f"NE {band}"
                + (" +vaso" if vaso else "")
                + (" +other" if other else "")
            )
            colors[label] = "#%02x%02x%02x" % tuple(int(255 * v) for v in rgb)
            order.append(label)
    # Listed by agent set, then by descending dose, so the legend reads as the
    # two factors it is rather than as sixteen unrelated categories.
    return colors, order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--gridsize", type=int, default=34)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "38_layer_hexmaps", paths.log_home / "38_layer_hexmaps.log"
    )
    figures = paths.output_home / "figures" / paths.model_key

    spec = importlib.util.spec_from_file_location(
        "traj", pathlib.Path(__file__).resolve().parent / "09_trajectories.py"
    )
    traj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(traj)

    candidates = sorted(paths.patient_level_home.glob(
        f"trajectory_projections_{args.subset}_corrected-*.parquet"
    ))
    if not candidates:
        logger.error(f"no corrected projection for {args.subset}")
        return
    grid = pl.read_parquet(candidates[0]).select(
        "hospitalization_id", "offset_hours", "pacmap_1", "pacmap_2"
    )
    offsets = sorted(grid["offset_hours"].unique().to_list())
    spacing = (
        min(b - a for a, b in zip(offsets, offsets[1:]))
        if len(offsets) > 1 else 6.0
    )
    logger.info(
        f"projection carries {len(offsets)} offsets at {spacing:.0f}h spacing, "
        f"{offsets[0]:.0f} to {offsets[-1]:.0f}h"
    )

    states = add_treatment_symbol(pl.read_parquet(
        paths.patient_level_home / f"treatment_states_{args.subset}.parquet"
    ))
    joined = grid.join(
        states.select("hospitalization_id", "offset_hours", "region",
                      "treatment"),
        on=["hospitalization_id", "offset_hours"], how="inner",
    )
    patients = max(joined["hospitalization_id"].n_unique(), 1)
    per_patient = joined.height / patients
    logger.info(
        f"  {joined.height:,} grid points carry both layers, "
        f"{patients:,} patients ({per_patient:.1f} points each)"
    )
    tag = f"{spacing:.0f}h"

    from clifcs.palette import husl_colors

    treatment_colors, treatment_order = treatment_palette()
    layers = {
        "treatment": (
            treatment_colors, treatment_order, "treatment",
            "norepinephrine band crossed with the other agents, on a ramp "
            "ordered by intensity",
        ),
        "response": (
            husl_colors(list(HOURLY_REGION_ORDER)),
            list(HOURLY_REGION_ORDER), "region",
            "the action taken and the MAP response that followed it",
        ),
    }
    for layer, (colors, order, column, note) in layers.items():
        present = [
            level for level in order if (joined[column] == level).any()
        ]
        logger.info(
            f"  {layer}: {len(present)} of {len(order)} levels present"
        )
        subtitle = (
            f"{note}  |  grid at {spacing:.0f}h spacing, "
            f"{per_patient:.0f} points per patient"
        )
        for mode in ("plurality", "enrichment"):
            for flow, suffix in ((False, ""), (True, "_flow")):
                traj.plot_state_hexbin(
                    joined, "pacmap_1", "pacmap_2", column, colors, present,
                    figures / (
                        f"layer_{layer}_hex_{mode}{suffix}_{tag}_"
                        f"{args.subset}.png"
                    ),
                    f"{args.subset} shock: {layer} layer by hexagon, {mode}"
                    + (" (with flow)" if flow else ""),
                    subtitle, logger, mode=mode, gridsize=args.gridsize,
                    min_count=args.min_count, flow=flow,
                    step_hours=spacing if flow else None,
                )
        traj.plot_state_small_multiples(
            joined, "pacmap_1", "pacmap_2", column, colors, present,
            figures / f"layer_{layer}_panels_{tag}_{args.subset}.png",
            f"{args.subset} shock: each {layer} level on its own",
            logger, step_hours=spacing,
        )
    logger.info("38 done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
shared figure plumbing
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt

from clifcs.palette import GRID, INK_PRIMARY, INK_SECONDARY, SURFACE

DPI = 500


def style() -> None:
    """recessive axes and grid; text in ink tokens, never in a series colour"""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE["light"],
            "axes.facecolor": SURFACE["light"],
            "savefig.facecolor": SURFACE["light"],
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "font.size": 8,
            "legend.frameon": False,
            "figure.dpi": 110,
        }
    )


def save_figure(figure, out_path: pathlib.Path, attempts: int = 6, **kwargs) -> None:
    """
    save a figure, retrying if the destination is locked

    output_home normally sits on a synced drive (Box, OneDrive, Dropbox). When
    the sync client is mid-upload on a file being overwritten, the open fails
    with EINVAL rather than blocking, which kills a run several minutes in and
    leaves half the figures regenerated. Writing to a temporary name first
    keeps the existing file intact if every attempt fails.
    """
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staged = out_path.with_suffix(".tmp.png")
    figure.savefig(staged, dpi=kwargs.pop("dpi", DPI),
                   bbox_inches=kwargs.pop("bbox_inches", "tight"), **kwargs)

    for attempt in range(attempts):
        try:
            staged.replace(out_path)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * 2**attempt)


def quantile_colour(values):
    """
    map values to their quantile rank, with ticks labelled by actual value

    Colouring a right-skewed variable by magnitude wastes almost the whole
    ramp: ICU length of stay has a median near 4 days and a tail past 50, so
    every typical patient lands in the first few percent of the scale and the
    panel reads as uniformly dark. Ranking first spreads the cloud across the
    full ramp; the colourbar is then labelled with the value at each quartile,
    so the reader still sees real units.

    Returns (ranks, finite mask, tick positions, tick labels).
    """
    import numpy as np
    from scipy.stats import rankdata

    finite = np.isfinite(values)
    ranks = np.full(values.shape, np.nan)
    if finite.sum() == 0:
        return ranks, finite, [], []
    ranks[finite] = rankdata(values[finite], method="average") / finite.sum()
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = [f"{np.nanquantile(values[finite], q):.3g}" for q in quantiles]
    return ranks, finite, quantiles, labels


def key_string(value) -> str:
    """
    canonical string form of a categorical level

    polars renders booleans as "true"/"false" while Python's str() gives
    "True"/"False". A colour dict keyed one way and looked up the other
    silently returns the fallback grey for every row, which is how the
    mortality and ventilation panels came out blank.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower()

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

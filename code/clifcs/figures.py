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


def save_figure(figure, out_path: pathlib.Path, attempts: int = 6,
                manuscript: bool = True, **kwargs) -> None:
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
    if manuscript:
        _save_manuscript(figure, out_path)

    for attempt in range(attempts):
        try:
            staged.replace(out_path)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * 2**attempt)


def save_table(frame, out_path: pathlib.Path, attempts: int = 6) -> None:
    """
    write a CSV, retrying if the destination is locked or transiently absent

    The same hazard `save_figure` guards against, on the other kind of
    output. A sync client mid-operation can make an EXISTING directory report
    itself as absent for a second or two, not merely lock a file: a run here
    died with FileNotFoundError on output/tables/GEM-all while the directory
    was present before and after. So the parent is recreated on each attempt
    rather than once, and the write is staged so a failure leaves the previous
    table intact rather than truncated.
    """
    out_path = pathlib.Path(out_path)
    staged = out_path.with_suffix(".tmp.csv")
    for attempt in range(attempts):
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_csv(staged)
            staged.replace(out_path)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * 2**attempt)


#: resolution for the manuscript variants. 500 is above most journals' 300 dpi
#: floor with room to be cropped or scaled down without resampling artefacts.
MANUSCRIPT_DPI = 500


def _save_manuscript(figure, out_path: pathlib.Path) -> None:
    """
    the same figure stripped for publication: no axis furniture, transparent

    Hides the axis decoration rather than deleting it, then puts it back, so
    the figure object is unchanged for whatever the caller does next -- several
    callers save and then keep drawing.

    Legends are deliberately kept. They sit on their own axes here, and a
    reader cannot reconstruct which colour was which phenotype from the caption
    alone; axis labels and titles are recoverable from the caption, a palette
    is not.
    """
    import matplotlib.pyplot as plt  # noqa: F401  (ensures a backend is set)

    target = out_path.parent / "manuscript" / out_path.name
    target.parent.mkdir(parents=True, exist_ok=True)

    state = []
    for ax in figure.axes:
        state.append((ax, ax.axison, ax.get_title(), ax.get_xlabel(),
                      ax.get_ylabel()))
        ax.set_axis_off()
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
    suptitle = figure._suptitle
    suptitle_text = suptitle.get_text() if suptitle is not None else None
    if suptitle is not None:
        suptitle.set_visible(False)
    # figure-level annotations (the explanatory footers) go too
    texts = [(t, t.get_visible()) for t in figure.texts]
    for text, _ in texts:
        text.set_visible(False)

    try:
        figure.savefig(target, dpi=MANUSCRIPT_DPI, bbox_inches="tight",
                       transparent=True)
    except OSError:
        pass  # a locked manuscript copy must not kill the run
    finally:
        for ax, axison, title, xlabel, ylabel in state:
            if axison:
                ax.set_axis_on()
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        if suptitle is not None:
            suptitle.set_visible(True)
            suptitle.set_text(suptitle_text)
        for text, visible in texts:
            text.set_visible(visible)


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

#!/usr/bin/env python3

"""
small-cell suppression for the aggregate tables a site sends

A count of 2 in a `seymour_phenotype x shock_type x site` cell is closer to
naming a patient than to being a statistic, and the breakdown tables this
consortium asked for are exactly the ones that produce such cells: four Seymour
groups times four Bhavani groups times six shock types times however many sites
is a lot of cells over a cohort that is not that large once it is divided.

Three things have to happen, and doing only the first is the usual mistake.

1. **Primary suppression** -- blank any cell holding 1 to `min_cell - 1`.

2. **Complementary suppression** -- if exactly one cell in a row is blanked and
   the row total is published, subtracting gives it straight back. So blanking
   one cell forces blanking a second: the next-smallest, which loses the least.

3. **Derived statistics inherit it** -- a mortality of 1.000 on n = 3 names
   three patients' outcomes whether or not the 3 is printed. Any rate, median or
   mean computed over a suppressed cell is suppressed with it.

`min_cell = 11` is the default because that is the threshold CMS applies to
beneficiary cell sizes and the one most consortium DUAs have been written
against; it is configurable because the right number is a policy decision and
this module should not be the thing that encodes it.

Suppressed cells become null rather than zero. A zero is a claim that nobody was
in the cell, which is a different and sometimes very interesting statement --
"no patient at this site was assigned Bhavani D" is a data-quality finding, and
conflating it with "between one and ten were" would hide it.
"""

from __future__ import annotations

import polars as pl

#: CMS cell-size policy, and what most consortium DUAs are written against
DEFAULT_MIN_CELL = 11

#: columns that are a count of patients, by this pipeline's naming convention
COUNT_PREFIX = "n"


def count_columns(frame: pl.DataFrame) -> list[str]:
    """the integer columns that hold a patient count"""
    return [
        c
        for c in frame.columns
        if (c == COUNT_PREFIX or c.startswith("n_"))
        and frame.schema[c]
        in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]


def suppress_counts(
    frame: pl.DataFrame,
    *,
    min_cell: int = DEFAULT_MIN_CELL,
    derived: tuple[str, ...] = (),
    by: tuple[str, ...] = (),
    logger=None,
) -> pl.DataFrame:
    """
    blank every cell a reader could resolve to fewer than `min_cell` patients

    `derived` names the statistics computed over the count -- mortality,
    median LOS -- which are suppressed wherever their count is. `by` names the
    grouping whose row totals make complementary suppression necessary; pass the
    columns that are *held constant* across a row of the published table, e.g.
    `by=("site",)` for a per-site phenotype breakdown.

    Returns a new frame; does not mutate.
    """
    counts = count_columns(frame)
    if not counts:
        return frame

    out = frame
    n_primary = 0
    n_complementary = 0

    for column in counts:
        small = (pl.col(column) > 0) & (pl.col(column) < min_cell)
        n_primary += int(out.select(small.sum()).item() or 0)
        out = out.with_columns(
            pl.when(small).then(None).otherwise(pl.col(column)).alias(column)
        )

        # -------------------------------------------------- complementary pass
        # One blanked cell in a row whose total is published is recoverable by
        # subtraction, so blank the next-smallest survivor too. Done per `by`
        # group, which is the unit a published row corresponds to.
        if not by:
            continue
        suppressed_per_group = (
            out.group_by(by)
            .agg(
                n_suppressed=pl.col(column).is_null().sum(),
                n_cells=pl.len(),
            )
            .filter(
                (pl.col("n_suppressed") == 1) & (pl.col("n_cells") > 2)
            )
            .select(by)
        )
        if suppressed_per_group.height == 0:
            continue

        # within each exposed group, the smallest surviving cell
        exposed = out.join(suppressed_per_group, on=list(by), how="semi")
        victims = (
            exposed.filter(pl.col(column).is_not_null())
            .sort(column)
            .group_by(by, maintain_order=True)
            .head(1)
            .select(*by, **{f"_victim_{column}": pl.col(column)})
        )
        if victims.height == 0:
            continue
        n_complementary += victims.height
        out = (
            out.join(victims, on=list(by), how="left")
            .with_columns(
                pl.when(pl.col(column) == pl.col(f"_victim_{column}"))
                .then(None)
                .otherwise(pl.col(column))
                .alias(column)
            )
            .drop(f"_victim_{column}")
        )

    # ------------------------------------------------------- derived statistics
    # Keyed to the primary count `n` when present, else to the first count
    # column -- a mortality rate is a rate over one denominator, and that is
    # the one it was computed on.
    anchor = COUNT_PREFIX if COUNT_PREFIX in counts else counts[0]
    present_derived = [c for c in derived if c in out.columns]
    if present_derived:
        out = out.with_columns(
            [
                pl.when(pl.col(anchor).is_null())
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in present_derived
            ]
        )

    if logger is not None and (n_primary or n_complementary):
        logger.info(
            f"  suppressed {n_primary} cell(s) below {min_cell}"
            + (
                f" and {n_complementary} complementary cell(s) that subtraction "
                "would have recovered"
                if n_complementary
                else ""
            )
        )
    return out


def suppression_note(min_cell: int = DEFAULT_MIN_CELL) -> str:
    """the sentence that has to appear under any table this produced"""
    return (
        f"Cells holding fewer than {min_cell} patients are suppressed (shown "
        "blank, not zero), as is one further cell per row where a single "
        "suppression would otherwise be recoverable by subtraction, and any "
        "rate or median computed over a suppressed cell."
    )

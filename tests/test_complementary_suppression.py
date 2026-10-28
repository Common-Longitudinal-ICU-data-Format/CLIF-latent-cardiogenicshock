#!/usr/bin/env python3

"""
complementary suppression, which was inert in every exported table

Four mechanical defects between them meant it never fired once: `by` was built
as "every candidate column present", which swept in the varying dimension and
left one row per group; `by=()` was read as "skip" rather than "one group";
`n_cells > 2` excluded the two-cell case, which is the most exposed of all; and
the victim was the smallest survivor including zeros, so the pass could
"protect" a cell by blanking a zero and give the suppressed cell back anyway.
"""

from __future__ import annotations

import pathlib
import sys

import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from clifcs.export import suppress  # noqa: E402

MIN_CELL = 11


def ns(frame, by=()):
    return suppress.suppress_counts(frame, min_cell=MIN_CELL, by=by)["n"].to_list()


def group(counts, g="a"):
    return pl.DataFrame(
        {
            "g": [g] * len(counts),
            "label": [chr(ord("w") + i) for i in range(len(counts))],
            "n": counts,
        }
    )


def test_one_blanked_cell_in_a_distribution_forces_a_second():
    assert ns(group([5, 200, 300, 400]), by=("g",)) == [None, None, 300, 400]


def test_a_two_cell_group_blanks_both():
    """blank one of two and the other is the total minus a blank"""
    assert ns(group([5, 400]), by=("g",)) == [None, None]


def test_the_victim_is_never_a_zero_cell():
    """blanking a zero protects nothing -- the remaining sum still subtracts"""
    out = ns(group([5, 0, 300, 400]), by=("g",))
    assert out[1] == 0, "the zero must survive; it is not a usable victim"
    assert out.count(None) == 2, "a positive cell has to be sacrificed instead"


def test_by_empty_tuple_treats_the_whole_frame_as_one_group():
    """the single-distribution case, which used to be skipped entirely"""
    assert ns(group([5, 200, 300, 400]), by=()) == [None, None, 300, 400]


def test_by_none_opts_out():
    """for a table whose rows do not partition anything"""
    out = suppress.suppress_counts(
        group([5, 200, 300, 400]), min_cell=MIN_CELL, by=None
    )["n"].to_list()
    assert out == [None, 200, 300, 400], "only primary suppression should apply"


def test_a_part_whose_complement_is_small_is_blanked():
    """the real Rush row: a variable charted in 5668 of 5677 names the other 9"""
    out = suppress.suppress_counts(
        pl.DataFrame(
            {
                "variable": ["heart_rate"],
                "n_charted": [5668],
                "n_total": [5677],
                "share_charted": [5668 / 5677],
            }
        ),
        min_cell=MIN_CELL,
        by=None,
    )
    assert out["n_charted"][0] is None
    assert out["share_charted"][0] is None, "the share would restore it"

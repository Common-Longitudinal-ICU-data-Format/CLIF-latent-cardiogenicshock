#!/usr/bin/env python3

"""
what the clinician is doing, as a state layer beside what the patient is doing

The response states -- responsive, deteriorating, wean held and the rest --
describe the patient. They are also, it turned out, largely recoverable from
the dose and pressure LEVELS: norepinephrine dose alone separates "responsive"
from everything else at AUC 0.856 and "wean held" at 0.854, against 0.66-0.80
for the two-dimensional projection. So a comparison of responses that does not
hold dose fixed is partly a comparison of treatment intensity.

This layer makes the treatment explicit so it can be held fixed. A treatment
state is the norepinephrine band crossed with whether vasopressin is running
and whether any other vasoactive is:

    none / low / mid / high   x   +/- vasopressin   x   +/- other

Sixteen cells, of which the populated ones are what "essentially the same
treatment handbook" means operationally. Within a cell every patient-hour is
receiving comparable support, so a difference in response is a difference in
the patient rather than in the plan.

WHAT THIS STILL DOES NOT FIX. Patients are not assigned to treatment cells. A
clinician arrives at "high norepinephrine plus vasopressin" by reading a
patient who was not responding, so the cell is selected on the very thing the
response compares. Holding treatment fixed removes the dose confound from the
measurement; it does not make the comparison randomised, and no amount of
stratification will.
"""

from __future__ import annotations

import polars as pl


def label_treatment(frame, cfg):
    """
    one treatment state per patient-hour, from the dose level and the agents

    Read from the dose LEVEL at the hour, not the change: the question here is
    what support the patient was on, which is a different variable from what
    was done to them over the previous hour.
    """
    edges = [float(v) for v in cfg.treatment.ne_bands]
    names = list(cfg.treatment.ne_band_names)
    others = [a for a in cfg.treatment.other_agents
              if f"dose_{a}" in frame.columns]

    band = pl.when(pl.col("dose_norepinephrine") <= 0).then(pl.lit("none"))
    for low, high, name in zip(edges[:-1], edges[1:], names):
        band = band.when(
            (pl.col("dose_norepinephrine") > low)
            & (pl.col("dose_norepinephrine") <= high)
        ).then(pl.lit(name))
    band = band.otherwise(pl.lit(names[-1]))

    on_other = pl.lit(False)
    for agent in others:
        on_other = on_other | (pl.col(f"dose_{agent}") > 0)

    return frame.with_columns(
        ne_band=band.alias("ne_band"),
        on_vasopressin=(pl.col("dose_vasopressin") > 0)
        if "dose_vasopressin" in frame.columns else pl.lit(False),
        on_other_pressor=on_other,
    ).with_columns(
        treatment=pl.concat_str([
            pl.lit("NE "), pl.col("ne_band"),
            pl.when(pl.col("on_vasopressin")).then(pl.lit(" +vaso"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("on_other_pressor")).then(pl.lit(" +other"))
            .otherwise(pl.lit("")),
        ])
    )


#: one hex digit per treatment cell, so a patient's treatment is a string the
#: same shape as their response string. The encoding is additive and therefore
#: readable: the low nibble is the norepinephrine band, +4 means vasopressin,
#: +8 means another agent. So "6" is mid-dose norepinephrine with vasopressin
#: and "E" is high with both.
TREATMENT_BANDS = ("none", "low", "mid", "high")
TREATMENT_SYMBOLS = "0123456789ABCDEF"


def treatment_code(band: str, vaso: bool, other: bool) -> str:
    index = TREATMENT_BANDS.index(band) + (4 if vaso else 0) + (8 if other else 0)
    return TREATMENT_SYMBOLS[index]


def treatment_label(symbol: str) -> str:
    index = TREATMENT_SYMBOLS.find(symbol)
    if index < 0:
        return symbol
    band = TREATMENT_BANDS[index % 4]
    parts = [f"NE {band}"]
    if index & 4:
        parts.append("+vaso")
    if index & 8:
        parts.append("+other")
    return " ".join(parts)


def add_treatment_symbol(frame):
    """the single-character treatment code, alongside the readable label"""
    mapping = {}
    for band in TREATMENT_BANDS:
        for vaso in (False, True):
            for other in (False, True):
                label = (
                    f"NE {band}"
                    + (" +vaso" if vaso else "")
                    + (" +other" if other else "")
                )
                mapping[label] = treatment_code(band, vaso, other)
    return frame.with_columns(
        treatment_symbol=pl.col("treatment").replace_strict(
            mapping, default="?", return_dtype=pl.String
        )
    )


def collapse(sequence: str) -> str:
    """run-length collapse: the ORDER of what was done, not its durations"""
    out = []
    for character in sequence:
        if not out or out[-1] != character:
            out.append(character)
    return "".join(out)

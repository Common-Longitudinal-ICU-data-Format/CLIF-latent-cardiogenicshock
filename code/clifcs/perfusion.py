#!/usr/bin/env python3

"""
MAP adequacy crossed with vasopressor exposure, as a four-state variable

Lives here rather than in 09 because two scripts need it: 09 draws the states
over the flow field, and 11 clusters patients on their sequence through them.
"""

from __future__ import annotations

import polars as pl


#: MAP must EXCEED this to enter the adequate state, and fall BELOW `MAP_LOW`
#: to leave it. The gap is the hysteresis band: a patient sitting on 65 with
#: normal measurement noise would otherwise change group at almost every grid
#: point, and a transition matrix built from that measures the noise rather
#: than the patient. Inside the band the previous state is carried forward.
MAP_HIGH = 65.0
MAP_LOW = 55.0

#: the four groups, drawn largest-first so the most acute lands on top. The
#: palette encodes the 2x2 directly: warm means vasopressors are running, dark
#: means the MAP is low, so "low and on pressors" is the darkest warm colour
#: and reads as the worst cell without needing the legend.
PERFUSION_ORDER = (
    "MAP >65, no vasopressors",
    "MAP >65, on vasopressors",
    "MAP <65, no vasopressors",
    "MAP <65, on vasopressors",
)
PERFUSION_COLORS = {
    "MAP >65, no vasopressors": "#7fb3c8",
    "MAP >65, on vasopressors": "#e8a33d",
    "MAP <65, no vasopressors": "#9b6a8f",
    "MAP <65, on vasopressors": "#8f2230",
}


def perfusion_state(frame: pl.DataFrame, logger) -> pl.DataFrame:
    """
    MAP adequacy (with hysteresis) crossed with whether pressors are running

    The MAP side is a Schmitt trigger, not a threshold. Crossing up needs
    MAP > 65, crossing down needs MAP < 55, and between the two the patient
    keeps whatever state they were last in. A plain cut at 65 makes a patient
    hovering there oscillate every 6 h, which would dominate a transition
    matrix with measurement noise.

    Consequences of that, both deliberate:

      - the state is PATH-DEPENDENT. Two patients with identical MAP at some
        grid point can be in different groups, because they arrived from
        different sides. That is the point of a trigger and it is why this
        cannot be computed per row.
      - a NULL MAP carries the previous state forward rather than breaking the
        run. A missing measurement is not evidence that perfusion changed.

    The pressor side gets no hysteresis. Starting or stopping a vasopressor is
    a recorded clinical action, not a noisy measurement, so there is nothing to
    debounce -- `n_vasopressors` is built up from per-agent rates and is a true
    zero when nothing is running, never null.

    Seeding: the run starts at the first UNAMBIGUOUS observation (above 65 or
    below 55) and that state is carried backwards over any leading timepoints
    inside the band, since the alternative is to invent a state for them from
    the very measurement the band says is uninformative. A patient whose MAP
    never leaves [55, 65] has no unambiguous observation at all and falls back
    to the plain cut; the count is logged because it is the one group whose
    assignment the hysteresis did not decide.
    """
    needed = {"map", "n_vasopressors"}
    missing = needed - set(frame.columns)
    if missing:
        logger.warning(
            f"perfusion state needs {sorted(missing)}: skipping the panels"
        )
        return frame

    ordered = frame.sort(["hospitalization_id", "offset_hours"])
    ids = ordered["hospitalization_id"].to_list()
    maps = ordered["map"].to_list()

    # forward pass, per patient
    states: list[bool | None] = [None] * len(ids)
    previous_id = None
    state: bool | None = None
    for i, (pid, value) in enumerate(zip(ids, maps)):
        if pid != previous_id:
            state = None
            previous_id = pid
        if value is not None:
            if value > MAP_HIGH:
                state = True
            elif value < MAP_LOW:
                state = False
        states[i] = state

    # backward fill of the leading run, per patient, from the first resolved
    # state. Walked in reverse so a whole leading block fills in one pass.
    following: bool | None = None
    previous_id = None
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] != previous_id:
            following = None
            previous_id = ids[i]
        if states[i] is None:
            states[i] = following
        else:
            following = states[i]

    unresolved = sum(1 for s in states if s is None)
    ordered = ordered.with_columns(
        map_adequate=pl.Series(states, dtype=pl.Boolean)
    ).with_columns(
        # the fallback for a patient who never left the band
        map_adequate=pl.when(pl.col("map_adequate").is_not_null())
        .then(pl.col("map_adequate"))
        .otherwise(pl.col("map") >= MAP_HIGH)
    )
    if unresolved:
        logger.info(
            f"perfusion state: {unresolved:,} of {len(ids):,} patient-timepoints "
            f"({unresolved / len(ids):.1%}) never left the {MAP_LOW:g}-"
            f"{MAP_HIGH:g} band and fall back to the plain cut"
        )

    ordered = ordered.with_columns(
        perfusion_state=pl.concat_str([
            pl.when(pl.col("map_adequate")).then(pl.lit("MAP >65"))
            .otherwise(pl.lit("MAP <65")),
            pl.when(pl.col("n_vasopressors") > 0)
            .then(pl.lit(", on vasopressors"))
            .otherwise(pl.lit(", no vasopressors")),
        ])
    )

    counts = ordered["perfusion_state"].value_counts(sort=True)
    logger.info("perfusion state, patient-timepoints:")
    for level, n in zip(counts["perfusion_state"].to_list(),
                        counts["count"].to_list()):
        logger.info(f"    {level:<28} {n:>8,}  {n / ordered.height:>6.1%}")
    return ordered


# =================================================== the richer state space


#: A dose change counts only if it clears BOTH a relative and an absolute
#: floor. The relative floor alone makes 0.01 -> 0.02 mcg/kg/min a "100%
#: increase", which is a titration artefact rather than a decision; the
#: absolute floor alone would call a 0.30 -> 0.32 change meaningful on a
#: patient already at a high dose. This is the same deadband reasoning as the
#: MAP band: without it the trend flips at nearly every grid point and a motif
#: built from the sequence measures charting noise.
DOSE_RELATIVE = 0.25
DOSE_ABSOLUTE = 0.01

#: The alphabet, numbered so a sequence reads as a string of digits and a
#: motif like "10" or "3012" is legible. Index IS the symbol.
#:
#: Two MAP levels crossed with: no pressors, or one of three dose trends
#: crossed with whether norepinephrine is running alone or with something
#: else. 14 states plus a terminal symbol.
#:
#: Composition applies only when pressors are running, which is why this is
#: not a clean 2x4x2 -- "no pressors" has no trend and no composition, so the
#: count is 2 x (1 + 3 x 2) = 14 rather than 16.
PERFUSION_STATES = tuple(
    f"MAP {m}65, {label}"
    for m in (">", "<")
    for label in (
        "no pressors",
        "NE alone, rising",
        "NE alone, falling",
        "NE alone, steady",
        "NE+other, rising",
        "NE+other, falling",
        "NE+other, steady",
    )
) + ("record ended",)

#: index -> symbol, for writing sequences as digit strings. Beyond 9 the
#: symbols run A, B, C... so every state stays one character and a motif
#: stays readable.
STATE_SYMBOLS = "0123456789ABCDE"


def state_symbol(index: int) -> str:
    return STATE_SYMBOLS[index] if 0 <= index < len(STATE_SYMBOLS) else "?"


def rich_perfusion_state(frame: "pl.DataFrame", logger) -> "pl.DataFrame":
    """
    MAP adequacy x pressor dose trend x norepinephrine composition

    Extends the four-state version in two directions the earlier analysis
    could not see. The pressor side is now a TREND rather than a presence
    flag, so "on pressors and climbing" is distinguishable from "on pressors
    and weaning" -- the distinction that separated P2 from P4 at twice the
    mortality while their peak doses were nearly identical. And it records
    whether norepinephrine is running alone or alongside another agent, since
    reaching for a second pressor is a decision rather than a dose.

    THE TREND IS A DERIVATIVE, with two consequences. It needs the previous
    grid point, so the first window of a run has nothing to compare against
    and is called steady -- biasing the first window toward steady, which is
    stated rather than hidden. And because the state already encodes change, a
    SEQUENCE of these states encodes change-of-change: a motif reading
    "rising then falling" is a second-order statement, and should be read as
    one.

    Null dose carries forward as no pressors; a null MAP carries the previous
    adequacy, exactly as in the four-state version.
    """
    needed = {"map", "nee", "nee_norepinephrine", "n_other_pressors"}
    missing = needed - set(frame.columns)
    if missing:
        logger.warning(
            f"rich perfusion state needs {sorted(missing)}: skipping. "
            "Rebuild the state table so it carries the per-agent columns."
        )
        return frame

    ordered = frame.sort(["hospitalization_id", "offset_hours"])
    ids = ordered["hospitalization_id"].to_list()
    maps = ordered["map"].to_list()
    nee = [0.0 if v is None else float(v) for v in ordered["nee"].to_list()]
    ne_dose = [
        0.0 if v is None else float(v)
        for v in ordered["nee_norepinephrine"].to_list()
    ]
    others = [
        0 if v is None else int(v) for v in ordered["n_other_pressors"].to_list()
    ]

    adequate: list[bool | None] = [None] * len(ids)
    previous_id, state = None, None
    for i, (pid, value) in enumerate(zip(ids, maps)):
        if pid != previous_id:
            state, previous_id = None, pid
        if value is not None:
            if value > MAP_HIGH:
                state = True
            elif value < MAP_LOW:
                state = False
        adequate[i] = state
    following, previous_id = None, None
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] != previous_id:
            following, previous_id = None, ids[i]
        if adequate[i] is None:
            adequate[i] = following
        else:
            following = adequate[i]

    labels: list[str] = []
    previous_id, previous_dose = None, None
    for i, pid in enumerate(ids):
        if pid != previous_id:
            previous_id, previous_dose = pid, None
        dose = nee[i]
        if dose <= 0:
            arm = "no pressors"
        else:
            if previous_dose is None or previous_dose <= 0:
                trend = "steady"
            else:
                change = dose - previous_dose
                relative = abs(change) / max(previous_dose, 1e-9)
                if relative >= DOSE_RELATIVE and abs(change) >= DOSE_ABSOLUTE:
                    trend = "rising" if change > 0 else "falling"
                else:
                    trend = "steady"
            composition = "NE+other" if others[i] > 0 else "NE alone"
            if ne_dose[i] <= 0 and others[i] > 0:
                # pressors running but none of them norepinephrine; grouped
                # with NE+other rather than given a state of its own, which
                # would add two sparse symbols for a small group
                composition = "NE+other"
            arm = f"{composition}, {trend}"
        high = adequate[i]
        if high is None:
            high = (maps[i] or 0) >= MAP_HIGH
        labels.append(f"MAP {'>' if high else '<'}65, {arm}")
        previous_dose = dose

    index = {name: i for i, name in enumerate(PERFUSION_STATES)}
    ordered = ordered.with_columns(
        rich_state=pl.Series(labels, dtype=pl.String),
    ).with_columns(
        rich_symbol=pl.col("rich_state").replace_strict(
            {name: state_symbol(i) for name, i in index.items()},
            default="?", return_dtype=pl.String,
        )
    )

    counts = ordered["rich_state"].value_counts(sort=True)
    logger.info(
        f"rich perfusion state: {len(PERFUSION_STATES)} symbols "
        f"({DOSE_RELATIVE:.0%} relative and {DOSE_ABSOLUTE} absolute dose "
        "deadband)"
    )
    for level, n in zip(counts["rich_state"].to_list(), counts["count"].to_list()):
        logger.info(
            f"    {state_symbol(index.get(level, -1))}  {level:<30} "
            f"{n:>8,}  {n / ordered.height:>6.1%}"
        )
    return ordered

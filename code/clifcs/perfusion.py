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

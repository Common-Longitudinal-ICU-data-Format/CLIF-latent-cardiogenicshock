#!/usr/bin/env python3

"""
MAP and pressor state at sub-window resolution, from the raw CLIF tables

The coarse analysis gives each patient one letter per 6-hour window: 29
letters for a week. That throws away everything that happens INSIDE a window.
A patient whose norepinephrine was titrated up three times and then weaned
between hour 12 and hour 18 carries the same single letter as one who sat
untouched, because both are summarised by whatever was running at hour 18.

This builds the finer object. Within each 6-hour window a patient gets as
many letters as they had distinct states, terminated by an end-of-window
marker, so a sentence looks like

    MAP<65 no pressors -> MAP<65 NE rising -> MAP<65 NE rising  _
    MAP<65 NE+other rising -> MAP>65 NE+other steady -> ...     _

and the `_` is now a real boundary with something on either side of it,
instead of a separator between single letters that would block every merge.

WHAT COUNTS AS AN EVENT, and why it is not every measurement. The timeline is
the union of two sources, which are not the same kind of thing:

  * DOSE CHANGES are read as events, one per administration record. A new
    infusion rate is a recorded clinical action. Its timing is the decision
    itself, so event-driven is the honest representation.

  * MAP is resampled on a fixed sub-grid, NOT read per measurement. Charting
    frequency is not a clinical decision: a patient with an arterial line
    yields a MAP every minute and one on cuff pressures yields a handful a
    day. One letter per measurement would make sequence length a measure of
    monitoring intensity, and this project has already shown position in the
    projection explains R2 0.45 of log stream length -- the documentation
    confound does not need another channel into the analysis.

RUNS ARE COLLAPSED WITHIN THE WINDOW, which is what keeps length meaningful.
After collapsing, the number of letters in a window is the number of observed
STATE CHANGES, bounded by the sub-grid. Without it, length would again track
how often the window happened to be sampled. It can be turned off, but the
default is on and the reason is the same confound.

STALENESS IS BOUNDED, which the coarse version does not bother with. At 6-hour
resolution a two-hour-old MAP is a reasonable reading of the chart. At hourly
resolution, carrying an 18-hour-old value across eighteen checkpoints
manufactures a long run of identical letters out of one measurement, and after
run-collapsing that run reads as "stable" -- stability invented by sparse
charting. Beyond `tolerance_hours` the MAP is treated as unknown.

THE DOSE TREND MEANS SOMETHING DIFFERENT HERE. The deadband is applied between
consecutive EVENTS rather than between 6-hour grid points, so "rising" is a
finer derivative than the coarse analysis's "rising" and the two are not
interchangeable. A titration that the coarse version sees as one net rise
appears here as several.
"""

from __future__ import annotations

import polars as pl

from clifcs.clif import scan
from clifcs.labels import norepinephrine_equivalents
from clifcs.perfusion import MAP_HIGH, MAP_LOW

#: the sub-window MAP checkpoint spacing and how stale a MAP may be before it
#: is treated as unknown rather than carried forward
RESOLUTION_MINUTES = 60
TOLERANCE_HOURS = 4.0

#: end of a 6-hour window. Chosen outside STATE_SYMBOLS so it can never
#: collide with a state, and merges are forbidden across it.
WINDOW_END = "_"

#: MAP values outside this are artefacts; the raw table reached 1,040 mmHg
MAP_RANGE = (20.0, 200.0)


def _event_timeline(anchors, doses, end_hours, resolution_minutes):
    """
    every moment worth evaluating a state at, per patient

    The union of the fixed MAP checkpoints and the actual dose-change records.
    Returned with `offset_hours` so the caller can bucket into windows without
    re-deriving it from timestamps.
    """
    checkpoints = (
        anchors.select("hospitalization_id", "onset")
        .join(
            pl.DataFrame({
                "tick": list(range(
                    0, int(end_hours * 60 // resolution_minutes) + 1
                ))
            }),
            how="cross",
        )
        .with_columns(
            event_dttm=pl.col("onset")
            + pl.duration(minutes=pl.col("tick") * resolution_minutes)
        )
        .select("hospitalization_id", "event_dttm")
    )

    if doses is not None and doses.height:
        changes = (
            doses.select("hospitalization_id", event_dttm=pl.col("admin_dttm"))
            .join(anchors.select("hospitalization_id", "onset"),
                  on="hospitalization_id", how="inner")
            .filter(
                (pl.col("event_dttm") >= pl.col("onset"))
                & (pl.col("event_dttm")
                   <= pl.col("onset") + pl.duration(hours=end_hours))
            )
            .select("hospitalization_id", "event_dttm")
        )
        timeline = pl.concat([checkpoints, changes], how="vertical")
    else:
        timeline = checkpoints

    timeline = (
        timeline.unique(subset=["hospitalization_id", "event_dttm"])
        .join(anchors.select("hospitalization_id", "onset"),
              on="hospitalization_id", how="inner")
        .with_columns(
            offset_hours=(
                pl.col("event_dttm") - pl.col("onset")
            ).dt.total_seconds() / 3600.0
        )
        .drop("onset")
        .sort(["hospitalization_id", "event_dttm"])
    )
    return timeline


def _physiology_at_events(vitals, timeline, doses, tolerance_hours):
    """
    MAP and per-agent pressor rates as they stood at each event

    Backward as-of per agent, then summed. A single join keyed only on the
    patient would return whichever agent was charted last rather than
    everything running concurrently -- the same trap the coarse version
    documents.
    """
    events = timeline.sort("event_dttm")
    if vitals is not None and vitals.height:
        out = events.join_asof(
            vitals, left_on="event_dttm", right_on="recorded_dttm",
            by="hospitalization_id", strategy="backward",
        )
        # A MAP older than the tolerance is not evidence about now. Carrying
        # it would invent a run of identical letters from one measurement,
        # which after run-collapsing reads as stability.
        out = out.with_columns(
            map=pl.when(
                (pl.col("event_dttm") - pl.col("recorded_dttm"))
                .dt.total_seconds() <= tolerance_hours * 3600
            ).then(pl.col("map")).otherwise(None)
        ).drop("recorded_dttm")
    else:
        out = events.with_columns(map=pl.lit(None, dtype=pl.Float64))

    out = out.with_columns(
        nee=pl.lit(0.0), nee_norepinephrine=pl.lit(0.0),
        n_other_pressors=pl.lit(0, dtype=pl.Int64),
        n_vasopressors=pl.lit(0, dtype=pl.Int64),
    )
    if doses is None or doses.height == 0:
        return out

    for agent in doses["med_category"].unique().to_list():
        # NOT f"nee_{agent}": for norepinephrine that is literally
        # "nee_norepinephrine", the name of one of the running totals, so the
        # join overwrote the total and the drop below then deleted it.
        column = "_rate"
        part = (
            doses.filter(pl.col("med_category") == agent)
            .select("hospitalization_id", "admin_dttm",
                    **{column: pl.col("nee")})
            .sort("admin_dttm")
        )
        out = (
            out.sort("event_dttm")
            .join_asof(
                part, left_on="event_dttm", right_on="admin_dttm",
                by="hospitalization_id", strategy="backward",
            )
            .with_columns(pl.col(column).fill_null(0.0))
            .with_columns(
                nee=pl.col("nee") + pl.col(column),
                n_vasopressors=pl.col("n_vasopressors")
                + (pl.col(column) > 0).cast(pl.Int64),
                **(
                    {"nee_norepinephrine": pl.col(column)}
                    if agent == "norepinephrine"
                    else {"n_other_pressors": pl.col("n_other_pressors")
                          + (pl.col(column) > 0).cast(pl.Int64)}
                ),
            )
            .drop(column, "admin_dttm", strict=False)
        )
    return out


def _label_events(frame, dose_relative, dose_absolute):
    """
    the 15-state label at every event, with hysteresis and a dose deadband

    Deliberately a re-implementation rather than a call into
    `rich_perfusion_state`: that one is keyed to `offset_hours` on a regular
    grid and seeds its trend from the previous grid point. Here the gap
    between consecutive rows is irregular, so the same code would silently
    compare a 1-minute gap against a 4-hour one and call both a trend.
    """
    from clifcs.perfusion import PERFUSION_STATES, state_symbol

    ordered = frame.sort(["hospitalization_id", "event_dttm"])
    ids = ordered["hospitalization_id"].to_list()
    maps = ordered["map"].to_list()
    nee = [0.0 if v is None else float(v) for v in ordered["nee"].to_list()]
    ne_dose = [
        0.0 if v is None else float(v)
        for v in ordered["nee_norepinephrine"].to_list()
    ]
    others = [
        0 if v is None else int(v)
        for v in ordered["n_other_pressors"].to_list()
    ]

    # MAP adequacy: Schmitt trigger forward, then seeded backwards from the
    # first unambiguous reading, exactly as the coarse version
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

    symbols, labels = [], []
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
                if relative >= dose_relative and abs(change) >= dose_absolute:
                    trend = "rising" if change > 0 else "falling"
                else:
                    trend = "steady"
            composition = "NE+other" if others[i] > 0 else "NE alone"
            if ne_dose[i] <= 0 and others[i] > 0:
                # another agent is running without norepinephrine; the coarse
                # state space has no cell for that, so it is read as NE+other
                composition = "NE+other"
            arm = f"{composition}, {trend}"
        previous_dose = dose
        side = ">" if adequate[i] else "<"
        name = f"MAP {side}65, {arm}"
        labels.append(name)
        symbols.append(
            state_symbol(PERFUSION_STATES.index(name))
            if name in PERFUSION_STATES else "?"
        )
    return ordered.with_columns(
        fine_state=pl.Series(labels), fine_symbol=pl.Series(symbols)
    )


def windowed_sentences(labelled, step_hours, end_hours, collapse=True):
    """
    one string per patient: the states inside each window, then WINDOW_END

    Run-collapsing is applied WITHIN a window only. Across a boundary a
    repeated state is kept, because the marker has to stay where it is for the
    windows to line up with the coarse analysis -- collapsing across it would
    make window k's length depend on window k-1's last state.
    """
    frame = labelled.filter(
        (pl.col("offset_hours") >= 0) & (pl.col("offset_hours") < end_hours)
    ).with_columns(
        window=(pl.col("offset_hours") // step_hours).cast(pl.Int64)
    ).sort(["hospitalization_id", "offset_hours"])

    rows = []
    n_windows = int(end_hours // step_hours)
    for (patient,), part in frame.group_by(
        ["hospitalization_id"], maintain_order=True
    ):
        by_window = {}
        for window, symbol in zip(
            part["window"].to_list(), part["fine_symbol"].to_list()
        ):
            bucket = by_window.setdefault(int(window), [])
            if collapse and bucket and bucket[-1] == symbol:
                continue
            bucket.append(symbol)
        pieces, lengths = [], []
        for window in range(n_windows):
            letters = by_window.get(window, [])
            # An empty window is not "no pressors and a fine MAP"; it is a
            # patient with no record, which the coarse version writes as
            # "record ended". Left empty here so the marker alone shows it.
            pieces.append("".join(letters) + WINDOW_END)
            lengths.append(len(letters))
        rows.append({
            "hospitalization_id": patient,
            "fine_sentence": "".join(pieces),
            "fine_letters": int(sum(lengths)),
            "windows_with_record": int(sum(1 for n in lengths if n)),
            "max_letters_in_a_window": int(max(lengths)) if lengths else 0,
            "windows_with_a_change": int(sum(1 for n in lengths if n > 1)),
        })
    return pl.DataFrame(rows)


def fine_state_sequences(
    data_dir, anchors, logger, step_hours=6.0, end_hours=168.0,
    resolution_minutes=RESOLUTION_MINUTES, tolerance_hours=TOLERANCE_HOURS,
    dose_relative=None, dose_absolute=None, collapse=True, batch=3000,
):
    """
    the whole pipeline for one site: raw tables in, windowed sentences out

    `anchors` needs hospitalization_id and onset. Batched by patient because
    the event timeline is two orders of magnitude larger than the 29-point
    grid it replaces -- a week at hourly resolution plus every dose record is
    a few hundred rows per patient, and the per-agent as-of joins multiply
    that before they collapse.
    """
    from clifcs.perfusion import DOSE_ABSOLUTE, DOSE_RELATIVE

    dose_relative = DOSE_RELATIVE if dose_relative is None else dose_relative
    dose_absolute = DOSE_ABSOLUTE if dose_absolute is None else dose_absolute

    ids = anchors["hospitalization_id"].to_list()

    # Read the two raw tables ONCE for the whole site. Both were originally
    # pulled inside the batch loop, which re-scanned every medication and
    # vitals row per batch -- the scan dominated, so the cost scaled with the
    # number of batches rather than with the number of patients.
    vitals = (
        scan(data_dir, "clif_vitals", dttm_cols=("recorded_dttm",))
        .filter(
            pl.col("vital_category") == "map",
            pl.col("vital_value").is_between(*MAP_RANGE),
        )
        .join(anchors.lazy().select("hospitalization_id").unique(),
              on="hospitalization_id", how="inner")
        .select("hospitalization_id", "recorded_dttm",
                map=pl.col("vital_value"))
        .sort("recorded_dttm")
        .collect()
    )
    all_doses = norepinephrine_equivalents(
        data_dir, anchors.select("hospitalization_id")
    )
    logger.info(
        f"    {vitals.height:,} MAP measurements and "
        f"{all_doses.height:,} pressor administration records"
    )

    parts = []
    for start in range(0, len(ids), batch):
        chunk_ids = ids[start:start + batch]
        chunk = anchors.filter(pl.col("hospitalization_id").is_in(chunk_ids))
        doses = (
            all_doses.filter(pl.col("hospitalization_id").is_in(chunk_ids))
            if all_doses is not None and all_doses.height else None
        )
        timeline = _event_timeline(
            chunk, doses, end_hours, resolution_minutes
        )
        if timeline.height == 0:
            continue
        measured = _physiology_at_events(
            vitals.filter(pl.col("hospitalization_id").is_in(chunk_ids)),
            timeline, doses, tolerance_hours,
        )
        labelled = _label_events(measured, dose_relative, dose_absolute)
        parts.append(windowed_sentences(
            labelled, step_hours, end_hours, collapse=collapse
        ))
        logger.info(
            f"    batch {start // batch + 1}: {chunk.height:,} patients, "
            f"{timeline.height:,} events "
            f"({timeline.height / max(chunk.height, 1):.0f} per patient)"
        )
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="vertical")

#!/usr/bin/env python3

"""
what the clinician did, and what the blood pressure did next

Each decision step is labelled by an ACTION read from the change in
norepinephrine-equivalent dose across the step, paired with the RESPONSE read
from the change in MAP over the window after it. The eight states are the
cross of those two, plus the two off-pressor states where no action was
available to take.

WHY THIS IS NOT THE 15-STATE PERFUSION SPACE. That encoding labels a window by
what was true in it -- MAP band, pressor arm, dose trend. This one labels a
window by what was DONE and what FOLLOWED, which is a different object: the
same "NE rising" window is a responsive escalation or a refractory one
depending on what the MAP did afterwards, and the perfusion space cannot tell
those apart. The two are complementary and neither replaces the other.

THE RESPONSE WINDOW IS STRICTLY AFTER THE ACTION. MAP is read from the end of
the step forward, never across it, so no state is labelled using a measurement
that preceded the dose change it is supposed to be responding to.

WHAT THIS CANNOT ESTABLISH. A "responsive escalation" is not evidence the dose
raised the MAP. Clinicians escalate when the pressure is falling and the
pressure often recovers for reasons of its own, so the label describes a
temporal coincidence in the record. Confounding by indication runs through
every state here, and the counterfactual -- what the MAP would have done
without the change -- is not in the data.

ONE DEADBAND, TWO PLACES. A dose change must clear both a relative and an
absolute floor to count as an action, and a MAP change must clear a millimetre
band to count as a response. Without the first, 0.01 -> 0.02 is a doubling;
without the second, two cuff readings differing by noise classify a response.
"""

from __future__ import annotations

import polars as pl

from clifcs.clif import scan

#: the eight states, in the order the report lists them
STATES = (
    "responsive escalation",
    "refractory escalation",
    "tolerated wean",
    "failed wean",
    "stable hold",
    "drifting hold",
    "off pressors, MAP >=65",
    "off pressors, MAP <65",
)

#: one character each, so a trajectory can be written as a string the way the
#: perfusion sequences are. Kept distinct from perfusion.STATE_SYMBOLS on
#: purpose: a reader who confuses the two encodings should see unfamiliar
#: letters rather than plausible ones.
STATE_SYMBOLS = {
    "responsive escalation": "R",
    "refractory escalation": "X",
    "tolerated wean": "W",
    "failed wean": "F",
    "stable hold": "S",
    "drifting hold": "D",
    "off pressors, MAP >=65": "o",
    "off pressors, MAP <65": "u",
}

#: what each state needs to be decidable. A step missing its MAP response
#: cannot be labelled at all, and is left null rather than guessed.
UNDECIDABLE = "undecidable"


def classify_action(nee_start, nee_end, relative_minimum, absolute_minimum):
    """
    escalate / de-escalate / no change, from the dose at each end of the step

    BOTH floors must be cleared. The relative floor alone calls 0.01 -> 0.02 a
    doubling; the absolute floor alone ignores scale, so a 0.02 move counts the
    same on a patient at 0.03 as on one at 0.6.

    STARTING FROM ZERO is handled by the absolute floor alone, since a relative
    change from zero is undefined rather than infinite. Stopping TO zero is
    likewise a de-escalation whenever the absolute floor is cleared, which it
    is for any dose above the floor.
    """
    if nee_start is None or nee_end is None:
        return None
    change = float(nee_end) - float(nee_start)
    if abs(change) < absolute_minimum:
        return "no change"
    if nee_start <= 0:
        return "escalate" if change > 0 else "no change"
    if abs(change) / max(float(nee_start), 1e-9) < relative_minimum:
        return "no change"
    return "escalate" if change > 0 else "de-escalate"


def classify_state(action, map_end, map_response, re_escalated, adequate,
                   deadband):
    """
    the state, from the action and what the MAP did over the response window

    `map_response` is MAP at the end of the response window minus MAP at the
    end of the step -- strictly after the action. `re_escalated` says whether
    the dose went back up inside that window, which only the wean states use.

    OFF-PRESSOR STATES TAKE PRECEDENCE and are decided by MAP level, not by
    MAP change: the question for a patient on no support is whether the
    pressure is adequate, not whether it moved three millimetres.
    """
    if action is None or map_end is None:
        return UNDECIDABLE
    if action == "off":
        return (
            "off pressors, MAP >=65" if map_end >= adequate
            else "off pressors, MAP <65"
        )
    if map_response is None:
        return UNDECIDABLE
    if action == "escalate":
        return (
            "responsive escalation" if map_response > deadband
            else "refractory escalation"
        )
    if action == "de-escalate":
        # Two ways a wean fails and both count: the pressure falls, or the dose
        # is put back up. The second is the clinician's own verdict on the
        # first, and excluding it would score a wean as tolerated precisely
        # when it was reversed quickly enough to keep the MAP from moving.
        if re_escalated:
            return "failed wean"
        if map_response < -deadband:
            return "failed wean"
        return (
            "tolerated wean" if map_end + map_response >= adequate
            else "failed wean"
        )
    return "stable hold" if map_response >= -deadband else "drifting hold"


def encode(hourly, cfg, logger):
    """
    one row per (patient, decision step) with its action, response and state

    `hourly` needs hospitalization_id, offset_hours, nee, map and
    on_vasopressin on a one-hour grid. Steps are taken every
    `cfg.action.step_hours`; the response window is the next
    `cfg.response.hours` after the step ends, so a step at t uses dose at t and
    t+step and MAP at t+step and t+step+response.

    THE LAST STEPS OF EVERY PATIENT ARE UNDECIDABLE by construction -- there is
    no response window after the record ends. They are labelled `undecidable`
    and counted, never dropped quietly, because dropping them would make the
    state distribution a distribution over steps that happened to be followed
    by more record.
    """
    step = float(cfg.action.step_hours)
    response_hours = float(cfg.response.hours)
    adequate = float(cfg.map.adequate)
    deadband = float(cfg.response.deadband_mmhg)
    relative = float(cfg.action.relative_minimum)
    absolute = float(cfg.action.absolute_minimum)

    frame = hourly.sort(["hospitalization_id", "offset_hours"])
    # lookup by (patient, hour) so the step arithmetic reads the grid directly
    # rather than relying on row offsets, which break wherever a patient is
    # missing an hour
    key_nee, key_map, key_vaso = {}, {}, {}
    for patient, hour, nee, mean_map, vaso in zip(
        frame["hospitalization_id"].to_list(),
        frame["offset_hours"].to_list(),
        frame["nee"].to_list(),
        frame["map"].to_list(),
        frame["on_vasopressin"].to_list(),
    ):
        key = (patient, round(float(hour), 3))
        key_nee[key] = nee
        key_map[key] = mean_map
        key_vaso[key] = vaso

    # EACH PATIENT STOPS AT THEIR OWN LAST HOUR. An earlier version took the
    # maximum across the cohort and generated steps to 168h for everyone, so
    # bounding the hourly series at discharge changed nothing -- the steps were
    # still created and simply found no rows, turning an absent step into an
    # undecidable one. The total step count not moving when 18% of
    # patient-hours were dropped is what gave it away.
    extents = (
        frame.group_by("hospitalization_id")
        .agg(last_hour=pl.col("offset_hours").max())
    )
    horizon = dict(zip(
        extents["hospitalization_id"].to_list(),
        [float(h) for h in extents["last_hour"].to_list()],
    ))

    rows = []
    for patient, last_hour in horizon.items():
        t = 0.0
        while t + step <= last_hour + 1e-9:
            start = (patient, round(t, 3))
            end = (patient, round(t + step, 3))
            after = (patient, round(t + step + response_hours, 3))
            nee_start = key_nee.get(start)
            nee_end = key_nee.get(end)
            map_end = key_map.get(end)
            map_after = key_map.get(after)

            if nee_start is not None and nee_end is not None \
                    and nee_start <= 0 and nee_end <= 0:
                action = "off"
            else:
                action = classify_action(
                    nee_start, nee_end, relative, absolute
                )
            # re-escalation anywhere inside the response window, not only at
            # its end: a dose put up and taken down again inside two hours is
            # still the clinician reversing the wean
            re_escalated = False
            if action == "de-escalate" and nee_end is not None:
                probe = t + step
                while probe <= t + step + response_hours + 1e-9:
                    value = key_nee.get((patient, round(probe, 3)))
                    if value is not None and classify_action(
                        nee_end, value, relative, absolute
                    ) == "escalate":
                        re_escalated = True
                        break
                    probe += 1.0

            response = (
                None if map_after is None or map_end is None
                else float(map_after) - float(map_end)
            )
            rows.append({
                "hospitalization_id": patient,
                "offset_hours": t,
                "nee_start": nee_start,
                "nee_end": nee_end,
                "map_end": map_end,
                "map_response": response,
                "on_vasopressin": bool(key_vaso.get(end) or False),
                "action": action,
                "re_escalated": re_escalated,
                "state": classify_state(
                    action, map_end, response, re_escalated, adequate, deadband
                ),
            })
            t += step

    out = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        symbol=pl.col("state").replace_strict(
            STATE_SYMBOLS, default="?", return_dtype=pl.String
        )
    )
    decidable = out.filter(pl.col("state") != UNDECIDABLE)
    logger.info(
        f"  {out.height:,} decision steps within the record, "
        f"{decidable.height:,} decidable "
        f"({decidable.height / max(out.height, 1):.1%}); the rest have no "
        "response window or no MAP"
    )
    return out


def _plausible_map(data_dir, anchors, cfg, low, high, logger):
    """
    MAP readings that survive a cross-check against the same timestamp's SBP

    A RANGE FILTER CANNOT CATCH THE REAL ERROR HERE. Systolic values are
    written into the MAP field: among readings above 130 the median MAP and the
    median SBP at the same timestamp are both 152, and 48% of them have
    MAP >= SBP, which cannot happen. Those readings sit inside any sane range,
    so [20, 200] passes them through.

    TWO CHECKS, BOTH NEEDING A PAIRED READING. MAP must fall below the systolic
    at the same timestamp, and where the diastolic is also present, MAP must sit
    within a tolerance of (SBP + 2*DBP)/3. The median gap is 3 mmHg overall
    against 31 mmHg among readings above 130, so the tolerance separates
    rounding from mislabelling.

    THE RATE IS STRONGLY SITE-DEPENDENT -- 3.48% of UCMC readings fail against
    0.38% at NU and 0.04% at RUSH -- so this is not a cosmetic clean-up. Left
    uncorrected it biases the three sites differently, and because the error
    inflates MAP it pushes states toward "adequate" and can manufacture a
    response to a dose change. An unpaired MAP is KEPT: absence of a
    simultaneous systolic is not evidence against the reading.
    """
    plausibility = cfg.map.get("plausibility", {})
    require_below = bool(plausibility.get("require_below_sbp", True))
    max_gap = plausibility.get("max_gap_from_calculated")

    wide = (
        scan(data_dir, "clif_vitals", dttm_cols=("recorded_dttm",))
        .filter(
            pl.col("vital_category").is_in(["map", "sbp", "dbp"]),
            pl.col("vital_value").is_not_null(),
        )
        .join(anchors.lazy().select("hospitalization_id").unique(),
              on="hospitalization_id", how="inner")
        .select("hospitalization_id", "recorded_dttm", "vital_category",
                "vital_value")
        .collect()
        .pivot(
            values="vital_value",
            index=["hospitalization_id", "recorded_dttm"],
            on="vital_category", aggregate_function="first",
        )
    )
    if "map" not in wide.columns:
        logger.warning("no MAP readings at all")
        return pl.DataFrame(
            schema={"hospitalization_id": pl.String,
                    "recorded_dttm": pl.Datetime("us"), "map": pl.Float64}
        )
    frame = wide.filter(pl.col("map").is_between(low, high))
    before = frame.height

    if require_below and "sbp" in frame.columns:
        frame = frame.filter(
            pl.col("sbp").is_null() | (pl.col("map") < pl.col("sbp"))
        )
    if max_gap is not None and {"sbp", "dbp"} <= set(frame.columns):
        calculated = (pl.col("sbp") + 2 * pl.col("dbp")) / 3
        frame = frame.filter(
            pl.col("sbp").is_null() | pl.col("dbp").is_null()
            | ((pl.col("map") - calculated).abs() <= float(max_gap))
        )
    dropped = before - frame.height
    logger.info(
        f"  MAP plausibility: {dropped:,} of {before:,} readings dropped "
        f"({dropped / max(before, 1):.2%}) as systolics in the MAP field"
    )
    return (
        frame.select("hospitalization_id", "recorded_dttm", "map")
        .sort("recorded_dttm")
    )


def hourly_map(data_dir, anchors, cfg, end_hours, logger):
    """
    MAP on the same regular grid the dose sits on

    Lives beside the state encoder rather than in `nee` because it is the other
    half of one pairing, and the two must share a grid exactly: a step reads
    dose at t and MAP at t+step, so an hour present in one series and absent
    from the other silently makes that step undecidable.

    NULL, NOT ZERO, BEYOND THE TOLERANCE -- the opposite of the dose side, and
    the asymmetry is the point. An absent infusion record means no infusion;
    an absent MAP means nobody measured it. Carrying an 18-hour-old pressure
    across eighteen hourly steps would manufacture a long flat response and
    every one of those steps would read as a stable hold.
    """
    low, high = [float(v) for v in cfg.map.range]
    step = int(cfg.map.resolution_minutes)
    tolerance = float(cfg.map.tolerance_hours) * 3600
    ticks = int(end_hours * 60 // step) + 1

    grid = (
        anchors.select("hospitalization_id", "onset")
        .join(pl.DataFrame({"tick": list(range(ticks))}), how="cross")
        .with_columns(
            event_dttm=pl.col("onset")
            + pl.duration(minutes=pl.col("tick") * step),
            offset_hours=pl.col("tick") * (step / 60.0),
        )
        .drop("onset", "tick")
        .sort("event_dttm")
    )
    vitals = _plausible_map(data_dir, anchors, cfg, low, high, logger)
    if vitals.is_empty():
        logger.warning("no MAP measurements in range")
        return grid.with_columns(map=pl.lit(None, dtype=pl.Float64))
    out = grid.join_asof(
        vitals, left_on="event_dttm", right_on="recorded_dttm",
        by="hospitalization_id", strategy="backward",
    ).with_columns(
        map=pl.when(
            (pl.col("event_dttm") - pl.col("recorded_dttm"))
            .dt.total_seconds() <= tolerance
        ).then(pl.col("map")).otherwise(None)
    ).drop("recorded_dttm")
    logger.info(
        f"  hourly MAP: {out['map'].is_not_null().mean():.1%} of patient-hours "
        f"have a measurement within {cfg.map.tolerance_hours:.0f}h"
    )
    return out

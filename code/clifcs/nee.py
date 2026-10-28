#!/usr/bin/env python3

"""
hourly norepinephrine-equivalent dose, with vasopressin held out

`labels.norepinephrine_equivalents` returns one row per administration record
and folds every agent, vasopressin included, into a single rate. That is right
for asking how much vasoactive support a patient is on. It is wrong for asking
what the clinician DID, which is what the action-response encoding needs:

  * VASOPRESSIN IS IN THE SUM, AND ALSO FLAGGED. It runs at a fixed rate and
    is rarely titrated, but starting it is still an escalation decision and
    its pressor effect is real -- so it counts toward the dose, per Goradia,
    and the flag plus a titratable-only series are published beside it. An
    earlier version excluded it from the sum outright, which understated
    vasoactive burden for exactly the sickest patients and would have biased a
    dose-response fit downward.
  * THE SERIES MUST BE REGULAR. Administration records arrive when something
    changed, so differencing them measures charting cadence as much as dose.
    Everything here is resampled onto a fixed grid first.

labels.NOREPINEPHRINE_EQUIVALENTS IS NOT MODIFIED. Seven-plus scripts read it
and every prior NEE figure would move. The conversion used here comes from
config/dynamics.yaml instead, and the two deliberately differ: dopamine is
1/100 there and 1/150 in labels.py, which is a real disagreement between
published tables rather than a bug in either.
"""

from __future__ import annotations

import polars as pl

from clifcs.clif import scan
from clifcs.labels import DEFAULT_WEIGHT_KG, STOPPED_ACTIONS, _weights


def agent_rates(data_dir, anchors, cfg, logger):
    """
    one row per administration record: agent, time, and its NEE contribution

    Deliberately a re-implementation of the conversion rather than a call into
    `labels.norepinephrine_equivalents`, because that function sums across
    agents before returning and the sum is exactly what has to be split here.
    The unit handling, the explicit-stop rule and the weight fallback are kept
    identical to it; only the factor table and the grouping differ.
    """
    conversion = {
        agent: dict(units) for agent, units in cfg.nee.conversion.items()
    }
    vaso = list(cfg.nee.vasopressin_agents)
    # Vasopressin's factors live under their own key so the sum can be built
    # with or without them from one table, rather than needing two tables that
    # could drift apart.
    if cfg.nee.get("vasopressin") is not None:
        for agent in vaso:
            conversion.setdefault(agent, dict(cfg.nee.vasopressin))
    wanted = sorted(set(conversion) | set(vaso))

    weights = _weights(data_dir, anchors)
    meds = (
        scan(data_dir, "clif_medication_admin_continuous_converted",
             dttm_cols=("admin_dttm",))
        .filter(pl.col("med_category").is_in(wanted))
        .join(anchors.lazy().select("hospitalization_id"),
              on="hospitalization_id", how="inner")
        .join(weights.lazy(), on="hospitalization_id", how="left")
        .with_columns(
            weight_kg=pl.col("weight_kg").fill_null(DEFAULT_WEIGHT_KG),
            dose=pl.col("med_dose_converted").fill_null(pl.col("med_dose")),
            unit=pl.col("med_dose_unit_converted").fill_null(
                pl.col("med_dose_unit")
            ),
        )
        # an explicit stop means the rate goes to zero, not that the last rate
        # persists -- without this a stopped pressor contributes forever
        .with_columns(
            dose=pl.when(
                pl.col("mar_action_category").is_in(list(STOPPED_ACTIONS))
            ).then(0.0).otherwise(pl.col("dose"))
        )
        .with_columns(
            dose=pl.when(pl.col("unit") == "mcg/min")
            .then(pl.col("dose") / pl.col("weight_kg"))
            .otherwise(pl.col("dose")),
            unit=pl.when(pl.col("unit") == "mcg/min")
            .then(pl.lit("mcg/kg/min")).otherwise(pl.col("unit")),
        )
        .collect()
    )
    if meds.is_empty():
        logger.warning("no vasoactive administration records")
        return meds

    factor = pl.lit(0.0)
    for agent, units in conversion.items():
        for unit, value in units.items():
            factor = (
                pl.when(
                    (pl.col("med_category") == agent) & (pl.col("unit") == unit)
                )
                .then(pl.lit(float(value)))
                .otherwise(factor)
            )
    # `running` is carried separately from `nee` because VASOPRESSIN HAS NO
    # CONVERSION FACTOR BY DESIGN -- it is held out of the summed dose -- so
    # its `nee` is zero on every row. Using `nee > 0` to detect that it was
    # running therefore reported nobody on vasopressin at any site, which is
    # how this was caught.
    out = meds.with_columns(
        nee=(pl.col("dose").fill_null(0.0) * factor),
        running=(pl.col("dose").fill_null(0.0) > 0),
        is_vasopressin=pl.col("med_category").is_in(vaso),
    ).select(
        "hospitalization_id", "admin_dttm", "med_category", "dose", "unit",
        "nee", "running", "is_vasopressin",
    )
    unconverted = out.filter(
        ~pl.col("is_vasopressin") & (pl.col("nee") == 0) & (pl.col("dose") > 0)
    )
    if unconverted.height:
        # A nonzero dose that converted to a zero rate means the unit was not
        # in the table. Silently treating it as no pressor would understate
        # every NEE for that patient, so it is named rather than dropped.
        pairs = unconverted.group_by("med_category", "unit").len().sort(
            "len", descending=True
        )
        logger.warning(
            f"{unconverted.height:,} records had a dose but no conversion "
            f"factor: {pairs.head(6).to_dicts()}"
        )
    return out


def hourly_nee(rates, anchors, cfg, end_hours, logger):
    """
    NEE and the vasopressin flag on a regular grid, per patient

    Backward as-of per agent and then summed, never a single join keyed only on
    the patient: that would return whichever agent was charted last rather than
    everything running concurrently. The same trap `state.trajectory_state`
    documents.

    A grid point with no prior record inside the tolerance gets NEE zero rather
    than null. That is a real asymmetry with the MAP side, and it is
    deliberate: an absent infusion record means no infusion, whereas an absent
    MAP means nobody measured it. Charting a running pressor is not optional in
    any of these units.
    """
    step = int(cfg.nee.resolution_minutes)
    tolerance = float(cfg.nee.tolerance_hours) * 3600
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
    if rates.is_empty():
        return grid.with_columns(
            nee=pl.lit(0.0), nee_titratable=pl.lit(0.0),
            on_vasopressin=pl.lit(False),
            n_titratable=pl.lit(0, dtype=pl.Int64),
        )

    include_vaso = bool(cfg.nee.get("include_vasopressin_in_nee", True))
    out = grid.with_columns(
        nee=pl.lit(0.0), nee_titratable=pl.lit(0.0),
        n_titratable=pl.lit(0, dtype=pl.Int64), on_vasopressin=pl.lit(False),
    )
    for agent in rates["med_category"].unique().to_list():
        part = (
            rates.filter(pl.col("med_category") == agent)
            .select("hospitalization_id", "admin_dttm", "nee", "running",
                    "is_vasopressin")
            .sort("admin_dttm")
        )
        is_vaso = bool(part["is_vasopressin"][0])
        out = (
            out.sort("event_dttm")
            .join_asof(
                part.select("hospitalization_id", "admin_dttm",
                            _rate=pl.col("nee"), _running=pl.col("running")),
                left_on="event_dttm", right_on="admin_dttm",
                by="hospitalization_id", strategy="backward",
            )
            .with_columns(
                _fresh=(
                    (pl.col("event_dttm") - pl.col("admin_dttm"))
                    .dt.total_seconds() <= tolerance
                )
            )
            .with_columns(
                _rate=pl.when(pl.col("_fresh"))
                .then(pl.col("_rate")).otherwise(0.0).fill_null(0.0),
                _running=pl.when(pl.col("_fresh"))
                .then(pl.col("_running")).otherwise(False).fill_null(False),
            )
        )
        if is_vaso:
            out = out.with_columns(
                on_vasopressin=pl.col("on_vasopressin") | pl.col("_running")
            )
            if include_vaso:
                out = out.with_columns(nee=pl.col("nee") + pl.col("_rate"))
        else:
            out = out.with_columns(
                nee=pl.col("nee") + pl.col("_rate"),
                nee_titratable=pl.col("nee_titratable") + pl.col("_rate"),
                n_titratable=pl.col("n_titratable")
                + (pl.col("_rate") > 0).cast(pl.Int64),
            )
        out = out.drop("_rate", "_running", "_fresh", "admin_dttm",
                       strict=False)

    logger.info(
        f"  hourly NEE: {out.height:,} patient-hours, "
        f"{(out['nee'] > 0).mean():.1%} on any vasoactive, "
        f"{(out['nee_titratable'] > 0).mean():.1%} on a titratable one, "
        f"{out['on_vasopressin'].mean():.1%} on vasopressin"
        + ("" if include_vaso else "  [vasopressin NOT in the sum]")
    )
    return out


#: agents kept as their own hourly column. Doses stay in their NATIVE units --
#: mcg/kg/min for the catecholamines, u/min for vasopressin -- rather than
#: being converted to norepinephrine equivalents, because the question "how far
#: does the pressure move when I turn the norepinephrine up" is asked in the
#: units the drug is ordered in. Converting first would express a change in
#: norepinephrine as a change in a composite and make the x-axis unreadable at
#: the bedside.
PANEL_AGENTS = (
    "norepinephrine", "vasopressin", "epinephrine", "phenylephrine",
)


def hourly_by_agent(rates, anchors, cfg, end_hours, logger,
                    agents=PANEL_AGENTS):
    """
    one hourly column per agent, in that agent's own dose units

    `hourly_nee` sums every agent into one rate, which is right for asking how
    much support a patient is on and wrong for asking what a change in ONE drug
    did -- the sum cannot distinguish turning the norepinephrine up from
    starting vasopressin.

    Backward as-of per agent with the same staleness rule as the summed
    version: beyond the tolerance a dose reads zero rather than carrying
    forward, because an absent infusion record means no infusion.
    """
    step = int(cfg.nee.resolution_minutes)
    tolerance = float(cfg.nee.tolerance_hours) * 3600
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
    out = grid
    for agent in agents:
        column = f"dose_{agent}"
        part = rates.filter(pl.col("med_category") == agent) \
            if not rates.is_empty() else rates
        if part.is_empty():
            out = out.with_columns(pl.lit(0.0).alias(column))
            logger.warning(f"  no {agent} records; column is all zero")
            continue
        part = (
            part.select("hospitalization_id", "admin_dttm",
                        _dose=pl.col("dose"))
            .sort("admin_dttm")
        )
        out = (
            out.sort("event_dttm")
            .join_asof(
                part, left_on="event_dttm", right_on="admin_dttm",
                by="hospitalization_id", strategy="backward",
            )
            .with_columns(
                **{column: pl.when(
                    (pl.col("event_dttm") - pl.col("admin_dttm"))
                    .dt.total_seconds() <= tolerance
                ).then(pl.col("_dose")).otherwise(0.0).fill_null(0.0)}
            )
            .drop("_dose", "admin_dttm", strict=False)
        )
    running = {
        agent: float((out[f"dose_{agent}"] > 0).mean()) for agent in agents
    }
    logger.info(
        "  hourly per-agent coverage: "
        + ", ".join(f"{a} {s:.1%}" for a, s in running.items())
    )
    return out

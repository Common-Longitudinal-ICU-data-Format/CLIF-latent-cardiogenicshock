#!/usr/bin/env python3

"""
reading the dose-change / pressure-response plane as clinical regions

The ratio |dMAP| / |dNE| fails here because its denominator is usually tiny: an
hour with a 0.01 change in norepinephrine and five millimetres of measurement
noise scores five hundred times more "sensitive" than one with a 0.1 change and
the same swing. Both cohorts confirmed it -- the observed spread of the log
ratio sat BELOW a permutation null.

Regions keep what the ratio was reaching for and drop the division. The sign of
the dose change and the size of the pressure response together carry the
clinical reading:

  highly responsive     the pressure rose substantially on no extra drug, or
                        on a minimal increase -- the best cell on the plane
  responsive            escalated, and the pressure rose
  nonresponsive         escalated, and the pressure did not move
  deteriorating         escalated, and the pressure fell anyway
  wean held             de-escalated, pressure flat or up
  wean with a dip       de-escalated, pressure fell but by less than `big`
  failed wean           de-escalated, pressure fell substantially

THE ASYMMETRY IS DELIBERATE. A small dip after a wean is tolerable and a large
one is not, so the wean side is split at `big_mmhg` while the escalation side
is split at the deadband -- the question on one side is "did it work" and on
the other "did it cost too much".

NONE OF THESE LABELS IS A CAUSAL CLAIM. "Deteriorating" describes an hour in
which the dose went up and the pressure went down; it does not say the dose
failed, because the clinician escalated precisely when the pressure was
falling. Confounding by indication runs through every cell.
"""

from __future__ import annotations

import numpy as np
import polars as pl

#: region -> how to read it, in order from best to worst. The order is used
#: for plotting and for the composition tables, so it is defined once here.
REGION_ORDER = (
    "highly responsive",
    "responsive",
    "wean held",
    "wean with a dip",
    "nonresponsive",
    "failed wean",
    "deteriorating",
)


def label_regions(events, cfg):
    """
    one region per dose-change hour, from the sign of d_ne and the size of d_map

    `events` needs d_ne and d_map. Rows keep every original column so the
    region can be crossed with regimen, site or time afterwards.
    """
    deadband = float(cfg.response.deadband_mmhg)
    big = float(cfg.regions.big_mmhg)
    minimal = float(cfg.regions.minimal_increase)

    rose_a_lot = pl.col("d_map") >= big
    rose = pl.col("d_map") > deadband
    fell = pl.col("d_map") < -deadband
    fell_a_lot = pl.col("d_map") <= -big
    went_up = pl.col("d_ne") > 0
    barely_up = pl.col("d_ne") <= minimal

    return events.with_columns(
        region=pl.when(rose_a_lot & (barely_up | ~went_up))
        # the pressure recovered substantially on little or no extra drug
        .then(pl.lit("highly responsive"))
        .when(went_up & rose)
        .then(pl.lit("responsive"))
        .when(went_up & fell)
        .then(pl.lit("deteriorating"))
        .when(went_up)
        .then(pl.lit("nonresponsive"))
        .when(~went_up & fell_a_lot)
        .then(pl.lit("failed wean"))
        .when(~went_up & fell)
        .then(pl.lit("wean with a dip"))
        .otherwise(pl.lit("wean held"))
        .alias("region")
    )


def dose_bins(events, cfg):
    """
    the MAP response averaged inside bins of |d_ne|, separately up and down

    THE BINNED ANSWER TO THE QUESTION THE RATIO COULD NOT ANSWER. If the
    response scales with the size of the dose change, the mean response rises
    across bins; if it is flat, the dose change tells you only its direction.
    No denominator, so no instability where the dose barely moved.
    """
    edges = [float(v) for v in cfg.regions.dose_bins]
    rows = []
    for direction, predicate in (
        ("increase", pl.col("d_ne") > 0), ("decrease", pl.col("d_ne") < 0)
    ):
        part = events.filter(predicate)
        for low, high in zip(edges[:-1], edges[1:]):
            cell = part.filter(
                (pl.col("d_ne").abs() >= low) & (pl.col("d_ne").abs() < high)
            )
            if cell.height < 50:
                continue
            values = cell["d_map"].to_numpy()
            rows.append({
                "direction": direction,
                "dose_change_from": low,
                "dose_change_to": high if high < 999 else None,
                "hours": cell.height,
                "patients": cell["hospitalization_id"].n_unique(),
                "mean_d_map": float(values.mean()),
                "ci_low": float(
                    values.mean() - 1.96 * values.std() / np.sqrt(len(values))
                ),
                "ci_high": float(
                    values.mean() + 1.96 * values.std() / np.sqrt(len(values))
                ),
                "median_d_map": float(np.median(values)),
                "sd_d_map": float(values.std()),
            })
    return pl.DataFrame(rows)


def by_starting_map(events, cfg):
    """the same binned response, split by the MAP the hour started at"""
    edges = [float(v) for v in cfg.regions.starting_map_bins]
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        stratum = events.filter(
            (pl.col("map") >= low) & (pl.col("map") < high)
        )
        if stratum.height < 200:
            continue
        for direction, predicate in (
            ("increase", pl.col("d_ne") > 0), ("decrease", pl.col("d_ne") < 0)
        ):
            cell = stratum.filter(predicate)
            if cell.height < 50:
                continue
            values = cell["d_map"].to_numpy()
            rows.append({
                "starting_map_from": low,
                "starting_map_to": high if high < 299 else None,
                "direction": direction,
                "hours": cell.height,
                "mean_d_map": float(values.mean()),
                "ci_low": float(
                    values.mean() - 1.96 * values.std() / np.sqrt(len(values))
                ),
                "ci_high": float(
                    values.mean() + 1.96 * values.std() / np.sqrt(len(values))
                ),
            })
    return pl.DataFrame(rows)


def patient_composition(labelled, cfg, logger):
    """
    each patient's share of hours in each region, with enough hours to mean it

    This is the PATIENT-level question, which is not the hour-level one. An
    earlier test found no between-patient variation in the binary rate of
    "pressure moved the way the dose was pushed" -- among patients with ten or
    more titrations the observed spread was smaller than binomial noise around
    one shared rate. Regions use the MAGNITUDE of the response and seven cells
    rather than two, so they carry strictly more information and the question
    is worth asking again rather than assumed settled.
    """
    floor = int(cfg.regions.min_hours_per_patient)
    counts = (
        labelled.group_by("hospitalization_id")
        .agg(
            hours=pl.len(),
            **{
                f"n_{region.replace(' ', '_')}":
                (pl.col("region") == region).sum()
                for region in REGION_ORDER
            },
        )
        .filter(pl.col("hours") >= floor)
    )
    shares = counts.with_columns([
        (pl.col(f"n_{region.replace(' ', '_')}") / pl.col("hours"))
        .alias(region.replace(" ", "_"))
        for region in REGION_ORDER
    ])
    logger.info(
        f"  {shares.height:,} patients with at least {floor} labelled "
        f"titration-hours (median {shares['hours'].median():.0f} hours each)"
    )
    return shares


def between_patient_variance(shares, logger, seed=42, rounds=500):
    """
    do patients differ in composition by more than multinomial noise

    THE TEST THAT DECIDES WHETHER A PATIENT-LEVEL SENSITIVITY EXISTS. Each
    patient's region counts are compared against a null in which every patient
    draws from the same cohort-wide region probabilities with their own number
    of hours. If the observed spread does not exceed that null, the apparent
    differences between patients are the differences between small samples.

    The permutation keeps each patient's hour COUNT and reassigns which
    regions those hours fell in, which is the right null: it holds fixed how
    much was observed about each patient and breaks only the link between a
    patient and their outcomes.
    """
    regions = [r.replace(" ", "_") for r in REGION_ORDER]
    counts = np.column_stack([
        shares[f"n_{region}"].to_numpy() for region in regions
    ]).astype(float)
    totals = shares["hours"].to_numpy().astype(float)
    pooled = counts.sum(axis=0) / max(totals.sum(), 1.0)

    observed = np.array([
        float(np.var(counts[:, i] / totals)) for i in range(len(regions))
    ])
    rng = np.random.default_rng(seed)
    null = np.zeros((rounds, len(regions)))
    for draw in range(rounds):
        simulated = np.array([
            rng.multinomial(int(n), pooled) for n in totals
        ], dtype=float)
        null[draw] = [
            float(np.var(simulated[:, i] / totals)) for i in range(len(regions))
        ]

    rows = []
    logger.info("between-patient variance in region share, against a "
                "multinomial null:")
    for i, region in enumerate(REGION_ORDER):
        p = float((null[:, i] >= observed[i]).mean())
        excess = float(observed[i] - null[:, i].mean())
        rows.append({
            "region": region,
            "cohort_share": float(pooled[i]),
            "observed_variance": float(observed[i]),
            "null_variance": float(null[:, i].mean()),
            "excess_variance": excess,
            "excess_sd": float(np.sqrt(max(excess, 0.0))),
            "p_exceeds_null": p,
        })
        logger.info(
            f"    {region:<20} share {pooled[i]:>6.1%}  observed var "
            f"{observed[i]:.5f}  null {null[:, i].mean():.5f}  "
            f"excess sd {np.sqrt(max(excess, 0.0)):.4f}  p={p:.3f}"
        )
    return pl.DataFrame(rows)


def transitions(labelled, logger, max_gap_hours: float | None = None):
    """
    consecutive region pairs within a patient, in order

    `gap_hours` is the time between the two titrations, which varies from one
    hour to several days. It is carried rather than collapsed, because a
    "transition" across two days is a different object from one across an hour
    and pooling them would describe neither.
    """
    ordered = labelled.sort(["hospitalization_id", "offset_hours"])
    pairs = ordered.with_columns(
        next_region=pl.col("region").shift(-1).over("hospitalization_id"),
        next_hours=pl.col("offset_hours").shift(-1).over("hospitalization_id"),
    ).filter(pl.col("next_region").is_not_null()).with_columns(
        gap_hours=pl.col("next_hours") - pl.col("offset_hours")
    )
    if max_gap_hours is not None:
        before = pairs.height
        pairs = pairs.filter(pl.col("gap_hours") <= max_gap_hours)
        logger.info(
            f"  {before - pairs.height:,} of {before:,} pairs dropped for "
            f"spanning more than {max_gap_hours:.0f}h"
        )
    logger.info(
        f"  {pairs.height:,} consecutive titration pairs, median gap "
        f"{pairs['gap_hours'].median():.1f}h "
        f"(IQR {pairs['gap_hours'].quantile(0.25):.1f}"
        f"-{pairs['gap_hours'].quantile(0.75):.1f})"
    )
    return pairs


def transition_matrix(pairs, alpha: float = 1.0):
    """
    row-normalised region-to-region probabilities with Dirichlet smoothing

    `alpha` is added to every cell before normalising, so a region seen a
    handful of times does not produce a row of zeros and ones. Phase 3 needs
    the same treatment for the same reason, which is why the smoothing lives
    here rather than in a figure.
    """
    rows = []
    for source in REGION_ORDER:
        part = pairs.filter(pl.col("region") == source)
        total = part.height
        row = {"from_region": source, "n": total}
        counts = {
            target: part.filter(pl.col("next_region") == target).height
            for target in REGION_ORDER
        }
        denominator = total + alpha * len(REGION_ORDER)
        for target in REGION_ORDER:
            row[target] = (counts[target] + alpha) / denominator
        rows.append(row)
    return pl.DataFrame(rows)


def persistence_test(pairs, logger, seed=42, rounds=400):
    """
    does being in a region make the NEXT titration land in it again

    THE NULL IS A WITHIN-PATIENT SHUFFLE, which is the whole point. Shuffling
    each patient's own sequence preserves how much that patient favours each
    region and destroys only the order, so an excess above this null cannot be
    explained by "this patient is often in wean held". It has to mean that
    being in a region now changes where the next titration lands -- which is
    what a state has to do before a Markov chain over these regions means
    anything.

    A cohort-wide shuffle would be the wrong null: patients differ in
    composition (five of seven regions vary beyond multinomial noise), so
    pooling them would show "persistence" that is only composition.
    """
    rng = np.random.default_rng(seed)
    observed = float(
        (pairs["region"] == pairs["next_region"]).mean()
    )

    sequences = (
        pairs.sort(["hospitalization_id", "offset_hours"])
        .group_by("hospitalization_id", maintain_order=True)
        .agg(path=pl.col("region"))
    )["path"].to_list()
    null = np.empty(rounds)
    for draw in range(rounds):
        same = 0
        total = 0
        for path in sequences:
            if len(path) < 2:
                continue
            shuffled = rng.permutation(np.asarray(path, dtype=object))
            same += int((shuffled[:-1] == shuffled[1:]).sum())
            total += len(shuffled) - 1
        null[draw] = same / max(total, 1)
    p = float((null >= observed).mean())
    logger.info(
        f"  stays in the same region: observed {observed:.1%}, "
        f"within-patient shuffle null {null.mean():.1%} "
        f"+/- {null.std():.1%}, p={p:.4f}"
    )
    return observed, float(null.mean()), float(null.std()), p


#: The HOURLY state space. Every hour gets exactly one label, including hours
#: in which nothing was changed -- which the event-based `REGION_ORDER` drops
#: entirely, and which are 85% of hours. Without them a "transition" spans
#: whatever gap happened to separate two titrations, median 2h but up to days,
#: so the sequence is neither a Markov chain nor a clinical narrative.
HOURLY_REGION_ORDER = (
    "highly responsive",
    "responsive",
    "nonresponsive",
    "deteriorating",
    "wean held",
    "wean with a dip",
    "failed wean",
    "stable hold",
    "drifting hold",
    "off pressors",
)


def label_hourly_regions(hourly, cfg):
    """
    one region per patient-hour, holds and off-pressor hours included

    `hourly` is the full grid from 29, not the dose-change events: every hour
    is labelled, so consecutive labels are one hour apart by construction and
    a transition means the same thing everywhere.

    The three added states are not decoration. A hold with the pressure
    drifting down is a distinct clinical situation from a hold with it steady,
    and an hour off pressors is not a dose-response observation at all -- but
    all three have to be in the state space or the chain cannot represent a
    patient sitting still, which is most of what patients do.
    """
    deadband = float(cfg.response.deadband_mmhg)
    big = float(cfg.regions.big_mmhg)
    minimal = float(cfg.regions.minimal_increase)
    floor = float(cfg.action.absolute_minimum)
    adequate = float(cfg.map.adequate)

    frame = hourly.sort(["hospitalization_id", "offset_hours"]).with_columns(
        d_ne=pl.col("dose_norepinephrine").diff().over("hospitalization_id"),
        d_map=(pl.col("map").shift(-1) - pl.col("map"))
        .over("hospitalization_id"),
        gap_back=pl.col("offset_hours").diff().over("hospitalization_id"),
        gap_forward=(pl.col("offset_hours").shift(-1) - pl.col("offset_hours"))
        .over("hospitalization_id"),
    ).filter(
        (pl.col("gap_back") == 1.0) & (pl.col("gap_forward") == 1.0)
        & pl.col("d_ne").is_not_null() & pl.col("d_map").is_not_null()
    )

    on_pressors = pl.col("dose_norepinephrine") > 0
    went_up = pl.col("d_ne") >= floor
    came_down = pl.col("d_ne") <= -floor
    rose_a_lot = pl.col("d_map") >= big
    rose = pl.col("d_map") > deadband
    fell = pl.col("d_map") < -deadband
    fell_a_lot = pl.col("d_map") <= -big
    barely_up = pl.col("d_ne") <= minimal

    return frame.with_columns(
        region=pl.when(~on_pressors & ~went_up)
        # no drug running and none started: not a dose-response hour at all
        .then(pl.lit("off pressors"))
        .when(rose_a_lot & (barely_up | came_down))
        .then(pl.lit("highly responsive"))
        .when(went_up & rose)
        .then(pl.lit("responsive"))
        .when(went_up & fell)
        .then(pl.lit("deteriorating"))
        .when(went_up)
        .then(pl.lit("nonresponsive"))
        .when(came_down & fell_a_lot)
        .then(pl.lit("failed wean"))
        .when(came_down & fell)
        .then(pl.lit("wean with a dip"))
        .when(came_down)
        .then(pl.lit("wean held"))
        .when(fell)
        .then(pl.lit("drifting hold"))
        .otherwise(pl.lit("stable hold"))
        .alias("region"),
        map_adequate=pl.col("map") >= adequate,
    )


def lagged_transitions(labelled, logger, lag_hours: int = 2,
                       order=HOURLY_REGION_ORDER):
    """
    state at hour t against state at hour t+lag, with no shared measurement

    LAG 2 IS THE MINIMUM THAT AVOIDS THE OVERLAP, and the overlap is not a
    detail. A state at hour t is built from map(t) and map(t+1); a state at
    t+1 is built from map(t+1) and map(t+2). They SHARE map(t+1), so a single
    low reading makes hour t read as deterioration and hour t+1 as recovery,
    with no physiology involved. Measured on the event-based encoding, the
    correlation between consecutive dMAP values was -0.401 at the overlapping
    lag and -0.011 and -0.009 at non-overlapping ones: the artefact is
    isolated and it is large.

    At lag 2 the windows are disjoint on BOTH axes -- state t reads
    map(t), map(t+1) and ne(t-1), ne(t); state t+2 reads map(t+2), map(t+3)
    and ne(t+1), ne(t+2) -- so a shared reading cannot manufacture a
    transition in either direction.

    Anything inferred about persistence from the adjacent lag is unusable,
    which includes the earlier conclusion that these regions have no dynamics.
    """
    ordered = labelled.sort(["hospitalization_id", "offset_hours"])
    pairs = ordered.with_columns(
        next_region=pl.col("region").shift(-lag_hours)
        .over("hospitalization_id"),
        next_hours=pl.col("offset_hours").shift(-lag_hours)
        .over("hospitalization_id"),
        next_d_map=pl.col("d_map").shift(-lag_hours).over("hospitalization_id"),
    ).filter(
        pl.col("next_region").is_not_null()
        # exactly `lag_hours` apart: a shift of two rows is only a shift of two
        # hours when no hour between them was dropped
        & ((pl.col("next_hours") - pl.col("offset_hours")) == float(lag_hours))
    )
    overlap = float(
        np.corrcoef(pairs["d_map"], pairs["next_d_map"])[0, 1]
    ) if pairs.height > 100 else float("nan")
    logger.info(
        f"  {pairs.height:,} transitions at lag {lag_hours}h; "
        f"corr(dMAP_t, dMAP_t+{lag_hours}) = {overlap:+.3f} "
        "(near zero confirms the windows do not share a reading)"
    )
    return pairs


#: one character per hourly state, for writing a patient's course as a string.
#: Case carries meaning: capitals are the consequential states, lower case the
#: quiet ones, so a sequence reads at a glance.
STATE_SYMBOL = {
    "highly responsive": "H",
    "responsive": "R",
    "nonresponsive": "N",
    "deteriorating": "D",
    "wean held": "W",
    "wean with a dip": "w",
    "failed wean": "F",
    "stable hold": "S",
    "drifting hold": "d",
    "off pressors": "o",
}
SYMBOL_STATE = {v: k for k, v in STATE_SYMBOL.items()}


def trim_trailing_off(sequence: str, off: str = "o", keep: int = 1) -> str:
    """
    collapse a TRAILING run of off-pressor hours to at most `keep`

    A patient who recovers and stays off pressors contributes one hour of
    useful information -- that they came off -- and then a hundred hours
    saying the same thing again. Those hours are 78% of the grid and they
    dominate every distance, every plurality map and every clustering, while
    describing a patient who has already left the problem.

    ONLY THE TRAILING RUN IS TOUCHED. Off-pressor hours sitting BETWEEN periods
    of support are a relapse, which is the opposite of uninformative, so they
    are kept in full. The distinction is the whole point: "came off and stayed
    off" compresses to one symbol, "came off and went back on" does not
    compress at all.
    """
    if not sequence:
        return sequence
    end = len(sequence)
    while end > 0 and sequence[end - 1] == off:
        end -= 1
    trailing = len(sequence) - end
    if trailing <= keep:
        return sequence
    return sequence[:end] + off * keep


def state_sequences(labelled, logger, trim=True, keep=1):
    """one string per patient, optionally with the trailing off-run collapsed"""
    ordered = labelled.sort(["hospitalization_id", "offset_hours"])
    grouped = (
        ordered.with_columns(
            symbol=pl.col("region").replace_strict(
                STATE_SYMBOL, default="?", return_dtype=pl.String
            )
        )
        .group_by("hospitalization_id", maintain_order=True)
        .agg(sequence=pl.col("symbol").str.join(""))
    )
    raw = grouped["sequence"].to_list()
    if not trim:
        return grouped
    trimmed = [trim_trailing_off(s, keep=keep) for s in raw]
    before = sum(len(s) for s in raw)
    after = sum(len(s) for s in trimmed)
    logger.info(
        f"  trailing off-pressor runs collapsed to {keep}: "
        f"{before:,} hours -> {after:,} ({1 - after / max(before, 1):.1%} "
        "removed); relapse gaps kept in full"
    )
    return grouped.with_columns(
        sequence=pl.Series(trimmed),
        hours_raw=pl.Series([len(s) for s in raw]),
        hours_trimmed=pl.Series([len(s) for s in trimmed]),
    )

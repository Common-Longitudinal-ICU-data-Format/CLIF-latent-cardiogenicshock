#!/usr/bin/env python3

"""
39_sequence_alignment.py -- where two courses stop being the same illness

BPE answered "which substrings recur" and the answer was short and ubiquitous:
the common words are two and three hours long, most of the cohort carries them,
and a motif shared by most of the cohort cannot be where outcomes part. This
asks the complementary question, the one genomics asks of two homologous
sequences -- align them from their common origin and read off the point at
which the alignment stops paying.

    python code/39_sequence_alignment.py --subset cardiogenic

THREE CHOICES WORTH ARGUING WITH.

  The alignment is ANCHORED at shock onset and free at both ends. Every string
  starts at the same clinical event, so a free start gap would let the aligner
  slide one course forward in time until it matched -- which is the claim under
  test, not an allowance to make. Both ends are free, so the endpoint is the
  divergence point by construction.

  The substitution matrix is MEASURED, not chosen. Edit distance scores
  `stable hold` vs `drifting hold` the same as `wean held` vs `deteriorating`.
  Here a substitution costs what the two states' behaviour differs by: the
  Jensen-Shannon distance between their forward and backward neighbour
  distributions. A BLOSUM-style count over aligned columns was tried first and
  collapsed -- see `clifcs.align.similarity_matrix` for why.

  Courses are CAPPED at --cap-hours. The cohort runs to 167 hours and the DP is
  quadratic in length, but the real reason is the question: divergence in the
  first days of shock is the thing being asked about, and a cap keeps a pair's
  score from being dominated by a week of recovery neither patient was in.

WHAT A DIVERGENCE POINT DOES NOT TELL YOU. It is the hour two RECORDED courses
stopped agreeing, not the hour one patient's illness changed. Nothing here
identifies a cause, and the pair was selected into its treatment stratum by
everything that happened before the divergence.

RECORD LENGTH IS THE STANDING THREAT TO EVERY NUMBER BELOW, because a course
cannot diverge after it ends and courses end when patients die. Two guards run
against it and both are reported rather than assumed: `--min-tail` requires
both patients to have further hours on record past the divergence, so the event
is observed and not censored, and the random control is LENGTH-MATCHED to the
treatment-matched set, so the two pair sets can be compared at all. The
correlation between divergence hour and the shorter course's length is logged
every run; if it is large the run is measuring length, whatever else it says.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.align import (
    align_many, background_frequencies, similarity_matrix,
    transition_distributions,
)
from clifcs.config import CONFIG_HOME
from clifcs.regions import HOURLY_REGION_ORDER, STATE_SYMBOL, trim_trailing_off
from clifcs.treatment import add_treatment_symbol, collapse

ALPHABET = "".join(STATE_SYMBOL[state] for state in HOURLY_REGION_ORDER)


def build_strings(labelled, logger, cap_hours):
    """one response string and one treatment string per patient"""
    frame = add_treatment_symbol(labelled).with_columns(
        response_symbol=pl.col("region").replace_strict(
            STATE_SYMBOL, default="?", return_dtype=pl.String
        )
    ).sort(["hospitalization_id", "offset_hours"])
    grouped = frame.group_by("hospitalization_id", maintain_order=True).agg(
        response=pl.col("response_symbol").str.join(""),
        treatment=pl.col("treatment_symbol").str.join(""),
    )
    responses = [
        trim_trailing_off(s, keep=1)[:cap_hours]
        for s in grouped["response"].to_list()
    ]
    grouped = grouped.with_columns(
        response=pl.Series(responses),
        hours=pl.Series([len(s) for s in responses]),
        treatment_collapsed=pl.col("treatment").map_elements(
            collapse, return_dtype=pl.String
        ),
    ).filter(pl.col("hours") >= 4)
    logger.info(
        f"  {grouped.height:,} patients with at least 4 hours after trimming "
        f"and the {cap_hours}h cap; median {int(grouped['hours'].median())}h"
    )
    return grouped


def sample_pairs(grouped, logger, seed, n_matched, min_group, tolerance=3):
    """
    pairs matched on the treatment course, and a LENGTH-MATCHED random control

    The control is the point. A divergence-hour distribution on its own says
    nothing -- any two strings over a 10-letter alphabet stop agreeing
    somewhere. What is interpretable is matched against random.

    THE CONTROL HAD TO BE LENGTH-MATCHED, not merely random. Patients who share
    a collapsed treatment course have systematically shorter records -- median
    13 hours against 20 for random pairs -- and a course cannot diverge after it
    ends, so an unmatched control made treatment-matched pairs look like they
    diverged earlier when all that differed was how long they were observed.
    Each control pair is therefore drawn to have the same two course lengths,
    within `tolerance` hours, as a treatment-matched pair it stands in for.
    """
    rng = np.random.default_rng(seed)
    lengths = np.array(grouped["hours"].to_list())
    courses = grouped["treatment_collapsed"].to_list()

    buckets = {}
    for index, course in enumerate(courses):
        buckets.setdefault(course, []).append(index)
    usable = {k: v for k, v in buckets.items() if len(v) >= min_group}
    logger.info(
        f"  {len(usable):,} collapsed treatment courses shared by at least "
        f"{min_group} patients, covering "
        f"{sum(len(v) for v in usable.values()):,} patients"
    )
    if not usable:
        return {}

    weights = np.array([len(v) for v in usable.values()], dtype=float)
    keys = list(usable)
    matched = []
    for which in rng.choice(len(keys), size=n_matched, p=weights / weights.sum()):
        members = usable[keys[which]]
        a, b = rng.choice(len(members), size=2, replace=False)
        matched.append((members[a], members[b]))

    # one length-matched control pair per treatment-matched pair
    order = np.argsort(lengths)
    sorted_lengths = lengths[order]
    control = []
    for a, b in matched:
        target = sorted((int(lengths[a]), int(lengths[b])))
        picks = []
        for want in target:
            low = int(np.searchsorted(sorted_lengths, want - tolerance, "left"))
            high = int(np.searchsorted(sorted_lengths, want + tolerance, "right"))
            if high <= low:
                picks = []
                break
            picks.append(int(order[rng.integers(low, high)]))
        if len(picks) == 2 and picks[0] != picks[1]:
            control.append((picks[0], picks[1]))

    logger.info(
        f"  {len(matched):,} treatment-matched pairs, {len(control):,} "
        f"length-matched control pairs (+/-{tolerance}h on both courses)"
    )
    return {"matched treatment": matched, "length-matched random": control}


def kaplan_meier(time, event):
    """
    survival of "still the same illness" against hour

    Time-to-divergence is right-censored by construction: when a record ends
    while two courses are still agreeing, all that is known is that they had
    not parted by then. Kaplan-Meier is the estimator that uses those pairs
    instead of discarding them, which matters here because they are three
    quarters of the data.
    """
    order = np.argsort(time, kind="stable")
    time, event = np.asarray(time)[order], np.asarray(event)[order]
    at_risk = len(time)
    survival, times = 1.0, []
    curve = []
    index = 0
    while index < len(time):
        now = time[index]
        stop = index
        deaths = 0
        while stop < len(time) and time[stop] == now:
            deaths += int(event[stop])
            stop += 1
        if deaths > 0 and at_risk > 0:
            survival *= 1.0 - deaths / at_risk
            times.append(float(now))
            curve.append(survival)
        at_risk -= stop - index
        index = stop
    return np.array(times), np.array(curve)


def divergence_quantile(times, curve, fraction=0.25):
    """
    the hour by which `fraction` of pairs have parted

    NOT the median. Only 14-21% of pairs ever diverge before one record ends,
    so the survival curve never reaches 0.5 and a median divergence time is
    undefined -- it was reported as nan on the first run that used it. An early
    quantile is the quantity the data can actually support.
    """
    reached = np.where(curve <= 1.0 - fraction)[0]
    return float(times[reached[0]]) if len(reached) else float("nan")


def logrank(time_a, event_a, time_b, event_b):
    """two-sample log-rank, comparing whole curves rather than medians"""
    time = np.concatenate([time_a, time_b]).astype(float)
    event = np.concatenate([event_a, event_b]).astype(int)
    group = np.concatenate([np.zeros(len(time_a)), np.ones(len(time_b))])
    order = np.argsort(time, kind="stable")
    time, event, group = time[order], event[order], group[order]
    risk_a, risk_b = len(time_a), len(time_b)
    observed = expected = variance = 0.0
    index = 0
    while index < len(time):
        now = time[index]
        stop, deaths, deaths_b = index, 0, 0
        while stop < len(time) and time[stop] == now:
            if event[stop]:
                deaths += 1
                deaths_b += int(group[stop] == 1)
            stop += 1
        total = risk_a + risk_b
        if deaths > 0 and total > 1 and risk_a > 0 and risk_b > 0:
            observed += deaths_b
            expected += deaths * risk_b / total
            variance += (
                deaths * risk_a * risk_b * (total - deaths)
                / (total * total * (total - 1))
            )
        for position in range(index, stop):
            if group[position] == 1:
                risk_b -= 1
            else:
                risk_a -= 1
        index = stop
    if variance <= 0:
        return float("nan"), float("nan")
    from scipy.stats import norm
    z = (observed - expected) / np.sqrt(variance)
    return float(z), float(2.0 * norm.sf(abs(z)))


def summarise(name, results, logger):
    """the reason mix and the divergence-hour distribution for one pair set"""
    reasons = Counter(r["reason"] for r in results)
    total = max(len(results), 1)
    logger.info(f"  {name}: {len(results):,} pairs")
    for reason, count in reasons.most_common():
        logger.info(f"      {reason:<20} {count:>7,}  {count / total:>6.1%}")
    diverged = [r for r in results if r["diverged"]]
    if not diverged:
        logger.warning(f"      no diverging pairs in {name}")
        return diverged, reasons
    hours = np.array([r["left_hour"] for r in diverged], dtype=float)
    identity = np.array([r["identity"] for r in diverged], dtype=float)
    logger.info(
        f"      divergence hour: median {np.median(hours):.0f}, "
        f"IQR {np.percentile(hours, 25):.0f}-{np.percentile(hours, 75):.0f}, "
        f"p90 {np.percentile(hours, 90):.0f}"
    )
    if np.isfinite(identity).any():
        logger.info(
            "      shared-prefix identity: median "
            f"{np.nanmedian(identity):.2f}"
        )
    return diverged, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--cap-hours", type=int, default=72)
    parser.add_argument("--gap-open", type=float, default=4.0)
    parser.add_argument("--gap-extend", type=float, default=1.0)
    parser.add_argument("--min-tail", type=int, default=12,
                        help="hours both courses must still have on "
                             "record past the divergence for it to count "
                             "as observed rather than censored")
    parser.add_argument("--matched-pairs", type=int, default=20000)
    parser.add_argument("--min-group", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "39_sequence_alignment", paths.log_home / "39_sequence_alignment.log"
    )
    _cfg = OmegaConf.load(CONFIG_HOME / "dynamics.yaml")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"subset {args.subset}; cap {args.cap_hours}h; gap open "
        f"{args.gap_open} extend {args.gap_extend}; min tail "
        f"{args.min_tail}h"
    )

    labelled = pl.read_parquet(
        paths.patient_level_home / f"treatment_states_{args.subset}.parquet"
    )
    grouped = build_strings(labelled, logger, args.cap_hours)
    sequences = grouped["response"].to_list()

    logger.info("building the substitution matrix:")
    background = background_frequencies(sequences, ALPHABET)
    matrix = similarity_matrix(sequences, ALPHABET, logger)
    forward, _backward = transition_distributions(sequences, ALPHABET)
    _ = forward
    rows = []
    for i, a in enumerate(ALPHABET):
        row = {"state": HOURLY_REGION_ORDER[i], "symbol": a,
               "background": float(background[i])}
        for j, b in enumerate(ALPHABET):
            row[b] = float(matrix[i, j])
        rows.append(row)
    pl.DataFrame(rows).write_csv(
        tables / f"alignment_substitution_matrix_{args.subset}.csv"
    )
    off = [
        (float(matrix[i, j]), HOURLY_REGION_ORDER[i], HOURLY_REGION_ORDER[j])
        for i in range(len(ALPHABET)) for j in range(i + 1, len(ALPHABET))
    ]
    off.sort(reverse=True)
    logger.info("  most substitutable state pairs (cheapest mismatches):")
    for value, a, b in off[:5]:
        logger.info(f"      {a:<20} ~ {b:<20} {value:+.2f}")
    logger.info("  least substitutable:")
    for value, a, b in off[-5:]:
        logger.info(f"      {a:<20} ~ {b:<20} {value:+.2f}")

    logger.info("sampling pairs:")
    sets = sample_pairs(
        grouped, logger, args.seed, args.matched_pairs, args.min_group,
    )
    if not sets:
        logger.error("no treatment course is shared widely enough to match on")
        return

    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet").select(
        "hospitalization_id", "in_hospital_mortality"
    )
    outcome = (
        grouped.select("hospitalization_id")
        .join(cohort, on="hospitalization_id", how="left")
    )["in_hospital_mortality"].to_list()
    identifiers = grouped["hospitalization_id"].to_list()

    logger.info("aligning:")
    out_rows = []
    for name, pairs in sets.items():
        if not pairs:
            continue
        left = [sequences[a] for a, _ in pairs]
        right = [sequences[b] for _, b in pairs]
        results = align_many(
            left, right, ALPHABET, matrix, args.gap_open, args.gap_extend,
            logger=logger, min_tail=args.min_tail,
        )
        summarise(name, results, logger)
        for (a, b), result in zip(pairs, results):
            out_rows.append({
                "pair_set": name,
                "left": identifiers[a],
                "right": identifiers[b],
                "left_hours": len(sequences[a]),
                "right_hours": len(sequences[b]),
                "diverged": result["diverged"],
                "reason": result["reason"],
                "divergence_hour": result["left_hour"],
                "event": int(result["diverged"]),
                "censoring_hour": min(
                    result["left_hour"] + result["left_remaining"],
                    result["right_hour"] + result["right_remaining"],
                ),
                "identity": result["identity"],
                "left_remaining": result["left_remaining"],
                "right_remaining": result["right_remaining"],
                "score": result["alignment_score"],
                "discordant": (
                    None if outcome[a] is None or outcome[b] is None
                    else int(outcome[a] != outcome[b])
                ),
            })
    table = pl.DataFrame(out_rows, infer_schema_length=None)
    table.write_csv(tables / f"alignment_pairs_{args.subset}.csv")

    # TIME TO DIVERGENCE, TREATED AS A SURVIVAL PROBLEM. The divergence hour is
    # bounded above by the shorter record, so it correlates with length however
    # pairs are chosen; filtering on a required tail discarded 77% of pairs and
    # barely moved the correlation. Censoring is the instrument that fits the
    # problem -- a pair still agreeing when a record ended is informative about
    # the hours it was observed for, and nothing more.
    logger.info("time to divergence, censored at the end of the shorter record:")
    curves = {}
    for name in sets:
        part = table.filter(pl.col("pair_set") == name)
        if part.height < 50:
            continue
        time = part["divergence_hour"].to_numpy().astype(float)
        event = part["event"].to_numpy().astype(int)
        times, curve = kaplan_meier(time, event)
        curves[name] = (time, event)
        still = float(curve[-1]) if len(curve) else 1.0
        logger.info(
            f"  {name}: {event.sum():,} divergences in {part.height:,} pairs "
            f"({event.mean():.1%} uncensored); first decile hour "
            f"{divergence_quantile(times, curve, 0.10):.0f}, first quartile "
            f"{divergence_quantile(times, curve, 0.25):.0f}; "
            f"{still:.1%} still agreeing at the last observed hour"
        )
    if len(curves) == 2:
        (first, second) = list(curves)
        z, pvalue = logrank(*curves[first], *curves[second])
        logger.info(
            f"  log-rank {first} vs {second}: z {z:+.2f}, p {pvalue:.3g} "
            f"(positive z means {second} diverges sooner)"
        )

    # the confound check still runs, on the events only, where it is expected to
    # be positive; it is reported so the censored comparison above is read as
    # the primary one and this as the reason it had to be
    logger.info("residual length dependence among uncensored divergences:")
    for name in sets:
        part = table.filter(
            (pl.col("pair_set") == name) & (pl.col("event") == 1)
        )
        if part.height < 50:
            continue
        shorter = np.minimum(
            part["left_hours"].to_numpy(), part["right_hours"].to_numpy()
        ).astype(float)
        hour = part["divergence_hour"].to_numpy().astype(float)
        correlation = (
            float(np.corrcoef(shorter, hour)[0, 1])
            if shorter.std() > 0 and hour.std() > 0 else float("nan")
        )
        logger.info(
            f"  {name}: corr(divergence hour, shorter course) "
            f"{correlation:+.3f} over {part.height:,} events"
        )

    logger.info("do pairs that diverge sooner end up in different outcomes?")
    for name in sets:
        part = table.filter(
            (pl.col("pair_set") == name) & pl.col("discordant").is_not_null()
        )
        if part.height < 100:
            continue
        concordant = part.filter(pl.col("discordant") == 0)
        discordant = part.filter(pl.col("discordant") == 1)
        if min(concordant.height, discordant.height) < 30:
            continue
        z, pvalue = logrank(
            concordant["divergence_hour"].to_numpy().astype(float),
            concordant["event"].to_numpy().astype(int),
            discordant["divergence_hour"].to_numpy().astype(float),
            discordant["event"].to_numpy().astype(int),
        )
        rates = []
        for label, side in (("same outcome", concordant),
                            ("different outcome", discordant)):
            times, curve = kaplan_meier(
                side["divergence_hour"].to_numpy().astype(float),
                side["event"].to_numpy().astype(int),
            )
            rates.append(
                f"{label}: {side['event'].mean():.1%} diverge, first decile "
                f"hour {divergence_quantile(times, curve, 0.10):.0f}"
            )
        logger.info(f"  {name}: " + "; ".join(rates))
        logger.info(
            f"      log-rank z {z:+.2f}, p {pvalue:.3g} (positive means pairs "
            "with different outcomes diverge sooner)"
        )

    logger.info("39 done")


if __name__ == "__main__":
    main()

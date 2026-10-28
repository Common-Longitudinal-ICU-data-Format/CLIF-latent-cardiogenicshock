#!/usr/bin/env python3

"""
19_wean_durability.py -- when a wean holds, and when the pressors go back on

94% of "0_" words follow another "0_": the patient was already off pressors and
nothing happened. The WEANING words are the ones where a window opens on
pressors and closes without them, and the commonest of those is "30_" --
norepinephrine running steady, then off, inside six hours, with no documented
taper. It outnumbers the tapered "3230_" by two to three times.

This takes every occurrence of a weaning word and asks what decides whether it
is followed by "0_" repeating, or by pressors going back on.

    python code/19_wean_durability.py --subset cardiogenic

A MID-WINDOW WEAN IS NOT A TRIAL OFF PRESSORS, and pooling it with one that
closes the window wrecks the headline. A word ending in "_" closes its 6-hour
window, so nothing more is recorded in it; a word without the marker is
followed by further states IN THE SAME SIX HOURS. Those further states are
pressors 98% of the time -- not because the trial failed but because, had the
patient stayed off, the window would have closed and the word would have ended
in "_". Mid-window weans restart at 98.0% with a median of zero consecutive
"0_" after them; window-closing weans restart at 36.9% with a median of nine.

The mid-window events are better read as transient gaps -- a pump change, a
line swap, the granularity of infusion charting -- than as decisions to trial a
patient off support. The durability analysis therefore runs on WINDOW-CLOSING
weans only, and the mid-window ones are counted and reported separately rather
than dropped silently. An earlier version pooled them and reported a 57%
restart rate, which was the average of a real 37% and a definitional 98%.

A FIXED LOOKAHEAD, NOT "DID IT EVER FAIL". The obvious definition -- no pressor
word anywhere in the rest of the sentence -- is dominated by how much sentence
was left. A wean at hour 150 has four windows in which to fail and a wean at
hour 12 has twenty-five, so "held" would largely mean "weaned late". Every
occurrence is instead scored over the SAME 24-hour horizon, and occurrences
without a full 24 hours of record remaining are dropped rather than counted as
successes.

DEATH IS NOT A WEAN THAT HELD. A patient whose record stops two windows after a
wean has no subsequent pressor word, which the naive definition reads as
success. Those occurrences are excluded by the same runway requirement, and the
exclusion count is reported.

ONE PATIENT CAN CONTRIBUTE SEVERAL OCCURRENCES, and the comparisons below do
not adjust for that. The unit is the wean, not the patient, because the
question is about weans; but a patient weaned four times is four rows, and
patients who are weaned repeatedly are not a random sample of patients. The
per-patient occurrence count is reported so the clustering is visible.

WHAT THIS IS NOT. Weaning is not randomised. A clinician stops pressors when
they judge the patient ready, so any predictor of restart is at least partly a
predictor of what the clinician already knew. The taper comparison in
particular is confounded by indication: a patient tapered slowly may be one the
clinician was worried about.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import types

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.perfusion import PERFUSION_STATES, STATE_SYMBOLS

#: symbols meaning a vasopressor is running, on either side of the MAP cut
ON_PRESSORS = set("123456") | set("89ABCD")
#: the weaning words worth separating: a window that opens on pressors and
#: closes off them, grouped by whether the dose was walked down on the way
#: Matched on the word with its trailing marker STRIPPED. Whether the window
#: happened to end at the wean is a separate question from how the wean was
#: done, and conflating them put "30" -- the same abrupt stop, mid-window --
#: into "other" alongside genuinely different things. "30" is two thirds as
#: common as "30_", so a third of the abrupt norepinephrine weans were being
#: misfiled, and "other" then carried an 85% restart rate that was mostly an
#: artefact: a mid-window wean is followed by more states IN THAT WINDOW, and
#: had the patient stayed off, the window would have closed there instead.
WEAN_FAMILIES = {
    "abrupt, NE alone": ("30",),
    "tapered, NE alone": ("320", "3230", "323230", "32320"),
    "abrupt, combination": ("60",),
    "tapered, combination": ("650", "6560", "656560", "65650"),
}
STEP_HOURS = 6.0


def spell(word: str) -> str:
    pieces = []
    for character in word:
        if character == "_":
            pieces.append("[window ends]")
            continue
        index = STATE_SYMBOLS.find(character)
        pieces.append(
            PERFUSION_STATES[index]
            if 0 <= index < len(PERFUSION_STATES) else character
        )
    return " -> ".join(pieces)


def bears_pressors(word: str) -> bool:
    return any(c in ON_PRESSORS for c in word)


def find_weans(sentences, ids, horizon_windows, logger):
    """
    every occurrence of a weaning word, with what happened over the next 24h

    `position` is the index of the word, which is NOT the window index: a word
    may span several states but never more than one window, so the window a
    word closes is the count of window-closing words up to and including it.
    """
    rows = []
    dropped = 0
    for patient, words in zip(ids, sentences):
        closing = 0
        for index, word in enumerate(words):
            closes = word.endswith("_")
            if closes:
                closing += 1
            if not (bears_pressors(word) and word.rstrip("_").endswith("0")):
                continue
            # the tail, measured in WINDOWS rather than words, so the horizon
            # means the same thing for every occurrence
            tail, seen = [], 0
            for later in words[index + 1:]:
                tail.append(later)
                if later.endswith("_"):
                    seen += 1
                if seen >= horizon_windows:
                    break
            if seen < horizon_windows:
                dropped += 1
                continue
            # WHAT LED INTO THE TRIAL. The weaning word itself is close to
            # uninformative -- "30_" and "60_" both mostly say "came off
            # pressors", which is what every wean says. The clinical situation
            # is in the window BEFORE: going into a trial from "3_" (steady
            # single agent) is not the same as going into it from "646_" (the
            # combination being moved about) or from "D_" (hypotensive on a
            # combination), even though all three can be followed by the same
            # "30_".
            lead_in = words[index - 1] if index else None
            lead_in_2 = words[index - 2] if index > 1 else None
            restarted = any(bears_pressors(w) for w in tail)
            run = 0
            for later in words[index + 1:]:
                if later == "0_":
                    run += 1
                else:
                    break
            # A CLOSING word ends window `closing - 1` (the counter was just
            # incremented for it). A MID-WINDOW word sits in window `closing`,
            # which has not closed yet -- using `closing - 1` for both put
            # every mid-window wean in the PREVIOUS window and gave the first
            # word of a sentence an hour of -6, which the fine grid does not
            # have. The symptom was a minimum of -6h; the cost was a six-hour
            # misalignment against every table joined on this column.
            index_of_window = closing - 1 if closes else closing
            rows.append({
                "hospitalization_id": patient,
                "word": word,
                "window": index_of_window,
                "hours": index_of_window * STEP_HOURS,
                "windows_of_record_after": sum(
                    1 for w in words[index + 1:] if w.endswith("_")
                ),
                "preceding_word": lead_in,
                "preceding_word_2": lead_in_2,
                "lead_in": (
                    f"{lead_in} {word}" if lead_in else f"[start] {word}"
                ),
                "restarted_within_24h": restarted,
                "consecutive_0_after": run,
            })
    logger.info(
        f"  {len(rows):,} weans with a full {horizon_windows * STEP_HOURS:.0f}h "
        f"of record after them; {dropped:,} dropped for running out of record "
        "(death, discharge, or the end of the week)"
    )
    return pl.DataFrame(rows)


def family_of(word):
    core = word.rstrip("_")
    for name, members in WEAN_FAMILIES.items():
        if core in members:
            return name
    # A word that comes off pressors and goes back on inside one window is a
    # restart by construction, not a wean whose durability is in question.
    if any(
        core[i] == "0" and core[i + 1] in ON_PRESSORS
        for i in range(len(core) - 1)
    ):
        return "restarted inside the window"
    return "other"


def per_word_report(weans, fine, labels, physiology, logger, min_weans=60):
    """
    one row per weaning word: who carries it, and whether it holds

    19's dissection was written for a single word. The report needs every word
    that precedes a trial off pressors, with the patient-level outcomes on the
    same row -- otherwise a reader has to carry the restart rate here and the
    mortality from another table and trust that the denominators match.

    TWO DIFFERENT DENOMINATORS SIT ON EACH ROW and the column names say which.
    `weans` and `restarted_within_24h` count OCCURRENCES; `patients` and every
    outcome count PATIENTS who carry the word at least once. A patient weaned
    four times contributes four occurrences and one outcome.
    """
    joined = weans
    if physiology is not None and not physiology.is_empty():
        shared = [
            c for c in ("map", "nee", "n_vasopressors")
            if c in physiology.columns
        ]
        joined = weans.join(
            physiology.select(
                "hospitalization_id",
                pl.col("offset_hours").alias("hours"), *shared,
            ),
            on=["hospitalization_id", "hours"], how="left",
        )

    outcomes = [
        c for c in ("in_hospital_mortality", "icu_los_days", "icu_free_days",
                    "ever_imv", "new_imv_after", "ever_rrt", "new_rrt_after")
        if c in labels.columns
    ]
    patient_level = labels.select(["hospitalization_id"] + outcomes).join(
        fine.select("hospitalization_id", "fine_letters", "fine_bpe_words",
                    "site"),
        on="hospitalization_id", how="inner",
    )

    rows = []
    for word in joined["word"].unique().to_list():
        part = joined.filter(pl.col("word") == word)
        if part.height < min_weans:
            continue
        carriers = part["hospitalization_id"].unique().to_list()
        people = patient_level.filter(
            pl.col("hospitalization_id").is_in(carriers)
        )
        row = {
            "word": word,
            "family": family_of(word),
            "closes_window": word.endswith("_"),
            "weans": part.height,
            "patients": people.height,
            "restarted_within_24h": float(part["restarted_within_24h"].mean()),
            "median_hours_from_onset": float(part["hours"].median()),
            "median_sentence_words": float(people["fine_bpe_words"].median())
            if people.height else None,
            "median_fine_letters": float(people["fine_letters"].median())
            if people.height else None,
        }
        for column in outcomes:
            values = people[column].drop_nulls()
            if values.is_empty():
                continue
            row[column] = float(
                values.cast(pl.Float64).mean()
                if values.dtype in (pl.Boolean,)
                or column.startswith(("ever_", "new_", "in_hospital"))
                else values.median()
            )
        # what separates the occurrences that held, within this word
        held = part.filter(~pl.col("restarted_within_24h"))
        back = part.filter(pl.col("restarted_within_24h"))
        for column in ("map", "nee", "hours"):
            if column not in part.columns:
                continue
            a = held[column].cast(pl.Float64).drop_nulls()
            b = back[column].cast(pl.Float64).drop_nulls()
            # `Series.var()` is None for a single element, so a word with
            # exactly one held or one restarted occurrence crashed the pooled
            # standard deviation rather than skipping the comparison.
            if a.len() < 2 or b.len() < 2:
                continue
            pooled = float(np.sqrt((a.var() + b.var()) / 2)) or 1.0
            row[f"smd_{column}"] = float((b.mean() - a.mean()) / pooled)
        rows.append(row)

    if not rows:
        logger.warning("no word carried enough weans for a per-word row")
        return pl.DataFrame()
    table = pl.DataFrame(rows, infer_schema_length=None).sort(
        "weans", descending=True
    )
    logger.info("per-word summary (occurrences, then patient-level outcomes):")
    for row in table.head(12).iter_rows(named=True):
        logger.info(
            f"    {row['word']:<10} {row['weans']:>6,} weans  "
            f"restart {row['restarted_within_24h']:>6.1%}  "
            f"died {row.get('in_hospital_mortality') or float('nan'):>6.1%}  "
            f"SMD(MAP) {row.get('smd_map') or float('nan'):+.3f}"
        )
    return table


def lead_in_report(weans, labels, logger, min_events=60, top=20):
    """
    the trial characterised by what led INTO it, not by the wean itself

    "30_" and "60_" are close to uninformative on their own: both mostly say
    "came off pressors", which is what every weaning word says. What separates
    one trial from another is the window before -- entering a trial from "3_",
    a steady single agent, is a different clinical situation from entering it
    from "646_", the combination being actively moved, or from "D_",
    hypotensive on a combination.

    Grouped on (preceding word, weaning word) rather than on the preceding word
    alone, because the same lead-in followed by a single-agent stop and by a
    combination stop are not the same event.
    """
    outcomes = [
        c for c in ("in_hospital_mortality", "icu_free_days", "icu_los_days",
                    "new_imv_after", "new_rrt_after")
        if c in labels.columns
    ]
    rows = []
    for key in weans["lead_in"].unique().to_list():
        part = weans.filter(pl.col("lead_in") == key)
        if part.height < min_events:
            continue
        carriers = part["hospitalization_id"].unique().to_list()
        people = labels.filter(pl.col("hospitalization_id").is_in(carriers))
        preceding = part["preceding_word"][0]
        row = {
            "lead_in": key,
            "preceding_word": preceding,
            "weaning_word": part["word"][0],
            "events": part.height,
            "patients": people.height,
            "restarted_within_24h": float(part["restarted_within_24h"].mean()),
            "median_hours_from_onset": float(part["hours"].median()),
            "preceding_meaning": spell(preceding) if preceding else "[start]",
        }
        for column in outcomes:
            values = people[column].drop_nulls()
            if values.is_empty():
                continue
            share = column.startswith(("new_", "in_hospital"))
            row[column] = float(
                values.cast(pl.Float64).mean() if share else values.median()
            )
        rows.append(row)
    if not rows:
        logger.warning("no lead-in cleared the floor")
        return pl.DataFrame()
    table = pl.DataFrame(rows, infer_schema_length=None).sort(
        "restarted_within_24h"
    )
    logger.info(
        "trials by what led into them (most durable first), "
        f"{table.height} combinations above {min_events} events:"
    )
    for row in table.head(top).iter_rows(named=True):
        logger.info(
            f"    {row['lead_in']:<20} {row['events']:>6,} events  "
            f"restart {row['restarted_within_24h']:>6.1%}  "
            f"died {row.get('in_hospital_mortality') or float('nan'):>6.1%}  "
            f"| from: {str(row['preceding_meaning'])[:48]}"
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic"))
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--word", default="30_",
                        help="the word whose occurrences are dissected")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "19_wean_durability", paths.log_home / "19_wean_durability.log"
    )
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)
    horizon = int(args.horizon_hours // STEP_HOURS)

    fine = pl.read_parquet(
        paths.patient_level_home / f"fine_state_sequences_{args.subset}.parquet"
    ).filter(pl.col("fine_bpe_sentence").is_not_null())
    ids = fine["hospitalization_id"].to_list()
    sentences = [s.split() for s in fine["fine_bpe_sentence"].to_list()]
    logger.info(f"{args.subset}: {len(ids):,} patients")

    # ---- 1. what precedes coming off pressors
    from collections import Counter
    before = Counter()
    for words in sentences:
        for index, word in enumerate(words):
            if word == "0_" and index:
                before[words[index - 1]] += 1
    total = sum(before.values())
    logger.info(f"what precedes a '0_' ({total:,} occurrences):")
    rows = []
    for word, n in before.most_common(15):
        rows.append({
            "preceding_word": word,
            "occurrences": n,
            "share": n / max(total, 1),
            "already_off_pressors": not bears_pressors(word),
            "meaning": spell(word),
        })
        logger.info(
            f"    {word:<10} {n:>8,} {n / max(total, 1):>6.1%}  "
            f"{'(already off)' if not bears_pressors(word) else 'A WEAN':<14} "
            f"{spell(word)[:58]}"
        )
    pl.DataFrame(rows).write_csv(
        tables / f"wean_preceding_words_{args.subset}.csv"
    )

    weans = find_weans(sentences, ids, horizon, logger)
    if weans.is_empty():
        logger.error("no weans with enough runway")
        return
    weans = weans.with_columns(
        family=pl.col("word").map_elements(family_of, return_dtype=pl.String),
        closes_window=pl.col("word").str.ends_with("_"),
    )
    per_patient = weans.group_by("hospitalization_id").len()
    logger.info(
        f"  {per_patient.height:,} patients contribute them, "
        f"median {per_patient['len'].median():.0f} each, "
        f"max {per_patient['len'].max()}"
    )

    # ---- 2. abrupt versus tapered
    # Split before anything is summarised: the two kinds of event answer
    # different questions and their restart rates differ by sixty points.
    transient = weans.filter(~pl.col("closes_window"))
    weans = weans.filter(pl.col("closes_window"))
    logger.info(
        f"  {transient.height:,} mid-window events set aside as transient gaps "
        f"(restart {transient['restarted_within_24h'].mean():.1%}, median "
        f"{transient['consecutive_0_after'].median():.0f} consecutive '0_' "
        f"after); {weans.height:,} window-closing trials kept"
    )
    pl.DataFrame([{
        "kind": "window-closing trial off pressors",
        "events": weans.height,
        "restarted_within_24h": float(weans["restarted_within_24h"].mean()),
        "median_consecutive_0_after": float(
            weans["consecutive_0_after"].median()
        ),
    }, {
        "kind": "mid-window gap (not a trial)",
        "events": transient.height,
        "restarted_within_24h": float(
            transient["restarted_within_24h"].mean()
        ),
        "median_consecutive_0_after": float(
            transient["consecutive_0_after"].median()
        ),
    }]).write_csv(tables / f"wean_kinds_{args.subset}.csv")

    logger.info("restart within 24h, by how the wean was done:")
    family_rows = []
    for family in list(WEAN_FAMILIES) + [
        "restarted inside the window", "other"
    ]:
        part = weans.filter(pl.col("family") == family)
        if part.height < 20:
            continue
        family_rows.append({
            "family": family,
            "weans": part.height,
            "closed_the_window": float(part["closes_window"].mean()),
            "patients": part["hospitalization_id"].n_unique(),
            "restarted_within_24h": float(part["restarted_within_24h"].mean()),
            "median_hours_from_onset": float(part["hours"].median()),
            "median_consecutive_0_after": float(
                part["consecutive_0_after"].median()
            ),
        })
        logger.info(
            f"    {family:<24} {part.height:>6,} weans  "
            f"restart {part['restarted_within_24h'].mean():>6.1%}  "
            f"at {part['hours'].median():>5.0f}h"
        )
    pl.DataFrame(family_rows).write_csv(
        tables / f"wean_by_family_{args.subset}.csv"
    )

    # ---- 3. within ONE word, what separates the weans that held
    # Both the window-closing and the mid-window spelling of the same wean:
    # they are one clinical event and splitting them halves the sample for no
    # reason the question cares about.
    target = weans.filter(
        pl.col("word").str.strip_chars_end("_") == args.word.rstrip("_")
    )
    physiology = None
    if target.height < 100:
        logger.warning(f"only {target.height} occurrences of {args.word!r}")
        return
    logger.info(
        f"dissecting {target.height:,} occurrences of {args.word!r} "
        f"({spell(args.word)}):"
    )

    # physiology at the wean window, recomputed: 11 builds it but writes only
    # the states, and the question needs the numbers underneath them
    spec = importlib.util.spec_from_file_location(
        "longitudinal",
        pathlib.Path(__file__).resolve().parent / "11_longitudinal_clusters.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cohort = module.cohort_ids(paths, args.subset, logger)
    physiology = module.perfusion_sequences(paths, cohort, module.OFFSETS, logger)
    if not physiology.is_empty():
        target = target.join(
            physiology.select(
                "hospitalization_id",
                pl.col("offset_hours").alias("hours"),
                "map", "nee", "n_vasopressors",
                *[c for c in ("on_imv",) if c in physiology.columns],
            ),
            on=["hospitalization_id", "hours"], how="left",
        )

    states = paths.patient_level_home / f"longitudinal_states_{args.subset}.parquet"
    if states.is_file():
        long = pl.read_parquet(states)
        topics = [c for c in long.columns if c.startswith("topic_")]
        target = target.join(
            long.select(
                "hospitalization_id",
                pl.col("offset_hours").alias("hours"),
                *topics,
                *[c for c in ("growth_cluster",) if c in long.columns],
            ),
            on=["hospitalization_id", "hours"], how="left",
        )
    target = target.join(
        fine.select("hospitalization_id", "site"),
        on="hospitalization_id", how="left",
    )

    held = target.filter(~pl.col("restarted_within_24h"))
    back = target.filter(pl.col("restarted_within_24h"))
    logger.info(
        f"  {held.height:,} held ({held.height / target.height:.1%}), "
        f"{back.height:,} back on pressors within 24h"
    )

    comparison = []
    for column in ("hours", "map", "nee", "n_vasopressors",
                   "windows_of_record_after"):
        if column not in target.columns:
            continue
        a = held[column].cast(pl.Float64).drop_nulls()
        b = back[column].cast(pl.Float64).drop_nulls()
        if a.is_empty() or b.is_empty():
            continue
        pooled = float(np.sqrt((a.var() + b.var()) / 2)) or 1.0
        comparison.append({
            "measure": column,
            "held_median": float(a.median()),
            "restarted_median": float(b.median()),
            "standardised_difference": float((b.mean() - a.mean()) / pooled),
        })
        logger.info(
            f"    {column:<26} held {a.median():>8.3g}  "
            f"restarted {b.median():>8.3g}  "
            f"SMD {(b.mean() - a.mean()) / pooled:+.3f}"
        )
    if comparison:
        pl.DataFrame(comparison).write_csv(
            tables / f"wean_held_vs_restarted_{args.subset}.csv"
        )

    for column in ("site", "growth_cluster"):
        if column not in target.columns:
            continue
        logger.info(f"    restart rate by {column}:")
        grouped = (
            target.filter(pl.col(column).is_not_null())
            .group_by(column)
            .agg(
                weans=pl.len(),
                restarted=pl.col("restarted_within_24h").cast(pl.Float64).mean(),
            )
            .sort(column)
        )
        for row in grouped.iter_rows(named=True):
            logger.info(
                f"      {str(row[column]):<8} {row['weans']:>6,} weans  "
                f"{row['restarted']:>6.1%} restart"
            )
        grouped.write_csv(
            tables / f"wean_restart_by_{column}_{args.subset}.csv"
        )
    target.write_parquet(
        paths.patient_level_home / f"wean_occurrences_{args.subset}.parquet"
    )

    # Every weaning word on one table, for the standalone BPE report: the
    # restart rate beside the patient-level outcomes, so the denominators are
    # visible rather than inferred from two tables that may not match.
    labels = pl.read_parquet(
        paths.patient_level_home
        / f"longitudinal_clusters_{args.subset}.parquet"
    )
    summary = per_word_report(weans, fine, labels, physiology, logger)
    if not summary.is_empty():
        summary.write_csv(tables / f"wean_word_outcomes_{args.subset}.csv")
    leads = lead_in_report(weans, labels, logger)
    if not leads.is_empty():
        leads.write_csv(tables / f"wean_lead_in_{args.subset}.csv")
    weans.write_parquet(
        paths.patient_level_home / f"wean_all_{args.subset}.parquet"
    )
    logger.info("19 done")


if __name__ == "__main__":
    main()

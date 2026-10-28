#!/usr/bin/env python3

"""
17_fine_grained_states.py -- MAP and pressor states inside the 6-hour window

The coarse analysis gives a patient one letter per 6-hour window, so a week is
29 letters and everything that happened between grid points is gone. A patient
titrated up three times and then weaned between hour 12 and hour 18 carries the
same letter as one left untouched, because both are summarised by whatever was
running at hour 18.

This goes back to the raw CLIF tables and builds the finer sequence: within
each window, as many letters as the patient had distinct states, closed by an
end-of-window marker.

    python code/17_fine_grained_states.py --subset cardiogenic

    3_3_3_3A3_34646_6_6_6_
    |  |  |  |  | |
    window 1 is one state; window 5 holds three, a rise then two falls

THE MARKER EARNS ITS PLACE HERE AND NOT IN THE COARSE VERSION. On the 6-hour
sequences one letter IS one window, so a separator after every letter would
make every letter its own word and block every merge -- BPE would learn
nothing. Here there are several letters per window, so "_" is a real boundary
with something on either side, and a BPE word becomes a within-window
manoeuvre rather than an arbitrary span.

THE LENGTH OF THESE SEQUENCES IS THE THING TO DISTRUST. Two guards are built
in and a third is reported rather than fixed:

  * MAP is resampled hourly rather than read per measurement, because
    monitoring intensity is not a clinical decision -- an arterial line gives a
    MAP a minute, a cuff gives a handful a day.
  * Runs are collapsed within each window, so a window's length counts STATE
    CHANGES, not samples.
  * Dose changes ARE read as events, because a new infusion rate is a recorded
    action rather than an observation. That is the honest representation and
    it is also the remaining confound: a site that re-charts a running
    infusion hourly will show more "events" than one that charts only genuine
    changes. The per-site length table below is the check, and it is the first
    thing to read -- if the sites disagree on letters per patient, the
    vocabulary is partly a vocabulary of charting habits.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.bpe import apply_merges, learn_merges, vocabulary_table
from clifcs.fine_states import (
    RESOLUTION_MINUTES, TOLERANCE_HOURS, WINDOW_END, fine_state_sequences,
)
from clifcs.perfusion import PERFUSION_STATES, STATE_SYMBOLS

STEP_HOURS = 6.0
END_HOURS = 168.0


def site_lengths(frame, logger):
    """
    letters per patient by site -- the validity check, read before anything else

    If one hospital charts twice as many states per patient as another, the
    vocabulary BPE learns is partly a vocabulary of charting habits, and every
    comparison downstream inherits that. Reported as a ratio against the
    pooled median so the number can be read without the three medians in hand.
    """
    pooled = float(frame["fine_letters"].median())
    rows = []
    for site in sorted(frame["site"].unique().to_list()):
        part = frame.filter(pl.col("site") == site)
        median = float(part["fine_letters"].median())
        rows.append({
            "site": site,
            "patients": part.height,
            "median_letters": median,
            "ratio_to_pooled": median / max(pooled, 1e-9),
            "median_windows_with_a_change": float(
                part["windows_with_a_change"].median()
            ),
            "median_max_letters_in_a_window": float(
                part["max_letters_in_a_window"].median()
            ),
        })
        logger.info(
            f"    {site:<6} {part.height:>6,} patients  median "
            f"{median:>5.0f} letters ({median / max(pooled, 1e-9):>4.2f}x "
            f"pooled)  {part['windows_with_a_change'].median():>4.0f} windows "
            "with a change"
        )
    spread = max(r["ratio_to_pooled"] for r in rows) / min(
        r["ratio_to_pooled"] for r in rows
    )
    logger.info(
        f"  widest site-to-site ratio {spread:.2f}x"
        + ("  -- read the vocabulary with that in mind" if spread > 1.5 else "")
    )
    return pl.DataFrame(rows).with_columns(widest_ratio=pl.lit(spread))


def versus_coarse(fine, coarse, logger, step_hours=STEP_HOURS):
    """
    what the 6-hour summary could not see

    The coarse letter for a window is the state at the window's END. So for
    every window holding more than one fine state, the coarse version kept the
    last and dropped the rest. Two counts follow, and they say different
    things: how often a window held any change at all, and how often the
    change was one the coarse letter actively misrepresents -- the window
    STARTED in a different state than the one it was labelled with.
    """
    coarse_map = dict(zip(
        coarse["hospitalization_id"].to_list(),
        coarse["sequence_raw"].to_list(),
    ))
    windows = total = changed = misread = 0
    first_vs_last = Counter()
    for patient, sentence in zip(
        fine["hospitalization_id"].to_list(), fine["fine_sentence"].to_list()
    ):
        reference = coarse_map.get(patient)
        if reference is None:
            continue
        total += 1
        for index, chunk in enumerate(sentence.split(WINDOW_END)[:-1]):
            if not chunk or index >= len(reference):
                continue
            windows += 1
            if len(chunk) > 1:
                changed += 1
                if chunk[0] != chunk[-1]:
                    misread += 1
                    first_vs_last[(chunk[0], chunk[-1])] += 1
    logger.info(
        f"  {total:,} patients matched against their coarse sequence over "
        f"{windows:,} windows"
    )
    logger.info(
        f"    {changed:,} windows ({changed / max(windows, 1):.1%}) held more "
        "than one state"
    )
    logger.info(
        f"    {misread:,} windows ({misread / max(windows, 1):.1%}) began in a "
        "different state than the one the coarse letter records"
    )

    def name(symbol):
        position = STATE_SYMBOLS.find(symbol)
        return (
            PERFUSION_STATES[position]
            if 0 <= position < len(PERFUSION_STATES) else symbol
        )

    rows = [
        {
            "began_as": name(a), "recorded_as": name(b),
            "windows": n, "share_of_misread": n / max(misread, 1),
        }
        for (a, b), n in first_vs_last.most_common(20)
    ]
    logger.info("  the transitions the coarse letter hides, commonest first:")
    for row in rows[:8]:
        logger.info(
            f"      {row['began_as']:<34} -> {row['recorded_as']:<34} "
            f"{row['windows']:>6,}"
        )
    summary = pl.DataFrame([{
        "patients": total, "windows": windows,
        "windows_with_a_change": changed,
        "share_with_a_change": changed / max(windows, 1),
        "windows_misread_by_coarse": misread,
        "share_misread_by_coarse": misread / max(windows, 1),
    }])
    return summary, pl.DataFrame(rows) if rows else pl.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--merges", type=int, default=150,
                        help="more than the coarse 100: the alphabet is the "
                             "same but the sequences are several times longer")
    parser.add_argument("--resolution-minutes", type=int,
                        default=RESOLUTION_MINUTES)
    parser.add_argument("--tolerance-hours", type=float,
                        default=TOLERANCE_HOURS)
    parser.add_argument("--no-collapse", action="store_true",
                        help="keep every event as a letter; makes sequence "
                             "length a measure of charting frequency")
    parser.add_argument("--end-hours", type=float, default=END_HOURS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "17_fine_grained_states", paths.log_home / "17_fine_grained_states.log"
    )
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    # The same definition 11 uses, imported rather than restated: a second
    # copy of the cardiogenic filter is a second thing to keep in step.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "longitudinal",
        pathlib.Path(__file__).resolve().parent / "11_longitudinal_clusters.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cohort = module.cohort_ids(paths, args.subset, logger)
    onset = pl.read_parquet(paths.cohort_home / "cs_onset.parquet").select(
        "hospitalization_id",
        onset=pl.col("cs_onset_dttm").cast(pl.Datetime("us")),
    )
    logger.info(
        f"{args.subset}: {cohort.height:,} patients, "
        f"{args.resolution_minutes}-minute MAP checkpoints, "
        f"{args.tolerance_hours:.0f}h staleness tolerance, "
        f"runs {'kept' if args.no_collapse else 'collapsed'} within a window"
    )

    parts = []
    for site in paths.sites:
        ids = cohort.filter(
            pl.col("site") == site
        )["hospitalization_id"].to_list()
        if not ids:
            continue
        anchors = onset.filter(pl.col("hospitalization_id").is_in(ids))
        logger.info(f"  [{site}] {anchors.height:,} anchors")
        built = fine_state_sequences(
            paths.raw_subset(site), anchors, logger,
            step_hours=STEP_HOURS, end_hours=args.end_hours,
            resolution_minutes=args.resolution_minutes,
            tolerance_hours=args.tolerance_hours,
            collapse=not args.no_collapse,
        )
        if not built.is_empty():
            parts.append(built.with_columns(site=pl.lit(site)))
    if not parts:
        logger.error("no sequences built")
        return
    frame = pl.concat(parts, how="vertical")
    logger.info(
        f"{frame.height:,} patients, median {frame['fine_letters'].median():.0f} "
        f"letters over {int(args.end_hours // STEP_HOURS)} windows "
        f"(coarse would give {int(args.end_hours // STEP_HOURS)})"
    )

    logger.info("letters per patient by site:")
    site_lengths(frame, logger).write_csv(
        tables / f"fine_state_site_lengths_{args.subset}.csv"
    )

    # ---- BPE over the fine sequences, with "_" as a hard boundary
    sequences = frame["fine_sentence"].to_list()
    logger.info(f"learning {args.merges} BPE merges with '_' as a boundary:")
    # The marker is a SUFFIX, not a separator: merges absorb it rightward
    # so a window-closing word reads "320_", and no word spans two windows.
    merges = learn_merges(
        sequences, n_merges=args.merges, suffix=WINDOW_END, logger=logger
    )
    crossing = [m for m in merges if WINDOW_END in m[0]]
    if crossing:
        logger.error(f"{len(crossing)} merges crossed the window boundary")
    vocabulary, segmented, compression = vocabulary_table(
        sequences, merges, PERFUSION_STATES, logger
    )
    vocabulary.write_csv(tables / f"fine_bpe_vocabulary_{args.subset}.csv")
    pl.DataFrame({
        "step": list(range(1, len(merges) + 1)),
        "left": [a for a, _ in merges],
        "right": [b for _, b in merges],
        "merged": [a + b for a, b in merges],
    }).write_csv(tables / f"fine_bpe_merges_{args.subset}.csv")

    # How much of the vocabulary is genuinely within-window structure rather
    # than the runs the coarse version would also have found. A word of length
    # one is a single state; the interesting ones are longer AND sit inside a
    # window rather than spanning the whole of it.
    multi = vocabulary.filter(
        (pl.col("length") > 1) & (pl.col("word") != WINDOW_END)
    )
    logger.info(
        f"  {multi.height:,} multi-state words, "
        f"{compression:.2f} letters per word"
    )
    # Every word here is movement, and that is structural rather than a
    # finding: runs are collapsed within a window and merges cannot cross a
    # window boundary, so no word can contain the same state twice in a row.
    # The coarse vocabulary's commonest entries were "00", "000" and "0000" --
    # three words all meaning "stayed off pressors". None of those can exist
    # in this vocabulary, which is most of why it is worth having.
    repeated = [w for w in multi["word"].to_list() if len(set(w)) == 1]
    if repeated:
        logger.error(
            f"  {len(repeated)} words repeat a single state, which the "
            "collapsing should have made impossible"
        )
    logger.info(
        f"  all {multi.height:,} are within-window movement by construction: "
        "collapsing plus the window boundary leave no way to write a run"
    )

    # ---- what 6-hour resolution missed
    labels_path = (
        paths.patient_level_home
        / f"longitudinal_clusters_{args.subset}.parquet"
    )
    if labels_path.is_file():
        labels = pl.read_parquet(labels_path)
        column = (
            "state_sequence" if "state_sequence" in labels.columns else None
        )
        if column:
            coarse = labels.select(
                "hospitalization_id", sequence_raw=pl.col(column)
            )
            logger.info("against the 6-hour sequences:")
            summary, hidden = versus_coarse(frame, coarse, logger)
            summary.write_csv(
                tables / f"fine_versus_coarse_{args.subset}.csv"
            )
            if not hidden.is_empty():
                hidden.write_csv(
                    tables / f"fine_hidden_transitions_{args.subset}.csv"
                )
        else:
            logger.warning("no state_sequence column; comparison skipped")
    else:
        logger.warning(f"no {labels_path.name}; run 11 first for the comparison")

    out = (
        paths.patient_level_home / f"fine_state_sequences_{args.subset}.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.with_columns(
        fine_bpe_sentence=pl.Series([" ".join(t) for t in segmented]),
        fine_bpe_words=pl.Series([len(t) for t in segmented]),
    ).write_parquet(out)
    logger.info(f"wrote {out} ({frame.height:,} patients)")
    logger.info("17 done")


if __name__ == "__main__":
    main()

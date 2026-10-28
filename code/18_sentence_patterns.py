#!/usr/bin/env python3

"""
18_sentence_patterns.py -- what order the BPE words come in

The vocabulary says which units exist and the neighbour columns say what sits
either side of each one. This asks the next question: which ORDERED patterns
recur across patients, and which of them separate the latent classes.

    python code/18_sentence_patterns.py --subset cardiogenic --sequences fine

WHY NOT "THE COMMONEST SENTENCE". Between 93% and 98% of these sentences occur
exactly once, and the only one that is common is the degenerate "never on
pressors". The empirical distribution over whole sentences is therefore almost
flat and has nothing to report. Sub-sentences do: a pattern that recurs across
patients whose weeks otherwise differ is the answerable version of the
question.

TWO VIEWS, KEPT SEPARATE. Word n-grams are contiguous -- strictly "what comes
next". Sequential patterns allow gaps, so a wean followed three days later by
coming off pressors counts as the same story whatever sat between. A trigram
cannot express that unless the gap is exactly zero, which is why both are here
and why the tables carry a `contiguous` column rather than being pooled.

LATENT CLASSES ONLY. Patterns are cross-tabulated against the latent class
growth clusters and nothing else. The other four clusterings in this project
are not used here, deliberately -- this is not a comparison of clustering
methods.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.patterns import (
    pattern_by_cluster, prefix_span, word_ngrams,
)

#: which sentence to read. The fine-grained ones carry a median of ~58 words
#: against the coarse ones' 4, so a gapped pattern has somewhere to put a gap;
#: on the coarse sentences a trigram nearly spans the whole week.
SOURCES = {
    "fine": ("fine_state_sequences", "fine_bpe_sentence"),
    "coarse": ("longitudinal_clusters", "bpe_sentence"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic"))
    parser.add_argument("--sequences", default="fine",
                        choices=tuple(SOURCES) + ("both",))
    parser.add_argument("--min-pattern-share", type=float, default=0.05,
                        help="per-patient support floor for a gapped pattern")
    parser.add_argument("--min-ngram-share", type=float, default=0.01)
    parser.add_argument("--max-pattern-words", type=int, default=5)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "18_sentence_patterns", paths.log_home / "18_sentence_patterns.log"
    )
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    labels = pl.read_parquet(
        paths.patient_level_home
        / f"longitudinal_clusters_{args.subset}.parquet"
    )
    if "growth_cluster" not in labels.columns:
        logger.error("no growth_cluster; run 11 first")
        return
    assignment = dict(zip(
        labels["hospitalization_id"].to_list(),
        labels["growth_cluster"].to_list(),
    ))

    wanted = tuple(SOURCES) if args.sequences == "both" else (args.sequences,)
    for source in wanted:
        stem, column = SOURCES[source]
        path = paths.patient_level_home / f"{stem}_{args.subset}.parquet"
        if not path.is_file():
            logger.warning(f"no {path.name}; {source} skipped")
            continue
        frame = pl.read_parquet(path)
        if column not in frame.columns:
            logger.warning(f"{path.name} carries no {column}; skipped")
            continue
        frame = frame.filter(
            pl.col(column).is_not_null() & (pl.col(column) != "")
        )
        ids = frame["hospitalization_id"].to_list()
        sentences = [s.split() for s in frame[column].to_list()]
        lengths = sorted(len(s) for s in sentences)
        logger.info(
            f"=== {source} sentences: {len(sentences):,} patients, median "
            f"{lengths[len(lengths) // 2]} words, "
            f"{len({w for s in sentences for w in s}):,} distinct words"
        )

        # how flat the sentence-level distribution is, stated rather than
        # assumed: this is the number that makes sub-sentences the only
        # answerable version of "what is common"
        whole = frame[column].value_counts(sort=True)
        logger.info(
            f"  whole sentences: {whole.height:,} distinct, commonest covers "
            f"{whole['count'][0] / len(sentences):.1%} of patients, "
            f"{(whole['count'] == 1).sum() / whole.height:.0%} occur once"
        )

        grams = word_ngrams(
            sentences, sizes=(2, 3), min_patients=args.min_ngram_share
        )
        if not grams.is_empty():
            logger.info(f"  {grams.height:,} contiguous n-grams above the floor")
            for row in grams.head(10).iter_rows(named=True):
                logger.info(
                    f"    {row['pattern']:<30} "
                    f"{row['share_of_patients']:>6.1%} of patients"
                )
            grams.write_csv(
                tables / f"sentence_ngrams_{source}_{args.subset}.csv"
            )

        logger.info(
            f"  mining gapped patterns at a {args.min_pattern_share:.0%} "
            "per-patient support floor:"
        )
        gapped = prefix_span(
            sentences, min_patients=args.min_pattern_share,
            max_length=args.max_pattern_words,
        )
        if gapped.is_empty():
            logger.warning("  no gapped pattern cleared the floor")
            continue
        runs = gapped.filter(pl.col("is_a_run"))
        moves = gapped.filter(~pl.col("is_a_run"))
        logger.info(
            f"  {gapped.height:,} closed gapped patterns: {runs.height:,} are "
            f"a single word repeated, {moves.height:,} involve more than one"
        )
        # The runs are reported but not led with. "0_ 0_ 0_" is three
        # consecutive windows off pressors -- true, and carried by three
        # quarters of the cohort, and not a sequence of care. Leading with it
        # buries everything that is.
        logger.info("  the commonest runs (one word repeated):")
        for row in runs.head(4).iter_rows(named=True):
            logger.info(
                f"    {row['pattern']:<34} "
                f"{row['share_of_patients']:>6.1%} of patients"
            )
        logger.info("  the commonest patterns involving a CHANGE of word:")
        for row in moves.head(14).iter_rows(named=True):
            logger.info(
                f"    {row['pattern']:<34} "
                f"{row['share_of_patients']:>6.1%} of patients "
                f"({row['distinct_words']} distinct words)"
            )
        gapped.write_csv(
            tables / f"sentence_patterns_{source}_{args.subset}.csv"
        )

        # ---- which patterns separate the latent classes
        logger.info("  patterns against the latent classes:")
        # Patterns involving a change come first, then runs, so the class
        # comparison is not dominated by "this class stayed on the same
        # regimen for longer" -- which is true of the sicker classes by
        # definition and crowds out everything else.
        candidates = (
            moves.head(200)["pattern"].to_list()
            + runs.head(50)["pattern"].to_list()
        )
        enrichment = pattern_by_cluster(
            candidates, sentences, ids, assignment, logger
        )
        if not enrichment.is_empty():
            enrichment.write_csv(
                tables / f"sentence_patterns_by_class_{source}_{args.subset}.csv"
            )
    logger.info("18 done")


if __name__ == "__main__":
    main()

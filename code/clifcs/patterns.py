#!/usr/bin/env python3

"""
ordered patterns over a BPE sentence: n-grams and frequent subsequences

Two ways of asking what order the words come in, kept separate because they
answer different questions.

WORD n-GRAMS are CONTIGUOUS. A bigram counts only where one word directly
follows another, which is the "what comes next" question and is what the
vocabulary table's neighbour columns show, lifted into a feature.

SEQUENTIAL PATTERNS ALLOW GAPS. "320_ then later 0_" counts however many words
sit between, which is the "what order does care unfold in" question. Contiguous
n-grams cannot express it: a wean followed three days later by coming off
pressors is the same clinical story whether or not something happened in
between, and a trigram would miss it unless the gap were exactly zero.

WHY GAPS MATTER MORE THAN USUAL HERE. 93-98% of these sentences occur exactly
once, so the empirical distribution over whole sentences is almost flat -- the
only common one is the degenerate "never on pressors". Frequency at the
sentence level has no answer. Frequency at the sub-sentence level does, and
allowing gaps is what makes a sub-sentence recur across patients whose weeks
otherwise differ.

SUPPORT IS COUNTED PER PATIENT, NOT PER OCCURRENCE. A pattern appearing eight
times in one patient's week is one patient's worth of evidence. Counting
occurrences would rank the vocabulary by whoever was in hospital longest.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl


def word_ngrams(sentences, sizes=(2, 3), min_patients=0.01):
    """
    contiguous runs of n words, with the patient count that carries each

    `min_patients` is a FRACTION of patients, not a count, so the floor scales
    with the cohort rather than admitting rarer patterns in bigger ones.
    """
    floor = max(int(min_patients * len(sentences)), 2)
    carriers, occurrences = Counter(), Counter()
    for words in sentences:
        seen = set()
        for size in sizes:
            for start in range(len(words) - size + 1):
                gram = tuple(words[start:start + size])
                occurrences[gram] += 1
                seen.add(gram)
        carriers.update(seen)
    rows = [
        {
            "pattern": " ".join(gram),
            "words": len(gram),
            "patients": n,
            "share_of_patients": n / max(len(sentences), 1),
            "occurrences": occurrences[gram],
            "contiguous": True,
        }
        for gram, n in carriers.items() if n >= floor
    ]
    return pl.DataFrame(rows).sort("patients", descending=True)


def _project(sentences, prefix):
    """
    for each sentence containing `prefix` in order, what remains after it

    The projected database of PrefixSpan: having matched the prefix, only the
    tail can extend it, so each recursion works on a strictly smaller problem
    than the one above it.
    """
    projected = []
    for words in sentences:
        position = 0
        for item in prefix:
            while position < len(words) and words[position] != item:
                position += 1
            if position == len(words):
                break
            position += 1
        else:
            projected.append(words[position:])
    return projected


def prefix_span(sentences, min_patients=0.05, max_length=5,
                max_patterns=4000, closed=True):
    """
    frequent ordered subsequences, gaps allowed, by per-patient support

    Depth-first over the projected databases. The support of a prefix can only
    fall as the prefix grows, so a prefix below the floor is pruned with
    everything beneath it -- which is what makes the search tractable at a
    vocabulary of this size.

    `max_patterns` is a guard rather than a parameter of the method: a low
    floor on a long sentence can enumerate combinatorially, and a run that
    silently takes an hour is worse than one that stops and says so.

    ONLY CLOSED PATTERNS ARE RETURNED by default, and without this the output
    is unreadable. A pattern is closed when no longer pattern containing it
    has the SAME support. The raw enumeration returns "6_ 6_ 6_", then
    "6_ 6_ 6_ 6_", then "6_ 6_ 6_ 6_ 6_" as three findings carried by very
    nearly the same patients -- so a cross-tabulation against four classes
    reported 659 of 1,193 tests as significant, which is a statement about
    redundancy in the pattern set rather than about the data. Keeping the
    longest representative of each support class leaves one row per distinct
    group of patients.

    `distinct_words` is carried because the commonest patterns are runs of a
    single repeated word -- "0_ 0_ 0_" is three consecutive windows off
    pressors. Those are real but they are not sequences of care in any
    interesting sense, and this is the column that separates them.
    """
    floor = max(int(min_patients * len(sentences)), 2)
    found = {}

    def grow(prefix, projected):
        if len(found) >= max_patterns or len(prefix) >= max_length:
            return
        nxt = Counter()
        for tail in projected:
            nxt.update(set(tail))
        for word, support in nxt.most_common():
            if support < floor:
                break
            extended = prefix + (word,)
            found[extended] = support
            grow(extended, _project(projected, (word,)))

    grow((), [list(s) for s in sentences])
    keep = {p: s for p, s in found.items() if len(p) > 1}
    if closed:
        by_support = {}
        for pattern, support in keep.items():
            by_support.setdefault(support, []).append(pattern)
        closed_set = set()
        for support, group in by_support.items():
            # longest first, so a pattern is dropped only against a strictly
            # longer one already kept at the same support
            for pattern in sorted(group, key=len, reverse=True):
                if not any(
                    len(other) > len(pattern)
                    and _contains_in_order(other, pattern)
                    for other in closed_set
                ):
                    closed_set.add(pattern)
        keep = {p: s for p, s in keep.items() if p in closed_set}
    rows = [
        {
            "pattern": " ".join(pattern),
            "words": len(pattern),
            "distinct_words": len(set(pattern)),
            "is_a_run": len(set(pattern)) == 1,
            "patients": support,
            "share_of_patients": support / max(len(sentences), 1),
            "contiguous": False,
        }
        for pattern, support in keep.items()
    ]
    return (
        pl.DataFrame(rows).sort("patients", descending=True)
        if rows else pl.DataFrame()
    )


def pattern_by_cluster(patterns, sentences, ids, assignment, logger,
                       min_cell=11, top=25):
    """
    which patterns distinguish one latent class from the rest

    Each pattern is tested as a per-patient presence flag, one class against
    all others pooled, so the comparison is "does this class do this more than
    everyone else" rather than a free-for-all between four groups.

    SMALL CELLS ARE SUPPRESSED AT 11, the CMS floor this project uses
    throughout, because the per-class tables are thin enough that a cell of
    three would otherwise be publishable.

    Bonferroni over every (pattern, class) test. It is the conservative
    choice and that is deliberate: the pattern set was itself selected for
    frequency, so the tests are neither independent nor a fair sample of
    hypotheses, and a sharper correction would promise a calibration this
    design cannot deliver.
    """
    from scipy.stats import chi2_contingency

    index = {patient: i for i, patient in enumerate(ids)}
    classes = sorted(set(assignment.values()))
    members = {c: [] for c in classes}
    for patient, label in assignment.items():
        if patient in index and label in members:
            members[label].append(index[patient])

    rows = []
    for pattern in patterns:
        words = tuple(pattern.split())
        # presence is tested on the ORDERED pattern, not on the word set: a
        # patient carrying every word in the wrong order does not carry the
        # pattern, and testing membership of the bag would quietly turn an
        # order-aware analysis back into a bag-of-words one
        carries = [
            _contains_in_order(sentence, words) for sentence in sentences
        ]
        total = sum(carries)
        if total < min_cell:
            continue
        for label in classes:
            rows_in = members[label]
            if len(rows_in) < min_cell:
                continue
            inside = sum(carries[i] for i in rows_in)
            outside = total - inside
            n_in, n_out = len(rows_in), len(sentences) - len(rows_in)
            if min(inside, outside) < min_cell:
                continue
            table = [[inside, n_in - inside], [outside, n_out - outside]]
            chi2, p, _, _ = chi2_contingency(table, correction=False)
            rate_in = inside / max(n_in, 1)
            rate_out = outside / max(n_out, 1)
            rows.append({
                "cluster": label,
                "pattern": pattern,
                "words": len(words),
                "patients_in_cluster_with_pattern": inside,
                "prevalence_in_cluster": rate_in,
                "prevalence_elsewhere": rate_out,
                "log2_ratio": float(
                    np.log2(max(rate_in, 1e-9) / max(rate_out, 1e-9))
                ),
                "p": float(p),
            })
    if not rows:
        logger.warning("no pattern cleared the small-cell floor by class")
        return pl.DataFrame()
    frame = pl.DataFrame(rows)
    frame = frame.with_columns(
        p_adjusted=(pl.col("p") * frame.height).clip(upper_bound=1.0)
    ).sort("p")
    significant = frame.filter(pl.col("p_adjusted") < 0.05)
    logger.info(
        f"  {frame.height:,} (pattern, class) tests, {significant.height:,} "
        "significant after Bonferroni"
    )
    for row in significant.head(top).iter_rows(named=True):
        logger.info(
            f"    {row['cluster']}  {row['pattern']:<28} "
            f"{row['prevalence_in_cluster']:>6.1%} vs "
            f"{row['prevalence_elsewhere']:>6.1%}  "
            f"log2 {row['log2_ratio']:+.2f}"
        )
    return frame


def _contains_in_order(words, pattern):
    """is `pattern` a subsequence of `words`, gaps allowed"""
    position = 0
    for item in pattern:
        while position < len(words) and words[position] != item:
            position += 1
        if position == len(words):
            return False
        position += 1
    return True

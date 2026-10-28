#!/usr/bin/env python3

"""
byte-pair encoding over the MAP/pressor state sequences

States are letters, merged units are words, a trajectory is a sentence. BPE
learns the words instead of being told their length.

HOW THIS DIFFERS FROM THE n-GRAM MOTIFS. The motif analysis enumerates every
2- to 4-gram and prunes by frequency, which gives a BAG of all substrings at a
fixed maximum length. BPE greedily merges the commonest adjacent pair, which
gives a SEGMENTATION into variable-length units. Three consequences:

  * Long runs collapse. The raw motif vocabulary's top three entries were
    "00", "000" and "0000" in 73-78% of patients -- three separate words all
    meaning "stayed off pressors", crowding out the vocabulary. BPE merges
    them into one unit and keeps it distinct from "00", so duration is
    represented compactly rather than either exploded or, as under run-length
    encoding, thrown away.
  * A sentence has word ORDER. Only a segmentation supports asking what a
    trajectory starts with, or whether a unit appears as a prefix or a suffix.
    A bag of overlapping substrings cannot.
  * Unit length is learned. Nothing caps a word at four states, so a recurring
    seven-state manoeuvre can become a single word if the data contains one.

BOUNDARY SYMBOLS ARE HARD. Merges never cross one. For the coarse sequences
that is the terminal "E" alone: allowing merges across it would let BPE learn
words that encode death, and every downstream comparison would inherit an
outcome it was supposed to be predicting. The fine-grained sequences add "_"
at each 6-hour window edge, which is what makes a word there a within-window
manoeuvre rather than an arbitrary span -- and is only worth doing because
those sequences carry several letters per window. On the coarse sequences,
where one letter IS one window, a "_" after every letter would make every
letter its own word and block every merge.

VOCABULARIES ARE FITTED PER COHORT, by choice. The same string therefore means
different things in the cardiogenic and septic reports, and the two
vocabularies are not comparable -- a word is whatever was frequent in its own
cohort. Stated here because nothing downstream can detect it.
"""

from __future__ import annotations

from collections import Counter

import polars as pl


def learn_merges(sequences, n_merges: int = 100, boundary: str = "E",
                 suffix: str | None = None, logger=None):
    """
    the merge list, in the order BPE learned it

    Returns [(left, right), ...]. Applying them in order to a fresh sequence
    reproduces the same segmentation, which is what makes the vocabulary
    portable to a patient the fit never saw.
    """
    corpus = [
        [c for c in sequence] for sequence in sequences if sequence
    ]
    merges: list[tuple[str, str]] = []
    for step in range(n_merges):
        pairs = Counter()
        for tokens in corpus:
            for a, b in zip(tokens, tokens[1:]):
                # A HARD boundary is never merged at all. For the coarse
                # sequences that is the terminal "E": merging across it would
                # let BPE learn words that encode death, and every downstream
                # comparison would inherit an outcome it was meant to be
                # predicting. `boundary` is a SET of characters, not one --
                # `boundary in (a, b)` would compare the whole string against
                # each token and match nothing.
                if a in boundary or b in boundary:
                    continue
                # A SUFFIX boundary is absorbed rightward instead, the way
                # SentencePiece attaches its word marker. "320" + "_" becomes
                # the word "320_", which means "weaned, and that is how the
                # window ended" and is a different thing from a "320" sitting
                # mid-window. What stays forbidden is extending PAST a marker:
                # once a token ends with one, it cannot take another token on
                # its right, so no word ever spans two windows and "_" only
                # ever appears as a token's last character.
                if suffix and a.endswith(suffix):
                    continue
                pairs[(a, b)] += 1
        if not pairs:
            break
        (left, right), count = pairs.most_common(1)[0]
        if count < 2:
            break
        merged = left + right
        for index, tokens in enumerate(corpus):
            out, i = [], 0
            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):
                    out.append(merged)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            corpus[index] = out
        merges.append((left, right))
        if logger is not None and (step + 1) % 25 == 0:
            logger.info(
                f"    merge {step + 1:>3}: {left!r}+{right!r} -> {merged!r} "
                f"({count:,} occurrences)"
            )
    return merges


def apply_merges(sequence: str, merges) -> list[str]:
    """segment one sequence with a learned merge list"""
    tokens = [c for c in sequence]
    for left, right in merges:
        merged = left + right
        out, i = [], 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                out.append(merged)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return tokens


def _neighbours(segmented):
    """
    counts of where each word sits and what sits either side of it

    Shared by `vocabulary_table` and `word_context` so the two can never
    disagree about what precedes what -- they were separate tables with
    separate passes over the same segmentation, and only one of them carried
    the context at all.

    NEIGHBOURS ARE LITERALLY ADJACENT. With the window marker absorbed into
    the token that closes a window -- "320_" the way SentencePiece writes
    "ing_" -- the next token simply IS the next word, and no skipping is
    needed. An earlier version stood "_" alone between every pair and had to
    skip it; attaching it instead makes the question "what came before this"
    answerable by reading the sequence.

    A BARE "_" still survives where a window held no record at all, since it
    has nothing on its left to attach to. That is informative rather than
    noise: a word followed by "_" is a word followed by a window in which
    nothing was charted.
    """
    occurrences, carriers = Counter(), Counter()
    opens, closes = Counter(), Counter()
    before, after = {}, {}
    for tokens in segmented:
        if not tokens:
            continue
        carriers.update(set(tokens))
        last = len(tokens) - 1
        for position, word in enumerate(tokens):
            occurrences[word] += 1
            if position == 0:
                opens[word] += 1
            if position == last:
                closes[word] += 1
        for position, word in enumerate(tokens):
            if position > 0:
                before.setdefault(word, Counter())[tokens[position - 1]] += 1
            if position < last:
                after.setdefault(word, Counter())[tokens[position + 1]] += 1
    n_patients = sum(1 for t in segmented if t)
    return occurrences, carriers, opens, closes, before, after, n_patients


def _render_neighbours(counter, denominator, edge_label, boundary="_", top=3):
    """
    the commonest neighbours as one readable cell

    The window marker is spelled out rather than printed as "_". On the
    fine-grained sequences it is the commonest neighbour of almost every word,
    and "_ 61%" in a table column means nothing to a reader who has not read
    the tokeniser -- what it means is that the word starts, or ends, a 6-hour
    window.
    """
    if not counter:
        return edge_label
    pieces = []
    for word, n in counter.most_common(top):
        shown = (
            "(window edge)" if word == boundary
            else "(record ended)" if word == "E" else word
        )
        pieces.append(f"{shown} {n / max(denominator, 1):.0%}")
    return ", ".join(pieces)


def vocabulary_table(sequences, merges, states, logger, top: int = 30):
    """
    the learned words, how common they are, and what they spell out

    `compression` is the ratio of letters to words across the corpus. It is
    worth reading as a result rather than a diagnostic: a cohort that
    compresses hard is one whose trajectories are stereotyped, and a patient
    who compresses little is following a course the cohort rarely takes.
    """
    from clifcs.perfusion import STATE_SYMBOLS

    def spell(word: str) -> str:
        names = []
        for character in word:
            position = STATE_SYMBOLS.find(character)
            names.append(
                states[position] if 0 <= position < len(states) else character
            )
        # a run of one state reads better as "x N" than as the name repeated
        if len(set(names)) == 1 and len(names) > 1:
            return f"{names[0]}  x{len(names)}"
        return "  ->  ".join(names)

    segmented = [apply_merges(s, merges) for s in sequences]
    letters = sum(len(s) for s in sequences)
    words = sum(len(t) for t in segmented)
    (occurrences, carriers, opens, closes, before, after,
     n_patients) = _neighbours(segmented)

    logger.info(
        f"BPE: {len(merges)} merges, {len(carriers):,} distinct words, "
        f"compression {letters / max(words, 1):.2f} letters per word"
    )
    rows = []
    for word, n in occurrences.most_common():
        share = carriers[word] / max(len(sequences), 1)
        # Denominators exclude the occurrences that HAVE no neighbour on that
        # side: a word appearing only at the start of a sentence has no
        # preceding word, and dividing by its total occurrences would make
        # every share of its predecessors sum to less than one for a reason
        # that has nothing to do with what precedes it.
        rows.append({
            "word": word,
            "length": len(word),
            "occurrences": n,
            "patients": carriers[word],
            "share_of_patients": share,
            "opens_sentence": opens[word] / max(n, 1),
            "closes_sentence": closes[word] / max(n, 1),
            "commonly_preceded_by": _render_neighbours(
                before.get(word), sum(before.get(word, {}).values()),
                "(always opens)",
            ),
            "commonly_followed_by": _render_neighbours(
                after.get(word), sum(after.get(word, {}).values()),
                "(always closes)",
            ),
            "meaning": spell(word),
        })
    for row in rows[:top]:
        if row["length"] > 1:
            logger.info(
                f"    {row['word']:<10} {row['share_of_patients']:>6.1%} of "
                f"patients   {row['meaning'][:72]}"
            )
    return pl.DataFrame(rows), segmented, letters / max(words, 1)


def edit_distance(a, b) -> int:
    """Levenshtein between two token sequences, on WORDS not letters"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, token in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (token != other),
            ))
        previous = current
    return previous[-1]


def word_context(segmented, states, logger, top: int = 25):
    """
    for each common word: how often it appears, where, and what surrounds it

    A vocabulary alone says which units exist. This says how they are USED --
    whether a word opens a sentence, closes one, or sits inside; and which
    words habitually come before and after it. That is the prefix/suffix
    question, and it only has an answer because BPE produces a segmentation:
    a bag of overlapping n-grams has no notion of what follows what.

    `share_of_patients` is the fraction of patients whose sentence contains
    the word at least once, which is the honest denominator for "how common
    is this" -- `occurrences` double-counts a patient who repeats a word, and
    a word repeated eight times in one patient is not eight patients' worth of
    evidence.

    START and END are reported as shares of that word's own occurrences, so a
    word that only ever opens a sentence reads 100% even if it is rare.
    """
    from collections import Counter

    from clifcs.perfusion import STATE_SYMBOLS

    def spell(word: str) -> str:
        names = []
        for character in word:
            position = STATE_SYMBOLS.find(character)
            names.append(
                states[position] if 0 <= position < len(states) else character
            )
        if len(set(names)) == 1 and len(names) > 1:
            return f"{names[0]}  x{len(names)}"
        return "  ->  ".join(names)

    (occurrences, carriers, opens, closes, before, after,
     n_patients) = _neighbours(segmented)
    rows = []
    logger.info("the commonest BPE words, and what surrounds them:")
    for word, count in occurrences.most_common(top):
        preceding = before.get(word, Counter())
        following = after.get(word, Counter())
        inner = max(count - opens[word], 1)
        outer = max(count - closes[word], 1)
        top_before = _render_neighbours(preceding, inner, "(sentence start)")
        top_after = _render_neighbours(following, outer, "(sentence end)")
        rows.append({
            "word": word,
            "length": len(word),
            "share_of_patients": carriers[word] / max(n_patients, 1),
            "occurrences": count,
            "opens_sentence": opens[word] / count,
            "closes_sentence": closes[word] / count,
            "commonly_preceded_by": top_before,
            "commonly_followed_by": top_after,
            "meaning": spell(word),
        })
        logger.info(
            f"    {word:<12} {carriers[word] / max(n_patients, 1):>6.1%} of "
            f"patients  opens {opens[word] / count:>5.0%}  "
            f"closes {closes[word] / count:>5.0%}"
        )
        logger.info(f"        before: {top_before}")
        logger.info(f"        after:  {top_after}")
        logger.info(f"        means:  {spell(word)[:88]}")
    return pl.DataFrame(rows)

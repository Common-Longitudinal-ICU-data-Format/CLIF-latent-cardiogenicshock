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

THE TERMINAL SYMBOL IS A HARD BOUNDARY. Merges never cross it. Allowing them
would let BPE learn words that encode death, and every downstream comparison
would inherit an outcome it was supposed to be predicting.

VOCABULARIES ARE FITTED PER COHORT, by choice. The same string therefore means
different things in the cardiogenic and septic reports, and the two
vocabularies are not comparable -- a word is whatever was frequent in its own
cohort. Stated here because nothing downstream can detect it.
"""

from __future__ import annotations

from collections import Counter

import polars as pl


def learn_merges(sequences, n_merges: int = 100, boundary: str = "E",
                 logger=None):
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
                # never merge across the terminal symbol
                if boundary in (a, b):
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
    carriers = Counter()
    for tokens in segmented:
        carriers.update(set(tokens))
    occurrences = Counter()
    for tokens in segmented:
        occurrences.update(tokens)

    logger.info(
        f"BPE: {len(merges)} merges, {len(carriers):,} distinct words, "
        f"compression {letters / max(words, 1):.2f} letters per word"
    )
    rows = []
    for word, n in occurrences.most_common():
        share = carriers[word] / max(len(sequences), 1)
        rows.append({
            "word": word,
            "length": len(word),
            "occurrences": n,
            "patients": carriers[word],
            "share_of_patients": share,
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

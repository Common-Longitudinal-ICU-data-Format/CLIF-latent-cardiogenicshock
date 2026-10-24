#!/usr/bin/env python3

"""
state motifs as words, and trajectories as sentences of them

Each patient's path through the perfusion states is a string. This chops that
string into short overlapping subsequences -- "10" for a wean, "01" for an
escalation, "3012" for a longer manoeuvre -- keeps the ones common enough to
be worth naming, and represents the patient as a bag of them. Clustering that
representation groups patients who make the same MOVES, wherever in the week
they make them.

WHY THIS AND NOT THE SEQUENCE DIRECTLY. The existing clusterings one-hot each
position and cluster on Hamming distance, which scores two patients by whether
they are in the same state at the same hour. That treats a wean at +60h and a
wean at +96h as maximally different at every position where they disagree,
even though both are the same clinical manoeuvre. A motif is position-free, so
both contribute the same "10" and land together.

RAW AND RUN-LENGTH ENCODED, both. On the raw string "1111110000" the motifs
carry duration -- "11", "11", "10", "00" -- so holding a state for five
windows looks different from holding it for one. Collapsing runs first gives
"10", which matches any wean regardless of how long either phase lasted.
Duration is sometimes the signal and sometimes the nuisance, and which one it
is here is a question rather than an assumption, so both are built.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl


def run_length_encode(sequence: str) -> str:
    """'1111110000' -> '10'; adjacent duplicates collapsed"""
    if not sequence:
        return sequence
    out = [sequence[0]]
    for character in sequence[1:]:
        if character != out[-1]:
            out.append(character)
    return "".join(out)


def extract_motifs(sequences, sizes=(2, 3, 4), min_patients: float = 0.01):
    """
    the n-grams common enough to be words, with the patients carrying each

    `min_patients` is a share of patients, not of occurrences: a motif that
    appears a hundred times in one patient is an idiosyncrasy, while one
    appearing once in a hundred patients is a pattern. Counting distinct
    patients is what tells those apart.
    """
    per_patient = []
    for sequence in sequences:
        found = Counter()
        for size in sizes:
            for start in range(len(sequence) - size + 1):
                found[sequence[start:start + size]] += 1
        per_patient.append(found)

    carriers = Counter()
    for found in per_patient:
        carriers.update(found.keys())
    floor = max(2, int(round(min_patients * len(sequences))))
    vocabulary = sorted(m for m, n in carriers.items() if n >= floor)
    return vocabulary, per_patient, carriers


def motif_matrix(vocabulary, per_patient):
    """tf-idf over motifs, patients by vocabulary"""
    index = {motif: i for i, motif in enumerate(vocabulary)}
    counts = np.zeros((len(per_patient), len(vocabulary)), dtype=np.float32)
    for row, found in enumerate(per_patient):
        for motif, n in found.items():
            position = index.get(motif)
            if position is not None:
                counts[row, position] = n
    # sublinear tf, because a motif occurring eight times in one patient is
    # not eight times the evidence of occurring once
    tf = np.where(counts > 0, 1.0 + np.log(np.maximum(counts, 1.0)), 0.0)
    document_frequency = (counts > 0).sum(axis=0)
    idf = np.log((1 + len(per_patient)) / (1 + document_frequency)) + 1.0
    weighted = tf * idf
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    return np.divide(weighted, np.maximum(norms, 1e-9))


def describe_motifs(vocabulary, carriers, states, n_patients, logger,
                    top: int = 20):
    """log the commonest motifs, spelled out in state names"""
    def spell(motif: str) -> str:
        from clifcs.perfusion import STATE_SYMBOLS
        names = []
        for character in motif:
            position = STATE_SYMBOLS.find(character)
            names.append(
                states[position] if 0 <= position < len(states) else character
            )
        return "  ->  ".join(names)

    logger.info(f"motif vocabulary: {len(vocabulary):,} motifs")
    ranked = sorted(vocabulary, key=lambda m: -carriers[m])[:top]
    rows = []
    for motif in ranked:
        share = carriers[motif] / max(n_patients, 1)
        logger.info(f"    {motif:<6} {share:>6.1%} of patients   {spell(motif)}")
        rows.append({
            "motif": motif,
            "length": len(motif),
            "patients": carriers[motif],
            "share_of_patients": share,
            "meaning": spell(motif),
        })
    return pl.DataFrame(rows)

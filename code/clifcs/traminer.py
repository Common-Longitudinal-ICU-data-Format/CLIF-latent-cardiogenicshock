#!/usr/bin/env python3

"""
OMspell sequence dissimilarity, via TraMineR in R

WHY NOT THE ALIGNER WE ALREADY HAD. Handbook discovery used a semi-global
alignment written here. Benchmarked on a test set built from the phenotypes we
expect to find -- a course that stays on low pressors, one that weans
repeatedly, one held at a high dose -- each perturbed the way real data is
messy (a single-hour spike, a spurious off-pressor hour, a shifted start), that
aligner came last of fourteen measures: 58% nearest-neighbour purity with a
NEGATIVE margin. A perturbed low-pressor course was more often closer to a
different phenotype than to its own clean sibling. Run-length collapsing fixed
the duration sensitivity and broke something else, because after collapsing an
eight-hour plateau and a one-hour artefact are the same character.

OMspell edits SPELLS and prices their duration separately through `expcost`, so
duration is discounted rather than discarded. It scored 100% purity with the
widest margin of the fourteen.

WHAT THIS COSTS US. An R dependency in the middle of a Python pipeline, and a
CSV round-trip per call. The round-trip is affordable only because discovery
already clusters a SAMPLE and assigns the cohort to medoids: the full 6,269 x
6,269 matrix would be 314 MB and is never built, here or before.

WHAT IT DOES NOT FIX. Nothing about this makes the handbooks causal, and
nothing about it addresses treatment-by-indication. It makes "these two courses
were managed alike" a more honest claim on messy data; that is all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

#: project scratch, never under Box -- Box Drive holds open handles on synced
#: files and a matrix written there can be locked mid-write
SCRATCH = Path(os.environ.get("CLIFCS_SCRATCH",
                              Path.home() / "clif-cs-scratch"))

R_SCRIPT = Path(__file__).resolve().parents[1] / "R" / "omspell.R"


def find_rscript() -> str:
    """Rscript from the environment, the PATH, or the usual Windows install"""
    explicit = os.environ.get("CLIFCS_RSCRIPT")
    if explicit:
        return explicit
    found = shutil.which("Rscript")
    if found:
        return found
    candidates = sorted(
        Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"), reverse=True
    )
    if candidates:
        return str(candidates[0])
    raise RuntimeError(
        "Rscript not found. Install R and TraMineR, or set CLIFCS_RSCRIPT. "
        "Handbook discovery can still run with --distance align."
    )


def available() -> bool:
    """can we reach R and TraMineR at all? asked before a long run, not during"""
    try:
        rscript = find_rscript()
    except RuntimeError:
        return False
    probe = (
        'lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library",'
        ' paste(R.version$major, substr(R.version$minor, 1, 1), sep = "."));'
        'if (dir.exists(lib)) .libPaths(c(lib, .libPaths()));'
        'cat(as.character(requireNamespace("TraMineR", quietly = TRUE)))'
    )
    try:
        done = subprocess.run([rscript, "-e", probe], capture_output=True,
                              text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "TRUE" in done.stdout


def _write_sequences(sequences, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("sequence\n")
        for sequence in sequences:
            handle.write(f"{sequence}\n")


def _alphabet(sequences) -> str:
    return "".join(sorted({character for s in sequences for character in s}))


def _run(arguments, logger, label):
    rscript = find_rscript()
    done = subprocess.run([rscript, str(R_SCRIPT), *arguments],
                          capture_output=True, text=True)
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-6:]
        raise RuntimeError(
            f"OMspell ({label}) failed:\n  " + "\n  ".join(tail)
        )
    if logger is not None:
        for line in done.stdout.strip().splitlines():
            if line.startswith("wrote"):
                logger.info(f"    R: {line}")


def omspell_matrix(sequences, logger=None, expcost=0.5, indel=1.0,
                   norm="auto", costs="TRATE", tag="sample"):
    """
    square OMspell dissimilarity among `sequences`

    Returned normalised to [0, 1] by default (`norm="auto"` selects the
    Yujian-Bo normalised metric for OMspell), so the fit floors and linkage
    cutoffs mean the same thing whatever the cohort's sequence lengths are.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    source = SCRATCH / f"omspell_{tag}_in.csv"
    target = SCRATCH / f"omspell_{tag}_out.csv"
    _write_sequences(sequences, source)
    if logger is not None:
        logger.info(
            f"  OMspell over {len(sequences):,} courses "
            f"(expcost {expcost}, {costs} costs)"
        )
    _run([
        "--mode=pairwise", f"--input={source}", f"--output={target}",
        f"--alphabet={_alphabet(sequences)}", f"--expcost={expcost}",
        f"--indel={indel}", f"--norm={norm}", f"--sm={costs}",
    ], logger, "pairwise")
    matrix = np.loadtxt(target, delimiter=",", dtype=np.float32)
    if matrix.shape != (len(sequences), len(sequences)):
        raise RuntimeError(
            f"OMspell returned {matrix.shape}, expected "
            f"{(len(sequences), len(sequences))}"
        )
    return matrix


def omspell_to_refs(sequences, references, logger=None, expcost=0.5,
                    indel=1.0, norm="auto", costs="TRATE", tag="assign"):
    """
    (n, k) OMspell dissimilarity from every sequence to each reference course

    The references are appended to the sequences rather than compared across
    two seqdef objects, so both sides share one alphabet and one measured
    substitution matrix. Comparing against costs estimated from a different set
    would make the distances incommensurable with the clustering that produced
    the medoids.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    combined = list(references) + list(sequences)
    source = SCRATCH / f"omspell_{tag}_in.csv"
    target = SCRATCH / f"omspell_{tag}_out.csv"
    _write_sequences(combined, source)
    index = ",".join(str(i + 1) for i in range(len(references)))  # R is 1-based
    if logger is not None:
        logger.info(
            f"  OMspell assigning {len(sequences):,} courses to "
            f"{len(references)} medoids"
        )
    _run([
        "--mode=refs", f"--input={source}", f"--output={target}",
        f"--alphabet={_alphabet(combined)}", f"--expcost={expcost}",
        f"--indel={indel}", f"--norm={norm}", f"--sm={costs}",
        f"--refs={index}",
    ], logger, "refs")
    matrix = np.loadtxt(target, delimiter=",", dtype=np.float32)
    matrix = matrix.reshape(len(combined), len(references))
    return matrix[len(references):, :]


def pam(square, k, logger=None, max_swaps=200):
    """
    partitioning around medoids on a precomputed dissimilarity

    WHY NOT AVERAGE LINKAGE. The aligner's handbooks were cut from an
    average-linkage tree, which worked on alignment distances and fails on
    these: OMspell dissimilarities sit in a narrow band (mean 0.65 on the
    cardiogenic sample) and average linkage chains, giving 511/56/29/2/1/1
    where ward gives 196/107/102/75/67/53. Four of six handbooks were
    singletons -- a clustering artefact, not a finding about treatment.

    PAM is used rather than ward because discovery's next step needs a MEDOID,
    and PAM optimises medoids directly instead of extracting one after the fact
    from a cut whose objective was something else. It is also what the sequence
    -analysis literature pairs with seqdist.

    BUILD then SWAP, both deterministic, so no seed enters here: the only
    randomness in discovery stays the choice of which courses are sampled.
    """
    n = len(square)
    # BUILD: the first medoid minimises total distance, each next one is
    # whichever point most reduces the cost of what is already chosen
    chosen = [int(np.argmin(square.sum(axis=1)))]
    while len(chosen) < k:
        nearest = square[:, chosen].min(axis=1)
        gain = np.maximum(nearest[None, :] - square, 0).sum(axis=1)
        gain[chosen] = -np.inf
        chosen.append(int(np.argmax(gain)))
    medoids = np.array(chosen)

    # SWAP: exchange a medoid for a non-medoid whenever that lowers total cost
    cost = square[:, medoids].min(axis=1).sum()
    for _ in range(max_swaps):
        best = (cost, None, None)
        outside = np.setdiff1d(np.arange(n), medoids)
        for position in range(k):
            trial = medoids.copy()
            for candidate in outside:
                trial[position] = candidate
                trial_cost = square[:, trial].min(axis=1).sum()
                if trial_cost < best[0] - 1e-9:
                    best = (trial_cost, position, candidate)
        if best[1] is None:
            break
        cost, position, candidate = best
        medoids[position] = candidate
    labels = np.argmin(square[:, medoids], axis=1)
    if logger is not None:
        sizes = np.bincount(labels, minlength=k)
        logger.info(
            f"    PAM cost {cost:.1f}; cluster sizes "
            + ", ".join(str(int(s)) for s in sorted(sizes, reverse=True))
        )
    return labels, medoids

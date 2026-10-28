#!/usr/bin/env python3

"""
tests for clifcs.align

The load-bearing test is `test_batch_matches_a_reference_implementation`. The
batched kernel advances every pair through cell (i, j) together, which is the
only axis an affine-gap recurrence leaves free to vectorise, and that
restructuring is exactly the kind that produces a plausible-looking matrix with
the wrong contents. A naive Gotoh written independently is the check.

`test_traceback_cost_equals_the_score` is the second. Affine gaps need a
backpointer per LAYER: with one pointer array the traceback cannot tell a gap
that was opened from one that was continued, so it silently returns a path
whose cost differs from the score that was reported for it. Recomputing the
path's own cost catches that and nothing else does.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))

from clifcs.align import (  # noqa: E402
    align_batch, align_many, divergence_point, encode, similarity_matrix,
    transition_distributions, walk,
)

ALPHABET = "HRNDWwFSdo"
OPEN, EXTEND = 4.0, 1.0


def reference(left, right, matrix, alphabet, gap_open=OPEN, gap_extend=EXTEND):
    """plain Gotoh: anchored start, endpoint free anywhere"""
    index = {s: i for i, s in enumerate(alphabet)}
    la, lb = len(left), len(right)
    negative = -1e9
    m = [[negative] * (lb + 1) for _ in range(la + 1)]
    x = [[negative] * (lb + 1) for _ in range(la + 1)]
    y = [[negative] * (lb + 1) for _ in range(la + 1)]
    m[0][0] = 0.0
    for j in range(1, lb + 1):
        y[0][j] = -(gap_open + (j - 1) * gap_extend)
    for i in range(1, la + 1):
        x[i][0] = -(gap_open + (i - 1) * gap_extend)
        for j in range(1, lb + 1):
            score = float(matrix[index[left[i - 1]], index[right[j - 1]]])
            m[i][j] = max(m[i-1][j-1], x[i-1][j-1], y[i-1][j-1]) + score
            x[i][j] = max(m[i-1][j] - gap_open, x[i-1][j] - gap_extend)
            y[i][j] = max(m[i][j-1] - gap_open, y[i][j-1] - gap_extend)
    best = 0.0
    for i in range(la + 1):
        for j in range(lb + 1):
            best = max(best, m[i][j])
    return best


def path_cost(path, gap_open=OPEN, gap_extend=EXTEND):
    """what the traceback itself costs, charged affinely"""
    total, run = 0.0, None
    for left_index, _right_index, score in path:
        if score is not None:
            total += score
            run = None
        else:
            kind = "x" if left_index >= 0 else "y"
            total -= gap_extend if run == kind else gap_open
            run = kind
    return total


@pytest.fixture
def matrix():
    rng = np.random.default_rng(7)
    size = len(ALPHABET)
    raw = rng.normal(0.0, 2.0, (size, size)).astype(np.float32)
    built = ((raw + raw.T) / 2.0).astype(np.float32)
    np.fill_diagonal(built, 3.0)
    return built


@pytest.fixture
def random_pairs():
    rng = np.random.default_rng(11)
    pairs = []
    for _ in range(120):
        la = int(rng.integers(2, 14))
        lb = int(rng.integers(2, 14))
        pairs.append((
            "".join(rng.choice(list(ALPHABET), la)),
            "".join(rng.choice(list(ALPHABET), lb)),
        ))
    return pairs


def test_batch_matches_a_reference_implementation(matrix, random_pairs):
    left = [a for a, _ in random_pairs]
    right = [b for _, b in random_pairs]
    left_codes, left_len, _ = encode(left, ALPHABET)
    right_codes, right_len, _ = encode(right, ALPHABET)
    best, _bi, _bj, _ptr = align_batch(
        left_codes, left_len, right_codes, right_len, matrix, OPEN, EXTEND
    )
    for k, (a, b) in enumerate(random_pairs):
        assert float(best[k]) == pytest.approx(
            reference(a, b, matrix, ALPHABET), abs=1e-3
        ), f"score disagrees on {a} / {b}"


def test_traceback_cost_equals_the_score(matrix, random_pairs):
    left = [a for a, _ in random_pairs]
    right = [b for _, b in random_pairs]
    left_codes, left_len, _ = encode(left, ALPHABET)
    right_codes, right_len, _ = encode(right, ALPHABET)
    best, bi, bj, pointers = align_batch(
        left_codes, left_len, right_codes, right_len, matrix, OPEN, EXTEND
    )
    for k in range(len(random_pairs)):
        path = walk(
            pointers[k], matrix, left_codes[k], right_codes[k], bi[k], bj[k]
        )
        assert path_cost(path) == pytest.approx(float(best[k]), abs=1e-3)


def test_start_is_anchored_not_free(matrix):
    """
    a shared course offset by three hours must NOT align for free

    If the start were free the aligner would slide one course forward until it
    matched, which is the claim under test rather than an allowance to make.
    """
    identity = np.full((len(ALPHABET), len(ALPHABET)), -2.0, dtype=np.float32)
    np.fill_diagonal(identity, 3.0)
    aligned = align_many(
        ["SSSWWWddd", "dddSSSWWWddd"], ["SSSWWWddd", "SSSWWWddd"],
        ALPHABET, identity, OPEN, EXTEND,
    )
    assert aligned[0]["alignment_score"] > aligned[1]["alignment_score"], (
        "an offset course scored as well as an aligned one, so the start "
        "gap is being given away"
    )


def test_endpoint_is_not_pinned_to_an_edge():
    """
    the aligner must not buy its way to the last row with interior gaps

    Pinning the endpoint to the final row or column -- the textbook
    semi-global rule -- forces one course to be consumed in full, and on this
    pair it produced SSSS-----D against SSSSWWWWWW: eight points of gap spent
    to reach a single mismatch. The alignment should stop at hour 4.
    """
    identity = np.full((len(ALPHABET), len(ALPHABET)), -2.0, dtype=np.float32)
    np.fill_diagonal(identity, 3.0)
    result = align_many(
        ["SSSSDDDDDD"], ["SSSSWWWWWW"], ALPHABET, identity, OPEN, EXTEND,
    )[0]
    assert result["left_hour"] == 4
    assert result["gap_columns"] == 0


def test_a_prefix_is_censored_not_a_divergence():
    """a record that simply ended is not evidence the courses parted"""
    identity = np.full((len(ALPHABET), len(ALPHABET)), -2.0, dtype=np.float32)
    np.fill_diagonal(identity, 3.0)
    short, long = "SSSddd", "SSSdddWWWooo"
    result = align_many([short], [long], ALPHABET, identity, OPEN, EXTEND)[0]
    assert result["reason"] == "ran_out"
    assert result["diverged"] is False


def test_identical_courses_agree_throughout():
    identity = np.full((len(ALPHABET), len(ALPHABET)), -2.0, dtype=np.float32)
    np.fill_diagonal(identity, 3.0)
    result = align_many(
        ["SSSdddWWW"], ["SSSdddWWW"], ALPHABET, identity, OPEN, EXTEND
    )[0]
    assert result["reason"] == "agreed_throughout"
    assert result["identity"] == pytest.approx(1.0)


def test_courses_that_never_agree_diverge_at_onset():
    """
    hour zero is a divergence and must be reported, not dropped

    Bucketing these as "no shared prefix" truncates the hour distribution away
    from exactly the pairs that parted earliest, which are the ones the
    analysis is looking for.
    """
    identity = np.full((len(ALPHABET), len(ALPHABET)), -2.0, dtype=np.float32)
    np.fill_diagonal(identity, 3.0)
    result = align_many(
        ["DDDDDDDDDDDD"], ["WWWWWWWWWWWW"], ALPHABET, identity, OPEN, EXTEND
    )[0]
    assert result["diverged"] is True
    assert result["reason"] == "diverged_at_onset"
    assert result["left_hour"] == 0


def test_similarity_matrix_never_prefers_a_mismatch():
    """
    identity must outscore every substitution, or the aligner can prefer a
    mismatch to a match -- which a BLOSUM-style bootstrap actually produced
    here, finishing with four negative self-scores
    """
    rng = np.random.default_rng(3)
    sequences = [
        "".join(rng.choice(list(ALPHABET), int(rng.integers(6, 40))))
        for _ in range(400)
    ]
    built = similarity_matrix(sequences, ALPHABET)
    diagonal = np.diag(built)
    off = built[~np.eye(len(ALPHABET), dtype=bool)]
    assert np.all(diagonal > 0), "a state scored against itself must be positive"
    assert off.max() < diagonal.min(), "a substitution outscored an identity"


def test_transition_distributions_are_normalised():
    rng = np.random.default_rng(5)
    sequences = [
        "".join(rng.choice(list(ALPHABET), int(rng.integers(6, 40))))
        for _ in range(200)
    ]
    forward, backward = transition_distributions(sequences, ALPHABET)
    assert np.allclose(forward.sum(axis=1), 1.0)
    assert np.allclose(backward.sum(axis=1), 1.0)


def test_divergence_point_handles_an_empty_path():
    result = divergence_point([], 0, 0, 0, 0)
    assert result["diverged"] is False
    assert result["reason"] == "agreed_throughout"

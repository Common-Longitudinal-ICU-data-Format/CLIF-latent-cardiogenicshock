#!/usr/bin/env python3

"""
semi-global sequence alignment over the hourly state strings, and the
divergence point it hands back

BPE found short words. A short word is the "ing" problem: `dd`, `SS`, `oo` are
everywhere, they carry almost no mortality gap, and a motif shared by most of
the cohort cannot be where two courses part company. Alignment asks the
question the other way round -- not "which substring recurs" but "up to which
hour were these two patients having the same illness, and where did they stop".

WHY SEMI-GLOBAL AND NOT LOCAL. Smith-Waterman finds the best-matching
subregion anywhere in either sequence, which is the right tool when you do not
know where the homology starts. Here you do: every string starts at shock
onset, which is a genuine common origin in the sense a shared ancestor is. So
the alignment is ANCHORED at position 0 -- no free gap before the first column
-- and left free at the end, because two courses ending at different hours
should not be charged for hours one of them did not live to have.

THE SUBSTITUTION MATRIX IS MEASURED, NOT ASSUMED. Edit distance scores every
mismatch at 1, which says `stable hold` versus `drifting hold` is as big a
difference as `wean held` versus `deteriorating`. Clinically it is not, and
that uniform cost is part of why matching on raw strings found nobody.
`similarity_matrix` scores a substitution by how alike the two states BEHAVE --
the Jensen-Shannon distance between their forward and backward neighbour
distributions. Its docstring records why the obvious BLOSUM-style approach of
counting aligned columns was tried first and had to be abandoned.

TWO THINGS THE DIVERGENCE POINT IS NOT.

  It is not causal. The hour at which two aligned courses part is the hour
  their recorded states stopped agreeing. Nothing here identifies what made
  them part, and the patient who diverges upward was selected to by everything
  that came before.

  It is not a survival time. Free end gaps mean a 12-hour course aligns to the
  first 12 hours of a 90-hour one at no cost, which is correct -- they DID
  match over the hours both were observed -- but it also means a pair can
  "diverge" simply because one record ended. `divergence_point` returns None
  for that case rather than a number, and the None rate is reported, because
  silently scoring it would make the divergence point a length artefact
  wearing a physiology label.
"""

from __future__ import annotations

import numpy as np

NEGATIVE = np.float32(-1e9)

#: traceback pointers, one int8 per cell
STOP, DIAGONAL, UP, LEFT = 0, 1, 2, 3

#: written into the MATCH pointer plane when a local alignment restarts at
#: a cell. It has to be distinct from the three layer indices, which are
#: 0, 1, 2, and from the zero the array is initialised to -- a cell the DP
#: never reached reads as MATCH and must not read as RESET.
RESET = 3


def encode(sequences, alphabet):
    """
    strings -> a padded integer matrix plus their true lengths

    Padding is -1 and is never scored; the length array bounds every loop, so
    a padded column cannot contribute to a score by accident.
    """
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    lengths = np.array([len(s) for s in sequences], dtype=np.int32)
    width = int(lengths.max()) if len(lengths) else 0
    codes = np.full((len(sequences), width), -1, dtype=np.int16)
    unknown = 0
    for row, sequence in enumerate(sequences):
        for column, symbol in enumerate(sequence):
            if symbol in index:
                codes[row, column] = index[symbol]
            else:
                unknown += 1
    return codes, lengths, unknown


def background_frequencies(sequences, alphabet):
    """corpus state frequencies, the null the log-odds are taken against"""
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    counts = np.zeros(len(alphabet), dtype=np.float64)
    for sequence in sequences:
        for symbol in sequence:
            if symbol in index:
                counts[index[symbol]] += 1.0
    total = counts.sum()
    return counts / total if total else counts


def transition_distributions(sequences, alphabet, smoothing=0.5):
    """
    each state's forward and backward neighbour distributions

    Forward is where a state goes next, backward is what it arrives from. Both
    are used because forward alone cannot tell `stable hold` from `off
    pressors` -- each overwhelmingly persists, so their forward rows look
    nearly identical while the states are clinically opposite.
    """
    size = len(alphabet)
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    counts = np.full((size, size), smoothing, dtype=np.float64)
    for sequence in sequences:
        for a, b in zip(sequence, sequence[1:]):
            if a in index and b in index:
                counts[index[a], index[b]] += 1.0
    forward = counts / counts.sum(axis=1, keepdims=True)
    backward = (counts / counts.sum(axis=0, keepdims=True)).T
    return forward, backward


def _js_distance(p, q):
    """Jensen-Shannon distance in [0, 1] -- the metric square root of the
    divergence, taken in bits so the bound is exactly one"""
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(p > 0, p * np.log2(p / np.where(m > 0, m, 1.0)), 0.0)
        right = np.where(q > 0, q * np.log2(q / np.where(m > 0, m, 1.0)), 0.0)
    return float(np.sqrt(max(0.5 * left.sum() + 0.5 * right.sum(), 0.0)))


def similarity_matrix(sequences, alphabet, logger=None, match=3.0,
                      mismatch_at_median=-1.0, margin=0.5):
    """
    a substitution matrix from how alike two states BEHAVE, not from alignments

    WHY THIS REPLACED A BLOSUM-STYLE BOOTSTRAP, which is the obvious thing to
    try and which was tried first. Counting what lands in the same alignment
    column requires alignments, and the only alignments available are the ones
    made with the matrix being estimated. Because extension only passes through
    positively scoring columns, any state the current matrix declines to align
    is never counted, falls to the pseudocount floor, earns a large negative
    log-odds and is then guaranteed not to be aligned on the next round. It is
    a degenerate fixed point, and it was reached: four states finished with
    NEGATIVE self-scores -- `highly responsive` at the -6 clip on a 5.9%
    background, meaning the aligner was penalised for matching a
    highly-responsive hour to a highly-responsive hour. Round-to-round change
    went 1.067 then 0.895; it was not converging, it was walking into the clip.
    Real BLOSUM escapes this only because its blocks were curated
    independently of the matrix, and there is no such external anchor here.

    So substitutability is measured WITHOUT alignment: two states are alike to
    the extent that their forward and backward neighbour distributions are
    alike. The Jensen-Shannon distance between them is a metric bounded in
    [0, 1], which is mapped linearly to a score so that an identical state
    scores `match` and a state at the median off-diagonal distance scores
    `mismatch_at_median`.

    THE DIAGONAL IS ENFORCED, not estimated. A matrix that scores any
    substitution at or above identity is malformed -- it lets the aligner
    prefer a mismatch to a match -- so off-diagonals are capped at
    `match - margin` after the mapping.
    """
    size = len(alphabet)
    forward, backward = transition_distributions(sequences, alphabet)
    distance = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i + 1, size):
            value = 0.5 * (
                _js_distance(forward[i], forward[j])
                + _js_distance(backward[i], backward[j])
            )
            distance[i, j] = distance[j, i] = value

    off = ~np.eye(size, dtype=bool)
    median = float(np.median(distance[off]))
    if median <= 0:
        return (np.eye(size, dtype=np.float32) * np.float32(match))
    slope = (match - mismatch_at_median) / median
    matrix = match - slope * distance
    np.fill_diagonal(matrix, match)
    ceiling = match - margin
    matrix = np.where(off, np.minimum(matrix, ceiling), matrix)
    matrix = matrix.astype(np.float32)

    if logger:
        positive = float(np.mean(matrix[off] > 0))
        logger.info(
            f"  behavioural similarity matrix: diagonal {match:+.1f} "
            f"throughout, off-diagonal mean {float(matrix[off].mean()):+.2f}, "
            f"{positive:.0%} of off-diagonals positive"
        )
        logger.info(
            f"  median Jensen-Shannon distance between state behaviours "
            f"{median:.3f} (maps to {mismatch_at_median:+.1f})"
        )
    return matrix


#: DP layers. M is a match/mismatch column, X consumes the left sequence
#: against a gap, Y consumes the right one.
MATCH, GAP_RIGHT, GAP_LEFT = 0, 1, 2


def align_batch(left_codes, left_len, right_codes, right_len, matrix,
                gap_open=4.0, gap_extend=1.0, want_path=True, local=False):
    """
    affine-gap alignment, vectorised ACROSS PAIRS, anchored or local

    The Gotoh recurrence is sequential in both i and j -- Iy[i][j] needs
    Iy[i][j-1] -- so there is no way to vectorise inside a single alignment.
    What CAN be vectorised is the pair axis: every pair advances through cell
    (i, j) together, so the inner work is an O(1) numpy operation on an
    N-vector and the loop runs max_i * max_j times no matter how many pairs are
    aligned. For 40-hour courses that is 1,600 iterations whether N is 10 or
    100,000.

    THE ENDPOINT IS UNRESTRICTED. Free end gaps on both sequences means the
    alignment may stop at any cell; see the comment at the collection step for
    what goes wrong when it is pinned to an edge instead.

    `local` MAKES THE START UNRESTRICTED TOO, which is Smith-Waterman, and it
    abandons a premise this module was built on. The anchored mode exists
    because every course starts at shock onset and that is a genuine common
    origin -- a shared ancestor, not a coincidence -- so sliding one course
    forward in time until it matched would have been assuming the answer.

    What overrides that premise is the data. Courses like

        A   3200 11221100000012 5550
        C   00000 11122210000043112 5444

    share a long stretch that begins at hour 4 in one and hour 5 in the other,
    after a prologue that matches nothing. Anchored at onset the aligner is
    charged for that prologue and the shared stretch is never found. Local
    alignment finds it and reports WHERE IT STARTED as well as where it ended,
    so the anchoring assumption becomes a measurement -- a pair whose shared
    region really does begin at hour 0 will be reported that way.

    The price is that a local alignment always finds something: two random
    courses will share two or three runs. `divergence_point` therefore takes a
    minimum aligned length below which a pair is reported as having no shared
    course at all, rather than a two-run "agreement" and a divergence hour.

    ALL THREE LAYERS GET BACKPOINTERS. A single pointer array is the usual
    shortcut and it is wrong under affine gaps: it cannot represent "this cell
    continued an existing gap" as distinct from "this cell opened one", so the
    traceback silently reconstructs a path with a different cost than the score
    it was handed. The memory is the price of the traceback being the same
    alignment as the score -- chunk the pair axis rather than dropping a layer.

    Returns the best score, the cell it was found in, and the pointers.
    """
    n = int(left_codes.shape[0])
    la = int(left_len.max()) if n else 0
    lb = int(right_len.max()) if n else 0
    open_cost = np.float32(gap_open)
    extend_cost = np.float32(gap_extend)

    m_previous = np.full((n, lb + 1), NEGATIVE, dtype=np.float32)
    x_previous = np.full((n, lb + 1), NEGATIVE, dtype=np.float32)
    y_previous = np.full((n, lb + 1), NEGATIVE, dtype=np.float32)

    if local:
        # FREE START ON BOTH SEQUENCES: the top row costs nothing, so the
        # alignment may begin at any pair of positions. The gap layers stay
        # unreachable at the border -- a gap cannot be opened before the
        # alignment has started.
        m_previous[:, :] = np.float32(0.0)
    else:
        # ANCHORED START: row 0 and column 0 pay the full affine penalty. A
        # free start gap would let the aligner slide one course forward in time
        # until it matched, which is exactly the claim being tested.
        m_previous[:, 0] = np.float32(0.0)
        for j in range(1, lb + 1):
            y_previous[:, j] = np.float32(-(gap_open + (j - 1) * gap_extend))

    pointers = (
        np.zeros((n, 3, la + 1, lb + 1), dtype=np.int8) if want_path else None
    )

    best = np.zeros(n, dtype=np.float32)
    best_i = np.zeros(n, dtype=np.int32)
    best_j = np.zeros(n, dtype=np.int32)

    for i in range(1, la + 1):
        m_current = np.full((n, lb + 1), NEGATIVE, dtype=np.float32)
        x_current = np.full((n, lb + 1), NEGATIVE, dtype=np.float32)
        y_current = np.full((n, lb + 1), NEGATIVE, dtype=np.float32)
        if local:
            m_current[:, 0] = np.float32(0.0)
        else:
            x_current[:, 0] = np.float32(-(gap_open + (i - 1) * gap_extend))

        left_symbol = left_codes[:, i - 1]
        left_ok = left_symbol >= 0
        safe_left = np.clip(left_symbol, 0, None)
        for j in range(1, lb + 1):
            right_symbol = right_codes[:, j - 1]
            safe_right = np.clip(right_symbol, 0, None)
            score = np.where(
                left_ok & (right_symbol >= 0),
                matrix[safe_left, safe_right],
                np.float32(0.0),
            ).astype(np.float32)

            previous = np.stack(
                (m_previous[:, j - 1], x_previous[:, j - 1],
                 y_previous[:, j - 1])
            )
            came_from = np.argmax(previous, axis=0)
            reached = previous[came_from, np.arange(n)] + score
            if local:
                # Smith-Waterman's clamp. A cell whose best extension scores
                # below zero starts a fresh alignment instead, and is marked
                # RESET so the traceback knows to stop there rather than
                # walking back through a region that contributed nothing.
                restart = reached < 0
                m_current[:, j] = np.where(restart, np.float32(0.0), reached)
                came_from = np.where(restart, np.int64(RESET), came_from)
            else:
                m_current[:, j] = reached

            open_x = m_previous[:, j] - open_cost
            extend_x = x_previous[:, j] - extend_cost
            x_current[:, j] = np.maximum(open_x, extend_x)

            open_y = m_current[:, j - 1] - open_cost
            extend_y = y_current[:, j - 1] - extend_cost
            y_current[:, j] = np.maximum(open_y, extend_y)

            if want_path:
                pointers[:, MATCH, i, j] = came_from.astype(np.int8)
                pointers[:, GAP_RIGHT, i, j] = np.where(
                    open_x >= extend_x, MATCH, GAP_RIGHT
                ).astype(np.int8)
                pointers[:, GAP_LEFT, i, j] = np.where(
                    open_y >= extend_y, MATCH, GAP_LEFT
                ).astype(np.int8)

        # FREE END GAPS ON BOTH SIDES, which means the endpoint may be ANY
        # cell, not one on the last row or column. Restricting it to an edge
        # is the textbook "semi-global" rule and it is wrong for this
        # question: it forces one course to be consumed in full, so the
        # aligner buys its way to the edge with interior gaps -- SSSS-----D
        # against SSSSWWWWWW, paying 8 points of gap to reach one mismatch.
        # Letting the alignment stop anywhere is the BLAST extension rule, and
        # where it stops is exactly the divergence point.
        within = (left_len >= i)
        for j in range(lb + 1):
            usable = within & (right_len >= j)
            if not usable.any():
                continue
            candidate = np.where(usable, m_current[:, j], NEGATIVE)
            better = candidate > best
            best = np.where(better, candidate, best)
            best_i = np.where(better, i, best_i)
            best_j = np.where(better, j, best_j)

        m_previous, x_previous, y_previous = m_current, x_current, y_current

    return best, best_i, best_j, pointers


def walk(pointers, matrix, left_code, right_code, end_i, end_j):
    """
    one pair's alignment path, back from its end cell to the anchor

    `pointers` is this pair's slice, shape (3, la+1, lb+1). Returns the columns
    in forward order as (left_index, right_index, score) with -1 for a gap and
    a None score on gap columns. Taken one pair at a time: the path is only
    needed for the divergence point and for printing examples.
    """
    path = []
    i, j, layer = int(end_i), int(end_j), MATCH
    while i > 0 or j > 0:
        if i == 0:
            path.append((-1, j - 1, None))
            j -= 1
            continue
        if j == 0:
            path.append((i - 1, -1, None))
            i -= 1
            continue
        following = int(pointers[layer, i, j])
        if layer == MATCH and following == RESET:
            # a local alignment restarted here, so this cell scores zero and
            # the alignment BEGINS after it: the column is not part of the
            # path and neither is anything before it
            break
        if layer == MATCH:
            a, b = int(left_code[i - 1]), int(right_code[j - 1])
            path.append((i - 1, j - 1, float(matrix[a, b])))
            i -= 1
            j -= 1
        elif layer == GAP_RIGHT:
            path.append((i - 1, -1, None))
            i -= 1
        else:
            path.append((-1, j - 1, None))
            j -= 1
        layer = following
    path.reverse()
    return path


def collapse_runs(sequence: str):
    """
    the ORDER of states, with the hour each run begins and how long it lasts

    WHY THE DIVERGENCE ALIGNMENT NEEDS THIS. Aligned hour by hour, two courses
    with the same order of states but different durations in them score as
    disagreeing from the first hour the durations part. Worked example, with
    an ordinal alphabet:

        A   3200112211000000125550
        B     12100000000011200000

    Those share a long stretch -- A's 11221100000012 against B's
    121000000000112 -- and hour-by-hour they are mostly mismatches, because A
    holds each state for a different number of hours than B. Collapsed they
    are IDENTICAL, both 121012, and what is left to explain is the part after
    the shared order runs out: A goes to 5550 and B to 0000.

    That is the question the section is asking, so the collapse belongs in the
    alignment rather than only in the clustering, where it already was. The
    cost is explicit and is NOT small: duration is discarded, so a patient who
    spent two hours in a state and one who spent twenty are the same here, and
    any reading of the divergence hour as "how long they stayed together"
    becomes wrong. It is the hour at which the first DIVERGING RUN begins.

    Returns (collapsed, starts, lengths): the collapsed string, the hour each
    run begins at, and the number of hours it covers. `starts` is what maps an
    alignment endpoint in collapsed coordinates back to a clock hour.
    """
    collapsed, starts, lengths = [], [], []
    for hour, symbol in enumerate(sequence):
        if collapsed and collapsed[-1] == symbol:
            lengths[-1] += 1
            continue
        collapsed.append(symbol)
        starts.append(hour)
        lengths.append(1)
    return "".join(collapsed), starts, lengths


def _to_hours(end, starts, total_hours):
    """an alignment endpoint in collapsed runs, as a clock hour"""
    if end >= len(starts):
        return int(total_hours)
    return int(starts[end])


def divergence_point(path, end_i, end_j, left_length, right_length,
                     min_tail=1, min_aligned=0):
    """
    where an anchored alignment stopped paying, and whether that means anything

    With the endpoint unrestricted the alignment extends from the origin for
    exactly as long as extending is worth points, so THE END CELL IS THE
    DIVERGENCE POINT -- there is nothing further to search for.

WHAT `diverged` IS FOR, AND WHY IT IS NOT A FILTER. The divergence hour is
    bounded above by the shorter record -- a 20-hour course cannot part company
    at hour 25 -- so it correlates with length no matter how pairs are
    selected, and an attempt to filter that away failed: requiring 12 further
    hours on record discarded 77% of pairs and left the correlation at +0.64.
    Exclusion is the wrong instrument. `diverged` is instead an EVENT INDICATOR
    and `left_hour` a TIME, so a pair whose record ran out while the two were
    still agreeing is a censored observation rather than a discarded one, and
    the length bound is carried by the censoring mechanism where it belongs.

    `min_tail` survives as a tightening option: at its default of 1 a pair is
    censored as soon as either course is exhausted, which is the honest reading.
    Raising it demands a margin of still-agreeing hours before an event counts.
    """
    scored = [s for _, _, s in path if s is not None]
    # Where the shared region BEGAN. Under anchored alignment this is always
    # (0, 0) and carries no information; under local alignment it is the
    # measurement that replaces the anchoring assumption.
    aligned_cols = [(a, b) for a, b, sc in path if a >= 0 and b >= 0 and sc is not None]
    start_i = int(aligned_cols[0][0]) if aligned_cols else int(end_i)
    start_j = int(aligned_cols[0][1]) if aligned_cols else int(end_j)
    left_remaining = int(left_length) - int(end_i)
    right_remaining = int(right_length) - int(end_j)
    matched = sum(
        1 for a, b, s in path if a >= 0 and b >= 0 and s is not None
    )
    identical = sum(1 for a, b, s in path if s is not None and s > 0)

    if matched < int(min_aligned):
        # LOCAL ALIGNMENT ALWAYS FINDS SOMETHING. Two unrelated courses share
        # two or three runs by chance, and scoring that as "agreed, then
        # diverged" would hand back a divergence hour for every pair in the
        # cohort. Below the floor the pair is reported as having no shared
        # course, which is a different statement from having parted.
        reason, diverged = "no_shared_course", False
    elif left_remaining <= 0 and right_remaining <= 0:
        reason, diverged = "agreed_throughout", False
    elif min(left_remaining, right_remaining) < max(int(min_tail), 1):
        reason, diverged = "ran_out", False
    elif matched == 0:
        # they never agreed at all; that is a divergence AT ONSET and must be
        # reported as hour zero, not dropped, or the hour distribution is
        # truncated away from exactly the pairs that parted earliest
        reason, diverged = "diverged_at_onset", True
    else:
        reason, diverged = "diverged", True

    return {
        "diverged": diverged,
        "reason": reason,
        "left_start": start_i,
        "right_start": start_j,
        "left_hour": int(end_i),
        "right_hour": int(end_j),
        "left_remaining": left_remaining,
        "right_remaining": right_remaining,
        "columns": len(path),
        "matched_columns": matched,
        "gap_columns": len(path) - matched,
        "positive_columns": identical,
        # NOT the alignment score: this sums the matched columns only and
        # charges no gap penalty, so two alignments with the same number of
        # matches tie here however much gap one of them bought to get them.
        # `align_many` adds the real score as `alignment_score`.
        "matched_score": float(sum(scored)) if scored else 0.0,
        "identity": identical / matched if matched else float("nan"),
    }


def column_counts(aligned, size):
    """aligned-column state pairs, for the next round of re-estimation"""
    counts = np.zeros((size, size), dtype=np.float64)
    for left_code, right_code, path in aligned:
        for a, b, score in path:
            if a >= 0 and b >= 0 and score is not None:
                i, j = int(left_code[a]), int(right_code[b])
                if i >= 0 and j >= 0:
                    counts[i, j] += 1.0
    return counts


def render(path, left_code, right_code, alphabet, limit=64):
    """the two aligned strings with a match line between them, for the log"""
    top, middle, bottom = [], [], []
    for a, b, score in path[:limit]:
        top.append(alphabet[int(left_code[a])] if a >= 0 else "-")
        bottom.append(alphabet[int(right_code[b])] if b >= 0 else "-")
        if a >= 0 and b >= 0:
            if left_code[a] == right_code[b]:
                middle.append("|")
            elif score is not None and score > 0:
                middle.append("+")
            else:
                middle.append(" ")
        else:
            middle.append(" ")
    return "".join(top), "".join(middle), "".join(bottom)


def align_many(left, right, alphabet, matrix, gap_open=4.0, gap_extend=1.0,
               budget_mb=256.0, logger=None, keep_paths=False,
               min_tail=1, collapse=False, local=False, min_aligned=0):
    """
    align a list of sequence pairs, chunked so the pointer arrays fit in memory

    `collapse` aligns the RUN-LENGTH COLLAPSED courses instead of the hourly
    ones and maps the endpoint back to a clock hour. See `collapse_runs` for
    the worked example of why: hour-by-hour, two courses with the same order
    of states but different durations in them disagree from the first hour the
    durations part, which is a statement about length of stay wearing a
    physiology label. With `collapse`, `left_hour` and `right_hour` are the
    hour at which the first DIVERGING RUN begins, and `left_run`/`right_run`
    carry the endpoint in collapsed coordinates. `min_tail` is then counted in
    runs, not hours.

    `local` drops the anchor at shock onset so the shared region may begin
    anywhere, and `min_aligned` is the floor below which a local match is not
    called a shared course at all. The two belong together: without the floor,
    local alignment reports a divergence hour for every pair, because any two
    courses share two or three runs by chance.

    The traceback needs three int8 pointers per cell per pair, so a chunk costs
    3 * n * (la+1) * (lb+1) bytes. At the 167-hour maximum in this cohort that
    is 173 MB for 2,048 pairs, which is why the chunk size is derived from the
    LONGEST PAIR IN THE CHUNK rather than fixed. Pairs are sorted by length
    first so that short pairs -- the median course is 20 hours -- travel in
    large chunks and only the long tail is cut small.
    """
    order = sorted(
        range(len(left)), key=lambda k: max(len(left[k]), len(right[k]))
    )
    results = [None] * len(left)
    paths = [] if keep_paths else None
    start, chunks = 0, 0
    while start < len(order):
        longest = max(
            len(left[order[start]]), len(right[order[start]])
        )
        per_pair = 3 * (longest + 1) * (longest + 1)
        size = max(1, int(budget_mb * 1_000_000 / max(per_pair, 1)))
        block = order[start:start + size]
        # the block may contain longer pairs than the one it was sized on, so
        # the budget is re-checked against the block's true maximum
        longest = max(max(len(left[k]), len(right[k])) for k in block)
        per_pair = 3 * (longest + 1) * (longest + 1)
        size = max(1, int(budget_mb * 1_000_000 / max(per_pair, 1)))
        block = order[start:start + size]

        if collapse:
            left_runs = [collapse_runs(left[k]) for k in block]
            right_runs = [collapse_runs(right[k]) for k in block]
            left_text = [r[0] for r in left_runs]
            right_text = [r[0] for r in right_runs]
        else:
            left_runs = right_runs = None
            left_text = [left[k] for k in block]
            right_text = [right[k] for k in block]
        left_codes, left_len, _ = encode(left_text, alphabet)
        right_codes, right_len, _ = encode(right_text, alphabet)
        best, best_i, best_j, pointers = align_batch(
            left_codes, left_len, right_codes, right_len, matrix,
            gap_open, gap_extend, want_path=True, local=local,
        )
        for position, k in enumerate(block):
            path = walk(
                pointers[position], matrix, left_codes[position],
                right_codes[position], best_i[position], best_j[position],
            )
            summary = divergence_point(
                path, best_i[position], best_j[position],
                left_len[position], right_len[position], min_tail,
                min_aligned,
            )
            summary["alignment_score"] = float(best[position])
            if collapse:
                # The endpoint is in runs; the rest of the pipeline reads
                # hours, so both are reported rather than one silently
                # standing in for the other.
                _, left_starts, _ = left_runs[position]
                _, right_starts, _ = right_runs[position]
                summary["left_run"] = summary["left_hour"]
                summary["right_run"] = summary["right_hour"]
                summary["left_start_run"] = summary["left_start"]
                summary["right_start_run"] = summary["right_start"]
                summary["left_hour"] = _to_hours(
                    summary["left_run"], left_starts, len(left[k])
                )
                summary["right_hour"] = _to_hours(
                    summary["right_run"], right_starts, len(right[k])
                )
                summary["left_start"] = _to_hours(
                    summary["left_start_run"], left_starts, len(left[k])
                )
                summary["right_start"] = _to_hours(
                    summary["right_start_run"], right_starts, len(right[k])
                )
                summary["shared_hours"] = (
                    summary["left_hour"] - summary["left_start"]
                )
                summary["left_remaining_hours"] = (
                    len(left[k]) - summary["left_hour"]
                )
                summary["right_remaining_hours"] = (
                    len(right[k]) - summary["right_hour"]
                )
                summary["collapsed"] = True
            results[k] = summary
            if keep_paths:
                paths.append(
                    (k, left_codes[position], right_codes[position], path)
                )
        start += len(block)
        chunks += 1
    if logger:
        logger.info(
            f"  {len(left):,} pairs aligned in {chunks} chunks"
            + (" on run-length collapsed courses" if collapse else "")
            + (", locally" if local else ", anchored at onset")
        )
    return (results, paths) if keep_paths else results


def align_scores(left, right, alphabet, matrix, gap_open=4.0, gap_extend=1.0,
                 budget_mb=256.0):
    """
    alignment scores only -- no traceback, no divergence point

    Clustering needs a distance between every pair and nothing else, and the
    traceback is what makes `align_many` expensive: the pointer arrays are
    three int8 per cell per pair, against three float32 per cell per ROW here.
    Dropping them lets a chunk hold tens of thousands of pairs instead of
    thousands, which is the difference between clustering being feasible and
    not.
    """
    order = sorted(
        range(len(left)), key=lambda k: max(len(left[k]), len(right[k]))
    )
    scores = np.zeros(len(left), dtype=np.float32)
    start = 0
    while start < len(order):
        longest = max(len(left[order[start]]), len(right[order[start]]))
        size = max(1, int(budget_mb * 1_000_000 / (3 * 4 * (longest + 1) * 2)))
        block = order[start:start + size]
        # NO COLLAPSE HERE, deliberately. `align_many` can align run-length
        # collapsed courses because its question is where two courses part;
        # this function feeds handbook DISCOVERY, where the clustering is over
        # dose-hours and collapsing would redefine what a handbook is rather
        # than make the divergence point honest.
        left_codes, left_len, _ = encode([left[k] for k in block], alphabet)
        right_codes, right_len, _ = encode([right[k] for k in block], alphabet)
        best, _i, _j, _p = align_batch(
            left_codes, left_len, right_codes, right_len, matrix,
            gap_open, gap_extend, want_path=False,
        )
        for position, k in enumerate(block):
            scores[k] = best[position]
        start += len(block)
    return scores


def distance_matrix(sequences, alphabet, matrix, gap_open=4.0, gap_extend=1.0,
                    match=3.0, logger=None):
    """
    condensed pairwise alignment distance, normalised to [0, 1]

    The raw score is not a distance: it grows with length, so two long similar
    courses outscore two short identical ones. Dividing by the best score the
    SHORTER course could possibly earn -- `match` per hour, which is what it
    would score against itself -- gives a fraction of achievable agreement, and
    one minus that is a distance bounded at 1 for courses that never agree.

    IT IS NOT A METRIC and nothing here assumes it is. Alignment distances
    violate the triangle inequality in general, so this feeds average-linkage
    clustering and medoid assignment, both of which only need an ordering, and
    never a method that embeds the distances in a space.
    """
    n = len(sequences)
    left, right, where = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            left.append(sequences[i])
            right.append(sequences[j])
            where.append((i, j))
    if logger:
        logger.info(f"  {len(left):,} pairwise alignments over {n:,} courses")
    scores = align_scores(
        left, right, alphabet, matrix, gap_open, gap_extend
    )
    condensed = np.zeros(len(left), dtype=np.float32)
    for position, (i, j) in enumerate(where):
        ceiling = match * min(len(sequences[i]), len(sequences[j]))
        condensed[position] = 1.0 - (
            scores[position] / ceiling if ceiling > 0 else 0.0
        )
    return np.clip(condensed, 0.0, 1.0)

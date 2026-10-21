"""
where the representation moves if the model reads one more token

The trajectory figures show where a patient went. This asks where they WOULD
go next, by taking their token prefix at a chosen hour, appending one
hypothetical event, and re-embedding. The frozen model is a deterministic
function of the token sequence, so appending a token and reading the hidden
state is exactly what the model would have done had that event been charted.

WHAT THIS IS NOT, and the reason it is easy to misread. It is not a treatment
effect and not a causal counterfactual. The model learned associations between
what is charted and what follows, so if large pressor doses preceded death in
the training data then appending a large pressor dose moves the representation
toward wherever dying patients sit -- whether or not the pressor caused
anything. Reading "the embedding moves toward the bad region when we increase
the pressor" as "increasing the pressor is bad" has the inference backwards:
the dose is charted because the patient is deteriorating.

What it does answer is narrower and still worth asking: which charted events
the representation is most sensitive to, and in which direction it moves. If
the model barely responds to a doubling of the vasopressor dose, that says
something about the representation.
"""

from __future__ import annotations

import re

import polars as pl


def _quantile(name: str) -> int | None:
    """the decile index baked into a token name, if it carries one"""
    match = re.search(r"_Q(\d+)$", name)
    return int(match.group(1)) if match else None


def _stem(name: str) -> str:
    """the token name with its quantile suffix removed"""
    return re.sub(r"_Q\d+$", "", name)


def shift_quantile(names: dict, prefix: list, stem: str, delta: int):
    """
    the token for the same variable, `delta` deciles up or down

    Finds the most recent token for `stem` in the patient own prefix and
    returns the id of the same variable at a shifted quantile. Relative to the
    patient rather than absolute, because "the clinician increased the dose" is
    a statement about a change from where they were, and an absolute target
    would be an increase for one patient and a decrease for another.

    Returns (token_id, from_q, to_q), or None when the patient has no token for
    that variable -- you cannot increase a pressor dose for somebody who is not
    on one, and inventing a Q5 from nothing is a different intervention.
    """
    by_name = {v: k for k, v in names.items()}
    last = None
    for token in reversed(prefix):
        name = names.get(int(token))
        if name and _stem(name) == stem and _quantile(name) is not None:
            last = name
            break
    if last is None:
        return None
    current = _quantile(last)
    target = max(0, min(9, current + delta))
    if target == current:
        return None
    # the vocabulary is sparse: not every decile of every variable was seen in
    # training. Walk outward to the nearest quantile that exists in the
    # intended direction, and give up rather than silently flipping direction.
    step = 1 if delta > 0 else -1
    probe = target
    while 0 <= probe <= 9:
        candidate = f"{stem}_Q{probe}"
        if candidate in by_name:
            return by_name[candidate], current, probe
        probe += step
    return None


def fixed_token(names: dict, prefix: list, token_name: str):
    """a named token, regardless of what the patient already has"""
    by_name = {v: k for k, v in names.items()}
    if token_name not in by_name:
        return None
    return by_name[token_name], None, None


#: the hypothetical next events, in tab10 order.
#:
#: Each is (key, label, resolver). A resolver returns None when the
#: intervention does not apply to that patient, which is a real answer and not
#: a failure: increasing a vasopressor dose is undefined for a patient on no
#: vasopressor, and the honest thing is to leave them out of that panel rather
#: than invent a starting dose.
INTERVENTIONS: tuple[tuple[str, str, object], ...] = (
    (
        "pressor_up",
        "Vasopressor dose increased (+2 deciles)",
        lambda names, prefix: shift_quantile(
            names, prefix, "MED-CTS//norepinephrine", +2
        ),
    ),
    (
        "pressor_down",
        "Vasopressor dose reduced (-2 deciles)",
        lambda names, prefix: shift_quantile(
            names, prefix, "MED-CTS//norepinephrine", -2
        ),
    ),
    (
        "map_down",
        "MAP falls (-2 deciles)",
        lambda names, prefix: shift_quantile(names, prefix, "VTL//map", -2),
    ),
    (
        "map_up",
        "MAP recovers (+2 deciles)",
        lambda names, prefix: shift_quantile(names, prefix, "VTL//map", +2),
    ),
    (
        "off_imv",
        "Taken off invasive ventilation",
        # there is no "not ventilated" token. Coming off the ventilator is
        # represented the way it is charted: the next device category.
        lambda names, prefix: fixed_token(names, prefix, "RESP//nasal_cannula"),
    ),
    (
        "start_inotrope",
        "Inotrope started (dobutamine)",
        lambda names, prefix: fixed_token(names, prefix, "MED-CTS//dobutamine_Q5"),
    ),
)


def append_token(frame: pl.DataFrame, names: dict, resolver) -> pl.DataFrame:
    """
    one extra token on the end of every patient prefix in `frame`

    `frame` is a cocoa *_for_inference.parquet: one row per patient with
    tokens_past, s_elapsed_past and last_valid. The appended event is given the
    same elapsed time as the last real token, so it reads as happening at the
    cut point rather than after some invented further gap -- the question is
    what the clinician does NOW.

    `last_valid` is incremented, because the extractor uses it to find the last
    token before padding and would otherwise read the state one token early and
    return the unmodified embedding. That failure is silent: the output is a
    valid embedding of the wrong sequence.

    Patients the intervention does not apply to are dropped.
    """
    rows = []
    tokens = frame["tokens_past"].to_list()
    elapsed = frame["s_elapsed_past"].to_list()
    last_valid = (
        frame["last_valid"].to_list()
        if "last_valid" in frame.columns
        else [len(t) for t in tokens]
    )
    subjects = frame["subject_id"].to_list()

    for subject, prefix, times, valid in zip(subjects, tokens, elapsed, last_valid):
        prefix = list(prefix or [])
        times = list(times or [])
        valid = int(valid)
        if valid < 1 or len(prefix) < valid or len(times) < valid:
            continue
        resolved = resolver(names, prefix[:valid])
        if resolved is None:
            continue
        token_id, from_q, to_q = resolved
        rows.append({
            "subject_id": subject,
            "tokens_past": prefix[:valid] + [int(token_id)],
            # repeat the last elapsed value AS-IS. Casting it to float mixed a
            # Float64 into an Int64 list and polars refused to build the
            # column; the elapsed times are integer seconds in the parquet and
            # the appended event happens at the cut point, so the existing
            # value is already the right one in the right type.
            "s_elapsed_past": times[:valid] + [times[valid - 1]],
            "last_valid": valid + 1,
            "from_quantile": from_q,
            "to_quantile": to_q,
        })
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)

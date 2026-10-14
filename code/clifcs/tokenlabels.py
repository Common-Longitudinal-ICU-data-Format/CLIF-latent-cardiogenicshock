#!/usr/bin/env python3

"""
the minimal ordered tokens: what a projection point can be labelled with

The consortium permits latent embeddings plus "a minimal set of ordered
tokens" -- enough to say which PaCMAP point corresponds to which MAP value,
without saying which patient or when. This module is that set.

For each allowlisted variable it takes the **last token before the anchor**,
which is the state the embedding was taken in, and exports its **decile rank**
(0-9) rather than the measured value. Nothing else: no timestamp, no sequence,
no count of how many times it was charted.

Why a rank is the right unit, and not a rounded value
-----------------------------------------------------
The tokenizer bins every numeric into deciles and bakes the bin into the token
name -- `VTL//map_Q3`. The model never saw 71.4 mmHg; it saw `map_Q3`. So the
rank is not a coarsening of the data the embedding was built from, it *is* the
data the embedding was built from, and a panel coloured by rank is showing the
model's own resolution rather than an approximation of it.

It also solves the cross-site comparability problem for free. The bin edges were
fitted once at training time and ship inside `tokenizer.yaml`, which every site
already must hold for its embeddings to mean anything. So rank 3 is the same
range of millimetres of mercury at every site, with nobody having to agree on
anything or exchange a quantile.

And the edges themselves are model metadata, not patient data, so the
coordinating site can render a legend reading "Q3 = 71-75 mmHg" from the
tokenizer alone -- which is the labelling the consortium asked for, obtained
without the sites sending any values at all.

Two things to know about the ranks
----------------------------------
**They are not always contiguous.** A decile split on a distribution piled up at
one value produces fewer than ten distinct bins: `spo2` has no Q3 or Q8, and
`gcs_total` has only Q0, Q1, Q2 and Q9, because most charted GCS totals are 15.
`levels()` reports the ranks that actually exist per variable, and a colour ramp
has to be built from that rather than from `range(10)`.

**Rank is ordinal, not linear.** The gap between Q0 and Q1 is a different number
of mmHg from the gap between Q8 and Q9. Ranks may be ordered, binned and used
for a sequential colour ramp; they may not be averaged. `bin_label` exists so a
figure says "71-75 mmHg" rather than "3.4".
"""

from __future__ import annotations

import functools
import re

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from clifcs.export.contract import TOKEN_LABEL_PREFIX

#: export column -> the tokenizer's qualified variable name
#:
#: The export name is the readable one and the tokenizer name is whatever cocoa
#: called it; `temperature` is `VTL//temp_c` and the labs are the *result*
#: stream (`LAB-RES`) rather than the order (`LAB-ORD`), which records that a
#: test was requested rather than what came back.
EXPORTED: dict[str, str] = {
    "map": "VTL//map",
    "sbp": "VTL//sbp",
    "heart_rate": "VTL//heart_rate",
    "spo2": "VTL//spo2",
    "respiratory_rate": "VTL//respiratory_rate",
    "temperature": "VTL//temp_c",
    "lactate": "LAB-RES//lactate",
    "creatinine": "LAB-RES//creatinine",
    "bicarbonate": "LAB-RES//bicarbonate",
    "gcs_total": "ASMT//gcs_total",
}

#: units, for the legend the coordinating site renders from the bin edges
UNITS: dict[str, str] = {
    "map": "mmHg",
    "sbp": "mmHg",
    "heart_rate": "bpm",
    "spo2": "%",
    "respiratory_rate": "/min",
    "temperature": "degC",
    "lactate": "mmol/L",
    "creatinine": "mg/dL",
    "bicarbonate": "mmol/L",
    "gcs_total": "",
}

#: `<qualified name>_Q<rank>`; the rank is the decile index, 0-9
_QUANTILE = re.compile(r"^(?P<name>.+)_Q(?P<rank>\d)$")


@functools.lru_cache(maxsize=4)
def _tokenizer(tokenizer_yaml: str) -> tuple[dict, dict]:
    cfg = OmegaConf.load(tokenizer_yaml)
    lookup = {str(name): int(i) for name, i in dict(cfg.lookup).items()}
    bins = {str(name): list(edges) for name, edges in dict(cfg.get("bins", {})).items()}
    return lookup, bins


@functools.lru_cache(maxsize=4)
def rank_table(tokenizer_yaml: str) -> dict[int, tuple[str, int]]:
    """
    token id -> (exported variable name, decile rank)

    Only ids belonging to an allowlisted variable appear, so a lookup against
    this is also the membership test: anything absent is a token we do not
    export a label for.
    """
    lookup, _ = _tokenizer(tokenizer_yaml)
    wanted = {qualified: exported for exported, qualified in EXPORTED.items()}
    out: dict[int, tuple[str, int]] = {}
    for name, token_id in lookup.items():
        match = _QUANTILE.match(name)
        if match is None:
            continue
        exported = wanted.get(match["name"])
        if exported is not None:
            out[token_id] = (exported, int(match["rank"]))
    return out


def levels(tokenizer_yaml: str) -> pl.DataFrame:
    """
    the ranks that exist per variable, with the range of values each covers

    This is the legend, and it is derived entirely from `tokenizer.yaml` -- it
    contains no patient data and is identical at every site, so it is written
    once by the coordinating site rather than sent by anyone.

    `lo`/`hi` are the bin's open bounds: the lowest rank has no lower bound and
    the highest none above, which is what a decile split means at the tails.
    """
    _, bins = _tokenizer(tokenizer_yaml)
    ranks = rank_table(tokenizer_yaml)

    by_variable: dict[str, list[int]] = {}
    for _, (variable, rank) in sorted(ranks.items()):
        by_variable.setdefault(variable, []).append(rank)

    rows = []
    for variable, present in sorted(by_variable.items()):
        edges = bins.get(EXPORTED[variable], [])
        for rank in sorted(set(present)):
            lo = float(edges[rank - 1]) if 0 < rank <= len(edges) else None
            hi = float(edges[rank]) if rank < len(edges) else None
            rows.append(
                {
                    "variable": variable,
                    "rank": rank,
                    "lo": lo,
                    "hi": hi,
                    "unit": UNITS.get(variable, ""),
                    "label": bin_label(variable, rank, lo, hi),
                }
            )
    return pl.DataFrame(rows)


def bin_label(variable: str, rank: int, lo: float | None, hi: float | None) -> str:
    """the printable range for one rank, e.g. `71-75 mmHg`"""
    unit = UNITS.get(variable, "")
    suffix = f" {unit}" if unit else ""

    def fmt(x: float) -> str:
        return f"{x:.0f}" if abs(x) >= 10 else f"{x:.1f}"

    if lo is None and hi is not None:
        return f"<{fmt(hi)}{suffix}"
    if hi is None and lo is not None:
        return f">={fmt(lo)}{suffix}"
    if lo is None and hi is None:
        return f"Q{rank}"
    return f"{fmt(lo)}-{fmt(hi)}{suffix}"


def last_pre_anchor_ranks(
    frame: pl.DataFrame, tokenizer_yaml: str, token_column: str = "tokens_past"
) -> pl.DataFrame:
    """
    one `tok_<variable>` column per allowlisted variable, holding the decile
    rank of that variable's last token before the anchor

    `frame` is what `clifcs.tokens.load_pre_anchor_tokens` returns: one row per
    patient with the token ids the model read, in order. Null where the variable
    was never charted before the anchor -- which is itself informative and is
    reported as a missingness rate rather than imputed.

    Implemented as a reverse scan per row rather than vectorised: the sequences
    are ragged (median 783 tokens, some tens of thousands) and the work is one
    dictionary lookup per token until every variable is found, which terminates
    early on most patients.
    """
    ranks = rank_table(tokenizer_yaml)
    variables = sorted(EXPORTED)
    target = len(variables)
    index = {v: i for i, v in enumerate(variables)}

    out = np.full((frame.height, target), -1, dtype=np.int8)
    for row, tokens in enumerate(frame[token_column].to_list()):
        if tokens is None or not len(tokens):
            continue
        found = 0
        seen = out[row]
        # walk backwards: the last token for a variable is the state the
        # embedding was taken in, and anything earlier is superseded
        for token_id in reversed(tokens):
            hit = ranks.get(int(token_id))
            if hit is None:
                continue
            slot = index[hit[0]]
            if seen[slot] < 0:
                seen[slot] = hit[1]
                found += 1
                if found == target:
                    break

    # built with -1 as the sentinel and nulled afterwards: a numpy array cannot
    # hold None without becoming dtype=object, which polars will not cast
    columns = [f"{TOKEN_LABEL_PREFIX}{v}" for v in variables]
    return (
        frame.select([c for c in frame.columns if c != token_column])
        .with_columns(
            [
                pl.Series(name, out[:, index[variable]], dtype=pl.Int8)
                for name, variable in zip(columns, variables)
            ]
        )
        .with_columns(
            [
                pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
                for c in columns
            ]
        )
    )


def coverage(frame: pl.DataFrame) -> pl.DataFrame:
    """
    per-variable share of patients whose last pre-anchor value was charted

    A site missing a variable entirely still produces every figure, coloured by
    nulls, and the figure looks fine. This is the table that catches it -- the
    same failure as the urine-output story in the README, where a specified
    criterion silently contributed nothing because the table was absent.
    """
    columns = [c for c in frame.columns if c.startswith(TOKEN_LABEL_PREFIX)]
    return pl.DataFrame(
        {
            "variable": [c.removeprefix(TOKEN_LABEL_PREFIX) for c in columns],
            "n_charted": [int(frame[c].is_not_null().sum()) for c in columns],
            "n_total": [frame.height] * len(columns),
            "share_charted": [
                float(frame[c].is_not_null().mean() or 0.0) for c in columns
            ],
        }
    ).sort("share_charted")

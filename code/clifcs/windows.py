#!/usr/bin/env python3

"""
per-window token composition, and the site-representativeness filter

The trajectory grid embeds a *cumulative prefix* at each timepoint: the +24h
embedding is the whole record up to +24h. That is right for an embedding and
wrong for clustering, because cumulative counts are monotone and a clustering
of them mostly recovers how long the record is. So the features here are
counted WITHIN each window instead -- window t holds the tokens with elapsed
time in (t - step, t].

Everything is derived from `tokens_times.parquet` and the onset table, so this
is token space: no model, no embedding, no derived physiology. The decile bins
are part of the vocabulary (`VTL//map_Q2` and `VTL//map_Q8` are different
tokens), so a composition over the vocabulary carries value information and not
only charting frequency.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def window_token_counts(
    paths, offsets, step_hours: float, keep_ids, logger
) -> pl.DataFrame:
    """
    tokens per (patient, window, token id), counted within the window

    `offsets` are the window END points: window t covers (t - step, t], so the
    t=0 window is the six hours leading up to shock onset and t=+48 is the six
    hours ending two days after it.

    Streamed and horizon-filtered per site for the same reason as
    `last_token_table` in 09 -- Rush's full stream explodes to 34M rows and
    everything outside the window range is irrelevant.
    """
    onset = (
        pl.read_parquet(paths.cohort_home / "cs_onset.parquet")
        .select(
            subject_id=pl.col("hospitalization_id").cast(pl.String),
            onset=pl.col("cs_onset_dttm").cast(pl.Datetime("us")),
        )
        .unique(subset="subject_id")
        .lazy()
    )
    low = min(offsets) - step_hours
    high = max(offsets)

    parts = []
    for site in paths.sites:
        counts = (
            pl.scan_parquet(paths.processed(site) / "tokens_times.parquet")
            .join(onset, on="subject_id", how="inner")
            .select("subject_id", "onset", "tokens", "times")
            .explode(["tokens", "times"])
            .with_columns(
                elapsed=(pl.col("times") - pl.col("onset")).dt.total_seconds() / 3600.0
            )
            .filter((pl.col("elapsed") > low) & (pl.col("elapsed") <= high))
            # ceil onto the window end: elapsed in (-6, 0] -> 0, (0, 6] -> 6
            .with_columns(
                window=((pl.col("elapsed") / step_hours).ceil() * step_hours)
            )
            .group_by("subject_id", "window", "tokens")
            .len()
            .rename({"subject_id": "hospitalization_id", "tokens": "token_id",
                     "len": "n"})
            .with_columns(site=pl.lit(site))
            .collect(engine="streaming")
        )
        if keep_ids is not None:
            counts = counts.filter(pl.col("hospitalization_id").is_in(keep_ids))
        logger.info(
            f"[{site}] {counts.height:,} (patient, window, token) cells, "
            f"{counts['hospitalization_id'].n_unique():,} patients"
        )
        parts.append(counts)
    return pl.concat(parts)


def site_representative_tokens(
    counts: pl.DataFrame, sites, logger, max_ratio: float = 2.0,
    min_prevalence: float = 0.01,
) -> list[int]:
    """
    the tokens every site uses, and uses at a comparable rate

    Token space carries no site correction. CORAL and LEACE act on embeddings;
    nothing acts on counts, and the sites chart very differently -- MED-INT is
    1.29% of tokens at UCMC, 1.74% at NU and 0.31% at Rush, and UNK runs
    7.8 / 8.8 / 14.8%. Clustering raw composition would substantially recover
    which hospital a patient was in.

    So a token enters the feature space only if

      * it appears at EVERY site -- a token one site never emits is a site
        indicator, however informative it looks, and
      * it is reasonably common, present in at least `min_prevalence` of that
        site's patient-windows, since a rare token's prevalence ratio is mostly
        sampling noise, and
      * its prevalence varies across sites by no more than `max_ratio`.

    Prevalence is the share of PATIENT-WINDOWS containing the token, not the
    share of tokens. A token charted once per window everywhere, at a site that
    charts twice as much overall, has the same prevalence and a halved count
    share -- prevalence is the quantity that asks "do these sites record this
    thing about the same patients", which is what the filter is for.
    """
    cells = (
        counts.group_by("site")
        .agg(total=pl.struct("hospitalization_id", "window").n_unique())
    )
    denominator = dict(zip(cells["site"].to_list(), cells["total"].to_list()))

    prevalence = (
        counts.group_by("site", "token_id")
        .agg(present=pl.struct("hospitalization_id", "window").n_unique())
        .with_columns(
            prevalence=pl.col("present")
            / pl.col("site").replace_strict(denominator, return_dtype=pl.Float64)
        )
    )
    wide = prevalence.pivot(on="site", index="token_id", values="prevalence")
    present = [s for s in sites if s in wide.columns]

    # a null here means the token never appeared at that site at all
    complete = wide.filter(
        pl.all_horizontal([pl.col(s).is_not_null() for s in present])
    )
    common = complete.filter(
        pl.min_horizontal([pl.col(s) for s in present]) >= min_prevalence
    )
    balanced = common.filter(
        pl.max_horizontal([pl.col(s) for s in present])
        / pl.min_horizontal([pl.col(s) for s in present])
        <= max_ratio
    )
    logger.info(
        f"site filter: {wide.height:,} tokens seen -> "
        f"{complete.height:,} at every site -> "
        f"{common.height:,} with prevalence >= {min_prevalence:.0%} everywhere -> "
        f"{balanced.height:,} within a {max_ratio:g}x prevalence ratio"
    )
    return sorted(balanced["token_id"].to_list())


def composition_matrix(counts: pl.DataFrame, token_ids, offsets, logger):
    """
    (patient, window) composition over the kept tokens, plus the window's volume

    Returns (patients, matrix, volume) where `matrix` is
    n_patients x n_windows x n_tokens holding each window's token proportions
    and `volume` is n_patients x n_windows holding log1p of the kept-token count.

    Proportions rather than counts: that divides out charting volume, which is
    the part of the signal most contaminated by site. Volume is kept as its own
    feature rather than discarded, because how much gets recorded about a
    patient per six hours is clinically real -- it rises with acuity -- it just
    should not be allowed to dominate the composition.

    A patient-window with no kept tokens gets a zero composition and zero
    volume. That is a real observation -- nothing comparable across sites was
    recorded in those six hours -- not a gap to impute.
    """
    index = {int(t): i for i, t in enumerate(token_ids)}
    window_index = {float(o): i for i, o in enumerate(offsets)}

    kept = counts.filter(pl.col("token_id").is_in(list(token_ids)))
    patients = sorted(kept["hospitalization_id"].unique().to_list())
    row = {p: i for i, p in enumerate(patients)}

    matrix = np.zeros((len(patients), len(offsets), len(token_ids)), dtype=np.float32)
    for pid, window, token, n in zip(
        kept["hospitalization_id"].to_list(),
        kept["window"].to_list(),
        kept["token_id"].to_list(),
        kept["n"].to_list(),
    ):
        w = window_index.get(float(window))
        if w is None:
            continue
        matrix[row[pid], w, index[int(token)]] += n

    volume = matrix.sum(axis=2)
    matrix = np.where(
        volume[:, :, None] > 0, matrix / np.maximum(volume[:, :, None], 1.0), 0.0
    )
    logger.info(
        f"composition: {len(patients):,} patients x {len(offsets)} windows x "
        f"{len(token_ids):,} tokens; "
        f"{(volume == 0).mean():.1%} of patient-windows have no kept tokens"
    )
    return patients, matrix.astype(np.float32), np.log1p(volume).astype(np.float32)

#!/usr/bin/env python3

"""
20_wean_in_embedding.py -- where the weans sit, and whether failure clusters

19 found that nothing measurable at the bedside separates a wean that holds
from one that does not. This asks whether the EMBEDDING separates them: are
there regions of the projection where weans tend to fail and regions where they
tend to hold?

    python code/20_wean_in_embedding.py --subset cardiogenic

ONLY WEANS INSIDE THE EMBEDDING WINDOW CAN BE PLACED. The grid runs -24h to
+48h in thirteen points; the wean sequences run the full seven days. About two
thirds of weans fall inside the window and the rest have no coordinate. Those
are dropped, and because weans get later as the week goes on, the ones dropped
are systematically the LATE weans -- so this figure describes early weaning
specifically, not weaning.

CLUSTERING IS TESTED, NOT EYEBALLED. A hexagon map of a rate always looks
patchy: with a few hundred events spread over a hundred cells, the per-cell
rates vary by sampling alone. Two tests are run against that:

  * a PERMUTATION test, shuffling the hold/restart labels among the weans that
    were placed and recomputing the spread of per-hexagon rates. This asks
    whether the map is patchier than chance given the same event positions.
  * MORAN'S I on the per-hexagon rate, which asks the sharper question of
    whether high-restart cells sit NEXT TO other high-restart cells. A map can
    be over-dispersed without being spatially organised, and only the second
    would mean the embedding knows something.

WHAT A POSITIVE RESULT WOULD AND WOULD NOT MEAN. Position in this projection
already tracks severity and, at R2 0.45, log stream length. So a region where
weans fail is very likely a region where patients are sicker, and finding one
is not evidence that the embedding knows something specific about weaning.

AND THE ANSWER IS THAT IT DOES NOT. The regions are real -- patchiness six
times the null and Moran's I around +0.54, both p < 0.0001 -- but at the
hexagon level the restart rate correlates -0.65 with hours from onset, -0.55
with log stream length and +0.32 to +0.61 with mortality. The high-restart
regions are the early, sicker, more heavily charted part of the map, which are
exactly the axes this projection was already known to encode.

DO NOT READ THOSE CORRELATIONS AGAINST 19'S EFFECT SIZES. They are ECOLOGICAL
correlations over ~65 hexagons, and 19's standardised differences are
individual-level: MAP separated held from restarted at SMD +0.002 per patient
while correlating +0.25 with the restart rate per hexagon. Those are not in
conflict. Aggregating averages away the within-cell noise and leaves only the
between-region component, which inflates the correlation -- an r of -0.65
across cells is entirely compatible with no association across patients. The
aggregate numbers say what the MAP of the projection is organised by; they say
nothing about what predicts an individual wean, and reading them as if they did
is the ecological fallacy.
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths


@functools.lru_cache(maxsize=1)
def _wean_module():
    """19, loaded once: the state names and family rules live there"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "wean", pathlib.Path(__file__).resolve().parent / "19_wean_durability.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spell(word: str):
    return _wean_module().spell(word)

#: CMS small-cell floor, applied per hexagon before any rate is published
MIN_CELL = 11


def hexbin_rates(x, y, outcome, gridsize, logger):
    """
    per-hexagon event count and restart rate, on a shared tiling

    matplotlib's hexbin is used for the tiling rather than a hand-rolled grid
    so that these cells line up with every other hexagon figure in the report.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    counts = axes.hexbin(x, y, gridsize=gridsize, mincnt=1)
    positions = counts.get_offsets()
    totals = counts.get_array()
    plt.close(figure)

    figure, axes = plt.subplots()
    hits = axes.hexbin(
        x, y, C=outcome.astype(float), gridsize=gridsize, reduce_C_function=np.sum,
        mincnt=1,
    )
    restarts = hits.get_array()
    plt.close(figure)

    keep = np.asarray(totals) >= MIN_CELL
    logger.info(
        f"  {int(keep.sum())} hexagons carry at least {MIN_CELL} weans "
        f"(of {len(totals)} non-empty, {int(np.asarray(totals).sum()):,} weans)"
    )
    return (
        np.asarray(positions)[keep],
        np.asarray(totals)[keep],
        np.asarray(restarts)[keep],
    )


def permutation_spread(totals, restarts, seed, rounds=2000):
    """
    is the map patchier than shuffling the labels would make it

    The statistic is the count-weighted variance of the per-hexagon rate.
    Weighted, because an unweighted variance is dominated by the thinnest
    cells, which are exactly the ones whose rates are least certain -- the
    test would then mostly measure how many small cells the tiling produced.
    """
    rate = restarts / np.maximum(totals, 1)
    overall = restarts.sum() / max(totals.sum(), 1)
    observed = float(np.average((rate - overall) ** 2, weights=totals))

    rng = np.random.default_rng(seed)
    n = int(totals.sum())
    successes = int(restarts.sum())
    null = np.empty(rounds)
    for draw in range(rounds):
        # redraw each hexagon's restart count from the pooled rate, keeping
        # every cell's event count fixed: the positions of the weans are not
        # in question, only which of them failed
        shuffled = rng.hypergeometric(
            successes, n - successes, totals.astype(int)
        ) if successes < n else totals
        null[draw] = np.average(
            (shuffled / np.maximum(totals, 1) - overall) ** 2, weights=totals
        )
    p = float((null >= observed).mean())
    return observed, float(null.mean()), p


def morans_i(positions, totals, restarts, seed, rounds=2000):
    """
    do high-restart hexagons sit next to other high-restart hexagons

    Neighbours are the cells within 1.5 hexagon pitches, the pitch taken as the
    smallest non-zero distance between centres. Over-dispersion without spatial
    organisation is what sampling noise looks like; organisation is what an
    embedding that knows something looks like.
    """
    rate = restarts / np.maximum(totals, 1)
    values = rate - np.average(rate, weights=totals)
    gaps = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=2
    )
    off = gaps[gaps > 0]
    if off.size == 0:
        return float("nan"), float("nan")
    pitch = off.min()
    weights = ((gaps > 0) & (gaps <= pitch * 1.5)).astype(float)
    if weights.sum() == 0:
        return float("nan"), float("nan")

    def statistic(v):
        numerator = float(v @ weights @ v)
        return (len(v) / weights.sum()) * numerator / float((v ** 2).sum())

    observed = statistic(values)
    rng = np.random.default_rng(seed)
    null = np.empty(rounds)
    for draw in range(rounds):
        null[draw] = statistic(rng.permutation(values))
    return float(observed), float((null >= observed).mean())


def plot_wean_map(frame, positions, totals, restarts, overall, paths, subset,
                  logger):
    """two panels: where the weans are, and where they fail"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from clifcs.figures import save_figure

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    axes[0].scatter(
        frame["pacmap_1"], frame["pacmap_2"], s=4, c="#d9d9d9",
        linewidths=0, rasterized=True,
    )
    weans = frame.filter(pl.col("is_wean"))
    axes[0].scatter(
        weans["pacmap_1"], weans["pacmap_2"], s=6, c="#2b5d8a",
        linewidths=0, alpha=0.5, rasterized=True,
    )
    axes[0].set_title(
        f"Where the weans happen\n{weans.height:,} weans against "
        f"{frame.height:,} grid points"
    )

    rate = restarts / np.maximum(totals, 1)
    # Diverging about the POOLED rate, not about 0.5. Half the weans restart,
    # so a palette centred on 0.5 would call the average cell "high" in one
    # cohort and "low" in another purely from the base rate.
    span = float(np.abs(rate - overall).max()) or 0.1
    scatter = axes[1].scatter(
        positions[:, 0], positions[:, 1], c=rate, s=np.sqrt(totals) * 14,
        cmap="RdBu_r", vmin=overall - span, vmax=overall + span,
        linewidths=0.4, edgecolors="white",
    )
    bar = figure.colorbar(scatter, ax=axes[1])
    bar.set_label(f"restart within 24h (pooled {overall:.0%})")
    axes[1].set_title(
        f"Where they fail\nhexagons with at least {MIN_CELL} weans, "
        "area proportional to count"
    )
    for panel in axes:
        panel.set_xlabel("PaCMAP 1")
        panel.set_ylabel("PaCMAP 2")
    figure.suptitle(
        f"{subset} shock: weaning in the corrected embedding, 0-48h only",
        fontsize=12,
    )
    figure.tight_layout()
    out = (
        paths.output_home / "figures" / paths.model_key
        / f"wean_embedding_{subset}.png"
    )
    save_figure(figure, out, manuscript=True)
    logger.info(f"wrote {out}")


def plot_per_word(grid, weans, paths, subset, gridsize, logger,
                  min_occurrences=60, top=12, column="lead_in"):
    """
    one figure per trial type: where it happens, split by whether it held

    KEYED ON THE LEAD-IN, not on the weaning word. "30_" and "60_" are close to
    uninformative on their own -- both mostly say "came off pressors", which is
    what every weaning word says, so a map per weaning word is nearly a map of
    weaning in general drawn several times over. The window BEFORE carries the
    clinical situation, so each panel is one (preceding word, weaning word)
    pair.

    SEPARATE FIGURES, not subplots, because the words differ by an order of
    magnitude in how often they occur. A shared panel invites reading the
    sparse words against the dense ones as though the density meant the same
    thing, and a 60-occurrence word tiled on the same scale as a 4,000-
    occurrence one is mostly empty space.

    EACH PANEL CARRIES THE WHOLE CLOUD IN GREY. Without it a reader cannot tell
    a region where this word never happens from a region where no patient ever
    goes, and those are entirely different statements.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from clifcs.figures import save_figure

    out_dir = (
        paths.output_home / "figures" / paths.model_key
        / f"wean_words_{subset}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if column not in weans.columns:
        logger.warning(f"no {column} column; per-word maps skipped")
        return []
    counts = weans.filter(pl.col(column).is_not_null())[column].value_counts(
        sort=True
    )
    written = []
    for word, n in zip(counts[column].to_list(), counts["count"].to_list()):
        if n < min_occurrences or len(written) >= top:
            continue
        part = weans.filter(pl.col(column) == word)
        held = part.filter(~pl.col("restarted_within_24h"))
        back = part.filter(pl.col("restarted_within_24h"))

        figure, axes = plt.subplots(figsize=(7.2, 6.2))
        axes.scatter(
            grid["pacmap_1"], grid["pacmap_2"], s=4, c="#e2e2e2",
            linewidths=0, rasterized=True,
        )
        axes.scatter(
            held["pacmap_1"], held["pacmap_2"], s=26, c="#2166ac",
            linewidths=0.3, edgecolors="white", alpha=0.85,
            label=f"held ({held.height:,})",
        )
        axes.scatter(
            back["pacmap_1"], back["pacmap_2"], s=26, c="#b2182b",
            linewidths=0.3, edgecolors="white", alpha=0.85,
            label=f"restarted within 24h ({back.height:,})",
        )
        lead = (
            part["preceding_word"][0]
            if "preceding_word" in part.columns else None
        )
        axes.set_title(
            "\n".join([
                word,
                "from: " + (
                    spell(lead) if lead else "[start of record]"
                ),
                f"{part.height:,} trials placed, "
                f"{part['restarted_within_24h'].mean():.0%} restarted "
                f"(all trials {weans['restarted_within_24h'].mean():.0%})",
            ]),
            fontsize=9,
        )
        axes.set_xlabel("PaCMAP 1")
        axes.set_ylabel("PaCMAP 2")
        axes.legend(loc="best", fontsize=8, frameon=False)
        figure.tight_layout()
        safe = word.replace("_", "W").replace(" ", "-")
        out = out_dir / f"wean_word_{safe}.png"
        save_figure(figure, out, manuscript=True)
        written.append((word, out))
        logger.info(
            f"    {word:<10} {part.height:>6,} placed, "
            f"{part['restarted_within_24h'].mean():>5.0%} restarted -> "
            f"{out.name}"
        )
    return written


def word_per_window(fine, logger, step_hours=6.0):
    """
    which BPE word occupies each 6-hour window, one row per (patient, window)

    A window may contain several words, so one has to be chosen to stand for
    it. The WINDOW-CLOSING word is used -- the one ending in the marker --
    because it is the only one guaranteed to exist exactly once per recorded
    window, and because it is the word whose span ends where the window does.
    Picking the first word instead would describe the window by something that
    may have been superseded four times before it closed.
    """
    rows_patient, rows_offset, rows_word = [], [], []
    for patient, sentence in zip(
        fine["hospitalization_id"].to_list(),
        fine["fine_bpe_sentence"].to_list(),
    ):
        if not sentence:
            continue
        window = 0
        for word in sentence.split():
            if not word.endswith("_"):
                continue
            rows_patient.append(patient)
            rows_offset.append(window * step_hours)
            rows_word.append(word)
            window += 1
    frame = pl.DataFrame({
        "hospitalization_id": rows_patient,
        "offset_hours": rows_offset,
        "bpe_word": rows_word,
    })
    logger.info(
        f"  {frame.height:,} (patient, window) slots carry a closing word, "
        f"{frame['bpe_word'].n_unique():,} distinct"
    )
    return frame


#: Words that describe the ABSENCE of vasoactive support. They are the
#: background of this cohort rather than a manoeuvre, and under plurality they
#: win most of the map by headcount alone -- "0_" takes 32% of all slots and
#: 118 hexagons while "313_" takes 3% of slots and none at all. The
#: background-excluded variants drop them from the colour competition so the
#: question becomes which SUPPORTED state localises where.
BACKGROUND_WORDS = ("0_", "7_", "_")


def plot_word_hexmaps(grid, fine, paths, subset, logger, top=14,
                      gridsize=34, min_count=10):
    """
    the projection tiled by which BPE word owns each hexagon

    FOUR MAPS, because one weighting cannot answer the question. Plurality
    weights by raw count, so a cohort that is mostly off pressors produces a
    map that is mostly "off pressors": in the septic plurality map "0_" held
    118 hexagons and 32% of points while four words at 2-3% of points each won
    ZERO hexagons between them. That is the statistic behaving correctly and
    the figure being useless.

      plurality               the commonest word in the hexagon
      enrichment              the word most concentrated here relative to its
                              OWN total, so size cancels and a rare word shows
                              wherever it localises
      ...-supported           the same two with the off-pressor background
                              words dropped from the competition entirely, so
                              each hexagon is coloured by which SUPPORTED state
                              owns it

    The supported variants are the ones to read for localisation. The full ones
    are kept because dropping two thirds of the points changes what a hexagon's
    count means, and a reader should be able to see the denominator they were
    computed on.

    THE PALETTE IS husl, NOT THE DEEPER hls. States and words take the evenly
    spaced husl hues; the darker `cluster_colors` palette is reserved for
    trajectories and sentence-level clusterings, so a reader can tell at a
    glance whether a colour denotes a state or a group of patients. An earlier
    version of this figure used `cluster_colors` and broke that convention.
    """
    import importlib.util

    from clifcs.palette import husl_colors

    spec = importlib.util.spec_from_file_location(
        "traj", pathlib.Path(__file__).resolve().parent / "09_trajectories.py"
    )
    traj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(traj)

    per_window = word_per_window(fine, logger)
    joined = grid.join(
        per_window, on=["hospitalization_id", "offset_hours"], how="inner"
    )
    if joined.is_empty():
        logger.warning("no grid point carried a BPE word; hexmaps skipped")
        return
    logger.info(
        f"  {joined.height:,} grid points matched to a word "
        f"({joined.height / max(grid.height, 1):.0%} of the grid)"
    )

    counts = joined["bpe_word"].value_counts(sort=True)
    chosen = counts["bpe_word"].to_list()[:top]
    pooled = f"other word ({counts.height - len(chosen)} of them)"
    joined = joined.with_columns(
        word_level=pl.when(pl.col("bpe_word").is_in(chosen))
        .then(pl.col("bpe_word"))
        .otherwise(pl.lit(pooled))
    )
    # THE POOLED TAIL DOES NOT COMPETE. Leaving "other word" in the level set
    # let an aggregate of 72 words run against individual words under
    # plurality, which it won almost everywhere by virtue of being pooled --
    # 215 hexagons and 40% of points, while seven named words won none at all.
    # Comparing an aggregate against its own constituents is not a comparison.
    # Its size is reported in the subtitle instead.
    order = list(chosen)
    colors = husl_colors(chosen)
    colors[pooled] = "#b9b9b9"
    for word in chosen:
        share = counts.filter(pl.col("bpe_word") == word)["count"][0]
        logger.info(
            f"    {word:<10} {share:>7,} slots "
            f"({share / joined.height:>5.1%})   {spell(word)[:52]}"
        )

    supported = [w for w in order if w not in BACKGROUND_WORDS]
    held_out = [w for w in chosen if w in BACKGROUND_WORDS]
    logger.info(
        f"    background words dropped from the supported variants: "
        f"{', '.join(held_out) or 'none'}"
    )
    for mode in ("plurality", "enrichment"):
        for scope, levels in (("", order), ("_supported", supported)):
            pooled_share = (
                joined.filter(pl.col("word_level") == pooled).height
                / max(joined.height, 1)
            )
            note = (
                f"window-closing words; the {len(chosen)} commonest shown, "
                f"the remaining {counts.height - len(chosen)} words "
                f"({pooled_share:.0%} of slots) excluded from the competition "
                "rather than pooled into a level that would win it"
            )
            if scope:
                note += (
                    "  |  off-pressor words ("
                    + ", ".join(held_out)
                    + ") dropped, so each hexagon is coloured by which "
                    "SUPPORTED state owns it"
                )
            traj.plot_state_hexbin(
                joined, "pacmap_1", "pacmap_2", "word_level", colors, levels,
                paths.output_home / "figures" / paths.model_key
                / f"wean_word_hex_{mode}{scope}_{subset}.png",
                f"{subset} shock: BPE word by hexagon, {mode}"
                + (", supported states only" if scope else ""),
                note, logger, mode=mode, gridsize=gridsize,
                min_count=min_count, flow=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic"))
    parser.add_argument("--gridsize", type=int, default=22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "20_wean_in_embedding", paths.log_home / "20_wean_in_embedding.log"
    )
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    wean_file = (
        paths.patient_level_home / f"wean_occurrences_{args.subset}.parquet"
    )
    if not wean_file.is_file():
        logger.error(f"no {wean_file.name}; run 19 first")
        return
    weans = pl.read_parquet(wean_file)

    candidates = sorted(paths.patient_level_home.glob(
        f"trajectory_projections_{args.subset}_corrected-*.parquet"
    ))
    if not candidates:
        logger.error(f"no corrected projection for {args.subset}")
        return
    logger.info(f"reading {candidates[0].name}")
    grid = pl.read_parquet(candidates[0]).select(
        "hospitalization_id", "offset_hours", "pacmap_1", "pacmap_2", "site",
        *[c for c in ("last_valid", "in_hospital_mortality")
          if c in pl.read_parquet_schema(candidates[0])],
    )
    if "last_valid" in grid.columns:
        grid = grid.with_columns(
            log_stream=(pl.col("last_valid").fill_null(0) + 1).log()
        )

    placed = weans.join(
        grid, left_on=["hospitalization_id", "hours"],
        right_on=["hospitalization_id", "offset_hours"], how="inner",
    )
    logger.info(
        f"{placed.height:,} of {weans.height:,} weans fall inside the "
        f"embedding window and have a coordinate "
        f"({placed.height / max(weans.height, 1):.0%})"
    )
    # The dropped ones are not a random sample: weans get later as the week
    # goes on, so what is lost is specifically the late weans.
    missed = weans.filter(~pl.col("hours").is_in(
        grid["offset_hours"].unique().to_list()
    ))
    if not missed.is_empty():
        logger.info(
            f"  dropped weans sit at a median {missed['hours'].median():.0f}h "
            f"against {placed['hours'].median():.0f}h for those kept, and "
            f"restart at {missed['restarted_within_24h'].mean():.1%} against "
            f"{placed['restarted_within_24h'].mean():.1%}"
        )
    if placed.height < 200:
        logger.error("too few placed weans to map")
        return

    outcome = placed["restarted_within_24h"].to_numpy()
    overall = float(outcome.mean())
    positions, totals, restarts = hexbin_rates(
        placed["pacmap_1"].to_numpy(), placed["pacmap_2"].to_numpy(),
        outcome, args.gridsize, logger,
    )
    if len(totals) < 5:
        logger.error("too few hexagons above the small-cell floor")
        return

    rate = restarts / np.maximum(totals, 1)
    logger.info(
        f"  pooled restart {overall:.1%}; per-hexagon range "
        f"{rate.min():.1%} to {rate.max():.1%}"
    )
    spread, expected, p_spread = permutation_spread(totals, restarts, args.seed)
    logger.info(
        f"  patchiness (count-weighted variance of the rate): observed "
        f"{spread:.5f}, expected under the null {expected:.5f}, p={p_spread:.4f}"
    )
    moran, p_moran = morans_i(positions, totals, restarts, args.seed)
    logger.info(
        f"  Moran's I on the per-hexagon rate: {moran:+.4f}, p={p_moran:.4f} "
        "(do high-restart cells neighbour each other)"
    )

    # The confound, measured rather than asserted: if the high-restart regions
    # are simply the sicker or more heavily documented ones, these say so.
    # EVERY COVARIATE IS BINNED ON THE SAME POINT SET as the rate, because
    # hexbin returns only its non-empty cells in its own order. Masking each
    # covariate separately gave each one a different tiling, and pairing them
    # by array index then compared unrelated cells -- the correlations it
    # produced were meaningless. Rows missing any covariate are dropped once,
    # up front, so one tiling serves all of them.
    #
    # LOG STREAM LENGTH IS THE COVARIATE THAT MATTERS and was missing from the
    # first version. Position in this projection explains R2 0.45 of it, so a
    # region of the map is partly a region of documentation density; testing
    # MAP and dose while omitting it would have answered the easy question and
    # skipped the hard one. Mortality is here for the same reason: it stands
    # in for severity in general, where MAP at one instant does not.
    covariates = [
        (column, label) for column, label in (
            ("hours", "hours from onset"),
            ("map", "MAP at the wean"),
            ("nee", "norepinephrine equivalents"),
            ("log_stream", "log event-stream length"),
            ("in_hospital_mortality", "died in hospital"),
        ) if column in placed.columns
    ]
    context = []
    if covariates:
        complete = placed.drop_nulls([c for c, _ in covariates])
        logger.info(
            f"  covariate check on {complete.height:,} weans with every "
            "covariate present"
        )
        if complete.height >= 200:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            x = complete["pacmap_1"].to_numpy()
            y = complete["pacmap_2"].to_numpy()
            outcome_here = complete["restarted_within_24h"].to_numpy().astype(float)
            figure, axes = plt.subplots()
            counts_here = axes.hexbin(x, y, gridsize=args.gridsize, mincnt=1)
            totals_here = np.asarray(counts_here.get_array())
            plt.close(figure)
            figure, axes = plt.subplots()
            rate_here = np.asarray(axes.hexbin(
                x, y, C=outcome_here, gridsize=args.gridsize, mincnt=1,
                reduce_C_function=np.mean,
            ).get_array())
            plt.close(figure)
            keep_here = totals_here >= MIN_CELL
            for column, label in covariates:
                values = complete[column].cast(pl.Float64).to_numpy()
                figure, axes = plt.subplots()
                binned = np.asarray(axes.hexbin(
                    x, y, C=values, gridsize=args.gridsize, mincnt=1,
                    reduce_C_function=np.mean,
                ).get_array())
                plt.close(figure)
                pair = float(np.corrcoef(
                    binned[keep_here], rate_here[keep_here]
                )[0, 1])
                context.append({
                    "covariate": label,
                    "hexagons": int(keep_here.sum()),
                    "correlation_with_restart_rate": pair,
                })
                logger.info(
                    f"    per-hexagon {label:<28} correlates {pair:+.3f} with "
                    "the restart rate"
                )

    pl.DataFrame([{
        "weans_placed": placed.height,
        "weans_total": weans.height,
        "hexagons": len(totals),
        "pooled_restart": overall,
        "rate_min": float(rate.min()),
        "rate_max": float(rate.max()),
        "weighted_variance": spread,
        "weighted_variance_null": expected,
        "p_patchiness": p_spread,
        "morans_i": moran,
        "p_morans_i": p_moran,
    }]).write_csv(tables / f"wean_embedding_{args.subset}.csv")
    if context:
        pl.DataFrame(context).write_csv(
            tables / f"wean_embedding_covariates_{args.subset}.csv"
        )

    placed_ids = set(zip(
        placed["hospitalization_id"].to_list(), placed["hours"].to_list()
    ))
    frame = grid.with_columns(
        is_wean=pl.struct("hospitalization_id", "offset_hours").map_elements(
            lambda r: (r["hospitalization_id"], r["offset_hours"]) in placed_ids,
            return_dtype=pl.Boolean,
        )
    )
    plot_wean_map(
        frame, positions, totals, restarts, overall, paths, args.subset, logger
    )

    # One figure per weaning word, which is the view that lets a reader ask
    # whether a PARTICULAR manoeuvre has a region, rather than whether weaning
    # in general does.
    everything = (
        paths.patient_level_home / f"wean_all_{args.subset}.parquet"
    )
    if everything.is_file():
        logger.info("per-word maps:")
        placed_all = pl.read_parquet(everything).join(
            grid, left_on=["hospitalization_id", "hours"],
            right_on=["hospitalization_id", "offset_hours"], how="inner",
        )
        plot_per_word(
            grid, placed_all, paths, args.subset, args.gridsize, logger
        )
    else:
        logger.warning(f"no {everything.name}; per-word maps skipped")

    # The whole-vocabulary view: every hexagon coloured by which BPE word owns
    # it, on the same tiling and with the same plurality/enrichment pair the
    # care-state maps use.
    fine = pl.read_parquet(
        paths.patient_level_home
        / f"fine_state_sequences_{args.subset}.parquet"
    )
    logger.info("BPE word hexagon maps:")
    plot_word_hexmaps(grid, fine, paths, args.subset, logger)
    logger.info("20 done")


if __name__ == "__main__":
    main()

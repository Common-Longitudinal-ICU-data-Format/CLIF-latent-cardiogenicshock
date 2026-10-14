#!/usr/bin/env python3

"""
08 -- the model's learned token space

A UMAP of the embedding table itself: one point per vocabulary entry, not per
patient. This asks whether the model organised clinical concepts into coherent
structure during training -- whether labs sit near labs, vasopressors near
vasopressors, and whether dose or value quantiles of the same measurement lie
on an ordered path.

It is the one figure here that needs no patient data at all, and it is a
precondition for reading anything else: if the token space is incoherent, the
patient representations built from it cannot mean much.

Tokens are weighted by how often they actually occur in the cohort, because
the embedding table also holds rows for tokens the model barely saw, whose
positions are close to their random initialisation and would otherwise read as
real structure.

Usage
    python code/08_token_space.py
    python code/08_token_space.py --model GEM-mimic-icu --min-count 300
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from safetensors.numpy import load_file

from clifcs import get_logger, load_paths
from clifcs.figures import save_figure
from clifcs.palette import (
    CATEGORICAL,
    CATEGORICAL_8,
    CONTEXT,
    GRID,
    HIGHLIGHT,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    shade_ramp,
)
from clifcs.tokens import load_pre_anchor_tokens, token_counts_by_site, vocabulary

DPI = 500

#: cocoa bins every continuous value into this many population deciles
#: and fuses the bin into the token name (tokenization.yaml: n_bins)
DECILES = 10

#: panels per decile-run figure; the leading measurements by volume,
#: because past a couple of dozen the panels are too small to read
MAX_RUN_PANELS = 24

#: families grouped for colour; everything else falls into "other"
FAMILY_GROUPS = {
    "labs": ("LAB-ORD", "LAB-RES"),
    "vitals": ("VTL",),
    "medications": ("MED-CTS", "MED-INT"),
    "respiratory": ("RESP",),
    "assessments": ("ASMT",),
    "severity / events": ("SOFA", "LABEL"),
    "location & admin": ("XFR-IN", "XFR-OUT", "ADMN", "DSCG", "CODE"),
    "demographics": ("AGE", "SEX", "RACE", "ETHN"),
}
#: One hue only. Nine groups cannot be told apart by colour in a scatter -- the
#: palette validates three slots under the all-pairs gate -- so each group gets
#: its own panel with the rest as grey context. HIGHLIGHT and CONTEXT live in
#: the palette module because 03 draws the shock-type panels the same way, and a
#: second copy of the context grey is a second thing to keep in step.


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE["light"],
            "axes.facecolor": SURFACE["light"],
            "savefig.facecolor": SURFACE["light"],
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "font.size": 8,
            "legend.frameon": False,
        }
    )


def embedding_table(model_home: pathlib.Path, logger) -> np.ndarray:
    """
    the input embedding matrix, vocab x hidden

    Read straight from the safetensors file rather than by instantiating the
    model: nothing here needs a forward pass, and loading 300 MB of weights to
    read one matrix would be wasteful.
    """
    weights = load_file(str(model_home / "model.safetensors"))
    for key in ("model.embed_tokens.weight", "embed_tokens.weight"):
        if key in weights:
            table = np.asarray(weights[key], dtype=np.float64)
            logger.info(f"embedding table {key}: {table.shape}")
            return table
    raise KeyError(
        f"no embedding matrix in {model_home / 'model.safetensors'}; "
        f"keys include {sorted(weights)[:5]}"
    )


def cohort_token_counts(paths, logger) -> pl.DataFrame:
    """how often each token occurs in this cohort's pre-anchor streams"""
    names, _ = vocabulary(str(paths.tokenizer_yaml))
    frames = []
    for site in paths.sites:
        try:
            frames.append(load_pre_anchor_tokens(paths, site, "24h"))
        except FileNotFoundError:
            logger.warning(f"no winnowed tokens for {site}; skipping")
    if not frames:
        raise FileNotFoundError("no tokenized data; run 01 first")
    tokens = pl.concat(frames)

    # the winnowed stream covers everyone embedded, which is a superset of the
    # cohort once an enrollment criterion is tightened; count only the cohort
    cohort_file = paths.cohort_home / "cohort.parquet"
    if cohort_file.is_file():
        keep = pl.read_parquet(cohort_file).select("hospitalization_id", "site")
        before = tokens.height
        tokens = tokens.join(keep, on=["hospitalization_id", "site"], how="inner")
        logger.info(f"token counts from the cohort: {before} -> {tokens.height} patients")

    counts = token_counts_by_site(tokens, names)
    return counts.group_by("token_id", "token", "family").agg(n=pl.col("n").sum())


def group_of(family: str) -> str:
    for group, members in FAMILY_GROUPS.items():
        if family in members:
            return group
    return "other"


def slug(group: str) -> str:
    """filename-safe form of a family name"""
    return re.sub(r"[^a-z0-9]+", "-", group.lower()).strip("-")


def decile_of(token: str) -> int | None:
    """the decile a fused token carries, or None if it carries no value"""
    found = re.search(r"_Q(\d+)$", token)
    return int(found.group(1)) if found else None


def plot_family(
    coords: np.ndarray,
    meta: pl.DataFrame,
    group: str,
    out_path: pathlib.Path,
    title: str,
    logger,
) -> None:
    """
    one family, as a pair: the whole space on the left, the family zoomed on
    the right, with the zoom window drawn on the left panel

    Paired rather than tabbed because the two panels answer halves of the same
    question -- where in the vocabulary this family sits, and what is inside it
    -- and flipping between them loses the correspondence. The rectangle is the
    link between the two.

    Colour carries two things at once. Hue names the measurement: the leading
    eight by frequency, with the tail folded into one grey series. Lightness
    within that hue names the decile, because cocoa never hands the model a
    number -- it bins every continuous value into ten population deciles and
    fuses the bin into the token, so sodium_Q0 through _Q9 are ten unrelated
    vocabulary entries whose ordering the model had to learn from context.
    Ramping light to dark across Q0..Q9 makes that learned ordering legible as
    geometry: where it was learned, the shades lie along a path. Tokens that
    carry no value keep the undiluted hue.

    Identity is carried by the legend over a zoomed layout, not by hue alone,
    since eight slots exceed what the all-pairs colour gate supports in a
    scatter; the deciles are a within-hue ordinal ramp, which is a separate
    channel and is read against its own key rather than pairwise.
    """
    group_values = np.asarray(meta["group"].to_list())
    counts = meta["n"].to_numpy()
    tokens = np.asarray(meta["token"].to_list())
    stems = np.asarray(
        meta.with_columns(stem=pl.col("token").str.replace(r"_Q\d+$", ""))["stem"]
        .to_list()
    )
    mask = group_values == group
    local = np.flatnonzero(mask)
    if local.size == 0:
        return

    sizes = 3.5 + 5.5 * (np.log1p(counts) / np.log1p(counts.max())) ** 2
    deciles = [decile_of(t) for t in tokens]

    by_stem: dict[str, np.ndarray] = {}
    for stem in dict.fromkeys(stems[local]):
        by_stem[stem] = local[stems[local] == stem]
    ranked = sorted(by_stem, key=lambda t: -counts[by_stem[t]].sum())
    shown, folded = ranked[: len(CATEGORICAL_8)], ranked[len(CATEGORICAL_8):]

    # how much of this family is actually binned, stated rather than implied
    binned = sum(deciles[i] is not None for i in local)

    px, py = coords[local, 0], coords[local, 1]
    # Squared about the family's centre: a family spread wide and flat gives a
    # window of that shape, and under equal aspect the zoom panel then fills
    # horizontally and wastes most of its height. A square window fills a
    # square panel, and keeps the aspect honest in both panels.
    span = max(np.ptp(px), np.ptp(py), 1.0) * 1.18
    mid_x, mid_y = (px.min() + px.max()) / 2, (py.min() + py.max()) / 2
    window = (mid_x - span / 2, mid_x + span / 2, mid_y - span / 2, mid_y + span / 2)

    # Placed by hand rather than by tight_layout: both scatter panels are
    # held to equal aspect with adjustable="box", which tight_layout cannot
    # honour, and it responds by resizing the axes after the limits are set --
    # which let the zoom panel spill past the window it claims to show.
    figure = plt.figure(figsize=(11.2, 4.6))
    axes = [
        figure.add_axes([0.015, 0.045, 0.40, 0.84]),
        figure.add_axes([0.435, 0.045, 0.40, 0.84]),
        figure.add_axes([0.850, 0.045, 0.14, 0.84]),
    ]
    legend_ax = axes[2]
    legend_ax.axis("off")

    ramps = {stem: shade_ramp(colour, DECILES)
             for colour, stem in zip(CATEGORICAL_8, shown)}
    folded_ramp = shade_ramp("#8a8980", DECILES)

    def point_colours(index: np.ndarray, ramp: list[str], base: str) -> list[str]:
        """the shade each token takes: its decile within the hue, else the hue"""
        return [
            base if deciles[i] is None else ramp[min(deciles[i], len(ramp) - 1)]
            for i in index
        ]

    handles = []
    for position, ax in enumerate(axes[:2]):
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=3.5, c=CONTEXT,
                   linewidths=0, rasterized=True, zorder=1)
        if folded:
            index = np.concatenate([by_stem[t] for t in folded])
            ax.scatter(coords[index, 0], coords[index, 1],
                       s=sizes[index] * (1.0 if position == 0 else 3.2),
                       c=point_colours(index, folded_ramp, "#8a8980"),
                       linewidths=0, alpha=0.75, rasterized=True, zorder=2)
            if position == 0:
                handles.append(
                    plt.Line2D([], [], marker="o", linestyle="none", markersize=4,
                               markerfacecolor="#8a8980", markeredgecolor="none",
                               label=f"{len(folded)} others")
                )
        for colour, stem in zip(CATEGORICAL_8, shown):
            index = by_stem[stem]
            if position == 1:
                decile_path(ax, coords, index, deciles, ramps[stem])
            ax.scatter(coords[index, 0], coords[index, 1],
                       s=sizes[index] * (1.0 if position == 0 else 3.2),
                       c=point_colours(index, ramps[stem], colour),
                       linewidths=0, alpha=0.9, rasterized=True, zorder=3)
            if position == 0:
                handles.insert(
                    len(handles) - (1 if folded else 0),
                    plt.Line2D([], [], marker="o", linestyle="none", markersize=4,
                               markerfacecolor=colour, markeredgecolor="none",
                               label=stem[:30]),
                )
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    # left: the whole space, with the right panel's window outlined
    axes[0].add_patch(
        plt.Rectangle(
            (window[0], window[2]), window[1] - window[0], window[3] - window[2],
            fill=False, edgecolor=INK_PRIMARY, linewidth=0.9, zorder=4,
        )
    )
    axes[0].set_title("whole vocabulary", loc="left", fontsize=7.5, pad=4)
    axes[0].set_aspect("equal", adjustable="box")

    # right: the same points, cropped to the window
    axes[1].set_xlim(window[0], window[1])
    axes[1].set_ylim(window[2], window[3])
    axes[1].set_title("zoomed to the boxed region", loc="left", fontsize=7.5, pad=4)
    axes[1].set_aspect("equal", adjustable="box")

    legend_ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(-0.05, 1.0),
        fontsize=6.2, labelcolor=INK_SECONDARY, handletextpad=0.5,
        borderpad=0, labelspacing=0.4,
    )
    if binned:
        decile_key(legend_ax, binned, int(local.size))
    figure.suptitle(title, x=0.015, ha="left", fontsize=9.5, y=0.975)
    save_figure(figure, out_path)
    plt.close(figure)
    logger.info(
        f"wrote {out_path.name} ({binned}/{local.size} tokens carry a decile)"
    )


def decile_path(ax, coords: np.ndarray, index: np.ndarray,
                deciles: list, ramp: list[str]) -> None:
    """
    join one measurement's deciles in order, Q0 to Q9

    The shade ramp alone is hard to read at this scale, because ten deciles of
    one measurement often land within a few points of each other and the eye
    compares touching dots rather than following a sequence. Drawing the
    sequence makes the question answerable directly: if the model recovered the
    ordering the path is smooth, and if it did not the path doubles back. It is
    the same claim the Spearman coefficients make numerically, in a form that
    can be checked by looking.

    Drawn under the markers and in the measurement's own mid shade so it reads
    as belonging to that series rather than as a separate layer.
    """
    ordered = sorted(
        (i for i in index if deciles[i] is not None), key=lambda i: deciles[i]
    )
    if len(ordered) < 3:
        return
    ax.plot(coords[ordered, 0], coords[ordered, 1],
            color=ramp[len(ramp) // 2], linewidth=0.7, alpha=0.5,
            solid_capstyle="round", zorder=2.5)


def decile_key(ax, binned: int, total: int) -> None:
    """
    the within-hue ramp, drawn once as swatches rather than repeated per series

    Lightness is a single ordinal channel shared by every hue on the plot, so
    one key serves all of them, in neutral grey to avoid implying it belongs to
    any one measurement. It sits below the hue legend because it is read
    second: first which measurement, then which decile of it.
    """
    # saturation_floor=0 so the neutral stays neutral: the floor is there
    # to keep a hue from washing out, and a grey has no hue to keep
    swatches = shade_ramp("#6b6b6b", DECILES, saturation_floor=0.0)
    left, width, height = -0.05, 0.9 / DECILES, 0.035
    base = 0.30
    ax.text(left, base + height + 0.05, "decile within each colour",
            transform=ax.transAxes, fontsize=6.2, color=INK_SECONDARY)
    for step, shade in enumerate(swatches):
        ax.add_patch(
            plt.Rectangle(
                (left + step * width, base), width, height,
                transform=ax.transAxes, facecolor=shade, edgecolor="none",
                clip_on=False,
            )
        )
    ax.text(left, base - 0.02, "Q0", transform=ax.transAxes,
            fontsize=5.8, color=INK_SECONDARY, ha="left", va="top")
    ax.text(left + 0.9, base - 0.02, "Q9", transform=ax.transAxes,
            fontsize=5.8, color=INK_SECONDARY, ha="right", va="top")
    ax.text(left, base - 0.10,
            f"{binned} of {total} tokens carry a\nbinned value; the rest carry none",
            transform=ax.transAxes, fontsize=5.6, color=INK_SECONDARY,
            ha="left", va="top", linespacing=1.5)


def plot_decile_runs(
    coords: np.ndarray,
    meta: pl.DataFrame,
    coherence: pl.DataFrame,
    group: str,
    out_path: pathlib.Path,
    title: str,
    logger,
) -> None:
    """
    one small panel per measurement, cropped to that measurement's own deciles

    The family figures colour deciles as a ramp within each hue, which shows
    that the gradient is there but not whether it is ordered: a measurement's
    ten deciles span 1.1% of the labs window and 3.2% of the vitals window, so
    at family scale a decile run is a few pixels and no crop fixes it -- the
    windows are set by genuine spread, not by outliers (the central-98% window
    is within 1.14x of the full extent for every family).

    Cropping each measurement to its own run makes the run fill its panel by
    construction, which is what makes the claim checkable: a model that
    recovered the ordering draws a path that advances light to dark without
    doubling back. The Spearman coefficient on each panel is the same quantity
    measured in the native 1024-d space, so the picture and the number can be
    read against each other rather than taken on trust.

    Hues follow CATEGORICAL_8 in the same frequency order as the family figure,
    so the leading measurements keep the colour they had there. Past eight the
    hues repeat, which is harmless here and would not be in the family figure:
    each panel is cropped, labelled and read alone, so hue is decoration rather
    than the thing carrying identity.
    """
    scores = {
        row["measurement"]: (row["spearman"], row["spearman_umap"])
        for row in coherence.iter_rows(named=True)
    }
    counts = meta["n"].to_numpy()
    stems = np.asarray(
        meta.with_columns(stem=pl.col("token").str.replace(r"_Q\d+$", ""))["stem"]
        .to_list()
    )
    tokens = np.asarray(meta["token"].to_list())
    deciles = [decile_of(t) for t in tokens]
    local = np.flatnonzero(np.asarray(meta["group"].to_list()) == group)

    runs = []
    for stem in dict.fromkeys(stems[local]):
        index = np.array([i for i in local[stems[local] == stem]
                          if deciles[i] is not None])
        if index.size >= 5:
            runs.append((stem, index[np.argsort([deciles[i] for i in index])]))
    if not runs:
        return
    runs.sort(key=lambda pair: -counts[pair[1]].sum())
    runs = runs[:MAX_RUN_PANELS]

    columns = min(6, len(runs))
    rows = math.ceil(len(runs) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(1.62 * columns, 1.90 * rows), squeeze=False
    )
    for position, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(position < len(runs))
            ax.spines[side].set_color(GRID)
        if position >= len(runs):
            ax.set_visible(False)
            continue

        stem, index = runs[position]
        ramp = shade_ramp(CATEGORICAL_8[position % len(CATEGORICAL_8)], DECILES)
        x, y = coords[index, 0], coords[index, 1]
        span = max(np.ptp(x), np.ptp(y), 1e-6) * 1.35
        ax.set_xlim((x.min() + x.max()) / 2 - span / 2,
                    (x.min() + x.max()) / 2 + span / 2)
        ax.set_ylim((y.min() + y.max()) / 2 - span / 2,
                    (y.min() + y.max()) / 2 + span / 2)
        ax.set_aspect("equal", adjustable="box")

        ax.plot(x, y, color=ramp[len(ramp) // 2], linewidth=0.8, alpha=0.55,
                solid_capstyle="round", zorder=2)
        ax.scatter(x, y, s=46,
                   c=[ramp[min(deciles[i], DECILES - 1)] for i in index],
                   linewidths=0.4, edgecolors=SURFACE["light"], zorder=3)
        ax.annotate("Q0", (x[0], y[0]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=5.0,
                    color=INK_SECONDARY, zorder=4)
        ax.annotate(f"Q{deciles[index[-1]]}", (x[-1], y[-1]),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=5.0, color=INK_SECONDARY, zorder=4)

        name = stem.split("//")[-1]
        # native first because it is the claim about the model; "here" is
        # the projection, and where the two part company the drawn path is
        # the casualty rather than the evidence
        pair = scores.get(stem)
        label = name[:22] if pair is None else (
            f"{name[:22]}\nρ {pair[0]:+.2f} native \u00b7 "
            f"{pair[1]:+.2f} here"
        )
        ax.set_title(label, fontsize=5.5, color=INK_PRIMARY, loc="center",
                     pad=3, linespacing=1.4)

    figure.suptitle(title, x=0.012, ha="left", fontsize=9.5, y=0.995)
    # h_pad clears the two-line titles, which sit above an equal-aspect box
    # and would otherwise land on the row above
    figure.tight_layout(rect=(0, 0, 1, 0.945), h_pad=2.4, w_pad=1.1)
    save_figure(figure, out_path)
    plt.close(figure)
    logger.info(f"wrote {out_path.name} ({len(runs)} measurements)")


def quantile_coherence(
    vectors: np.ndarray, coords: np.ndarray, meta: pl.DataFrame, logger
) -> pl.DataFrame:
    """
    do the quantile bins of one measurement lie on an ordered path?

    cocoa bins numeric values into deciles, so `LAB-RES//lactate_Q0` through
    `_Q9` are ten separate tokens the model had to relate on its own. If it
    learned the ordering, consecutive quantiles sit closer than distant ones.
    Reported as the Spearman correlation between |quantile difference| and
    distance between the tokens, per measurement with at least five bins.

    Measured in the native 1024-d space, under the cosine distance the UMAP
    was built on. The projection is reported alongside but is not the claim:
    a correlation computed on 2-d coordinates cannot separate what the model
    learned from how UMAP chose to lay it out, and would overstate the result
    whenever the layout happens to be kind.
    """
    rows = []
    base = (
        meta.with_columns(
            stem=pl.col("token").str.replace(r"_Q\d$", ""),
            quantile=pl.col("token").str.extract(r"_Q(\d)$", 1).cast(pl.Int64),
        )
        .with_row_index("row")
        .filter(pl.col("quantile").is_not_null())
    )
    for stem, part in base.group_by("stem"):
        stem = stem[0] if isinstance(stem, tuple) else stem
        if part.height < 5:
            continue
        index = part["row"].to_numpy()
        q = part["quantile"].to_numpy()
        native = vectors[index]
        native = native / np.linalg.norm(native, axis=1, keepdims=True)
        flat = coords[index]
        dq, d_native, d_flat = [], [], []
        for i in range(len(index)):
            for j in range(i + 1, len(index)):
                dq.append(abs(q[i] - q[j]))
                d_native.append(1.0 - float(native[i] @ native[j]))
                d_flat.append(float(np.linalg.norm(flat[i] - flat[j])))
        if len(dq) < 6:
            continue

        def spearman(a, b):
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            return float(np.corrcoef(ra, rb)[0, 1])

        rows.append({
            "measurement": stem,
            "n_bins": part.height,
            "spearman": spearman(dq, d_native),
            "spearman_umap": spearman(dq, d_flat),
        })

    table = pl.DataFrame(rows).sort("spearman", descending=True)
    if table.height:
        logger.info(
            f"quantile ordering across {table.height} binned measurements, "
            f"in the native 1024-d space (cosine): median Spearman "
            f"{table['spearman'].median():.3f}, "
            f"{(table['spearman'] > 0).mean():.0%} positive"
        )
        logger.info(
            f"    for comparison, in the 2-d UMAP projection: median "
            f"{table['spearman_umap'].median():.3f}, "
            f"{(table['spearman_umap'] > 0).mean():.0%} positive"
        )
        for row in table.head(5).iter_rows(named=True):
            logger.info(
                f"    {row['measurement']:<34} native {row['spearman']:+.3f}  "
                f"umap {row['spearman_umap']:+.3f}"
            )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--min-count",
        type=int,
        default=300,
        help="drop tokens occurring fewer times than this in the cohort; rare "
        "tokens sit near their random initialisation and would read as structure",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--role",
        choices=("site", "pooled"),
        default=None,
        help="`pooled` reads received bundles from output/export/ and refuses "
        "to touch raw CLIF data; `site` reads this machine's own files. "
        "Defaults to `role` in paths.yaml, then to `site`.",
    )
    args = parser.parse_args()

    paths = load_paths(model=args.model, role=args.role)
    paths.mkdirs()
    logger = get_logger("08_token_space", paths.log_home / "08_token_space.log")
    style()

    import umap

    table = embedding_table(paths.model_home, logger)
    counts = cohort_token_counts(paths, logger)
    names, families = vocabulary(str(paths.tokenizer_yaml))

    meta = (
        counts.filter(pl.col("n") >= args.min_count)
        .with_columns(group=pl.col("family").map_elements(group_of, return_dtype=pl.String))
        .sort("token_id")
    )
    index = meta["token_id"].to_numpy()
    if index.max() >= table.shape[0]:
        raise IndexError(
            f"token id {index.max()} exceeds the embedding table ({table.shape[0]} rows)"
        )
    vectors = table[index]
    logger.info(
        f"{len(index)} of {len(names)} tokens occur >= {args.min_count} times; "
        f"embedding dim {vectors.shape[1]}"
    )

    coords = umap.UMAP(
        n_neighbors=15, min_dist=0.25, n_components=2, metric="cosine",
        random_state=args.seed,
    ).fit_transform(vectors)

    # ahead of the figures because the decile-run panels are labelled with
    # these coefficients: the picture and the number should come from one pass
    coherence = quantile_coherence(vectors, coords, meta, logger)

    groups = [g for g in list(FAMILY_GROUPS) + ["other"]
              if (meta["group"] == g).any()]
    for group in groups:
        members = int((meta["group"] == group).sum())
        measurements = (
            meta.filter(pl.col("group") == group)
            .with_columns(stem=pl.col("token").str.replace(r"_Q\d+$", ""))["stem"]
            .n_unique()
        )
        plot_family(
            coords,
            meta,
            group,
            paths.figure_home / f"token_space_{slug(group)}.png",
            f"{paths.model_key} · {group}: {members} tokens, "
            f"{measurements} measurements",
            logger,
        )
        plot_decile_runs(
            coords,
            meta,
            coherence,
            group,
            paths.figure_home / f"token_deciles_{slug(group)}.png",
            f"{paths.model_key} \u00b7 {group}: decile runs, each cropped to "
            f"its own measurement",
            logger,
        )

    if coherence.height:
        coherence.write_csv(paths.table_home / "token_quantile_ordering.csv")
    meta.with_columns(
        umap_1=pl.Series(coords[:, 0]), umap_2=pl.Series(coords[:, 1])
    ).write_csv(paths.table_home / "token_space_umap.csv")
    logger.info("08 done")


if __name__ == "__main__":
    main()

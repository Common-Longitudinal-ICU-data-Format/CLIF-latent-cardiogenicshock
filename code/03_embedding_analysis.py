#!/usr/bin/env python3

"""
03 -- embedding analysis: PCA, UMAP and PaCMAP of the frozen-model representations

Reads output/patient_level_data/, projects the embeddings to 2-d with PCA,
UMAP and PaCMAP, and writes one scatterplot per (projection x colouring x
anchor).

All three projections are computed and saved every run. PaCMAP is what the
report shows: it is built to preserve global structure as well as local
neighbourhoods, where UMAP optimises the local graph and will happily tear
apart or fuse clusters that are genuinely near each other. UMAP is kept because
it is what the earlier figures used and the two should be comparable.

Colourings: contributing site, Zweck cardiogenic shock phenotype, ICU length of
stay, in-hospital mortality.

Plot conventions, as specified: dpi=500, the legend sits outside the scatter
axes, and the scatter keeps a 1:1 data aspect ratio so distances in the
projection are not visually distorted along one axis.

Usage
    python code/03_embedding_analysis.py
    python code/03_embedding_analysis.py --anchor 24h --max-points 20000
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clifcs import ANCHORS, get_logger, load_paths
from clifcs.figures import key_string, save_figure
from clifcs.palette import (
    CONTEXT,
    GRID,
    HIGHLIGHT,
    INK_PRIMARY,
    INK_SECONDARY,
    mortality_colors,
    phenotype_colors,
    continuous_cmap,
    site_colors,
    SURFACE,
)
from clifcs.phenotype import PHENOTYPES
from clifcs.shock_type import SHOCK_TYPES

DPI = 500
#: PCA components kept before UMAP and PaCMAP, to denoise and speed up the
#: neighbour graph
PCA_COMPONENTS = 50

#: neighbours for both UMAP and PaCMAP. Shared deliberately -- see `project`.
N_NEIGHBORS = 30

LOS_CMAP = continuous_cmap()


def style() -> None:
    """recessive axes and grid; text in ink tokens, never in a series colour"""
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
            "figure.dpi": 110,
        }
    )


# ============================================================== projections


def embedding_matrix(frame: pl.DataFrame) -> np.ndarray:
    columns = sorted(c for c in frame.columns if c.startswith("emb_"))
    if not columns:
        raise RuntimeError("no emb_* columns found in the patient-level data")
    return frame.select(columns).to_numpy().astype(np.float64)


def project(
    frame: pl.DataFrame, seed: int, logger
) -> tuple[pl.DataFrame, np.ndarray]:
    """
    attach PCA, UMAP and PaCMAP coordinates

    All three run on z-scored embeddings, and the two neighbour-graph methods
    are initialised from the leading PCA components rather than the raw 1024-d
    vectors: the graph is then computed from the same denoised subspace for
    every anchor, which makes the anchors' layouts comparable, and it is far
    faster.

    UMAP and PaCMAP are both given `N_NEIGHBORS`. PaCMAP would otherwise pick
    its own count from the sample size, and matching them is what makes the two
    panels a comparison of the two algorithms rather than of two neighbourhood
    scales.

    PaCMAP is told not to apply its own PCA (`apply_pca=False`): it is already
    receiving PCA scores, and letting it re-project them would mean the saved
    `PCA_COMPONENTS`-dimensional subspace is not actually what it embedded.
    """
    import pacmap
    import umap

    raw = embedding_matrix(frame)
    logger.info(f"  embeddings: {raw.shape[0]} patients x {raw.shape[1]} dims")

    scaled = StandardScaler().fit_transform(raw)

    pca_full = PCA(n_components=min(PCA_COMPONENTS, scaled.shape[1]), random_state=seed)
    pca_scores = pca_full.fit_transform(scaled)
    explained = pca_full.explained_variance_ratio_
    logger.info(
        f"  PCA: PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%}, "
        f"first {len(explained)} PCs {explained.sum():.1%} of variance"
    )

    logger.info(f"  UMAP: building neighbour graph (k={N_NEIGHBORS})")
    umap_coords = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=seed,
    ).fit_transform(pca_scores)

    logger.info(f"  PaCMAP: building neighbour graph (k={N_NEIGHBORS})")
    # float32: PaCMAP's neighbour search runs through faiss, which is
    # single-precision, and handing it float64 just copies the array first
    pacmap_coords = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        MN_ratio=0.5,
        FP_ratio=2.0,
        apply_pca=False,
        random_state=seed,
    ).fit_transform(np.ascontiguousarray(pca_scores, dtype=np.float32))

    return frame.with_columns(
        pca_1=pl.Series(pca_scores[:, 0]),
        pca_2=pl.Series(pca_scores[:, 1]),
        umap_1=pl.Series(umap_coords[:, 0]),
        umap_2=pl.Series(umap_coords[:, 1]),
        pacmap_1=pl.Series(pacmap_coords[:, 0]),
        pacmap_2=pl.Series(pacmap_coords[:, 1]),
    ), explained


# =================================================================== plotting


def _new_figure():
    """
    scatter axes on the left, legend axes on the right

    The legend lives in its own axes rather than inside the scatter: the
    requirement is that it sits off the plot, and a separate axes also keeps it
    from ever occluding a point.
    """
    figure, (scatter_ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(6.4, 4.6),
        gridspec_kw={"width_ratios": [4.0, 1.25], "wspace": 0.04},
    )
    legend_ax.axis("off")
    return figure, scatter_ax, legend_ax


def _finish(scatter_ax, title: str, xlabel: str, ylabel: str) -> None:
    scatter_ax.set_title(title, loc="left", pad=8)
    scatter_ax.set_xlabel(xlabel)
    scatter_ax.set_ylabel(ylabel)
    scatter_ax.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
    scatter_ax.set_axisbelow(True)
    for side in ("top", "right"):
        scatter_ax.spines[side].set_visible(False)
    # equal data aspect: one unit on x is one unit on y, so the projection is
    # not stretched along either axis
    scatter_ax.set_aspect("equal", adjustable="datalim")


def _marker_size(n: int) -> float:
    """smaller marks as the cloud gets denser, so structure stays visible"""
    if n > 20000:
        return 1.2
    if n > 8000:
        return 2.5
    if n > 2000:
        return 5.0
    return 9.0


def plot_categorical(
    frame: pl.DataFrame,
    x: str,
    y: str,
    column: str,
    colors: dict,
    order,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: pathlib.Path,
    rng: np.random.Generator,
) -> None:
    """
    one scatter per category

    Drawing order is shuffled across categories rather than category-by-category
    so the last group plotted does not sit on top of the others and exaggerate
    its own extent.
    """
    figure, scatter_ax, legend_ax = _new_figure()
    size = _marker_size(frame.height)

    shuffled = frame.with_columns(
        _order=pl.Series(rng.permutation(frame.height))
    ).sort("_order")

    values = shuffled[column].to_list()
    xs = np.asarray(shuffled[x].to_list())
    ys = np.asarray(shuffled[y].to_list())
    point_colors = [colors[v] for v in values]

    scatter_ax.scatter(
        xs, ys, s=size, c=point_colors, linewidths=0, alpha=0.75, rasterized=True
    )
    _finish(scatter_ax, title, xlabel, ylabel)

    counts = frame[column].value_counts()
    lookup = dict(zip(counts[column].to_list(), counts["count"].to_list()))
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor=colors[key],
            markeredgecolor="none",
            label=f"{label}  n={lookup.get(key, 0):,}",
        )
        for key, label in order
    ]
    legend_ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        labelcolor=INK_SECONDARY,
        handletextpad=0.6,
        borderpad=0,
        fontsize=7,
    )

    save_figure(figure, out_path)
    plt.close(figure)


def plot_categorical_facets(
    frame: pl.DataFrame,
    x: str,
    y: str,
    column: str,
    order,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: pathlib.Path,
    columns: int = 3,
) -> None:
    """
    one panel per category: that category in the highlight hue, the rest grey

    This is the documented answer past three series, and shock type has six.
    A scatter shows every series at once, so the all-pairs colourblind gate
    applies and the palette validates three slots under it -- six hues cannot
    be told apart no matter how they are chosen. Giving each category its own
    panel moves identity out of colour and into position on the page, so the
    one hue only has to separate from the grey context layer.

    It is also the more honest figure for this particular question. Shock types
    overlap heavily in the latent space, and in a single pooled scatter whichever
    category is drawn last looks larger than it is. Here every panel shows the
    same cloud, so "septic shock is everywhere" and "cardiogenic shock sits in
    one corner" are both readable, and the reader can compare panel to panel
    rather than trusting an overplotted legend.
    """
    levels = [(key, label) for key, label in order]
    rows = int(np.ceil(len(levels) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.55 * columns, 2.75 * rows),
        squeeze=False,
    )

    xs = np.asarray(frame[x].to_list())
    ys = np.asarray(frame[y].to_list())
    values = [key_string(v) for v in frame[column].to_list()]
    size = _marker_size(frame.height)

    for position, (key, label) in enumerate(levels):
        ax = axes[position // columns][position % columns]
        mask = np.array([v == key_string(key) for v in values])

        ax.scatter(
            xs[~mask], ys[~mask], s=size * 0.8, c=CONTEXT,
            linewidths=0, rasterized=True, zorder=1,
        )
        ax.scatter(
            xs[mask], ys[mask], s=size * 1.6, c=HIGHLIGHT,
            linewidths=0, alpha=0.85, rasterized=True, zorder=2,
        )
        share = mask.sum() / max(len(mask), 1)
        ax.set_title(
            f"{label}\nn={int(mask.sum()):,} ({share:.1%})",
            loc="left", pad=5, fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    for position in range(len(levels), rows * columns):
        axes[position // columns][position % columns].axis("off")

    figure.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK_PRIMARY)
    # axes are unlabelled on purpose: the units of a neighbour-graph projection
    # are arbitrary, and six repeated axis labels crowd out the panel titles
    figure.supxlabel(
        f"{xlabel} / {ylabel}  -- axes shared, scales arbitrary",
        x=0.01, ha="left", fontsize=7, color=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(figure, out_path)
    plt.close(figure)


def plot_continuous(
    frame: pl.DataFrame,
    x: str,
    y: str,
    column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str,
    out_path: pathlib.Path,
    rng: np.random.Generator,
    upper_percentile: float = 99.0,
) -> None:
    """
    scatter coloured by a continuous value, with the colourbar off the scatter

    ICU length of stay is heavily right-skewed, so the scale is clipped at the
    99th percentile: on a raw scale a handful of multi-month stays compress
    everyone else into the lightest step and the plot shows nothing.
    """
    figure, scatter_ax, legend_ax = _new_figure()
    size = _marker_size(frame.height)

    shuffled = frame.with_columns(
        _order=pl.Series(rng.permutation(frame.height))
    ).sort("_order")

    values = np.asarray(shuffled[column].to_list(), dtype=float)
    vmax = float(np.nanpercentile(values, upper_percentile))
    vmin = float(np.nanmin(values))
    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))

    points = scatter_ax.scatter(
        np.asarray(shuffled[x].to_list()),
        np.asarray(shuffled[y].to_list()),
        s=size,
        c=values,
        cmap=LOS_CMAP,
        norm=norm,
        linewidths=0,
        alpha=0.8,
        rasterized=True,
    )
    _finish(scatter_ax, title, xlabel, ylabel)

    bar_ax = legend_ax.inset_axes([0.08, 0.35, 0.16, 0.6])
    colorbar = figure.colorbar(points, cax=bar_ax, extend="max")
    colorbar.set_label(label, color=INK_SECONDARY, fontsize=7)
    colorbar.ax.tick_params(labelsize=6, color=GRID, labelcolor=INK_SECONDARY)
    colorbar.outline.set_visible(False)

    median = float(np.nanmedian(values))
    legend_ax.text(
        0.08,
        0.22,
        f"median {median:.1f}\nscale clipped at\np{upper_percentile:g} = {vmax:.1f}",
        transform=legend_ax.transAxes,
        fontsize=6,
        color=INK_SECONDARY,
        va="top",
    )

    save_figure(figure, out_path)
    plt.close(figure)


# =================================================================== driver


#: covariates the plots need; every contributing site must ship these
REQUIRED_COVARIATES = (
    "site",
    "cs_phenotype",
    "icu_los_days",
    "in_hospital_mortality",
)


def load_anchor(paths, anchor: str, logger) -> pl.DataFrame:
    """
    pool every site's embedding file for one anchor

    Files are found by glob rather than named, so a collaborating site
    contributes by dropping its `embeddings_anchor-<anchor>_<SITE>.parquet`
    into patient_level_data/ -- no merge step, no code change. Sites differ in
    whether they can ship identifiers, so the union of columns is taken and
    only `REQUIRED_COVARIATES` are insisted on.
    """
    sources = sorted(
        paths.patient_level_home.glob(f"embeddings_anchor-{anchor}_*.parquet")
    )
    # a share-safe copy is a redaction of a file already in the list locally;
    # only read it when its unredacted twin is absent (i.e. it came from elsewhere)
    stems = {p.name.replace("_share-safe", "") for p in sources}
    sources = [
        p
        for p in sources
        if "_share-safe" not in p.name or p.name.replace("_share-safe", "") not in stems
    ]
    if not sources:
        logger.warning(
            f"[{anchor}] no embeddings_anchor-{anchor}_*.parquet in "
            f"{paths.patient_level_home}; run 02 first"
        )
        return pl.DataFrame()

    frames = []
    for source in sources:
        part = pl.read_parquet(source)
        missing = [c for c in REQUIRED_COVARIATES if c not in part.columns]
        if missing:
            raise KeyError(f"{source.name} is missing required columns {missing}")
        logger.info(
            f"[{anchor}] {part.height:>6} patients from {source.name} "
            f"({part['site'].unique().to_list()})"
        )
        frames.append(part)

    frame = pl.concat(frames, how="diagonal_relaxed")

    # Refresh covariates from the local cohort table where it has them, and
    # drop locally-processed patients the cohort no longer contains. 02 embeds
    # everyone in the winnowed stream, which becomes a superset the moment an
    # enrollment criterion is tightened, so without the drop this would keep
    # describing the old cohort. A patient from a site this machine does not
    # hold data for is a contribution from elsewhere: it is kept, on the
    # covariates shipped with it.
    cohort_file = paths.cohort_home / "cohort.parquet"
    if cohort_file.is_file() and "hospitalization_id" in frame.columns:
        cohort = pl.read_parquet(cohort_file)

        # Driven off whichever of these the cohort file actually has, rather
        # than a fixed list. A covariate can be absent on either side: added to
        # the cohort after an embedding file was written (shock_type), or
        # present locally but stripped from a share-safe file sent by another
        # site. Naming them unconditionally meant adding one column to the
        # cohort broke 03 until 02 was rerun, which is a long way round for a
        # join.
        refreshable = [
            c
            for c in (
                "cs_phenotype",
                "icu_los_days",
                "in_hospital_mortality",
                "cardiac_origin",
                "shock_type",
            )
            if c in cohort.columns
        ]
        local = cohort.select(
            "hospitalization_id",
            "site",
            *(pl.col(c).alias(f"{c}_local") for c in refreshable),
        )
        frame = frame.join(local, on=["hospitalization_id", "site"], how="left")

        # membership is decided on the join itself, not on any one covariate
        marker = "cs_phenotype_local" if "cs_phenotype" in refreshable else None
        local_sites = set(paths.sites)
        before = frame.height
        if marker is not None:
            frame = frame.filter(
                ~pl.col("site").is_in(list(local_sites))
                | pl.col(marker).is_not_null()
            )
            if frame.height != before:
                logger.info(
                    f"[{anchor}] dropped {before - frame.height} patients no "
                    f"longer in the cohort: {before} -> {frame.height}"
                )
            n_refreshed = int(frame[marker].is_not_null().sum())
        else:
            n_refreshed = 0

        frame = frame.with_columns(
            **{
                c: (
                    pl.coalesce(f"{c}_local", c)
                    if c in frame.columns
                    else pl.col(f"{c}_local")
                )
                for c in refreshable
            }
        ).drop(*(f"{c}_local" for c in refreshable))
        logger.info(
            f"[{anchor}] refreshed {len(refreshable)} covariates "
            f"({', '.join(refreshable)}) for {n_refreshed} locally-processed "
            f"patients from {cohort_file.name}"
        )

    return frame


def analyse_anchor(paths, anchor: str, args, logger) -> pl.DataFrame:
    frame = load_anchor(paths, anchor, logger)
    if frame.height == 0:
        return pl.DataFrame()

    sites = [s for s in frame["site"].unique().sort().to_list()]
    logger.info(f"[{anchor}] {frame.height} patients across {len(sites)} sites: {sites}")

    rng = np.random.default_rng(args.seed)
    if args.max_points and frame.height > args.max_points:
        logger.info(f"[{anchor}] subsampling to {args.max_points} for the plots")
        keep = rng.choice(frame.height, size=args.max_points, replace=False)
        frame = frame[np.sort(keep)]

    frame, explained = project(frame, args.seed, logger)

    figure_home = paths.figure_home / f"anchor-{anchor}"
    figure_home.mkdir(parents=True, exist_ok=True)

    projections = (
        (
            "pca",
            "pca_1",
            "pca_2",
            f"PC1 ({explained[0]:.1%})",
            f"PC2 ({explained[1]:.1%})",
            "PCA",
        ),
        ("umap", "umap_1", "umap_2", "UMAP-1", "UMAP-2", "UMAP"),
        ("pacmap", "pacmap_1", "pacmap_2", "PaCMAP-1", "PaCMAP-2", "PaCMAP"),
    )

    anchor_label = {
        "24h": "24 h after admission",
        "onset": "cardiogenic shock onset",
    }[anchor]

    for key, x, y, xlabel, ylabel, pretty in projections:
        base = f"{pretty} of {paths.model_key} embeddings, {anchor_label}"

        plot_categorical(
            frame,
            x,
            y,
            "site",
            site_colors(sites),
            [(s, s) for s in sites],
            f"{base}\ncoloured by contributing site",
            xlabel,
            ylabel,
            figure_home / f"{key}_site.png",
            rng,
        )

        phenotype_frame = frame.filter(pl.col("cs_phenotype").is_not_null())
        plot_categorical(
            phenotype_frame,
            x,
            y,
            "cs_phenotype",
            phenotype_colors(),
            [(p, p) for p in PHENOTYPES],
            f"{base}\ncoloured by cardiogenic shock phenotype (Zweck 2021)",
            xlabel,
            ylabel,
            figure_home / f"{key}_phenotype.png",
            rng,
        )

        if "shock_type" in frame.columns:
            shock_frame = frame.filter(pl.col("shock_type").is_not_null())
            plot_categorical_facets(
                shock_frame,
                x,
                y,
                "shock_type",
                [(s, s) for s in SHOCK_TYPES],
                f"{base}\nby shock type from coded diagnoses",
                xlabel,
                ylabel,
                figure_home / f"{key}_shock_type.png",
            )

        plot_categorical(
            frame,
            x,
            y,
            "in_hospital_mortality",
            mortality_colors(),
            [(False, "survived"), (True, "died in hospital")],
            f"{base}\ncoloured by in-hospital mortality",
            xlabel,
            ylabel,
            figure_home / f"{key}_mortality.png",
            rng,
        )

        plot_continuous(
            frame.filter(pl.col("icu_los_days").is_not_null()),
            x,
            y,
            "icu_los_days",
            f"{base}\ncoloured by ICU length of stay",
            xlabel,
            ylabel,
            "ICU LOS (days)",
            figure_home / f"{key}_icu_los.png",
            rng,
        )

    logger.info(f"[{anchor}] wrote 8 figures to {figure_home}")

    # the table view the contrast-relief rule asks for, and a record of the
    # coordinates so the figures can be remade without re-embedding
    projection_cols = [
        c
        for c in (
            "hospitalization_id",
            "patient_id",
            "patient_id_pseudonym",
            "hospitalization_id_pseudonym",
        )
        if c in frame.columns
    ]
    optional = [
        c
        for c in (
            "cardiac_origin",
            "shock_type",
            "age_at_admission",
            "sex_category",
            "split",
        )
        if c in frame.columns
    ]
    frame.select(
        *projection_cols,
        "site",
        "anchor",
        "pca_1",
        "pca_2",
        "umap_1",
        "umap_2",
        "pacmap_1",
        "pacmap_2",
        "cs_phenotype",
        "icu_los_days",
        "in_hospital_mortality",
        *optional,
    ).write_parquet(paths.patient_level_home / f"projections_anchor-{anchor}.parquet")

    purity = neighbourhood_purity(frame, anchor, args.seed, logger)
    purity.write_csv(paths.table_home / f"neighbourhood_purity_anchor-{anchor}.csv")

    return summarize_anchor(frame, anchor)


def neighbourhood_purity(
    frame: pl.DataFrame, anchor: str, seed: int, logger, k: int = 30
) -> pl.DataFrame:
    """
    how much of the embedding's geometry each label actually explains

    For each patient, the share of its k nearest neighbours carrying the same
    label, against the share expected if the label were scattered at random
    (sum of squared class proportions). Lift near 1.0 means the label is not
    what the space is organised by.

    This is the check that decides whether the scatterplots can be read
    clinically at all. A foundation model trained on event streams picks up
    documentation practice -- which units exist, which assessment scales get
    charted, how often labs are drawn -- as readily as physiology, and those
    are site-specific. If site dominates, apparent phenotype or outcome
    structure in the plot is mostly site composition showing through, and the
    honest comparison is within site rather than across the pooled cloud.

    Measured in the 50-d PCA space rather than the 2-d UMAP, because UMAP
    distorts neighbourhoods by construction and would flatter whatever
    structure it happened to lay out.
    """
    from sklearn.neighbors import NearestNeighbors

    coordinates = PCA(
        n_components=min(PCA_COMPONENTS, frame.height - 1), random_state=seed
    ).fit_transform(StandardScaler().fit_transform(embedding_matrix(frame)))

    k = min(k, frame.height - 1)
    neighbours = (
        NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        .fit(coordinates)
        .kneighbors(coordinates, return_distance=False)[:, 1:]
    )

    rows = []
    for label in ("site", "cs_phenotype", "in_hospital_mortality"):
        if label not in frame.columns:
            continue
        values = np.asarray(frame[label].cast(pl.String).to_list())
        _, codes = np.unique(values, return_inverse=True)
        purity = float(np.mean(codes[neighbours] == codes[:, None]))
        proportions = np.bincount(codes) / len(codes)
        chance = float((proportions**2).sum())
        rows.append(
            {
                "anchor": anchor,
                "label": label,
                "k": k,
                "purity": purity,
                "chance": chance,
                "lift": purity / chance if chance else float("nan"),
            }
        )

    table = pl.DataFrame(rows)
    logger.info(f"[{anchor}] neighbourhood purity in {coordinates.shape[1]}-d PCA space (k={k}):")
    for row in table.iter_rows(named=True):
        logger.info(
            f"    {row['label']:<24} {row['purity']:>6.1%} observed vs "
            f"{row['chance']:>6.1%} by chance  ({row['lift']:.2f}x)"
        )
    site_lift = table.filter(pl.col("label") == "site")["lift"]
    if site_lift.len() and float(site_lift[0]) > 1.3:
        logger.warning(
            f"[{anchor}] site dominates the embedding geometry "
            f"({float(site_lift[0]):.2f}x). Structure in the phenotype and "
            "outcome plots is substantially site composition; compare within "
            "site before reading anything clinical into it."
        )
    return table


def summarize_anchor(frame: pl.DataFrame, anchor: str) -> pl.DataFrame:
    """per-site counts and outcome rates, as the figures' accompanying table"""
    return (
        frame.group_by("site")
        .agg(
            anchor=pl.lit(anchor),
            n=pl.len(),
            mortality=pl.col("in_hospital_mortality").mean(),
            median_icu_los_days=pl.col("icu_los_days").median(),
            pct_cardiac_origin=(
                pl.col("cardiac_origin").mean()
                if "cardiac_origin" in frame.columns
                else pl.lit(None, dtype=pl.Float64)
            ),
            **{
                f"pct_{name.split()[0]}": (pl.col("cs_phenotype") == name).mean()
                for name in PHENOTYPES
            },
        )
        .sort("site")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", choices=ANCHORS, default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="which checkpoint's embeddings to analyse (default: default_model)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="subsample before projecting (0 = use every patient)",
    )
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "03_embedding_analysis", paths.log_home / "03_embedding_analysis.log"
    )
    style()

    summaries = [
        summary
        for anchor in ((args.anchor,) if args.anchor else ANCHORS)
        if (summary := analyse_anchor(paths, anchor, args, logger)).height
    ]
    if summaries:
        table = pl.concat(summaries)
        table.write_csv(paths.table_home / "embedding_summary.csv")
        logger.info(f"wrote {paths.table_home / 'embedding_summary.csv'}")
        for row in table.iter_rows(named=True):
            logger.info(
                f"  {row['anchor']:<6} {row['site']:<5} n={row['n']:<6} "
                f"mortality {row['mortality']:.1%}  "
                f"median ICU LOS {row['median_icu_los_days']:.1f}d"
            )

    logger.info("03 done")


if __name__ == "__main__":
    main()

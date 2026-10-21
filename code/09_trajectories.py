#!/usr/bin/env python3

"""
09 -- patient trajectories through the latent space

Embeds each patient repeatedly, on a grid of timepoints around cardiogenic
shock onset, and draws where they move.

Two figures:
  scatter     every patient-timepoint in one projection, coloured by time
              relative to onset, by phenotype, and by outcome
  flow        the mean displacement field, as a streamplot

The flow plot is modelled on scVelo's velocity embedding, with one important
difference in its favour. RNA velocity has to *infer* direction from spliced
and unspliced read ratios, because a cell is destroyed when it is measured and
only one snapshot per cell exists. We observe the same patient at t and t+6h,
so the displacement is measured rather than estimated. The arrows are data.

The caveat is the projection, not the velocity: UMAP is not metric, so a
displacement in 2-d is not proportional to the displacement in 1024-d. The
field is therefore also computed in PCA space, where distances are linear, and
the two are reported together. If they disagree, the UMAP field is an artefact
of the layout.

Usage
    python code/09_trajectories.py --stage extract     # slow, GPU
    python code/09_trajectories.py --stage plot
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clifcs import ANCHORS, get_logger, load_paths
from clifcs.config import CONFIG_HOME
from clifcs.grid_winnower import GridWinnower
from clifcs.figures import key_string, quantile_colour, save_figure
from clifcs.palette import (
    husl_colors,
    mortality_colors,
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    phenotype_colors,
    continuous_cmap,
    SURFACE,
    site_colors,
)
from clifcs.phenotype import PHENOTYPES
from clifcs.sepsis_phenotype import BHAVANI_LABELS, SEYMOUR_LABELS
from clifcs.state import (
    VARIABLES,
    gradient_alignment,
    position_encodes,
    trajectory_state,
)

DPI = 500
SPLITS = ("train", "tuning", "held_out")
PCA_COMPONENTS = 50


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


def offsets(args) -> list[float]:
    return list(np.arange(args.start_hours, args.end_hours + 1e-9, args.step_hours))


#: which patients the grid is extracted for. The full cohort at 26 grid points
#: is 13x the work of one anchor -- about 12-20 hours for 40,000 patients on a
#: single consumer GPU -- and the figure this analysis exists for is the flow
#: among cardiogenic shock patients, who are a quarter of that.
SUBSETS = ("all", "cardiogenic", "septic")


def subset_filter(name: str) -> "pl.Expr | None":
    """the cohort predicate for a --subset value"""
    if name == "all":
        return None
    if name == "septic":
        # the coded-septic denominator, which is also exactly the set Seymour
        # and Bhavani are assigned over -- a septic flow figure coloured by
        # either phenotype has no meaning outside it
        return pl.col("sepsis_cohort").fill_null(False)
    # the union of the two readings of cardiogenic shock. Extracting the union
    # covers both the Zweck panel and the by-definition panel from one pass.
    return (pl.col("shock_type") == "cardiogenic") | pl.col(
        "cardiac_origin"
    ).fill_null(False)


def grid_home(paths, site: str, offset: float, subset: str = "all") -> pathlib.Path:
    # The subset is part of the path. Without it a later --subset all run would
    # find the cardiogenic shards already present, skip them, and silently
    # build a whole-cohort figure out of a quarter of the cohort.
    tag = f"{'m' if offset < 0 else 'p'}{abs(int(round(offset))):03d}"
    return (
        paths.scratch_home
        / "trajectory"
        / paths.model_key
        / subset
        / site
        / tag
    )


# =============================================================== extraction


def _clear_datasets_cache(logger) -> None:
    """
    drop the HuggingFace `datasets` cache for the grid point just finished

    Safe because nothing reads it. The transform cotorra maps is a closure, so
    the fingerprint changes every call and `datasets` rebuilds from scratch
    regardless of what is on disk -- the cache is write-only by construction.

    Failures are logged and swallowed: a cache that could not be cleared is a
    disk-space problem for the next grid point, not a reason to lose the
    embeddings this one just produced.
    """
    cache = os.environ.get("HF_DATASETS_CACHE")
    if not cache:
        return
    home = pathlib.Path(cache)
    if not home.is_dir():
        return
    freed = 0
    for child in home.iterdir():
        try:
            if child.is_dir():
                freed += sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                shutil.rmtree(child, ignore_errors=True)
            else:
                freed += child.stat().st_size
                child.unlink()
        except Exception as exc:
            # Exception, not OSError. Two runs sharing this cache will fight
            # over it -- one deletes an arrow file the other has open and
            # Windows raises PermissionError from inside shutil, past
            # ignore_errors. A cache that will not clear is a disk-space
            # problem for the next grid point, never a reason to lose the
            # embeddings this one just produced.
            #
            # The real fix is not to run two extracts at once; this only stops
            # that mistake from destroying hours of work.
            logger.warning(f"could not clear {child.name} from the datasets cache: {exc}")
    if freed:
        logger.info(f"  cleared {freed / 2**30:.1f} GiB of datasets cache")


def extract_grid(paths, args, logger) -> None:
    """winnow and embed every patient at every grid point"""
    from clifcs.tokens import SPLITS as TOKEN_SPLITS  # noqa: F401

    # was a dynamic __import__ of "02_model_embedding", whose leading digit
    # made it unimportable by name; it is a module now
    from clifcs import embedding as embedding_module

    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    onset_file = paths.cohort_home / "cs_onset.parquet"
    shared = ("tokenizer.yaml", "subject_splits.parquet", "tokens_times.parquet")

    info = embedding_module.describe_model(paths, logger)
    extraction_cfg = OmegaConf.load(CONFIG_HOME / "extraction.yaml")
    extraction_cfg.extract.batch_size = int(args.batch_size)

    predicate = subset_filter(args.subset)
    if predicate is not None:
        before = cohort.height
        cohort = cohort.filter(predicate)
        logger.info(
            f"subset {args.subset!r}: {cohort.height:,} of {before:,} patients "
            f"({cohort.height / max(before, 1):.1%})"
        )

    for site in paths.sites:
        keep = set(cohort.filter(pl.col("site") == site)["hospitalization_id"].to_list())
        source = paths.processed(site)
        for offset in offsets(args):
            destination = grid_home(paths, site, offset, args.subset)
            out_file = destination / "embeddings.parquet"
            if out_file.is_file() and not args.force:
                logger.info(f"[{site} t={offset:+.0f}h] exists, skipping")
                continue
            destination.mkdir(parents=True, exist_ok=True)
            for name in shared:
                if not (destination / name).is_file():
                    shutil.copy2(source / name, destination / name)

            GridWinnower(
                winnowing_cfg=CONFIG_HOME / "winnowing_onset.yaml",
                processed_data_home=destination,
                onset_parquet=onset_file,
                offset_hours=offset,
                min_tokens_past=args.min_tokens,
            ).save_all()

            cfg_path = destination / "extraction-resolved.yaml"
            OmegaConf.save(extraction_cfg, cfg_path)
            t0 = time.perf_counter()
            extractor = embedding_module.build_extractor(
                paths, destination, destination, cfg_path, keep="last"
            )
            extractor.extract()

            frames = []
            for split in SPLITS:
                features = sorted(
                    destination.glob(f"features-{split}*-{paths.model_home.name}.parquet")
                )
                inference = destination / f"{split}_for_inference.parquet"
                if not features or not inference.is_file():
                    continue
                vectors = pl.concat(
                    [pl.read_parquet(f, columns=["features"]) for f in features]
                )
                keys = pl.read_parquet(inference, columns=["subject_id", "last_valid"])
                if vectors.height != keys.height:
                    raise RuntimeError(
                        f"[{site} t={offset:+.0f}h/{split}] {vectors.height} feature "
                        f"rows vs {keys.height} inference rows"
                    )
                frames.append(keys.with_columns(vectors["features"]))

            if not frames:
                logger.warning(f"[{site} t={offset:+.0f}h] nobody reaches this point")
                continue
            frame = pl.concat(frames).rename({"subject_id": "hospitalization_id"})
            frame = frame.filter(pl.col("hospitalization_id").is_in(list(keep)))
            matrix = np.vstack(frame["features"].to_list()).astype(np.float32)
            (
                frame.drop("features")
                .with_columns(site=pl.lit(site), offset_hours=pl.lit(float(offset)))
                .hstack(
                    pl.from_numpy(
                        matrix, schema=[f"emb_{i:04d}" for i in range(matrix.shape[1])]
                    )
                )
                .write_parquet(out_file)
            )
            logger.info(
                f"[{site} t={offset:+.0f}h] {frame.height} patients in "
                f"{time.perf_counter() - t0:.0f}s -> {out_file.parent.name}"
            )

            # The raw shards are large and fully regenerable -- and so is every
            # other working file in here. Only embeddings.parquet is wanted
            # afterwards; the winnowed inputs the extractor consumed
            # (tokens_times, *_for_inference, the per-split copies) are about
            # 820 MB per grid point, which is 32 GB across a 39-point grid
            # against 0.26 GB of output. Deleting them does not affect resume,
            # which keys on embeddings.parquet.
            for f in destination.iterdir():
                if f.is_file() and f.name != out_file.name:
                    try:
                        f.unlink()
                    except OSError as exc:
                        logger.warning(f"could not remove {f.name}: {exc}")

            # AND the datasets cache, which is the thing that actually fills the
            # disk. cotorra maps a closure, so `datasets` cannot fingerprint the
            # transform, warns that it "will consider it to be different from
            # the previous calls and recompute everything", and writes a fresh
            # build every single time. The cache is therefore never once read
            # back -- every lookup misses by construction -- and it accumulates
            # about a gigabyte per grid point.
            #
            # clifcs.config already redirects it out of the home directory into
            # scratch, which fixed where it lands and not how much of it there
            # is: 119 GB over four days killed an earlier run at grid point 23
            # of 26, and this one died at 28 shards of 78 with 28.6 GB of cache
            # against 0.68 GB of actual output. Bounding it to one grid point is
            # what makes 09 runnable on a normal disk.
            _clear_datasets_cache(logger)
    logger.info(f"extraction done (embedding dim {info['embedding_dim']})")


# ================================================================ projection


def counterfactual_home(paths, site: str, variant: str) -> pathlib.Path:
    """working directory for one site and one hypothetical next token"""
    return paths.scratch_home / "counterfactual" / paths.model_key / site / variant


def extract_counterfactuals(paths, args, patients, logger) -> pl.DataFrame:
    """
    embed each chosen patient prefix with one extra hypothetical token on it

    Only the patients being plotted are embedded. The grid extract pays for
    every patient at every timepoint because the field needs them; this needs
    one timepoint and a handful of patients, so it is a few dozen sequences
    rather than a hundred thousand and takes seconds rather than hours.

    The baseline variant is the unmodified prefix, re-embedded here rather than
    taken from the grid. That is deliberate: it is the control the hypotheticals
    are measured against, and reading it from a different code path would make
    any difference between them partly a difference in how they were produced.
    """
    from clifcs import embedding as embedding_module
    from clifcs.counterfactual import INTERVENTIONS, append_token
    from clifcs.tokens import SPLITS as TOKEN_SPLITS  # noqa: F401
    from clifcs.tokens import vocabulary

    names, _ = vocabulary(str(paths.tokenizer_yaml))
    wanted = {str(p) for p in patients}
    extraction_cfg = OmegaConf.load(CONFIG_HOME / "extraction.yaml")
    extraction_cfg.extract.batch_size = int(args.batch_size)
    shared = ("tokenizer.yaml", "subject_splits.parquet", "tokens_times.parquet")

    variants = [("baseline", "Actual prefix (no extra token)", None)]
    variants += [(k, label, res) for k, label, res in INTERVENTIONS]

    collected = []
    for site in paths.sites:
        source = paths.processed(site)
        if not (source / "tokens_times.parquet").is_file():
            continue

        # winnow this site once, at the hour the branch leaves from
        staging = counterfactual_home(paths, site, "_staging")
        staging.mkdir(parents=True, exist_ok=True)
        for name in shared:
            shutil.copy2(source / name, staging / name)
        GridWinnower(
            winnowing_cfg=CONFIG_HOME / "winnowing_onset.yaml",
            processed_data_home=staging,
            onset_parquet=paths.cohort_home / "cs_onset.parquet",
            offset_hours=float(args.cf_hours),
            min_tokens_past=args.min_tokens,
        ).save_all()

        base = {}
        for split in SPLITS:
            path = staging / f"{split}_for_inference.parquet"
            if not path.is_file():
                continue
            frame = pl.read_parquet(path)
            frame = frame.filter(pl.col("subject_id").cast(pl.String).is_in(list(wanted)))
            if frame.height:
                base[split] = frame
        if not base:
            logger.info(f"[{site}] none of the chosen patients reach +{args.cf_hours:g}h")
            continue
        logger.info(
            f"[{site}] {sum(f.height for f in base.values())} chosen patient(s) "
            f"at +{args.cf_hours:g}h"
        )

        for key, label, resolver in variants:
            home = counterfactual_home(paths, site, key)
            home.mkdir(parents=True, exist_ok=True)
            for name in shared:
                shutil.copy2(source / name, home / name)

            wrote = 0
            for split, frame in base.items():
                if resolver is None:
                    out = frame
                else:
                    modified = append_token(frame, names, resolver)
                    if not modified.height:
                        continue
                    # keep every column the extractor expects, overriding only
                    # the three the hypothetical changes
                    out = (
                        frame.drop(
                            [c for c in ("tokens_past", "s_elapsed_past", "last_valid")
                             if c in frame.columns]
                        )
                        .join(
                            modified.drop("from_quantile", "to_quantile"),
                            on="subject_id",
                            how="inner",
                        )
                    )
                if not out.height:
                    continue
                out.write_parquet(home / f"{split}_for_inference.parquet")
                wrote += out.height
            if wrote == 0:
                logger.info(f"  {key}: does not apply to any chosen patient")
                continue

            cfg_path = home / "extraction-resolved.yaml"
            OmegaConf.save(extraction_cfg, cfg_path)
            extractor = embedding_module.build_extractor(
                paths, home, home, cfg_path, keep="last"
            )
            extractor.extract()

            for split in SPLITS:
                features = sorted(
                    home.glob(f"features-{split}*-{paths.model_home.name}.parquet")
                )
                inference = home / f"{split}_for_inference.parquet"
                if not features or not inference.is_file():
                    continue
                vectors = pl.concat(
                    [pl.read_parquet(f, columns=["features"]) for f in features]
                )
                keys = pl.read_parquet(inference, columns=["subject_id"])
                if vectors.height != keys.height:
                    logger.warning(
                        f"  {site}/{key}/{split}: {vectors.height} features vs "
                        f"{keys.height} rows; skipped"
                    )
                    continue
                matrix = np.vstack(vectors["features"].to_list()).astype(np.float32)
                collected.append(
                    keys.with_columns(site=pl.lit(site), variant=pl.lit(key))
                    .rename({"subject_id": "hospitalization_id"})
                    .hstack(
                        pl.from_numpy(
                            matrix,
                            schema=[f"emb_{i:04d}" for i in range(matrix.shape[1])],
                        )
                    )
                )
            logger.info(f"  {key}: embedded {wrote} sequence(s)")
            for f in home.glob("features-*.parquet"):
                f.unlink()
            _clear_datasets_cache(logger)

    if not collected:
        return pl.DataFrame()
    return pl.concat(collected, how="diagonal_relaxed")


def load_grid(paths, args, logger) -> pl.DataFrame:
    parts = [
        pl.read_parquet(grid_home(paths, site, offset, args.subset) / "embeddings.parquet")
        for site in paths.sites
        for offset in offsets(args)
        if (grid_home(paths, site, offset, args.subset) / "embeddings.parquet").is_file()
    ]
    if not parts:
        raise FileNotFoundError("no trajectory embeddings; run --stage extract first")
    frame = pl.concat(parts, how="diagonal_relaxed")
    logger.info(
        f"{frame.height} patient-timepoints, "
        f"{frame['hospitalization_id'].n_unique()} patients, "
        f"{frame['offset_hours'].n_unique()} grid points"
    )
    return frame


#: which 2-d projection the figures are drawn in.
#:
#: UMAP and PaCMAP are both computed and both saved to
#: trajectory_projections.parquet every run; this only selects which one is
#: plotted. Set once from --projection in main() and read by `proj_cols` and
#: `proj_labels`, so no figure can end up with one projection's coordinates
#: under another's axis labels -- which has happened here before.
PROJECTION = "pacmap"


def proj_cols() -> tuple[str, str]:
    """the coordinate columns the figures should read"""
    return f"{PROJECTION}_1", f"{PROJECTION}_2"


def proj_labels() -> tuple[str, str]:
    """axis labels matching `proj_cols`"""
    if PROJECTION == "pca":
        return "PC1", "PC2"
    pretty = {"umap": "UMAP", "pacmap": "PaCMAP"}[PROJECTION]
    return f"{pretty}-1", f"{pretty}-2"


def project(frame: pl.DataFrame, seed: int, logger) -> pl.DataFrame:
    """
    one projection over every patient-timepoint, so positions are comparable

    UMAP and PaCMAP are both fitted, on the same PCA scores with the same
    neighbour count, and both are saved. PaCMAP is what the figures use by
    default: it is built to preserve global structure alongside local
    neighbourhoods, which matters more here than anywhere else in the pipeline
    because the question being asked of this layout is about *motion* across
    it, and a projection that tears neighbouring regions apart invents
    displacement that the 1024-d space does not contain.
    """
    import pacmap
    import umap

    columns = sorted(c for c in frame.columns if c.startswith("emb_"))
    scaled = StandardScaler().fit_transform(
        frame.select(columns).to_numpy().astype(np.float64)
    )
    pca = PCA(n_components=PCA_COMPONENTS, random_state=seed)
    scores = pca.fit_transform(scaled)
    logger.info(
        f"PCA: PC1 {pca.explained_variance_ratio_[0]:.1%}, "
        f"first {PCA_COMPONENTS} {pca.explained_variance_ratio_.sum():.1%}"
    )
    logger.info("UMAP: building neighbour graph")
    umap_coords = umap.UMAP(
        n_neighbors=30, min_dist=0.3, n_components=2, random_state=seed
    ).fit_transform(scores)
    logger.info("PaCMAP: building neighbour graph")
    pacmap_coords = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=30,
        MN_ratio=0.5,
        FP_ratio=2.0,
        apply_pca=False,
        random_state=seed,
    ).fit_transform(np.ascontiguousarray(scores, dtype=np.float32))
    return frame.with_columns(
        umap_1=pl.Series(umap_coords[:, 0]),
        umap_2=pl.Series(umap_coords[:, 1]),
        pacmap_1=pl.Series(pacmap_coords[:, 0]),
        pacmap_2=pl.Series(pacmap_coords[:, 1]),
        pca_1=pl.Series(scores[:, 0]),
        pca_2=pl.Series(scores[:, 1]),
    )


def displacement_field(
    frame: pl.DataFrame, x: str, y: str, bins: int = 26, smooth: float = 1.0
):
    """
    mean displacement per grid cell, from consecutive timepoints of one patient

    Each patient contributes one vector per consecutive pair, placed at the
    earlier position. Cells holding fewer than a handful of vectors are left
    empty rather than drawn from noise.
    """
    ordered = frame.sort("hospitalization_id", "offset_hours")
    dx = (
        ordered.with_columns(
            next_x=pl.col(x).shift(-1).over("hospitalization_id"),
            next_y=pl.col(y).shift(-1).over("hospitalization_id"),
            next_t=pl.col("offset_hours").shift(-1).over("hospitalization_id"),
        )
        .drop_nulls(["next_x", "next_y", "next_t"])
        .with_columns(
            vx=pl.col("next_x") - pl.col(x), vy=pl.col("next_y") - pl.col(y)
        )
    )
    px, py = dx[x].to_numpy(), dx[y].to_numpy()
    vx, vy = dx["vx"].to_numpy(), dx["vy"].to_numpy()

    lo_x, hi_x = np.percentile(frame[x].to_numpy(), [0.5, 99.5])
    lo_y, hi_y = np.percentile(frame[y].to_numpy(), [0.5, 99.5])
    edges_x = np.linspace(lo_x, hi_x, bins + 1)
    edges_y = np.linspace(lo_y, hi_y, bins + 1)

    counts, _, _ = np.histogram2d(px, py, bins=[edges_x, edges_y])
    sum_x, _, _ = np.histogram2d(px, py, bins=[edges_x, edges_y], weights=vx)
    sum_y, _, _ = np.histogram2d(px, py, bins=[edges_x, edges_y], weights=vy)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = np.where(counts >= 5, sum_x / counts, 0.0)
        mean_y = np.where(counts >= 5, sum_y / counts, 0.0)
    mean_x = gaussian_filter(mean_x, smooth)
    mean_y = gaussian_filter(mean_y, smooth)

    centres_x = 0.5 * (edges_x[:-1] + edges_x[1:])
    centres_y = 0.5 * (edges_y[:-1] + edges_y[1:])
    return centres_x, centres_y, mean_x.T, mean_y.T, counts.T, len(px)


# ================================================================== plotting


def plot_scatter(frame, x, y, column, colors, order, title, label, out, logger,
                 continuous=False):
    figure, (ax, legend_ax) = plt.subplots(
        1, 2, figsize=(6.4, 4.6), gridspec_kw={"width_ratios": [4.0, 1.25], "wspace": 0.04}
    )
    legend_ax.axis("off")
    rng = np.random.default_rng(0)
    shuffled = frame.with_columns(_o=pl.Series(rng.permutation(frame.height))).sort("_o")
    px, py = shuffled[x].to_numpy(), shuffled[y].to_numpy()
    size = 1.2 if frame.height > 60000 else 2.5

    if continuous:
        cmap = continuous_cmap()
        values = shuffled[column].to_numpy()
        points = ax.scatter(px, py, s=size, c=values, cmap=cmap, linewidths=0,
                            alpha=0.8, rasterized=True)
        bar = figure.colorbar(points, cax=legend_ax.inset_axes([0.08, 0.3, 0.16, 0.6]))
        bar.set_label(label, color=INK_SECONDARY, fontsize=7)
        bar.ax.tick_params(labelsize=6, color=GRID, labelcolor=INK_SECONDARY)
        bar.outline.set_visible(False)
    else:
        # One scatter per category, in `order`, so the LAST level named is drawn
        # on top. For Zweck that is I -> II -> III, which puts the smallest and
        # most distinctive group above the largest: III cardiometabolic is ~5%
        # of the cohort and I non-congested is ~70%, so a single pooled scatter
        # buries III under points drawn after it and the figure reads as though
        # III were absent.
        #
        # Shuffling is kept WITHIN each category and dropped between them. The
        # within-category shuffle is what stops row order in the parquet showing
        # up as structure; the across-category shuffle was what made the layering
        # unpredictable.
        # Both sides through key_string, for the reason spelled out in
        # plot_flow: it lowercases, so a raw-vs-canonical comparison silently
        # matches nothing.
        canon = np.array([key_string(v) for v in shuffled[column].to_list()])
        palette = {key_string(k): v for k, v in colors.items()}
        levels = list(order) if order else sorted(set(canon))
        seen = set()
        for level in levels:
            key = key_string(level)
            if key in seen:
                continue
            seen.add(key)
            mask = canon == key
            if not mask.any():
                continue
            ax.scatter(px[mask], py[mask], s=size,
                       c=palette.get(key, "#cccccc"), linewidths=0,
                       alpha=0.8, rasterized=True, zorder=2)
        counts = frame[column].cast(pl.String).value_counts()
        lookup = dict(zip(counts[column].to_list(), counts["count"].to_list()))
        legend_ax.legend(
            handles=[
                plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                           markerfacecolor=colors[k], markeredgecolor="none",
                           label=f"{k}  n={lookup.get(k, 0):,}")
                for k in order
            ],
            loc="upper left", bbox_to_anchor=(0, 1), labelcolor=INK_SECONDARY,
            fontsize=7, handletextpad=0.6, borderpad=0,
        )

    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel(proj_labels()[0])
    ax.set_ylabel(proj_labels()[1])
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out}")


def cell_coherence(frame: pl.DataFrame, x: str, y: str, bins: int = 26) -> float:
    """median over cells of |sum of displacements| / sum of |displacements|"""
    ordered = frame.sort("hospitalization_id", "offset_hours")
    dx = (
        ordered.with_columns(
            next_x=pl.col(x).shift(-1).over("hospitalization_id"),
            next_y=pl.col(y).shift(-1).over("hospitalization_id"),
        )
        .drop_nulls(["next_x", "next_y"])
        .with_columns(vx=pl.col("next_x") - pl.col(x), vy=pl.col("next_y") - pl.col(y))
    )
    px, py = dx[x].to_numpy(), dx[y].to_numpy()
    vx, vy = dx["vx"].to_numpy(), dx["vy"].to_numpy()
    speed = np.hypot(vx, vy)
    edges_x = np.linspace(*np.percentile(frame[x].to_numpy(), [0.5, 99.5]), bins + 1)
    edges_y = np.linspace(*np.percentile(frame[y].to_numpy(), [0.5, 99.5]), bins + 1)
    counts, _, _ = np.histogram2d(px, py, [edges_x, edges_y])
    sum_x, _, _ = np.histogram2d(px, py, [edges_x, edges_y], weights=vx)
    sum_y, _, _ = np.histogram2d(px, py, [edges_x, edges_y], weights=vy)
    sum_speed, _, _ = np.histogram2d(px, py, [edges_x, edges_y], weights=speed)
    ok = counts >= 5
    if not ok.any():
        return float("nan")
    return float(
        np.median(np.hypot(sum_x[ok], sum_y[ok]) / np.maximum(sum_speed[ok], 1e-9))
    )


def apply_correction(paths, frame: pl.DataFrame, method: str, anchor: str, logger):
    """
    put every grid point through the affine map 05 fitted on the anchor rows

    The site effect dominates this geometry -- 97% of a patient's neighbours in
    the anchor space carry their own site -- so a flow field drawn on raw
    embeddings is substantially a picture of three hospitals charting
    differently, with the motion of interest underneath it. Correcting first is
    what makes the remaining displacement readable as physiology.

    THE ORDER MATTERS AND IS EASY TO GET WRONG. The map is affine on the
    1024-d embedding, so it has to be applied before the projection: correcting
    PaCMAP coordinates instead would be applying a linear map to a non-linear
    embedding of the thing that actually needed correcting, which is
    meaningless. PCA is therefore refit downstream, on corrected vectors.

    The map is LOADED, not refitted. 05 fits it on one row per patient at the
    anchor; refitting here would fit it on up to 26 rows per patient and weight
    the result by whose grid is complete, and the map would then change with
    every change to the grid.

    Read the result with 04 in mind. The sites separate because each emits
    token types the others never do, so an affine map closes the linear channel
    rather than making the representations exchangeable -- 05 reports a
    non-linear probe that still recovers site after LEACE has driven the linear
    one to chance. "Site effect removed" means removed to first order.
    """
    if method == "none":
        return frame

    from clifcs.correction import AffineCorrection

    source = (
        paths.patient_level_home / "corrections" / f"anchor-{anchor}_{method}.npz"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} not found. Fit and save it first:\n"
            f"    python code/05_site_correction.py --anchor {anchor} --save-map\n"
            "09 deliberately does not fit its own: the map has to be the one "
            "measured at the anchor, or the correction is not the one 05 "
            "reports numbers for."
        )
    correction = AffineCorrection.load(source)
    columns = sorted(c for c in frame.columns if c.startswith("emb_"))
    x = frame.select(columns).to_numpy().astype(np.float64)
    sites = np.asarray(frame["site"].to_list())
    corrected = correction.apply(x, sites)
    logger.info(
        f"applied the {correction.name} map from {source.name} to "
        f"{frame.height:,} grid points ({len(columns)}-d)"
    )
    return pl.concat(
        [
            frame.select([c for c in frame.columns if not c.startswith("emb_")]),
            pl.from_numpy(
                corrected.astype(np.float32),
                schema=[f"emb_{i:04d}" for i in range(corrected.shape[1])],
            ),
        ],
        how="horizontal",
    )


#: how a patient qualifies as cardiogenic, as a LABEL rather than a cohort
#:
#: Splitting the cohort fitted a separate field to each definition, so the two
#: figures differed both in who was in them and in what the arrows were computed
#: from -- and nothing on the page said which difference you were looking at.
#: One field over the union, with the definition as the hue, holds the arrows
#: fixed and varies only the colour. It also puts the overlap on the figure:
#: the two definitions agree on about a quarter of their union, which is the
#: single most important caveat about comparing them and was previously a line
#: in a log.
CS_DEFINITION_LEVELS: tuple[str, ...] = ("both", "ICD code only", "treatment only")


def cardiogenic_definition() -> pl.Expr:
    """which of the two readings of cardiogenic shock a patient satisfies"""
    coded = pl.col("shock_type") == "cardiogenic"
    treated = pl.col("cardiac_origin").fill_null(False)
    return (
        pl.when(coded & treated).then(pl.lit("both"))
        .when(coded).then(pl.lit("ICD code only"))
        .when(treated).then(pl.lit("treatment only"))
        .otherwise(pl.lit(None))
    )


def plot_cardiogenic_flow(frame, px, py, home, paths, args, logger) -> None:
    """
    the headline figures: latent flow among cardiogenic shock patients

    One flow field over the union of the two definitions, drawn twice with
    different hues beneath it -- Zweck phenotype, and which definition the
    patient satisfies. Both carry the arrows, because a panel without them
    answers a different question from the one this section is asking.

    The field is computed on the union rather than per definition for the
    reason in CS_DEFINITION_LEVELS: holding the arrows fixed is what makes two
    panels comparable.
    """
    needed = {"shock_type", "cardiac_origin"}
    if not needed <= set(frame.columns):
        logger.warning(
            f"cannot label cardiogenic definition; missing "
            f"{sorted(needed - set(frame.columns))}"
        )
        return

    labelled = frame.with_columns(cs_definition=cardiogenic_definition())
    part = labelled.filter(pl.col("cs_definition").is_not_null())
    if part.height == 0:
        logger.warning("no cardiogenic patients to draw")
        return

    n_patients = part["hospitalization_id"].n_unique()
    breakdown = (
        part.group_by("cs_definition")
        .agg(patients=pl.col("hospitalization_id").n_unique())
        .sort("patients", descending=True)
    )
    logger.info(f"cardiogenic flow: {n_patients:,} patients in the union")
    for row in breakdown.iter_rows(named=True):
        logger.info(
            f"    {row['cs_definition']:<16} {row['patients']:>6,} patients "
            f"({row['patients'] / n_patients:.1%} of the union)"
        )

    subtitle = (
        f"{n_patients:,} patients meeting either definition of cardiogenic "
        f"shock; one field over all of them, mean displacement between "
        f"consecutive {args.step_hours:g}h timepoints, cells with <5 vectors blank"
    )

    plot_flow(
        part, px, py,
        home / "trajectory_flow_cardiogenic_phenotype.png",
        f"{paths.model_key}: latent flow, cardiogenic shock by Zweck phenotype",
        subtitle,
        logger, labels=proj_labels(),
        colour_by={
            "key": "cs_phenotype",
            "colors": phenotype_colors(),
            "order": list(PHENOTYPES),
        },
    )

    plot_flow(
        part, px, py,
        home / "trajectory_flow_cardiogenic_definition.png",
        f"{paths.model_key}: latent flow, cardiogenic shock by how it was identified",
        subtitle + "; the ICD code and the treatment-based flag agree on only "
        "part of this set, so they are not two independent findings",
        logger, labels=proj_labels(),
        colour_by={
            "key": "cs_definition",
            "colors": husl_colors(CS_DEFINITION_LEVELS),
            "order": list(CS_DEFINITION_LEVELS),
        },
    )


#: the two sepsis phenotype systems, as flow underlays
#:
#: Both are REFITS, not transfers: neither paper publishes centroids, so the
#: clusters are this cohort's and the labels come from matching each centroid
#: profile to the published description. Read the flow panels as "do these four
#: classes occupy different parts of the space and move differently", never as
#: a replication of Seymour or Bhavani.
#:
#: The denominator is the coded septic-shock flag, which fails in both
#: directions and fails differently at each site. That is the weakest part of
#: both systems and it is upstream of every panel here.
SEPSIS_SCHEMES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("seymour_phenotype", SEYMOUR_LABELS, "Seymour 2019 alpha/beta/gamma/delta"),
    ("bhavani_phenotype", BHAVANI_LABELS, "Bhavani 2022 group A-D"),
)


def plot_septic_flow(frame, px, py, home, paths, args, logger) -> None:
    """
    latent flow among coded septic shock, by each sepsis phenotype system

    The septic counterpart of plot_cardiogenic_flow, and built the same way:
    ONE field over the whole septic set, drawn repeatedly with different hues
    beneath it. Holding the arrows fixed is what lets two panels be compared --
    fitting a field per phenotype would vary both the population and the
    arrows, with nothing saying which difference you were looking at.

    Requires a grid extracted with --subset septic. Running it against the
    cardiogenic grid would draw the 23% of septic patients who also met a
    cardiac-origin criterion -- patients on inotropes or with a high troponin,
    which is atypical for septic shock -- and present them as septic shock.
    """
    if "sepsis_cohort" not in frame.columns:
        logger.warning("no sepsis_cohort column; skipping the septic panels")
        return
    part = frame.filter(pl.col("sepsis_cohort").fill_null(False))
    if part.height == 0:
        logger.warning("no coded-septic patients in this grid")
        return

    n_patients = part["hospitalization_id"].n_unique()
    total = frame["hospitalization_id"].n_unique()
    logger.info(
        f"septic flow: {n_patients:,} coded-septic patients of {total:,} in the grid"
    )
    if n_patients < 0.5 * total:
        logger.warning(
            f"  only {n_patients / max(total, 1):.0%} of this grid is "
            "coded-septic -- these panels are drawn over a subset selected for "
            "something else. Re-extract with --subset septic."
        )

    subtitle = (
        f"{n_patients:,} patients carrying a septic shock code; one field over "
        f"all of them, mean displacement between consecutive "
        f"{args.step_hours:g}h timepoints, cells with <5 vectors blank"
    )

    for key, labels, scheme in SEPSIS_SCHEMES:
        if key not in part.columns:
            logger.info(f"  {key} absent, no panel")
            continue
        labelled = part.filter(pl.col(key).is_not_null())
        if labelled.height == 0:
            logger.info(f"  {key} entirely null, no panel")
            continue
        assigned = labelled["hospitalization_id"].n_unique()
        plot_flow(
            labelled, px, py,
            home / f"trajectory_flow_septic_{key}.png",
            f"{paths.model_key}: latent flow, septic shock by {scheme}",
            subtitle
            + f"; {assigned:,} of {n_patients:,} carry a {key.split('_')[0]} label",
            logger, labels=proj_labels(),
            colour_by={
                "key": key,
                "colors": husl_colors(labels),
                "order": list(labels),
            },
        )

    # Zweck over the septic set too. Zweck is a cardiogenic shock phenotyping
    # applied here to septic patients, which is not what it was derived for --
    # it is drawn because the comparison is the point of the whole pipeline,
    # not because the label is claimed to mean the same thing.
    if "cs_phenotype" in part.columns:
        plot_flow(
            part.filter(pl.col("cs_phenotype").is_not_null()), px, py,
            home / "trajectory_flow_septic_zweck.png",
            f"{paths.model_key}: latent flow, septic shock by Zweck phenotype",
            subtitle + "; Zweck was derived in cardiogenic shock, so this is a "
            "comparison and not a claim that the label transfers",
            logger, labels=proj_labels(),
            colour_by={
                "key": "cs_phenotype",
                "colors": phenotype_colors(),
                "order": list(PHENOTYPES),
            },
        )

    if "in_hospital_mortality" in part.columns:
        plot_flow(
            part, px, py,
            home / "trajectory_flow_septic_mortality.png",
            f"{paths.model_key}: latent flow, septic shock by outcome",
            subtitle,
            logger, labels=proj_labels(),
            colour_by={
                "key": "in_hospital_mortality",
                "colors": mortality_colors(),
                "order": [False, True],
                "names": {False: "survived", True: "died in hospital"},
            },
        )


def attach_state(paths, frame: pl.DataFrame, logger) -> pl.DataFrame:
    """physiology at each grid point, for use as a flow-field underlay"""
    cache = paths.patient_level_home / "trajectory_state.parquet"
    wanted = set(
        frame.select("hospitalization_id", "offset_hours").unique().rows()
    )
    if cache.is_file():
        cached = pl.read_parquet(cache)
        if set(cached.select("hospitalization_id", "offset_hours").rows()) == wanted:
            logger.info(f"reusing {cache.name}")
            return frame.join(cached, on=["hospitalization_id", "offset_hours"],
                              how="left")

    onset = pl.read_parquet(paths.cohort_home / "cs_onset.parquet").select(
        "hospitalization_id", "site", "cs_onset_dttm"
    )
    parts = []
    for site in paths.sites:
        grid = (
            frame.select("hospitalization_id", "site", "offset_hours")
            .filter(pl.col("site") == site)
            .join(onset.drop("site"), on="hospitalization_id", how="inner")
            .with_columns(
                grid_dttm=pl.col("cs_onset_dttm")
                + pl.duration(seconds=(pl.col("offset_hours") * 3600).cast(pl.Int64))
            )
            .select("hospitalization_id", "offset_hours", "grid_dttm")
        )
        logger.info(f"[{site}] physiology at {grid.height:,} grid points")
        parts.append(trajectory_state(paths.raw_subset(site), grid))

    state = pl.concat(parts, how="diagonal_relaxed")
    state.write_parquet(cache)
    logger.info(f"wrote {cache.name}")
    return frame.join(state, on=["hospitalization_id", "offset_hours"], how="left")


def plot_state_flow_single(frame, x, y, name, out, title, logger) -> None:
    """
    the flow field over ONE physiological variable, on its own axes

    plot_state_flow packs every available variable into a 4-column grid, which
    is useful as an overview and unreadable as evidence: eleven panels at
    3.6 inches each means the colour ramp spans a few hundred pixels and the
    streamlines are hairlines. One variable per figure gives each the whole
    canvas, and the report puts them in a tab strip so they are still compared
    by flicking between them rather than by scrolling.

    The flow field is identical in every one of these -- only the underlay
    changes -- so a difference between two of them is a difference in the
    underlay and never in the arrows.
    """
    label, _ = VARIABLES[name]
    values = frame[name].to_numpy().astype(float)
    finite = np.isfinite(values)
    if finite.sum() < 100:
        logger.info(f"  {name}: {int(finite.sum())} finite values, no panel")
        return

    cx, cy, vx, vy, counts, _ = displacement_field(frame, x, y)
    figure, ax = plt.subplots(figsize=(6.0, 5.2))

    mesh = ax.hexbin(
        frame[x].to_numpy()[finite], frame[y].to_numpy()[finite],
        C=values[finite], reduce_C_function=np.mean, mincnt=5, gridsize=52,
        cmap=continuous_cmap(), linewidths=0, alpha=0.8, zorder=1,
    )
    cells = mesh.get_array()
    good = cells.compressed() if hasattr(cells, "compressed") else cells
    good = good[np.isfinite(good)]
    if good.size:
        mesh.set_clim(*np.nanpercentile(good, [2, 98]))
    bar = figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label(label, color=INK_SECONDARY, fontsize=8)
    bar.ax.tick_params(labelsize=7, color=GRID, labelcolor=INK_SECONDARY)
    bar.outline.set_visible(False)

    # arrows above the underlay, explicitly -- see plot_flow
    ax.streamplot(
        cx, cy, np.ma.array(vx, mask=counts < 5), np.ma.array(vy, mask=counts < 5),
        color=INK_PRIMARY, linewidth=0.7, density=1.4, arrowsize=0.9,
        arrowstyle="->", zorder=3,
    )

    ax.set_title(title, loc="left", pad=8)
    xlabel, ylabel = proj_labels()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.6)
    figure.tight_layout()
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out.name}")


def plot_state_flow(frame, x, y, out, title, logger, columns: int = 4):
    """
    the flow field over each physiological variable in turn

    The arrows are identical in every panel; only the underlay changes. Where
    the colour gradient runs along the arrows, that variable is what is
    changing as patients move.
    """
    available = [v for v in VARIABLES if v in frame.columns
                 and frame[v].is_not_null().sum() > frame.height * 0.2]
    if not available:
        logger.warning("no state variables available to underlay")
        return pl.DataFrame()

    cx, cy, vx, vy, counts, _ = displacement_field(frame, x, y)
    rows = int(np.ceil(len(available) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 3.3 * rows),
                                squeeze=False)
    cmap = continuous_cmap()

    px_all, py_all = frame[x].to_numpy(), frame[y].to_numpy()
    ordered = frame.sort("hospitalization_id", "offset_hours")
    step = (
        ordered.with_columns(
            nx=pl.col(x).shift(-1).over("hospitalization_id"),
            ny=pl.col(y).shift(-1).over("hospitalization_id"),
        ).drop_nulls(["nx", "ny"])
    )
    sx, sy = step[x].to_numpy(), step[y].to_numpy()
    svx = (step["nx"] - step[x]).to_numpy()
    svy = (step["ny"] - step[y]).to_numpy()

    report = []
    for position, name in enumerate(available):
        ax = axes[position // columns][position % columns]
        label, worse_high = VARIABLES[name]
        values = frame[name].to_numpy().astype(float)
        finite = np.isfinite(values)
        if finite.sum() < 100:
            ax.axis("off")
            continue
        points = ax.hexbin(
            px_all[finite], py_all[finite], C=values[finite],
            reduce_C_function=np.mean, mincnt=5, gridsize=48,
            cmap=cmap, linewidths=0,
        )
        cells = points.get_array()
        good = cells.compressed() if hasattr(cells, "compressed") else cells
        lo, hi = np.nanpercentile(good[np.isfinite(good)], [2, 98])
        points.set_clim(lo, hi)
        masked_x = np.ma.array(vx, mask=counts < 5)
        masked_y = np.ma.array(vy, mask=counts < 5)
        # zorder above the underlay rather than relying on call order: a
        # streamplot is a LineCollection plus FancyArrowPatches, and collections
        # and patches do not share a default zorder, so the arrows could land
        # under the hue they are meant to be read against.
        ax.streamplot(cx, cy, masked_x, masked_y, color=INK_PRIMARY,
                      linewidth=0.55, density=1.1, arrowsize=0.75,
                      arrowstyle="->", zorder=3)

        alignment, cells = gradient_alignment(sx, sy, svx, svy,
                                              step[name].to_numpy().astype(float))
        # two separate questions, and they can disagree: how much of the
        # variable a patient's position accounts for, and whether patients
        # travel along that variable's gradient
        encoded = position_encodes(px_all, py_all, values)
        report.append({"variable": name, "label": label,
                       "worse_is_high": worse_high,
                       "position_r2": encoded,
                       "flow_gradient_cosine": alignment, "cells": cells})
        arrow = "up-gradient" if alignment > 0.05 else (
            "down-gradient" if alignment < -0.05 else "no trend")
        ax.set_title(
            label + "\n"
            + f"position explains R² {encoded:.2f}   ·   flow {arrow} "
              f"(cos {alignment:+.2f})",
            loc="left", fontsize=7.2, pad=4,
        )
        bar = figure.colorbar(points, ax=ax, fraction=0.046, pad=0.02)
        bar.ax.tick_params(labelsize=5.5, color=GRID, labelcolor=INK_SECONDARY)
        bar.outline.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    for position in range(len(available), rows * columns):
        axes[position // columns][position % columns].axis("off")

    figure.suptitle(title, x=0.02, ha="left", fontsize=10, y=0.998)
    figure.text(0.02, 0.004,
                "identical flow field in every panel; only the underlay changes. "
                "R² is how much of the variable a patient's position accounts for "
                "(30-NN, held out); cos is the mean cosine between the flow and the "
                "variable's spatial gradient, +1 moving toward higher values. The two "
                "answer different questions: where patients sit, and where they go",
                fontsize=7, color=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0.02, 1, 0.97))
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out.name}")
    return pl.DataFrame(report)


def colourings(frame: pl.DataFrame, paths) -> list[dict]:
    """
    the ways of labelling a patient-timepoint, as one spec per figure

    Kept in one place because the same list drives the standalone UMAPs, the
    per-patient trajectory overlays and the report's tabs, and they have to
    agree or a tab will label a figure it is not showing.
    """
    specs = [
        {"key": "site", "label": "contributing site", "kind": "categorical",
         "colors": site_colors(paths.sites), "order": list(paths.sites)},
        {"key": "in_hospital_mortality", "label": "in-hospital mortality",
         "kind": "categorical", "colors": mortality_colors(),
         "order": [False, True],
         "names": {False: "survived", True: "died in hospital"}},
        {"key": "cs_phenotype", "label": "cardiogenic shock phenotype",
         "kind": "categorical", "colors": phenotype_colors(),
         "order": list(PHENOTYPES)},
        {"key": "ever_imv", "label": "ever invasively ventilated",
         "kind": "categorical", "colors": mortality_colors(),
         "order": [False, True], "names": {False: "never", True: "ventilated"}},
        {"key": "n_vasopressors", "label": "vasoactive agents running",
         "kind": "continuous"},
        {"key": "map", "label": "MAP (mmHg)", "kind": "continuous"},
        {"key": "hours_from_admission", "label": "hours from admission",
         "kind": "continuous"},
        {"key": "icu_los_days", "label": "ICU length of stay (days)",
         "kind": "continuous"},
    ]
    return [s for s in specs if s["key"] in frame.columns]


def plot_colouring_grid(frame, specs, out, title, logger, columns: int = 4):
    """every labelling of the trajectory cloud, as one small-multiple sheet"""
    rows = int(np.ceil(len(specs) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.9 * columns, 3.5 * rows),
                                squeeze=False)
    rng = np.random.default_rng(0)
    order = rng.permutation(frame.height)
    shuffled = frame.with_columns(_o=pl.Series(order)).sort("_o")
    _px, _py = proj_cols()
    px, py = shuffled[_px].to_numpy(), shuffled[_py].to_numpy()
    size = 1.1 if frame.height > 60000 else 2.2

    for position, spec in enumerate(specs):
        ax = axes[position // columns][position % columns]
        if spec["kind"] == "continuous":
            values = shuffled[spec["key"]].to_numpy().astype(float)
            finite = np.isfinite(values)
            if finite.sum() < 100:
                ax.axis("off")
                continue
            # Hexbin of the local mean rather than 75,000 individual points.
            # Overplotted, a high-variance variable reads as confetti whatever
            # the colormap, because neighbouring patients differ far more than
            # neighbouring regions do; averaging within a cell shows the
            # spatial pattern, which is the thing being asked about.
            points = ax.hexbin(
                px[finite], py[finite], C=values[finite],
                reduce_C_function=np.mean, mincnt=5, gridsize=52,
                cmap=continuous_cmap(), linewidths=0,
            )
            cells = points.get_array()
            lo, hi = np.nanpercentile(cells.compressed() if hasattr(cells, "compressed")
                                      else cells[np.isfinite(cells)], [2, 98])
            points.set_clim(lo, hi)
            bar = figure.colorbar(points, ax=ax, fraction=0.046, pad=0.02)
            bar.ax.tick_params(labelsize=5.5, color=GRID, labelcolor=INK_SECONDARY)
            bar.outline.set_visible(False)
        else:
            colours = {key_string(k): v for k, v in spec["colors"].items()}
            values = [key_string(v) for v in shuffled[spec["key"]].to_list()]
            ax.scatter(px, py, s=size,
                       c=[colours.get(v, "#cccccc") for v in values],
                       linewidths=0, alpha=0.8, rasterized=True, zorder=1)
            names = spec.get("names", {})
            ax.legend(
                handles=[
                    plt.Line2D([], [], marker="o", linestyle="none", markersize=4,
                               markerfacecolor=spec["colors"][k],
                               markeredgecolor="none",
                               label=str(names.get(k, k))[:28])
                    for k in spec["order"] if k in spec["colors"]
                ],
                loc="upper left", bbox_to_anchor=(0, -0.01), fontsize=5.8,
                labelcolor=INK_SECONDARY, handletextpad=0.5, borderpad=0,
                ncol=2,
            )
        ax.set_title(spec["label"], loc="left", fontsize=8, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    for position in range(len(specs), rows * columns):
        axes[position // columns][position % columns].axis("off")
    figure.suptitle(title, x=0.02, ha="left", fontsize=10, y=0.998)
    figure.tight_layout(rect=(0, 0.01, 1, 0.97))
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out.name}")


def plot_single_colouring(frame, spec, out, title, logger):
    """
    one labelling, one full-size panel

    The small-multiple sheet is for comparing labellings against each other;
    this is for reading one of them properly, and is what the report tabs
    between.
    """
    figure, (ax, legend_ax) = plt.subplots(
        1, 2, figsize=(6.6, 5.0),
        gridspec_kw={"width_ratios": [4.3, 1.15], "wspace": 0.04},
    )
    legend_ax.axis("off")
    rng = np.random.default_rng(0)
    shuffled = frame.with_columns(_o=pl.Series(rng.permutation(frame.height))).sort("_o")
    _px, _py = proj_cols()
    px, py = shuffled[_px].to_numpy(), shuffled[_py].to_numpy()

    if spec["kind"] == "continuous":
        values = shuffled[spec["key"]].to_numpy().astype(float)
        finite = np.isfinite(values)
        mesh = ax.hexbin(
            px[finite], py[finite], C=values[finite], reduce_C_function=np.mean,
            mincnt=5, gridsize=64, cmap=continuous_cmap(), linewidths=0,
        )
        cells = mesh.get_array()
        good = cells.compressed() if hasattr(cells, "compressed") else cells
        lo, hi = np.nanpercentile(good[np.isfinite(good)], [2, 98])
        mesh.set_clim(lo, hi)
        bar = figure.colorbar(mesh, cax=legend_ax.inset_axes([0.10, 0.30, 0.16, 0.58]))
        bar.set_label(spec["label"], color=INK_SECONDARY, fontsize=7)
        bar.ax.tick_params(labelsize=6, color=GRID, labelcolor=INK_SECONDARY)
        bar.outline.set_visible(False)
        legend_ax.text(
            0.10, 0.22,
            "hexagon = local mean\nof the patients in it",
            transform=legend_ax.transAxes, fontsize=6, color=INK_SECONDARY,
            va="top",
        )
    else:
        colours = {key_string(k): v for k, v in spec["colors"].items()}
        values = [key_string(v) for v in shuffled[spec["key"]].to_list()]
        ax.scatter(px, py, s=1.6, c=[colours.get(v, "#cccccc") for v in values],
                   linewidths=0, alpha=0.8, rasterized=True)
        counts = frame[spec["key"]].to_list()
        tally = {}
        for v in counts:
            tally[key_string(v)] = tally.get(key_string(v), 0) + 1
        names = spec.get("names", {})
        legend_ax.legend(
            handles=[
                plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                           markerfacecolor=spec["colors"][k], markeredgecolor="none",
                           label=f"{names.get(k, k)}  n={tally.get(key_string(k), 0):,}")
                for k in spec["order"] if k in spec["colors"]
            ],
            loc="upper left", bbox_to_anchor=(0, 1), fontsize=7,
            labelcolor=INK_SECONDARY, handletextpad=0.6, borderpad=0,
        )

    ax.set_title(title, loc="left", pad=8, fontsize=9)
    ax.set_xlabel(proj_labels()[0])
    ax.set_ylabel(proj_labels()[1])
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out.name}")


def path_quality(frame, patient, cx, cy, vx, vy, counts, extent) -> float:
    """
    how well one patient's path illustrates the averaged flow field

    WHAT THIS SCORES, because the first version got it wrong. It used the mean
    cosine between each step and the local field vector and nothing else.
    Cosine is scale-free, so a patient who teleports across the whole map in
    roughly the right direction scored a perfect 1.0, identically to one
    gliding along a streamline -- and a step spanning twenty cells was judged
    against the field in the cell it happened to start in. The criterion was
    therefore selecting FOR big jumps, which is exactly what the figures showed.

    Four things now, all of which a usable illustration needs:

      steps       at least four usable ones. A mean over two is not a tendency.
      direction   the FRACTION of steps pointing with the field, not the mean
                  cosine -- one long well-aligned step cannot carry a path that
                  otherwise wanders.
      scale       steps comparable to the field's own resolution. A step longer
                  than two grid cells is not following the field, it is
                  crossing it, and the local vector cannot describe it.
      smoothness  net displacement over path length. A path that doubles back
                  covers ground without going anywhere, and reads as noise
                  however well each individual step lines up.

    Returns 0 for a path that fails any of them, so sorting descending puts the
    usable ones first and never promotes a teleporter.
    """
    _px, _py = proj_cols()
    path = frame.filter(pl.col("hospitalization_id") == patient).sort("offset_hours")
    if path.height < 5:
        return 0.0
    tx, ty = path[_px].to_numpy(), path[_py].to_numpy()
    dx, dy = np.diff(tx), np.diff(ty)
    lengths = np.hypot(dx, dy)

    # the field's own resolution: displacement_field uses `bins` cells across
    # the 0.5-99.5 percentile span, so one cell is that span over `bins`
    cell = extent / 26.0
    if cell <= 0:
        return 0.0

    ix = np.clip(np.searchsorted(cx, tx[:-1]) - 1, 0, len(cx) - 1)
    iy = np.clip(np.searchsorted(cy, ty[:-1]) - 1, 0, len(cy) - 1)

    cosines, steps = [], []
    for sx, sy, length, i, j in zip(dx, dy, lengths, ix, iy):
        if counts[j, i] < 5:
            continue
        fx, fy = vx[j, i], vy[j, i]
        field = np.hypot(fx, fy)
        if length < 1e-9 or field < 1e-9 or not np.isfinite(field):
            continue
        cosines.append((sx * fx + sy * fy) / (length * field))
        steps.append(length)

    if len(cosines) < 4:
        return 0.0
    cosines = np.asarray(cosines)
    steps = np.asarray(steps)

    # LENGTH FILTERS APPLY TO EVERY STEP, not just the scoreable ones. Steps
    # landing in empty cells are skipped above because there is no vector to
    # compare them against -- but skipping them from the length test too meant
    # a path could jump right across the plot through empty space and still be
    # judged on the handful of small steps it took afterwards. That is what put
    # a patient with one enormous opening stride at the top of the ranking.
    if np.median(lengths) > 1.5 * cell:
        return 0.0
    if lengths.max() > 0.15 * extent:
        return 0.0
    # and most of the path has to be somewhere the field actually exists, or
    # "follows the field" is a statement about a minority of it
    if len(cosines) / len(lengths) < 0.7:
        return 0.0

    aligned = float(np.mean(cosines > 0.3))
    if aligned < 0.6:
        return 0.0

    travelled = float(steps.sum())
    net = float(np.hypot(tx[-1] - tx[0], ty[-1] - ty[0]))
    directness = net / travelled if travelled > 0 else 0.0

    return aligned * float(np.mean(cosines)) * (0.5 + 0.5 * directness)


def pick_flow_followers(frame: pl.DataFrame, x, y, logger, n: int = 8,
                        min_points: int = 8, pool: int = 900, seed: int = 7):
    """
    patients whose own path follows the flow field, spread across the groups

    Two stages. A stratified random pool first, so every (outcome, phenotype)
    combination is represented and the result is not simply the eight patients
    with the longest paths; then the best-aligned from each stratum.

    `min_points` is higher than the plain sampler's: a path needs enough steps
    for "follows the field" to mean anything, and six grid points is three
    steps at the coarsest useful resolution.
    """
    cx, cy, vx, vy, counts, _ = displacement_field(frame, x, y)
    extent = float(max(np.ptp(frame[x].to_numpy()), np.ptp(frame[y].to_numpy())))

    eligible = (
        frame.group_by("hospitalization_id")
        .agg(points=pl.len())
        .filter(pl.col("points") >= min_points)
    )
    if eligible.height == 0:
        return []

    keys = [c for c in ("in_hospital_mortality", "cs_phenotype") if c in frame.columns]
    table = (
        frame.select("hospitalization_id", *keys)
        .unique(subset=["hospitalization_id"])
        .join(eligible, on="hospitalization_id", how="inner")
        .sort("hospitalization_id")
    )

    rng = np.random.default_rng(seed)
    strata: list[list] = []
    if keys:
        for (_, group) in table.group_by(keys, maintain_order=True):
            ids = group["hospitalization_id"].to_list()
            rng.shuffle(ids)
            strata.append(ids[: max(1, pool // max(len(keys), 1))])
    else:
        ids = table["hospitalization_id"].to_list()
        rng.shuffle(ids)
        strata = [ids[:pool]]

    # Score every candidate, keep only those that clear the hard filters, and
    # take the best from each stratum. Taking the best available per stratum
    # regardless of score is what put teleporting patients in the figures: a
    # stratum where nothing qualifies should contribute nothing, not its least
    # bad member.
    chosen: list = []
    for bucket in strata:
        scored = [
            (path_quality(frame, pid, cx, cy, vx, vy, counts, extent), pid)
            for pid in bucket
        ]
        scored = [(s, p) for s, p in scored if s > 0.0]
        if scored:
            scored.sort(reverse=True)
            chosen.append(scored[0])

    chosen.sort(reverse=True)
    if chosen:
        logger.info(
            "  path quality of the chosen: "
            + ", ".join(f"{s:.2f}" for s, _ in chosen[:n])
        )
    return [pid for _, pid in chosen[:n]]


def pick_patients(frame: pl.DataFrame, n: int = 8, min_points: int = 5, seed: int = 7):
    """
    patients with enough timepoints to show a path, spread across the groups

    A uniform draw reproduces the cohort's imbalance, which for eight panels
    meant eight survivors and seven of one phenotype -- technically a random
    sample and useless as an illustration, because the reader cannot tell a
    phenotype-I path from a phenotype-III one when only phenotype I is shown.

    Sampling is therefore stratified on (died, Zweck phenotype) and drawn
    round-robin across the strata, so the panel shows variety. These are
    examples, never evidence: nothing is measured from them, and the averaged
    field beside them is what carries any claim.
    """
    eligible = (
        frame.group_by("hospitalization_id")
        .agg(points=pl.len())
        .filter(pl.col("points") >= min_points)
    )
    if eligible.height == 0:
        return []

    keys = [c for c in ("in_hospital_mortality", "cs_phenotype") if c in frame.columns]
    table = (
        frame.select("hospitalization_id", *keys)
        .unique(subset=["hospitalization_id"])
        .join(eligible, on="hospitalization_id", how="inner")
        .sort("hospitalization_id")
    )

    rng = np.random.default_rng(seed)
    if not keys:
        ids = table["hospitalization_id"].to_list()
        picked = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
        return [ids[int(i)] for i in sorted(picked)]

    strata = []
    for (_, group) in table.group_by(keys, maintain_order=True):
        ids = group["hospitalization_id"].to_list()
        rng.shuffle(ids)
        strata.append(ids)
    rng.shuffle(strata)

    chosen: list = []
    while len(chosen) < n and any(strata):
        for bucket in strata:
            if bucket and len(chosen) < n:
                chosen.append(bucket.pop())
    return chosen


def plot_patient_path_single(frame, patient, index, out, title, logger) -> None:
    """
    ONE patient per figure, so the HTML can lay them out however it likes

    Previously eight patients shared one 4-column canvas, which made each panel
    about three inches wide: the hour labels overlapped whatever the spacing
    rule, because the problem was the panel size and not the rule. A figure per
    patient gives each the whole canvas and lets the report arrange them.

    Labels are placed by walking the path and skipping any point closer than a
    fixed fraction of the panel to the last label placed, ends always kept. At
    full size that leaves them legible even where a patient hovers in one spot.
    """
    _px, _py = proj_cols()
    path = frame.filter(pl.col("hospitalization_id") == patient).sort("offset_hours")
    if path.height < 2:
        return

    figure, ax = plt.subplots(figsize=(5.4, 5.0))
    px, py = frame[_px].to_numpy(), frame[_py].to_numpy()
    ax.scatter(px, py, s=1.0, c="#dcdbd6", linewidths=0, alpha=0.8,
               rasterized=True, zorder=1)

    tx, ty = path[_px].to_numpy(), path[_py].to_numpy()
    ax.plot(tx, ty, "-", color=INK_PRIMARY, linewidth=1.3, zorder=3, alpha=0.9)
    ax.scatter(tx, ty, s=34, c=INK_PRIMARY, zorder=4, linewidths=0)

    # ONE label per end, carrying both the word and the hour. Annotating "end"
    # and "+48h" separately put two strings on the same coordinate, which was
    # the last collision left after the spacing rule was fixed -- the rule
    # cannot help when the figure deliberately labels one point twice.
    hours_all = path["offset_hours"].to_list()
    ax.annotate(f"start  {hours_all[0]:+.0f}h", (tx[0], ty[0]), fontsize=8,
                color=INK_PRIMARY, xytext=(7, 7), textcoords="offset points",
                zorder=5, fontweight="bold")
    ax.annotate(f"end  {hours_all[-1]:+.0f}h", (tx[-1], ty[-1]), fontsize=8,
                color=INK_PRIMARY, xytext=(7, -13), textcoords="offset points",
                zorder=5, fontweight="bold")

    # Collision test against EVERY label already placed, not just the previous
    # one. Walking in time order and comparing to the last placed label fails
    # for a path that wanders off and comes back: the last label is somewhere
    # else by then, so the point gets labelled on top of one from six hours
    # earlier. That is what left "+12h", "+30h", "+48h" and "end" stacked on a
    # single dot.
    span = max(np.ptp(px), np.ptp(py), 1e-6)
    gap = 0.055 * span
    placed: list[tuple[float, float]] = [(tx[0], ty[0]), (tx[-1], ty[-1])]
    hours = path["offset_hours"].to_list()
    for position, (hour, a, b) in enumerate(zip(hours, tx, ty)):
        if position in (0, len(hours) - 1):
            continue  # already carried by the start/end label
        if any(np.hypot(a - qx, b - qy) < gap for qx, qy in placed):
            continue
        ax.annotate(f"{hour:+.0f}h", (a, b), fontsize=7, color=INK_SECONDARY,
                    xytext=(5, 4), textcoords="offset points", zorder=5)
        placed.append((a, b))

    site = path["site"][0]
    died = bool(path["in_hospital_mortality"][0])
    phenotype = path["cs_phenotype"][0] if "cs_phenotype" in path.columns else None
    bits = [str(site), f"{path.height} timepoints",
            "died" if died else "survived"]
    if phenotype is not None:
        bits.append(str(phenotype))
    ax.set_title(f"{title}\n{' / '.join(bits)}", loc="left", pad=8, fontsize=9)
    xlabel, ylabel = proj_labels()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out.name} ({path.height} timepoints)")


def plot_patient_paths_plain(frame, patients, out, title, logger, columns: int = 4):
    """
    one panel per patient: where that patient goes, and in what order

    The smoothed field answers where patients go on average, which is not the
    same claim as any individual making that journey -- a flow field through a
    cell where half the patients move each way shows an arrow, and the
    coherence number beside it is the only hint. These panels are the
    unsmoothed version: black dots at each grid point, joined in time order,
    annotated with the hour relative to shock onset.

    The cloud behind is deliberately neutral grey. Colouring it invites reading
    the path against whatever the colour means, and the point here is only the
    path -- the coloured variants are the `patient_paths_<label>.png` figures.

    First and last points are always labelled, then every third, because a
    patient who barely moves stacks every label on one spot and a patient who
    moves a lot does not need eleven of them.
    """
    if not patients:
        logger.warning("no patient has enough timepoints to draw a path")
        return
    rows = int(np.ceil(len(patients) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.3 * columns, 3.5 * rows), squeeze=False
    )
    _px, _py = proj_cols()
    px, py = frame[_px].to_numpy(), frame[_py].to_numpy()

    for position, patient in enumerate(patients):
        ax = axes[position // columns][position % columns]
        ax.scatter(px, py, s=0.8, c="#dcdbd6", linewidths=0, alpha=0.8,
                   rasterized=True, zorder=1)

        path = frame.filter(
            pl.col("hospitalization_id") == patient
        ).sort("offset_hours")
        tx, ty = path[_px].to_numpy(), path[_py].to_numpy()
        ax.plot(tx, ty, "-", color=INK_PRIMARY, linewidth=1.1, zorder=3,
                alpha=0.9)
        ax.scatter(tx, ty, s=24, c=INK_PRIMARY, zorder=4, linewidths=0)

        # Label the ends always, then any point far enough from the last label
        # placed. An every-third-point rule collides badly here: a patient who
        # barely moves stacks "+30h" on "+48h" at the same coordinates, which
        # is unreadable and worse than no label. The threshold is a fraction of
        # the panel's own extent so it travels across projections.
        hours = path["offset_hours"].to_list()
        # Spacing threshold from the PANEL's extent, not the patient's. Using
        # the path's own extent is self-defeating: a patient who barely moves
        # has a tiny span, so the threshold shrinks with them and the labels
        # collide anyway -- which is precisely the case that needed fixing.
        span = max(np.ptp(px), np.ptp(py), 1e-6)
        gap = 0.05 * span
        last = None
        for index, (hour, a, b) in enumerate(zip(hours, tx, ty)):
            ends = index in (0, len(hours) - 1)
            if not ends and last is not None and np.hypot(a - last[0], b - last[1]) < gap:
                continue
            ax.annotate(f"{hour:+.0f}h", (a, b), fontsize=6, color=INK_PRIMARY,
                        xytext=(4, 3), textcoords="offset points", zorder=5)
            last = (a, b)

        site = path["site"][0]
        died = bool(path["in_hospital_mortality"][0])
        phenotype = path["cs_phenotype"][0] if "cs_phenotype" in path.columns else None
        bits = [str(site), f"{path.height} points",
                "died" if died else "survived"]
        if phenotype is not None:
            bits.append(str(phenotype))
        ax.set_title(" / ".join(bits), loc="left", fontsize=7, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    for position in range(len(patients), rows * columns):
        axes[position // columns][position % columns].axis("off")

    figure.suptitle(title, x=0.02, ha="left", fontsize=10, y=0.998)
    figure.text(
        0.02, 0.004,
        "each dot is one 6-hourly grid point, joined in time order and labelled "
        "with hours from shock onset; grey is every other patient-timepoint. "
        "These are individuals, not the averaged field -- a cell where half the "
        "patients move one way and half the other still draws a streamline",
        fontsize=7, color=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.97))
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out.name} ({len(patients)} patients)")


def plot_patient_paths(frame, spec, patients, out, title, logger):
    """
    one panel per patient: their path through the cloud, over one labelling

    The path is drawn in black over the coloured cloud and each point is
    annotated with its hour relative to shock onset, so the direction of travel
    is readable rather than inferred from the arrow of a smoothed field.
    """
    figure, axes = plt.subplots(1, len(patients),
                               figsize=(3.1 * len(patients), 3.4), squeeze=False)
    _px, _py = proj_cols()
    px, py = frame[_px].to_numpy(), frame[_py].to_numpy()

    if spec["kind"] == "continuous":
        values = frame[spec["key"]].to_numpy().astype(float)
        finite = np.isfinite(values)
    else:
        colours = {key_string(k): v for k, v in spec["colors"].items()}
        values = [colours.get(key_string(v), "#cccccc")
                  for v in frame[spec["key"]].to_list()]

    for position, patient in enumerate(patients):
        ax = axes[0][position]
        if spec["kind"] == "continuous":
            mesh = ax.hexbin(
                px[finite], py[finite], C=values[finite],
                reduce_C_function=np.mean, mincnt=5, gridsize=46,
                cmap=continuous_cmap(), linewidths=0, alpha=0.8,
            )
            cells = mesh.get_array()
            good = cells.compressed() if hasattr(cells, "compressed") else cells
            lo, hi = np.nanpercentile(good[np.isfinite(good)], [2, 98])
            mesh.set_clim(lo, hi)
        else:
            ax.scatter(px, py, s=0.9, c=values, linewidths=0, alpha=0.8,
                       rasterized=True)

        path = frame.filter(pl.col("hospitalization_id") == patient).sort("offset_hours")
        tx, ty = path[_px].to_numpy(), path[_py].to_numpy()
        ax.plot(tx, ty, "-", color=INK_PRIMARY, linewidth=1.0, zorder=3)
        ax.scatter(tx, ty, s=22, c=INK_PRIMARY, zorder=4, linewidths=0)
        # labelling every point collides where a patient barely moves; the
        # ends plus every third step still show the direction of travel
        hours = path["offset_hours"].to_list()
        for index, (hour, a, b) in enumerate(zip(hours, tx, ty)):
            if index not in (0, len(hours) - 1) and index % 3:
                continue
            ax.annotate(f"{hour:+.0f}h", (a, b), fontsize=5.2, color=INK_PRIMARY,
                        xytext=(4, 3), textcoords="offset points", zorder=5)
        site = path["site"][0]
        died = bool(path["in_hospital_mortality"][0])
        ax.set_title(
            f"{site} · {path.height} points · "
            + ("died" if died else "survived"),
            loc="left", fontsize=7, pad=3,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    figure.suptitle(title, x=0.01, ha="left", fontsize=9.5, y=1.02)
    figure.text(0.01, -0.04,
                "black path is one patient through the cloud; labels are hours "
                "relative to shock onset",
                fontsize=6.5, color=INK_SECONDARY)
    figure.tight_layout()
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out.name}")


def interpolate_into_projection(real_emb, real_xy, new_emb, k: int = 15):
    """
    place a new embedding in an existing PaCMAP layout, by its neighbours

    PaCMAP has no `transform`, so a point that was not in the fit cannot be
    placed exactly. This takes the k nearest real points in the 1024-d space and
    returns the inverse-distance-weighted mean of their 2-d coordinates.

    THE POSITION IS AN INTERPOLATION. It cannot land anywhere the real points do
    not already cover, so a hypothetical that moves the representation somewhere
    genuinely new gets dragged back toward the nearest familiar region, and the
    length of the arrow is NOT a distance in the latent space. It shows
    direction and rough neighbourhood, which is what the figure is for; the
    quantitative version of the same question is the cosine in the table.

    Distances are computed in the full 1024-d space rather than in the 2-d
    layout, so neighbours are neighbours in the representation and not merely
    in the picture.
    """
    from sklearn.neighbors import NearestNeighbors

    finder = NearestNeighbors(n_neighbors=min(k, len(real_emb))).fit(real_emb)
    distances, indices = finder.kneighbors(new_emb)
    weights = 1.0 / np.maximum(distances, 1e-9)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.einsum("ij,ijk->ik", weights, real_xy[indices])


def plot_counterfactual_paths(
    frame, cf_frame, patients, home, paths, args, logger
) -> pl.DataFrame:
    """
    one figure per patient: the real path, and where one more token would move it

    The path and its hypothetical branches on the same axes. Branches leave the
    patient position at the chosen hour and are coloured tab10, one hue per
    intervention, with the baseline re-embedding drawn as a hollow marker so the
    reader can see how far the hypotheticals sit from an unmodified prefix taken
    through the identical code path.

    Returns the per-patient, per-intervention displacement so the figures can be
    read against a number rather than by eye.
    """
    import matplotlib.cm as cm

    from clifcs.counterfactual import INTERVENTIONS

    if cf_frame.is_empty():
        logger.warning("no counterfactual embeddings; nothing to draw")
        return pl.DataFrame()

    _px, _py = proj_cols()
    emb_cols = sorted(c for c in frame.columns if c.startswith("emb_"))
    cf_cols = sorted(c for c in cf_frame.columns if c.startswith("emb_"))
    if emb_cols != cf_cols:
        logger.warning("counterfactual embeddings have a different width; skipped")
        return pl.DataFrame()

    real_emb = frame.select(emb_cols).to_numpy().astype(np.float64)
    real_xy = frame.select(_px, _py).to_numpy().astype(np.float64)
    placed = interpolate_into_projection(
        real_emb, real_xy, cf_frame.select(cf_cols).to_numpy().astype(np.float64)
    )
    cf = cf_frame.select("hospitalization_id", "variant").with_columns(
        cf_x=pl.Series(placed[:, 0]), cf_y=pl.Series(placed[:, 1])
    )

    order = ["baseline"] + [k for k, _, _ in INTERVENTIONS]
    labels = {"baseline": "Actual prefix (control)"}
    labels.update({k: label for k, label, _ in INTERVENTIONS})
    palette = {
        key: cm.tab10(i % 10) for i, key in enumerate(k for k in order if k != "baseline")
    }

    rows = []
    for index, patient in enumerate(patients, start=1):
        branches = cf.filter(pl.col("hospitalization_id") == patient)
        if branches.height == 0:
            continue
        path = frame.filter(
            pl.col("hospitalization_id") == patient
        ).sort("offset_hours")
        if path.height < 2:
            continue

        figure, ax = plt.subplots(figsize=(5.8, 5.2))
        ax.scatter(real_xy[:, 0], real_xy[:, 1], s=1.0, c="#dcdbd6",
                   linewidths=0, alpha=0.8, rasterized=True, zorder=1)

        tx, ty = path[_px].to_numpy(), path[_py].to_numpy()
        ax.plot(tx, ty, "-", color=INK_PRIMARY, linewidth=1.2, zorder=3, alpha=0.9)
        ax.scatter(tx, ty, s=26, c=INK_PRIMARY, zorder=4, linewidths=0)

        # the branch point: the patient position at the hour the hypotheticals
        # leave from, which is the grid point nearest args.cf_hours
        hours = np.asarray(path["offset_hours"].to_list(), dtype=float)
        at = int(np.argmin(np.abs(hours - float(args.cf_hours))))
        bx, by = float(tx[at]), float(ty[at])
        ax.scatter([bx], [by], s=110, facecolors="none", edgecolors=INK_PRIMARY,
                   linewidths=1.4, zorder=5)
        ax.annotate(f"{hours[at]:+.0f}h", (bx, by), fontsize=8, fontweight="bold",
                    color=INK_PRIMARY, xytext=(8, 8),
                    textcoords="offset points", zorder=6)

        lookup = dict(zip(branches["variant"].to_list(),
                          zip(branches["cf_x"].to_list(), branches["cf_y"].to_list())))
        for key in order:
            if key not in lookup:
                continue
            qx, qy = lookup[key]
            if key == "baseline":
                ax.scatter([qx], [qy], s=52, facecolors="none",
                           edgecolors=INK_SECONDARY, linewidths=1.2, zorder=6,
                           label=labels[key])
                continue
            colour = palette[key]
            ax.annotate(
                "", xy=(qx, qy), xytext=(bx, by), zorder=6,
                arrowprops=dict(arrowstyle="->", color=colour, linewidth=1.6,
                                shrinkA=2, shrinkB=2),
            )
            ax.scatter([qx], [qy], s=46, c=[colour], linewidths=0, zorder=7,
                       label=labels[key])
            rows.append({
                "hospitalization_id": patient,
                "intervention": key,
                "label": labels[key],
                "displacement": float(np.hypot(qx - bx, qy - by)),
            })

        ax.legend(loc="upper left", fontsize=6.5, frameon=False, markerscale=1.1,
                  handletextpad=0.5, borderpad=0.2)
        site = path["site"][0]
        died = bool(path["in_hospital_mortality"][0])
        ax.set_title(
            f"{paths.model_key}: patient {index}, hypothetical next events\n"
            f"{site} / {path.height} timepoints / "
            + ("died" if died else "survived"),
            loc="left", pad=8, fontsize=9,
        )
        xlabel, ylabel = proj_labels()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{ylabel}  -- branch positions are interpolated; "
                      "arrow length is not a distance")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        figure.tight_layout()
        out = home / f"counterfactual_path_{index:02d}.png"
        save_figure(figure, out)
        plt.close(figure)
        logger.info(f"wrote {out.name} ({branches.height} branches)")

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def plot_flow(frame, x, y, out, title, subtitle, logger, labels=None,
              colour_by=None):
    """streamplot of the displacement field over the point cloud"""
    cx, cy, vx, vy, counts, n_vectors = displacement_field(frame, x, y)

    # Coherence per cell: |sum of vectors| / sum of |vectors|. Near 1 the
    # patients in a cell move the same way; near 0 they cancel and the
    # streamlines are an integration drawn through noise. Pure noise over
    # n vectors gives roughly 1/sqrt(n), so a few tenths is already real
    # structure -- but the streamplot renders the directional part only, and
    # looks far more orderly than the underlying scatter, so the number
    # belongs on the figure.
    coherence = cell_coherence(frame, x, y)

    figure, ax = plt.subplots(figsize=(6.0, 5.2))

    # The point cloud goes UNDER the arrows. `colour_by` turns it from flat grey
    # into the hue the field is meant to be read against -- Zweck phenotype,
    # mortality -- so one figure answers both "where do these patients sit" and
    # "which way do they move", instead of asking the reader to hold two
    # side-by-side figures in their head.
    #
    # Categories are drawn in `order` so the last named lands on top, and zorder
    # is explicit rather than left to call order: a streamplot is a
    # LineCollection plus FancyArrowPatches, which do not share a default
    # zorder, so the arrows can otherwise end up beneath the scatter.
    px, py = frame[x].to_numpy(), frame[y].to_numpy()
    if colour_by:
        # key_string lowercases as well as normalising booleans, so BOTH sides
        # of every comparison have to go through it. Canonicalising the colour
        # dict and the level while leaving the series raw matched nothing at
        # all: "I non-congested" never equals "i non-congested", every mask
        # came out empty, and the figure rendered as arrows over a blank panel
        # with no legend. That is the failure key_string's own docstring
        # describes, in the other direction.
        values = np.array(
            [key_string(v) for v in frame[colour_by["key"]].to_list()]
        )
        colours = {key_string(k): v for k, v in colour_by["colors"].items()}
        for level in colour_by["order"]:
            mask = values == key_string(level)
            if not mask.any():
                logger.warning(
                    f"  {out.name}: no points for {level!r}; the hue underlay "
                    "will be missing that level"
                )
                continue
            pretty = colour_by.get("names", {}).get(level, level)
            ax.scatter(
                px[mask], py[mask], s=2.0,
                c=colours.get(key_string(level), "#cccccc"),
                linewidths=0, alpha=0.8, rasterized=True, zorder=1,
                label=f"{pretty} (n={int(mask.sum()):,})",
            )
        ax.legend(
            loc="upper left", fontsize=6, frameon=False,
            markerscale=3, handletextpad=0.4, borderpad=0.2,
        )
    else:
        ax.scatter(px, py, s=1.0, c="#dcdbd6", linewidths=0,
                   rasterized=True, alpha=0.8, zorder=1)

    mask = counts < 5
    vx = np.ma.array(vx, mask=mask)
    vy = np.ma.array(vy, mask=mask)
    ax.streamplot(
        cx, cy, vx, vy,
        color=INK_PRIMARY,
        linewidth=0.7,
        density=1.5,
        arrowsize=0.9,
        arrowstyle="->",
        zorder=3,
    )

    ax.set_title(title, loc="left", pad=8)
    # labelled from the columns actually plotted: this said "UMAP" on the PCA
    # panel, which is the one figure whose whole purpose is to not be the UMAP
    xlabel, ylabel = labels or (x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    figure.text(
        0.02, -0.01,
        f"{subtitle}; directional coherence {coherence:.2f} "
        f"(1 = patients in a cell all move alike, 0 = they cancel)",
        fontsize=7, color=INK_SECONDARY,
    )
    save_figure(figure, out)
    plt.close(figure)
    logger.info(
        f"wrote {out.name} ({n_vectors:,} displacement vectors, "
        f"{int((counts >= 5).sum())} populated cells, coherence {coherence:.3f})"
    )
    return coherence


# =================================================================== driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "plot", "all"), default="all")
    parser.add_argument("--model", default=None)
    parser.add_argument("--start-hours", type=float, default=-24.0)
    parser.add_argument("--end-hours", type=float, default=48.0)
    parser.add_argument("--step-hours", type=float, default=6.0)
    parser.add_argument(
        "--counterfactuals",
        action="store_true",
        help="also embed each plotted patient with one extra hypothetical token "
        "on the end -- pressor up or down, MAP up or down, off the ventilator, "
        "inotrope started -- and draw where that moves them. Needs the GPU, but "
        "only for the handful of patients being plotted, so it is seconds.",
    )
    parser.add_argument(
        "--cf-hours",
        type=float,
        default=24.0,
        help="the hour the hypothetical branches leave from (default +24h)",
    )
    parser.add_argument(
        "--subset",
        choices=SUBSETS,
        default="cardiogenic",
        help="which patients to extract the grid for. Defaults to cardiogenic "
        "-- the union of the ICD code and the cardiac_origin flag, about a "
        "quarter of the cohort -- because that is what the flow figures are "
        "for and the full cohort is 4x the GPU time. Shards are kept in "
        "separate directories per subset, so switching does not reuse them.",
    )
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--projection",
        choices=("pacmap", "umap", "pca"),
        default="pacmap",
        help="which projection the figures are drawn in; all are computed and "
        "saved regardless",
    )
    parser.add_argument(
        "--correction",
        choices=("none", "leace", "site-standardized"),
        default="leace",
        help="apply an affine site correction to every grid point before "
        "projecting. Defaults to leace: site dominates this geometry at 97%% "
        "neighbour purity, so an uncorrected flow field is substantially a "
        "picture of the hospitals. The map is loaded from the one 05 fitted "
        "at the anchor -- run 05 --save-map first. `none` draws the raw field.",
    )
    parser.add_argument(
        "--correction-anchor",
        choices=ANCHORS,
        default="onset",
        help="which anchor's fitted map to apply; onset matches the grid, "
        "which is centred on cs_onset_dttm",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    global PROJECTION
    PROJECTION = args.projection

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger("09_trajectories", paths.log_home / "09_trajectories.log")
    style()

    grid = offsets(args)
    logger.info(
        f"model {paths.model_key}, {len(grid)} grid points "
        f"from {grid[0]:+.0f}h to {grid[-1]:+.0f}h relative to shock onset"
    )

    if args.stage in ("extract", "all"):
        extract_grid(paths, args, logger)
    if args.stage == "extract":
        return

    frame = load_grid(paths, args, logger)
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet").select(
        # shock_type and cardiac_origin are the two readings of "cardiogenic
        # shock" the headline figure is drawn for; without them it silently
        # had nothing to filter on
        "hospitalization_id", "cs_phenotype", "in_hospital_mortality",
        "icu_los_days", "admission_dttm", "cs_onset_dttm",
        "shock_type", "cardiac_origin",
        # the two sepsis phenotype systems and their denominator. Without these
        # a septic flow figure has nothing to colour by, and the omission is
        # silent: the panel guards test for column presence.
        "seymour_phenotype", "bhavani_phenotype", "sepsis_cohort",
    )
    frame = frame.join(cohort, on="hospitalization_id", how="inner")

    # where in the admission this timepoint falls, and whether the patient was
    # ever ventilated at any point -- a patient-level fact, unlike the
    # timepoint-level on_imv the state table carries
    frame = frame.with_columns(
        hours_from_admission=(
            (pl.col("cs_onset_dttm") - pl.col("admission_dttm")).dt.total_seconds()
            / 3600.0
        )
        + pl.col("offset_hours")
    )
    frame = apply_correction(
        paths, frame, args.correction, args.correction_anchor, logger
    )
    frame = project(frame, args.seed, logger)

    # Corrected runs write beside the raw ones rather than over them, so the
    # pair can be compared. Same reasoning as 03's corrected-<method>/.
    # Scoped by subset as well as by correction. Both write
    # trajectory_flow_pacmap.png, so a septic run would otherwise overwrite the
    # cardiogenic one and the report would show one labelled as the other.
    home = paths.figure_home / "trajectory" / args.subset
    if args.correction != "none":
        home = home / f"corrected-{args.correction}"
    home.mkdir(parents=True, exist_ok=True)

    px, py = proj_cols()
    plot_scatter(frame, px, py, "offset_hours", None, None,
                 f"{paths.model_key}: patient-timepoints around shock onset",
                 "hours from onset", home / "trajectory_time.png", logger,
                 continuous=True)
    plot_scatter(frame, px, py, "site", site_colors(paths.sites),
                 list(paths.sites),
                 f"{paths.model_key}: patient-timepoints by site",
                 "", home / "trajectory_site.png", logger)
    plot_scatter(frame.filter(pl.col("cs_phenotype").is_not_null()),
                 px, py, "cs_phenotype", phenotype_colors(),
                 list(PHENOTYPES),
                 f"{paths.model_key}: patient-timepoints by phenotype",
                 "", home / "trajectory_phenotype.png", logger)

    plot_flow(frame, px, py, home / f"trajectory_flow_{PROJECTION}.png",
              f"{paths.model_key}: latent flow around shock onset",
              f"mean observed displacement between consecutive {args.step_hours:g}h "
              "timepoints; cells with <5 vectors blank", logger,
              labels=proj_labels())
    plot_flow(frame, "pca_1", "pca_2", home / "trajectory_flow_pca.png",
              f"{paths.model_key}: latent flow, PCA space",
              "the same field in a linear space, where distances are metric — "
              f"a cross-check on the {proj_labels()[0].rsplit('-', 1)[0]} field",
              logger, labels=("PC1", "PC2"))

    # The figure this analysis exists for
    logger.info("cardiogenic shock flow, both definitions:")
    plot_cardiogenic_flow(frame, px, py, home, paths, args, logger)
    logger.info("septic shock flow, by each phenotype system:")
    plot_septic_flow(frame, px, py, home, paths, args, logger)

    # Outcome as a COLOUR on one field, not two fields over two cohorts.
    # Splitting the cohort fits a separate field to each, so the two figures
    # differ both in who is in them and in what the arrows were computed from,
    # and there is no way to tell which difference you are looking at. One
    # field with the outcome underneath holds the arrows fixed and varies only
    # the hue.
    if "in_hospital_mortality" in frame.columns:
        plot_flow(frame, px, py,
                  home / "trajectory_flow_mortality.png",
                  f"{paths.model_key}: latent flow, coloured by outcome",
                  f"{frame['hospitalization_id'].n_unique():,} patients; one "
                  "field over everyone, hue is in-hospital mortality",
                  logger, labels=proj_labels(),
                  colour_by={
                      "key": "in_hospital_mortality",
                      "colors": mortality_colors(),
                      "order": [False, True],
                      "names": {False: "survived", True: "died in hospital"},
                  })

    frame = attach_state(paths, frame, logger)
    if "on_imv" in frame.columns:
        frame = frame.join(
            frame.group_by("hospitalization_id").agg(
                ever_imv=pl.col("on_imv").max() > 0
            ),
            on="hospitalization_id",
            how="left",
        )

    specs = colourings(frame, paths)
    plot_colouring_grid(
        frame, specs, home / "trajectory_labels.png",
        f"{paths.model_key}: patient-timepoints around shock onset", logger,
    )
    for spec in specs:
        plot_single_colouring(
            frame, spec, home / f"trajectory_label_{spec['key']}.png",
            f"{paths.model_key}: patient-timepoints around shock onset, "
            f"by {spec['label']}",
            logger,
        )

    # Patients whose own path follows the averaged field, so the panels show
    # what the arrows are an average of rather than a random walk. A biased
    # sample on purpose; the captions say so.
    patients = pick_flow_followers(frame, px, py, logger)
    if not patients:
        patients = pick_patients(frame)
        logger.warning(
            "no patient path aligned with the flow field; falling back to a "
            "stratified random sample"
        )
    logger.info(f"patient paths: {patients}")
    for index, patient in enumerate(patients, start=1):
        plot_patient_path_single(
            frame, patient, index,
            home / f"patient_path_{index:02d}.png",
            f"{paths.model_key}: patient {index} of {len(patients)}",
            logger,
        )

    if args.counterfactuals:
        logger.info(
            f"hypothetical next tokens at +{args.cf_hours:g}h for "
            f"{len(patients)} patient(s)"
        )
        cf_frame = extract_counterfactuals(paths, args, patients, logger)
        if cf_frame.height:
            # the same affine map the grid went through, or the branch and the
            # path would sit in different spaces and every arrow would be
            # measuring the correction rather than the token
            cf_frame = apply_correction(
                paths, cf_frame, args.correction, args.correction_anchor, logger
            )
            displacement = plot_counterfactual_paths(
                frame, cf_frame, patients, home, paths, args, logger
            )
            if displacement.height:
                out = paths.table_home / "counterfactual_displacement.csv"
                displacement.write_csv(out)
                logger.info(f"wrote {out.name}")
                summary = (
                    displacement.group_by("label")
                    .agg(
                        patients=pl.len(),
                        median_displacement=pl.col("displacement").median(),
                    )
                    .sort("median_displacement", descending=True)
                )
                logger.info("  median branch displacement by intervention:")
                for row in summary.iter_rows(named=True):
                    logger.info(
                        f"    {row['label']:<42} {row['median_displacement']:.2f} "
                        f"(n={row['patients']})"
                    )
    for spec in specs:
        plot_patient_paths(
            frame, spec, patients,
            home / f"patient_paths_{spec['key']}.png",
            f"{paths.model_key}: {len(patients)} patients, cloud coloured by {spec['label']}",
            logger,
        )
    alignment = plot_state_flow(
        frame, px, py, home / "trajectory_flow_state.png",
        f"{paths.model_key}: what changes as patients move", logger,
    )
    # and one figure per variable, which is what the report tabs through
    for name in VARIABLES:
        if name in frame.columns and frame[name].is_not_null().sum() > frame.height * 0.2:
            plot_state_flow_single(
                frame, px, py, name,
                home / f"trajectory_flow_state_{name}.png",
                f"{paths.model_key}: latent flow over {VARIABLES[name][0]}",
                logger,
            )
    if alignment.height:
        alignment.write_csv(paths.table_home / "trajectory_flow_alignment.csv")
        logger.info("position encodes the variable (R2) vs flow follows it (cos):")
        for row in alignment.sort("position_r2", descending=True).iter_rows(named=True):
            logger.info(
                f"    {row['label']:<42} R2 {row['position_r2']:>6.3f}   "
                f"cos {row['flow_gradient_cosine']:+.3f}"
            )

    frame.select(
        "hospitalization_id", "site", "offset_hours", "last_valid",
        "cs_phenotype", "in_hospital_mortality",
        "umap_1", "umap_2", "pacmap_1", "pacmap_2", "pca_1", "pca_2",
    # Suffixed on a corrected run, so it never overwrites the raw one. 11 reads
    # the unsuffixed file for its trajectory figure; a corrected run silently
    # replacing it would re-draw that figure in a different space without
    # anything in 11 saying so. Same reasoning as 03's corrected-<method>/.
    ).write_parquet(
        paths.patient_level_home
        / (
            f"trajectory_projections_corrected-{args.correction}.parquet"
            if args.correction != "none"
            else "trajectory_projections.parquet"
        )
    )
    logger.info("09 done")


if __name__ == "__main__":
    main()

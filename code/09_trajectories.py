#!/usr/bin/env python3

"""
09 -- patient trajectories through the latent space

Embeds each patient repeatedly, on a grid of timepoints around cardiogenic
shock onset, and draws where they move.

Two figures:
  scatter     every patient-timepoint in one UMAP, coloured by time relative
              to onset, by phenotype, and by outcome
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

from clifcs import get_logger, load_paths
from clifcs.config import CONFIG_HOME
from clifcs.grid_winnower import GridWinnower
from clifcs.figures import save_figure
from clifcs.palette import (
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    PHENOTYPE_COLORS,
    SCATTER_SEQUENTIAL,
    SURFACE,
    site_colors,
)
from clifcs.phenotype import PHENOTYPES

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


def grid_home(paths, site: str, offset: float) -> pathlib.Path:
    tag = f"{'m' if offset < 0 else 'p'}{abs(int(round(offset))):03d}"
    return paths.scratch_home / "trajectory" / paths.model_key / site / tag


# =============================================================== extraction


def extract_grid(paths, args, logger) -> None:
    """winnow and embed every patient at every grid point"""
    from clifcs.tokens import SPLITS as TOKEN_SPLITS  # noqa: F401

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    embedding_module = __import__("02_model_embedding")

    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    onset_file = paths.cohort_home / "cs_onset.parquet"
    shared = ("tokenizer.yaml", "subject_splits.parquet", "tokens_times.parquet")

    info = embedding_module.describe_model(paths, logger)
    extraction_cfg = OmegaConf.load(CONFIG_HOME / "extraction.yaml")
    extraction_cfg.extract.batch_size = int(args.batch_size)

    for site in paths.sites:
        keep = set(cohort.filter(pl.col("site") == site)["hospitalization_id"].to_list())
        source = paths.processed(site)
        for offset in offsets(args):
            destination = grid_home(paths, site, offset)
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

            # the raw shards are large and fully regenerable
            for f in destination.glob("features-*.parquet"):
                f.unlink()
    logger.info(f"extraction done (embedding dim {info['embedding_dim']})")


# ================================================================ projection


def load_grid(paths, args, logger) -> pl.DataFrame:
    parts = [
        pl.read_parquet(grid_home(paths, site, offset) / "embeddings.parquet")
        for site in paths.sites
        for offset in offsets(args)
        if (grid_home(paths, site, offset) / "embeddings.parquet").is_file()
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


def project(frame: pl.DataFrame, seed: int, logger) -> pl.DataFrame:
    """one UMAP over every patient-timepoint, so positions are comparable"""
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
    coords = umap.UMAP(
        n_neighbors=30, min_dist=0.3, n_components=2, random_state=seed
    ).fit_transform(scores)
    return frame.with_columns(
        umap_1=pl.Series(coords[:, 0]),
        umap_2=pl.Series(coords[:, 1]),
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
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("t", SCATTER_SEQUENTIAL)
        values = shuffled[column].to_numpy()
        points = ax.scatter(px, py, s=size, c=values, cmap=cmap, linewidths=0,
                            alpha=0.8, rasterized=True)
        bar = figure.colorbar(points, cax=legend_ax.inset_axes([0.08, 0.3, 0.16, 0.6]))
        bar.set_label(label, color=INK_SECONDARY, fontsize=7)
        bar.ax.tick_params(labelsize=6, color=GRID, labelcolor=INK_SECONDARY)
        bar.outline.set_visible(False)
    else:
        values = shuffled[column].cast(pl.String).to_list()
        ax.scatter(px, py, s=size, c=[colors[v] for v in values], linewidths=0,
                   alpha=0.75, rasterized=True)
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
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out}")


def plot_flow(frame, x, y, out, title, subtitle, logger):
    """streamplot of the displacement field over the point cloud"""
    cx, cy, vx, vy, counts, n_vectors = displacement_field(frame, x, y)
    figure, ax = plt.subplots(figsize=(6.0, 5.2))

    ax.scatter(frame[x].to_numpy(), frame[y].to_numpy(), s=1.0,
               c="#dcdbd6", linewidths=0, rasterized=True)

    speed = np.hypot(vx, vy)
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
    )

    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    figure.text(0.02, -0.01, subtitle, fontsize=7, color=INK_SECONDARY)
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out} ({n_vectors:,} displacement vectors, "
                f"{int((counts >= 5).sum())} populated cells)")
    return speed


# =================================================================== driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "plot", "all"), default="all")
    parser.add_argument("--model", default=None)
    parser.add_argument("--start-hours", type=float, default=-24.0)
    parser.add_argument("--end-hours", type=float, default=48.0)
    parser.add_argument("--step-hours", type=float, default=6.0)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

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
        "hospitalization_id", "cs_phenotype", "in_hospital_mortality"
    )
    frame = frame.join(cohort, on="hospitalization_id", how="inner")
    frame = project(frame, args.seed, logger)

    home = paths.figure_home / "trajectory"
    home.mkdir(parents=True, exist_ok=True)

    plot_scatter(frame, "umap_1", "umap_2", "offset_hours", None, None,
                 f"{paths.model_key}: patient-timepoints around shock onset",
                 "hours from onset", home / "trajectory_time.png", logger,
                 continuous=True)
    plot_scatter(frame, "umap_1", "umap_2", "site", site_colors(paths.sites),
                 list(paths.sites),
                 f"{paths.model_key}: patient-timepoints by site",
                 "", home / "trajectory_site.png", logger)
    plot_scatter(frame.filter(pl.col("cs_phenotype").is_not_null()),
                 "umap_1", "umap_2", "cs_phenotype", PHENOTYPE_COLORS,
                 list(PHENOTYPES),
                 f"{paths.model_key}: patient-timepoints by phenotype",
                 "", home / "trajectory_phenotype.png", logger)

    plot_flow(frame, "umap_1", "umap_2", home / "trajectory_flow_umap.png",
              f"{paths.model_key}: latent flow around shock onset",
              f"mean observed displacement between consecutive {args.step_hours:g}h "
              "timepoints; cells with <5 vectors blank", logger)
    plot_flow(frame, "pca_1", "pca_2", home / "trajectory_flow_pca.png",
              f"{paths.model_key}: latent flow, PCA space",
              "the same field in a linear space, where distances are metric — "
              "a cross-check on the UMAP field", logger)

    for survived, name in ((False, "died"), (True, "survived")):
        part = frame.filter(pl.col("in_hospital_mortality") != survived)
        if part.height > 2000:
            plot_flow(part, "umap_1", "umap_2",
                      home / f"trajectory_flow_{name}.png",
                      f"{paths.model_key}: latent flow, patients who {name}",
                      f"{part['hospitalization_id'].n_unique():,} patients",
                      logger)

    frame.select(
        "hospitalization_id", "site", "offset_hours", "last_valid",
        "cs_phenotype", "in_hospital_mortality",
        "umap_1", "umap_2", "pca_1", "pca_2",
    ).write_parquet(paths.patient_level_home / "trajectory_projections.parquet")
    logger.info("09 done")


if __name__ == "__main__":
    main()

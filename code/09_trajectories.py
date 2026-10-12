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
from clifcs.figures import key_string, quantile_colour, save_figure
from clifcs.palette import (
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
        cmap = continuous_cmap()
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
        ax.streamplot(cx, cy, masked_x, masked_y, color=INK_PRIMARY,
                      linewidth=0.55, density=1.1, arrowsize=0.75,
                      arrowstyle="->")

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
    px, py = shuffled["umap_1"].to_numpy(), shuffled["umap_2"].to_numpy()
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
                       linewidths=0, alpha=0.8, rasterized=True)
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


def pick_patients(frame: pl.DataFrame, n: int = 5, min_points: int = 5, seed: int = 7):
    """random patients with enough timepoints to show a path"""
    counts = (
        frame.group_by("hospitalization_id")
        .agg(points=pl.len())
        .filter(pl.col("points") >= min_points)
        .sort("hospitalization_id")
    )
    if counts.height == 0:
        return []
    rng = np.random.default_rng(seed)
    ids = counts["hospitalization_id"].to_list()
    chosen = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    return [ids[int(i)] for i in sorted(chosen)]


def plot_patient_paths(frame, spec, patients, out, title, logger):
    """
    one panel per patient: their path through the cloud, over one labelling

    The path is drawn in black over the coloured cloud and each point is
    annotated with its hour relative to shock onset, so the direction of travel
    is readable rather than inferred from the arrow of a smoothed field.
    """
    figure, axes = plt.subplots(1, len(patients),
                               figsize=(3.1 * len(patients), 3.4), squeeze=False)
    px, py = frame["umap_1"].to_numpy(), frame["umap_2"].to_numpy()

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
                cmap=continuous_cmap(), linewidths=0, alpha=0.75,
            )
            cells = mesh.get_array()
            good = cells.compressed() if hasattr(cells, "compressed") else cells
            lo, hi = np.nanpercentile(good[np.isfinite(good)], [2, 98])
            mesh.set_clim(lo, hi)
        else:
            ax.scatter(px, py, s=0.9, c=values, linewidths=0, alpha=0.5,
                       rasterized=True)

        path = frame.filter(pl.col("hospitalization_id") == patient).sort("offset_hours")
        tx, ty = path["umap_1"].to_numpy(), path["umap_2"].to_numpy()
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


def plot_flow(frame, x, y, out, title, subtitle, logger, labels=None):
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
        "hospitalization_id", "cs_phenotype", "in_hospital_mortality",
        "icu_los_days", "admission_dttm", "cs_onset_dttm",
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
                 "umap_1", "umap_2", "cs_phenotype", phenotype_colors(),
                 list(PHENOTYPES),
                 f"{paths.model_key}: patient-timepoints by phenotype",
                 "", home / "trajectory_phenotype.png", logger)

    plot_flow(frame, "umap_1", "umap_2", home / "trajectory_flow_umap.png",
              f"{paths.model_key}: latent flow around shock onset",
              f"mean observed displacement between consecutive {args.step_hours:g}h "
              "timepoints; cells with <5 vectors blank", logger,
              labels=("UMAP-1", "UMAP-2"))
    plot_flow(frame, "pca_1", "pca_2", home / "trajectory_flow_pca.png",
              f"{paths.model_key}: latent flow, PCA space",
              "the same field in a linear space, where distances are metric — "
              "a cross-check on the UMAP field", logger, labels=("PC1", "PC2"))

    for survived, name in ((False, "died"), (True, "survived")):
        part = frame.filter(pl.col("in_hospital_mortality") != survived)
        if part.height > 2000:
            plot_flow(part, "umap_1", "umap_2",
                      home / f"trajectory_flow_{name}.png",
                      f"{paths.model_key}: latent flow, patients who {name}",
                      f"{part['hospitalization_id'].n_unique():,} patients",
                      logger, labels=("UMAP-1", "UMAP-2"))

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

    patients = pick_patients(frame)
    logger.info(f"patient paths: {patients}")
    for spec in specs:
        plot_patient_paths(
            frame, spec, patients,
            home / f"patient_paths_{spec['key']}.png",
            f"{paths.model_key}: five patients, cloud coloured by {spec['label']}",
            logger,
        )
    alignment = plot_state_flow(
        frame, "umap_1", "umap_2", home / "trajectory_flow_state.png",
        f"{paths.model_key}: what changes as patients move", logger,
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
        "umap_1", "umap_2", "pca_1", "pca_2",
    ).write_parquet(paths.patient_level_home / "trajectory_projections.parquet")
    logger.info("09 done")


if __name__ == "__main__":
    main()

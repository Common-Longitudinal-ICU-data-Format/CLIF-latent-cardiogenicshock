#!/usr/bin/env python3

"""
11 -- how close are our patients to the trials the guidelines were built on?

Every cardiogenic shock recommendation comes from a trial that enrolled a narrow
slice of the people who present in shock. This asks where those slices sit in
the model's latent space, which of our patients fall inside them, and who falls
inside none of them.

Three figures and four tables:

  regions      one panel per trial: its eligible patients in colour against the
               whole cohort in grey, in the same projection 03 uses
  coverage     how many trials cover each patient, and what the patients
               covered by none look like
  trajectory   distance to each trial region over the time grid from 09, so a
               patient can be seen entering or leaving a region as they
               deteriorate

  trial_eligibility.csv       per trial: n eligible, what we could not apply
  trial_representativeness.csv  our eligible subset vs the enrolled population
  trial_coverage.csv          cross-tabulation of coverage against outcome
  patient_trial_alignment.parquet  per patient: eligibility, distance and rank

Run after 03 (it needs the projections) and, for the trajectory figure, after
09. Usage:

    python code/11_trial_alignment.py
    python code/11_trial_alignment.py --anchor onset
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clifcs import ANCHORS, get_logger, load_cohort_cfg, load_paths
from clifcs.concordance import (
    add_concordance,
    attach_procedures,
    attach_severity,
    outcomes_by_position,
)
from clifcs.config import CONFIG_HOME
from clifcs.figures import save_figure, style
from clifcs.interactive import render as render_interactive
from clifcs.palette import (
    CATEGORICAL,
    CONTEXT,
    GRID,
    HIGHLIGHT,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL_BLUE,
    site_colors,
)
from clifcs.trials import (
    add_distances,
    criteria_frame,
    add_eligibility,
    attach_features,
    compare_to_published,
    fit_regions,
    load_trials,
    inside_all_null,
    assign_guideline,
    rank_and_orphans,
    table1_comparison,
    table1_position,
)

#: the projection the figures are drawn in; 03 writes all three
PROJECTION = "pacmap"

#: local-rate ramp, from the palette's single-hue blue steps
#:
#: A rate is a magnitude, so it takes a sequential one-hue scale rather than a
#: categorical palette. Built from SEQUENTIAL_BLUE's full 100-700 range because
#: a hexagon is an area encoding, where the palest step is meant to recede into
#: the surface -- unlike a scatter mark, which would simply vanish.
LOCAL_RATE_CMAP = LinearSegmentedColormap.from_list(
    "local_rate", list(SEQUENTIAL_BLUE)
)

#: reference palette slot 2, for "eligible but outside the published IQRs"
#:
#: Slots 1 and 2 against the context grey is a three-series panel, which is
#: what the all-pairs gate validates -- worst pair orange<->aqua is not in play
#: here, and blue<->orange clears every floor comfortably.
OUTSIDE_COLOUR = CATEGORICAL[1]


def embedding_matrix(frame: pl.DataFrame) -> np.ndarray:
    columns = sorted(c for c in frame.columns if c.startswith("emb_"))
    return frame.select(columns).to_numpy().astype(np.float64)


def load_anchor(paths, anchor: str, logger) -> pl.DataFrame:
    """the per-site embedding files for one anchor, restricted to the cohort"""
    sources = sorted(
        paths.patient_level_home.glob(f"embeddings_anchor-{anchor}_*.parquet")
    )
    sources = [p for p in sources if "share-safe" not in p.name]
    if not sources:
        raise FileNotFoundError(
            f"no embeddings for anchor {anchor} in {paths.patient_level_home}; "
            "run 02 first"
        )
    frame = pl.concat([pl.read_parquet(p) for p in sources], how="diagonal_relaxed")
    logger.info(f"[{anchor}] {frame.height} patients from {len(sources)} site files")
    return frame


def pca_coords(frame: pl.DataFrame, components: int, seed: int, logger) -> np.ndarray:
    """
    the subspace the regions are fitted in

    Refit here rather than read from 03's saved projections because the regions
    need the full `components`-dimensional scores, and 03 only persists the
    leading two. Same scaler, same component count and same seed, so the
    subspace is the one 03 drew.
    """
    raw = embedding_matrix(frame)
    scaled = StandardScaler().fit_transform(raw)
    pca = PCA(n_components=min(components, scaled.shape[1]), random_state=seed)
    scores = pca.fit_transform(scaled)
    logger.info(
        f"  PCA subspace: {scores.shape[1]} components, "
        f"{pca.explained_variance_ratio_.sum():.1%} of variance"
    )
    return scores


def attach_projection(paths, frame: pl.DataFrame, anchor: str, logger) -> pl.DataFrame:
    """the 2-d coordinates 03 saved, for drawing only"""
    path = paths.patient_level_home / f"projections_anchor-{anchor}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run 03 first")
    wanted = [f"{PROJECTION}_1", f"{PROJECTION}_2"]
    projections = pl.read_parquet(path)
    missing = [c for c in wanted if c not in projections.columns]
    if missing:
        raise KeyError(
            f"{path.name} has no {missing}; rerun 03 so it writes {PROJECTION} "
            "coordinates"
        )
    return frame.join(
        projections.select("hospitalization_id", "site", *wanted),
        on=["hospitalization_id", "site"],
        how="inner",
        validate="1:1",
    )


# ================================================================== figures


def _marker_size(n: int) -> float:
    if n > 20000:
        return 1.2
    if n > 8000:
        return 2.5
    return 5.0


def plot_regions(frame, trials, regions, out_path, anchor, model_key) -> None:
    """
    one panel per trial: the local rate at which patients meet its criteria

    Hexagon local means, not one mark per patient. Drawn as points, 4,397
    eligible patients spread across 34,142 grey ones saturate every cluster, so
    the first six panels came out looking identical and the figure said only
    "trial-eligible patients are everywhere" -- which is true, and is exactly
    the thing the panels are supposed to let you *measure* rather than assert.
    A local rate separates "this cluster is 60% eligible" from "this cluster is
    8% eligible", which the point version could not.

    This is the convention 09 already uses for a continuous value over a dense
    cloud, for the same reason.

    The colour scale is shared across panels and runs from 0 to 1, so a panel
    can be compared with its neighbour. Each panel's title carries the trial's
    cohort-wide prevalence, which is the number a hexagon's shade should be
    read against: a 20% hexagon is enrichment for DanGer Shock (0.6% overall)
    and depletion for SOAP II (89%).
    """
    x, y = f"{PROJECTION}_1", f"{PROJECTION}_2"
    xs = frame[x].to_numpy()
    ys = frame[y].to_numpy()

    # Three trials reduce to the same computable criteria once the parts we
    # cannot evaluate are dropped (AMI plus shock), so their panels are
    # identical and would silently imply three independent pieces of evidence.
    # The duplicate is named on the panel instead.
    seen: dict[bytes, str] = {}
    duplicate_of: dict[str, str] = {}
    for trial in trials:
        signature = frame[f"elig_{trial.key}"].to_numpy().tobytes()
        if signature in seen:
            duplicate_of[trial.key] = seen[signature]
        else:
            seen[signature] = trial.label

    columns = 4
    rows = int(np.ceil(len(trials) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.75 * columns, 3.1 * rows), squeeze=False
    )

    mesh = None
    for position, trial in enumerate(trials):
        ax = axes[position // columns][position % columns]
        eligible = frame[f"elig_{trial.key}"].to_numpy().astype(float)
        prevalence = eligible.mean()

        # the cohort's own outline, so empty space reads as empty rather than
        # as zero eligibility
        ax.scatter(xs, ys, s=1.0, c=CONTEXT, linewidths=0, rasterized=True,
                   zorder=1)
        mesh = ax.hexbin(
            xs, ys, C=eligible, reduce_C_function=np.mean,
            gridsize=58, mincnt=8, cmap=LOCAL_RATE_CMAP, vmin=0.0, vmax=1.0,
            linewidths=0, zorder=2,
        )

        # Two lines at most. The enrichment threshold lives in
        # trial_eligibility_<anchor>.csv; three-line titles collided with the
        # neighbouring panel and cost more than they bought.
        region = next(r for r in regions if r.key == trial.key)
        note = f"{int(eligible.sum()):,} eligible / {prevalence:.1%} of cohort"
        if region.fitted and region.centroid_separation is not None:
            note += f" / sep {region.centroid_separation:.2f}"
        else:
            note += " / region not fitted"
        label = trial.label
        if trial.key in duplicate_of:
            label += f" = {duplicate_of[trial.key]}"
        ax.set_title(f"{label} ({trial.year})" + chr(10) + note, loc="left",
                     pad=4, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    for position in range(len(trials), rows * columns):
        axes[position // columns][position % columns].axis("off")

    if mesh is not None:
        bar = figure.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.016,
                              pad=0.012)
        bar.set_label("share of local patients meeting the trial's criteria",
                      color=INK_SECONDARY, fontsize=7)
        bar.ax.tick_params(labelsize=6, color=GRID, labelcolor=INK_SECONDARY)
        bar.outline.set_visible(False)

    figure.suptitle(
        f"{model_key}: where each trial's population sits, "
        f"{PROJECTION.upper()}, anchor {anchor}  "
        "(“sep” is centroid separation, in within-group spreads)",
        x=0.01, ha="left", fontsize=9, color=INK_PRIMARY,
    )
    figure.subplots_adjust(hspace=0.26, wspace=0.05, top=0.90, bottom=0.02)
    save_figure(figure, out_path)
    plt.close(figure)


def plot_eligible_vs_tested(frame, trials, out_path, anchor, model_key) -> None:
    """
    per trial, a pair of panels: who would qualify, and who matches the people
    actually enrolled

    The left panel of each pair is the binary question the criteria answer --
    eligible or not. The right panel splits the eligible patients by whether
    they fall inside every interquartile range the paper published.

    The pair is the point. Enrolment criteria are thresholds, so they admit a
    much wider range of patients than a trial ends up containing: somebody can
    satisfy every DanGer Shock criterion with a systolic pressure of 60 and a
    heart rate of 70 and still sit below both published IQRs, meaning the
    result was never observed in anyone much like them. The left panel counts
    them in; the right panel is where that shows.

    Three colours per panel, which is what the all-pairs colourblind gate
    allows -- the panels are scatters with every series on screen together.
    """
    x, y = f"{PROJECTION}_1", f"{PROJECTION}_2"
    xs, ys = frame[x].to_numpy(), frame[y].to_numpy()
    size = _marker_size(frame.height)

    pairs_per_row = 2
    rows = int(np.ceil(len(trials) / pairs_per_row))
    figure, axes = plt.subplots(
        rows, pairs_per_row * 2,
        figsize=(2.5 * pairs_per_row * 2, 2.9 * rows), squeeze=False,
    )

    def blank(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    for position, trial in enumerate(trials):
        row = position // pairs_per_row
        base = (position % pairs_per_row) * 2
        eligible = frame[f"elig_{trial.key}"].to_numpy()

        # ---- left: eligible or not
        ax = axes[row][base]
        ax.scatter(xs[~eligible], ys[~eligible], s=size, c=CONTEXT,
                   linewidths=0, rasterized=True, zorder=1)
        ax.scatter(xs[eligible], ys[eligible], s=size, c=HIGHLIGHT,
                   linewidths=0, alpha=0.9, rasterized=True, zorder=2)
        ax.set_title(
            f"{trial.label} ({trial.year})\nwould qualify: "
            f"{int(eligible.sum()):,} ({eligible.mean():.1%})",
            loc="left", pad=4, fontsize=7,
        )
        blank(ax)

        # ---- right: of those, inside the published population
        ax = axes[row][base + 1]
        outside_col = f"t1_outside_{trial.key}"
        compared = frame[f"t1_n_compared_{trial.key}"].to_numpy()
        usable = eligible & (compared > 0)
        if outside_col in frame.columns and usable.any():
            n_outside = frame[outside_col].fill_null(-1).to_numpy()
            inside = usable & (n_outside == 0)
            outside = usable & (n_outside > 0)
            ax.scatter(xs[~usable], ys[~usable], s=size, c=CONTEXT,
                       linewidths=0, rasterized=True, zorder=1)
            ax.scatter(xs[outside], ys[outside], s=size, c=OUTSIDE_COLOUR,
                       linewidths=0, alpha=0.9, rasterized=True, zorder=2)
            ax.scatter(xs[inside], ys[inside], s=size, c=HIGHLIGHT,
                       linewidths=0, alpha=0.9, rasterized=True, zorder=3)
            share = inside.sum() / max(usable.sum(), 1)
            n_vars = int(np.median(compared[usable]))
            ax.set_title(
                f"inside all {n_vars} published IQRs\n"
                f"{int(inside.sum()):,} of {int(usable.sum()):,} eligible "
                f"({share:.1%})",
                loc="left", pad=4, fontsize=7,
            )
        else:
            ax.scatter(xs, ys, s=size, c=CONTEXT, linewidths=0,
                       rasterized=True, zorder=1)
            ax.set_title("no usable published IQRs", loc="left", pad=4,
                         fontsize=7, color=INK_SECONDARY)
        blank(ax)

    for position in range(len(trials), rows * pairs_per_row):
        row = position // pairs_per_row
        base = (position % pairs_per_row) * 2
        axes[row][base].axis("off")
        axes[row][base + 1].axis("off")

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5,
               markerfacecolor=HIGHLIGHT, markeredgecolor="none",
               label="eligible (left) / inside every published IQR (right)"),
        Line2D([], [], marker="o", linestyle="none", markersize=5,
               markerfacecolor=OUTSIDE_COLOUR, markeredgecolor="none",
               label="eligible but outside at least one published IQR"),
        Line2D([], [], marker="o", linestyle="none", markersize=5,
               markerfacecolor=CONTEXT, markeredgecolor="none",
               label="not eligible"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  fontsize=7, labelcolor=INK_SECONDARY,
                  bbox_to_anchor=(0.5, -0.012))
    figure.suptitle(
        f"{model_key}: would qualify, and resembles who was enrolled "
        f"({PROJECTION.upper()}, anchor {anchor})",
        x=0.01, ha="left", fontsize=9, color=INK_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.965))
    save_figure(figure, out_path)
    plt.close(figure)


def plot_coverage(frame, out_path, anchor, model_key, paths) -> None:
    """
    how many trials cover each patient, and how the uncovered differ

    Three panels: the distribution of coverage, mortality against coverage, and
    the projection coloured by whether any trial covers the patient at all.
    """
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))

    # ---- coverage histogram
    ax = axes[0]
    counts = (
        frame.group_by("n_eligible_trials")
        .agg(n=pl.len())
        .sort("n_eligible_trials")
    )
    ax.bar(counts["n_eligible_trials"].to_list(), counts["n"].to_list(),
           color=HIGHLIGHT, width=0.72, linewidth=0)
    for k, n in zip(counts["n_eligible_trials"].to_list(), counts["n"].to_list()):
        ax.text(k, n, f" {n / frame.height:.0%}", ha="center", va="bottom",
                fontsize=6.5, color=INK_SECONDARY, rotation=90)
    ax.set_xlabel("trials whose criteria the patient meets")
    ax.set_ylabel("patients")
    ax.set_title("Guideline coverage", loc="left", pad=6)
    ax.set_ylim(0, max(counts["n"].to_list()) * 1.22)

    # ---- mortality by coverage
    ax = axes[1]
    by = (
        frame.group_by("n_eligible_trials")
        .agg(n=pl.len(), mortality=pl.col("in_hospital_mortality").mean())
        .sort("n_eligible_trials")
        .filter(pl.col("n") >= 25)
    )
    ax.plot(by["n_eligible_trials"].to_list(), by["mortality"].to_list(),
            marker="o", markersize=5, linewidth=1.6, color=HIGHLIGHT)
    ax.set_xlabel("trials whose criteria the patient meets")
    ax.set_ylabel("in-hospital mortality")
    # deliberately not "severity rises with coverage": it does not rise
    # monotonically, because the count mixes trials with very different
    # severity thresholds
    ax.set_title("Mortality by coverage count", loc="left", pad=6)
    ax.set_ylim(0, max(by["mortality"].to_list()) * 1.2 if by.height else 1)

    # ---- orphans in the projection
    ax = axes[2]
    x, y = f"{PROJECTION}_1", f"{PROJECTION}_2"
    xs, ys = frame[x].to_numpy(), frame[y].to_numpy()
    orphan = frame["evidence_orphan"].to_numpy()
    size = _marker_size(frame.height)
    ax.scatter(xs[~orphan], ys[~orphan], s=size * 0.8, c=CONTEXT, linewidths=0,
               rasterized=True, zorder=1)
    ax.scatter(xs[orphan], ys[orphan], s=size * 1.5, c=HIGHLIGHT, linewidths=0,
               alpha=0.8, rasterized=True, zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(
        f"No trial covers them\n{int(orphan.sum()):,} patients "
        f"({orphan.mean():.1%})",
        loc="left", pad=6,
    )

    for ax in axes[:2]:
        ax.grid(True, axis="y", color=GRID, linewidth=0.4, alpha=0.7)
        ax.set_axisbelow(True)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axes[2].spines[side].set_visible(False)

    figure.suptitle(
        f"{model_key}: guideline coverage, anchor {anchor}",
        x=0.01, ha="left", fontsize=9, color=INK_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, out_path)
    plt.close(figure)


def plot_trajectory_distance(paths, frame, trials, regions, cfg, out_path,
                             seed, logger) -> bool:
    """
    distance to each trial region over the grid of timepoints from 09

    The question this answers that a single anchor cannot: do patients *move*
    into the populations the trials enrolled? A patient who is outside every
    region at onset and inside one twelve hours later was always going to be a
    trial patient; one who moves the other way was briefly eligible for a
    recommendation that no longer describes them.

    Returns False if 09's output is absent or does not cover this cohort, since
    the figure would otherwise be drawn from a different set of patients than
    everything else here.
    """
    path = paths.patient_level_home / "trajectory_projections.parquet"
    if not path.is_file():
        logger.info("  09 has not been run; skipping the trajectory figure")
        return False

    trajectory = pl.read_parquet(path)
    cohort_ids = set(frame["hospitalization_id"].to_list())
    traj_ids = set(trajectory["hospitalization_id"].to_list())
    if len(cohort_ids - traj_ids) > 0.02 * len(cohort_ids):
        logger.warning(
            f"  09's trajectory covers {len(traj_ids):,} patients against this "
            f"cohort's {len(cohort_ids):,}; skipping the trajectory figure "
            "rather than drawing it from a different population"
        )
        return False

    # 09 saves only the 2-d coordinates, so the regions -- which live in the
    # 50-d PCA subspace -- cannot be applied to it directly. What is comparable
    # is eligibility, which is a property of the patient rather than of the
    # timepoint, so the figure shows where each trial's eligible patients are
    # over time rather than a distance.
    elig_cols = [f"elig_{t.key}" for t in trials]
    trajectory = trajectory.join(
        frame.select("hospitalization_id", "site", *elig_cols, "evidence_orphan"),
        on=["hospitalization_id", "site"],
        how="inner",
    )
    if trajectory.height == 0:
        logger.warning("  trajectory join produced no rows; skipping")
        return False

    fitted = [t for t in trials if next(r for r in regions if r.key == t.key).fitted]
    panels = fitted + [None]  # the last panel is the orphans
    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.1 * columns, 2.9 * rows), squeeze=False
    )

    px, py = "pacmap_1", "pacmap_2"
    if px not in trajectory.columns:
        px, py = "umap_1", "umap_2"
        logger.info(f"  09 wrote no PaCMAP coordinates; drawing in {px}/{py}")

    xs = trajectory[px].to_numpy()
    ys = trajectory[py].to_numpy()
    offsets = trajectory["offset_hours"].to_numpy()
    size = 0.8 if trajectory.height > 100000 else 1.6

    for position, trial in enumerate(panels):
        ax = axes[position // columns][position % columns]
        if trial is None:
            mask = trajectory["evidence_orphan"].to_numpy()
            title = f"covered by no trial\n{int(mask.sum()):,} patient-timepoints"
        else:
            mask = trajectory[f"elig_{trial.key}"].to_numpy()
            title = f"{trial.label}\n{int(mask.sum()):,} patient-timepoints"

        ax.scatter(xs, ys, s=size * 0.7, c=CONTEXT, linewidths=0,
                   rasterized=True, zorder=1)
        if mask.any():
            points = ax.scatter(
                xs[mask], ys[mask], s=size * 1.8, c=offsets[mask],
                cmap="cubehelix", linewidths=0, alpha=0.8, rasterized=True,
                zorder=2,
            )
        ax.set_title(title, loc="left", pad=5, fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    for position in range(len(panels), rows * columns):
        axes[position // columns][position % columns].axis("off")

    bar = figure.colorbar(points, ax=axes.ravel().tolist(), fraction=0.015,
                          pad=0.01)
    bar.set_label("hours from shock onset", color=INK_SECONDARY, fontsize=7)
    bar.ax.tick_params(labelsize=6, color=GRID, labelcolor=INK_SECONDARY)
    bar.outline.set_visible(False)

    figure.suptitle(
        f"Trial-eligible patients over their trajectory ({px.split('_')[0]})",
        x=0.01, ha="left", fontsize=9, color=INK_PRIMARY,
    )
    save_figure(figure, out_path)
    plt.close(figure)
    return True


# =================================================================== driver


def analyse(paths, anchor: str, cohort_cfg, trial_cfg, args, logger) -> None:
    trials = load_trials(trial_cfg)
    logger.info(f"=== anchor {anchor}: {len(trials)} trials")

    frame = load_anchor(paths, anchor, logger)
    frame = attach_projection(paths, frame, anchor, logger)

    # the eligibility criteria read columns 02 does not carry
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    keep = [
        c
        for c in (
            "hospitalization_id", "site", "met_vasoactive_support",
            "met_sustained_hypotension", "gcs_min_in_window", "lactate",
            "cs_onset_dttm", "age_at_admission", "sex_category",
            # the per-agent flags are derived from this; without it they all
            # silently read False, which is how every "norepinephrine running"
            # comparison first came out at 0.0% against a published 85.5%
            "vasoactives_used",
        )
        if c in cohort.columns
    ]
    frame = frame.drop(
        [c for c in ("lactate", "age_at_admission", "sex_category")
         if c in frame.columns and c in keep]
    ).join(cohort.select(keep), on=["hospitalization_id", "site"], how="inner",
           validate="1:1")

    per_site = []
    for site in paths.sites:
        part = frame.filter(pl.col("site") == site)
        if part.height == 0:
            continue
        part = attach_features(paths.site_data[site], part, cohort_cfg, logger)
        part = attach_procedures(paths.site_data[site], part, logger)
        per_site.append(part)
    frame = pl.concat(per_site, how="diagonal_relaxed")

    frame = add_eligibility(frame, trials)
    # Raw PCA scores, not whitened: 03 measures neighbourhood purity in this
    # same space, and the two numbers are only comparable if the neighbour
    # graph is built the same way.
    coords = pca_coords(frame, int(trial_cfg.regions.pca_components), args.seed, logger)
    regions = fit_regions(frame, coords, trials, trial_cfg, logger)
    frame = add_distances(frame, coords, regions)
    frame = rank_and_orphans(frame, regions)

    # how far each eligible patient sits outside the population the paper
    # actually reported, in interquartile ranges
    logger.info("position of eligible patients within the published baselines:")
    for trial in trials:
        frame = table1_position(frame, trial, logger)
    frame = attach_severity(paths, frame, logger)
    frame = add_concordance(frame, trials, logger)
    comparisons = [table1_comparison(frame, trial, logger) for trial in trials]
    comparisons = [c for c in comparisons if c.height]
    table1 = (
        pl.concat(comparisons, how="diagonal_relaxed") if comparisons
        else pl.DataFrame()
    )

    # ----------------------------------------------------------------- tables
    paths.table_home.mkdir(parents=True, exist_ok=True)
    eligibility = pl.DataFrame(
        [
            {
                "trial": t.label,
                "year": t.year,
                "intervention": t.intervention,
                "guideline": t.guideline,
                "citation": t.citation,
                "n_eligible": int(frame[f"elig_{t.key}"].sum()),
                "share_eligible": float(frame[f"elig_{t.key}"].mean()),
                "n_inside_region": int(frame[f"inside_{t.key}"].sum()),
                "n_inside_not_eligible": int(
                    (frame[f"inside_{t.key}"] & ~frame[f"elig_{t.key}"]).sum()
                ),
                "mortality_eligible": float(
                    frame.filter(pl.col(f"elig_{t.key}"))[
                        "in_hospital_mortality"
                    ].mean()
                )
                if frame[f"elig_{t.key}"].sum()
                else None,
                "centroid_separation_spreads": next(
                    (r.centroid_separation for r in regions if r.key == t.key), None
                ),
                "affinity_threshold": next(
                    (r.threshold for r in regions if r.key == t.key), None
                ),
                "threshold_neighbours": next(
                    (r.threshold_neighbours for r in regions if r.key == t.key), None
                ),
                "criteria_not_applied": "; ".join(t.not_implementable) or "none",
            }
            for t in trials
        ]
    )
    eligibility.write_csv(paths.table_home / f"trial_eligibility_{anchor}.csv")

    # the published criteria, with what we can and cannot apply. Anchor-free,
    # but written per anchor so the report can find it the same way as the rest.
    criteria = criteria_frame(trial_cfg)
    if criteria.height:
        criteria.write_csv(paths.table_home / "trial_criteria.csv")

    if table1.height:
        table1.write_csv(paths.table_home / f"trial_table1_{anchor}.csv")

        summary = []
        for trial in trials:
            column = f"t1_outside_{trial.key}"
            compared = f"t1_n_compared_{trial.key}"
            usable = frame.filter(
                pl.col(f"elig_{trial.key}") & (pl.col(compared) > 0)
            )
            if usable.height == 0:
                continue
            k = int(usable[compared].median())
            observed = float((usable[column] == 0).mean())
            null = inside_all_null(k)
            summary.append(
                {
                    "trial": trial.label,
                    "n_eligible": int(frame[f"elig_{trial.key}"].sum()),
                    "n_comparable": usable.height,
                    "n_variables_compared": k,
                    "share_inside_all_iqrs": observed,
                    # what a representative cohort would give, 0.5^k. Without
                    # it the share is unreadable: it falls with every variable
                    # added whether or not the cohort differs at all.
                    "null_inside_all_iqrs": null,
                    "ratio_to_null": observed / null if null > 0 else None,
                    "median_variables_outside": float(usable[column].median()),
                    "median_max_distance_iqrs": float(
                        usable[f"t1_maxdist_{trial.key}"].median()
                    ),
                }
            )
        if summary:
            inside = pl.DataFrame(summary)
            inside.write_csv(paths.table_home / f"trial_inside_tested_{anchor}.csv")
            logger.info(
                "inside every published IQR, against the 0.5^k a representative "
                "cohort would give:"
            )
            for row in inside.iter_rows(named=True):
                logger.info(
                    f"  {row['trial']:<22} {row['share_inside_all_iqrs']:>6.1%} "
                    f"of {row['n_comparable']:>6,} across "
                    f"{row['n_variables_compared']} IQRs  "
                    f"(null {row['null_inside_all_iqrs']:>5.1%}, "
                    f"ratio {row['ratio_to_null']:.2f})"
                )
            logger.info(
                "  a ratio near 1 means the eligible patients are as "
                "concentrated as a representative sample; read the per-variable "
                "shifts above for where they actually differ"
            )

    frame = assign_guideline(frame, trials, logger)
    outcomes = outcomes_by_position(frame, trials, logger)
    if outcomes.height:
        outcomes.write_csv(paths.table_home / f"trial_outcomes_{anchor}.csv")

    representativeness = compare_to_published(frame, trials, logger)
    representativeness.write_csv(
        paths.table_home / f"trial_representativeness_{anchor}.csv"
    )

    coverage = (
        frame.group_by("n_eligible_trials")
        .agg(
            n=pl.len(),
            share=pl.len() / frame.height,
            mortality=pl.col("in_hospital_mortality").mean(),
            median_age=pl.col("age_at_admission").median(),
            pct_female=(pl.col("sex_category") == "Female").mean(),
            median_lactate=pl.col("lactate").median(),
            median_icu_los=pl.col("icu_los_days").median(),
        )
        .sort("n_eligible_trials")
    )
    coverage.write_csv(paths.table_home / f"trial_coverage_{anchor}.csv")

    logger.info("coverage by number of trials a patient qualifies for:")
    for row in coverage.iter_rows(named=True):
        logger.info(
            f"  {row['n_eligible_trials']} trial(s): n={row['n']:>6,} "
            f"({row['share']:>5.1%})  mortality {row['mortality']:>5.1%}  "
            f"age {row['median_age']:>3.0f}  women {row['pct_female']:>5.1%}  "
            f"lactate {row['median_lactate'] or float('nan'):>4.1f}"
        )

    orphans = frame.filter(pl.col("evidence_orphan"))
    logger.info(
        f"evidence orphans (eligible for nothing, inside no region): "
        f"{orphans.height:,} ({orphans.height / frame.height:.1%})"
    )
    if orphans.height:
        logger.info(
            f"  vs the rest: mortality "
            f"{orphans['in_hospital_mortality'].mean():.1%} against "
            f"{frame.filter(~pl.col('evidence_orphan'))['in_hospital_mortality'].mean():.1%}"
            f", median age {orphans['age_at_admission'].median():.0f}, "
            f"women {(orphans['sex_category'] == 'Female').mean():.1%}"
        )

    keep_cols = [
        "hospitalization_id", "site", "age_at_admission", "sex_category",
        "lactate", "sbp_at_onset", "in_hospital_mortality", "icu_los_days",
        "shock_type", "cs_phenotype", "n_eligible_trials", "n_regions_inside",
        "nearest_trial", "nearest_lift", "evidence_orphan", "criteria_orphan",
        "assigned_guideline", "guideline_fit",
        "n_specific_trials", "n_specific_inside",
        f"{PROJECTION}_1", f"{PROJECTION}_2",
    ]
    keep_cols += [
        c
        for c in frame.columns
        if c.startswith(
            ("elig_", "inside_", "affinity_", "lift_", "dx_", "t1_",
             "concordant_", "had_", "on_")
        )
    ]
    keep_cols += [
        c
        for c in (
            "map_at_onset", "dbp_at_onset", "heart_rate_at_onset", "temp_at_onset",
            "ph_at_onset", "bmi", "weight_kg", "height_cm",
            "imv_before_onset", "male", "sofa_total", "sofa_non_cv",
        )
        if c in frame.columns
    ]
    paths.patient_level_home.mkdir(parents=True, exist_ok=True)
    frame.select([c for c in keep_cols if c in frame.columns]).write_parquet(
        paths.patient_level_home / f"patient_trial_alignment_{anchor}.parquet"
    )

    # ---------------------------------------------------------------- figures
    home = paths.figure_home / f"anchor-{anchor}"
    home.mkdir(parents=True, exist_ok=True)
    plot_regions(frame, trials, regions, home / "trial_regions.png", anchor,
                 paths.model_key)
    plot_eligible_vs_tested(frame, trials, home / "trial_eligible_vs_tested.png",
                            anchor, paths.model_key)
    plot_coverage(frame, home / "trial_coverage.png", anchor, paths.model_key,
                  paths)
    if anchor == "onset":
        plot_trajectory_distance(
            paths, frame, trials, regions, trial_cfg,
            paths.figure_home / "trajectory" / "trial_eligibility_paths.png",
            args.seed, logger,
        )
    # the interactive view, as an HTML fragment 07 embeds verbatim
    fragment = paths.table_home / f"trial_interactive_{anchor}.html"
    fragment.write_text(
        render_interactive(frame, trials, uid=anchor.replace("-", "")),
        encoding="utf-8",
    )
    logger.info(
        f"[{anchor}] interactive fragment -> {fragment.name} "
        f"({fragment.stat().st_size / 1e6:.1f} MB)"
    )
    logger.info(f"[{anchor}] figures -> {home}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", choices=ANCHORS, default=None,
                        help="one anchor only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, help="which checkpoint's embeddings")
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger("11_trial_alignment",
                        paths.log_home / "11_trial_alignment.log")
    style()

    cohort_cfg = load_cohort_cfg()
    trial_cfg = OmegaConf.load(CONFIG_HOME / "trials.yaml")

    for anchor in (args.anchor,) if args.anchor else ANCHORS:
        analyse(paths, anchor, cohort_cfg, trial_cfg, args, logger)

    logger.info("11 done")


if __name__ == "__main__":
    main()

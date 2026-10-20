#!/usr/bin/env python3

"""
13 -- undifferentiated shock: no coded mechanism, and what the model makes of it

The cohort is selected on physiology, so it is mechanism-agnostic by
construction and 54.6% of it carries no shock diagnosis code at all. This asks
what those 18,626 patients resemble in the frozen model's latent space, whether
a linear probe trained on the coded patients will assign them a mechanism, and
whether the treatment they were given matches the assignment.

THREE THINGS THAT DECIDE HOW ANY NUMBER HERE READS

1. The two sites code different mechanisms, not different amounts. The uncoded
   share is close -- NU 56.6%, UCMC 48.9% -- but UCMC codes cardiogenic 2.5
   times as often as NU and NU codes septic 1.7 times as often as UCMC. The
   coded patients are what the centroids and the probe are built from, so that
   mix difference means neither transports. Every centroid, every neighbourhood
   and every probe is fitted and scored within site.

2. The uncoded are much less sick: 6.2% in-hospital mortality pooled (4.6% NU,
   11.2% UCMC) against 25.2% cardiogenic, 29.6% septic and 43.3% mixed. Most of
   them are not a diagnostic puzzle but a mild population the physiological
   criteria admitted.

3. Whether a diagnosis arrives later is not answerable. The diagnosis table is
   hospitalization-level and untimed, and an uncoded patient has no shock code
   at any point by definition. What is reported instead is whether a mechanism
   was recorded by some other means -- a positive culture, or a non-shock code
   implying one -- somewhere in the admission.

Figures and tables:

  groups        who the groups are, per site, with severity beside the counts
  proximity     cosine distance to each mechanism centroid, undifferentiated
                against the coded patients as the reference
  probe         AUROC per class, within-site and cross-site, and calibration
  snapshots     the same probe at 0 h, +6 h and +24 h from onset -- how much
                separation is treatment leaking back in
  treatments    what each group was given, and whether the undifferentiated
                were treated as the probe would have assigned them

Runs after 01 (the cohort and shock_type) and 02 (the embeddings). The snapshot
stage additionally needs 09's grid, and degrades to the offsets that exist.

Usage
    python code/13_undifferentiated.py
    python code/13_undifferentiated.py --anchor onset --skip-snapshots
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
from omegaconf import OmegaConf

from clifcs import ANCHORS, get_logger, load_cohort_cfg, load_paths
from clifcs import site_embedding_files
from clifcs.concordance import attach_severity
from clifcs.config import CONFIG_HOME
from clifcs.figures import save_figure, style
from clifcs.palette import (
    CATEGORICAL,
    CONTEXT,
    GRID,
    HIGHLIGHT,
    INK_PRIMARY,
    INK_SECONDARY,
    shock_type_colors,
)
from clifcs.trials import attach_features
from clifcs.undifferentiated import (
    GROUPS,
    MECHANISMS,
    NOT_ASSESSABLE,
    TREATMENT_SIGNATURE,
    UNDIFFERENTIATED,
    assignment_concordance,
    centroid_proximity,
    concordance_adjusted,
    describe_groups,
    first_agent_and_antibiotic,
    first_agent_mix,
    fit_probe,
    knn_composition,
    mechanism_evidence,
    reference_separation,
    snapshot_separation,
    treatment_profile,
)


def load_anchor(paths, anchor: str, logger) -> pl.DataFrame:
    """the per-site embedding files for one anchor, pooled"""
    sources = site_embedding_files(paths.patient_level_home, anchor)
    if not sources:
        raise FileNotFoundError(
            f"no embeddings for anchor {anchor} in {paths.patient_level_home}; "
            "run 02 first"
        )
    frame = pl.concat([pl.read_parquet(p) for p in sources], how="diagonal_relaxed")
    logger.info(f"[{anchor}] {frame.height:,} patients from {len(sources)} site files")
    return frame


def attach_cohort(paths, frame: pl.DataFrame, cfg, logger) -> pl.DataFrame:
    """
    join the cohort, the severity score and the treatment columns

    The embedding files are a superset the moment an enrolment criterion is
    tightened -- 02 embeds everyone in the winnowed stream -- so the join is
    inner on purpose. A patient the cohort no longer contains is not described
    here under the old criteria.
    """
    # Every cohort column, not a hand-picked subset. The first version of this
    # named the dozen columns the tables report and broke twice: attach_features
    # reads `gcs_min_in_window` and `vasoactives_used`, neither of which any
    # table here displays, and a subset that happens to satisfy one helper is
    # one refactor away from not satisfying the next.
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    embeddings = frame.select(
        "hospitalization_id", "site", *(c for c in frame.columns if c.startswith("emb_"))
    )
    frame = cohort.join(
        embeddings, on=["hospitalization_id", "site"], how="inner"
    )
    logger.info(f"  {frame.height:,} patients with both a cohort row and an embedding")

    frame = attach_severity(paths, frame, logger)

    from clifcs.freedays import add_free_days

    parts = []
    cohort_cfg = load_cohort_cfg()
    for site in paths.sites:
        part = frame.filter(pl.col("site") == site)
        if not part.height:
            continue
        part = attach_features(paths.site_data[site], part, cohort_cfg, logger)
        # joined, not assigned: this one returns only the columns it computes,
        # unlike attach_features which returns the frame it was given
        part = part.join(
            first_agent_and_antibiotic(paths.site_data[site], part, cfg, logger),
            on="hospitalization_id",
            how="left",
            validate="1:1",
        )
        part = add_free_days(paths.raw_subset(site), part, logger)
        parts.append(part)
    frame = pl.concat(parts, how="diagonal_relaxed")

    # the two treatment signatures the concordance table can score
    frame = frame.with_columns(
        abx_within_6h_flag=(pl.col("hours_to_first_antibiotic") <= 6).fill_null(False),
        cardiogenic_treatment=(
            pl.col("on_dobutamine").fill_null(False)
            | pl.col("on_milrinone").fill_null(False)
            | pl.col("cardiac_mcs").fill_null(False)
        ),
    )
    return frame


# ================================================================== figures


def figure_groups(groups: pl.DataFrame, out_path: pathlib.Path, anchor: str):
    """
    who the groups are: size, mortality and severity, per site

    The single most important fact in this section is that the uncoded group is
    both the largest and much the least sick, and in a table that is two
    columns a reader has to hold in their head at once. Size as bars and
    mortality as a point on a second axis puts the two in one shape: the tall
    bar with the low dot is the whole argument.

    Per site and never pooled, for the reason the module docstring gives.
    """
    sites = groups["site"].unique(maintain_order=True).to_list()
    figure, axes = plt.subplots(
        1, len(sites), figsize=(5.0 * len(sites), 3.9), squeeze=False, sharey=True
    )
    order = [g for g in GROUPS if g in set(groups["shock_type"])]
    hues = shock_type_colors()
    for column, site in enumerate(sites):
        axis = axes[0][column]
        block = groups.filter(pl.col("site") == site)
        # Share of the site, not the raw count. NU contributes 25,007 patients
        # and UCMC 9,135, so on a shared count axis UCMC's bars are a third the
        # height of NU's whatever its composition is -- which is the one thing
        # this figure must not say, since composition within a site is the only
        # comparison the coding differences permit.
        shares, counts, deaths = [], [], []
        for group in order:
            row = block.filter(pl.col("shock_type") == group)
            shares.append(float(row["share_of_site"][0]) if row.height else 0.0)
            counts.append(int(row["n"][0]) if row.height else 0)
            deaths.append(float(row["mortality"][0]) if row.height else float("nan"))
        positions = np.arange(len(order))
        axis.bar(
            positions, shares, width=0.68,
            color=[hues.get(g, CONTEXT) for g in order], alpha=0.85,
        )
        for position, share, n in zip(positions, shares, counts):
            axis.annotate(
                f"{n:,}", (position, share), xytext=(0, 2),
                textcoords="offset points", ha="center", fontsize=6.5,
                color=INK_SECONDARY,
            )
        axis.set_title(site, fontsize=10)
        axis.set_ylabel("share of the site" if column == 0 else "")
        axis.set_xticks(positions)
        axis.set_xticklabels(order, rotation=35, ha="right", fontsize=7.5)
        axis.grid(axis="y", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)

        twin = axis.twinx()
        twin.plot(
            positions, deaths, "o", markersize=7, color=INK_PRIMARY,
            markerfacecolor="white", markeredgewidth=1.6,
        )
        twin.set_ylim(0, max(0.55, float(np.nanmax(deaths)) * 1.15))
        twin.set_ylabel(
            "in-hospital mortality" if column == len(sites) - 1 else "",
            fontsize=8,
        )
        twin.tick_params(labelsize=7)
        twin.grid(False)
        for position, value in zip(positions, deaths):
            if not np.isnan(value):
                twin.annotate(
                    f"{value:.1%}", (position, value), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=6.5,
                    color=INK_SECONDARY,
                )

    figure.suptitle(
        f"Share of each site (bars) against in-hospital mortality (points), "
        f"{anchor} anchor",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, out_path)
    plt.close(figure)


def _rate_heatmap(axis, frame: pl.DataFrame, row_key: str, columns: list,
                  labels: dict, title: str, vmax: float = 1.0):
    """a group-by-measure grid of rates, annotated, since the cells are few"""
    rows = frame[row_key].to_list()
    grid = np.full((len(rows), len(columns)), np.nan)
    for r, _ in enumerate(rows):
        for c, column in enumerate(columns):
            value = frame[column][r]
            if value is not None:
                grid[r, c] = float(value)
    image = axis.imshow(grid, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
    axis.set_xticks(np.arange(len(columns)))
    axis.set_xticklabels(
        [labels.get(c, c) for c in columns], rotation=40, ha="right", fontsize=6.5
    )
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(rows, fontsize=7)
    axis.set_title(title, fontsize=9, loc="left")
    for r in range(len(rows)):
        for c in range(len(columns)):
            if np.isnan(grid[r, c]):
                continue
            axis.text(
                c, r, f"{grid[r, c]:.0%}", ha="center", va="center", fontsize=5.5,
                color="white" if grid[r, c] < vmax * 0.55 else INK_PRIMARY,
            )
    return image


def figure_evidence(evidence: pl.DataFrame, out_path: pathlib.Path, anchor: str):
    """
    mechanism evidence from outside the shock codes, as a grid

    Six groups by six flags is a table nobody reads across, and the thing worth
    seeing is a pattern: coded septic patients light up the septic column at
    ~99% and nothing else does. A grid shows that in one glance and shows the
    weak flags as weak -- the hypovolemic column is pale everywhere, which is
    the honest reading of a flag with 21% sensitivity.
    """
    shares = [
        c for c in evidence.columns
        if c.startswith(("hint_", "culture_")) or c == "any_hint"
    ]
    if not shares:
        return
    labels = {
        "hint_septic": "infection code",
        "hint_cardiogenic": "cardiac code",
        "hint_hypovolemic": "haemorrhage code",
        "culture_sent": "culture sent",
        "culture_positive": "culture positive",
        "culture_unresulted": "culture unresulted",
        "any_hint": "any code hint",
    }
    sites = evidence["site"].unique(maintain_order=True).to_list()
    figure, axes = plt.subplots(
        1, len(sites), figsize=(4.6 * len(sites), 3.6), squeeze=False
    )
    image = None
    for column, site in enumerate(sites):
        block = (
            evidence.filter(pl.col("site") == site)
            .with_columns(
                _order=pl.col("shock_type").cast(pl.String).replace_strict(
                    {g: i for i, g in enumerate(GROUPS)}, default=99,
                    return_dtype=pl.Int32,
                )
            )
            .sort("_order")
        )
        image = _rate_heatmap(
            axes[0][column], block, "shock_type", shares, labels, site
        )
    if image is not None:
        bar = figure.colorbar(image, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
        bar.ax.tick_params(labelsize=6, labelcolor=INK_SECONDARY)
        bar.outline.set_visible(False)
    figure.suptitle(
        f"Evidence of a mechanism from outside the shock codes ({anchor} anchor)",
        fontsize=10.5, x=0.01, ha="left",
    )
    figure.supxlabel(
        "not a later diagnosis: none of these carry a timestamp",
        fontsize=7, color=INK_SECONDARY, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(figure, out_path)
    plt.close(figure)


def figure_treatments(treatments: pl.DataFrame, out_path: pathlib.Path,
                      anchor: str):
    """
    what each group was given, as a grid of rates

    This replaces the widest table in the section -- seven agents times two
    windows plus steroids, support and antibiotic timing, which ran off the
    side of the page. The question is comparative (does the uncoded group get
    treated like the septic one?) and a grid answers it by column.

    Peri-onset flags only. The admission-wide `ever_` columns are deliberately
    left in the table rather than drawn beside them: 11 found the two windows
    give opposite answers, and putting them in adjacent cells of one grid is an
    invitation to read across.
    """
    agents = [c for c in treatments.columns if c.startswith("on_")]
    extras = [c for c in ("inotrope", "mcs", "abx_within_6h") if c in treatments.columns]
    measures = agents + extras
    if not measures:
        return
    labels = {c: c.replace("on_", "").replace("_", " ") for c in measures}
    labels.update({
        "on_any_steroid": "any steroid",
        "abx_within_6h": "antibiotics < 6 h",
        "mcs": "mechanical support",
        "inotrope": "any inotrope",
    })
    sites = treatments["site"].unique(maintain_order=True).to_list()
    figure, axes = plt.subplots(
        1, len(sites), figsize=(0.52 * len(measures) * len(sites) + 3.2, 3.8),
        squeeze=False,
    )
    image = None
    for column, site in enumerate(sites):
        block = (
            treatments.filter(pl.col("site") == site)
            .with_columns(
                _order=pl.col("shock_type").cast(pl.String).replace_strict(
                    {g: i for i, g in enumerate(GROUPS)}, default=99,
                    return_dtype=pl.Int32,
                )
            )
            .sort("_order")
        )
        image = _rate_heatmap(
            axes[0][column], block, "shock_type", measures, labels, site
        )
    if image is not None:
        bar = figure.colorbar(image, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
        bar.ax.tick_params(labelsize=6, labelcolor=INK_SECONDARY)
        bar.outline.set_visible(False)
    figure.suptitle(
        f"Treatment at shock onset, by coded mechanism ({anchor} anchor)",
        fontsize=10.5, x=0.01, ha="left",
    )
    figure.supxlabel(
        "share of the group with that drug running in the six hours either side "
        "of onset; fluids, transfusion and echo are unavailable at both sites",
        fontsize=7, color=INK_SECONDARY, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.93))
    save_figure(figure, out_path)
    plt.close(figure)


def figure_neighbours(neighbours: pl.DataFrame, out_path: pathlib.Path,
                      anchor: str, k: int = 25):
    """
    what each group's nearest coded neighbours are, as stacked shares

    The kNN table is three mechanisms times three values of k times two sites,
    and the only comparison that matters is whether the uncoded group's
    neighbourhood composition differs from the coded groups' own. Stacked bars
    summing to 1 make that one read.
    """
    wanted = [f"knn{k}_{m}" for m in MECHANISMS]
    if any(c not in neighbours.columns for c in wanted):
        return
    sites = neighbours["site"].unique(maintain_order=True).to_list()
    hues = shock_type_colors()
    figure, axes = plt.subplots(
        1, len(sites), figsize=(4.4 * len(sites), 3.7), squeeze=False, sharey=True
    )
    for column, site in enumerate(sites):
        axis = axes[0][column]
        block = neighbours.filter(pl.col("site") == site)
        order = [g for g in GROUPS if g in set(block["shock_type"])]
        positions = np.arange(len(order))
        bottom = np.zeros(len(order))
        for mechanism in MECHANISMS:
            values = []
            for group in order:
                row = block.filter(pl.col("shock_type") == group)
                values.append(
                    float(row[f"knn{k}_{mechanism}"][0]) if row.height else 0.0
                )
            values = np.array(values)
            axis.bar(
                positions, values, bottom=bottom, width=0.68,
                color=hues.get(mechanism, CONTEXT), alpha=0.9,
                label=mechanism if column == 0 else None,
            )
            bottom += values
        axis.set_xticks(positions)
        axis.set_xticklabels(order, rotation=35, ha="right", fontsize=7.5)
        axis.set_title(site, fontsize=10)
        axis.set_ylabel("share of the k nearest coded neighbours"
                        if column == 0 else "")
        axis.grid(axis="y", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
    axes[0][0].legend(fontsize=7.5, frameon=False, loc="upper left")
    figure.suptitle(
        f"Mechanism composition of the {k} nearest coded neighbours "
        f"({anchor} anchor, within site)",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, out_path)
    plt.close(figure)


def figure_concordance(concordance: pl.DataFrame, out_path: pathlib.Path,
                       anchor: str):
    """
    outcome by whether the treatment matched the probe's assignment

    Four cells per site and class -- treated or not, confidently assigned or
    not -- which is sixteen rows of a table and one readable panel.

    THE SHAPE IS NOT AN EFFECT. A clinician gives an inotrope because they read
    the same record the probe reads, so the treated group is the group whose
    mechanism was legible to both. The figure is drawn because the comparison
    was asked for, and the caption says what it cannot support.
    """
    if not concordance.height or "treated_as_assigned" not in concordance.columns:
        return
    sites = concordance["site"].unique(maintain_order=True).to_list()
    classes = [
        c for c in MECHANISMS
        if c in set(concordance["probe_assignment"].to_list())
    ]
    if not classes:
        return
    figure, axes = plt.subplots(
        1, len(sites), figsize=(4.6 * len(sites), 3.8), squeeze=False, sharey=True
    )
    for column, site in enumerate(sites):
        axis = axes[0][column]
        block = concordance.filter(pl.col("site") == site)
        ticks, offset = [], 0.0
        for klass in classes:
            for confident in (True, False):
                for treated, colour in ((True, CATEGORICAL[0]), (False, CATEGORICAL[1])):
                    row = block.filter(
                        (pl.col("probe_assignment") == klass)
                        & (pl.col("probe_confident") == confident)
                        & (pl.col("treated_as_assigned") == treated)
                    )
                    if not row.height:
                        offset += 1.0
                        continue
                    n = int(row["n"][0])
                    value = float(row["mortality"][0])
                    axis.bar(
                        offset, value, width=0.78, color=colour,
                        alpha=0.9 if confident else 0.42,
                    )
                    axis.annotate(
                        f"{n:,}", (offset, value), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=5.5,
                        color=INK_SECONDARY,
                    )
                    offset += 1.0
                ticks.append((offset - 1.0, f"{klass}{chr(10)}"
                              f"{'confident' if confident else 'ambiguous'}"))
            offset += 0.6
        axis.set_xticks([t for t, _ in ticks])
        axis.set_xticklabels([label for _, label in ticks], fontsize=6,
                             rotation=30, ha="right")
        axis.set_title(site, fontsize=10)
        axis.set_ylabel("in-hospital mortality" if column == 0 else "")
        axis.grid(axis="y", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=7,
                   color=CATEGORICAL[0], label="treated as assigned"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=7,
                   color=CATEGORICAL[1], label="not treated as assigned"),
    ]
    axes[0][0].legend(handles=handles, fontsize=7, frameon=False, loc="upper left")
    figure.suptitle(
        f"Mortality by whether treatment matched the probe ({anchor} anchor)",
        fontsize=10.5,
    )
    figure.supxlabel(
        "confounded by indication and not adjustable: the clinician read the "
        "same record the probe reads. Numbers are cell sizes; pale bars are "
        "the ambiguously assigned.",
        fontsize=7, color=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.93))
    save_figure(figure, out_path)
    plt.close(figure)


def figure_proximity(proximity: pl.DataFrame, out_path: pathlib.Path, anchor: str):
    """
    cosine distance to each mechanism centroid, one panel per site

    The undifferentiated distribution against the coded patients' own. The
    comparison the spec asks for is the overlap: if the uncoded sit where the
    coded septic patients sit, the representation is saying something; if every
    group sits on top of every centroid, it is not, and the panels show that
    directly rather than through a summary statistic.
    """
    sites = proximity["site"].unique(maintain_order=True).to_list()
    figure, axes = plt.subplots(
        len(sites), len(MECHANISMS),
        figsize=(4.0 * len(MECHANISMS), 3.1 * len(sites)),
        squeeze=False,
    )
    for row, site in enumerate(sites):
        block = proximity.filter(pl.col("site") == site)
        for column, mechanism in enumerate(MECHANISMS):
            axis = axes[row][column]
            values = f"cos_{mechanism}"
            if values not in block.columns:
                axis.set_axis_off()
                continue
            coded = block.filter(
                pl.col("shock_type").cast(pl.String) == mechanism
            )[values].drop_nulls().to_numpy()
            uncoded = block.filter(
                pl.col("shock_type").cast(pl.String) == UNDIFFERENTIATED
            )[values].drop_nulls().to_numpy()
            if not len(coded) or not len(uncoded):
                axis.set_axis_off()
                continue
            bins = np.linspace(
                float(min(coded.min(), uncoded.min())),
                float(max(coded.max(), uncoded.max())),
                50,
            )
            axis.hist(
                uncoded, bins=bins, density=True, color=CONTEXT,
                label=f"undifferentiated (n={len(uncoded):,})",
            )
            axis.hist(
                coded, bins=bins, density=True, histtype="step", linewidth=1.6,
                color=HIGHLIGHT, label=f"coded {mechanism} (n={len(coded):,})",
            )
            axis.set_title(f"{site} -- distance to {mechanism}", fontsize=10)
            axis.set_xlabel("cosine distance to centroid", fontsize=9)
            axis.set_ylabel("density", fontsize=9)
            axis.legend(fontsize=7.5, frameon=False)
    figure.suptitle(
        f"How far the undifferentiated sit from each coded mechanism "
        f"({anchor} anchor, within site)",
        fontsize=12,
    )
    figure.tight_layout()
    save_figure(figure, out_path)


def figure_probe(metrics: pl.DataFrame, out_path: pathlib.Path, anchor: str):
    """AUROC per class, within-site beside cross-site"""
    if not metrics.height:
        return
    classes = [c for c in MECHANISMS if c in metrics["class"].unique().to_list()]
    scorings = ["within-site", "cross-site"]
    figure, axis = plt.subplots(figsize=(1.9 * len(classes) + 3.4, 3.8))
    width = 0.38
    positions = np.arange(len(classes))
    for i, scoring in enumerate(scorings):
        heights, errors = [], []
        for klass in classes:
            rows = metrics.filter(
                (pl.col("scoring") == scoring) & (pl.col("class") == klass)
            )
            heights.append(rows["auroc"].mean() if rows.height else np.nan)
            errors.append(
                rows["auroc"].max() - rows["auroc"].min() if rows.height > 1 else 0.0
            )
        axis.bar(
            positions + (i - 0.5) * width, heights, width,
            yerr=errors, capsize=3,
            color=CATEGORICAL[i], label=scoring,
        )
    axis.axhline(0.5, color=INK_SECONDARY, linewidth=1, linestyle="--")
    axis.text(
        len(classes) - 0.5, 0.505, "chance", fontsize=8, color=INK_SECONDARY,
        ha="right",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(classes)
    axis.set_ylim(0.4, 1.0)
    axis.set_ylabel("AUROC (one vs rest)")
    axis.set_title(
        f"Linear probe for the coded mechanism, {anchor} anchor", fontsize=11
    )
    axis.legend(fontsize=8.5, frameon=False)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    figure.tight_layout()
    save_figure(figure, out_path)


def figure_snapshots(separation: pl.DataFrame, out_path: pathlib.Path):
    """
    the same probe at 0 h, +6 h and +24 h from onset

    The point of the figure is the slope, not the level. At 0 h the model has
    read nothing after the shock began; above it sits whatever the next hours
    added, which is the patient deteriorating and the clinician acting, in
    unknown proportion. A steep rise means the later snapshots are reading
    treatment, and no probe scored on them can be called a finding about the
    patient.
    """
    if not separation.height:
        return
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    for i, klass in enumerate(MECHANISMS):
        rows = (
            separation.filter(
                (pl.col("class") == klass) & (pl.col("scoring") == "within-site")
            )
            .group_by("offset_hours")
            .agg(pl.col("auroc").mean())
            .sort("offset_hours")
        )
        if not rows.height:
            continue
        axis.plot(
            rows["offset_hours"].to_list(), rows["auroc"].to_list(),
            marker="o", linewidth=1.8, color=CATEGORICAL[i % len(CATEGORICAL)],
            label=klass,
        )
    axis.axhline(0.5, color=INK_SECONDARY, linewidth=1, linestyle="--")
    axis.set_xlabel("hours from shock onset the embedding was taken")
    axis.set_ylabel("AUROC (within site, one vs rest)")
    axis.set_title(
        "Mechanism separation against how much post-onset record the model read",
        fontsize=10.5,
    )
    axis.legend(fontsize=9, frameon=False)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    figure.tight_layout()
    save_figure(figure, out_path)


def figure_entropy(predictions: pl.DataFrame, out_path: pathlib.Path, anchor: str):
    """
    how certain the probe is, by what the coders said

    The validation the config describes. `mixed` is excluded from training, so
    if the entropy carries information those patients should sit higher on it
    than patients the coders called one thing -- and the undifferentiated, who
    the coders called nothing, are the group the figure exists to place.
    """
    if not predictions.height or "probe_entropy" not in predictions.columns:
        return
    groups = [
        g
        for g in GROUPS
        if predictions.filter(
            pl.col("shock_type").cast(pl.String) == g
        )["probe_entropy"].drop_nulls().len()
        > 20
    ]
    if not groups:
        return
    figure, axis = plt.subplots(figsize=(1.15 * len(groups) + 3.2, 4.0))
    data = [
        predictions.filter(pl.col("shock_type").cast(pl.String) == g)[
            "probe_entropy"
        ].drop_nulls().to_numpy()
        for g in groups
    ]
    parts = axis.violinplot(data, showmedians=True, widths=0.85)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(HIGHLIGHT if groups[i] == UNDIFFERENTIATED else CONTEXT)
        body.set_alpha(0.85)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color(INK_SECONDARY)
            parts[key].set_linewidth(1.0)
    axis.set_xticks(np.arange(1, len(groups) + 1))
    axis.set_xticklabels(
        [f"{g}\n(n={len(d):,})" for g, d in zip(groups, data)], fontsize=8
    )
    axis.set_ylabel("normalised entropy of the probe (1 = no information)")
    axis.set_title(
        f"How undecided the probe is, by what the coders said ({anchor} anchor)",
        fontsize=10.5,
    )
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    figure.tight_layout()
    save_figure(figure, out_path)


def figure_overlay(paths, frame: pl.DataFrame, out_path: pathlib.Path, anchor: str,
                   logger):
    """
    the undifferentiated overlaid on the coded mechanisms, in 03's projection

    Read from the projection 03 saved rather than recomputed, so this is the
    same layout as the rest of the report and the panels can be compared to it.

    One panel per mechanism, that mechanism in a single highlight hue over
    every other patient in grey. Four categories cannot be separated by colour
    in one scatter -- the palette validates three slots under the all-pairs
    gate -- so the identity is carried by the panel title instead.
    """
    path = paths.patient_level_home / f"projections_anchor-{anchor}.parquet"
    if not path.is_file():
        logger.warning(f"  no {path.name}; overlay figure skipped (run 03)")
        return
    projections = pl.read_parquet(path)
    columns = [c for c in ("pacmap_1", "pacmap_2") if c in projections.columns]
    if len(columns) < 2:
        logger.warning("  projections carry no pacmap coordinates; overlay skipped")
        return
    # A projection written under the pooled role is keyed by the salted
    # `row_key`, because record numbers do not leave a site. There is nothing to
    # join it to here, and inventing a key would silently pair the wrong
    # patients with the wrong coordinates.
    if "hospitalization_id" not in projections.columns:
        logger.warning(
            f"  {path.name} is keyed by row_key, not hospitalization_id -- it "
            "was written under the pooled role. Overlay skipped; re-run 03 for "
            f"anchor {anchor} as a site to draw it."
        )
        return
    joined = frame.select("hospitalization_id", "site", "shock_type").join(
        projections.select("hospitalization_id", "site", *columns),
        on=["hospitalization_id", "site"],
        how="inner",
    )
    if not joined.height:
        logger.warning("  projections do not join the cohort; overlay skipped")
        return

    panels = [UNDIFFERENTIATED, *MECHANISMS]
    figure, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.5))
    x, y = joined[columns[0]].to_numpy(), joined[columns[1]].to_numpy()
    labels = joined["shock_type"].cast(pl.String).to_numpy()
    for axis, panel in zip(np.atleast_1d(axes), panels):
        members = labels == panel
        axis.scatter(x, y, s=2, c=CONTEXT, linewidths=0, rasterized=True)
        axis.scatter(
            x[members], y[members], s=2.5, c=HIGHLIGHT, linewidths=0,
            rasterized=True,
        )
        axis.set_title(f"{panel} (n={int(members.sum()):,})", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"Where each group sits in the {anchor}-anchor PaCMAP projection",
        fontsize=11.5,
    )
    figure.tight_layout()
    save_figure(figure, out_path)


# ================================================================== summary


def _md_table(frame: pl.DataFrame, columns, formats=None) -> list[str]:
    """a markdown table of the named columns, numerics right-aligned"""
    formats = formats or {}
    columns = [c for c in columns if c in frame.columns]
    if not columns:
        return []
    numeric = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64)
    alignment = [
        "---:" if frame.schema[c] in numeric else "---" for c in columns
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(alignment) + "|",
    ]
    for row in frame.select(columns).iter_rows(named=True):
        cells = []
        for column in columns:
            value = row[column]
            if value is None:
                cells.append("")
            elif column in formats:
                cells.append(formats[column](value))
            elif isinstance(value, bool):
                cells.append("yes" if value else "no")
            elif isinstance(value, float):
                cells.append(f"{value:.3f}")
            elif isinstance(value, int):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _pct(value) -> str:
    return "" if value is None else f"{value:.1%}"


def write_summary(path: pathlib.Path, results: dict, anchor: str, logger) -> None:
    """the written findings, with the limitations that decide how to read them"""
    lines = [f"# Undifferentiated shock (anchor `{anchor}`)", ""]
    lines += [
        "Patients meeting the shock criteria with no coded mechanism: what they",
        "resemble in the frozen model's representation, whether a linear probe",
        "will assign them a mechanism, and whether they were treated as though",
        "someone had.",
        "",
        "> **Every comparison is within site.** The two hospitals leave a similar",
        "> share uncoded (NU 56.6%, UCMC 48.9%) but reach for different mechanisms",
        "> when they do code one: UCMC codes cardiogenic 2.5 times as often, NU",
        "> codes septic 1.7 times as often. The coded patients define the",
        "> centroids and train the probe, so that mix difference means neither",
        "> transports, and pooling would let coding practice read as a finding",
        "> about mechanism.",
        "",
    ]

    groups = results.get("groups")
    if groups is not None and groups.height:
        lines += ["## Who the groups are", ""]
        lines += _md_table(
            groups,
            ["site", "shock_type", "n", "share_of_site", "mortality",
             "median_age", "median_sofa", "median_lactate", "pct_on_vasoactive"],
            {
                "share_of_site": _pct,
                "mortality": _pct,
                "pct_on_vasoactive": _pct,
                "median_age": lambda v: f"{v:.0f}",
                "median_sofa": lambda v: f"{v:.0f}",
                "median_lactate": lambda v: f"{v:.1f}",
                "median_icu_los": lambda v: f"{v:.1f}",
                "pct_female": _pct,
            },
        )
        lines.append("")

    evidence = results.get("evidence")
    if evidence is not None and evidence.height:
        lines += [
            "## Evidence of a mechanism from somewhere else",
            "",
            "**This is not a later diagnosis.** The hospital diagnosis table has",
            "no timestamp, and an uncoded patient has no shock code at any point",
            "in the admission by definition, so whether a mechanism was",
            "recognised *after* the shock is not answerable with this data. What",
            "follows is whether a mechanism was recorded by some other means",
            "anywhere in the admission: a positive culture, or a non-shock code",
            "implying one.",
            "",
        ]
        shares = ["hint_septic", "hint_cardiogenic", "hint_hypovolemic",
                  "culture_sent", "culture_positive", "culture_unresulted",
                  "any_hint"]
        lines += _md_table(
            evidence,
            ["site", "shock_type", "n", *shares],
            {c: _pct for c in shares},
        )
        lines.append("")

    reference = results.get("reference")
    if reference is not None and reference.height:
        lines += [
            "## What a distance means, on patients whose label is known",
            "",
            "The benchmark. A mean cosine distance from the uncoded to the septic",
            "centroid is uninterpretable until the coded patients have their own",
            "distances beside it: if coded septic patients sit as far from their",
            "own centroid as coded cardiogenic ones do, the mechanisms are not",
            "separated and no distance carries information. Own-centroid",
            "distances are leave-one-out, so a patient is never scored against a",
            "centroid they helped define.",
            "",
        ]
        lines += _md_table(
            reference,
            ["site", "label", "centroid", "n", "mean_cosine", "median_cosine",
             "is_own_centroid"],
            {"mean_cosine": lambda v: f"{v:.4f}",
             "median_cosine": lambda v: f"{v:.4f}"},
        )
        lines.append("")

    metrics = results.get("probe_metrics")
    if metrics is not None and metrics.height:
        lines += [
            "## The probe",
            "",
            "Multinomial logistic regression on the 50-d PCA scores, grouped by",
            "patient so the same person cannot straddle a fold boundary. The",
            "cross-site rows are the ones that matter: a probe trained and tested",
            "inside one hospital can reach a high AUROC by learning that",
            "hospital's documentation habits, and the drop from within-site to",
            "cross-site is how much of the within-site number was exactly that.",
            "",
        ]
        lines += _md_table(
            metrics,
            ["scoring", "trained_on", "tested_on", "class", "n", "n_positive",
             "auroc", "brier"],
        )
        lines.append("")

    separation = results.get("snapshots")
    if separation is not None and separation.height:
        lines += [
            "## How much of the separation is treatment leaking back in",
            "",
            "The medication vocabulary is part of what the model reads, so an",
            "embedding taken at +24 h has seen the antibiotics and the inotrope",
            "the working diagnosis produced. At 0 h it has read nothing after the",
            "shock began. The rise from 0 h is attributable to what happened",
            "next, which is the patient deteriorating *and* the clinician acting,",
            "in a proportion nothing here can separate.",
            "",
        ]
        rolled = (
            separation.filter(pl.col("scoring") == "within-site")
            .group_by("offset_hours", "class")
            .agg(pl.col("auroc").mean(), pl.col("n").sum())
            .sort("class", "offset_hours")
        )
        lines += _md_table(rolled, ["class", "offset_hours", "n", "auroc"])
        lines.append("")

    for key, heading in (
        ("treatments", "What each group was given"),
        ("first_agents", "Which vasoactive came first"),
        ("concordance", "Treated as the probe would have assigned them"),
        ("concordance_adjusted", "The same contrast with severity in the model"),
    ):
        table = results.get(key)
        if table is not None and table.height:
            lines += [f"## {heading}", ""]
            lines += _md_table(table, table.columns)
            lines.append("")

    lines += ["## Limitations", ""]
    lines += [
        "**The group is defined by a coding decision, not a clinical one.** A",
        "patient with no shock code may have had an undifferentiated",
        "presentation, or may have been straightforwardly septic and poorly",
        "documented. The two are indistinguishable here, and the 2.5-fold",
        "between-site gap in *which* mechanism gets coded says the second",
        "happens often enough to matter.",
        "",
        "**The uncoded are mostly not a diagnostic puzzle.** Their in-hospital",
        "mortality is 6.2% pooled -- 4.6% at NU, 11.2% at UCMC -- against 25 to",
        "43% for every coded group. Most of them",
        "are a mild population the physiological criteria admitted and nobody",
        "thought was in shock, so a result that they sit far from every mechanism",
        "centroid is in the first instance a result about severity.",
        "",
        "**A probe is not a diagnosis.** It is fitted to ICD codes assigned over a",
        "whole admission by people who had the chart, the imaging and the",
        "follow-up. An AUROC above chance says the pre-onset record carries",
        "mechanism-relevant signal; it does not say the assignment is right for",
        "any individual patient, and the calibration table is the relevant check",
        "rather than the AUROC.",
        "",
        "**Confounding by indication governs the concordance table.** Treatment",
        "follows the clinician's reading of the same record the probe reads. A",
        "concordant patient is one where two readings agree, and outcomes differ",
        "between concordant and discordant partly because the clinician was right",
        "and partly because a clinician who reaches a confident diagnosis is",
        "looking at a different patient. Adjusting for severity removes the part",
        "severity carries and leaves the judgement, which is unmeasured.",
        "",
    ]
    for name, reason in NOT_ASSESSABLE:
        lines += [f"**Not assessable: {name}.** {reason}", ""]
    for klass, signature in TREATMENT_SIGNATURE.items():
        if not signature["ascertainable"]:
            lines += [
                f"**The {klass} class has no scoreable treatment.** "
                + signature["means"],
                "",
            ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  summary -> {path}")


# ================================================================= analysis


def summarise_evidence(frame: pl.DataFrame, cfg) -> pl.DataFrame:
    """mechanism-evidence rates per (site, group)"""
    hints = [f"hint_{name}" for name in cfg.later_evidence.code_groups]
    present = [c for c in hints if c in frame.columns]
    aggregations = [pl.len().alias("n")]
    aggregations += [pl.col(c).mean().alias(c) for c in present]
    for column in ("culture_sent", "culture_positive", "culture_unresulted"):
        if column in frame.columns:
            aggregations.append(pl.col(column).mean().alias(column))
    if present:
        aggregations.append(
            pl.any_horizontal(*(pl.col(c) for c in present)).mean().alias("any_hint")
        )
    return (
        frame.group_by("site", "shock_type").agg(aggregations).sort("site", "shock_type")
    )


def analyse(paths, anchor: str, cfg, args, logger) -> None:
    """one anchor, start to finish"""
    logger.info(f"=== anchor {anchor} ===")
    frame = load_anchor(paths, anchor, logger)
    frame = attach_cohort(paths, frame, cfg, logger)

    results: dict[str, pl.DataFrame] = {}

    # ----------------------------------------------------------- the cohort
    logger.info("the groups:")
    results["groups"] = describe_groups(frame, logger)

    logger.info("evidence of a mechanism from elsewhere:")
    evidence_parts = []
    for site in paths.sites:
        part = frame.filter(pl.col("site") == site)
        if not part.height:
            continue
        evidence_parts.append(
            part.select("hospitalization_id", "site", "shock_type").join(
                mechanism_evidence(paths.site_data[site], part, cfg, logger),
                on="hospitalization_id",
                how="left",
            )
        )
    if evidence_parts:
        evidence = pl.concat(evidence_parts, how="diagonal_relaxed")
        results["evidence"] = summarise_evidence(evidence, cfg)
        frame = frame.join(
            evidence.drop("site", "shock_type"),
            on="hospitalization_id",
            how="left",
        )

    # --------------------------------------------------------- the geometry
    logger.info("distance to each mechanism centroid:")
    proximity = centroid_proximity(frame, cfg, args.seed, logger)
    results["proximity"] = proximity
    results["reference"] = reference_separation(proximity, logger)

    logger.info("neighbour composition:")
    neighbours = knn_composition(frame, cfg, args.seed, logger)
    if neighbours.height:
        results["neighbours"] = (
            neighbours.group_by("site", "shock_type")
            .agg(
                pl.len().alias("n"),
                *(
                    pl.col(c).mean().alias(c)
                    for c in neighbours.columns
                    if c.startswith("knn") and not c.endswith("majority")
                ),
            )
            .sort("site", "shock_type")
        )

    # ------------------------------------------------------------ the probe
    logger.info("the probe:")
    probe = fit_probe(frame, cfg, args.seed, logger)
    if probe:
        results["probe_metrics"] = probe["metrics"]
        results["probe_calibration"] = probe["calibration"]
        results["probe_predictions"] = probe["predictions"]
        frame = frame.join(
            probe["predictions"].drop("site", "shock_type"),
            on="hospitalization_id",
            how="left",
        )

        # does the probe agree with the coders where both spoke, and with the
        # other evidence where the coders did not
        if "hint_septic" in frame.columns:
            # The only external check available on an assignment: a patient the
            # probe calls septic ought to carry an infection code or a positive
            # culture more often than one it calls cardiogenic. Agreement here
            # is weak evidence -- both readings come from the same record -- but
            # disagreement would be strong evidence against the probe.
            checks = [pl.len().alias("n")]
            checks += [
                pl.col(f"hint_{m}").mean().alias(f"hint_{m}")
                for m in MECHANISMS
                if f"hint_{m}" in frame.columns
            ]
            if "culture_positive" in frame.columns:
                checks.append(
                    pl.col("culture_positive").mean().alias("culture_positive")
                )
            checks.append(
                pl.col("in_hospital_mortality").mean().alias("mortality")
            )
            results["probe_vs_evidence"] = (
                frame.filter(
                    (pl.col("shock_type").cast(pl.String) == UNDIFFERENTIATED)
                    & pl.col("probe_assignment").is_not_null()
                )
                .group_by("site", "probe_assignment", "probe_confident")
                .agg(checks)
                .sort("site", "probe_assignment", "probe_confident")
            )

    # -------------------------------------------------------- the snapshots
    if not args.skip_snapshots:
        logger.info("the same probe at each snapshot offset:")
        separation, assignments = snapshot_separation(
            paths, frame, cfg, args.seed, logger
        )
        if separation.height:
            results["snapshots"] = separation
            results["snapshot_assignments"] = assignments

    # -------------------------------------------------------- the treatments
    logger.info("treatment:")
    results["treatments"] = treatment_profile(frame, cfg, logger)
    mix = first_agent_mix(frame, logger)
    if mix.height:
        results["first_agents"] = mix
    if "probe_assignment" in frame.columns:
        concordance = assignment_concordance(frame, logger)
        if concordance.height:
            results["concordance"] = concordance
        scored = frame.filter(
            (pl.col("shock_type").cast(pl.String) == UNDIFFERENTIATED)
            & pl.col("probe_assignment").is_in(
                [k for k, v in TREATMENT_SIGNATURE.items() if v["ascertainable"]]
            )
        ).with_columns(
            treated_as_assigned=pl.when(pl.col("probe_assignment") == "septic")
            .then(pl.col("abx_within_6h_flag"))
            .otherwise(pl.col("cardiogenic_treatment"))
        )
        adjusted = concordance_adjusted(scored, logger)
        if adjusted.height:
            results["concordance_adjusted"] = adjusted

    # ----------------------------------------------------------- the outputs
    paths.table_home.mkdir(parents=True, exist_ok=True)
    per_patient = ("proximity", "probe_predictions", "snapshot_assignments")
    for name, table in results.items():
        if table is None or not table.height:
            continue
        if name in per_patient:
            # one row per patient: parquet, beside the other patient-level
            # files, so a 34k-row table does not land in the CSV directory the
            # report reads
            table.write_parquet(
                paths.patient_level_home / f"undiff_{name}_{anchor}.parquet"
            )
            logger.info(f"  undiff_{name}_{anchor}.parquet ({table.height:,} rows)")
            continue
        table.write_csv(paths.table_home / f"undiff_{name}_{anchor}.csv")
        logger.info(f"  undiff_{name}_{anchor}.csv ({table.height} rows)")

    # what could not be assessed, as a table so the report can render it
    pl.DataFrame(
        {
            "comparison": [name for name, _ in NOT_ASSESSABLE],
            "why_not": [reason for _, reason in NOT_ASSESSABLE],
        }
    ).write_csv(paths.table_home / "undiff_not_assessable.csv")

    style()
    figure_home = paths.figure_home / f"anchor-{anchor}"
    figure_home.mkdir(parents=True, exist_ok=True)
    # the table-shaped results, drawn. Each of these replaces a table that was
    # too wide to read across, and the tables stay in the report beneath them.
    for key, draw in (
        ("groups", figure_groups),
        ("evidence", figure_evidence),
        ("treatments", figure_treatments),
    ):
        table = results.get(key)
        if table is not None and table.height:
            draw(table, figure_home / f"undiff_{key}.png", anchor)
    if results.get("neighbours") is not None and results["neighbours"].height:
        figure_neighbours(
            results["neighbours"], figure_home / "undiff_neighbours.png", anchor
        )
    if results.get("concordance") is not None and results["concordance"].height:
        figure_concordance(
            results["concordance"], figure_home / "undiff_concordance.png", anchor
        )
    if results.get("proximity") is not None and results["proximity"].height:
        figure_proximity(
            results["proximity"], figure_home / "undiff_proximity.png", anchor
        )
    if results.get("probe_metrics") is not None and results["probe_metrics"].height:
        figure_probe(
            results["probe_metrics"], figure_home / "undiff_probe.png", anchor
        )
    if results.get("probe_predictions") is not None:
        figure_entropy(
            results["probe_predictions"],
            figure_home / "undiff_entropy.png",
            anchor,
        )
    if results.get("snapshots") is not None and results["snapshots"].height:
        figure_snapshots(results["snapshots"], figure_home / "undiff_snapshots.png")
    figure_overlay(paths, frame, figure_home / "undiff_overlay.png", anchor, logger)
    logger.info(f"  figures -> {figure_home}")

    write_summary(
        paths.output_home / f"undifferentiated_summary_{anchor}.md",
        results,
        anchor,
        logger,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--anchor", choices=ANCHORS, help="one anchor only")
    parser.add_argument("--model", help="which checkpoint's embeddings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="skip the 0/+6/+24h leakage check, which needs 09's grid",
    )
    args = parser.parse_args()

    paths = load_paths(model=args.model) if args.model else load_paths()
    paths.mkdirs()
    logger = get_logger(
        "13_undifferentiated", paths.log_home / "13_undifferentiated.log"
    )
    cfg = OmegaConf.load(
        CONFIG_HOME / "undifferentiated.yaml"
    ).undifferentiated

    for anchor in ([args.anchor] if args.anchor else ANCHORS):
        analyse(paths, anchor, cfg, args, logger)
    logger.info("13 done")


if __name__ == "__main__":
    main()

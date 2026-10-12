#!/usr/bin/env python3

"""
10 -- how the two sites differ, and how each model sees the prediction targets

Two figures the report needs, which do not belong to any earlier script:

  site features   distributions of the clinical variables at each site, so the
                  site effect in the embedding can be read against how
                  different the patients and their charting actually are
  target UMAPs    each checkpoint's embedding, coloured by the thing being
                  predicted, side by side -- does a model that separates the
                  outcome visually also predict it better?

Usage
    python code/10_site_features.py
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clifcs import ANCHORS, get_logger, load_cohort_cfg, load_paths
from clifcs.figures import key_string, save_figure, style
from clifcs.palette import (
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    husl_colors,
    site_colors,
)

MODELS = ("GEM-all", "GEM-ucmc-icu", "GEM-mimic-icu")

#: continuous cohort variables worth comparing between sites
CONTINUOUS = [
    ("age_at_admission", "age (years)"),
    ("lactate", "lactate at onset (mmol/L)"),
    ("creatinine", "creatinine at onset (mg/dL)"),
    ("bicarbonate", "bicarbonate at onset (mmol/L)"),
    ("egfr", "eGFR (mL/min/1.73m2)"),
    ("platelet_count", "platelets at onset (10^3/uL)"),
    ("icu_los_days", "ICU length of stay (days)"),
    ("hospital_los_days", "hospital length of stay (days)"),
    ("hours_admission_to_cs_onset", "admission to shock onset (h)"),
]

#: proportions worth comparing between sites
PROPORTIONS = [
    ("in_hospital_mortality", "died in hospital"),
    ("cardiac_origin", "cardiac origin"),
    ("met_sustained_hypotension", "SBP < 90 for > 30 min"),
    ("met_vasoactive_support", "vasoactive infusion"),
    ("met_ams", "altered mental status"),
    ("met_lactate", "lactate >= 2"),
    ("met_aki", "creatinine AKI"),
    ("cardiac_inotrope", "inotrope infusion"),
    ("cardiac_mcs", "MCS procedure"),
    ("cardiac_troponin", "troponin >= 500"),
    # coded mechanism. Not mutually exclusive: a patient can carry more than one
    # shock code, which is what the `mixed` level of `shock_type` collapses.
    ("shock_septic", "coded septic shock"),
    ("shock_cardiogenic", "coded cardiogenic shock"),
    ("shock_hypovolemic", "coded hypovolemic shock"),
    ("shock_other", "coded other/unspecified shock"),
]


def plot_site_features(cohort: pl.DataFrame, paths, out, logger) -> pl.DataFrame:
    """
    distributions and proportions side by side, one colour per site

    Drawn because the embedding separates the sites almost perfectly and the
    obvious question is whether the patients differ that much. Mostly they do
    not, which is what makes the separation a documentation finding.
    """
    sites = [s for s in paths.sites if s in set(cohort["site"].to_list())]
    colours = site_colors(sites)
    available = [(c, label) for c, label in CONTINUOUS if c in cohort.columns]
    proportions = [(c, label) for c, label in PROPORTIONS if c in cohort.columns]

    columns = 4
    rows = int(np.ceil(len(available) / columns)) + 1
    figure, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 2.9 * rows),
                                squeeze=False)

    rowsout = []
    for position, (column, label) in enumerate(available):
        ax = axes[position // columns][position % columns]
        series = {}
        for site in sites:
            values = (
                cohort.filter(pl.col("site") == site)[column]
                .cast(pl.Float64)
                .drop_nulls()
                .to_numpy()
            )
            series[site] = values
        pooled = np.concatenate(list(series.values())) if series else np.array([])
        if pooled.size < 50:
            ax.axis("off")
            continue
        lo, hi = np.nanpercentile(pooled, [1, 99])
        bins = np.linspace(lo, hi, 36)
        for site in sites:
            ax.hist(np.clip(series[site], lo, hi), bins=bins, density=True,
                    histtype="step", linewidth=1.4, color=colours[site], label=site)
            rowsout.append({
                "variable": label, "site": site,
                "median": float(np.nanmedian(series[site])),
                "p25": float(np.nanpercentile(series[site], 25)),
                "p75": float(np.nanpercentile(series[site], 75)),
                "n": int(series[site].size),
            })
        ax.set_title(label, loc="left", fontsize=7.8, pad=4)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        if position == 0:
            ax.legend(fontsize=6.5, labelcolor=INK_SECONDARY, handlelength=1.2)

    for position in range(len(available), rows * columns - columns):
        axes[position // columns][position % columns].axis("off")

    # the proportions go in one wide panel across the bottom row
    gs = axes[rows - 1][0].get_gridspec()
    for ax in axes[rows - 1]:
        ax.remove()
    wide = figure.add_subplot(gs[rows - 1, :])
    labels = [label for _, label in proportions]
    y = np.arange(len(labels))
    height = 0.8 / max(len(sites), 1)
    for offset, site in enumerate(sites):
        part = cohort.filter(pl.col("site") == site)
        values = [
            float(part[column].cast(pl.Boolean).fill_null(False).mean())
            for column, _ in proportions
        ]
        wide.barh(y + offset * height - 0.4 + height / 2, values, height=height,
                  color=colours[site], label=site, linewidth=0)
        for column, label in proportions:
            rowsout.append({
                "variable": label, "site": site,
                "median": float(part[column].cast(pl.Boolean).fill_null(False).mean()),
                "p25": None, "p75": None, "n": part.height,
            })
    wide.set_yticks(y, labels, fontsize=7)
    wide.invert_yaxis()
    wide.set_xlabel("share of the site's cohort", fontsize=7.5)
    wide.tick_params(labelsize=6.5)
    wide.grid(True, axis="x", color=GRID, linewidth=0.4)
    wide.set_axisbelow(True)
    for side in ("top", "right", "left"):
        wide.spines[side].set_visible(False)
    wide.legend(fontsize=6.5, labelcolor=INK_SECONDARY, loc="lower right")

    figure.suptitle("How the sites differ clinically", x=0.02, ha="left",
                    fontsize=10, y=0.998)
    figure.tight_layout(rect=(0, 0.01, 1, 0.97))
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out.name}")
    return pl.DataFrame(rowsout)


def plot_target_umaps(paths, anchor: str, out, logger) -> None:
    """
    each checkpoint's embedding, coloured by the prediction targets

    One row per target, one column per checkpoint, every panel on its own UMAP
    of that checkpoint's embedding. If a model that visibly separates an
    outcome also predicts it better, the two should agree; section 6 says they
    largely do not.
    """
    labels_file = paths.cohort_home / f"downstream_labels_anchor-{anchor}.parquet"
    if not labels_file.is_file():
        logger.warning(f"no downstream labels for {anchor}; run 06 first")
        return
    labels = pl.read_parquet(labels_file)
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet").select(
        "hospitalization_id", "site", "in_hospital_mortality"
    )

    targets = [("site", "contributing site")]
    for task in ("mortality_24h", "new_imv_24h", "pressor_escalation_24h"):
        if f"{task}__label" in labels.columns:
            targets.append((task, task.replace("_", " ")))

    figure, axes = plt.subplots(
        len(targets), len(MODELS),
        figsize=(3.5 * len(MODELS), 3.2 * len(targets)), squeeze=False,
    )
    two = husl_colors(range(2))
    binary = {"false": list(two.values())[0], "true": list(two.values())[1]}

    for column, model in enumerate(MODELS):
        model_paths = load_paths(model=model)
        sources = sorted(
            p for p in model_paths.patient_level_home.glob(
                f"embeddings_anchor-{anchor}_*.parquet")
            if "_share-safe" not in p.name and "_corrected-" not in p.name
        )
        if not sources:
            for row in range(len(targets)):
                axes[row][column].axis("off")
            logger.warning(f"{model}: no embeddings for {anchor}")
            continue
        frame = pl.concat([pl.read_parquet(p) for p in sources],
                          how="diagonal_relaxed")
        frame = frame.join(cohort.drop("site"), on="hospitalization_id", how="inner")
        frame = frame.join(labels.drop("site"), on="hospitalization_id", how="left")

        embedding = frame.select(
            sorted(c for c in frame.columns if c.startswith("emb_"))
        ).to_numpy().astype(np.float64)
        import umap

        coords = umap.UMAP(n_neighbors=30, min_dist=0.2, random_state=42).fit_transform(
            PCA(n_components=50, random_state=42).fit_transform(
                StandardScaler().fit_transform(embedding)
            )
        )
        logger.info(f"{model}: projected {len(coords):,} patients")

        for row, (target, label) in enumerate(targets):
            ax = axes[row][column]
            if target == "site":
                colours = site_colors(paths.sites)
                values = [key_string(v) for v in frame["site"].to_list()]
                lookup = {key_string(k): v for k, v in colours.items()}
                handles = [(s, colours[s]) for s in paths.sites]
            else:
                eligible = frame[f"{target}__eligible"].fill_null(False).to_numpy()
                values = [
                    key_string(v) if ok else "none"
                    for v, ok in zip(
                        frame[f"{target}__label"].fill_null(False).to_list(), eligible
                    )
                ]
                lookup = {**binary, "none": "#e4e3df"}
                handles = [("no", binary["false"]), ("yes", binary["true"]),
                           ("not eligible", "#e4e3df")]
            ax.scatter(coords[:, 0], coords[:, 1], s=1.4,
                       c=[lookup.get(v, "#e4e3df") for v in values],
                       linewidths=0, alpha=0.8, rasterized=True)
            if column == 0:
                ax.set_ylabel(label, fontsize=8, color=INK_PRIMARY)
            if row == 0:
                ax.set_title(model, loc="left", fontsize=9, pad=5)
            if column == len(MODELS) - 1:
                ax.legend(
                    handles=[
                        plt.Line2D([], [], marker="o", linestyle="none",
                                   markersize=4, markerfacecolor=c,
                                   markeredgecolor="none", label=n)
                        for n, c in handles
                    ],
                    loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6,
                    labelcolor=INK_SECONDARY, handletextpad=0.5, borderpad=0,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)

    figure.suptitle(
        f"Each checkpoint's embedding at the {anchor} anchor, coloured by target",
        x=0.02, ha="left", fontsize=10, y=0.999,
    )
    figure.tight_layout(rect=(0, 0.01, 1, 0.975))
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", choices=ANCHORS, default="24h")
    args = parser.parse_args()

    paths = load_paths()
    paths.mkdirs()
    load_cohort_cfg()
    logger = get_logger("10_site_features", paths.log_home / "10_site_features.log")
    style()

    shared = paths.output_home / "figures" / "shared"
    shared.mkdir(parents=True, exist_ok=True)

    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    table = plot_site_features(cohort, paths, shared / "site_features.png", logger)
    table.write_csv(paths.output_home / "tables" / "site_features.csv")

    plot_target_umaps(
        paths, args.anchor, shared / f"target_umaps_anchor-{args.anchor}.png", logger
    )
    logger.info("10 done")


if __name__ == "__main__":
    main()

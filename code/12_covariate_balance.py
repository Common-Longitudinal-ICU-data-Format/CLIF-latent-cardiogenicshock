#!/usr/bin/env python3

"""
Love plot: covariate balance across the longitudinal clusters

A love plot was built for two arms, where one dot per covariate shows the
standardised mean difference between treated and control before and after
weighting. There are four clusters here, so "before and after" is replaced by
one dot per cluster: the SMD of each baseline covariate in that cluster against
the pooled remainder.

WHY IT MATTERS FOR THE ADJUSTED ODDS RATIOS. A regression can report an
adjusted effect for a cluster whose covariate distribution barely overlaps the
others', but that number is extrapolation, not comparison. |SMD| above 0.1 is
the usual flag for imbalance and above 0.25 for severe imbalance; a cluster
sitting out past 0.25 on the covariates that drive mortality is one whose
adjusted odds ratio should not be read as "like-for-like".

Baseline is measured AT SHOCK ONSET, before the trajectories diverge. SOFA now
comes from sofa_grid.parquet -- the worst score in the 24h ending at onset --
which is the covariate that was unavailable while sofa_total was null.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, "code")
from clifcs import get_logger, load_paths
from clifcs.figures import save_figure
from clifcs.palette import (
    GRID, INK_PRIMARY, INK_SECONDARY, colors_for, husl_colors,
)

logger = get_logger("love")
paths = load_paths(role="site")

COVARIATES = [
    ("sofa_total", "SOFA (worst in 24h to onset)"),
    ("map", "MAP"),
    ("lactate", "lactate"),
    ("nee", "norepinephrine equivalents"),
    ("n_vasopressors", "vasoactive agents"),
    ("gcs_total", "GCS total"),
    ("creatinine", "creatinine"),
    ("platelet_count", "platelets"),
    ("on_imv", "invasive ventilation"),
    ("age_at_admission", "age"),
]

labels = pl.read_parquet(
    paths.patient_level_home / "longitudinal_clusters_septic.parquet"
)
cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet").select(
    "hospitalization_id", "age_at_admission", "in_hospital_mortality"
)
state = (
    pl.read_parquet(paths.patient_level_home / "trajectory_state.parquet")
    .filter(pl.col("offset_hours") == 0.0)
    .drop("sofa_total")
)
sofa = (
    pl.read_parquet(paths.patient_level_home / "sofa_grid.parquet")
    .filter(pl.col("offset_hours") == 0.0)
    .select("hospitalization_id", "sofa_total")
)
frame = (
    labels.join(cohort, on="hospitalization_id", how="inner")
    .join(state.drop("offset_hours"), on="hospitalization_id", how="left")
    .join(sofa, on="hospitalization_id", how="left")
)
logger.info(f"{frame.height:,} patients with cluster, outcome and baseline")

present = [(c, n) for c, n in COVARIATES if c in frame.columns]


def smd(part, rest, column):
    """standardised mean difference, pooled SD, nulls dropped pairwise"""
    a = part[column].cast(pl.Float64).drop_nulls().to_numpy()
    b = rest[column].cast(pl.Float64).drop_nulls().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled == 0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


rows = []
for column_name, display in present:
    for stratification in ("growth_cluster", "perfusion_cluster",
                           "sequence_cluster"):
        if stratification not in frame.columns:
            continue
        for level in sorted(frame[stratification].drop_nulls().unique().to_list()):
            part = frame.filter(pl.col(stratification) == level)
            rest = frame.filter(
                (pl.col(stratification) != level)
                & pl.col(stratification).is_not_null()
            )
            rows.append({
                "stratification": stratification,
                "cluster": level,
                "covariate": display,
                "smd": smd(part, rest, column_name),
            })

table = pl.DataFrame(rows)
out_csv = (
    paths.output_home / "tables" / paths.model_key / "covariate_balance_septic.csv"
)
out_csv.parent.mkdir(parents=True, exist_ok=True)
table.write_csv(out_csv)
logger.info(f"wrote {out_csv}")

for stratification, tag in (
    ("growth_cluster", "growth"),
    ("perfusion_cluster", "perfusion"),
    ("sequence_cluster", "sequence"),
):
    part = table.filter(pl.col("stratification") == stratification)
    if part.is_empty():
        continue
    clusters = sorted(part["cluster"].unique().to_list())
    colours = colors_for(stratification, clusters)
    names = [n for _, n in present][::-1]

    figure, ax = plt.subplots(figsize=(6.6, 0.42 * len(names) + 1.6))
    ax.axvspan(-0.1, 0.1, color="#eef1ee", zorder=0)
    for threshold in (-0.25, 0.25):
        ax.axvline(threshold, color=GRID, linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0.0, color=INK_SECONDARY, linewidth=0.8, zorder=1)

    for cluster in clusters:
        values = [
            part.filter(
                (pl.col("cluster") == cluster) & (pl.col("covariate") == name)
            )["smd"].to_list()
            for name in names
        ]
        xs = [v[0] if v and v[0] is not None else np.nan for v in values]
        ax.scatter(xs, range(len(names)), s=34, color=colours[cluster],
                   edgecolors="#ffffff", linewidths=0.6, zorder=3,
                   label=cluster)

    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("standardised mean difference vs all other clusters",
                  fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(
        f"Baseline covariate balance at shock onset — {stratification}",
        loc="left", fontsize=9, pad=8,
    )
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_SECONDARY,
              loc="lower right")
    ax.grid(axis="y", color=GRID, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    figure.text(
        0.01, 0.01,
        "Shaded band is |SMD| < 0.1, the usual threshold for balance; dashed "
        "lines mark 0.25, beyond which an adjusted estimate is extrapolating "
        "rather than comparing.",
        fontsize=6, color=INK_SECONDARY,
    )
    target = paths.figure_home / f"covariate_balance_{tag}_septic.png"
    save_figure(figure, target)
    plt.close(figure)
    worst = part.drop_nulls("smd").sort(pl.col("smd").abs(), descending=True).head(3)
    logger.info(f"wrote {target}")
    for row in worst.iter_rows(named=True):
        logger.info(
            f"    worst imbalance: {row['cluster']} {row['covariate']} "
            f"SMD {row['smd']:+.2f}"
        )

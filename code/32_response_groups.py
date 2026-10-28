#!/usr/bin/env python3

"""
32_response_groups.py -- are there groups responding differently to the same
handbook, and what distinguishes them

Q1 step 1: cluster patients on their hourly-state composition and then try to
kill the clustering. Four gates, any of which should retire it.

    python code/32_response_groups.py --subset cardiogenic

COMPOSITIONAL DATA NEEDS A LOG-RATIO. State shares sum to one, so they live on
a simplex, not in Euclidean space: raw proportions are linearly dependent and a
distance computed on them is distorted by whichever component happens to be
large. The centred log-ratio transform removes the constraint. Zeros are
handled by a small additive offset, which is crude but the alternative --
dropping patients with any empty state -- would drop almost everyone.

k IS CHOSEN BY STABILITY, NOT SILHOUETTE. Silhouette always finds something: it
measures whether k-means produced a compact partition, not whether the
partition is real. This project has already been bitten by that. Split-half
adjusted Rand across repeated splits asks whether the same patients land
together when the cohort is cut in half, which is the question that matters.

THE GATES ARE NOT DIAGNOSTICS, THEY ARE TESTS THE CLUSTERING CAN FAIL. A
clustering that is a redescription of non-cardiovascular SOFA, or of site, or
that rests on patients with four observed hours, should not proceed to Q2 and
Q3 -- and the point of running all four before any outcome analysis is that the
outcome cannot then be used to argue the clusters were real after all.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import polars as pl
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.config import CONFIG_HOME
from clifcs.regions import HOURLY_REGION_ORDER

#: states that describe a patient ON support. "off pressors" is excluded from
#: the composition because it is 78% of hours and would make every distance a
#: distance in how long the patient was off drug -- which is an outcome, not a
#: response pattern.
ON_SUPPORT = tuple(r for r in HOURLY_REGION_ORDER if r != "off pressors")
MIN_HOURS = 10
OFFSET = 0.5


def composition(labelled, logger, min_hours=MIN_HOURS):
    """per-patient share of on-support hours in each state, CLR-transformed"""
    on = labelled.filter(pl.col("region") != "off pressors")
    counts = (
        on.group_by("hospitalization_id")
        .agg(
            on_hours=pl.len(),
            **{
                f"n_{r.replace(' ', '_')}": (pl.col("region") == r).sum()
                for r in ON_SUPPORT
            },
        )
        .filter(pl.col("on_hours") >= min_hours)
    )
    logger.info(
        f"  {counts.height:,} patients with at least {min_hours} on-support "
        f"hours (median {counts['on_hours'].median():.0f})"
    )
    raw = np.column_stack([
        counts[f"n_{r.replace(' ', '_')}"].to_numpy() for r in ON_SUPPORT
    ]).astype(float)
    # additive offset before the log: a patient who never entered a state has a
    # true zero, and log(0) is not available. The offset is small relative to
    # the median hour count, so it shifts a zero toward "rare" rather than
    # toward "typical".
    shares = (raw + OFFSET) / (raw + OFFSET).sum(axis=1, keepdims=True)
    clr = np.log(shares) - np.log(shares).mean(axis=1, keepdims=True)
    return counts, shares, clr


def choose_k(clr, logger, seed=42, candidates=(2, 3, 4, 5, 6), rounds=12):
    """
    k by split-half adjusted Rand, with silhouette reported alongside for contrast

    The two usually disagree and the disagreement is the point. Silhouette
    rewards compactness, which k-means delivers by construction at any k;
    split-half ARI asks whether the partition is reproducible, which is the
    only version of the question that can come back negative.
    """
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    rows = []
    for k in candidates:
        scores = []
        for _ in range(rounds):
            order = rng.permutation(len(clr))
            half = len(order) // 2
            a, b = order[:half], order[half:]
            model_a = KMeans(k, n_init=10, random_state=seed).fit(clr[a])
            model_b = KMeans(k, n_init=10, random_state=seed).fit(clr[b])
            # each half's model labels the WHOLE cohort; agreement between the
            # two labellings is what reproducibility means here
            scores.append(adjusted_rand_score(
                model_a.predict(clr), model_b.predict(clr)
            ))
        full = KMeans(k, n_init=10, random_state=seed).fit(clr)
        sample = rng.choice(
            len(clr), size=min(4000, len(clr)), replace=False
        )
        rows.append({
            "k": k,
            "split_half_ari": float(np.mean(scores)),
            "ari_sd": float(np.std(scores)),
            "silhouette": float(silhouette_score(
                clr[sample], full.labels_[sample]
            )),
            "smallest_cluster_share": float(
                np.bincount(full.labels_, minlength=k).min() / len(clr)
            ),
        })
        logger.info(
            f"    k={k}: split-half ARI {np.mean(scores):.3f} "
            f"+/- {np.std(scores):.3f}   silhouette {rows[-1]['silhouette']:.3f}"
            f"   smallest cluster {rows[-1]['smallest_cluster_share']:.1%}"
        )
    return pl.DataFrame(rows)


def gate_severity(frame, labels, logger):
    """
    is the clustering a redescription of non-cardiovascular SOFA

    Non-CV deliberately: the cardiovascular component is scored from MAP and
    vasopressor dose, which is what the states are built from, so including it
    would remove the very thing being tested.
    """
    # `attach_severity` names it `sofa_non_cv`, not `severity`. Looking for
    # the wrong name made the gate skip itself with a warning that read like a
    # data problem rather than a wiring one -- which is exactly how a gate
    # stops gating.
    column = next(
        (c for c in ("sofa_non_cv", "severity") if c in frame.columns), None
    )
    if column is None:
        logger.error(
            "  no non-CV SOFA column; the severity gate did NOT run, so these "
            "clusters are unchecked against severity"
        )
        return pl.DataFrame()
    rows = []
    for cluster in sorted(set(labels.tolist())):
        part = frame.filter(pl.Series(labels == cluster))
        values = part[column].drop_nulls()
        rows.append({
            "cluster": f"R{cluster + 1}",
            "patients": part.height,
            "median_non_cv_sofa": float(values.median())
            if not values.is_empty() else None,
            "iqr_low": float(values.quantile(0.25))
            if not values.is_empty() else None,
            "iqr_high": float(values.quantile(0.75))
            if not values.is_empty() else None,
        })
    table = pl.DataFrame(rows)
    spread = (
        table["median_non_cv_sofa"].max() - table["median_non_cv_sofa"].min()
        if table["median_non_cv_sofa"].null_count() == 0 else float("nan")
    )
    logger.info(
        f"  non-CV SOFA medians span {spread:.1f} points across clusters"
        + ("  -- small, so the clusters are not simply severity"
           if spread == spread and spread < 2 else "")
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--k", type=int, default=None,
                        help="override the stability-chosen k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "32_response_groups", paths.log_home / "32_response_groups.log"
    )
    cfg = OmegaConf.load(CONFIG_HOME / "dynamics.yaml")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    labelled_file = (
        paths.patient_level_home / f"hourly_regions_{args.subset}.parquet"
    )
    if not labelled_file.is_file():
        logger.error(f"no {labelled_file.name}; run 31 first")
        return
    labelled = pl.read_parquet(labelled_file)
    logger.info(
        f"{labelled.height:,} hourly states, "
        f"{labelled['hospitalization_id'].n_unique():,} patients"
    )

    counts, shares, clr = composition(labelled, logger)
    if counts.height < 500:
        logger.error("too few patients clear the hour floor to cluster")
        return

    logger.info("choosing k by split-half reproducibility:")
    selection = choose_k(clr, logger, seed=args.seed)
    selection.write_csv(tables / f"response_group_k_{args.subset}.csv")
    best = (
        args.k if args.k
        else int(selection.sort("split_half_ari", descending=True)["k"][0])
    )
    chosen_ari = float(
        selection.filter(pl.col("k") == best)["split_half_ari"][0]
    )
    logger.info(
        f"  chosen k={best} (split-half ARI {chosen_ari:.3f})"
        + ("  -- WEAK: below 0.5, the partition is not reproducible"
           if chosen_ari < 0.5 else "")
    )

    model = KMeans(best, n_init=20, random_state=args.seed).fit(clr)
    labels = model.labels_
    frame = counts.with_columns(
        response_group=pl.Series([f"R{v + 1}" for v in labels])
    )
    for index, region in enumerate(ON_SUPPORT):
        frame = frame.with_columns(
            pl.Series(region.replace(" ", "_"), shares[:, index])
        )

    # ---- what each group's composition actually looks like
    logger.info("group composition (share of on-support hours):")
    profile_rows = []
    for group in sorted(frame["response_group"].unique().to_list()):
        part = frame.filter(pl.col("response_group") == group)
        row = {"response_group": group, "patients": part.height,
               "median_on_hours": float(part["on_hours"].median())}
        for region in ON_SUPPORT:
            row[region] = float(part[region.replace(" ", "_")].mean())
        profile_rows.append(row)
        top = sorted(
            ON_SUPPORT, key=lambda r: -row[r]
        )[:3]
        logger.info(
            f"    {group} n={part.height:>5,}  "
            + ", ".join(f"{r} {row[r]:.0%}" for r in top)
        )
    pl.DataFrame(profile_rows).write_csv(
        tables / f"response_group_profile_{args.subset}.csv"
    )

    # ---- gate: site
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    keep = [
        c for c in ("hospitalization_id", "site", "in_hospital_mortality")
        if c in cohort.columns
    ]
    frame = frame.join(cohort.select(keep), on="hospitalization_id", how="left")
    site_rows = []
    for group in sorted(frame["response_group"].unique().to_list()):
        part = frame.filter(pl.col("response_group") == group)
        row = {"response_group": group, "patients": part.height}
        for site in sorted(frame["site"].drop_nulls().unique().to_list()):
            row[site] = part.filter(pl.col("site") == site).height / part.height
        site_rows.append(row)
    site_table = pl.DataFrame(site_rows)
    site_table.write_csv(tables / f"response_group_sites_{args.subset}.csv")
    logger.info("  site composition by group:")
    for row in site_table.iter_rows(named=True):
        sites = ", ".join(
            f"{k} {v:.0%}" for k, v in row.items()
            if k not in ("response_group", "patients")
        )
        logger.info(f"    {row['response_group']}: {sites}")

    # ---- gate: severity (non-CV SOFA, from the existing concordance module)
    from clifcs.concordance import attach_severity
    try:
        severity = attach_severity(paths, cohort, logger)
        if "sofa_non_cv" in severity.columns:
            frame = frame.join(
                severity.select("hospitalization_id", "sofa_non_cv",
                                "sofa_total"),
                on="hospitalization_id", how="left",
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  severity unavailable: {exc}")
    severity_table = gate_severity(frame, labels, logger)
    if not severity_table.is_empty():
        severity_table.write_csv(
            tables / f"response_group_severity_{args.subset}.csv"
        )

    # ---- gate: sparsity
    thin = frame.filter(pl.col("on_hours") < 20)
    logger.info(
        f"  sparsity: {thin.height:,} of {frame.height:,} patients "
        f"({thin.height / frame.height:.1%}) rest on fewer than 20 on-support "
        "hours"
    )
    frame = frame.with_columns(
        thin_composition=pl.col("on_hours") < 20
    )

    # ---- outcome, reported but NOT used to justify the clustering
    if "in_hospital_mortality" in frame.columns:
        logger.info("mortality by group (reported after the gates, not before):")
        for group in sorted(frame["response_group"].unique().to_list()):
            part = frame.filter(
                (pl.col("response_group") == group)
                & pl.col("in_hospital_mortality").is_not_null()
            )
            logger.info(
                f"    {group} n={part.height:>5,}  died "
                f"{part['in_hospital_mortality'].mean():.1%}"
            )
    frame.write_parquet(
        paths.patient_level_home / f"response_groups_{args.subset}.parquet"
    )
    logger.info("32 done")


if __name__ == "__main__":
    main()

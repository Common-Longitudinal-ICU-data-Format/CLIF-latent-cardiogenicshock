#!/usr/bin/env python3

"""
11_longitudinal_clusters.py -- longitudinal clusters in septic shock

Clusters patients on how their TOKEN composition moves over the two days after
shock onset, by two methods that assume different things (state sequences and
latent class growth), then characterises the clusters by treatment and outcome
and checks that they are not hospitals.

    python code/11_longitudinal_clusters.py --subset septic

Writes output/patient_level_data/<model>/longitudinal_clusters_<subset>.parquet,
which 09 joins to colour the projection.

Feature space is token counts WITHIN each six-hour window, restricted to tokens
every site records at a comparable rate -- see clifcs.windows. Nothing in this
script touches the model or the embeddings; the embedding is where the clusters
get DRAWN, which is only an interesting place to draw them because they were
found somewhere else.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.longitudinal import (
    care_states,
    growth_clusters,
    growth_coefficients,
    sequence_clusters,
    shuffled_null,
    split_half_stability,
)
from clifcs.tokens import vocabulary
from clifcs.windows import (
    composition_matrix,
    site_representative_tokens,
    window_token_counts,
)

#: +0h to +48h in six-hour steps. The -24h to -6h points are excluded on
#: purpose: a patient only has them if they were admitted before shock onset,
#: so including them would cluster partly on admission timing.
OFFSETS = [float(h) for h in range(0, 49, 6)]
STEP_HOURS = 6.0


def cohort_ids(paths, subset: str, logger):
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    if subset == "septic":
        cohort = cohort.filter(pl.col("sepsis_cohort").fill_null(False))
    elif subset == "cardiogenic":
        cohort = cohort.filter(
            (pl.col("shock_type") == "cardiogenic")
            | pl.col("cardiac_origin").fill_null(False)
        )
    logger.info(f"subset {subset!r}: {cohort.height:,} patients")
    return cohort


def characterise(frame: pl.DataFrame, paths, label: str, logger) -> pl.DataFrame:
    """
    treatment and outcome by cluster

    Treatment is reported because it is what a clinician will ask first, but
    read it knowing the feature space includes treatment tokens -- vasopressor
    and ventilation events are part of the composition the clusters were built
    from, so a treatment difference is partly by construction. ICU length of
    stay and mortality are not in the feature space at all and are the checks
    that carry weight.
    """
    rows = []
    for cluster, part in sorted(frame.group_by(label), key=lambda kv: kv[0]):
        key = cluster[0] if isinstance(cluster, tuple) else cluster
        row = {label: key, "n": part.height}
        for column, name, how in (
            ("nee_max", "peak norepinephrine equivalents", "median"),
            ("nee_median", "median norepinephrine equivalents", "median"),
            ("n_vasopressors_max", "peak vasoactive agents", "median"),
            ("icu_los_days", "ICU length of stay (days)", "median"),
            ("ever_imv", "invasive ventilation", "share"),
            ("ever_rrt", "renal replacement", "share"),
            ("in_hospital_mortality", "died in hospital", "share"),
        ):
            if column not in part.columns:
                continue
            values = part[column].drop_nulls()
            if values.is_empty():
                continue
            if how == "share":
                row[name] = float(values.cast(pl.Float64).mean())
            else:
                row[name] = float(values.median())
        rows.append(row)
    table = pl.DataFrame(rows)
    logger.info(f"  {label} by treatment and outcome:")
    for row in table.iter_rows(named=True):
        parts = "  ".join(
            f"{k}={v:.3g}" for k, v in row.items() if k not in (label, "n")
        )
        logger.info(f"    {row[label]}  n={row['n']:>6,}  {parts}")
    return table


def site_composition(frame: pl.DataFrame, label: str, sites, logger) -> None:
    """
    is the partition a partition of hospitals?

    Cramer's V between cluster and site, on the contingency table. Reported
    rather than a probe accuracy because the question is association strength,
    not decodability, and V is comparable across different k. Roughly: below
    0.1 the partition is near-independent of site, above 0.3 it is substantially
    a site partition and the clusters should not be read as clinical.
    """
    table = (
        frame.group_by(label, "site").len()
        .pivot(on="site", index=label, values="len")
        .fill_null(0)
    )
    present = [s for s in sites if s in table.columns]
    observed = table.select(present).to_numpy().astype(float)
    total = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    chi2 = float(((observed - expected) ** 2 / np.maximum(expected, 1e-9)).sum())
    v = float(np.sqrt(chi2 / (total * (min(observed.shape) - 1))))
    logger.info(f"  {label} vs site: Cramer's V {v:.3f}")
    for row in table.iter_rows(named=True):
        shares = {s: row[s] for s in present}
        n = sum(shares.values())
        spread = "  ".join(f"{s} {row[s] / max(n, 1):>5.1%}" for s in present)
        logger.info(f"    {row[label]}  n={n:>6,}  {spread}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="septic",
                        choices=("septic", "cardiogenic", "all"))
    parser.add_argument("--states", type=int, default=6,
                        help="discrete care states the windows are reduced to")
    parser.add_argument("--clusters", type=int, default=4,
                        help="k for both clustering methods")
    parser.add_argument("--max-ratio", type=float, default=2.0,
                        help="largest allowed between-site prevalence ratio for "
                             "a token to enter the feature space")
    parser.add_argument("--min-prevalence", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    parser.add_argument("--role", choices=("site", "pooled"), default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model, role=args.role)
    paths.mkdirs()
    logger = get_logger(
        "11_longitudinal_clusters", paths.log_home / "11_longitudinal_clusters.log"
    )
    logger.info(
        f"subset {args.subset}, windows {OFFSETS[0]:+.0f}h to {OFFSETS[-1]:+.0f}h "
        f"in {STEP_HOURS:g}h steps ({len(OFFSETS)} positions)"
    )

    cohort = cohort_ids(paths, args.subset, logger)
    counts = window_token_counts(
        paths, OFFSETS, STEP_HOURS,
        cohort["hospitalization_id"].to_list(), logger,
    )
    token_ids = site_representative_tokens(
        counts, list(paths.sites), logger,
        max_ratio=args.max_ratio, min_prevalence=args.min_prevalence,
    )
    if len(token_ids) < 20:
        logger.error(
            f"only {len(token_ids)} tokens survived the site filter; raise "
            "--max-ratio or lower --min-prevalence"
        )
        return
    patients, matrix, volume = composition_matrix(
        counts, token_ids, OFFSETS, logger
    )

    # a patient needs something recorded in most windows to have a trajectory
    enough = (volume > 0).sum(axis=1) >= len(OFFSETS) - 2
    logger.info(
        f"{int(enough.sum()):,} of {len(patients):,} patients have tokens in at "
        f"least {len(OFFSETS) - 2} of {len(OFFSETS)} windows"
    )
    patients = [p for p, k in zip(patients, enough) if k]
    matrix, volume = matrix[enough], volume[enough]

    states, topics, _model = care_states(matrix, args.states, args.seed, logger)
    sequence_labels, one_hot = sequence_clusters(
        states, args.clusters, args.seed, logger
    )

    # volume joins the growth features: how much is recorded per six hours is
    # clinically real and is not in the composition, which is normalised by it
    growth_features = np.column_stack([
        growth_coefficients(topics, logger),
        growth_coefficients(volume[:, :, None], logger),
    ])
    growth_labels, growth_scaled, _bic = growth_clusters(
        growth_features, args.clusters, args.seed, logger
    )

    logger.info("validation:")

    def fit_kmeans(x, k, seed):
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(x)
        return model.labels_, model.cluster_centers_

    logger.info("  state sequences")
    split_half_stability(one_hot, args.clusters, args.seed, fit_kmeans, logger)
    logger.info("  latent class growth")
    split_half_stability(growth_scaled, args.clusters, args.seed, fit_kmeans, logger)

    agreement = adjusted_rand_score(sequence_labels, growth_labels)
    logger.info(f"  the two methods agree: ARI {agreement:.3f}")

    # the temporal null: same pipeline, windows permuted within each patient
    null_matrix = shuffled_null(matrix, args.states, args.clusters, args.seed, logger)
    null_states, _nt, _nm = care_states(
        null_matrix, args.states, args.seed, logger
    )
    null_labels, null_one_hot = sequence_clusters(
        null_states, args.clusters, args.seed, logger
    )
    null_growth_features = np.column_stack([
        growth_coefficients(_nt, logger),
        growth_coefficients(volume[:, :, None], logger),
    ])
    null_growth_labels, null_growth_scaled, _nb = growth_clusters(
        null_growth_features, args.clusters, args.seed, logger
    )

    def quality(x, labels):
        return silhouette_score(
            x, labels, sample_size=min(5000, len(x)), random_state=args.seed
        )

    logger.info("  time-shuffled null, silhouette real vs shuffled:")
    logger.info(
        f"    state sequences      {quality(one_hot, sequence_labels):.3f}  vs  "
        f"{quality(null_one_hot, null_labels):.3f}"
    )
    logger.info(
        f"    latent class growth  {quality(growth_scaled, growth_labels):.3f}  vs  "
        f"{quality(null_growth_scaled, null_growth_labels):.3f}"
    )
    logger.info(
        "    a shuffled score close to the real one means the partition is of "
        "which states a patient visits, not of the order they visit them in"
    )

    # characterisation
    assignments = pl.DataFrame({
        "hospitalization_id": patients,
        "sequence_cluster": [f"S{int(v) + 1}" for v in sequence_labels],
        "growth_cluster": [f"G{int(v) + 1}" for v in growth_labels],
    }).with_columns(
        state_sequence=pl.Series([
            "".join(str(int(s)) for s in row) for row in states
        ])
    )

    labels_frame = assignments.join(
        cohort.select([
            c for c in ("hospitalization_id", "site", "in_hospital_mortality",
                        "icu_los_days")
            if c in cohort.columns
        ]),
        on="hospitalization_id", how="left",
    )

    # Renal replacement from the CRRT token family rather than from a derived
    # table. It is asked of the clusters as an outcome, and CRRT tokens are
    # among those the site filter may well have dropped for being charted at
    # very different rates -- so this reads the UNFILTERED counts on purpose.
    # It is a characterisation, not a feature, and does not have to be
    # site-comparable to be worth reporting; the per-site split below says how
    # far it can be trusted.
    if paths.tokenizer_yaml_or_none is not None:
        names, families = vocabulary(str(paths.tokenizer_yaml))
        crrt = [i for i, fam in families.items() if fam == "CRRT"]
        if crrt:
            rrt = (
                counts.filter(pl.col("token_id").is_in(crrt))
                .group_by("hospitalization_id")
                .agg(ever_rrt=pl.lit(True))
            )
            labels_frame = labels_frame.join(
                rrt, on="hospitalization_id", how="left"
            ).with_columns(ever_rrt=pl.col("ever_rrt").fill_null(False))
            logger.info(
                f"renal replacement: {int(labels_frame['ever_rrt'].sum()):,} of "
                f"{labels_frame.height:,} patients have a CRRT token in the window"
            )

    # treatment, read off the state table 09 already wrote
    state_file = paths.patient_level_home / "trajectory_state.parquet"
    if state_file.is_file():
        per_patient = (
            pl.read_parquet(state_file)
            .filter(pl.col("offset_hours").is_between(OFFSETS[0], OFFSETS[-1]))
            .group_by("hospitalization_id")
            .agg(
                nee_max=pl.col("nee").max(),
                nee_median=pl.col("nee").median(),
                n_vasopressors_max=pl.col("n_vasopressors").max(),
                ever_imv=pl.col("on_imv").max() > 0,
            )
        )
        labels_frame = labels_frame.join(
            per_patient, on="hospitalization_id", how="left"
        )
    else:
        logger.warning(
            f"{state_file.name} not found: treatment columns omitted. Run 09 "
            "--stage plot first."
        )

    for label in ("sequence_cluster", "growth_cluster"):
        characterise(labels_frame, paths, label, logger)
        site_composition(labels_frame, label, list(paths.sites), logger)

    out = (
        paths.patient_level_home
        / f"longitudinal_clusters_{args.subset}.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    assignments.write_parquet(out)
    logger.info(f"wrote {out} ({assignments.height:,} patients)")
    logger.info("11 done")


if __name__ == "__main__":
    main()

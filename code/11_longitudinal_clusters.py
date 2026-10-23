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
    state_movement,
    split_half_stability,
)
from clifcs.tokens import vocabulary
from clifcs.windows import (
    composition_matrix,
    site_representative_tokens,
    window_token_counts,
)

#: +0h to +168h in six-hour steps -- seven days, 29 window positions. The
#: -24h to -6h points are excluded on purpose: a patient only has them if they
#: were admitted before shock onset, so including them would cluster partly on
#: admission timing.
#:
#: Seven days changes what absence means. Over 48h nearly every enrolled
#: patient is still being charted, so an empty window is a recording gap. Over
#: seven days most are not -- median ICU stay in these strata runs 1.8 to 11.4
#: days -- so a trailing run of empty windows is mostly discharge or death.
#: That is informative, and it is also partly the outcome, so `--min-windows`
#: fixes how much of the acute phase a patient must have before they can be
#: clustered at all, and the share of each cluster's windows that are empty is
#: reported so the contribution of record length is visible rather than
#: assumed away.
OFFSETS = [float(h) for h in range(0, 169, 6)]
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
    rows = []
    for row in table.iter_rows(named=True):
        n = sum(row[s] for s in present)
        spread = "  ".join(f"{s} {row[s] / max(n, 1):>5.1%}" for s in present)
        logger.info(f"    {row[label]}  n={n:>6,}  {spread}")
        rows.append({
            label: row[label], "n": n,
            **{f"share_{s}": row[s] / max(n, 1) for s in present},
        })
    return pl.DataFrame(rows).with_columns(cramers_v=pl.lit(v))


def describe_states(counts, token_ids, states, patients, offsets, names, logger):
    """
    what each care state actually contains

    "state 3" is not a finding, it is an index. A state is only interpretable
    through the tokens over-represented in the windows assigned to it, so this
    computes, per state, each token's prevalence among that state's windows
    against its prevalence overall.

    Lift rather than raw prevalence, with a prevalence floor. The commonest
    tokens are common everywhere -- a vital charted in 90% of all windows is in
    90% of every state's windows and says nothing about any of them -- so
    ranking by prevalence returns the same list for all six states. Ranking by
    lift alone instead surfaces rare tokens whose ratio is noise, hence the
    floor: a token has to be in at least a quarter of the state's windows
    before its lift is worth reading.
    """
    state_of = {
        (p, float(o)): int(states[i, j])
        for i, p in enumerate(patients)
        for j, o in enumerate(offsets)
    }
    kept = counts.filter(pl.col("token_id").is_in(list(token_ids))).with_columns(
        care_state=pl.struct("hospitalization_id", "window").map_elements(
            lambda r: state_of.get(
                (r["hospitalization_id"], float(r["window"])), -1
            ),
            return_dtype=pl.Int64,
        )
    ).filter(pl.col("care_state") >= 0)

    cells = (
        kept.select("hospitalization_id", "window", "care_state").unique()
        .group_by("care_state").len().rename({"len": "cells"})
    )
    per_state = (
        kept.select("hospitalization_id", "window", "care_state", "token_id")
        .unique().group_by("care_state", "token_id").len()
        .rename({"len": "present"})
        .join(cells, on="care_state")
        .with_columns(prevalence=pl.col("present") / pl.col("cells"))
    )
    total = kept.select("hospitalization_id", "window").unique().height
    overall = (
        kept.select("hospitalization_id", "window", "token_id").unique()
        .group_by("token_id").len().rename({"len": "n"})
        .with_columns(base=pl.col("n") / total)
    )
    lift = (
        per_state.join(overall, on="token_id")
        .with_columns(
            lift=pl.col("prevalence") / pl.col("base"),
            token=pl.col("token_id").cast(pl.Int64).replace_strict(
                names, default="<unmapped>", return_dtype=pl.String
            ),
        )
    )

    rows = []
    logger.info("what each care state contains:")
    for state in sorted(lift["care_state"].unique().to_list()):
        part = (
            lift.filter(
                (pl.col("care_state") == state) & (pl.col("prevalence") >= 0.25)
            )
            .sort("lift", descending=True)
            .head(8)
        )
        n_cells = int(cells.filter(pl.col("care_state") == state)["cells"][0])
        label = "nothing recorded" if state == 0 else f"state {state}"
        logger.info(f"  {label} ({n_cells:,} patient-windows)")
        for row in part.iter_rows(named=True):
            logger.info(
                f"      {row['token']:<34} in {row['prevalence']:>5.1%} of its "
                f"windows ({row['lift']:.2f}x overall)"
            )
            rows.append({
                "care_state": label,
                "patient_windows": n_cells,
                "token": row["token"],
                "prevalence_in_state": row["prevalence"],
                "prevalence_overall": row["base"],
                "lift": row["lift"],
            })
    return pl.DataFrame(rows)


def differential_within_cluster(
    counts, labels_frame, token_ids, names, label: str, logger,
    min_cell: int = 11, top: int = 8,
):
    """
    within each severity stratum, what separates the patients who died

    The clusters stratify by severity, so a token enriched among deaths across
    the whole cohort tells you mainly that sicker patients die. Asking the
    question INSIDE a stratum holds severity roughly fixed and asks what
    distinguishes outcome among patients who look similarly sick -- which is
    the question worth asking of a severity stratification, rather than a
    reason to discard it.

    Prevalence is per PATIENT (did this token ever appear in their windows),
    not per window, so a patient charted twice as often does not count twice.
    Cells below `min_cell` patients are not reported, on the same CMS
    small-cell rule the rest of the pipeline uses.

    Bonferroni is over every token-by-stratum test performed, not per stratum,
    because the strata are examined together and correcting within each would
    under-correct the family that is actually being looked at.
    """
    from scipy.stats import chi2_contingency

    patients = labels_frame.select("hospitalization_id", label, "in_hospital_mortality")
    patients = patients.filter(pl.col("in_hospital_mortality").is_not_null())
    present = (
        counts.filter(pl.col("token_id").is_in(list(token_ids)))
        .select("hospitalization_id", "token_id")
        .unique()
        .join(patients, on="hospitalization_id", how="inner")
    )
    if present.is_empty():
        logger.warning("no tokens to test within clusters")
        return pl.DataFrame()

    strata = sorted(patients[label].unique().to_list())
    denominators = (
        patients.group_by(label, "in_hospital_mortality").len()
        .pivot(on="in_hospital_mortality", index=label, values="len")
        .fill_null(0)
    )
    died_col = "true" if "true" in denominators.columns else "True"
    alive_col = "false" if "false" in denominators.columns else "False"
    if died_col not in denominators.columns or alive_col not in denominators.columns:
        logger.warning(f"mortality column not two-valued: {denominators.columns}")
        return pl.DataFrame()
    totals = {
        row[label]: (row[died_col], row[alive_col])
        for row in denominators.iter_rows(named=True)
    }

    hits = (
        present.group_by(label, "token_id", "in_hospital_mortality").len()
        .pivot(on="in_hospital_mortality", index=[label, "token_id"], values="len")
        .fill_null(0)
    )

    rows = []
    for row in hits.iter_rows(named=True):
        stratum = row[label]
        died_n, alive_n = totals.get(stratum, (0, 0))
        died_hit, alive_hit = row.get(died_col, 0), row.get(alive_col, 0)
        if min(died_hit, alive_hit) < min_cell:
            continue
        table = np.array([
            [died_hit, died_n - died_hit],
            [alive_hit, alive_n - alive_hit],
        ])
        # Both marginals, not just the rows. A token present in every patient
        # of one outcome group leaves a zero column marginal, and chi2 then
        # divides by a zero expected frequency. Such a token is also
        # uninformative by construction -- it does not vary within the stratum.
        if (
            (table < 0).any()
            or table.sum(axis=1).min() == 0
            or table.sum(axis=0).min() == 0
        ):
            continue
        chi2, p, _, _ = chi2_contingency(table, correction=False)
        died_rate = died_hit / max(died_n, 1)
        alive_rate = alive_hit / max(alive_n, 1)
        rows.append({
            label: stratum,
            "token": names.get(int(row["token_id"]), str(row["token_id"])),
            "died_prevalence": died_rate,
            "survived_prevalence": alive_rate,
            "log2_ratio": float(np.log2(max(died_rate, 1e-9) / max(alive_rate, 1e-9))),
            "p": float(p),
        })

    if not rows:
        logger.warning("every token-by-stratum cell fell below the small-cell floor")
        return pl.DataFrame()

    table = pl.DataFrame(rows)
    table = table.with_columns(
        p_adjusted=(pl.col("p") * table.height).clip(upper_bound=1.0)
    ).sort("p")
    significant = table.filter(pl.col("p_adjusted") < 0.05)
    logger.info(
        f"  died vs survived within {label}: {table.height:,} tests, "
        f"{significant.height:,} significant after Bonferroni"
    )
    for stratum in strata:
        part = significant.filter(pl.col(label) == stratum)
        if part.is_empty():
            logger.info(f"    {stratum}: nothing survives correction")
            continue
        up = part.filter(pl.col("log2_ratio") > 0).sort("log2_ratio", descending=True)
        down = part.filter(pl.col("log2_ratio") < 0).sort("log2_ratio")
        died_n, alive_n = totals.get(stratum, (0, 0))
        logger.info(
            f"    {stratum} ({died_n:,} died, {alive_n:,} survived), "
            f"{part.height} tokens differ:"
        )
        for direction, rows_ in (("enriched in deaths", up), ("enriched in survivors", down)):
            for r in rows_.head(top).iter_rows(named=True):
                logger.info(
                    f"      {direction:<22} {r['token']:<32} "
                    f"{r['died_prevalence']:>6.1%} vs {r['survived_prevalence']:>6.1%}  "
                    f"log2 {r['log2_ratio']:+.2f}"
                )
    return table


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
    parser.add_argument(
        "--min-windows", type=int, default=8,
        help="how many of the EARLIEST windows a patient must be charted in to "
             "be clustered; 8 is the first 48h. Trailing windows may be empty, "
             "which over seven days mostly means discharge or death.",
    )
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

    # A patient needs a complete enough ACUTE phase to be comparable, and is
    # then allowed to run out of record. Requiring coverage across all 29
    # windows would keep only patients who stayed a week, which is survivorship
    # selection that removes the fulminant group entirely -- the one with the
    # highest mortality.
    early = int(args.min_windows)
    enough = (volume[:, :early] > 0).sum(axis=1) >= early - 1
    logger.info(
        f"{int(enough.sum()):,} of {len(patients):,} patients have tokens in at "
        f"least {early - 1} of the first {early} windows "
        f"(through {OFFSETS[early - 1]:+.0f}h)"
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

    (paths.output_home / 'tables' / paths.model_key).mkdir(
        parents=True, exist_ok=True
    )
    movement = state_movement(states, sequence_labels, logger)
    vocab, _fam = (
        vocabulary(str(paths.tokenizer_yaml))
        if paths.tokenizer_yaml_or_none is not None else ({}, {})
    )
    if vocab:
        describe_states(
            counts, token_ids, states, patients, OFFSETS, vocab, logger
        ).write_csv(
            paths.output_home / "tables" / paths.model_key
            / f"care_state_tokens_{args.subset}.csv"
        )
    changes = (states[:, 1:] != states[:, :-1]).sum(axis=1)
    pl.DataFrame({
        "changes": list(range(int(changes.max()) + 1)),
        "patients": [int((changes == k).sum()) for k in range(int(changes.max()) + 1)],
    }).with_columns(
        share=pl.col("patients") / int(len(changes)),
        possible_transitions=pl.lit(states.shape[1] - 1),
    ).write_csv(
        paths.output_home / "tables" / paths.model_key
        / f"state_movement_{args.subset}.csv"
    )
    empty = (volume == 0).mean(axis=1)
    logger.info("  share of each cluster's windows with nothing recorded:")
    for cluster in sorted(set(sequence_labels.tolist())):
        mask = sequence_labels == cluster
        logger.info(
            f"    S{cluster + 1}: {empty[mask].mean():.1%} "
            f"(a proxy for how much of the week the patient was still in record)"
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

    names, _families = (
        vocabulary(str(paths.tokenizer_yaml))
        if paths.tokenizer_yaml_or_none is not None else ({}, {})
    )
    differential = []
    tables_home = paths.output_home / "tables" / paths.model_key
    tables_home.mkdir(parents=True, exist_ok=True)
    for label in ("sequence_cluster", "growth_cluster"):
        summary = characterise(labels_frame, paths, label, logger)
        if not summary.is_empty():
            summary.write_csv(
                tables_home / f"cluster_characteristics_{label}_{args.subset}.csv"
            )
        spread = site_composition(
            labels_frame, label, list(paths.sites), logger
        )
        if spread is not None and not spread.is_empty():
            spread.write_csv(
                tables_home / f"cluster_sites_{label}_{args.subset}.csv"
            )
        table = differential_within_cluster(
            counts, labels_frame, token_ids, names, label, logger
        )
        if not table.is_empty():
            # Renamed to a common column before the concat. Keeping each
            # stratification's own column name meant the diagonal concat gave
            # every sequence_cluster row a null growth_cluster and vice versa,
            # so the published table carried a column of "None" next to the
            # one that held the value.
            differential.append(
                table.rename({label: "cluster"})
                .with_columns(stratification=pl.lit(label))
                .select(
                    "stratification", "cluster", "token", "died_prevalence",
                    "survived_prevalence", "log2_ratio", "p", "p_adjusted",
                )
            )

    if differential:
        out = (
            paths.output_home / "tables" / paths.model_key
            / f"cluster_mortality_tokens_{args.subset}.csv"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.concat(differential, how="diagonal_relaxed").write_csv(out)
        logger.info(f"wrote {out}")

    out = (
        paths.patient_level_home
        / f"longitudinal_clusters_{args.subset}.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    assignments.write_parquet(out)
    logger.info(f"wrote {out} ({assignments.height:,} patients)")

    # The same labels in long form, keyed by (patient, window), which is what
    # anything wanting to colour a per-timepoint figure needs to join on.
    #
    # Note which label lives at which grain. The CARE STATE is per
    # (patient, window) -- it is the thing that changes as a patient moves. The
    # cluster is per patient and is carried down every row of their trajectory,
    # constant along it by construction.
    #
    # This is a label table and nothing more. The clustering and the embedding
    # are separate analyses that share only the (patient, time) key: the
    # embedding reads every token in the cumulative prefix with no site filter,
    # this reads 366 site-balanced tokens counted inside one six-hour window.
    # Same patient, same instant, two different views of it -- which is the
    # whole reason colouring one by the other says anything.
    long = pl.DataFrame({
        "hospitalization_id": np.repeat(patients, len(OFFSETS)),
        "offset_hours": np.tile(np.array(OFFSETS, dtype=np.float64), len(patients)),
        "care_state": [
            "nothing recorded" if int(v) == 0 else f"state {int(v)}"
            for v in states.reshape(-1)
        ],
    }).join(
        assignments.select(
            "hospitalization_id", "sequence_cluster", "growth_cluster"
        ),
        on="hospitalization_id", how="left",
    )
    long_out = (
        paths.patient_level_home
        / f"longitudinal_states_{args.subset}.parquet"
    )
    long.write_parquet(long_out)
    logger.info(
        f"wrote {long_out} ({long.height:,} patient-windows, "
        f"{len(OFFSETS)} per patient)"
    )
    logger.info("11 done")


if __name__ == "__main__":
    main()

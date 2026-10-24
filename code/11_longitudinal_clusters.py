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

from collections import Counter

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.longitudinal import (
    check_factor_labels,
    care_states,
    growth_clusters,
    growth_coefficients,
    sequence_clusters,
    state_movement,
    split_half_stability,
)
from clifcs.perfusion import (
    PERFUSION_ORDER, PERFUSION_STATES, perfusion_state, state_symbol,
)
from clifcs.state import trajectory_state
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
            ("map_median", "median MAP (mmHg)", "median"),
            ("map_min", "lowest MAP (mmHg)", "median"),
            ("hours_hypotensive", "hours with MAP <65", "median"),
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


def perfusion_sequences(paths, cohort, offsets, logger) -> pl.DataFrame:
    """
    each patient's path through the four MAP-by-pressor states, over seven days

    Built here rather than read from 09, because 09's state table stops at
    +48h: it is keyed to the embedding grid, and the embedding grid is 13
    points from -24h to +48h. This needs 29 points out to a week, so the
    physiology is re-asof-joined onto the wider grid.

    Returns one row per (patient, window) with `perfusion_state`, plus the
    per-patient string. A window after the record ends carries a null state --
    the patient has no MAP and no infusion because they are discharged or dead,
    which is not the same as a perfused patient off pressors.
    """
    onset = pl.read_parquet(paths.cohort_home / "cs_onset.parquet").select(
        "hospitalization_id",
        onset=pl.col("cs_onset_dttm").cast(pl.Datetime("us")),
    )
    keep = set(cohort["hospitalization_id"].to_list())

    parts = []
    for site in paths.sites:
        site_ids = set(
            cohort.filter(pl.col("site") == site)["hospitalization_id"].to_list()
        )
        if not site_ids:
            continue
        grid = (
            onset.filter(pl.col("hospitalization_id").is_in(list(site_ids)))
            .join(
                pl.DataFrame({"offset_hours": [float(o) for o in offsets]}),
                how="cross",
            )
            .with_columns(
                grid_dttm=pl.col("onset")
                + pl.duration(seconds=(pl.col("offset_hours") * 3600).cast(pl.Int64))
            )
            .select("hospitalization_id", "offset_hours", "grid_dttm")
        )
        state = trajectory_state(paths.raw_subset(site), grid)
        logger.info(
            f"[{site}] physiology on {grid.height:,} seven-day grid points"
        )
        parts.append(state.with_columns(site=pl.lit(site)))

    if not parts:
        return pl.DataFrame()
    state = pl.concat(parts, how="diagonal_relaxed")
    state = perfusion_state(state, logger)
    return state


def cluster_perfusion_sequences(
    state: pl.DataFrame, offsets, n_clusters: int, seed: int, logger
):
    """
    cluster patients on their sequence through the four perfusion states

    The states here are CLINICALLY DEFINED -- a MAP threshold with hysteresis
    crossed with whether pressors run -- not learned from the data like the six
    care states. That is the point of doing both: the care-state clustering
    asks what groupings the record falls into, this asks how patients move
    through a partition a clinician would recognise without being told.

    Five symbols, not four: a window after the record ends is its own symbol.
    Folding it into "MAP >65, no vasopressors" would say a discharged patient
    is a well-perfused one, and folding it into the low state would say a dead
    patient is hypotensive. Both are wrong and the second flatters the
    clustering, since death is the outcome these strata get judged against.
    """
    symbols = list(PERFUSION_ORDER) + ["record ended"]
    index = {s: i for i, s in enumerate(symbols)}
    wide = (
        state.with_columns(
            symbol=pl.col("perfusion_state").fill_null("record ended")
        )
        .pivot(on="offset_hours", index="hospitalization_id", values="symbol")
    )
    columns = [str(float(o)) for o in offsets if str(float(o)) in wide.columns]
    if len(columns) < 2:
        logger.warning("not enough perfusion windows to cluster")
        return pl.DataFrame()

    patients = wide["hospitalization_id"].to_list()
    codes = np.full((len(patients), len(columns)), index["record ended"], dtype=int)
    for j, column in enumerate(columns):
        values = wide[column].to_list()
        codes[:, j] = [index.get(v if v is not None else "record ended",
                                index["record ended"]) for v in values]

    one_hot = np.zeros((len(patients), len(columns) * len(symbols)), dtype=np.float32)
    for j in range(len(columns)):
        one_hot[np.arange(len(patients)), j * len(symbols) + codes[:, j]] = 1.0

    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = model.fit_predict(one_hot)
    changes = (codes[:, 1:] != codes[:, :-1]).sum(axis=1)
    logger.info(
        f"perfusion-sequence clusters: k={n_clusters}, sizes "
        f"{np.bincount(labels, minlength=n_clusters).tolist()}; "
        f"{int((changes > 0).sum()):,} of {len(changes):,} patients "
        f"({(changes > 0).mean():.1%}) change perfusion state at least once, "
        f"median {np.median(changes):.0f} of {len(columns) - 1}"
    )

    short = {s: str(i) for i, s in enumerate(symbols)}
    strings = ["".join(str(c) for c in row) for row in codes]
    out = pl.DataFrame({
        "hospitalization_id": patients,
        "perfusion_cluster": [f"P{int(v) + 1}" for v in labels],
        "perfusion_sequence": strings,
        "perfusion_changes": changes.tolist(),
    })
    logger.info("  symbol key: " + ", ".join(
        f"{short[s]}={s}" for s in symbols
    ))
    for cluster in sorted(out["perfusion_cluster"].unique().to_list()):
        part = out.filter(pl.col("perfusion_cluster") == cluster)
        top = part["perfusion_sequence"].value_counts(sort=True).head(3)
        logger.info(f"  {cluster} n={part.height:,}")
        for seq, n in zip(top["perfusion_sequence"].to_list(),
                          top["count"].to_list()):
            logger.info(f"      {seq}  x{n}")
    return out


def cluster_sequence_table(states, labels, offsets, logger, top: int = 4):
    """
    the commonest sequences in each cluster, and how it spends its time

    Both, because neither alone is honest. The modal sequence is vivid but
    thin -- with six symbols over 29 positions the most common string is
    typically a few per cent of its cluster, and reading it as "what this
    cluster looks like" is the mistake that led me to call these static when
    98.6% of patients move. Occupancy is the mass: the share of all this
    cluster's windows spent in each state.
    """
    rows = []
    strings = ["".join(str(int(s)) for s in row) for row in states]
    for cluster in sorted(set(labels.tolist())):
        mask = labels == cluster
        name = f"S{cluster + 1}"
        members = [s for s, k in zip(strings, mask) if k]
        counter = Counter(members)
        occupancy = Counter(int(v) for row in states[mask] for v in row)
        total = sum(occupancy.values())
        for rank, (sequence, n) in enumerate(counter.most_common(top), start=1):
            rows.append({
                "cluster": name,
                "patients": int(mask.sum()),
                "rank": rank,
                "sequence": sequence,
                "patients_with_this_sequence": n,
                "share_of_cluster": n / max(int(mask.sum()), 1),
                "occupancy": "  ".join(
                    f"{'none' if s == 0 else s}={occupancy[s] / total:.0%}"
                    for s, _ in occupancy.most_common(4)
                ),
            })
        logger.info(
            f"  {name} n={int(mask.sum()):,}  modal {counter.most_common(1)[0][0]} "
            f"(x{counter.most_common(1)[0][1]}, "
            f"{counter.most_common(1)[0][1] / max(int(mask.sum()), 1):.1%} of the cluster)"
        )
    return pl.DataFrame(rows)


def factor_token_table(model, token_ids, names, logger, top: int = 12):
    """
    the tokens each NMF factor loads on, for naming the factors by hand

    `model.components_` is factors-by-tokens and non-negative, which is the
    property that makes NMF worth using here: a factor is a set of events that
    co-occur, so its top loadings read as a description rather than as a
    contrast between two things the way a PCA loading does.

    Loadings are reported both raw and as a share of the factor's total mass.
    The share is what says whether a factor is CONCENTRATED -- a factor whose
    top twelve tokens carry 80% of its mass is one clinical pattern, while one
    carrying 15% is a diffuse mixture that probably should not be given a
    single name.
    """
    components = model.components_
    rows = []
    logger.info("NMF factor loadings (top tokens per factor):")
    for index in range(components.shape[0]):
        weights = components[index]
        total = float(weights.sum())
        order = np.argsort(weights)[::-1][:top]
        covered = float(weights[order].sum()) / max(total, 1e-12)
        logger.info(
            f"  factor {index + 1:02d}  top {top} tokens carry "
            f"{covered:.0%} of its mass"
        )
        for rank, position in enumerate(order, start=1):
            token = names.get(int(token_ids[position]), str(token_ids[position]))
            share = float(weights[position]) / max(total, 1e-12)
            rows.append({
                "factor": f"factor {index + 1:02d}",
                "rank": rank,
                "token": token,
                "loading": float(weights[position]),
                "share_of_factor": share,
                "top_tokens_cover": covered,
            })
            if rank <= 6:
                logger.info(f"       {rank:>2}. {token:<34} {share:>6.1%}")
    return pl.DataFrame(rows)


def motif_clusters(perfusion, offsets, n_clusters, seed, logger, tables_home,
                   subset: str = "septic"):
    """
    cluster patients on the motifs their state sequence contains

    Returns (labels_frame, long_states). The labels frame carries one row per
    patient with a cluster from the raw sequence and one from the run-length
    encoded sequence; the long frame carries the per-(patient, window) state
    so a projection can be coloured by it.
    """
    from sklearn.cluster import KMeans

    from clifcs.motifs import (
        describe_motifs, extract_motifs, motif_matrix, run_length_encode,
    )
    from clifcs.perfusion import PERFUSION_STATES, rich_perfusion_state

    rich = rich_perfusion_state(perfusion, logger)
    if "rich_symbol" not in rich.columns:
        return pl.DataFrame(), pl.DataFrame()

    ordered = rich.sort(["hospitalization_id", "offset_hours"])
    grouped = (
        ordered.group_by("hospitalization_id", maintain_order=True)
        .agg(sequence=pl.col("rich_symbol").str.join(""))
    )
    patients = grouped["hospitalization_id"].to_list()
    raw = grouped["sequence"].to_list()
    collapsed = [run_length_encode(s) for s in raw]
    logger.info(
        f"{len(patients):,} sequences, median length {int(np.median([len(s) for s in raw]))} "
        f"raw and {int(np.median([len(s) for s in collapsed]))} after collapsing runs"
    )

    frames = {}
    for tag, sequences in (("raw", raw), ("rle", collapsed)):
        vocabulary, per_patient, carriers = extract_motifs(sequences)
        if len(vocabulary) < 4:
            logger.warning(f"{tag}: only {len(vocabulary)} motifs, skipping")
            continue
        logger.info(f"-- motifs from the {tag} sequences")
        described = describe_motifs(
            vocabulary, carriers, PERFUSION_STATES, len(sequences), logger
        )
        if not described.is_empty():
            described.write_csv(
                tables_home / f"motif_vocabulary_{tag}_{subset}.csv"
            )
        matrix = motif_matrix(vocabulary, per_patient)
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = model.fit_predict(matrix)
        logger.info(
            f"   {tag} motif clusters: sizes "
            f"{np.bincount(labels, minlength=n_clusters).tolist()}"
        )
        frames[tag] = pl.DataFrame({
            "hospitalization_id": patients,
            f"motif_cluster_{tag}": [f"M{tag[0].upper()}{v + 1}" for v in labels],
            f"sequence_{tag}": sequences,
        })

    if not frames:
        return pl.DataFrame(), pl.DataFrame()
    out = frames.get("raw", frames.get("rle"))
    for tag, frame in frames.items():
        if frame is not out:
            out = out.join(frame, on="hospitalization_id", how="outer_coalesce")

    # the collapsed 8-state view, for hexagons: MAP x trend, composition
    # dropped. Fifteen hues is past where a categorical palette separates, and
    # the composition split is the part a hexagon map can least afford.
    long_states = ordered.select(
        "hospitalization_id", "offset_hours", "rich_state", "rich_symbol"
    ).with_columns(
        coarse_state=pl.col("rich_state")
        .str.replace("NE alone, ", "")
        .str.replace("NE\\+other, ", "")
    )
    return out, long_states


#: Token families that HELP DEFINE the perfusion state, so finding them
#: enriched at a divergence is close to circular: a norepinephrine dose token
#: in the last shared window may simply BE the dose change that puts the
#: patient in a different state at the next one. They are tested and reported
#: like everything else, but flagged, because the interesting answer is which
#: NON-constitutive tokens anticipate the split.
CONSTITUTIVE_PREFIXES = (
    "VTL//map", "VTL//sbp", "VTL//dbp",
    "MED-CTS//norepinephrine", "MED-CTS//epinephrine",
    "MED-CTS//phenylephrine", "MED-CTS//dopamine",
    "MED-CTS//angiotensin", "MED-CTS//vasopressin",
)


def is_constitutive(token: str) -> bool:
    return any(token.startswith(p) for p in CONSTITUTIVE_PREFIXES)


def divergence_tokens(
    sequences, counts, token_ids, names, offsets, logger,
    prefix_length: int = 3, min_group: int = 40, min_cell: int = 11,
    top_prefixes: int = 6,
):
    """
    among patients with the same history, what precedes them going different ways

    The within-stratum mortality analysis holds SEVERITY roughly fixed. This
    holds the PATH fixed, which is stronger: every patient in a comparison has
    occupied the same states in the same order up to the same hour, so the
    groups differ only in what happens next. The question is what was in the
    record at the last shared window that distinguishes them.

    For each common prefix, patients are split by the state they move to next,
    the two largest destinations are compared, and each token is tested on its
    per-patient prevalence in the LAST SHARED WINDOW -- before the split, while
    the two groups are still indistinguishable by state.

    WHAT THIS IS NOT. It is association, not cause. Both groups are observed,
    not assigned, and a token appearing before a divergence may be a marker of
    something already underway rather than a lever. The constitutive flag
    separates the worst case: a pressor-dose token in the last shared window
    can be the very change that produces the next state, in which case the
    "finding" is that a dose increase precedes a dose increase.
    """
    from scipy.stats import chi2_contingency

    position = prefix_length - 1
    if position >= len(offsets) - 1:
        logger.warning("prefix longer than the grid; divergence skipped")
        return pl.DataFrame()
    shared_window = float(offsets[position])
    next_window = float(offsets[position + 1])

    prefixes = {}
    for patient, sequence in sequences.items():
        if len(sequence) <= prefix_length:
            continue
        prefixes.setdefault(sequence[:prefix_length], []).append(
            (patient, sequence[prefix_length])
        )
    ranked = sorted(prefixes.items(), key=lambda kv: -len(kv[1]))
    logger.info(
        f"divergence at {next_window:+.0f}h after a shared prefix of "
        f"{prefix_length} windows (tokens read from {shared_window:+.0f}h):"
    )

    present = (
        counts.filter(
            pl.col("token_id").is_in(list(token_ids))
            & (pl.col("window") == shared_window)
        )
        .select("hospitalization_id", "token_id")
        .unique()
    )
    by_patient = {}
    for patient, token in zip(
        present["hospitalization_id"].to_list(), present["token_id"].to_list()
    ):
        by_patient.setdefault(patient, set()).add(int(token))

    rows = []
    for prefix, members in ranked[:top_prefixes]:
        destinations = Counter(nxt for _, nxt in members)
        ordered = [d for d, n in destinations.most_common() if n >= min_group]
        if len(ordered) < 2:
            continue
        left, right = ordered[0], ordered[1]
        group_a = [p for p, nxt in members if nxt == left]
        group_b = [p for p, nxt in members if nxt == right]
        logger.info(
            f"  prefix {prefix!r} ({len(members):,} patients): "
            f"{len(group_a):,} go to {left!r} vs {len(group_b):,} to {right!r}"
        )

        tested = 0
        for token in token_ids:
            token = int(token)
            hit_a = sum(1 for p in group_a if token in by_patient.get(p, ()))
            hit_b = sum(1 for p in group_b if token in by_patient.get(p, ()))
            if min(hit_a, hit_b) < min_cell:
                continue
            table = np.array([
                [hit_a, len(group_a) - hit_a],
                [hit_b, len(group_b) - hit_b],
            ])
            if table.min() < 0 or table.sum(axis=0).min() == 0:
                continue
            chi2, p, _, _ = chi2_contingency(table, correction=False)
            rate_a = hit_a / max(len(group_a), 1)
            rate_b = hit_b / max(len(group_b), 1)
            name = names.get(token, str(token))
            rows.append({
                "prefix": prefix,
                "shared_window_hours": shared_window,
                "destination_a": left, "destination_b": right,
                "n_a": len(group_a), "n_b": len(group_b),
                "token": name,
                "constitutive": is_constitutive(name),
                "prevalence_a": rate_a, "prevalence_b": rate_b,
                "log2_ratio": float(
                    np.log2(max(rate_a, 1e-9) / max(rate_b, 1e-9))
                ),
                "p": float(p),
            })
            tested += 1
        logger.info(f"      {tested} tokens testable above the small-cell floor")

    if not rows:
        logger.warning("no divergence comparison cleared the floors")
        return pl.DataFrame()

    table = pl.DataFrame(rows)
    table = table.with_columns(
        p_adjusted=(pl.col("p") * table.height).clip(upper_bound=1.0)
    ).sort("p")
    significant = table.filter(pl.col("p_adjusted") < 0.05)
    logger.info(
        f"  {table.height:,} tests, {significant.height:,} significant after "
        f"Bonferroni ({significant.filter(~pl.col('constitutive')).height:,} "
        "of them not constitutive)"
    )
    for row in significant.filter(~pl.col("constitutive")).head(12).iter_rows(
        named=True
    ):
        logger.info(
            f"      {row['prefix']} -> {row['destination_a']} vs "
            f"{row['destination_b']}   {row['token']:<32} "
            f"{row['prevalence_a']:>6.1%} vs {row['prevalence_b']:>6.1%}  "
            f"log2 {row['log2_ratio']:+.2f}"
        )
    return table


def divergence_outcomes(sequences, labels_frame, offsets, logger,
                        prefix_length: int = 2, min_group: int = 40,
                        top_prefixes: int = 6):
    """
    do the two branches of a divergence end up differently, and treated differently

    The token analysis asks what was in the record BEFORE the split. This asks
    what happens AFTER it: among patients who were indistinguishable by state
    until a given hour, do those who go one way have different outcomes, and
    were they treated differently over the rest of the week?

    The two questions are deliberately separate columns rather than one table,
    because they answer to different standards. An outcome difference between
    branches is the thing worth knowing. A treatment difference is partly
    mechanism and partly definition -- the branch is DEFINED by what the
    pressors did next, so a difference in peak norepinephrine is close to
    tautological, while a difference in ventilation or dialysis is not. The
    constitutive columns are marked so the distinction survives into the table.

    Fisher's exact is not used and the comparisons are not corrected across
    prefixes: this is a descriptive companion to the token test, which carries
    the multiplicity burden. Read the effect sizes, not the stars.
    """
    position = prefix_length - 1
    if position >= len(offsets) - 1:
        return pl.DataFrame()

    prefixes = {}
    for patient, sequence in sequences.items():
        if len(sequence) <= prefix_length:
            continue
        prefixes.setdefault(sequence[:prefix_length], []).append(
            (patient, sequence[prefix_length])
        )
    ranked = sorted(prefixes.items(), key=lambda kv: -len(kv[1]))

    #: (column, label, whether the branch definition makes it near-circular)
    MEASURES = (
        ("in_hospital_mortality", "died in hospital", False),
        ("icu_los_days", "ICU length of stay (days)", False),
        ("ever_imv", "invasive ventilation", False),
        ("ever_rrt", "renal replacement", False),
        ("map_median", "median MAP (mmHg)", True),
        ("hours_hypotensive", "hours with MAP <65", True),
        ("nee_max", "peak norepinephrine equivalents", True),
        ("n_vasopressors_max", "peak vasoactive agents", True),
    )

    rows = []
    logger.info(
        f"divergence outcomes after a {prefix_length}-window prefix "
        f"(split at {offsets[position + 1]:+.0f}h):"
    )
    for prefix, members in ranked[:top_prefixes]:
        destinations = Counter(nxt for _, nxt in members)
        ordered = [d for d, n in destinations.most_common() if n >= min_group]
        if len(ordered) < 2:
            continue
        left, right = ordered[0], ordered[1]
        ids_a = [p for p, nxt in members if nxt == left]
        ids_b = [p for p, nxt in members if nxt == right]
        part_a = labels_frame.filter(
            pl.col("hospitalization_id").is_in(ids_a)
        )
        part_b = labels_frame.filter(
            pl.col("hospitalization_id").is_in(ids_b)
        )
        if part_a.height < min_group or part_b.height < min_group:
            continue
        logger.info(
            f"  {prefix!r}: {part_a.height:,} -> {left!r} vs "
            f"{part_b.height:,} -> {right!r}"
        )
        for column, label, constitutive in MEASURES:
            if column not in labels_frame.columns:
                continue
            values_a = part_a[column].cast(pl.Float64).drop_nulls()
            values_b = part_b[column].cast(pl.Float64).drop_nulls()
            if values_a.is_empty() or values_b.is_empty():
                continue
            share = column in ("in_hospital_mortality", "ever_imv", "ever_rrt")
            stat_a = float(values_a.mean() if share else values_a.median())
            stat_b = float(values_b.mean() if share else values_b.median())
            rows.append({
                "prefix": prefix,
                "destination_a": left, "destination_b": right,
                "n_a": part_a.height, "n_b": part_b.height,
                "measure": label,
                "branch_defining": constitutive,
                "value_a": stat_a, "value_b": stat_b,
                "difference": stat_a - stat_b,
                "is_share": share,
            })
            if not constitutive:
                logger.info(
                    f"      {label:<30} {stat_a:>8.3g} vs {stat_b:>8.3g}"
                )
    if not rows:
        logger.warning("no divergence outcome comparison cleared the floor")
        return pl.DataFrame()
    return pl.DataFrame(rows)


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


def plot_factor_dag(long, offsets, out, logger, min_share: float = 0.04,
                    state_column: str = "max_factor",
                    cluster_column: str = "growth_cluster",
                    labeller=None, short=None):
    """
    each latent class as a path through the 12 NMF factors, laid out to be read

    Not on the projection. The PaCMAP-placed graphs put nodes at their median
    embedding position, which ties the graph to the geometry but packs twelve
    nodes into whatever space the cloud happens to occupy -- unreadable past
    about six. Here the nodes are spread on a circle, so every edge is visible,
    and the cost is that distance on the page no longer means anything.

    Node colour is the factor's colour in the hexagon maps and node order round
    the circle is the MEAN WINDOW INDEX at which that factor dominates, so the
    ring is roughly a timeline and a cluster that progresses reads as motion
    round it rather than as a scribble.

    Arrow colour is the latent class's colour in the flow overlay. One panel
    per class, sharing the node layout, because four classes of arrows over
    twelve nodes on one axes is 500-odd possible edges and nothing legible.

    Each window is assigned to its single largest factor. That is a hard
    reading of a soft representation and it is the point of comparison: if
    these paths look like the care-state graph, the k-means hardening into six
    states lost little.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from clifcs.figures import save_figure
    from clifcs.palette import (
        CONTEXT, GRID, INK_SECONDARY, cluster_colors, husl_colors,
    )

    if state_column not in long.columns or cluster_column not in long.columns:
        logger.info(
            f"no {state_column} / {cluster_column} columns; graph skipped"
        )
        return
    frame = long.filter(
        pl.col(state_column).is_not_null()
        & pl.col(cluster_column).is_not_null()
    )
    if frame.is_empty():
        logger.warning(f"no {state_column} rows; graph skipped")
        return
    labeller = labeller or (lambda key: key)
    short = short or (lambda key: key.replace("factor ", "F"))

    factors = sorted(frame[state_column].unique().to_list())
    classes = sorted(frame[cluster_column].unique().to_list())
    factor_colors = husl_colors(factors)
    # the latent classes take the deeper ramp, since here they are thin arrows
    class_colors = cluster_colors(classes)

    # order the ring by when each factor tends to dominate
    index_of_window = {float(o): i for i, o in enumerate(offsets)}
    timing = (
        frame.with_columns(
            position=pl.col("offset_hours").replace_strict(
                index_of_window, default=None, return_dtype=pl.Float64
            )
        )
        .group_by(state_column)
        .agg(mean_position=pl.col("position").mean(), n=pl.len())
    )
    order = (
        timing.sort("mean_position")[state_column].to_list()
        if not timing.is_empty() else factors
    )
    order = [f for f in order if f in factors] + [f for f in factors if f not in order]
    angles = {
        f: 2.0 * np.pi * i / len(order) - np.pi / 2.0
        for i, f in enumerate(order)
    }
    nodes = {f: (np.cos(a), np.sin(a)) for f, a in angles.items()}
    occupancy = dict(zip(timing[state_column].to_list(), timing["n"].to_list()))
    largest = max(occupancy.values()) if occupancy else 1

    columns = 2
    rows = int(np.ceil(len(classes) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.6 * columns, 4.4 * rows), squeeze=False
    )

    for index, cluster in enumerate(classes):
        ax = axes[index // columns][index % columns]
        part = frame.filter(pl.col(cluster_column) == cluster).sort(
            "hospitalization_id", "offset_hours"
        )
        pairs = (
            part.with_columns(
                next_factor=pl.col(state_column).shift(-1).over("hospitalization_id"),
                next_t=pl.col("offset_hours").shift(-1).over("hospitalization_id"),
            )
            .drop_nulls(["next_factor", "next_t"])
            .filter(
                (pl.col("next_t") - pl.col("offset_hours")
                 - (offsets[1] - offsets[0])).abs() < 1e-6
            )
        )
        counts = {}
        for a, b in zip(pairs[state_column].to_list(),
                        pairs["next_factor"].to_list()):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        departures = {
            f: sum(n for (a, _), n in counts.items() if a == f) for f in order
        }

        drawn = 0
        for (a, b), n in sorted(counts.items(), key=lambda kv: kv[1]):
            if a == b or a not in nodes or b not in nodes:
                continue
            share = n / max(departures.get(a, 1), 1)
            if share < min_share:
                continue
            ax.annotate(
                "", xy=nodes[b], xytext=nodes[a],
                arrowprops=dict(
                    arrowstyle="-|>", color=class_colors[cluster],
                    alpha=0.8, linewidth=0.5 + 3.5 * share,
                    shrinkA=11, shrinkB=13,
                    connectionstyle="arc3,rad=0.18",
                ),
                zorder=2,
            )
            drawn += 1

        for factor in order:
            nx_, ny_ = nodes[factor]
            stay = counts.get((factor, factor), 0) / max(departures.get(factor, 1), 1)
            size = 120 + 520 * (occupancy.get(factor, 0) / largest)
            ax.scatter([nx_], [ny_], s=size, c=factor_colors[factor],
                       edgecolors="#ffffff", linewidths=1.0, zorder=3)
            label_r = 1.26
            ax.annotate(
                short(factor),
                xy=(np.cos(angles[factor]) * label_r,
                    np.sin(angles[factor]) * label_r),
                ha="center", va="center", fontsize=6, color=INK_SECONDARY,
                zorder=4,
            )
            if stay > 0:
                ax.annotate(
                    f"{stay:.0%}", xy=(nx_, ny_), ha="center", va="center",
                    fontsize=5, color="#ffffff", zorder=5, fontweight="bold",
                )

        n_patients = part["hospitalization_id"].n_unique()
        ax.set_title(
            f"{cluster}  —  {n_patients:,} patients, {drawn} transitions "
            f"above {min_share:.0%}",
            loc="left", fontsize=8, pad=6, color=class_colors[cluster],
        )
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal")
        ax.axis("off")
        logger.info(
            f"  {cluster}: {n_patients:,} patients, {len(counts):,} distinct "
            f"transitions, {drawn} drawn above {min_share:.0%}"
        )

    for index in range(len(classes), rows * columns):
        axes[index // columns][index % columns].axis("off")

    figure.suptitle(labeller("title"), x=0.01, ha="left", fontsize=10)
    figure.text(
        0.01, 0.005,
        "Nodes are the 12 NMF factors in the hexagon-map colours, placed on a "
        "circle ordered by the mean window at which each dominates, sized by "
        "how many windows it dominates; the figure inside is the share that "
        "stay. Arrows are in each class's flow-overlay colour, width is the "
        "share of departures. Each window is assigned to its single largest "
        "factor, which is a hard reading of a soft representation.",
        fontsize=5.5, color=INK_SECONDARY, wrap=True,
    )
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out}")


def plot_patient_factor_paths(long, offsets, patients, out_dir, logger,
                              state_column: str = "max_factor",
                              cluster_column: str = "growth_cluster",
                              labeller=None, short=None, stem="factor"):
    """
    one figure per patient: their own route through the 12 NMF factors

    Separate files rather than subplots, so the report can lay them out side by
    side at whatever width it likes and each is legible at full size. A
    six-panel grid put twelve labelled nodes into about three inches and
    nothing could be read.

    Node fill is the factor's own colour at full opacity -- the same colour it
    carries in the hexagon maps -- and the OUTLINE runs white at the earliest
    window to black at the latest, so colour answers "which factor" and the ring
    answers "when". The previous version had those the other way round, which
    made the twelve factors indistinguishable from each other.

    Factors the patient never visited are drawn faint, so "never went there" is
    visible rather than absent. Arrows are black: the arrow is the patient and
    the colour is the destination.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from clifcs.figures import save_figure
    from clifcs.longitudinal import check_factor_labels, factor_label
    from clifcs.palette import INK_PRIMARY, INK_SECONDARY, husl_colors

    if state_column not in long.columns:
        logger.info(f"no {state_column} column; per-patient paths skipped")
        return []
    frame = long.filter(pl.col(state_column).is_not_null())
    if frame.is_empty():
        logger.warning(f"no {state_column} rows; per-patient paths skipped")
        return []
    labeller = labeller or (lambda key: key)

    factors = sorted(frame[state_column].unique().to_list())
    factor_colors = husl_colors(factors)
    index_of_window = {float(o): i for i, o in enumerate(offsets)}
    timing = (
        frame.with_columns(
            position=pl.col("offset_hours").replace_strict(
                index_of_window, default=None, return_dtype=pl.Float64
            )
        )
        .group_by(state_column)
        .agg(mean_position=pl.col("position").mean())
        .sort("mean_position")
    )
    order = [f for f in timing[state_column].to_list() if f in factors]
    order += [f for f in factors if f not in order]
    angles = {
        f: 2.0 * np.pi * i / len(order) - np.pi / 2.0 for i, f in enumerate(order)
    }
    nodes = {f: (np.cos(a), np.sin(a)) for f, a in angles.items()}
    ramp = mcolors.LinearSegmentedColormap.from_list("time", ["#ffffff", "#111111"])

    written = []
    for patient in patients:
        track = (
            frame.filter(pl.col("hospitalization_id") == patient)
            .sort("offset_hours")
        )
        if track.height < 3:
            continue
        visited = track[state_column].to_list()
        hours = track["offset_hours"].to_list()
        cluster = (
            track[cluster_column].drop_nulls().to_list()
            if cluster_column in track.columns else []
        )

        figure, ax = plt.subplots(figsize=(5.4, 5.4))
        for factor in order:
            nx_, ny_ = nodes[factor]
            ax.scatter([nx_], [ny_], s=150, c=factor_colors[factor],
                       alpha=0.16, linewidths=0, zorder=2)
            ax.annotate(
                labeller(factor),
                xy=(np.cos(angles[factor]) * 1.34,
                    np.sin(angles[factor]) * 1.34),
                ha="center", va="center", fontsize=5.5, color=INK_SECONDARY,
                zorder=3, wrap=True,
            )

        span = max(len(hours) - 1, 1)
        for step, (factor, hour) in enumerate(zip(visited, hours)):
            nx_, ny_ = nodes[factor]
            if step + 1 < len(visited) and visited[step + 1] != factor:
                ax.annotate(
                    "", xy=nodes[visited[step + 1]], xytext=(nx_, ny_),
                    arrowprops=dict(
                        arrowstyle="-|>", color=INK_PRIMARY, alpha=0.8,
                        linewidth=1.1, shrinkA=11, shrinkB=13,
                        connectionstyle="arc3,rad=0.16",
                    ),
                    zorder=4,
                )
            ax.scatter([nx_], [ny_], s=210, c=[factor_colors[factor]],
                       edgecolors=[ramp(step / span)], linewidths=2.2,
                       zorder=5)
            ax.annotate(
                f"{hour:+.0f}h",
                xy=(nx_ * 1.15, ny_ * 1.15), ha="center", va="center",
                fontsize=5, color=INK_SECONDARY, zorder=6,
            )

        title = f"{patient}"
        if cluster:
            title = f"{title}  ({cluster[0]})"
        ax.set_title(
            f"{title}\n{len(set(visited))} of {len(order)} factors "
            "visited; node colour = factor, outline = time",
            loc="left", fontsize=7.5, pad=6,
        )
        ax.set_xlim(-1.62, 1.62)
        ax.set_ylim(-1.62, 1.62)
        ax.set_aspect("equal")
        ax.axis("off")

        target = out_dir / f"patient_{stem}_path_{len(written) + 1:02d}.png"
        save_figure(figure, target)
        plt.close(figure)
        written.append(target)
        logger.info(
            f"  {patient}: {len(set(visited))} distinct factors over "
            f"{len(visited)} windows -> {target.name}"
        )
    return written


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

    states, topics, nmf_model = care_states(
        matrix, args.states, args.seed, logger
    )
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
    logger.info("sequences per cluster:")
    cluster_sequence_table(states, sequence_labels, OFFSETS, logger).write_csv(
        paths.output_home / "tables" / paths.model_key
        / f"cluster_sequences_{args.subset}.csv"
    )
    vocab, _fam = (
        vocabulary(str(paths.tokenizer_yaml))
        if paths.tokenizer_yaml_or_none is not None else ({}, {})
    )
    if vocab:
        factor_token_table(nmf_model, token_ids, vocab, logger).write_csv(
            paths.output_home / "tables" / paths.model_key
            / f"nmf_factor_tokens_{args.subset}.csv"
        )
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

    # The clinically-defined counterpart: how patients move through the four
    # MAP-by-pressor states. Independent of everything above -- it reads
    # physiology, not tokens -- so it is a second opinion on the same patients
    # rather than a variation on the same features.
    perfusion = perfusion_sequences(paths, cohort, OFFSETS, logger)
    if not perfusion.is_empty():
        perfusion_labels = cluster_perfusion_sequences(
            perfusion, OFFSETS, args.clusters, args.seed, logger
        )
        if not perfusion_labels.is_empty():
            labels_frame = labels_frame.join(
                perfusion_labels, on="hospitalization_id", how="left"
            )
            assignments = assignments.join(
                perfusion_labels.select(
                    "hospitalization_id", "perfusion_cluster",
                    "perfusion_sequence",
                ),
                on="hospitalization_id", how="left",
            )
            symbols = list(PERFUSION_ORDER) + ["record ended"]
            # mortality on the same table, so a reader does not have to carry
            # the sequence here and the outcome from another section
            outcome = labels_frame.join(
                perfusion_labels.select("hospitalization_id", "perfusion_cluster"),
                on="hospitalization_id", how="inner",
            ) if "perfusion_cluster" not in labels_frame.columns else labels_frame
            mortality = {}
            if "in_hospital_mortality" in outcome.columns:
                grouped = (
                    outcome.filter(pl.col("in_hospital_mortality").is_not_null())
                    .group_by("perfusion_cluster")
                    .agg(died=pl.col("in_hospital_mortality").cast(pl.Float64).mean())
                )
                mortality = dict(zip(
                    grouped["perfusion_cluster"].to_list(),
                    grouped["died"].to_list(),
                ))
            rows = []
            for cluster in sorted(
                perfusion_labels["perfusion_cluster"].unique().to_list()
            ):
                part = perfusion_labels.filter(
                    pl.col("perfusion_cluster") == cluster
                )
                top = part["perfusion_sequence"].value_counts(sort=True).head(3)
                for rank, (sequence, n) in enumerate(
                    zip(top["perfusion_sequence"].to_list(),
                        top["count"].to_list()), start=1
                ):
                    rows.append({
                        "cluster": cluster,
                        "patients": part.height,
                        "died": mortality.get(cluster, float("nan")),
                        "rank": rank,
                        "sequence": sequence,
                        "patients_with_this_sequence": n,
                        "share_of_cluster": n / max(part.height, 1),
                        "median_changes": float(part["perfusion_changes"].median()),
                    })
            (paths.output_home / "tables" / paths.model_key).mkdir(
                parents=True, exist_ok=True
            )
            pl.DataFrame(rows).write_csv(
                paths.output_home / "tables" / paths.model_key
                / f"perfusion_sequences_{args.subset}.csv"
            )
            pl.DataFrame({
                "symbol": [str(i) for i in range(len(symbols))],
                "state": symbols,
            }).write_csv(
                paths.output_home / "tables" / paths.model_key
                / f"perfusion_symbol_key_{args.subset}.csv"
            )

    # The richer state space and the motif clustering on top of it.
    motif_labels, rich_long = (pl.DataFrame(), pl.DataFrame())
    if not perfusion.is_empty():
        tables_home_early = paths.output_home / "tables" / paths.model_key
        tables_home_early.mkdir(parents=True, exist_ok=True)
        motif_labels, rich_long = motif_clusters(
            perfusion, OFFSETS, args.clusters, args.seed, logger,
            tables_home_early, subset=args.subset,
        )
        if not motif_labels.is_empty():
            labels_frame = labels_frame.join(
                motif_labels.select(
                    [c for c in motif_labels.columns
                     if not c.startswith("sequence_")]
                ),
                on="hospitalization_id", how="left",
            )
            assignments = assignments.join(
                motif_labels, on="hospitalization_id", how="left"
            )

    # Matched-prefix divergence: among patients with the same path so far,
    # what was in the record just before they went different ways.
    if not motif_labels.is_empty() and "sequence_raw" in motif_labels.columns:
        sequences = dict(zip(
            motif_labels["hospitalization_id"].to_list(),
            motif_labels["sequence_raw"].to_list(),
        ))
        vocab_now, _fam = (
            vocabulary(str(paths.tokenizer_yaml))
            if paths.tokenizer_yaml_or_none is not None else ({}, {})
        )
        for prefix_length in (2, 3):
            found = divergence_tokens(
                sequences, counts, token_ids, vocab_now, OFFSETS, logger,
                prefix_length=prefix_length,
            )
            if not found.is_empty():
                found.write_csv(
                    tables_home_early
                    / f"divergence_tokens_prefix{prefix_length}_{args.subset}.csv"
                )

    # Treatment and haemodynamics over the SEVEN-DAY window, from the state
    # table this script built for the perfusion sequences. 09's
    # trajectory_state.parquet is keyed to the 13-point embedding grid and
    # stops at +48h, so aggregating it would describe the first two days and
    # label the result as a seven-day stratum.
    if not perfusion.is_empty():
        per_patient = (
            perfusion.group_by("hospitalization_id").agg(
                map_median=pl.col("map").median(),
                map_min=pl.col("map").min(),
                nee_max=pl.col("nee").max(),
                nee_median=pl.col("nee").median(),
                n_vasopressors_max=pl.col("n_vasopressors").max(),
                ever_imv=pl.col("on_imv").max() > 0,
                hours_hypotensive=(
                    pl.col("perfusion_state").str.starts_with("MAP <65").sum()
                    * STEP_HOURS
                ),
            )
        )
        labels_frame = labels_frame.join(
            per_patient, on="hospitalization_id", how="left"
        )
    else:
        logger.warning("no perfusion state table: treatment columns omitted")

    names, _families = (
        vocabulary(str(paths.tokenizer_yaml))
        if paths.tokenizer_yaml_or_none is not None else ({}, {})
    )
    differential = []
    # How much of the cohort each state accounts for -- asked for explicitly,
    # and previously only in the log.
    if not rich_long.is_empty():
        representation = (
            rich_long["rich_state"].value_counts(sort=True)
            .with_columns(
                share=pl.col("count") / rich_long.height,
                symbol=pl.col("rich_state").replace_strict(
                    {name: state_symbol(i)
                     for i, name in enumerate(PERFUSION_STATES)},
                    default="?", return_dtype=pl.String,
                ),
            )
            .select("symbol", "rich_state", "count", "share")
        )
        representation.write_csv(
            paths.output_home / "tables" / paths.model_key
            / f"state_representation_{args.subset}.csv"
        )

    tables_home = paths.output_home / "tables" / paths.model_key
    tables_home.mkdir(parents=True, exist_ok=True)
    stratifications = ["sequence_cluster", "growth_cluster"]
    for extra in ("perfusion_cluster", "motif_cluster_raw", "motif_cluster_rle"):
        if extra in labels_frame.columns:
            stratifications.append(extra)
    for label in stratifications:
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

    if not motif_labels.is_empty() and "sequence_raw" in motif_labels.columns:
        sequences_now = dict(zip(
            motif_labels["hospitalization_id"].to_list(),
            motif_labels["sequence_raw"].to_list(),
        ))
        for prefix_length in (2, 3):
            found = divergence_outcomes(
                sequences_now, labels_frame, OFFSETS, logger,
                prefix_length=prefix_length,
            )
            if not found.is_empty():
                found.write_csv(
                    tables_home
                    / f"divergence_outcomes_prefix{prefix_length}_{args.subset}.csv"
                )

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
    # The SOFT layer travels too. k-means hardens the 12 topic weights into
    # six states and that is what the sequences are built from, but the weights
    # themselves are the only view of what hardening discarded -- a window that
    # was 51/49 between two topics becomes indistinguishable from one that was
    # 99/1. `max_factor` is the argmax label the soft representation implies,
    # which is a different partition from the k-means states and worth
    # comparing against them.
    flat_topics = topics.reshape(-1, topics.shape[2])
    topic_columns = {
        f"topic_{i + 1:02d}": flat_topics[:, i].astype(np.float32)
        for i in range(flat_topics.shape[1])
    }
    dominant = flat_topics.argmax(axis=1)
    occupied = flat_topics.sum(axis=1) > 0

    long = pl.DataFrame({
        "hospitalization_id": np.repeat(patients, len(OFFSETS)),
        "offset_hours": np.tile(np.array(OFFSETS, dtype=np.float64), len(patients)),
        "care_state": [
            "nothing recorded" if int(v) == 0 else f"state {int(v)}"
            for v in states.reshape(-1)
        ],
        "max_factor": [
            f"factor {d + 1:02d}" if ok else None
            for d, ok in zip(dominant, occupied)
        ],
        **topic_columns,
    }).join(
        assignments.select(
            "hospitalization_id", "sequence_cluster", "growth_cluster"
        ),
        on="hospitalization_id", how="left",
    )
    # the per-(patient, window) perfusion state travels in the same long table,
    # so 09 can colour by it on exactly the same join key
    if not perfusion.is_empty():
        long = long.join(
            perfusion.select(
                "hospitalization_id", "offset_hours", "perfusion_state"
            ),
            on=["hospitalization_id", "offset_hours"],
            how="left",
        )
        if "perfusion_cluster" in labels_frame.columns:
            long = long.join(
                labels_frame.select(
                    "hospitalization_id", "perfusion_cluster"
                ).unique(),
                on="hospitalization_id", how="left",
            )

    if not rich_long.is_empty():
        long = long.join(
            rich_long, on=["hospitalization_id", "offset_hours"], how="left"
        )
        if not motif_labels.is_empty():
            long = long.join(
                motif_labels.select([
                    c for c in motif_labels.columns
                    if not c.startswith("sequence_")
                ]),
                on="hospitalization_id", how="left",
            )

    long_out = (
        paths.patient_level_home
        / f"longitudinal_states_{args.subset}.parquet"
    )
    long.write_parquet(long_out)
    # the same patients 09 picks as flow followers, so the two figures show the
    # same people; falls back to the longest records when that file is absent
    followers = []
    projections = sorted(
        paths.patient_level_home.glob(
            f"trajectory_projections_{args.subset}*.parquet"
        )
    )
    if projections:
        followers = (
            pl.read_parquet(projections[0], columns=["hospitalization_id"])
            ["hospitalization_id"].unique().to_list()
        )
    present = set(long["hospitalization_id"].to_list())
    followers = [p for p in followers if p in present][:6]
    if not followers:
        followers = (
            long.group_by("hospitalization_id").len().sort("len", descending=True)
            ["hospitalization_id"].to_list()[:6]
        )
    logger.info("individual patients through the NMF factors:")
    if vocab:
        check_factor_labels(nmf_model, token_ids, vocab, logger)
    factor_dir = paths.figure_home / f"patient_factor_paths_{args.subset}"
    factor_dir.mkdir(parents=True, exist_ok=True)
    plot_patient_factor_paths(long, OFFSETS, followers, factor_dir, logger)

    # The same two figures for the MAP/pressor state space, so the two
    # analyses can be compared panel for panel rather than by description.
    if "rich_state" in long.columns and "motif_cluster_rle" in long.columns:
        def state_name(key):
            if key == "title":
                return "Motif clusters: paths through the MAP/pressor states"
            from clifcs.perfusion import PERFUSION_STATES, state_symbol
            try:
                return f"{state_symbol(PERFUSION_STATES.index(key))} {key}"
            except ValueError:
                return key

        logger.info("motif-cluster paths through the MAP/pressor states:")
        plot_factor_dag(
            long, OFFSETS,
            paths.figure_home / f"longitudinal_state_dag_{args.subset}.png",
            logger, state_column="rich_state",
            cluster_column="motif_cluster_rle",
            labeller=state_name,
            short=lambda k: k if k == "title" else (
                "" if k not in PERFUSION_STATES
                else state_symbol(PERFUSION_STATES.index(k))
            ),
        )
        state_dir = paths.figure_home / f"patient_state_paths_{args.subset}"
        state_dir.mkdir(parents=True, exist_ok=True)
        logger.info("individual patients through the MAP/pressor states:")
        plot_patient_factor_paths(
            long, OFFSETS, followers, state_dir, logger,
            state_column="rich_state", cluster_column="motif_cluster_rle",
            labeller=state_name, stem="state",
        )

    logger.info("latent class paths through the NMF factors:")
    plot_factor_dag(
        long, OFFSETS,
        paths.figure_home / f"longitudinal_factor_dag_{args.subset}.png",
        logger,
    )
    logger.info(
        f"wrote {long_out} ({long.height:,} patient-windows, "
        f"{len(OFFSETS)} per patient)"
    )
    logger.info("11 done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
undifferentiated shock: no coded mechanism, and what the embedding makes of it

The cohort is selected on physiology and is therefore mechanism-agnostic by
construction. `clifcs.shock_type` attaches the coded mechanism and leaves
18,626 of 34,142 patients -- 54.6% -- with no shock code at all. This module
asks what those patients resemble in the frozen model's representation, whether
a linear probe trained on the coded patients will assign them a mechanism, and
whether the treatment they received matches the assignment.

THE THREE THINGS TO KNOW BEFORE READING ANY NUMBER FROM HERE

1. **The two sites code different mechanisms, not different amounts.** The
   uncoded share is close -- NU 56.6%, UCMC 48.9%, a 1.16-fold gap. What
   differs is which mechanism is reached for when one is coded: UCMC codes
   cardiogenic 2.5 times as often as NU (13.5% against 5.3%), NU codes septic
   1.7 times as often as UCMC (20.3% against 12.3%). The coded patients are the
   probe's training set and the centroids' definition, so a mechanism mix that
   differs that much means neither the priors nor the centroids transport.
   Every centroid is fitted within site, every neighbourhood is searched within
   site, and the probe is scored within-site and cross-site separately.

2. **The uncoded are much less sick.** 6.2% in-hospital mortality pooled --
   4.6% at NU, 11.2% at UCMC -- against 25.2% (cardiogenic), 29.6% (septic) and
   43.3% (mixed). Most of this group is
   not a diagnostic puzzle; it is a mild population that the physiological
   criteria admitted and nobody thought was in shock. A finding that the
   uncoded sit far from every subtype centroid is in the first instance a
   finding about severity, and is reported against a severity-matched
   comparison for that reason.

3. **Whether a diagnosis arrives later is not answerable.** The diagnosis table
   is hospitalization-level and untimed, and an uncoded patient has no shock
   code at any point in the admission by definition. What this module reports
   instead is whether a mechanism was recorded by some *other* means -- a
   positive culture, or a non-shock code that implies one -- somewhere in the
   admission. That is a weaker claim and is labelled as one everywhere it
   appears.

WHAT THIS MODULE DOES NOT CONTAIN

The treatment comparison is missing three of its arms: fluid volume (no
intake/output table), transfusion (no blood product table) and echo or POCUS
(no echo table at either site). The third is the serious one. Echo is the main
act of differentiating shock, so its absence means this analysis sees only the
drugs that follow the differentiation, never the differentiation itself.
See `code/config/undifferentiated.yaml`, which states each absence beside the
setting it would have governed.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from clifcs.clif import has_table, scan

#: the label `shock_type` gives a patient carrying no shock code
UNDIFFERENTIATED = "none coded"

#: the coded levels that name a mechanism, in reading order. `mixed` and
#: `other shock` are comparison groups, not mechanisms: see `GROUPS`.
MECHANISMS: tuple[str, ...] = ("septic", "cardiogenic", "hypovolemic")

#: every group the tables report, in reading order
GROUPS: tuple[str, ...] = (
    UNDIFFERENTIATED,
    "other shock",
    "mixed",
    *MECHANISMS,
)


def embedding_matrix(frame: pl.DataFrame) -> np.ndarray:
    """the emb_* columns as a dense array, in a stable column order"""
    columns = sorted(c for c in frame.columns if c.startswith("emb_"))
    if not columns:
        raise KeyError("no emb_* columns; the embeddings have not been computed")
    return frame.select(columns).to_numpy().astype(np.float64)


def pca_coordinates(frame: pl.DataFrame, components: int, seed: int) -> np.ndarray:
    """
    the 50-d subspace the neighbourhoods are measured in

    Same scaler, same component count and same seed as 03, 04 and
    `clifcs.purity`, so a purity figure from here is comparable to the 97% site
    purity the README quotes rather than being on its own scale.
    """
    scaled = StandardScaler().fit_transform(embedding_matrix(frame))
    return PCA(
        n_components=min(components, scaled.shape[1], frame.height - 1),
        random_state=seed,
    ).fit_transform(scaled)


# =============================================================== the cohort


def describe_groups(cohort: pl.DataFrame, logger) -> pl.DataFrame:
    """
    one row per (site, shock_type): how many, how sick, how they ended

    Site is a row key rather than a column to average over. The whole reason
    this table exists is that the uncoded share differs 3.2-fold between the
    two hospitals, and a pooled row would hide exactly that.
    """
    severity = [
        pl.len().alias("n"),
        pl.col("in_hospital_mortality").mean().alias("mortality"),
        pl.col("age_at_admission").median().alias("median_age"),
        pl.col("lactate").median().alias("median_lactate"),
        pl.col("icu_los_days").median().alias("median_icu_los"),
        (pl.col("sex_category") == "Female").mean().alias("pct_female"),
    ]
    if "sofa_total" in cohort.columns:
        severity.append(pl.col("sofa_total").median().alias("median_sofa"))
    if "met_vasoactive_support" in cohort.columns:
        severity.append(
            pl.col("met_vasoactive_support").mean().alias("pct_on_vasoactive")
        )

    table = (
        cohort.group_by("site", "shock_type")
        .agg(severity)
        .with_columns(
            share_of_site=pl.col("n") / pl.col("n").sum().over("site"),
        )
        .sort("site", "shock_type")
    )
    for row in table.filter(pl.col("shock_type") == UNDIFFERENTIATED).iter_rows(
        named=True
    ):
        logger.info(
            f"  [{row['site']}] {row['n']:,} undifferentiated "
            f"({row['share_of_site']:.1%} of the site), "
            f"{row['mortality']:.1%} mortality"
        )
    return table


def _normalized_code() -> pl.Expr:
    """ICD code with dots and whitespace stripped, upper-cased

    The same normalisation `clifcs.shock_type` uses, so a prefix written one
    way in this module's config matches the same rows it would match there.
    """
    return (
        pl.col("diagnosis_code")
        .cast(pl.String)
        .str.replace_all(r"[.\s]", "")
        .str.to_uppercase()
    )


def mechanism_evidence(data_dir, cohort: pl.DataFrame, cfg, logger) -> pl.DataFrame:
    """
    evidence of a mechanism from somewhere other than a shock code

    Returns one row per hospitalization in `cohort` with a boolean per
    mechanism and a positive-culture flag, so the join in the caller cannot
    drop patients.

    THIS IS NOT "THE DIAGNOSIS ARRIVED LATER". There is no timestamp on a
    hospital diagnosis, so an infection code and a positive culture say that a
    mechanism was recorded *during the admission*, not that it was recorded
    after the shock. Reading these columns as a sequence is the one thing they
    cannot support, and the summary says so in those words.

    Cultures are counted as positive only where something grew. The table holds
    one row per organism tested for, so `organism_category` is non-null on
    every UCMC row -- non-nullness is a record that a culture was sent, not
    that it was positive.
    """
    ids = cohort.select("hospitalization_id")
    spec = cfg.later_evidence
    groups = {k: list(v) for k, v in spec.code_groups.items()}

    columns: dict[str, pl.Expr] = {
        f"hint_{name}": pl.lit(False) for name in groups
    }
    columns["culture_positive"] = pl.lit(False)
    columns["culture_sent"] = pl.lit(False)
    columns["culture_unresulted"] = pl.lit(False)
    evidence = ids.with_columns(**columns)

    # ------------------------------------------------- mechanism-implying codes
    if has_table(data_dir, "clif_hospital_diagnosis"):
        diagnoses = (
            scan(data_dir, "clif_hospital_diagnosis")
            .select("hospitalization_id", "diagnosis_code")
            .with_columns(code=_normalized_code())
            .join(ids.lazy(), on="hospitalization_id", how="semi")
        )
        # One anchored alternation per group rather than an or-chain of
        # starts_with: the prefixes are alphanumeric, so nothing in them is a
        # regex metacharacter, and the single pass reads as the list it came
        # from.
        hits = diagnoses.group_by("hospitalization_id").agg(
            **{
                f"hint_{name}": pl.col("code")
                .str.contains("^(?:" + "|".join(prefixes) + ")")
                .any()
                for name, prefixes in groups.items()
            }
        )
        evidence = (
            evidence.drop(*(f"hint_{n}" for n in groups))
            .join(hits.collect(), on="hospitalization_id", how="left")
            .with_columns(
                **{
                    f"hint_{n}": pl.col(f"hint_{n}").fill_null(False)
                    for n in groups
                }
            )
        )
    else:
        logger.warning(
            "  no clif_hospital_diagnosis; every mechanism hint is False, "
            "which will read as 'no evidence' rather than as 'not looked for'"
        )

    # -------------------------------------------------------------- cultures
    if spec.culture_positive and has_table(data_dir, "clif_microbiology_culture"):
        no_growth = str(spec.no_growth_value)
        cultures = (
            scan(data_dir, "clif_microbiology_culture")
            .select("hospitalization_id", "organism_category")
            .join(ids.lazy(), on="hospitalization_id", how="semi")
            .group_by("hospitalization_id")
            .agg(
                culture_sent=pl.lit(True),
                culture_positive=(
                    pl.col("organism_category").is_not_null()
                    & (pl.col("organism_category") != no_growth)
                ).any(),
                culture_unresulted=pl.col("organism_category").is_null().any(),
            )
            .collect()
        )
        evidence = (
            evidence.drop("culture_positive", "culture_sent", "culture_unresulted")
            .join(cultures, on="hospitalization_id", how="left")
            .with_columns(
                culture_sent=pl.col("culture_sent").fill_null(False),
                culture_positive=pl.col("culture_positive").fill_null(False),
                culture_unresulted=pl.col("culture_unresulted").fill_null(False),
            )
        )
    else:
        logger.warning("  no clif_microbiology_culture; culture flags are all False")

    return evidence


# ============================================================== the geometry


def _cosine_distance_to(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """1 - cosine similarity, guarding the zero vector"""
    norms = np.linalg.norm(points, axis=1)
    centre_norm = np.linalg.norm(centre)
    if centre_norm == 0:
        return np.full(points.shape[0], np.nan)
    similarity = (points @ centre) / np.where(norms == 0, np.nan, norms) / centre_norm
    return 1.0 - similarity


def centroid_proximity(
    frame: pl.DataFrame, cfg, seed: int, logger
) -> pl.DataFrame:
    """
    cosine distance from every patient to each mechanism's centroid

    Centroids are the mean raw embedding of that mechanism's coded patients
    **at this site**, and the distances are measured only against the site's own
    centroids. Fitting them pooled would put NU's coding practice into UCMC's
    reference point, which is the one thing this analysis is organised to avoid.

    Two metrics, because a conclusion that depends on the metric should be
    visible rather than implicit:

      cos_{mechanism}   cosine distance in the raw 1024-d space. The primary
                        measure: an angle between representations does not
                        depend on the cohort's mean, which a distance in a
                        centred and rescaled subspace does.
      pca_{mechanism}   Euclidean distance in the 50-d PCA space 03 uses,
                        reported so the two rankings can be correlated.

    Also returns, per patient, which centroid is nearest and the margin between
    the nearest and the runner-up. A patient 0.01 nearer to septic than to
    cardiogenic has a nearest centroid and no meaningful assignment, and the
    margin is what distinguishes the two cases.

    A SELF-COMPARISON IS NOT A COMPARISON. A coded patient contributes to their
    own mechanism's centroid, so their distance to it is biased downward --
    more so the smaller the group. Their own-centroid distance is therefore
    recomputed leave-one-out, which for a mean is exact and costs nothing. The
    reference distributions in `reference_separation` depend on this.
    """
    geometry = cfg.geometry
    mechanisms = list(MECHANISMS)
    parts = []

    for site in frame["site"].unique(maintain_order=True).to_list():
        block = frame.filter(pl.col("site") == site)
        raw = embedding_matrix(block)
        pca = pca_coordinates(block, int(geometry.pca_components), seed)
        labels = block["shock_type"].cast(pl.String).to_numpy()

        columns: dict[str, np.ndarray] = {}
        for mechanism in mechanisms:
            members = labels == mechanism
            n = int(members.sum())
            if n == 0:
                logger.warning(
                    f"  [{site}] no {mechanism} patients; its centroid is null"
                )
                columns[f"cos_{mechanism}"] = np.full(block.height, np.nan)
                columns[f"pca_{mechanism}"] = np.full(block.height, np.nan)
                continue

            centre_raw = raw[members].mean(axis=0)
            centre_pca = pca[members].mean(axis=0)
            cos = _cosine_distance_to(raw, centre_raw)
            euclid = np.linalg.norm(pca - centre_pca, axis=1)

            # leave-one-out for the members, so a patient is not scored
            # against a centroid they helped define
            if n > 1:
                loo_raw = (n * centre_raw - raw[members]) / (n - 1)
                loo_pca = (n * centre_pca - pca[members]) / (n - 1)
                member_cos = 1.0 - np.einsum(
                    "ij,ij->i", raw[members], loo_raw
                ) / (
                    np.linalg.norm(raw[members], axis=1)
                    * np.linalg.norm(loo_raw, axis=1)
                )
                cos[members] = member_cos
                euclid[members] = np.linalg.norm(pca[members] - loo_pca, axis=1)

            columns[f"cos_{mechanism}"] = cos
            columns[f"pca_{mechanism}"] = euclid

        stacked = np.vstack([columns[f"cos_{m}"] for m in mechanisms])
        order = np.argsort(np.where(np.isnan(stacked), np.inf, stacked), axis=0)
        nearest = np.array([mechanisms[i] for i in order[0]])
        best = np.take_along_axis(stacked, order[:1], axis=0)[0]
        runner_up = (
            np.take_along_axis(stacked, order[1:2], axis=0)[0]
            if len(mechanisms) > 1
            else np.full(block.height, np.nan)
        )

        parts.append(
            block.select("hospitalization_id", "site", "shock_type").with_columns(
                **{k: pl.Series(v) for k, v in columns.items()},
                nearest_centroid=pl.Series(nearest),
                centroid_margin=pl.Series(runner_up - best),
            )
        )

    return pl.concat(parts, how="diagonal_relaxed")


def knn_composition(frame: pl.DataFrame, cfg, seed: int, logger) -> pl.DataFrame:
    """
    the mechanism composition of each patient's nearest coded neighbours

    For each k in the config, the share of the k nearest *coded* patients
    carrying each mechanism. Neighbours are searched in the 50-d PCA space and
    within site, for the same reason the centroids are fitted that way.

    Only coded patients are candidate neighbours. Searching among everyone
    would make the answer for an undifferentiated patient mostly "my neighbours
    are also undifferentiated", which is true, uninteresting, and guaranteed by
    the group being 54.6% of the cohort.

    A coded patient never counts as their own neighbour -- the self-match is
    dropped -- so these shares are comparable between the coded patients, where
    they are the reference, and the undifferentiated, where they are the
    measurement.
    """
    from sklearn.neighbors import NearestNeighbors

    ks = sorted(int(k) for k in cfg.geometry.neighbours)
    mechanisms = list(MECHANISMS)
    parts = []

    for site in frame["site"].unique(maintain_order=True).to_list():
        block = frame.filter(pl.col("site") == site)
        pca = pca_coordinates(block, int(cfg.geometry.pca_components), seed)
        labels = block["shock_type"].cast(pl.String).to_numpy()
        coded = np.isin(labels, mechanisms)
        if coded.sum() <= max(ks):
            logger.warning(
                f"  [{site}] only {int(coded.sum())} coded patients, fewer than "
                f"k={max(ks)}; neighbour composition skipped for this site"
            )
            continue

        # +1 so a coded patient's own self-match can be dropped without
        # shortening their neighbour list relative to an uncoded patient's
        index = NearestNeighbors(n_neighbors=max(ks) + 1).fit(pca[coded])
        _, neighbours = index.kneighbors(pca)
        coded_labels = labels[coded]
        coded_positions = np.flatnonzero(coded)

        # drop the self-match row-wise: for a coded patient the first
        # neighbour is themselves, for an uncoded patient it is a real
        # neighbour and must be kept
        is_self = neighbours == np.searchsorted(
            coded_positions, np.arange(block.height)
        )[:, None]
        is_self &= coded[:, None]
        keep = np.where(
            is_self.any(axis=1)[:, None],
            np.arange(1, max(ks) + 1)[None, :],
            np.arange(0, max(ks))[None, :],
        )
        neighbours = np.take_along_axis(neighbours, keep, axis=1)

        columns: dict[str, pl.Series] = {}
        for k in ks:
            window = coded_labels[neighbours[:, :k]]
            for mechanism in mechanisms:
                columns[f"knn{k}_{mechanism}"] = pl.Series(
                    (window == mechanism).mean(axis=1)
                )
            shares = np.vstack(
                [(window == m).mean(axis=1) for m in mechanisms]
            )
            columns[f"knn{k}_majority"] = pl.Series(
                [mechanisms[i] for i in shares.argmax(axis=0)]
            )
            columns[f"knn{k}_majority_share"] = pl.Series(shares.max(axis=0))

        parts.append(
            block.select("hospitalization_id", "site", "shock_type").with_columns(
                **columns
            )
        )

    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def reference_separation(proximity: pl.DataFrame, logger) -> pl.DataFrame:
    """
    what "close to a centroid" means, measured on patients whose label is known

    The benchmark the spec asks for. Without it, a mean cosine distance of 0.31
    from the undifferentiated to the septic centroid is a number with no scale:
    coded septic patients might sit at 0.30 and coded cardiogenic ones at 0.32,
    in which case the mechanisms are not separated at all and nothing can be
    read from any patient's distance.

    One row per (site, label, centroid): the mean distance from that label's
    patients to that centroid, with the diagonal -- own label to own centroid --
    computed leave-one-out upstream. The contrast worth reading is the gap
    between a row's diagonal entry and its off-diagonal ones.
    """
    rows = []
    for site in proximity["site"].unique(maintain_order=True).to_list():
        block = proximity.filter(pl.col("site") == site)
        for label in GROUPS:
            members = block.filter(pl.col("shock_type").cast(pl.String) == label)
            if not members.height:
                continue
            for mechanism in MECHANISMS:
                column = f"cos_{mechanism}"
                if column not in members.columns:
                    continue
                rows.append(
                    {
                        "site": site,
                        "label": label,
                        "centroid": mechanism,
                        "n": members.height,
                        "mean_cosine": members[column].mean(),
                        "median_cosine": members[column].median(),
                        "is_own_centroid": label == mechanism,
                    }
                )
    table = pl.DataFrame(rows)
    if table.height:
        own = table.filter(pl.col("is_own_centroid"))["mean_cosine"].mean()
        other = table.filter(
            pl.col("label").is_in(list(MECHANISMS)) & ~pl.col("is_own_centroid")
        )["mean_cosine"].mean()
        logger.info(
            f"  coded patients: mean cosine {own:.4f} to their own centroid "
            f"against {other:.4f} to the others -- a gap of {other - own:+.4f}"
        )
    return table


# ================================================================= the probe


def _normalised_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row, scaled so a uniform row is 1"""
    safe = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=1)
    return entropy / np.log(probabilities.shape[1])


def _calibration(truth: np.ndarray, predicted: np.ndarray, bins: int = 10):
    """observed against predicted in equal-width bins, plus the Brier score"""
    edges = np.linspace(0.0, 1.0, bins + 1)
    assigned = np.clip(np.digitize(predicted, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        members = assigned == b
        if not members.any():
            continue
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": int(members.sum()),
                "mean_predicted": float(predicted[members].mean()),
                "observed": float(truth[members].mean()),
            }
        )
    return rows, float(brier_score_loss(truth, predicted))


def fit_probe(frame: pl.DataFrame, cfg, seed: int, logger) -> dict:
    """
    a linear probe for the coded mechanism, scored within and across sites

    Multinomial logistic regression on the 50-d PCA scores, cross-validated
    with patients -- not hospitalizations -- as the grouping unit, so the same
    person cannot appear on both sides of a fold boundary.

    Returns a dict of frames: metrics (AUROC and Brier per class and scoring
    mode), calibration (reliability bins), and predictions (per patient, for
    every patient in the frame, including the ones never trained on).

    WHY THE CROSS-SITE NUMBER IS THE ONE THAT MATTERS. Coding practice differs
    between the sites far more than case mix plausibly does, so a probe trained
    and tested inside one hospital can reach a high AUROC by learning that
    hospital's documentation habits. The cross-site rows -- train on one, test
    on the other -- cannot be satisfied that way, and they are reported beside
    the within-site ones rather than instead of them, so the size of the drop
    is visible.

    WHAT A HIGH AUROC WOULD AND WOULD NOT MEAN. The labels are ICD codes
    assigned over a whole admission, and the embeddings at the onset anchor
    read only what happened before the shock. A probe that separates them is
    finding mechanism-relevant signal in the pre-shock record. A probe scored
    on a post-onset snapshot may instead be reading the treatment the mechanism
    provoked, which is what snapshot_separation measures.
    """
    spec = cfg.probe
    classes = list(spec.classes)
    group_column = str(spec.group_column)

    labels = frame["shock_type"].cast(pl.String).to_numpy()
    trainable = np.isin(labels, classes)
    if trainable.sum() < len(classes) * int(spec.folds):
        logger.warning("  too few coded patients to fit the probe; skipped")
        return {}

    coordinates = pca_coordinates(frame, int(cfg.geometry.pca_components), seed)
    sites = frame["site"].cast(pl.String).to_numpy()
    groups = frame[group_column].cast(pl.String).to_numpy()

    def _model():
        return LogisticRegression(
            C=float(spec.penalty_C),
            max_iter=int(spec.max_iter),
            random_state=seed,
        )

    metrics, calibration_rows = [], []
    held_out = np.full((frame.height, len(classes)), np.nan)

    # -------------------------------------------------------- within site
    for site in sorted(set(sites)):
        members = trainable & (sites == site)
        y = labels[members]
        if len(set(y)) < len(classes):
            logger.warning(f"  [{site}] not all classes present; within-site skipped")
            continue
        x = coordinates[members]
        folds = StratifiedGroupKFold(
            n_splits=int(spec.folds), shuffle=True, random_state=seed
        )
        predicted = np.zeros((int(members.sum()), len(classes)))
        for train, test in folds.split(x, y, groups=groups[members]):
            model = _model().fit(x[train], y[train])
            order = [list(model.classes_).index(c) for c in classes]
            predicted[test] = model.predict_proba(x[test])[:, order]
        held_out[np.flatnonzero(members)] = predicted

        for i, klass in enumerate(classes):
            truth = (y == klass).astype(int)
            if truth.sum() == 0 or truth.sum() == len(truth):
                continue
            bins, brier = _calibration(truth, predicted[:, i])
            metrics.append(
                {
                    "scoring": "within-site",
                    "trained_on": site,
                    "tested_on": site,
                    "class": klass,
                    "n": int(members.sum()),
                    "n_positive": int(truth.sum()),
                    "auroc": float(roc_auc_score(truth, predicted[:, i])),
                    "brier": brier,
                }
            )
            for row in bins:
                calibration_rows.append(
                    {
                        "scoring": "within-site",
                        "tested_on": site,
                        "class": klass,
                        **row,
                    }
                )

    # --------------------------------------------------------- cross site
    for train_site in sorted(set(sites)):
        for test_site in sorted(set(sites)):
            if train_site == test_site:
                continue
            train = trainable & (sites == train_site)
            test = trainable & (sites == test_site)
            if len(set(labels[train])) < len(classes) or not test.sum():
                continue
            model = _model().fit(coordinates[train], labels[train])
            order = [list(model.classes_).index(c) for c in classes]
            predicted = model.predict_proba(coordinates[test])[:, order]
            for i, klass in enumerate(classes):
                truth = (labels[test] == klass).astype(int)
                if truth.sum() == 0 or truth.sum() == len(truth):
                    continue
                bins, brier = _calibration(truth, predicted[:, i])
                metrics.append(
                    {
                        "scoring": "cross-site",
                        "trained_on": train_site,
                        "tested_on": test_site,
                        "class": klass,
                        "n": int(test.sum()),
                        "n_positive": int(truth.sum()),
                        "auroc": float(roc_auc_score(truth, predicted[:, i])),
                        "brier": brier,
                    }
                )
                for row in bins:
                    calibration_rows.append(
                        {
                            "scoring": "cross-site",
                            "tested_on": test_site,
                            "class": klass,
                            **row,
                        }
                    )

    # ----------------------------- the assignment, for everyone not trained on
    #
    # Fitted on every coded patient at that patient's own site. The uncoded are
    # the target; mixed and other shock are scored too, as the validation the
    # config describes -- if the entropy means anything, mixed should sit
    # higher on it than a patient the coders called one thing.
    assignment = np.full((frame.height, len(classes)), np.nan)
    for site in sorted(set(sites)):
        train = trainable & (sites == site)
        target = ~trainable & (sites == site)
        if len(set(labels[train])) < len(classes) or not target.sum():
            continue
        model = _model().fit(coordinates[train], labels[train])
        order = [list(model.classes_).index(c) for c in classes]
        assignment[np.flatnonzero(target)] = model.predict_proba(
            coordinates[target]
        )[:, order]

    # A coded patient's own probabilities are their cross-validated ones, so
    # the entropy column means the same thing in every group rather than being
    # optimistic for the patients the probe was fitted on.
    combined = np.where(np.isnan(assignment), held_out, assignment)
    entropy = np.full(frame.height, np.nan)
    usable = ~np.isnan(combined).any(axis=1)
    entropy[usable] = _normalised_entropy(combined[usable])
    assigned = np.full(frame.height, None, dtype=object)
    assigned[usable] = [classes[i] for i in combined[usable].argmax(axis=1)]

    predictions = frame.select(
        "hospitalization_id", "site", "shock_type"
    ).with_columns(
        **{f"p_{c}": pl.Series(combined[:, i]) for i, c in enumerate(classes)},
        probe_assignment=pl.Series(assigned, dtype=pl.String),
        probe_entropy=pl.Series(entropy),
        probe_confident=pl.Series(entropy) <= float(spec.max_entropy_confident),
        probe_source=pl.Series(
            np.where(
                np.isnan(assignment).any(axis=1), "cross-validated", "applied"
            )
        ),
    )

    metrics_frame = pl.DataFrame(metrics) if metrics else pl.DataFrame()
    if metrics_frame.height:
        for scoring in ("within-site", "cross-site"):
            subset = metrics_frame.filter(pl.col("scoring") == scoring)
            if subset.height:
                summary = ", ".join(
                    f"{row['class']} {row['auroc']:.3f}"
                    for row in subset.sort("class").iter_rows(named=True)
                )
                logger.info(f"  {scoring}: AUROC {summary}")
    return {
        "metrics": metrics_frame,
        "calibration": pl.DataFrame(calibration_rows)
        if calibration_rows
        else pl.DataFrame(),
        "predictions": predictions,
    }


# ====================================================== the leakage check


def load_snapshot(paths, offset_hours: float, logger) -> pl.DataFrame:
    """
    09's grid embeddings for one offset from shock onset, pooled across sites

    09 embeds every patient repeatedly on a grid around onset, writing one
    directory per (site, offset). Those files are what makes the leakage check
    possible without re-embedding: the +6 h snapshot is the same patient, the
    same model and the same anchor, read six hours later.

    Returns an empty frame when the offset has not been extracted, so a caller
    can degrade to the offsets that exist rather than failing.
    """
    tag = f"{'m' if offset_hours < 0 else 'p'}{abs(int(round(offset_hours))):03d}"
    parts = []
    for site in paths.sites:
        source = (
            paths.scratch_home
            / "trajectory"
            / paths.model_key
            / site
            / tag
            / "embeddings.parquet"
        )
        if not source.is_file():
            logger.warning(f"  [t={offset_hours:+.0f}h] {site}: no grid embeddings")
            continue
        parts.append(pl.read_parquet(source))
    if not parts:
        return pl.DataFrame()
    frame = pl.concat(parts, how="diagonal_relaxed")
    logger.info(f"  [t={offset_hours:+.0f}h] {frame.height:,} patients")
    return frame


def snapshot_separation(paths, cohort: pl.DataFrame, cfg, seed: int, logger):
    """
    how much mechanism separation grows after the clinician starts treating

    The medication vocabulary is part of what the model reads, so an embedding
    taken at +24 h has seen the antibiotics, the inotrope and the transfusion
    that the working diagnosis produced. A probe scored there is partly reading
    the treatment back out, and reporting only that number would pass off
    documented clinical judgement as latent structure.

    The same probe, the same classes and the same folds are therefore run at
    each offset in the config, and the three AUROCs are reported together. The
    rise from 0 h is the quantity of interest: at 0 h the model has read
    nothing after the shock began, so that row is the floor, and anything above
    it is attributable to what happened next -- which includes the patient
    deteriorating as well as the clinician acting, and cannot be split between
    the two here.

    Returns (separation, assignments): per-offset probe metrics, and the
    per-patient assignment at each offset so an undifferentiated patient can be
    followed across the three.
    """
    spec = cfg.snapshots
    offsets = [float(o) for o in spec.offsets_hours]
    keys = cohort.select(
        "hospitalization_id", "site", "shock_type", "patient_id"
    )

    separation, assignments = [], []
    for offset in offsets:
        snapshot = load_snapshot(paths, offset, logger)
        if not snapshot.height:
            continue
        frame = snapshot.drop(
            [c for c in ("site", "offset_hours") if c in snapshot.columns]
        ).join(keys, on="hospitalization_id", how="inner")
        if frame.height < 100:
            logger.warning(
                f"  [t={offset:+.0f}h] only {frame.height} patients join the "
                "cohort; offset skipped"
            )
            continue
        result = fit_probe(frame, cfg, seed, logger)
        if not result:
            continue
        separation.append(
            result["metrics"].with_columns(offset_hours=pl.lit(offset))
        )
        assignments.append(
            result["predictions"].with_columns(offset_hours=pl.lit(offset))
        )

    if not separation:
        return pl.DataFrame(), pl.DataFrame()
    return (
        pl.concat(separation, how="diagonal_relaxed"),
        pl.concat(assignments, how="diagonal_relaxed"),
    )


# ================================================================ treatments

#: the comparisons the spec asks for that no table at either site supports.
#: Carried as data so the summary and the report can state them rather than
#: leaving a reader to notice the columns are missing.
NOT_ASSESSABLE: tuple[tuple[str, str], ...] = (
    (
        "fluid volume in the first 6/24 h",
        "no intake/output table in either extract. Counting crystalloid "
        "administration events was considered and rejected: an event without a "
        "volume cannot separate a 30 mL/kg resuscitation from a carrier line.",
    ),
    (
        "transfusion",
        "no blood product table at either site.",
    ),
    (
        "echocardiography or POCUS",
        "no echo table, and ejection fraction appears in no vitals, labs or "
        "assessment category. Whether a study was done is as unavailable as "
        "what it showed, so whether the team looked for a mechanism at all "
        "cannot be answered. This is the serious omission: echo is the main act "
        "of differentiating shock, so what remains here is only the drugs that "
        "follow it.",
    ),
)


def first_agent_and_antibiotic(data_dir, cohort: pl.DataFrame, cfg, logger):
    """
    which vasoactive came first, and when antibiotics started, relative to onset

    `clifcs.trials.attach_features` already gives the peri-onset and
    admission-wide agent flags. What it does not give is order, which is the
    part of the spec about sequence: a patient started on norepinephrine and
    then given dobutamine was being treated for one thing and then another.

    Returns one row per hospitalization with `first_agent`,
    `hours_to_first_agent` and `hours_to_first_antibiotic`, each null where the
    event never happened inside the window.

    Antibiotic timing can be negative and usually is. Antibiotics given before
    the shock are not an error in the data -- they are the sepsis pathway
    working, and a positive-only reading would discard the patients it most
    applies to.
    """
    spec = cfg.treatments
    window = int(spec.sequence_window_hours)
    ids = cohort.select("hospitalization_id", "cs_onset_dttm")

    empty = cohort.select("hospitalization_id").with_columns(
        first_agent=pl.lit(None, dtype=pl.String),
        hours_to_first_agent=pl.lit(None, dtype=pl.Float64),
        hours_to_first_antibiotic=pl.lit(None, dtype=pl.Float64),
    )

    # ------------------------------------------------- the first vasoactive
    if has_table(data_dir, "clif_medication_admin_continuous"):
        agents = [str(a) for a in spec.agents]
        continuous = (
            scan(
                data_dir,
                "clif_medication_admin_continuous",
                dttm_cols=("admin_dttm",),
            )
            .select("hospitalization_id", "admin_dttm", "med_category")
            .filter(pl.col("med_category").is_in(agents))
            .join(ids.lazy(), on="hospitalization_id", how="inner")
            .with_columns(
                hours=(
                    pl.col("admin_dttm") - pl.col("cs_onset_dttm")
                ).dt.total_minutes()
                / 60.0
            )
            .filter(pl.col("hours").abs() <= window)
            .sort("hospitalization_id", "hours")
            .group_by("hospitalization_id")
            .agg(
                first_agent=pl.col("med_category").first(),
                hours_to_first_agent=pl.col("hours").first(),
            )
            .collect()
        )
        empty = (
            empty.drop("first_agent", "hours_to_first_agent")
            .join(continuous, on="hospitalization_id", how="left")
        )
    else:
        logger.warning("  no clif_medication_admin_continuous; no agent sequence")

    # ---------------------------------------------------------- antibiotics
    if has_table(data_dir, "clif_medication_admin_intermittent"):
        group = str(spec.antibiotic_med_group)
        antibiotics = (
            scan(
                data_dir,
                "clif_medication_admin_intermittent",
                dttm_cols=("admin_dttm",),
            )
            .select("hospitalization_id", "admin_dttm", "med_group")
            .filter(pl.col("med_group") == group)
            .join(ids.lazy(), on="hospitalization_id", how="inner")
            .with_columns(
                hours_to_first_antibiotic=(
                    pl.col("admin_dttm") - pl.col("cs_onset_dttm")
                ).dt.total_minutes()
                / 60.0
            )
            .group_by("hospitalization_id")
            .agg(pl.col("hours_to_first_antibiotic").min())
            .collect()
        )
        empty = empty.drop("hours_to_first_antibiotic").join(
            antibiotics, on="hospitalization_id", how="left"
        )
    else:
        logger.warning("  no clif_medication_admin_intermittent; no antibiotic timing")

    return empty


def treatment_profile(frame: pl.DataFrame, cfg, logger) -> pl.DataFrame:
    """
    what each group was actually given, per site

    One row per (site, shock_type). Agent columns are the peri-onset flags from
    `clifcs.trials.attach_features`, which read the medication table over onset
    plus or minus six hours; the admission-wide `ever_` flags are reported in
    their own columns rather than mixed in, because 11 found that the two
    windows give opposite answers and treating them as interchangeable produced
    a published-comparison error that had to be retracted.
    """
    agents = [str(a) for a in cfg.treatments.agents]
    aggregations = [pl.len().alias("n")]
    for agent in agents:
        if f"on_{agent}" in frame.columns:
            aggregations.append(pl.col(f"on_{agent}").mean().alias(f"on_{agent}"))
        if f"ever_{agent}" in frame.columns:
            aggregations.append(
                pl.col(f"ever_{agent}").mean().alias(f"ever_{agent}")
            )
    for column, alias in (
        ("on_any_steroid", "on_any_steroid"),
        ("cardiac_mcs", "mcs"),
        ("cardiac_inotrope", "inotrope"),
    ):
        if column in frame.columns:
            aggregations.append(pl.col(column).mean().alias(alias))
    for column in ("hours_to_first_antibiotic", "hours_to_first_agent"):
        if column in frame.columns:
            aggregations += [
                pl.col(column).median().alias(f"median_{column}"),
                pl.col(column).is_not_null().mean().alias(f"pct_with_{column}"),
            ]
    if "hours_to_first_antibiotic" in frame.columns:
        aggregations.append(
            (pl.col("hours_to_first_antibiotic") <= 6)
            .fill_null(False)
            .mean()
            .alias("abx_within_6h")
        )

    table = (
        frame.group_by("site", "shock_type").agg(aggregations).sort("site", "shock_type")
    )
    logger.info(f"  treatment profile: {table.height} site-group rows")
    return table


def first_agent_mix(frame: pl.DataFrame, logger) -> pl.DataFrame:
    """which agent came first, as a share within each (site, group)"""
    if "first_agent" not in frame.columns:
        return pl.DataFrame()
    return (
        frame.filter(pl.col("first_agent").is_not_null())
        .group_by("site", "shock_type", "first_agent")
        .agg(pl.len().alias("n"))
        .with_columns(
            share=pl.col("n") / pl.col("n").sum().over("site", "shock_type")
        )
        .sort("site", "shock_type", "n", descending=[False, False, True])
    )


#: Treatment signatures for the probe's three classes, and whether each is
#: ascertainable here. Two of the three are; the third is the reason the
#: concordance analysis is reported for a subset rather than for everyone.
TREATMENT_SIGNATURE: dict[str, dict] = {
    "septic": {
        "ascertainable": True,
        "means": "antibiotics started within 6 h of shock onset",
    },
    "cardiogenic": {
        "ascertainable": True,
        "means": "an inotrope running at onset, or mechanical circulatory support",
    },
    "hypovolemic": {
        "ascertainable": False,
        "means": (
            "volume resuscitation or transfusion, neither of which exists in "
            "either extract -- so a patient the probe calls hypovolemic cannot "
            "be scored as treated or untreated and is excluded from this table "
            "rather than counted as discordant"
        ),
    },
}


def assignment_concordance(frame: pl.DataFrame, logger) -> pl.DataFrame:
    """
    did the treatment match what the embedding says the mechanism was

    Only for the undifferentiated, and only for the two classes with an
    ascertainable signature. A patient the probe calls hypovolemic has no
    column that says whether they were volume-resuscitated, and scoring them as
    discordant because the data cannot see the treatment would manufacture the
    association this table exists to test.

    CONFOUNDING BY INDICATION IS THE WHOLE STORY HERE AND IS NOT ADJUSTABLE.
    A patient receives antibiotics because a clinician thought they were
    infected. The probe is reading the pre-onset record for the same
    information the clinician had. So concordance measures agreement between
    two readings of one patient, and outcomes differ between the groups partly
    because the clinician was right and partly because a clinician who reaches
    a confident diagnosis is looking at a different patient from one who does
    not. Nothing below separates those.
    """
    ascertainable = [
        k for k, v in TREATMENT_SIGNATURE.items() if v["ascertainable"]
    ]
    required = {
        "septic": "abx_within_6h_flag",
        "cardiogenic": "cardiogenic_treatment",
    }
    missing = [c for c in required.values() if c not in frame.columns]
    if missing:
        logger.warning(f"  concordance needs {missing}; table skipped")
        return pl.DataFrame()

    scored = frame.filter(
        (pl.col("shock_type").cast(pl.String) == UNDIFFERENTIATED)
        & pl.col("probe_assignment").is_in(ascertainable)
    ).with_columns(
        treated_as_assigned=pl.when(pl.col("probe_assignment") == "septic")
        .then(pl.col("abx_within_6h_flag"))
        .when(pl.col("probe_assignment") == "cardiogenic")
        .then(pl.col("cardiogenic_treatment"))
        .otherwise(None)
    )
    if not scored.height:
        return pl.DataFrame()

    outcomes = [
        pl.len().alias("n"),
        pl.col("in_hospital_mortality").mean().alias("mortality"),
    ]
    for column in ("pressor_free_days", "vent_free_days"):
        if column in frame.columns:
            outcomes.append(pl.col(column).mean().alias(column))
    if "sofa_non_cv" in frame.columns:
        outcomes.append(pl.col("sofa_non_cv").mean().alias("sofa_non_cv"))
    elif "sofa_total" in frame.columns:
        outcomes.append(pl.col("sofa_total").mean().alias("sofa_total"))

    return (
        scored.group_by(
            "site", "probe_assignment", "probe_confident", "treated_as_assigned"
        )
        .agg(outcomes)
        .sort("site", "probe_assignment", "probe_confident", "treated_as_assigned")
    )


def concordance_adjusted(frame: pl.DataFrame, logger) -> pl.DataFrame:
    """
    concordant against discordant treatment, with severity in the model

    The unadjusted contrast in `assignment_concordance` is dominated by who
    gets treated: a patient on antibiotics within six hours was recognised as
    infected, and recognition tracks how obviously sick they looked. Severity
    therefore goes in the model -- non-cardiovascular SOFA, age and site -- and
    the adjusted estimate is reported beside the raw one.

    It does not fix the problem. Adjusting for severity removes the part of the
    confounding that severity carries, and leaves the part carried by the
    clinician's judgement, which is not measured anywhere. The `confounding`
    column says this on every row rather than leaving it to a footnote.

    One row per (probe class, outcome): the treated-minus-untreated difference
    before and after adjustment.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("  statsmodels not installed; adjusted estimates skipped")
        return pl.DataFrame()

    severity = next(
        (c for c in ("sofa_non_cv", "sofa_total") if c in frame.columns), None
    )
    if severity is None or "treated_as_assigned" not in frame.columns:
        logger.warning("  no severity column; adjusted estimates skipped")
        return pl.DataFrame()

    rows = []
    outcomes = [
        ("in_hospital_mortality", "logistic"),
        ("pressor_free_days", "ols"),
        ("vent_free_days", "ols"),
    ]
    for klass in frame["probe_assignment"].unique().drop_nulls().to_list():
        block = frame.filter(pl.col("probe_assignment") == klass)
        for outcome, kind in outcomes:
            if outcome not in block.columns:
                continue
            usable = block.select(
                "treated_as_assigned", outcome, severity, "age_at_admission", "site"
            ).drop_nulls()
            if usable.height < 100 or usable["treated_as_assigned"].n_unique() < 2:
                continue
            treated = usable["treated_as_assigned"].to_numpy().astype(float)
            y = usable[outcome].to_numpy().astype(float)
            design = np.column_stack(
                [
                    np.ones(usable.height),
                    treated,
                    usable[severity].to_numpy().astype(float),
                    usable["age_at_admission"].to_numpy().astype(float),
                    (usable["site"].to_numpy() == usable["site"][0]).astype(float),
                ]
            )
            raw = np.column_stack([np.ones(usable.height), treated])
            try:
                if kind == "logistic":
                    unadjusted = sm.Logit(y, raw).fit(disp=0)
                    adjusted = sm.Logit(y, design).fit(disp=0)
                else:
                    unadjusted = sm.OLS(y, raw).fit()
                    adjusted = sm.OLS(y, design).fit()
            except Exception as error:  # separation, singular design
                logger.warning(f"  {klass}/{outcome}: {type(error).__name__}; skipped")
                continue
            rows.append(
                {
                    "probe_assignment": klass,
                    "outcome": outcome,
                    "model": kind,
                    "n": usable.height,
                    "n_treated": int(treated.sum()),
                    "effect_unadjusted": float(unadjusted.params[1]),
                    "p_unadjusted": float(unadjusted.pvalues[1]),
                    "effect_adjusted": float(adjusted.params[1]),
                    "p_adjusted": float(adjusted.pvalues[1]),
                    "adjusted_for": f"{severity}, age, site",
                    "confounding": (
                        "confounding by indication: treatment follows the "
                        "clinician's reading of the same record the probe reads, "
                        "and the judgement itself is unmeasured"
                    ),
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()

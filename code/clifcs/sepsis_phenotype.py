#!/usr/bin/env python3

"""
sepsis phenotypes: Seymour's four clinical classes and Bhavani's four
vital-sign trajectories

Seymour CW, Kennedy JN, Wang S, et al. "Derivation, Validation, and Potential
Treatment Implications of Novel Clinical Phenotypes for Sepsis."
JAMA 2019;321(20):2003-2017. doi:10.1001/jama.2019.5791

Bhavani SV, Semler M, Qian ET, et al. "Development and validation of novel
sepsis subphenotypes using trajectories of vital signs."
Intensive Care Med 2022;48(11):1582-1592. doi:10.1007/s00134-022-06890-z

Literature values live in config/phenotypes_sepsis.yaml; this module holds the
assignment logic.

Read this before reading the numbers
------------------------------------
Both schemes were derived in sepsis cohorts, and this pipeline has no sepsis
definition. The denominator is the *coded* septic-shock flag, which is a
surrogate for Sepsis-3 and a weak one -- and it is site-dependent, since septic
shock is coded on 20.3% of NU's cohort and 12.3% of UCMC's. Phenotype shares
are therefore reported **per site and never pooled into one percentage**, and
they are not comparable to the published distributions.

Both are refits rather than transfers, for different reasons:

* Seymour publishes per-phenotype *means* (Table 2) but not the cluster
  centroids, not the standard deviations needed to put those means on a common
  scale, and no classifier. A transfer is computed anyway, from Table 2 against
  our own dispersion, and reported as a cross-check -- not as the answer.
* Bhavani publishes no model parameters at all, only the four qualitative
  shapes. There is nothing to transfer.

In both cases clusters are labelled **deterministically from their centroid
profiles**, never by cluster index, which k-means assigns arbitrarily; and in
both cases the recovered labelling is checked against the paper's published
mortality ordering and group sizes, which is the only evidence on offer that
the clusters correspond to the published ones at all.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.cluster import KMeans

from clifcs.clif import has_table, scan

SEYMOUR_LABELS: tuple[str, ...] = ("alpha", "beta", "gamma", "delta")
BHAVANI_LABELS: tuple[str, ...] = ("A", "B", "C", "D")

#: which CLIF table each `source` in the config reads, and the columns it uses
_SOURCES = {
    "labs": ("clif_labs", "lab_category", "lab_value_numeric", "lab_result_dttm"),
    "vitals": ("clif_vitals", "vital_category", "vital_value", "recorded_dttm"),
    "assessment": (
        "clif_patient_assessments",
        "assessment_category",
        "numerical_value",
        "recorded_dttm",
    ),
}

#: physiologically impossible readings that a colour scale or a cluster centroid
#: cannot survive. Mirrors clifcs.state._VITALS -- MAP reached 1,040 mmHg in the
#: raw table before that filter existed.
_PLAUSIBLE = {
    "heart_rate": (10.0, 300.0),
    "respiratory_rate": (2.0, 80.0),
    "sbp": (20.0, 300.0),
    "dbp": (10.0, 200.0),
    "temp_c": (25.0, 43.0),
    "spo2": (30.0, 100.0),
    "gcs_total": (3.0, 15.0),
    "lactate": (0.1, 30.0),
    "creatinine": (0.1, 25.0),
    "bicarbonate": (2.0, 60.0),
    "bun": (1.0, 250.0),
    "platelet_count": (1.0, 2000.0),
    "wbc": (0.1, 200.0),
    "alt": (1.0, 10000.0),
    "ast": (1.0, 10000.0),
    "bilirubin_total": (0.1, 50.0),
    "albumin": (0.5, 7.0),
    "hemoglobin": (2.0, 25.0),
    "sodium": (100.0, 180.0),
    "chloride": (60.0, 140.0),
    "glucose_serum": (10.0, 1500.0),
    "inr": (0.5, 20.0),
    "crp": (0.0, 500.0),
    "esr": (0.0, 150.0),
    "po2_arterial": (20.0, 600.0),
    "age_at_admission": (18.0, 120.0),
}


def usable_variables(cfg) -> tuple[list[dict], list[dict]]:
    """split the configured Seymour variables into usable and not"""
    entries = [dict(v) for v in cfg.seymour.variables]
    usable = [e for e in entries if e.get("available", True)]
    unusable = [e for e in entries if not e.get("available", True)]
    return usable, unusable


def mark_sepsis_cohort(cohort: pl.DataFrame, cfg, logger) -> pl.DataFrame:
    """
    flag the denominator the two sepsis phenotypes are applied to

    Reuses the coded septic flag `clifcs.shock_type` already attaches, rather
    than re-reading the diagnosis table: the code set is the same one and
    duplicating it would let the two drift apart.
    """
    if "shock_septic" not in cohort.columns:
        raise KeyError(
            "sepsis_cohort needs `shock_septic` from clifcs.shock_type; run the "
            "cohort stage first"
        )
    out = cohort.with_columns(sepsis_cohort=pl.col("shock_septic"))

    for site, part in out.group_by("site"):
        site = site[0] if isinstance(site, tuple) else site
        n = int(part["sepsis_cohort"].sum())
        logger.info(
            f"[{site}] sepsis denominator (coded septic shock): {n} of "
            f"{part.height} ({n / max(part.height, 1):.1%})"
        )

    shares = {
        (s[0] if isinstance(s, tuple) else s): float(p["sepsis_cohort"].mean() or 0.0)
        for s, p in out.group_by("site")
    }
    if len(shares) > 1:
        lo, hi = min(shares.values()), max(shares.values())
        if lo > 0 and hi / lo > 1.5:
            logger.warning(
                f"the coded sepsis denominator differs {hi / lo:.1f}x between "
                f"sites ({', '.join(f'{s} {v:.1%}' for s, v in sorted(shares.items()))})"
                " -- phenotype shares are not comparable across sites and must "
                "not be pooled into a single percentage"
            )
    return out


# ============================================================ feature assembly


def _nearest_values(
    data_dir,
    anchors: pl.DataFrame,
    wanted: dict[str, str],
    source: str,
    lookback_hours: float,
    lookahead_hours: float,
) -> pl.DataFrame:
    """
    for each variable, the single value measured nearest the anchor in window

    `wanted` maps our variable name to the CLIF category that supplies it.
    `anchors` carries hospitalization_id and anchor_dttm.
    """
    table, category_col, value_col, time_col = _SOURCES[source]
    keys = anchors.select("hospitalization_id")
    if not wanted or not has_table(data_dir, table):
        return keys.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(n) for n in wanted]
        )

    window = anchors.lazy().select(
        "hospitalization_id",
        "anchor_dttm",
        win_start=pl.col("anchor_dttm") - pl.duration(hours=lookback_hours),
        win_end=pl.col("anchor_dttm") + pl.duration(hours=lookahead_hours),
    )
    by_category = {category: name for name, category in wanted.items()}

    long = (
        scan(data_dir, table, dttm_cols=(time_col,))
        .filter(
            pl.col(category_col).is_in(list(by_category)),
            pl.col(value_col).is_not_null(),
        )
        .join(window, on="hospitalization_id", how="inner")
        .filter(pl.col(time_col).is_between(pl.col("win_start"), pl.col("win_end")))
        .with_columns(
            gap_s=(pl.col(time_col) - pl.col("anchor_dttm")).dt.total_seconds().abs()
        )
        # nearest in time, ties broken toward the earlier reading so the answer
        # does not depend on row order
        .sort(["hospitalization_id", category_col, "gap_s", time_col])
        .unique(
            subset=["hospitalization_id", category_col],
            keep="first",
            maintain_order=True,
        )
        .select("hospitalization_id", category_col, pl.col(value_col).alias("value"))
        .collect()
    )
    if long.height == 0:
        return keys.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(n) for n in wanted]
        )

    wide = long.pivot(on=category_col, index="hospitalization_id", values="value")
    wide = wide.rename({c: by_category[c] for c in wide.columns if c in by_category})
    for name in wanted:
        if name not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    return keys.join(
        wide.select("hospitalization_id", *wanted), on="hospitalization_id", how="left"
    )


def _clip_implausible(frame: pl.DataFrame, columns) -> pl.DataFrame:
    """null out readings outside the physiologically possible range"""
    exprs = []
    for name in columns:
        bounds = _PLAUSIBLE.get(name)
        if bounds is None:
            continue
        lo, hi = bounds
        exprs.append(
            pl.when(pl.col(name).is_between(lo, hi))
            .then(pl.col(name))
            .otherwise(None)
            .alias(name)
        )
    return frame.with_columns(exprs) if exprs else frame


def seymour_features(
    data_dir, cohort: pl.DataFrame, anchors: pl.DataFrame, cfg, logger
) -> pl.DataFrame:
    """
    the Seymour input vector: one row per patient, one column per usable
    variable, measured nearest the anchor

    Seymour took the first 6 h of hospital presentation. This pipeline anchors
    on shock onset, so the window is taken around the anchor instead -- the
    phenotypes are being applied at a different point in the illness from the
    one they were derived at, which is recorded here and reported by
    `summarize_seymour` rather than left for a reader to infer.
    """
    usable, unusable = usable_variables(cfg)
    if unusable:
        logger.info(
            f"Seymour: {len(usable)} of {len(usable) + len(unusable)} published "
            "variables usable; omitted "
            + ", ".join(f"{e['name']} ({e['why']})" for e in unusable)
        )

    lookback = float(cfg.seymour.window_lookback_hours)
    lookahead = float(cfg.seymour.window_lookahead_hours)

    out = anchors.select("hospitalization_id")
    for source in ("labs", "vitals", "assessment"):
        wanted = {
            e["name"]: e["category"] for e in usable if e["source"] == source
        }
        if not wanted:
            continue
        out = out.join(
            _nearest_values(
                data_dir, anchors, wanted, source, lookback, lookahead
            ),
            on="hospitalization_id",
            how="left",
        )

    # variables that come from the cohort frame rather than a CLIF table
    from_cohort = {e["name"]: e["column"] for e in usable if e["source"] == "cohort"}
    if from_cohort:
        out = out.join(
            cohort.select(
                "hospitalization_id",
                *[
                    pl.col(column).cast(pl.Float64).alias(name)
                    for name, column in from_cohort.items()
                ],
            ),
            on="hospitalization_id",
            how="left",
        )

    names = [e["name"] for e in usable]
    out = _clip_implausible(out, names)
    return out.select("hospitalization_id", *names)


# ====================================================== Seymour: the labelling


#: the variables each phenotype is distinguished by, and the direction of
#: derangement. Taken from the paper's own characterisation: beta is renal with
#: the most chronic illness, gamma is inflammation with pulmonary dysfunction,
#: delta is hepatic dysfunction with shock.
#:
#: `+1` means a higher value is more characteristic of the phenotype.
_SEYMOUR_AXES: dict[str, dict[str, float]] = {
    # delta: AST 118 vs 23-30 is the single most distinctive number in Table 2
    "hepatic": {
        "ast": +1.0,
        "alt": +1.0,
        "bilirubin_total": +1.0,
        "inr": +1.0,
        "lactate": +1.0,
        "bicarbonate": -1.0,
    },
    # beta: creatinine 2.3 and BUN 38 against a 1.4/24 cohort median, oldest,
    # and distinctively *brady*cardic relative to the other three (84 vs 94-109)
    "renal": {
        "creatinine": +1.0,
        "bun": +1.0,
        "age": +1.0,
        "heart_rate": -1.0,
    },
    # gamma: CRP 16 and ESR 92 are its markers, with the lowest oxygenation
    "inflammatory": {
        "crp": +1.0,
        "esr": +1.0,
        "wbc": +1.0,
        "po2_arterial": -1.0,
        "spo2": -1.0,
    },
}


def _log_mask(names: list[str], cfg) -> np.ndarray:
    """which columns are log-transformed, per the config"""
    wanted = set(cfg.seymour.get("log_transform", []) or [])
    return np.array([name in wanted for name in names], dtype=bool)


def _standardize(
    values: np.ndarray,
    log_mask: np.ndarray | None = None,
    z_clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    log-transform the skewed columns, z-score on observed values, impute to the
    centre, and clip what is still extreme

    Median and standard deviation rather than mean and SD: Table 2 reports
    medians for most of these variables precisely because they are skewed, and
    a mean centre would put the centroid off the bulk of the distribution.

    The log transform is applied before centring and is what keeps a handful of
    extreme values from setting the scale for everyone -- see `log_transform` in
    the config for what happened without it.
    """
    values = values.copy()
    if log_mask is not None and log_mask.any():
        # log1p, not log: several of these legitimately reach zero (bands,
        # bilirubin, troponin) and log would send them to -inf, which then
        # propagates into the centre and scale as nan.
        columns = np.where(log_mask)[0]
        values[:, columns] = np.log1p(np.clip(values[:, columns], 0.0, None))

    with np.errstate(invalid="ignore"):
        centre = np.nanmedian(values, axis=0)
        scale = np.nanstd(values, axis=0)
    scale = np.where((scale == 0) | ~np.isfinite(scale), 1.0, scale)
    centre = np.where(np.isfinite(centre), centre, 0.0)

    z = (values - centre) / scale
    z = np.where(np.isfinite(z), z, 0.0)
    if z_clip:
        z = np.clip(z, -float(z_clip), float(z_clip))
    return z, centre, scale


def _axis_score(centroid: np.ndarray, index: dict[str, int], axis: str) -> float:
    """how strongly a centroid expresses one of the phenotype axes"""
    return float(
        sum(
            weight * centroid[index[name]]
            for name, weight in _SEYMOUR_AXES[axis].items()
            if name in index
        )
    )


def availability(features: pl.DataFrame) -> pl.DataFrame:
    """
    per-variable share of patients the variable was actually measured in

    Reported for every variable whether or not it clears the threshold, because
    "measured in 3% of patients" is a finding about this cohort and not only a
    reason to drop a column.
    """
    names = [c for c in features.columns if c != "hospitalization_id"]
    return pl.DataFrame(
        {
            "variable": names,
            "n_measured": [int(features[c].is_not_null().sum()) for c in names],
            "n_total": [features.height] * len(names),
            "availability": [
                float(features[c].is_not_null().mean() or 0.0) for c in names
            ],
        }
    ).sort("availability")


def select_available(
    features: pl.DataFrame, cfg, logger
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    drop clustering variables too sparsely measured to carry information

    Returns the reduced frame and the full availability table. See
    `min_availability` in the config for why this is a drop rather than an
    imputation, and for what it costs the gamma label specifically.
    """
    table = availability(features)
    threshold = float(cfg.seymour.get("min_availability", 0.0) or 0.0)
    dropped = table.filter(pl.col("availability") < threshold)

    if dropped.height:
        logger.warning(
            f"Seymour: dropping {dropped.height} variable(s) measured in fewer "
            f"than {threshold:.0%} of patients -- "
            + ", ".join(
                f"{r['variable']} {r['availability']:.0%}"
                for r in dropped.iter_rows(named=True)
            )
        )
        for marker in ("crp", "esr"):
            if marker in dropped["variable"].to_list():
                logger.warning(
                    f"  {marker} is one of the markers Table 2 separates gamma "
                    "by; with it absent, gamma is recovered from WBC and "
                    "oxygenation alone and is the weakest of the four labels"
                )
    keep = table.filter(pl.col("availability") >= threshold)["variable"].to_list()
    if len(keep) < 4:
        raise ValueError(
            f"only {len(keep)} Seymour variable(s) clear the "
            f"{threshold:.0%} availability threshold; clustering on that would "
            "not reproduce anything"
        )
    return features.select("hospitalization_id", *keep), table


def assign_seymour(
    features: pl.DataFrame, cfg, logger
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """
    refit k-means at k=4 and name the clusters from their centroid profiles

    The order the labels are claimed in matters and is not arbitrary: delta is
    the most distinctive profile in Table 2 and is taken first, then beta on
    renal derangement, then gamma on inflammation, with alpha -- "the fewest
    abnormalities" -- as the remainder. Claiming alpha first would let the
    largest cluster absorb whichever of the others happened to be least
    separated.
    """
    names = [c for c in features.columns if c != "hospitalization_id"]
    values = features.select(names).to_numpy().astype(float)
    log_mask = _log_mask(names, cfg)
    z, centre, scale = _standardize(
        values, log_mask, cfg.seymour.get("z_clip")
    )
    n_imputed = int(np.sum(~np.isfinite(values)))

    km = KMeans(
        n_clusters=int(cfg.seymour.n_clusters),
        n_init=int(cfg.seymour.assignment.n_init),
        random_state=int(cfg.seymour.assignment.random_state),
    ).fit(z)
    centroids = km.cluster_centers_
    index = {name: i for i, name in enumerate(names)}

    order: dict[int, str] = {}
    remaining = list(range(len(centroids)))
    for axis, label in (
        ("hepatic", "delta"),
        ("renal", "beta"),
        ("inflammatory", "gamma"),
    ):
        pick = max(remaining, key=lambda c: _axis_score(centroids[c], index, axis))
        order[pick] = label
        remaining.remove(pick)
    order[remaining[0]] = "alpha"

    labels = [order[c] for c in km.labels_]
    centroid_frame = pl.DataFrame(
        {
            "cluster": list(range(len(centroids))),
            "seymour_phenotype": [order[c] for c in range(len(centroids))],
            "n": [int((km.labels_ == c).sum()) for c in range(len(centroids))],
            **{
                f"{axis}_score": [
                    _axis_score(centroids[c], index, axis)
                    for c in range(len(centroids))
                ]
                for axis in _SEYMOUR_AXES
            },
            **{
                f"z_{name}": [
                    float(centroids[c][index[name]]) for c in range(len(centroids))
                ]
                for name in names
            },
        }
    ).sort("seymour_phenotype")

    logger.info(
        f"Seymour refit on {len(names)} variables, {features.height} patients"
        + (f"; {n_imputed} missing cells imputed to the centre" if n_imputed else "")
    )
    model = {
        "features": names,
        "centre": centre,
        "scale": scale,
        "centroids": centroids,
        "labels": [order[c] for c in range(len(centroids))],
        # the transform has to travel with the moments: a site that z-scored
        # raw AST against a centre computed on log AST would place every
        # patient in the same cluster
        "log_transform": [n for n, m in zip(names, log_mask) if m],
        "z_clip": float(cfg.seymour.get("z_clip") or 0.0) or None,
    }
    return (
        features.select("hospitalization_id").with_columns(
            seymour_phenotype=pl.Series(labels, dtype=pl.Enum(SEYMOUR_LABELS))
        ),
        centroid_frame,
        model,
    )


def transfer_seymour(features: pl.DataFrame, cfg, logger) -> pl.DataFrame:
    """
    assign by nearest published Table 2 profile, as a cross-check on the refit

    **This is an approximation and is not offered as the primary assignment.**
    Putting published values on a common scale needs the derivation cohort's
    dispersion, and the paper publishes only the per-phenotype means and the
    cohort column. Standardising them against *our* spread assumes the two
    cohorts are equally dispersed, which for a sepsis cohort against an
    undifferentiated shock cohort is the assumption most likely to be wrong.

    It is computed anyway because the agreement between it and the refit is
    informative in both directions: high agreement is evidence the refit found
    the published structure, and low agreement says the label is ours rather
    than the paper's.
    """
    published = dict(cfg.seymour.published)
    names = [
        c
        for c in features.columns
        if c != "hospitalization_id" and c in published
    ]
    if not names:
        raise KeyError("no Seymour features have published Table 2 values")

    values = features.select(names).to_numpy().astype(float)
    log_mask = _log_mask(names, cfg)
    z_clip = cfg.seymour.get("z_clip")
    z, centre, scale = _standardize(values, log_mask, z_clip)

    # The published per-phenotype values, put through the *same* transform as
    # our own before being standardised against our moments. Skipping the log
    # here was the first version's bug: a published AST of 118 compared against
    # a centre computed on log AST lands roughly 40 standard deviations out, and
    # every patient is assigned to whichever phenotype happens to be nearest the
    # origin.
    published_matrix = np.array(
        [
            [float(published[name][label]) for name in names]
            for label in SEYMOUR_LABELS
        ]
    )
    columns = np.where(log_mask)[0]
    if columns.size:
        published_matrix[:, columns] = np.log1p(
            np.clip(published_matrix[:, columns], 0.0, None)
        )
    centroids = (published_matrix - centre) / scale
    if z_clip:
        centroids = np.clip(centroids, -float(z_clip), float(z_clip))
    nearest = np.argmin(
        ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1
    )
    logger.info(
        f"Seymour transfer from published Table 2 over {len(names)} variables "
        "(approximate: the paper publishes means but not dispersion)"
    )
    return features.select("hospitalization_id").with_columns(
        seymour_transfer=pl.Series(
            [SEYMOUR_LABELS[i] for i in nearest], dtype=pl.Enum(SEYMOUR_LABELS)
        )
    )


# ===================================================== Bhavani: trajectories


def bhavani_trajectories(
    data_dir, anchors: pl.DataFrame, cfg, logger
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    hourly means of each vital over the window, one column per (vital, hour)

    Returns the trajectory matrix and a per-patient count of how many hourly
    blocks held a complete set, which is what the paper's minimum-of-three rule
    is applied to.

    The paper measures from hospital presentation; this measures from the
    anchor, for the same reason and with the same caveat as the Seymour window.
    """
    bh = cfg.bhavani
    variables = list(bh.variables)
    n_blocks = int(bh.window_hours) // int(bh.block_hours)
    block_hours = int(bh.block_hours)

    window = anchors.lazy().select(
        "hospitalization_id",
        "anchor_dttm",
        win_end=pl.col("anchor_dttm") + pl.duration(hours=int(bh.window_hours)),
    )

    long = (
        scan(data_dir, "clif_vitals", dttm_cols=("recorded_dttm",))
        .filter(
            pl.col("vital_category").is_in(variables),
            pl.col("vital_value").is_not_null(),
        )
        .join(window, on="hospitalization_id", how="inner")
        .filter(
            pl.col("recorded_dttm") >= pl.col("anchor_dttm"),
            pl.col("recorded_dttm") < pl.col("win_end"),
        )
        .with_columns(
            block=(
                (pl.col("recorded_dttm") - pl.col("anchor_dttm")).dt.total_seconds()
                // (3600 * block_hours)
            ).cast(pl.Int32)
        )
        # several readings in one hour are averaged, as the paper specifies
        .group_by("hospitalization_id", "vital_category", "block")
        .agg(value=pl.col("vital_value").mean())
        .collect()
    )

    keys = anchors.select("hospitalization_id")
    columns = [f"{v}_h{b}" for v in variables for b in range(n_blocks)]
    if long.height == 0:
        logger.warning("Bhavani: no vitals in the window at this site")
        empty = keys.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in columns]
        )
        return empty, keys.with_columns(n_complete_blocks=pl.lit(0, dtype=pl.Int32))

    long = long.filter(pl.col("block").is_between(0, n_blocks - 1)).with_columns(
        cell=pl.col("vital_category") + "_h" + pl.col("block").cast(pl.String)
    )
    wide = long.pivot(on="cell", index="hospitalization_id", values="value")
    for c in columns:
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    wide = keys.join(
        wide.select("hospitalization_id", *columns),
        on="hospitalization_id",
        how="left",
    )
    wide = _clip_implausible_cells(wide, variables, n_blocks)

    # Three counts, because the eligibility rule needs all three: blocks where
    # every variable was measured (the paper's literal rule), blocks where the
    # continuously charted ones were, and how many temperatures exist at all.
    core = [v for v in (list(bh.get("core_variables", [])) or variables) if v in variables]

    def blocks_with(subset: list[str], name: str) -> pl.DataFrame:
        return (
            long.filter(pl.col("vital_category").is_in(subset))
            .group_by("hospitalization_id", "block")
            .agg(n_vars=pl.col("vital_category").n_unique())
            .filter(pl.col("n_vars") == len(subset))
            .group_by("hospitalization_id")
            .agg(**{name: pl.len().cast(pl.Int32)})
        )

    counts = keys
    for subset, name in ((variables, "n_complete_blocks"), (core, "n_core_blocks")):
        counts = counts.join(
            blocks_with(list(subset), name), on="hospitalization_id", how="left"
        ).with_columns(pl.col(name).fill_null(0))

    temps = (
        long.filter(pl.col("vital_category") == "temp_c")
        .group_by("hospitalization_id")
        .agg(n_temp_observations=pl.len().cast(pl.Int32))
    )
    counts = counts.join(temps, on="hospitalization_id", how="left").with_columns(
        n_temp_observations=pl.col("n_temp_observations").fill_null(0)
    )
    return wide, counts


def _bhavani_eligible(counts: pl.DataFrame, cfg, logger) -> pl.DataFrame:
    """
    who has enough of a trajectory to be clustered

    Either the paper's literal rule -- three blocks in which every vital was
    measured -- or the split rule the config defaults to; see
    `require_complete_blocks` there for why the literal one is not the default
    and what it costs.
    """
    bh = cfg.bhavani
    if bool(bh.get("require_complete_blocks", False)):
        rule = pl.col("n_complete_blocks") >= int(bh.min_complete_blocks)
        logger.info(
            "Bhavani: applying the published rule literally "
            f"(>= {int(bh.min_complete_blocks)} blocks with every vital measured)"
        )
    else:
        rule = (pl.col("n_core_blocks") >= int(bh.get("min_core_blocks", 3))) & (
            pl.col("n_temp_observations") >= int(bh.get("min_temp_observations", 1))
        )
        logger.info(
            f"Bhavani: >= {int(bh.get('min_core_blocks', 3))} blocks of "
            f"{', '.join(bh.get('core_variables', []))} plus >= "
            f"{int(bh.get('min_temp_observations', 1))} temperature, with "
            "temperature carried forward (a documented departure from the "
            "published rule -- see config)"
        )
        strict = int((counts["n_complete_blocks"] >= int(bh.min_complete_blocks)).sum())
        logger.info(
            f"  the literal rule would admit {strict} of {counts.height} "
            f"({strict / max(counts.height, 1):.0%}); this one admits "
            f"{int(counts.select(rule.sum()).item())}"
        )
    return counts.filter(rule).select("hospitalization_id")


def _clip_implausible_cells(
    frame: pl.DataFrame, variables, n_blocks: int
) -> pl.DataFrame:
    """apply the plausible range to every (variable, hour) cell"""
    exprs = []
    for variable in variables:
        bounds = _PLAUSIBLE.get(variable)
        if bounds is None:
            continue
        lo, hi = bounds
        for block in range(n_blocks):
            cell = f"{variable}_h{block}"
            exprs.append(
                pl.when(pl.col(cell).is_between(lo, hi))
                .then(pl.col(cell))
                .otherwise(None)
                .alias(cell)
            )
    return frame.with_columns(exprs) if exprs else frame


def _fill_trajectories(
    matrix: np.ndarray, n_vars: int, n_blocks: int
) -> np.ndarray:
    """
    carry each vital forward into empty hours, then fall back to the median

    GBTM handles missing hours through the likelihood and needs no filling;
    k-means cannot, so this stands in. Last-observation-carried-forward within
    a variable is how a chart is read, and the column median is the fallback for
    a patient whose first hours hold nothing at all.
    """
    filled = matrix.copy()
    for v in range(n_vars):
        block = filled[:, v * n_blocks : (v + 1) * n_blocks]
        for h in range(1, n_blocks):
            missing = ~np.isfinite(block[:, h])
            block[missing, h] = block[missing, h - 1]
        # backward pass for leading gaps, so an hour-0 null takes hour 1's value
        for h in range(n_blocks - 2, -1, -1):
            missing = ~np.isfinite(block[:, h])
            block[missing, h] = block[missing, h + 1]
        filled[:, v * n_blocks : (v + 1) * n_blocks] = block

    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(filled, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    gaps = ~np.isfinite(filled)
    filled[gaps] = np.take(medians, np.where(gaps)[1])
    return filled


def assign_bhavani(
    trajectories: pl.DataFrame, complete: pl.DataFrame, cfg, logger
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """
    cluster the standardised trajectories and name the groups from their shapes

    The naming follows the paper's own two-way characterisation: temperature,
    heart rate and respiratory rate move together and split the four groups into
    a high pair (A, B) and a low pair (C, D); blood pressure then splits each
    pair, with A and D the hypotensive members. So the rule is "level first,
    pressure second", which is deterministic and does not depend on cluster
    index or on cluster size.
    """
    bh = cfg.bhavani
    variables = list(bh.variables)
    n_blocks = int(bh.window_hours) // int(bh.block_hours)
    columns = [f"{v}_h{b}" for v in variables for b in range(n_blocks)]

    eligible = _bhavani_eligible(complete, cfg, logger)
    usable = trajectories.join(eligible, on="hospitalization_id", how="semi")
    logger.info(
        f"Bhavani: {usable.height} of {trajectories.height} patients "
        f"({usable.height / max(trajectories.height, 1):.0%}) have a usable "
        "trajectory"
    )
    if usable.height < int(bh.n_clusters) * 10:
        logger.warning(
            "Bhavani: too few patients with usable trajectories to cluster; "
            "every patient will be left unassigned"
        )
        return (
            trajectories.select("hospitalization_id").with_columns(
                bhavani_phenotype=pl.lit(None, dtype=pl.Enum(BHAVANI_LABELS))
            ),
            pl.DataFrame(),
            {},
        )

    raw = usable.select(columns).to_numpy().astype(float)
    filled = _fill_trajectories(raw, len(variables), n_blocks)

    # standardised to the cohort's own mean and SD, as the paper specifies.
    # Under federation this is the step that has to use transferred moments
    # rather than each site's own, or group A means something different at each
    # site -- see `apply_trajectory_model`.
    centre = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale = np.where((scale == 0) | ~np.isfinite(scale), 1.0, scale)
    z = (filled - centre) / scale

    km = KMeans(
        n_clusters=int(bh.n_clusters),
        n_init=int(bh.assignment.n_init),
        random_state=int(bh.assignment.random_state),
    ).fit(z)
    centroids = km.cluster_centers_

    # mean standardised level of each variable group across all hours
    def level(centroid: np.ndarray, names: list[str]) -> float:
        slices = [
            centroid[variables.index(n) * n_blocks : (variables.index(n) + 1) * n_blocks]
            for n in names
            if n in variables
        ]
        return float(np.mean([s.mean() for s in slices])) if slices else 0.0

    vital_names = [n for n in ("temp_c", "heart_rate", "respiratory_rate") if n in variables]
    bp_names = [n for n in ("sbp", "dbp") if n in variables]

    vitals_level = {c: level(centroids[c], vital_names) for c in range(len(centroids))}
    bp_level = {c: level(centroids[c], bp_names) for c in range(len(centroids))}

    # Claim the most distinctive group first, as the Seymour labelling does.
    #
    # B is that group: it is the only one of the four the paper describes as
    # *hypertensive*, so it is the only one separated by the sign of a variable
    # rather than by its degree. Splitting into a high-vitals and a low-vitals
    # pair first was tried and is not safe here -- the middle two clusters came
    # out at -0.18 and -0.25 on the vitals axis, a near-tie, so which pair a
    # cluster joined was decided by noise and the pressure split was then
    # applied to the wrong pairing.
    order: dict[int, str] = {}
    remaining = list(range(len(centroids)))

    b = max(remaining, key=lambda c: bp_level[c])
    order[b] = "B"
    remaining.remove(b)

    # A is then the hyperthermic, tachycardic, tachypneic one of what is left
    a = max(remaining, key=lambda c: vitals_level[c])
    order[a] = "A"
    remaining.remove(a)

    # and the final two are both lower-vitals, split by pressure: D is the
    # "most hypotensive" group, C the normotensive one
    d, c_ = sorted(remaining, key=lambda c: bp_level[c])
    order[d], order[c_] = "D", "C"

    assigned = usable.select("hospitalization_id").with_columns(
        bhavani_phenotype=pl.Series(
            [order[c] for c in km.labels_], dtype=pl.Enum(BHAVANI_LABELS)
        )
    )
    centroid_frame = pl.DataFrame(
        {
            "cluster": list(range(len(centroids))),
            "bhavani_phenotype": [order[c] for c in range(len(centroids))],
            "n": [int((km.labels_ == c).sum()) for c in range(len(centroids))],
            "vitals_level": [vitals_level[c] for c in range(len(centroids))],
            "bp_level": [bp_level[c] for c in range(len(centroids))],
        }
    ).sort("bhavani_phenotype")

    # patients without enough trajectory are left null rather than forced into a
    # group: "not enough vitals charted in eight hours" is a real category and
    # assigning it to the nearest shape would invent a trajectory
    out = trajectories.select("hospitalization_id").join(
        assigned, on="hospitalization_id", how="left"
    )
    model = {
        "features": columns,
        "centre": centre,
        "scale": scale,
        "centroids": centroids,
        "labels": [order[c] for c in range(len(centroids))],
    }
    return out, centroid_frame, model


# ============================================================ model transfer


def save_model(path, model: dict, scheme: str, extra: dict | None = None) -> None:
    """
    persist a fitted phenotype model so contributing sites apply it

    Carries four things, and all four are load-bearing:

      * `features` -- the variable set, including which were dropped for
        sparsity. A site that recomputed this against its own availability
        would cluster in a different space.
      * `centre` / `scale` -- the derivation cohort's moments. A site that
        standardised against its own would re-centre the space and move
        patients between phenotypes without changing anything clinical.
      * `centroids` / `labels` -- the clusters and their deterministic names.

    This is the same contract `clifcs.phenotype.save_phenotype_model` defines
    for Zweck, extended with the scheme name so one file cannot be applied as
    the wrong phenotype system.
    """
    import json
    import pathlib

    payload = {
        "scheme": scheme,
        "features": list(model["features"]),
        "centre": [float(x) for x in model["centre"]],
        "scale": [float(x) for x in model["scale"]],
        "centroids": [[float(x) for x in row] for row in model["centroids"]],
        "labels": list(model["labels"]),
        # The transform, which has to travel with the moments and did not.
        #
        # This enumerated payload silently dropped `log_transform` and `z_clip`
        # because they were not named here, while `assign_seymour` put them in
        # the model dict specifically so they would travel. A receiving site
        # then found no transform, applied none, and compared raw AST -- which
        # runs from 1 to the thousands -- against centroids fitted on log1p(AST)
        # and standardised by log-space moments. Every patient lands in
        # whichever cluster is nearest the origin, and the site gets four
        # phenotype labels that look plausible and mean nothing.
        #
        # Written unconditionally, including as an empty list, so that
        # `apply_model` can tell "no transform was used" apart from "this file
        # predates the fix" -- see the check there.
        "log_transform": list(model.get("log_transform", [])),
        "z_clip": model.get("z_clip"),
        **(extra or {}),
    }
    pathlib.Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def apply_model(
    features: pl.DataFrame, path, scheme: str, labels: tuple[str, ...], logger
) -> pl.DataFrame:
    """
    label patients with a model fitted elsewhere, by nearest centroid

    Uses the *source* cohort's centre and scale, never this cohort's. Refuses a
    model fitted for a different scheme, and refuses one whose feature set this
    site cannot supply -- both of which would otherwise produce labels that look
    fine and mean nothing.
    """
    import json
    import pathlib

    model = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if model.get("scheme") != scheme:
        raise ValueError(
            f"{path} holds a {model.get('scheme')!r} model, not {scheme!r}"
        )
    names = model["features"]
    missing = [n for n in names if n not in features.columns]
    if missing:
        raise KeyError(
            f"the {scheme} model needs variable(s) absent at this site: "
            f"{missing}. They cannot be dropped locally -- the centroids were "
            "fitted in a space that includes them."
        )

    values = features.select(names).to_numpy().astype(float)
    # the source cohort's transform, moments and clip -- all three, or the
    # patient is placed in a different space from the centroids
    # Absent, not empty: a file written before the transform was carried
    # through. Refused rather than defaulted, because the default -- apply no
    # transform -- is exactly the wrong answer when the centroids were fitted
    # on log-transformed values, and it is wrong silently.
    if "log_transform" not in model:
        raise ValueError(
            f"{path} does not record which variables were log-transformed. It "
            "was written by a version of save_model that dropped that field, "
            "so there is no way to place a patient in the same space as its "
            "centroids. Regenerate it by re-running the cohort stage at the "
            "derivation site."
        )
    log_columns = [i for i, n in enumerate(names) if n in set(model["log_transform"])]
    if log_columns:
        values[:, log_columns] = np.log1p(np.clip(values[:, log_columns], 0.0, None))
    centre = np.asarray(model["centre"])
    scale = np.asarray(model["scale"])
    z = (values - centre) / scale
    z = np.where(np.isfinite(z), z, 0.0)
    if model.get("z_clip"):
        z = np.clip(z, -float(model["z_clip"]), float(model["z_clip"]))
    centroids = np.asarray(model["centroids"])
    nearest = np.argmin(
        ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1
    )
    logger.info(
        f"applied {scheme} model from {path} "
        f"({len(names)} features, {len(centroids)} clusters)"
    )
    return features.select("hospitalization_id").with_columns(
        pl.Series(
            f"{scheme}_phenotype",
            [model["labels"][i] for i in nearest],
            dtype=pl.Enum(labels),
        )
    )


# =============================================================== orchestration


def attach(
    paths, cohort: pl.DataFrame, cfg, logger, *, models: dict | None = None
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame], dict[str, dict]]:
    """
    attach both sepsis phenotypes to the cohort, fitting or applying as asked

    Fitted on the **pooled** sepsis subset for the same reason the Zweck
    phenotypes are: a site that clusters its own patients gets its own clusters,
    and the label stops meaning the same thing anywhere else. `models` maps
    `"seymour"` / `"bhavani"` to a saved model file, which is what a contributing
    site passes instead of refitting.

    Returns the cohort with `sepsis_cohort`, `seymour_phenotype` and
    `bhavani_phenotype` attached, a dict of tables to write, and a dict of
    fitted models to save (empty when applying).
    """
    models = models or {}
    tables: dict[str, pl.DataFrame] = {}
    fitted: dict[str, dict] = {}

    cohort = mark_sepsis_cohort(cohort, cfg, logger)
    sepsis = cohort.filter("sepsis_cohort")
    if sepsis.height == 0:
        logger.warning(
            "no patients carry a septic shock code; both sepsis phenotypes are "
            "left unassigned at this site"
        )
        return (
            cohort.with_columns(
                seymour_phenotype=pl.lit(None, dtype=pl.Enum(SEYMOUR_LABELS)),
                bhavani_phenotype=pl.lit(None, dtype=pl.Enum(BHAVANI_LABELS)),
            ),
            tables,
            fitted,
        )

    def anchors_for(site: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        part = sepsis.filter(pl.col("site") == site)
        return part, part.select(
            "hospitalization_id", anchor_dttm=pl.col("cs_onset_dttm")
        )

    # ------------------------------------------------------------- Seymour
    features = pl.concat(
        [
            seymour_features(paths.site_data[site], *anchors_for(site), cfg, logger)
            for site in paths.sites
            if sepsis.filter(pl.col("site") == site).height
        ]
    )
    tables["seymour_availability"] = availability(features)

    if "seymour" in models:
        seymour = apply_model(
            features, models["seymour"], "seymour", SEYMOUR_LABELS, logger
        )
    else:
        reduced, _ = select_available(features, cfg, logger)
        seymour, centroids, model = assign_seymour(reduced, cfg, logger)
        tables["seymour_centroids"] = centroids
        fitted["seymour"] = model
        if bool(cfg.seymour.assignment.get("report_agreement", False)):
            tables["seymour_agreement"] = _agreement(
                seymour, transfer_seymour(reduced, cfg, logger), logger
            )

    # ------------------------------------------------------------- Bhavani
    trajectories, counts = (
        pl.concat(frames)
        for frames in zip(
            *[
                bhavani_trajectories(
                    paths.site_data[site], anchors_for(site)[1], cfg, logger
                )
                for site in paths.sites
                if sepsis.filter(pl.col("site") == site).height
            ]
        )
    )
    if "bhavani" in models:
        bhavani = apply_model(
            trajectories, models["bhavani"], "bhavani", BHAVANI_LABELS, logger
        )
    else:
        bhavani, centroids, model = assign_bhavani(trajectories, counts, cfg, logger)
        if centroids.height:
            tables["bhavani_centroids"] = centroids
            fitted["bhavani"] = model

    out = (
        cohort.join(seymour, on="hospitalization_id", how="left")
        .join(bhavani, on="hospitalization_id", how="left")
    )

    # Summaries are per site and are written whether or not the model was
    # refit, since they are the breakdown tables the consortium asked for.
    assigned = out.filter("sepsis_cohort")
    tables["seymour_summary"] = summarize(
        assigned, "seymour_phenotype", SEYMOUR_LABELS, cfg.seymour, logger,
        scheme="seymour",
    )
    tables["bhavani_summary"] = summarize(
        assigned, "bhavani_phenotype", BHAVANI_LABELS, cfg.bhavani, logger,
        scheme="bhavani",
    )
    return out, tables, fitted


def _agreement(
    refit: pl.DataFrame, transfer: pl.DataFrame, logger
) -> pl.DataFrame:
    """
    how often the refit and the published-value transfer agree

    Reported rather than reconciled. High agreement is evidence the refit found
    the published structure; low agreement says the label is ours and the
    paper's name on it is a convenience -- the same caveat `clifcs.trials`
    records about naming a region after a trial.
    """
    joined = refit.join(transfer, on="hospitalization_id", how="inner")
    same = (
        joined["seymour_phenotype"].cast(pl.String)
        == joined["seymour_transfer"].cast(pl.String)
    )
    overall = float(same.mean() or 0.0)
    logger.info(
        f"Seymour: refit and published-value transfer agree on {overall:.1%} of "
        f"{joined.height} patients"
    )
    if overall < 0.5:
        logger.warning(
            "  under half -- the published Table 2 profiles and the refit "
            "clusters are not the same partition, so `seymour_phenotype` should "
            "be read as this cohort's four classes rather than as the paper's"
        )
    return (
        joined.group_by("seymour_phenotype", "seymour_transfer")
        # `n`, the count name the export suppresses and checks
        .len(name="n")
        .sort("seymour_phenotype", "seymour_transfer")
    )


# ================================================================= reporting


def summarize(
    frame: pl.DataFrame,
    column: str,
    labels: tuple[str, ...],
    cfg_block,
    logger,
    *,
    scheme: str,
) -> pl.DataFrame:
    """
    per-site phenotype breakdown with the published distribution beside it

    Per site and never pooled: the denominator is a diagnosis code whose use
    differs between hospitals, so one combined percentage would be an average of
    two coding practices rather than a property of patients.
    """
    sites = sorted(frame["site"].unique().to_list())
    rows = []
    for label in labels:
        part = frame.filter(pl.col(column) == label)
        row = {
            scheme: label,
            "n": part.height,
            "share": part.height / max(frame.height, 1),
            "mortality": float(part["in_hospital_mortality"].mean() or 0.0)
            if part.height
            else None,
            "median_icu_los_days": float(part["icu_los_days"].median() or 0.0)
            if part.height
            else None,
        }
        # one column per site rather than the hardcoded pair the Zweck
        # summary used, so a third site needs no code change
        for site in sites:
            at_site = frame.filter(pl.col("site") == site)
            row[f"n_{site}"] = int(
                at_site.filter(pl.col(column) == label).height
            )
            row[f"share_{site}"] = (
                at_site.filter(pl.col(column) == label).height
                / max(at_site.height, 1)
            )
        rows.append(row)

    n_unassigned = int(frame[column].is_null().sum())
    summary = pl.DataFrame(rows)

    logger.info(f"{scheme} phenotypes, n={frame.height}:")
    for row in summary.iter_rows(named=True):
        per_site = "  ".join(
            f"{s} {row[f'share_{s}']:.1%}" for s in sites
        )
        logger.info(
            f"  {row[scheme]:<8} n={row['n']:<6} {row['share']:.1%}  "
            f"mortality {row['mortality'] or 0:.1%}  [{per_site}]"
        )
    if n_unassigned:
        logger.info(
            f"  {'(none)':<8} n={n_unassigned:<6} "
            f"{n_unassigned / max(frame.height, 1):.1%}  not assignable"
        )

    _check_published(summary, column, labels, cfg_block, logger, scheme=scheme)
    return summary


def _check_published(
    summary: pl.DataFrame, column: str, labels, cfg_block, logger, *, scheme: str
) -> None:
    """
    compare the recovered labelling against what the paper reported

    A refit that does not reproduce the published mortality ordering has not
    recovered the published phenotypes, whatever the cluster names say. This is
    the same check `clifcs.phenotype.summarize` applies to Zweck's I < II < III.
    """
    outcomes = dict(cfg_block.get("published_outcomes", {}) or {})
    if not outcomes:
        return

    observed = {
        row[column if column in summary.columns else scheme]: row["mortality"]
        for row in summary.iter_rows(named=True)
    }

    if scheme == "seymour":
        # the paper's ordering is strict: alpha < beta < gamma < delta
        series = [observed.get(l) for l in ("alpha", "beta", "gamma", "delta")]
        if all(v is not None for v in series) and not all(
            a <= b for a, b in zip(series, series[1:])
        ):
            logger.warning(
                "Seymour: mortality does not increase across alpha -> beta -> "
                "gamma -> delta ("
                + ", ".join(
                    f"{l} {v:.1%}"
                    for l, v in zip(("alpha", "beta", "gamma", "delta"), series)
                )
                + ") -- the refit may not correspond to the published phenotypes"
            )
    else:
        # Bhavani's two cohorts disagree on whether B or C is lowest, so only
        # what both agree on is checked: A and D exceed B and C
        hi = [observed.get(l) for l in ("A", "D")]
        lo = [observed.get(l) for l in ("B", "C")]
        if all(v is not None for v in hi + lo) and min(hi) <= max(lo):
            logger.warning(
                "Bhavani: groups A and D are not both higher-mortality than B "
                "and C ("
                + ", ".join(f"{l} {observed[l]:.1%}" for l in ("A", "B", "C", "D"))
                + ") -- the recovered groups may not correspond to the published ones"
            )

    for row in summary.iter_rows(named=True):
        label = row[scheme]
        published = outcomes.get(label)
        if not published:
            continue
        share = published.get("share") or published.get("share_validation")
        if share and abs(row["share"] - float(share)) > 0.15:
            logger.warning(
                f"{scheme} {label}: {row['share']:.1%} of this cohort against "
                f"{float(share):.1%} published -- a >15 point difference, which "
                "for a coded sepsis denominator is expected but makes the share "
                "uninterpretable as a replication"
            )

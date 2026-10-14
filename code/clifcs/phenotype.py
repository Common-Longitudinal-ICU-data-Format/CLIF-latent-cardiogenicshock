#!/usr/bin/env python3

"""
cardiogenic shock phenotypes

Zweck E, Thayer KL, Helgestad OKL, et al. "Phenotyping Cardiogenic Shock."
J Am Heart Assoc. 2021;10(14):e020085. doi:10.1161/JAHA.120.020085

That paper applied consensus k-means to the laboratory variables most strongly
associated with in-hospital mortality and recovered three phenotypes:

  I   non-congested    relatively preserved labs
  II  cardiorenal      renal dysfunction without major metabolic derangement
  III cardiometabolic  renal plus hepatic and metabolic derangement

This is a reimplementation, not a transfer of the published model: the paper
reports the clustering variables and the direction of each phenotype's profile
but not the fitted centroids, so the clusters are refit here on the eight labs
it names (the four strongest being eGFR, lactate, bicarbonate and ALT) and then
*labelled deterministically* from their centroid profiles rather than by
cluster index, which k-means assigns arbitrarily. Mortality ordering
(I < II < III in the paper) is reported by `summarize` so the recovered labels
can be checked against it.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl
from sklearn.cluster import KMeans

from clifcs.clif import scan

#: phenotype labels, in the paper's order
PHENOTYPES: tuple[str, ...] = (
    "I non-congested",
    "II cardiorenal",
    "III cardiometabolic",
)

#: labs drawn straight from CLIF; eGFR is derived from creatinine
_MEASURED_LABS = (
    "lactate",
    "bicarbonate",
    "alt",
    "creatinine",
    "bun",
    "platelet_count",
    "wbc",
)


def nearest_labs(data_dir, cohort: pl.DataFrame, cfg) -> pl.DataFrame:
    """
    for each lab, the single value measured closest in time to CS onset within
    the configured window
    """
    ph = cfg.phenotype
    window = cohort.lazy().select(
        "hospitalization_id",
        "cs_onset_dttm",
        win_start=pl.col("cs_onset_dttm")
        - pl.duration(hours=float(ph.window_lookback_hours)),
        win_end=pl.col("cs_onset_dttm")
        + pl.duration(hours=float(ph.window_lookahead_hours)),
    )

    long = (
        scan(data_dir, "clif_labs", dttm_cols=("lab_result_dttm",))
        .filter(
            pl.col("lab_category").is_in(list(_MEASURED_LABS)),
            pl.col("lab_value_numeric").is_not_null(),
        )
        .join(window, on="hospitalization_id", how="inner")
        .filter(
            pl.col("lab_result_dttm").is_between(pl.col("win_start"), pl.col("win_end"))
        )
        .with_columns(
            gap_s=(pl.col("lab_result_dttm") - pl.col("cs_onset_dttm"))
            .dt.total_seconds()
            .abs()
        )
        # nearest in time; ties broken toward the earlier draw so the result does
        # not depend on row order
        .sort(["hospitalization_id", "lab_category", "gap_s", "lab_result_dttm"])
        .unique(
            subset=["hospitalization_id", "lab_category"],
            keep="first",
            maintain_order=True,
        )
        .select("hospitalization_id", "lab_category", "lab_value_numeric")
        .collect()
    )

    wide = long.pivot(
        on="lab_category", index="hospitalization_id", values="lab_value_numeric"
    )
    # guarantee every expected column exists even if a site never measured one
    missing = [c for c in _MEASURED_LABS if c not in wide.columns]
    if missing:
        wide = wide.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing]
        )
    return wide.select("hospitalization_id", *_MEASURED_LABS)


def add_egfr(df: pl.DataFrame) -> pl.DataFrame:
    """
    CKD-EPI 2021 creatinine equation (race-free)

    eGFR = 142 * min(Scr/k, 1)^a * max(Scr/k, 1)^-1.200 * 0.9938^age * 1.012 [female]
    with k = 0.7, a = -0.241 for female and k = 0.9, a = -0.302 for male.
    Left null when sex is not recorded as Male or Female, since the equation has
    no coefficients for that case.
    """
    female = pl.col("sex_category") == "Female"
    male = pl.col("sex_category") == "Male"
    kappa = pl.when(female).then(0.7).when(male).then(0.9).otherwise(None)
    alpha = pl.when(female).then(-0.241).when(male).then(-0.302).otherwise(None)
    ratio = pl.col("creatinine") / kappa

    return df.with_columns(
        egfr=pl.when(pl.col("creatinine").is_null() | kappa.is_null())
        .then(None)
        .otherwise(
            142.0
            * pl.min_horizontal(ratio, pl.lit(1.0)).pow(alpha)
            * pl.max_horizontal(ratio, pl.lit(1.0)).pow(-1.200)
            * pl.lit(0.9938).pow(pl.col("age_at_admission").cast(pl.Float64))
            * pl.when(female).then(1.012).otherwise(1.0)
        )
    )


def _features(df: pl.DataFrame, features, centre, scale) -> np.ndarray:
    """z-score the phenotype labs and median-impute what is missing"""
    values = df.select(features).to_numpy().astype(float)
    z = (values - centre) / scale
    return np.where(np.isfinite(z), z, 0.0)


def save_phenotype_model(path, features, centre, scale, centroids, labels) -> None:
    """
    persist the fitted phenotype model so other sites can apply it

    A site that refits k-means on its own patients gets its own clusters, and
    "phenotype II" then means something different at each site. Shipping the
    scaler and the centroids makes the label mean the same thing everywhere,
    which is the only way the pooled plot is interpretable.
    """
    pathlib.Path(path).write_text(
        json.dumps(
            {
                "features": list(features),
                "centre": [float(x) for x in centre],
                "scale": [float(x) for x in scale],
                "centroids": [[float(x) for x in row] for row in centroids],
                "labels": list(labels),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def apply_phenotype_model(df: pl.DataFrame, path, logger) -> pl.DataFrame:
    """
    label patients using a phenotype model fitted elsewhere

    Nearest centroid in the same z-scored space, using the *source* cohort's
    centre and scale -- not this cohort's, which would silently re-centre the
    space and move patients between phenotypes.
    """
    model = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    features = model["features"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"phenotype model needs columns absent here: {missing}")

    z = _features(
        df, features, np.asarray(model["centre"]), np.asarray(model["scale"])
    )
    centroids = np.asarray(model["centroids"])
    nearest = np.argmin(
        ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1
    )
    labels = [model["labels"][i] for i in nearest]
    logger.info(
        f"applied phenotype model from {path} "
        f"({len(features)} features, {len(centroids)} clusters)"
    )
    return df.with_columns(
        cs_phenotype=pl.Series(labels, dtype=pl.Enum(PHENOTYPES)),
        phenotype_features_imputed=pl.lit(False),
    )


def assign_phenotypes(df: pl.DataFrame, cfg) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    fit k-means on the z-scored phenotype labs and map clusters to the paper's
    phenotype names by centroid profile

    Returns the cohort with `cs_phenotype` attached, plus a per-cluster centroid
    table (in z-score units) for inspection.
    """
    ph = cfg.phenotype
    features = [lab.name for lab in ph.labs]
    primary = [lab.name for lab in ph.labs if lab.primary]

    values = df.select(features).to_numpy().astype(float)

    # z-score on observed values only, then median-impute, so that a missing lab
    # lands at the cohort centre instead of skewing the scale
    with np.errstate(invalid="ignore"):
        centre = np.nanmedian(values, axis=0)
        scale = np.nanstd(values, axis=0)
    scale = np.where((scale == 0) | ~np.isfinite(scale), 1.0, scale)
    centre = np.where(np.isfinite(centre), centre, 0.0)

    z = (values - centre) / scale
    n_missing = int(np.sum(~np.isfinite(z)))
    z = np.where(np.isfinite(z), z, 0.0)

    km = KMeans(
        n_clusters=int(ph.n_clusters),
        n_init=50,
        random_state=int(ph.random_state),
    ).fit(z)
    centroids = km.cluster_centers_

    # -------------------------------------------------- deterministic labelling
    # k-means numbers clusters arbitrarily, so name them from their profiles.
    idx = {name: i for i, name in enumerate(features)}

    def direction(name: str) -> float:
        """+1 if a higher value means worse organ function, -1 otherwise"""
        return -1.0 if name in ("egfr", "bicarbonate", "platelet_count") else 1.0

    def score(centroid: np.ndarray, names: list[str]) -> float:
        return float(
            sum(direction(n) * centroid[idx[n]] for n in names if n in idx)
        )

    metabolic = [n for n in ("lactate", "bicarbonate", "alt") if n in idx]
    renal = [n for n in ("egfr", "creatinine", "bun") if n in idx]

    order: dict[int, str] = {}
    remaining = list(range(len(centroids)))

    # III cardiometabolic: the most deranged metabolic / hepatic profile
    worst_metabolic = max(remaining, key=lambda c: score(centroids[c], metabolic))
    order[worst_metabolic] = PHENOTYPES[2]
    remaining.remove(worst_metabolic)

    # II cardiorenal: of what is left, the more deranged renal profile
    worst_renal = max(remaining, key=lambda c: score(centroids[c], renal))
    order[worst_renal] = PHENOTYPES[1]
    remaining.remove(worst_renal)

    # I non-congested: the remainder
    order[remaining[0]] = PHENOTYPES[0]

    labels = [order[c] for c in km.labels_]

    centroid_frame = pl.DataFrame(
        {
            "cluster": list(range(len(centroids))),
            "cs_phenotype": [order[c] for c in range(len(centroids))],
            "n": [int((km.labels_ == c).sum()) for c in range(len(centroids))],
            "metabolic_score": [
                score(centroids[c], metabolic) for c in range(len(centroids))
            ],
            "renal_score": [score(centroids[c], renal) for c in range(len(centroids))],
            **{
                f"z_{name}": [float(centroids[c][idx[name]]) for c in range(len(centroids))]
                for name in features
            },
        }
    ).sort("cs_phenotype")

    out = df.with_columns(
        cs_phenotype=pl.Series(labels, dtype=pl.Enum(PHENOTYPES)),
        phenotype_features_imputed=pl.lit(n_missing > 0),
    )
    model = {
        "features": features,
        "centre": centre,
        "scale": scale,
        "centroids": centroids,
        "labels": [order[c] for c in range(len(centroids))],
    }
    return out, centroid_frame, model


def summarize(df: pl.DataFrame, logger) -> pl.DataFrame:
    """
    per-phenotype size and in-hospital mortality

    The paper found mortality rising I < II < III; a recovered labelling that
    inverts that ordering is a signal the refit did not reproduce the published
    phenotypes and should be reported as such rather than taken at face value.
    """
    # One column per site present, rather than the hardcoded `n_ucmc`/`n_nu`
    # pair this used to build: a federated run has however many sites sent a
    # bundle, and naming two of them meant a third was silently omitted from
    # its own breakdown table.
    sites = sorted(df["site"].unique().drop_nulls().to_list())
    summary = (
        df.group_by("cs_phenotype")
        .agg(
            n=pl.len(),
            *[
                (pl.col("site") == site).sum().alias(f"n_{site}")
                for site in sites
            ],
            mortality=pl.col("in_hospital_mortality").mean(),
            median_icu_los_days=pl.col("icu_los_days").median(),
        )
        .sort("cs_phenotype")
    )
    logger.info("cardiogenic shock phenotypes (Zweck et al. 2021, refit):")
    for row in summary.iter_rows(named=True):
        per_site = ", ".join(f"{site} {row[f'n_{site}']}" for site in sites)
        logger.info(
            f"  {row['cs_phenotype']:<22} n={row['n']:<6} "
            f"({per_site})  "
            f"in-hospital mortality {row['mortality']:.1%}  "
            f"median ICU LOS {row['median_icu_los_days']:.1f}d"
        )

    mortality = summary.sort("cs_phenotype")["mortality"].to_list()
    if not all(a <= b for a, b in zip(mortality, mortality[1:])):
        logger.warning(
            "mortality does not increase monotonically across I -> II -> III; "
            "the refit clusters may not correspond to the published phenotypes"
        )
    return summary

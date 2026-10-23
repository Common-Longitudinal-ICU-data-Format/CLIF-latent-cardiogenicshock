"""
Do the latent classes predict mortality beyond baseline severity?

The clusters stratify by severity, so their raw mortality spread (11.6% to
67.4%) could be nothing but severity restated. This asks whether cluster
membership adds information once severity AT SHOCK ONSET is accounted for --
before the trajectories have diverged.

THE REVERSE-CAUSATION PROBLEM, which no adjustment fixes. "Record ended" is a
symbol in the clustering feature space, and a record ends because the patient
died. So G4 is partly identified BY dying, and an adjusted odds ratio for G4
is not "this trajectory raises mortality" but something closer to "dying looks
like this". The landmark analysis below is the honest version: restricted to
patients still in record at +48h, the trajectory up to that point is observed
strictly before the outcome it is asked to predict.
"""
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "code")
from clifcs import get_logger, load_paths

logger = get_logger("adjust")
paths = load_paths(role="site")

labels = pl.read_parquet(
    paths.patient_level_home / "longitudinal_clusters_septic.parquet"
)
cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet").select(
    "hospitalization_id", "site", "age_at_admission", "in_hospital_mortality"
)
state = pl.read_parquet(paths.patient_level_home / "trajectory_state.parquet")

# Baseline = the state at shock onset itself. Not a window average: the point
# of a baseline is to be fixed before the trajectory it is being compared
# against, and averaging over +0 to +48h would fold the trajectory into the
# covariate meant to be independent of it.
BASELINE = ["sofa_total", "map", "lactate", "nee", "n_vasopressors",
            "gcs_total", "creatinine", "platelet_count", "on_imv"]
# SOFA comes from sofa_grid, not from the state table: state.py leaves
# sofa_total null because stage_derive writes one row per encounter stamped at
# the end of the stay. sofa_grid holds the worst score in the 24h ending at
# each grid point, which is what a baseline covariate needs.
onset = (
    state.filter(pl.col("offset_hours") == 0.0)
    .select([c for c in ["hospitalization_id"] + BASELINE
             if c != "sofa_total"])
)
sofa_file = paths.patient_level_home / "sofa_grid.parquet"
if sofa_file.is_file():
    onset = onset.join(
        pl.read_parquet(sofa_file)
        .filter(pl.col("offset_hours") == 0.0)
        .select("hospitalization_id",
                pl.col("sofa_total").cast(pl.Float64)),
        on="hospitalization_id", how="left",
    )
    logger.info("SOFA at onset joined from sofa_grid.parquet")
else:
    logger.warning("no sofa_grid.parquet: SOFA omitted from the adjustment")

frame = (
    labels.join(cohort, on="hospitalization_id", how="inner")
    .join(onset, on="hospitalization_id", how="inner")
)
logger.info(
    f"{frame.height:,} patients with a cluster, an outcome and onset physiology"
)

rows_out = []
covariates = BASELINE + ["age_at_admission"]
# sofa_total is 100% null in trajectory_state -- the column is emitted and
# never populated, which is also why there is no flow panel for it. Drop any
# covariate with no observed values rather than imputing a median of NaN.
present = [
    c for c in covariates
    if c in frame.columns and frame[c].null_count() < frame.height
]
dropped = [c for c in covariates if c not in present]
if dropped:
    logger.warning(f"dropping covariates with no observed values: {dropped}")
logger.info(f"severity covariates at onset: {present}")


def design(part, with_cluster, column):
    x = part.select(present).to_numpy().astype(float)
    # median-impute, then standardise; a missing lab lands at the centre
    with np.errstate(invalid="ignore"):
        centre = np.nanmedian(x, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    x = np.where(np.isfinite(x), x, centre)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    x = (x - x.mean(axis=0)) / scale
    if with_cluster:
        levels = sorted(part[column].drop_nulls().unique().to_list())[1:]
        dummies = np.column_stack([
            (part[column] == level).cast(pl.Int8).to_numpy() for level in levels
        ]) if levels else np.zeros((part.height, 0))
        x = np.column_stack([x, dummies])
    return x


def cross_validated_auc(part, with_cluster, column, y):
    x = design(part, with_cluster, column)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train, test in folds.split(x, y):
        model = LogisticRegression(max_iter=2000, C=1.0).fit(x[train], y[train])
        scores.append(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1]))
    return float(np.mean(scores))


for column, name in (
    ("growth_cluster", "latent class growth (G1-G4)"),
    ("perfusion_cluster", "MAP/pressor clusters (P1-P4)"),
    ("sequence_cluster", "care-state sequence clusters (S1-S4)"),
):
    if column not in frame.columns:
        continue
    for label, part in (
        ("all patients", frame.filter(pl.col(column).is_not_null())),
        # LANDMARK: still being charted at +48h, so the sequence used to assign
        # the cluster is observed before the deaths it is asked to predict
        ("in record at +48h", frame.filter(
            pl.col(column).is_not_null()
            & pl.col("hospitalization_id").is_in(
                state.filter(
                    (pl.col("offset_hours") == 48.0)
                    & pl.col("map").is_not_null()
                )["hospitalization_id"]
            )
        )),
    ):
        if part.height < 200:
            continue
        y = part["in_hospital_mortality"].cast(pl.Int8).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        base = cross_validated_auc(part, False, column, y)
        full = cross_validated_auc(part, True, column, y)

        # adjusted odds ratios, fitted on everything
        x = design(part, True, column)
        model = LogisticRegression(max_iter=2000, C=1e6).fit(x, y)
        levels = sorted(part[column].drop_nulls().unique().to_list())
        reference = levels[0]
        coefficients = model.coef_[0][len(present):]

        print(f"\n=== {name}, {label}  (n={part.height:,}, "
              f"{y.mean():.1%} died)")
        print(f"    baseline severity alone      AUC {base:.4f}")
        print(f"    baseline + cluster           AUC {full:.4f}   "
              f"(+{full - base:+.4f})")
        print(f"    reference level: {reference}")
        for level, coefficient in zip(levels[1:], coefficients):
            crude = part.filter(pl.col(column) == level)["in_hospital_mortality"].mean()
            crude_ref = part.filter(
                pl.col(column) == reference
            )["in_hospital_mortality"].mean()
            print(f"      {level}  crude {crude:>6.1%} vs {crude_ref:.1%}   "
                  f"adjusted OR {np.exp(coefficient):>6.2f}")
            rows_out.append({
                "stratification": column, "cohort": label, "n": part.filter(
                    pl.col(column) == level).height,
                "died": float(y.mean()), "cluster": level,
                "crude_mortality": float(crude),
                "adjusted_odds_ratio": float(np.exp(coefficient)),
                "auc_severity_only": base, "auc_with_cluster": full,
            })


if rows_out:
    table = pl.DataFrame(rows_out)
    out = (
        paths.output_home / "tables" / paths.model_key
        / "adjusted_mortality_septic.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out)
    logger.info(f"wrote {out} ({table.height} rows)")

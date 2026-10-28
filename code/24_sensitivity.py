#!/usr/bin/env python3

"""
24_sensitivity.py -- phase 2: how much the pressure moves per unit of dose

Fits two models of the same thing and reports the comparison, because in this
cohort the data supports one far better than the other.

    python code/24_sensitivity.py --subset cardiogenic

THE GAIN MODEL IS NEARLY UNIDENTIFIED HERE AND THAT IS A RESULT. Dose and MAP
correlate at -0.002 in levels, -0.000 in differences, and between -0.004 and
+0.003 across lags of one to six hours. The conditional means separate cleanly
-- MAP rises 1.8 mmHg in the hour after an escalation and falls 1.4 after a
wean -- so the direction of a dose change carries signal and the magnitude does
not. A random walk with partial pooling will still produce a smooth, plausible
s(t) out of that, made almost entirely of shrinkage, which is why `data_share`
is computed and reported for every patient.

Run after 23, which writes the hourly dose and MAP series this reads.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.config import CONFIG_HOME
from clifcs.sensitivity import (
    direction_probability, fit_hyperparameters, saturating, smooth_one,
    unobserved_variance,
)

#: a patient needs this share of their posterior precision on s to come from
#: their own data before the estimate is called usable. Not a statistical
#: threshold -- a reporting one, so phase 3 can stratify on sensitivity only
#: where sensitivity was measured.
RELIABLE_DATA_SHARE = 0.30


def build_series(hourly, kind, reference, ed50):
    """per-patient (MAP, regressor) arrays on the hourly grid"""
    ordered = hourly.sort(["hospitalization_id", "offset_hours"])
    series, index = [], []
    for (patient,), part in ordered.group_by(
        ["hospitalization_id"], maintain_order=True
    ):
        y = part["map"].cast(pl.Float64).to_numpy().astype(np.float64)
        regressor = saturating(
            part["nee"].fill_null(0.0).to_numpy(), kind, reference, ed50
        )
        if np.isfinite(y).sum() < 2:
            continue
        series.append((y, regressor))
        index.append((patient, part["offset_hours"].to_numpy()))
    return series, index


def fit_gain(hourly, cfg, kind, logger, seed):
    """the specified model, plus the diagnostic that says whether to trust it"""
    reference = float(cfg.sensitivity.reference_dose)
    ed50 = float(cfg.sensitivity.ed50)
    series, index = build_series(hourly, kind, reference, ed50)
    logger.info(f"[{kind}] {len(series):,} patients with at least two MAPs")
    if not series:
        return pl.DataFrame(), {}

    prior_mean = [float(cfg.map.adequate), 0.0]
    prior_var = [
        float(cfg.sensitivity.prior_baseline_var),
        float(cfg.sensitivity.prior_slope_var),
    ]
    obs_var, walk_var = fit_hyperparameters(
        series, prior_mean, prior_var, logger, seed=seed
    )

    rows = []
    for (y, regressor), (patient, hours) in zip(series, index):
        state, cov = smooth_one(
            y, regressor, obs_var, walk_var, prior_mean, prior_var
        )
        baseline_free = unobserved_variance(len(y), walk_var, prior_var)
        share = 1.0 - np.clip(cov[:, 1, 1] / baseline_free, 0.0, 1.0)
        rows.append(pl.DataFrame({
            "hospitalization_id": [patient] * len(y),
            "offset_hours": hours.astype(np.float64),
            "s": state[:, 1],
            "s_sd": np.sqrt(np.clip(cov[:, 1, 1], 0.0, None)),
            "baseline": state[:, 0],
            "data_share": share,
        }))
    frame = pl.concat(rows, how="vertical")
    per_patient = frame.group_by("hospitalization_id").agg(
        data_share=pl.col("data_share").max(),
        s_final=pl.col("s").last(),
    )
    reliable = per_patient.filter(pl.col("data_share") >= RELIABLE_DATA_SHARE)
    summary = {
        "form": kind,
        "patients": per_patient.height,
        "observation_sd": float(np.sqrt(obs_var)),
        "slope_walk_sd": float(np.sqrt(walk_var[1])),
        "median_data_share": float(per_patient["data_share"].median()),
        "reliable_patients": reliable.height,
        "reliable_share": reliable.height / max(per_patient.height, 1),
        "median_s": float(frame["s"].median()),
        "median_s_sd": float(frame["s_sd"].median()),
        "share_s_within_1sd_of_zero": float(
            (frame["s"].abs() < frame["s_sd"]).mean()
        ),
    }
    logger.info(
        f"  [{kind}] median s {summary['median_s']:+.3f} mmHg per unit, "
        f"median sd {summary['median_s_sd']:.3f}; "
        f"{summary['share_s_within_1sd_of_zero']:.1%} of patient-hours have s "
        "within one sd of zero"
    )
    logger.info(
        f"  [{kind}] median data_share {summary['median_data_share']:.3f}; "
        f"{reliable.height:,} of {per_patient.height:,} patients "
        f"({summary['reliable_share']:.1%}) clear the "
        f"{RELIABLE_DATA_SHARE:.0%} floor"
    )
    return frame, summary


def titration_events(encoded, cfg, logger):
    """
    one row per titration, with whether the pressure moved the way it was pushed

    An escalation goes the right way if the MAP rose beyond the deadband; a
    wean goes the right way if it did NOT fall beyond it. Holds are excluded --
    there was no push, so there is nothing for the pressure to agree or
    disagree with.
    """
    deadband = float(cfg.response.deadband_mmhg)
    events = encoded.filter(
        pl.col("action").is_in(["escalate", "de-escalate"])
        & pl.col("map_response").is_not_null()
    ).with_columns(
        went_the_right_way=pl.when(pl.col("action") == "escalate")
        .then(pl.col("map_response") > deadband)
        .otherwise(pl.col("map_response") >= -deadband)
    )
    logger.info(
        f"  {events.height:,} titrations with a measured response "
        f"({events['went_the_right_way'].mean():.1%} went the way the dose "
        "was pushed)"
    )
    return events.select(
        "hospitalization_id", "offset_hours", "action", "went_the_right_way"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger("24_sensitivity", paths.log_home / "24_sensitivity.log")
    cfg = OmegaConf.load(args.config or CONFIG_HOME / "dynamics.yaml")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, tables / f"dynamics_config_{args.subset}.yaml")

    hourly_file = (
        paths.patient_level_home / f"hourly_dose_map_{args.subset}.parquet"
    )
    encoded_file = (
        paths.patient_level_home / f"action_response_{args.subset}.parquet"
    )
    if not hourly_file.is_file():
        logger.error(f"no {hourly_file.name}; run 23 first")
        return
    hourly = pl.read_parquet(hourly_file)
    encoded = pl.read_parquet(encoded_file)
    logger.info(
        f"{hourly.height:,} patient-hours, "
        f"{hourly['hospitalization_id'].n_unique():,} patients"
    )

    # ---- the specified gain model, both saturating forms
    summaries, frames = [], {}
    for kind in ("log", "emax"):
        frame, summary = fit_gain(hourly, cfg, kind, logger, args.seed)
        if summary:
            summaries.append(summary)
            frames[kind] = frame
    if summaries:
        pl.DataFrame(summaries).write_csv(
            tables / f"sensitivity_gain_summary_{args.subset}.csv"
        )
    preferred = cfg.sensitivity.get("preferred_form", "log")
    if preferred in frames:
        frames[preferred].write_parquet(
            paths.patient_level_home / f"sensitivity_{args.subset}.parquet"
        )
        logger.info(f"wrote s(t) from the {preferred!r} form")

    # ---- the direction model
    events = titration_events(encoded, cfg, logger)
    direction = direction_probability(
        events, logger,
        forgetting=float(cfg.sensitivity.forgetting_per_hour),
        prior_weight=float(cfg.sensitivity.direction_prior_events),
    )
    if not direction.is_empty():
        direction.write_parquet(
            paths.patient_level_home
            / f"sensitivity_direction_{args.subset}.parquet"
        )
        per_patient = direction.group_by("hospitalization_id").agg(
            p_last=pl.col("p_right_way").last(),
            events=pl.col("effective_events").max(),
        )
        logger.info(
            f"  direction: {per_patient.height:,} patients, median final "
            f"p {per_patient['p_last'].median():.3f}, "
            f"IQR {per_patient['p_last'].quantile(0.25):.3f}"
            f"-{per_patient['p_last'].quantile(0.75):.3f}"
        )
        # The comparison that matters: does the direction estimate actually
        # discriminate, or is it also shrinkage? Spread across patients is the
        # cheap version of that question.
        pl.DataFrame([{
            "model": "direction probability",
            "patients": per_patient.height,
            "median_p": float(per_patient["p_last"].median()),
            "iqr_p": float(
                per_patient["p_last"].quantile(0.75)
                - per_patient["p_last"].quantile(0.25)
            ),
            "median_effective_events": float(per_patient["events"].median()),
        }]).write_csv(
            tables / f"sensitivity_direction_summary_{args.subset}.csv"
        )

    # ---- do the two models agree about who is sensitive?
    if preferred in frames and not direction.is_empty():
        gain = frames[preferred].group_by("hospitalization_id").agg(
            s_final=pl.col("s").last(),
            data_share=pl.col("data_share").max(),
        )
        joined = gain.join(
            direction.group_by("hospitalization_id").agg(
                p_last=pl.col("p_right_way").last(),
                events=pl.col("effective_events").max(),
            ),
            on="hospitalization_id", how="inner",
        )
        if joined.height > 100:
            from scipy.stats import spearmanr
            rho = spearmanr(joined["s_final"], joined["p_last"]).statistic
            trusted = joined.filter(
                pl.col("data_share") >= RELIABLE_DATA_SHARE
            )
            rho_trusted = (
                spearmanr(trusted["s_final"], trusted["p_last"]).statistic
                if trusted.height > 100 else float("nan")
            )
            logger.info(
                f"  agreement between the two models: Spearman "
                f"{rho:+.3f} over {joined.height:,} patients, "
                f"{rho_trusted:+.3f} over the {trusted.height:,} whose gain "
                "estimate clears the data_share floor"
            )
            pl.DataFrame([{
                "patients": joined.height,
                "spearman_all": float(rho),
                "patients_reliable": trusted.height,
                "spearman_reliable": float(rho_trusted),
            }]).write_csv(
                tables / f"sensitivity_agreement_{args.subset}.csv"
            )
    logger.info("24 done")


if __name__ == "__main__":
    main()

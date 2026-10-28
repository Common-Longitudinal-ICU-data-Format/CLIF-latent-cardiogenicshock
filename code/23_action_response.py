#!/usr/bin/env python3

"""
23_action_response.py -- phase 1: encode what was done and what followed

Builds the hourly norepinephrine-equivalent dose and MAP series from the raw
CLIF tables, then labels every decision step with an action-response state.

    python code/23_action_response.py --subset cardiogenic

READS config/dynamics.yaml AND WRITES IT BESIDE THE OUTPUT. Every threshold --
the conversion factors, the dose deadband, the MAP band, the step and response
lengths -- lives there, and a copy is saved with each run, because a state
distribution is uninterpretable without the cuts that produced it.

THE COHORT IS IMPORTED, NOT REDEFINED. `cohort_ids` from 11 is the single
definition of who is in each subset; a second copy here would drift from it.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.action_response import STATES, UNDECIDABLE, encode, hourly_map
from clifcs.config import CONFIG_HOME
from clifcs.disposition import bound_to_record, dispositions
from clifcs.nee import agent_rates, hourly_nee

END_HOURS = 168.0


def load_cohort(paths, subset, logger):
    """11's cohort definition, imported rather than restated"""
    spec = importlib.util.spec_from_file_location(
        "longitudinal",
        pathlib.Path(__file__).resolve().parent / "11_longitudinal_clusters.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cohort_ids(paths, subset, logger)


def sanity_check(encoded, hourly, rates, logger, n=10, seed=42):
    """
    walk ten random patients from the raw records back up to their states

    Prints, for each, the raw administration rows and hourly series beside the
    labelled steps, so a reader can confirm by eye that an "escalate" lines up
    with a dose going up in the medication table rather than with an artefact
    of the resampling. Ten is few enough to actually read.
    """
    rng = np.random.default_rng(seed)
    decidable = encoded.filter(pl.col("state") != UNDECIDABLE)
    candidates = (
        decidable.filter(pl.col("action").is_in(["escalate", "de-escalate"]))
        ["hospitalization_id"].unique().to_list()
    )
    if not candidates:
        logger.warning("no patient had a titration to check")
        return
    picked = [
        candidates[i] for i in
        rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    ]
    logger.info(f"sanity check on {len(picked)} patients with titrations:")
    for patient in picked:
        steps = encoded.filter(
            (pl.col("hospitalization_id") == patient)
            & (pl.col("state") != UNDECIDABLE)
        ).sort("offset_hours")
        raw = rates.filter(
            pl.col("hospitalization_id") == patient
        ).sort("admin_dttm")
        logger.info(
            f"  patient {patient}: {steps.height} decidable steps, "
            f"{raw.height} administration records"
        )
        for row in steps.head(6).iter_rows(named=True):
            logger.info(
                f"      t={row['offset_hours']:>6.1f}h  "
                f"NEE {row['nee_start'] if row['nee_start'] is not None else float('nan'):>7.4f}"
                f" -> {row['nee_end'] if row['nee_end'] is not None else float('nan'):>7.4f}  "
                f"MAP {row['map_end'] if row['map_end'] is not None else float('nan'):>5.1f}"
                f" d{row['map_response'] if row['map_response'] is not None else float('nan'):>+6.1f}  "
                f"{row['action']:<12} {row['state']}"
            )
        for row in raw.head(4).iter_rows(named=True):
            logger.info(
                f"        raw: {row['admin_dttm']}  {row['med_category']:<16} "
                f"{row['dose']} {row['unit']}  -> NEE {row['nee']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--config", default=None,
                        help="override config/dynamics.yaml")
    parser.add_argument("--end-hours", type=float, default=END_HOURS)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "23_action_response", paths.log_home / "23_action_response.log"
    )
    cfg = OmegaConf.load(args.config or CONFIG_HOME / "dynamics.yaml")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, tables / f"dynamics_config_{args.subset}.yaml")
    logger.info(
        f"step {cfg.action.step_hours:.0f}h, response {cfg.response.hours:.0f}h, "
        f"dose floors {cfg.action.relative_minimum:.0%} and "
        f"{cfg.action.absolute_minimum}, MAP deadband "
        f"{cfg.response.deadband_mmhg:.0f} mmHg, adequate "
        f"{cfg.map.adequate:.0f} mmHg"
    )

    cohort = load_cohort(paths, args.subset, logger)
    onset = pl.read_parquet(paths.cohort_home / "cs_onset.parquet").select(
        "hospitalization_id",
        onset=pl.col("cs_onset_dttm").cast(pl.Datetime("us")),
    )

    parts, raw_parts, disp_parts = [], [], []
    for site in paths.sites:
        ids = cohort.filter(
            pl.col("site") == site
        )["hospitalization_id"].to_list()
        if not ids:
            continue
        anchors = onset.filter(pl.col("hospitalization_id").is_in(ids))
        logger.info(f"[{site}] {anchors.height:,} patients")
        rates = agent_rates(paths.raw_subset(site), anchors, cfg, logger)
        dose = hourly_nee(rates, anchors, cfg, args.end_hours, logger)
        pressure = hourly_map(
            paths.raw_subset(site), anchors, cfg, args.end_hours, logger
        )
        merged = dose.join(
            pressure.select("hospitalization_id", "offset_hours", "map"),
            on=["hospitalization_id", "offset_hours"], how="left",
        ).with_columns(site=pl.lit(site))
        # Bound every series at the hour the record stops. Without this the
        # grid asks about hours that never happened -- phase 1 first ran
        # unbounded and the undecidable share climbed from 16.5% on day 0 to
        # 53.3% on day 6, which was discharge, not sparse charting.
        disposition = dispositions(paths.raw_subset(site), anchors, logger)
        merged = bound_to_record(merged, disposition, logger)
        disp_parts.append(disposition.with_columns(site=pl.lit(site)))
        parts.append(merged)
        if not rates.is_empty():
            raw_parts.append(rates)

    if not parts:
        logger.error("no site produced an hourly series")
        return
    hourly = pl.concat(parts, how="diagonal_relaxed")
    disposition = pl.concat(disp_parts, how="diagonal_relaxed")
    disposition.write_parquet(
        paths.patient_level_home / f"disposition_{args.subset}.parquet"
    )
    rates = (
        pl.concat(raw_parts, how="diagonal_relaxed") if raw_parts
        else pl.DataFrame()
    )
    logger.info(f"{hourly.height:,} patient-hours across {len(parts)} sites")

    encoded = encode(hourly, cfg, logger)
    encoded = encoded.join(
        hourly.select("hospitalization_id", "site").unique(),
        on="hospitalization_id", how="left",
    )

    # ---- state frequencies
    decidable = encoded.filter(pl.col("state") != UNDECIDABLE)
    rows = []
    for state in STATES:
        part = decidable.filter(pl.col("state") == state)
        row = {
            "state": state,
            "steps": part.height,
            "share_of_decidable": part.height / max(decidable.height, 1),
            "patients": part["hospitalization_id"].n_unique(),
            "share_of_patients": part["hospitalization_id"].n_unique()
            / max(encoded["hospitalization_id"].n_unique(), 1),
        }
        for site in sorted(encoded["site"].drop_nulls().unique().to_list()):
            at_site = decidable.filter(pl.col("site") == site)
            row[f"share_at_{site}"] = (
                part.filter(pl.col("site") == site).height
                / max(at_site.height, 1)
            )
        rows.append(row)
    frequencies = pl.DataFrame(rows, infer_schema_length=None)
    frequencies.write_csv(
        tables / f"action_response_states_{args.subset}.csv"
    )
    logger.info("state frequencies (share of decidable steps):")
    for row in frequencies.iter_rows(named=True):
        logger.info(
            f"    {row['state']:<26} {row['steps']:>8,} steps "
            f"{row['share_of_decidable']:>6.1%}   "
            f"{row['patients']:>6,} patients"
        )
    undecided = encoded.filter(pl.col("state") == UNDECIDABLE)
    logger.info(
        f"  undecidable: {undecided.height:,} steps "
        f"({undecided.height / max(encoded.height, 1):.1%}), "
        f"{undecided.filter(pl.col('map_end').is_null()).height:,} for want of "
        "a MAP and the rest for want of a response window"
    )

    # ---- sequences, as strings, one per patient
    sequences = (
        encoded.sort(["hospitalization_id", "offset_hours"])
        .group_by("hospitalization_id", maintain_order=True)
        .agg(
            sequence=pl.col("symbol").str.join(""),
            steps=pl.len(),
            titrations=(
                pl.col("action").is_in(["escalate", "de-escalate"]).sum()
            ),
            failed_weans=(pl.col("state") == "failed wean").sum(),
        )
    )
    logger.info(
        f"sequences: median {sequences['steps'].median():.0f} steps, "
        f"median {sequences['titrations'].median():.0f} titrations, "
        f"{(sequences['titrations'] == 0).mean():.1%} of patients never titrated"
    )
    logger.info("the commonest sequences:")
    for row in sequences["sequence"].value_counts(sort=True).head(6).iter_rows(
        named=True
    ):
        logger.info(
            f"    {row['sequence'][:56]:<56} x{row['count']:,}"
        )
    logger.info("example sequences, one per decile of titration count:")
    ordered = sequences.sort("titrations")
    for q in range(0, 10, 2):
        index = min(
            int(q / 10 * ordered.height), max(ordered.height - 1, 0)
        )
        row = ordered.row(index, named=True)
        logger.info(
            f"    {row['titrations']:>3} titrations, "
            f"{row['failed_weans']:>2} failed weans: {row['sequence'][:48]}"
        )
    sequences.write_parquet(
        paths.patient_level_home
        / f"action_response_sequences_{args.subset}.parquet"
    )
    encoded.write_parquet(
        paths.patient_level_home / f"action_response_{args.subset}.parquet"
    )
    hourly.write_parquet(
        paths.patient_level_home / f"hourly_dose_map_{args.subset}.parquet"
    )

    if not rates.is_empty():
        sanity_check(encoded, hourly, rates, logger)
    logger.info("23 done")


if __name__ == "__main__":
    main()

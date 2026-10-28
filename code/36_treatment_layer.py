#!/usr/bin/env python3

"""
36_treatment_layer.py -- same treatment, diverging response

Two state layers: what the clinician is doing (norepinephrine band crossed with
the other agents) and what the patient is doing (the response states). This
crosses them and asks, within each treatment cell, whether patients diverge in
response by more than chance -- and whether that divergence predicts outcome.

    python code/36_treatment_layer.py --subset cardiogenic

WHY THE TREATMENT LAYER EXISTS. The response states are largely recoverable
from dose level alone -- norepinephrine dose separates "responsive" at AUC
0.856 and "wean held" at 0.854. So comparing responses across patients without
holding dose fixed compares treatment intensity with a response label on it.
Within a treatment cell that confound is gone by construction.

WHAT REMAINS CONFOUNDED. Patients are not assigned to treatment cells. A
clinician reaches "high norepinephrine plus vasopressin" by reading a patient
who was not responding, so cell membership is selected on the outcome the
response measures. Conditioning fixes the measurement, not the design.
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
from clifcs.regions import HOURLY_REGION_ORDER
from clifcs.treatment import label_treatment

ON_SUPPORT = tuple(r for r in HOURLY_REGION_ORDER if r != "off pressors")


def divergence_within_cell(frame, cfg, logger, seed=42, rounds=300):
    """
    inside one treatment cell, do patients differ in response composition

    The null gives every patient the cell's pooled response probabilities and
    their own hour count, so an excess cannot be explained by some patients
    being observed longer. This is the same multinomial test used on the
    event-based composition, now applied WITHIN a treatment stratum -- which is
    the version that cannot be explained by dose.
    """
    rng = np.random.default_rng(seed)
    floor = int(cfg.treatment.min_hours_per_cell)
    rows = []
    for cell in sorted(frame["treatment"].unique().to_list()):
        part = frame.filter(pl.col("treatment") == cell)
        if part.height < floor:
            continue
        counts = (
            part.group_by("hospitalization_id")
            .agg(
                hours=pl.len(),
                **{
                    f"n_{r.replace(' ', '_')}": (pl.col("region") == r).sum()
                    for r in ON_SUPPORT
                },
            )
            .filter(pl.col("hours") >= 5)
        )
        if counts.height < 100:
            continue
        matrix = np.column_stack([
            counts[f"n_{r.replace(' ', '_')}"].to_numpy() for r in ON_SUPPORT
        ]).astype(float)
        totals = counts["hours"].to_numpy().astype(float)
        pooled = matrix.sum(axis=0) / max(totals.sum(), 1.0)
        observed = np.array([
            float(np.var(matrix[:, i] / totals))
            for i in range(len(ON_SUPPORT))
        ])
        null = np.zeros((rounds, len(ON_SUPPORT)))
        for draw in range(rounds):
            simulated = np.array(
                [rng.multinomial(int(n), pooled) for n in totals], dtype=float
            )
            null[draw] = [
                float(np.var(simulated[:, i] / totals))
                for i in range(len(ON_SUPPORT))
            ]
        excess = observed - null.mean(axis=0)
        p = [(null[:, i] >= observed[i]).mean() for i in range(len(ON_SUPPORT))]
        real = [
            r for r, pv, ex in zip(ON_SUPPORT, p, excess)
            if pv < 0.05 and ex > 0
        ]
        rows.append({
            "treatment": cell,
            "hours": part.height,
            "patients": counts.height,
            "median_hours_per_patient": float(np.median(totals)),
            "states_diverging": len(real),
            "diverging": ", ".join(real),
            "max_excess_sd": float(np.sqrt(max(excess.max(), 0.0))),
        })
        logger.info(
            f"    {cell:<24} {part.height:>7,} hours  {counts.height:>5,} "
            f"patients  {len(real)}/{len(ON_SUPPORT)} states diverge  "
            f"max excess sd {np.sqrt(max(excess.max(), 0.0)):.3f}"
        )
    return pl.DataFrame(rows)


def outcome_within_cell(frame, labels, cfg, logger):
    """
    within a treatment cell, does response composition predict mortality

    The question Q1 was always after: among patients receiving comparable
    support, do those who respond differently also fare differently. Reported
    per cell, because pooling across cells reintroduces the dose confound the
    cells exist to remove.
    """
    floor = int(cfg.treatment.min_hours_per_cell)
    rows = []
    for cell in sorted(frame["treatment"].unique().to_list()):
        part = frame.filter(pl.col("treatment") == cell)
        if part.height < floor:
            continue
        counts = (
            part.group_by("hospitalization_id")
            .agg(
                hours=pl.len(),
                nonresponsive=(pl.col("region") == "nonresponsive").mean(),
                wean_held=(pl.col("region") == "wean held").mean(),
                deteriorating=(pl.col("region") == "deteriorating").mean(),
                drifting=(pl.col("region") == "drifting hold").mean(),
            )
            .filter(pl.col("hours") >= 5)
            .join(labels, on="hospitalization_id", how="inner")
            .drop_nulls("in_hospital_mortality")
        )
        if counts.height < 150:
            continue
        row = {
            "treatment": cell,
            "patients": counts.height,
            "mortality": float(counts["in_hospital_mortality"].mean()),
        }
        for measure in ("nonresponsive", "wean_held", "deteriorating",
                        "drifting"):
            # median split rather than tertiles: inside one cell the sample is
            # a few hundred, and a tertile of that is too thin for a stable
            # mortality estimate
            cut = float(counts[measure].median())
            high = counts.filter(pl.col(measure) > cut)
            low = counts.filter(pl.col(measure) <= cut)
            if high.height < 40 or low.height < 40:
                continue
            row[f"{measure}_high"] = float(high["in_hospital_mortality"].mean())
            row[f"{measure}_low"] = float(low["in_hospital_mortality"].mean())
            row[f"{measure}_diff"] = row[f"{measure}_high"] - row[f"{measure}_low"]
        rows.append(row)
        pieces = ", ".join(
            f"{m} {row[f'{m}_diff']:+.1%}"
            for m in ("nonresponsive", "wean_held", "deteriorating", "drifting")
            if f"{m}_diff" in row
        )
        logger.info(
            f"    {cell:<24} n={counts.height:>5,} mortality "
            f"{row['mortality']:>5.1%}   high-vs-low: {pieces}"
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "36_treatment_layer", paths.log_home / "36_treatment_layer.log"
    )
    cfg = OmegaConf.load(CONFIG_HOME / "dynamics.yaml")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    labelled = pl.read_parquet(
        paths.patient_level_home / f"hourly_regions_{args.subset}.parquet"
    )
    labelled = label_treatment(labelled, cfg)
    logger.info(
        f"{labelled.height:,} patient-hours, "
        f"{labelled['treatment'].n_unique()} treatment cells populated"
    )

    # ---- the treatment layer on its own
    rows = []
    for cell in sorted(labelled["treatment"].unique().to_list()):
        part = labelled.filter(pl.col("treatment") == cell)
        rows.append({
            "treatment": cell,
            "hours": part.height,
            "share": part.height / labelled.height,
            "patients": part["hospitalization_id"].n_unique(),
            "median_ne": float(part["dose_norepinephrine"].median()),
            "median_map": float(part["map"].median()),
        })
    treatment_table = pl.DataFrame(rows).sort("hours", descending=True)
    treatment_table.write_csv(
        tables / f"treatment_cells_{args.subset}.csv"
    )
    logger.info("treatment cells:")
    for row in treatment_table.iter_rows(named=True):
        logger.info(
            f"    {row['treatment']:<24} {row['hours']:>8,} hours "
            f"{row['share']:>6.1%}  {row['patients']:>6,} patients  "
            f"median NE {row['median_ne']:.3f}  MAP {row['median_map']:.0f}"
        )

    # ---- response composition crossed with treatment
    on_support = labelled.filter(pl.col("region") != "off pressors")
    cross = (
        on_support.group_by("treatment", "region").len()
        .join(
            on_support.group_by("treatment").agg(total=pl.len()),
            on="treatment",
        )
        .with_columns(share=pl.col("len") / pl.col("total"))
        .sort(["treatment", "share"], descending=[False, True])
    )
    cross.write_csv(tables / f"treatment_by_response_{args.subset}.csv")

    logger.info("within each treatment cell, do patients diverge in response?")
    divergence = divergence_within_cell(
        on_support, cfg, logger, seed=args.seed
    )
    if not divergence.is_empty():
        divergence.write_csv(
            tables / f"treatment_divergence_{args.subset}.csv"
        )

    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet").select(
        "hospitalization_id", "in_hospital_mortality"
    )
    logger.info("within each cell, does response composition track mortality?")
    outcomes = outcome_within_cell(on_support, cohort, cfg, logger)
    if not outcomes.is_empty():
        outcomes.write_csv(
            tables / f"treatment_outcome_{args.subset}.csv"
        )
    labelled.write_parquet(
        paths.patient_level_home / f"treatment_states_{args.subset}.parquet"
    )
    logger.info("36 done")


if __name__ == "__main__":
    main()

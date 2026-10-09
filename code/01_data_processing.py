#!/usr/bin/env python3

"""
01 -- data processing: cardiogenic shock cohort -> model-ready token streams

Runs six stages per site (UCMC, NU). Each writes its outputs and can be rerun
on its own with --stage, which matters because `derive` and `collate` are the
slow ones.

  cohort    identify cardiogenic shock, print the filter ledger, attach
            outcomes and Zweck phenotypes       -> output/cohort/
  subset    cut each site's CLIF tables down to the cohort   -> scratch/data-raw/
  derive    clifpy: respiratory-support waterfall, dose conversion, SOFA
  collate   cocoa collate                                   -> scratch/processed/
  tokenize  cocoa tokenize, under the model's frozen tokenizer
  winnow    cut each timeline at both anchors (24h, onset)

Usage
    python code/01_data_processing.py                      # all stages
    python code/01_data_processing.py --stage cohort       # just the cohort
    python code/01_data_processing.py --site NU --force    # redo one site
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import polars as pl
from omegaconf import OmegaConf

from clifcs import ANCHORS, get_logger, load_cohort_cfg, load_paths
from clifcs.clif import HOSPITALIZATION_LEVEL, PATIENT_LEVEL
from clifcs.cohort import (
    FilterLedger,
    attach_demographics,
    base_cohort,
    cardiac_origin,
    end_organ_criterion,
    hemodynamic_criterion,
)
from clifcs.config import CONFIG_HOME
from clifcs.onset_winnower import OnsetWinnower
from clifcs.outcomes import add_outcomes
from clifcs.phenotype import (
    add_egfr,
    apply_phenotype_model,
    assign_phenotypes,
    nearest_labs,
    save_phenotype_model,
    summarize,
)

STAGES = ("cohort", "subset", "derive", "collate", "tokenize", "winnow")

#: tables cocoa's collation.yaml reads, plus what clifpy needs to derive them
REQUIRED_TABLES = (
    "clif_hospitalization",
    "clif_patient",
    "clif_adt",
    "clif_code_status",
    "clif_crrt_therapy",
    "clif_labs",
    "clif_vitals",
    "clif_medication_admin_continuous",
    "clif_medication_admin_intermittent",
    "clif_patient_assessments",
    "clif_position",
    "clif_respiratory_support",
    "clif_patient_procedures",
)


# ============================================================ stage: cohort


def stage_cohort(paths, cfg, logger, phenotype_model=None) -> pl.DataFrame:
    """identify cardiogenic shock at each site and assemble the cohort table"""
    paths.cohort_home.mkdir(parents=True, exist_ok=True)
    ledger = FilterLedger()
    per_site = []

    for site in paths.sites:
        data_dir = paths.site_data[site]
        logger.info(f"=== {site}: {data_dir}")

        base = base_cohort(site, data_dir, cfg, ledger, logger)
        hemo = hemodynamic_criterion(site, data_dir, base, cfg, ledger, logger)
        shock = end_organ_criterion(site, data_dir, hemo, cfg, ledger, logger)
        shock = cardiac_origin(site, data_dir, shock, cfg, ledger, logger)

        shock = attach_demographics(data_dir, shock)
        shock = add_outcomes(data_dir, shock, cfg)
        labs = nearest_labs(data_dir, shock, cfg)
        shock = shock.join(labs, on="hospitalization_id", how="left")
        shock = add_egfr(shock)
        per_site.append(shock.with_columns(site=pl.lit(site)))

    # Sorted before clustering, not for tidiness: k-means++ seeding samples from
    # the rows in order, so an upstream join that reorders them shifts which
    # local optimum is reached even with a fixed random_state. A canonical order
    # makes the phenotype assignment reproducible.
    cohort = pl.concat(per_site, how="diagonal_relaxed").sort(
        ["site", "hospitalization_id"]
    )

    # Phenotypes are fit on the pooled cohort so every site lands on a shared
    # set of clusters. A site that cannot pool (a remote institution running
    # this on its own data) instead applies a model fitted elsewhere, so that
    # "phenotype II" means the same thing in every contributing cohort.
    centroids = None
    if phenotype_model:
        cohort = apply_phenotype_model(cohort, phenotype_model, logger)
    else:
        cohort, centroids, model = assign_phenotypes(cohort, cfg)
        save_phenotype_model(paths.cohort_home / "phenotype_model.json", **model)
    phenotype_summary = summarize(cohort, logger)

    paths.cohort_home.mkdir(parents=True, exist_ok=True)
    cohort.write_parquet(paths.cohort_home / "cohort.parquet")
    ledger_frame = ledger.to_frame()
    ledger_frame.write_csv(paths.cohort_home / "filter_ledger.csv")
    if centroids is not None:
        centroids.write_csv(paths.cohort_home / "phenotype_centroids.csv")
    phenotype_summary.write_csv(paths.cohort_home / "phenotype_summary.csv")

    # the onset table the onset-anchored winnower reads
    cohort.select("hospitalization_id", "patient_id", "site", "cs_onset_dttm").write_parquet(
        paths.cohort_home / "cs_onset.parquet"
    )

    report_ledger(ledger_frame, paths.sites, logger)
    logger.info(f"wrote cohort ({cohort.height} patients) to {paths.cohort_home}")
    return cohort


def report_ledger(ledger: pl.DataFrame, sites, logger) -> None:
    """print the attrition table the way it is meant to be read in the write-up"""
    logger.info("")
    logger.info("=" * 78)
    logger.info("COHORT ATTRITION (hospitalizations / patients)")
    logger.info("=" * 78)
    for site in sites:
        logger.info(f"-- {site}")
        for row in ledger.filter(pl.col("site") == site).iter_rows(named=True):
            note = f"   [{row['note']}]" if row["note"] else ""
            logger.info(
                f"   {row['step']:<58} "
                f"{row['n_hospitalizations']:>8} / {row['n_patients']:>8}{note}"
            )
    logger.info("=" * 78)


# ============================================================ stage: subset


def stage_subset(paths, cohort: pl.DataFrame, sites, logger, force: bool) -> None:
    """
    write a cohort-restricted copy of each site's CLIF tables

    Mirrors coreopsis recipes/preprocessing.py: hospitalization-level tables are
    filtered on hospitalization_id, patient-level tables on patient_id, and
    anything keyed on neither is copied through. Cutting the data down first is
    what makes the clifpy and cocoa stages tractable -- clif_labs alone is 1 GB
    at UCMC, and the cohort is a small fraction of it.
    """
    for site in sites:
        out_dir = paths.raw_subset(site)
        if out_dir.exists() and not force:
            logger.info(f"[{site}] subset exists, skipping (use --force to redo)")
            continue
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        keep = cohort.filter(pl.col("site") == site)
        hosp_ids = keep.select("hospitalization_id").lazy()
        pt_ids = keep.select("patient_id").unique().lazy()
        logger.info(f"[{site}] subsetting to {keep.height} hospitalizations -> {out_dir}")

        for table in REQUIRED_TABLES:
            src = paths.site_data[site] / f"{table}.parquet"
            if not src.is_file():
                logger.warning(f"[{site}] {table}.parquet absent; skipped")
                continue
            frame = pl.scan_parquet(src).drop("__index_level_0__", strict=False)
            names = frame.collect_schema().names()

            if table in HOSPITALIZATION_LEVEL and "hospitalization_id" in names:
                frame = frame.cast({"hospitalization_id": pl.String}).join(
                    hosp_ids, on="hospitalization_id", how="inner"
                )
            elif table in PATIENT_LEVEL and "patient_id" in names:
                frame = frame.cast({"patient_id": pl.String}).join(
                    pt_ids, on="patient_id", how="inner"
                )
            else:
                shutil.copy2(src, out_dir / src.name)
                logger.info(f"[{site}]   {table}: copied (no cohort key)")
                continue

            frame.sink_parquet(out_dir / f"{table}.parquet")
            n = pl.scan_parquet(out_dir / f"{table}.parquet").select(pl.len()).collect().item()
            logger.info(f"[{site}]   {table}: {n} rows")


# ============================================================ stage: derive


def stage_derive(paths, sites, logger, force: bool) -> None:
    """
    build the derived tables cocoa's collation.yaml expects

    Reproduces coreopsis recipes/run_clifpy.py then recipes/run_sofa_scoring.py,
    in that order: the first saves the converted medication tables, the second
    re-converts only the four SOFA vasopressors in memory without overwriting
    them.
    """
    import clifpy as cpy
    from clifpy.tables.respiratory_support import RespiratorySupport

    units = OmegaConf.load(CONFIG_HOME / "clifpy_units.yaml")

    targets = (
        "clif_respiratory_support_processed",
        "clif_medication_admin_continuous_converted",
        "clif_medication_admin_intermittent_converted",
        "clif_sofa",
    )

    for site in sites:
        data_dir = paths.raw_subset(site)
        if all((data_dir / f"{t}.parquet").is_file() for t in targets) and not force:
            logger.info(f"[{site}] derived tables exist, skipping")
            continue

        logger.info(f"[{site}] clifpy: respiratory-support waterfall")
        resp = RespiratorySupport.from_file(
            data_directory=str(data_dir), filetype="parquet", timezone="UTC"
        )
        processed = resp.waterfall()
        processed.df.to_parquet(data_dir / "clif_respiratory_support_processed.parquet")

        logger.info(f"[{site}] clifpy: dose conversion")
        orchestrator = cpy.ClifOrchestrator(
            data_directory=str(data_dir),
            filetype="parquet",
            timezone="UTC",
            output_directory=str(paths.log_home / f"clifpy-{site.lower()}"),
        )
        orchestrator.convert_dose_units_for_continuous_meds(
            preferred_units=OmegaConf.to_container(units.continuous),
            show_intermediate=True,
            override=True,
            save_to_table=True,
        )
        orchestrator.medication_admin_continuous.df_converted.to_parquet(
            data_dir / "clif_medication_admin_continuous_converted.parquet"
        )
        orchestrator.convert_dose_units_for_intermittent_meds(
            preferred_units=OmegaConf.to_container(units.intermittent),
            show_intermediate=True,
            override=True,
            save_to_table=True,
        )
        orchestrator.medication_admin_intermittent.df_converted.to_parquet(
            data_dir / "clif_medication_admin_intermittent_converted.parquet"
        )

        logger.info(f"[{site}] clifpy: SOFA scoring")
        sofa = cpy.ClifOrchestrator(
            data_directory=str(data_dir),
            filetype="parquet",
            timezone="UTC",
            output_directory=str(paths.log_home / f"clifpy-{site.lower()}"),
        )
        sofa.convert_dose_units_for_continuous_meds(
            preferred_units=OmegaConf.to_container(units.sofa_continuous),
            override=True,
        )
        sofa.compute_sofa_scores()
        sofa_df = (
            sofa.wide_df.sort_values("event_time")
            .groupby("encounter_block")
            .agg({"event_time": "max", "hospitalization_id": "last"})
            .join(sofa.sofa_df, how="inner", on="encounter_block", validate="1:1")
        )
        sofa_df.to_parquet(data_dir / "clif_sofa.parquet")
        logger.info(f"[{site}] derived tables written to {data_dir}")


# =================================================== stages: collate/tokenize


def stage_collate(paths, sites, logger, force: bool) -> None:
    """cocoa collate: denormalize the CLIF tables into a MEDS-like event stream"""
    from cocoa.collator import Collator

    for site in sites:
        out_dir = paths.processed(site)
        if (out_dir / "meds.parquet").is_file() and not force:
            logger.info(f"[{site}] meds.parquet exists, skipping collate")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{site}] collating -> {out_dir}")
        t0 = time.perf_counter()
        Collator(
            collation_cfg=CONFIG_HOME / "collation.yaml",
            raw_data_home=paths.raw_subset(site),
            processed_data_home=out_dir,
        ).save_all()
        logger.info(f"[{site}] collate done in {time.perf_counter() - t0:.1f}s")


def stage_tokenize(paths, sites, logger, force: bool) -> None:
    """
    cocoa tokenize under the model's own frozen tokenizer

    Loading the training-time tokenizer (rather than learning a new one) is the
    whole point: the model's 1344-row embedding table is only meaningful under
    that exact token -> id mapping, and the decile bin edges have to be the ones
    the numeric values were binned with during training. Codes the training data
    never contained map to UNK (id 0).
    """
    from cocoa.tokenizer import Tokenizer

    for site in sites:
        out_dir = paths.processed(site)
        if (out_dir / "tokens_times.parquet").is_file() and not force:
            logger.info(f"[{site}] tokens_times.parquet exists, skipping tokenize")
            continue
        logger.info(f"[{site}] tokenizing with frozen {paths.tokenizer_yaml.name}")
        tokenizer = Tokenizer(
            tokenization_cfg=CONFIG_HOME / "tokenization.yaml",
            processed_data_home=out_dir,
        ).load(paths.tokenizer_yaml, done_training=True)
        if tokenizer.is_training:
            raise RuntimeError(
                "tokenizer is still in training mode; it would relearn the "
                "vocabulary and invalidate the model's embeddings"
            )
        tokenizer.save_all()
        logger.info(f"[{site}] vocabulary size {len(tokenizer)}")


# ============================================================ stage: winnow


def stage_winnow(paths, sites, logger, force: bool) -> None:
    """
    cut each timeline at both anchors

    Each anchor gets its own directory holding the three files cocoa and cotorra
    look for by name (tokenizer.yaml, subject_splits.parquet,
    tokens_times.parquet) plus that anchor's *_for_inference.parquet, so the two
    anchors can be extracted independently without either overwriting the other.
    """
    from cocoa.winnower import Winnower

    shared = ("tokenizer.yaml", "subject_splits.parquet", "tokens_times.parquet")

    for site in sites:
        src = paths.processed(site)
        for anchor in ANCHORS:
            dest = paths.anchored(site, anchor)
            done = all(
                (dest / f"{split}_for_inference.parquet").is_file()
                for split in ("train", "tuning", "held_out")
            )
            if done and not force:
                logger.info(f"[{site}/{anchor}] already winnowed, skipping")
                continue

            dest.mkdir(parents=True, exist_ok=True)
            for name in shared:
                shutil.copy2(src / name, dest / name)

            if anchor == "24h":
                logger.info(f"[{site}/{anchor}] winnowing at 24h from admission")
                winnower = Winnower(
                    winnowing_cfg=CONFIG_HOME / "winnowing_24h.yaml",
                    processed_data_home=dest,
                )
            else:
                logger.info(f"[{site}/{anchor}] winnowing at cardiogenic shock onset")
                winnower = OnsetWinnower(
                    winnowing_cfg=CONFIG_HOME / "winnowing_onset.yaml",
                    processed_data_home=dest,
                    onset_parquet=paths.cohort_home / "cs_onset.parquet",
                )
            winnower.save_all()

            kept = sum(
                pl.scan_parquet(dest / f"{split}_for_inference.parquet")
                .select(pl.len())
                .collect()
                .item()
                for split in ("train", "tuning", "held_out")
            )
            total = (
                pl.scan_parquet(dest / "tokens_times.parquet")
                .select(pl.len())
                .collect()
                .item()
            )
            logger.info(
                f"[{site}/{anchor}] {kept}/{total} timelines reach the anchor"
            )


# =================================================================== driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("all", *STAGES), default="all", help="run a single stage"
    )
    parser.add_argument(
        "--from-stage", choices=STAGES, default=None, help="run this stage onward"
    )
    parser.add_argument(
        "--site",
        default=None,
        help="restrict the per-site stages to one site; the cohort stage always "
        "runs both, since the phenotypes are fit on the pooled cohort",
    )
    parser.add_argument(
        "--force", action="store_true", help="redo stages whose outputs already exist"
    )
    parser.add_argument(
        "--phenotype-model",
        default=None,
        help="apply a phenotype model fitted elsewhere (cohort/phenotype_model.json) "
        "instead of refitting; required at a site that cannot pool its labs with "
        "the derivation cohort",
    )
    args = parser.parse_args()

    paths = load_paths()
    paths.mkdirs()
    cfg = load_cohort_cfg()
    logger = get_logger("01_data_processing", paths.log_home / "01_data_processing.log")

    if args.from_stage:
        todo = STAGES[STAGES.index(args.from_stage) :]
    elif args.stage == "all":
        todo = STAGES
    else:
        todo = (args.stage,)
    if args.site and args.site not in paths.sites:
        parser.error(f"--site {args.site} is not configured; have {paths.sites}")
    sites = (args.site,) if args.site else paths.sites
    logger.info(f"stages: {', '.join(todo)} | sites: {', '.join(sites)}")

    cohort_file = paths.cohort_home / "cohort.parquet"
    cohort = None

    if "cohort" in todo:
        cohort = stage_cohort(paths, cfg, logger, args.phenotype_model)
    if "subset" in todo:
        if cohort is None:
            if not cohort_file.is_file():
                parser.error(f"{cohort_file} not found; run --stage cohort first")
            cohort = pl.read_parquet(cohort_file)
        stage_subset(paths, cohort, sites, logger, args.force)
    if "derive" in todo:
        stage_derive(paths, sites, logger, args.force)
    if "collate" in todo:
        stage_collate(paths, sites, logger, args.force)
    if "tokenize" in todo:
        stage_tokenize(paths, sites, logger, args.force)
    if "winnow" in todo:
        stage_winnow(paths, sites, logger, args.force)

    logger.info("01 done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
01 -- build this site's cohort and write the tables it sends back

Runs on one machine, against this site's own CLIF-2.1 extract. Reads the raw
tables, identifies the shock cohort, labels it, and writes a directory of
summary tables to `output/upload_to_box_<SITE>/`.

Nothing it writes to that directory is patient-level. Every number is a count,
a share or a median over at least `--min-cell` patients.

Ten stages. Each is rerunnable on its own, which matters because `derive` and
`collate` are slow:

  cohort       identify shock, print the filter ledger, attach outcomes, coded
               shock type, and the Zweck, Seymour and Bhavani phenotypes
  subset       cut this site's CLIF tables down to the cohort
  derive       respiratory-support waterfall, dose conversion, SOFA
  collate      assemble one timeline per patient
  tokenize     encode each timeline
  winnow       cut each timeline at both anchors (24h, onset)
  tokencounts  per-token and per-family prevalence
  baseline     median physiology and shares by coded mechanism
  trials       eligibility, position against published IQRs, guideline fit
  export       assemble the directory to send

Usage

    python code/01_site_tables.py --check        # verify setup, run nothing
    python code/01_site_tables.py                # all ten stages
    python code/01_site_tables.py --from-stage collate   # resume after a failure
    python code/01_site_tables.py --stage cohort --force # redo one stage

`--check` reads configuration and lists directories. It opens no data file and
is safe to run first, anywhere.

Every stage prints when it starts, when it finishes, and how long it took. If
one fails, the run stops and prints the stage that failed, the line it failed
on, and the command to resume from there.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import shutil
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import polars as pl
from omegaconf import OmegaConf

from clifcs import ANCHORS, get_logger, load_cohort_cfg, load_paths
from clifcs.clif import (
    HOSPITALIZATION_LEVEL,
    PATIENT_LEVEL,
    duckdb_timezone,
    naive_datetime_columns,
)
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
from clifcs.shock_type import shock_type_flags
from clifcs.shock_type import summarize as summarize_shock_type
from clifcs import sepsis_phenotype
from clifcs.phenotype import (
    add_egfr,
    apply_phenotype_model,
    PHENOTYPE_VARIANTS,
    assign_phenotype_variants,
    assign_phenotypes,
    variant_agreement,
    variant_model_path,
    nearest_labs,
    save_phenotype_model,
    summarize,
)

STAGES = (
    "cohort",
    # cheap, and depends only on cohort.parquet
    "phenotypes",
    "subset",
    "derive",
    "collate",
    "tokenize",
    "winnow",
    # the token stream is tokenizer-only, so the one comparison that needs no
    # model at all -- how differently the sites chart -- belongs here.
    # 04 draws its tornado from this table, which is why that figure now
    # survives for a site that shares nothing patient-level.
    "tokencounts",
    # who these patients are, split by the coded mechanism. Reads raw CLIF for
    # the physiology at onset, so it has to run site-side; everything it emits
    # is a median or a share.
    "baseline",
    "trials",
    "export",
)

#: the stages that need the training-time tokenizer, and nothing else does.
#:
#: Skipped with a warning when `paths.has_tokenizer` is false, rather than
#: failing the run. Everything 01 reports besides token prevalence -- the
#: cohort, the three phenotype breakdowns, the baseline tables by coded
#: mechanism, and the whole trial and guideline analysis -- is computed from
#: CLIF and the published literature and never sees a token. A site that cannot
#: get the tokenizer should still send all of that; requiring it meant a site
#: that could have contributed almost everything contributed nothing.
#:
#: They are skipped as a GROUP because they chain: `winnow` copies the
#: `tokenizer.yaml` and `tokens_times.parquet` that `tokenize` wrote into each
#: anchor directory, and `tokencounts` reads the winnowed stream. Running
#: `winnow` without `tokenize` fails on a missing file rather than degrading.
TOKEN_STAGES = ("tokenize", "winnow", "tokencounts")

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
    # read by clifcs.shock_type for the coded mechanism. Not collated by cocoa,
    # but kept in the subset so the cohort stage can be rerun against it
    # without going back to the full extract.
    "clif_hospital_diagnosis",
)


#: absent, these do not stop the run -- they quietly change what it reports, so
#: each gets a line of its own saying what goes wrong
DEGRADES_QUIETLY: dict[str, str] = {
    "clif_hospital_diagnosis": (
        "every diagnosis-based criterion reads as not met, so coded shock type "
        "and trial eligibility are understated rather than errored"
    ),
    "clif_patient_procedures": (
        "IABP, ECMO and microaxial flags all read false, so device concordance "
        "is silently wrong rather than absent"
    ),
    "clif_crrt_therapy": "renal support is invisible",
    "clif_code_status": "limitation-of-treatment context is lost",
}


def preflight(paths, sites, logger) -> bool:
    """
    check this machine can do the run, before anything expensive starts

    Reads configuration, lists directories and reads table schemas, never
    rows, so it is safe anywhere and takes about a second. Returns True if the run can proceed.

    Runs automatically at the start of every invocation, and on its own with
    `--check`. Automatically, because the two failures it catches -- a path
    that does not resolve and a table that is absent -- otherwise surface
    hours in, somewhere in the middle of `derive`, as an error about a column.
    """
    ok = True

    def line(label, value, good, note=""):
        nonlocal ok
        if not good:
            ok = False
        # failures at ERROR, so they survive a log filtered to warnings and
        # above -- which is how anyone reads a log they were sent
        emit = logger.info if good else logger.error
        emit(f"  [{'ok  ' if good else 'FAIL'}] {label:<16} {value}")
        if note:
            emit(f"         {note}")

    logger.info("--- checking this machine ---")
    logger.info(f"  python           {sys.version.split()[0]}")
    logger.info(f"  platform         {platform.system()} {platform.machine()}")

    logger.info("sites")
    for site in sites:
        directory = paths.site_data[site]
        if not directory.is_dir():
            line(site, directory, False, "directory not found")
            continue
        present = {f.stem for f in directory.glob("clif_*.parquet")}
        missing = [t for t in REQUIRED_TABLES if t not in present]
        line(
            site,
            directory,
            not missing,
            f"all {len(REQUIRED_TABLES)} required tables present "
            f"({len(present)} clif_* files in the directory)"
            if not missing
            else f"MISSING {len(missing)} of {len(REQUIRED_TABLES)} required tables",
        )
        for table in missing:
            logger.info(f"           - {table}.parquet")
            note = DEGRADES_QUIETLY.get(table)
            if note:
                logger.info(f"               {note}")
        # A count is not the check. Sites ship different extras, so the union is
        # larger than the required set at every site and matching on a count
        # would pass a directory missing a required table and holding two it
        # never reads.
        extra = sorted(present - set(REQUIRED_TABLES))
        if extra:
            logger.info(f"         also present, never read: {', '.join(extra)}")
        # a warning, not a failure: a site may store UTC without a zone, as
        # UCMC's respiratory_support does
        for table in sorted(present & set(REQUIRED_TABLES)):
            naive = naive_datetime_columns(directory, table)
            if naive:
                logger.warning(
                    f"         {table}: {', '.join(naive)} stored without a timezone; "
                    "read as UTC. If these are local times, convert them to UTC first."
                )

    logger.info("where this run will write")
    for label, target in (
        ("output_home", paths.output_home),
        ("scratch_home", paths.scratch_home),
    ):
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".write-probe"
            probe.touch()
            probe.unlink()
            line(label, target, True)
        except Exception as exc:
            line(label, target, False, f"not writable: {exc}")

    if not ok:
        logger.error("--- check FAILED; fix the lines marked FAIL above ---")
        logger.error(
            "    paths come from code/config/paths.yaml. Copy "
            "code/config/paths.example.yaml to it if you have not."
        )
    else:
        logger.info("--- check passed ---")
    return ok

#: one line per stage, printed when it starts. A reader watching a long run
#: should be able to tell what is happening without opening this file.
STAGE_PURPOSE: dict[str, str] = {
    "cohort": "identifying the shock cohort and attaching labels",
    "subset": "cutting the CLIF tables down to the cohort",
    "derive": "deriving respiratory support, dose equivalents and SOFA (slow)",
    "collate": "assembling one timeline per patient (slow)",
    "tokenize": "encoding each timeline",
    "winnow": "cutting each timeline at the 24h and onset anchors",
    "tokencounts": "counting token prevalence",
    "baseline": "summarising physiology by coded mechanism",
    "trials": "scoring trial eligibility, IQR position and guideline fit",
    "export": "assembling the directory to send",
}

#: what a reader should look at first when a given stage fails. Keyed by stage,
#: appended to the failure report -- a traceback says where the code broke, and
#: this says where the *cause* usually is, which is rarely the same place.
STAGE_HINT: dict[str, str] = {
    "cohort": (
        "check that clif_hospitalization, clif_adt, clif_vitals and "
        "clif_labs are populated and not just present. Run --check."
    ),
    "subset": (
        "usually a disk or permissions problem under scratch_home, or a CLIF "
        "table with an unexpected column type."
    ),
    "derive": (
        "the most memory-hungry stage. If the process was killed rather than "
        "raising, it ran out of memory -- rerun with more, or one site at a "
        "time with --site."
    ),
    "collate": (
        "also memory-hungry, and it reads what `derive` wrote: if `derive` "
        "half-finished, rerun it with --stage derive --force first."
    ),
    "tokenize": (
        "reads what `collate` wrote. Rerun --stage collate --force if its "
        "output looks truncated."
    ),
    "winnow": (
        "reads what `tokenize` wrote into scratch. Rerun --stage tokenize "
        "--force first."
    ),
    "tokencounts": "reads the winnowed timelines; rerun --stage winnow --force.",
    "baseline": (
        "reads raw CLIF for physiology at onset, so a missing or empty "
        "clif_vitals or clif_labs shows up here."
    ),
    "trials": (
        "reads raw CLIF plus what `derive` wrote. A missing "
        "clif_hospital_diagnosis or clif_patient_procedures degrades this "
        "stage rather than failing it, so a hard failure here is usually a "
        "column type."
    ),
    "export": (
        "every table is re-checked against the minimum cell size before it is "
        "written. A failure naming a column and a cell count means a table "
        "reached export unsuppressed -- that is a bug, not a configuration "
        "problem; report it with the column name."
    ),
}


class StageFailed(Exception):
    """a stage raised; carries what is needed to report it usefully"""

    def __init__(self, stage: str, cause: BaseException):
        super().__init__(f"stage {stage!r} failed: {cause}")
        self.stage = stage
        self.cause = cause


def _origin(cause: BaseException) -> str:
    """
    the deepest frame inside this repository, as file:line in function

    Not the deepest frame overall: that is usually inside polars or pandas and
    says nothing actionable. The last frame under `code/` is the line in this
    pipeline that led there, which is the one worth printing.
    """
    here = pathlib.Path(__file__).resolve().parent
    frames = [
        frame
        for frame in traceback.extract_tb(cause.__traceback__)
        if (
            here in pathlib.Path(frame.filename).resolve().parents
            or pathlib.Path(frame.filename).resolve().parent == here
        )
        # `run_stage` is in this file and is always in the chain, so without
        # excluding it a failure that happened entirely inside a dependency
        # would be reported at the line that called the stage -- which looks
        # like an answer and is not one.
        and frame.name != "run_stage"
    ]
    if not frames:
        return "no frame inside code/ -- the failure is in a dependency"
    frame = frames[-1]
    rel = pathlib.Path(frame.filename).resolve().relative_to(here.parent)
    return f"{rel.as_posix()}:{frame.lineno} in {frame.name}()"


def run_stage(name, position, total, logger, fn, *args, **kwargs):
    """
    run one stage, announcing it and timing it, and label any failure with it

    The announcement is the checkpoint: a run that dies leaves a log whose last
    `started` line names the stage, so the resume point is readable without a
    traceback. The timing is there because `derive` and `collate` take long
    enough that a reader needs to know whether a quiet run is working or hung.
    """
    logger.info(f"=== [{position}/{total}] {name}: {STAGE_PURPOSE.get(name, '')}")
    started = time.monotonic()
    try:
        result = fn(*args, **kwargs)
    except Exception as cause:
        raise StageFailed(name, cause) from cause
    logger.info(
        f"=== [{position}/{total}] {name} finished in "
        f"{time.monotonic() - started:,.0f}s"
    )
    return result


def report_failure(failure: StageFailed, logger) -> None:
    """print what failed, where, and how to pick up from there"""
    stage = failure.stage
    cause = failure.cause
    logger.error("=" * 70)
    logger.error(f"STAGE FAILED: {stage} -- {STAGE_PURPOSE.get(stage, '')}")
    logger.error("")
    logger.error(f"  {type(cause).__name__}: {cause}")
    logger.error(f"  raised at   {_origin(cause)}")
    hint = STAGE_HINT.get(stage)
    if hint:
        logger.error(f"  look at     {hint}")
    logger.error("")
    logger.error("  resume from this stage once it is fixed:")
    logger.error(f"      python code/01_site_tables.py --from-stage {stage}")
    logger.error("")
    logger.error("  earlier stages are not redone unless you pass --force.")
    logger.error("=" * 70)
    logger.error("full traceback follows", exc_info=cause)


# ============================================================ stage: cohort


def stage_cohort(
    paths,
    cfg,
    logger,
    phenotype_model=None,
    sepsis_cfg=None,
    sepsis_models: dict | None = None,
) -> pl.DataFrame:
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
        # what the coders said the mechanism was. A labelling, never a filter --
        # the cohort is selected on physiology and stays mechanism-agnostic.
        shock = shock.join(
            shock_type_flags(data_dir, shock, cfg, logger),
            on="hospitalization_id",
            how="left",
            validate="1:1",
        )
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
    #
    # Fitted once per cardiogenic denominator, not once over everyone. Zweck
    # derived I/II/III in cardiogenic shock; fitting on the full cohort
    # clusters a population that is 75% non-cardiogenic and then names the
    # clusters after his. See PHENOTYPE_VARIANTS.
    cohort, centroid_tables, models = assign_phenotype_variants(
        cohort, cfg, logger, model_base=phenotype_model
    )
    for column, model in models.items():
        save_phenotype_model(
            variant_model_path(
                paths.cohort_home / "phenotype_model.json", column
            ),
            **model,
        )
    for column, table in centroid_tables.items():
        table.write_csv(paths.cohort_home / f"phenotype_centroids_{column}.csv")
    centroids = centroid_tables.get("cs_phenotype")
    phenotype_summary = summarize(cohort, logger)
    shock_type_summary = summarize_shock_type(cohort, logger)

    # The two sepsis phenotype systems, applied to the coded-septic subset.
    # Fitted on the pooled subset for the same reason the Zweck phenotypes are,
    # and applied from a saved model at a contributing site.
    cohort, sepsis_tables, sepsis_fitted = sepsis_phenotype.attach(
        paths, cohort, sepsis_cfg, logger, models=sepsis_models
    )
    for name, table in sepsis_tables.items():
        table.write_csv(paths.cohort_home / f"{name}.csv")
    for scheme, model in sepsis_fitted.items():
        sepsis_phenotype.save_model(
            paths.cohort_home / f"{scheme}_model.json", model, scheme
        )

    paths.cohort_home.mkdir(parents=True, exist_ok=True)
    cohort.write_parquet(paths.cohort_home / "cohort.parquet")
    ledger_frame = ledger.to_frame()
    ledger_frame.write_csv(paths.cohort_home / "filter_ledger.csv")
    if centroids is not None:
        centroids.write_csv(paths.cohort_home / "phenotype_centroids.csv")
    phenotype_summary.write_csv(paths.cohort_home / "phenotype_summary.csv")
    shock_type_summary.write_csv(paths.cohort_home / "shock_type_summary.csv")

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


def sofa_by_hospitalization(wide, scores):
    """
    one row per encounter block: its last hospitalization and event time, with
    that block's own SOFA row from clifpy

    A merge on the `encounter_block` column. The previous `.join(on=...)`
    matched the grouped block numbers against `scores`' row positions, so
    every patient carried another block's score and `validate="1:1"` still
    passed.
    """
    per_block = (
        wide.sort_values("event_time")
        .groupby("encounter_block", as_index=False)
        .agg({"event_time": "max", "hospitalization_id": "last"})
    )
    return per_block.merge(scores, on="encounter_block", how="inner", validate="1:1")


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
        sofa_df = sofa_by_hospitalization(sofa.wide_df, sofa.sofa_df)
        logger.info(
            f"[{site}] SOFA rows for {len(sofa_df)} of "
            f"{sofa.wide_df['encounter_block'].nunique()} encounter blocks"
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
    encode each timeline under the shared frozen encoding

    Applying the shared encoding rather than learning a local one is the whole
    point: the decile bin edges have to be the ones every site is using, or the
    same clinical event gets a different id here than elsewhere and the
    prevalence tables stop being comparable. Codes the encoding does not contain
    map to UNK (id 0).
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
                "vocabulary, which would make the token ids local to this "
                "site and not comparable with any other"
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


# ==================================================== stage: token prevalence


def stage_tokencounts(paths, sites, logger, min_cell: int) -> None:
    """
    how often each token appears at this site, and in how many patients

    The one cross-site comparison that needs no model: a token is a decile bin
    fitted at training time, so the bin edges are identical at every site by
    construction and the counts are directly comparable without anyone agreeing
    on anything.

    This is what 04's tornado reads. It used to be computed at the coordinating
    site from the shipped token sequences, which meant the figure existed only
    for sites that had sent them -- and 04 skipped it rather than drawing a
    degraded version. Computing it here instead costs one pass over the stream
    and makes the comparison available for every site, including one that
    shares nothing patient-level at all. The site effect is the largest single
    finding in this pipeline (97% neighbour purity), so being able to decompose
    it into "because they chart differently" should not have been contingent on
    a data use agreement.

    Two tables per site and anchor:

      token_prevalence  one row per vocabulary entry: how many patients emitted
                        it at least once, and how many times it occurred in all
      token_families    the same, aggregated to the ~23 families

    `n_patients` is a patient count and is suppressed. `total_occurrences` is
    suppressed with it -- it is not a patient count, but a token occurring four
    times in one patient is as identifying as the patient count that would have
    been blanked, and publishing it beside a blanked cell would return it.
    """
    from clifcs.export import contract
    from clifcs.tokens import load_pre_anchor_tokens, vocabulary

    names, families = vocabulary(str(paths.tokenizer_yaml))
    vocab = pl.DataFrame(
        {
            "token_id": pl.Series(list(names), dtype=pl.Int64),
            "token_name": [names[i] for i in names],
            "family": [families[i] for i in names],
        }
    )

    for anchor in ANCHORS:
        for site in sites:
            try:
                tokens = load_pre_anchor_tokens(paths, site, anchor)
            except FileNotFoundError:
                logger.warning(
                    f"[{site}/{anchor}] no winnowed stream; run --stage winnow first"
                )
                continue
            n_cohort = tokens.height
            if n_cohort == 0:
                continue

            # one row per (patient, token occurrence). ~27M rows for 34k
            # patients at a median 783 tokens, which polars handles in one pass.
            exploded = (
                tokens.select("hospitalization_id", "tokens_past")
                .explode("tokens_past")
                .rename({"tokens_past": "token_id"})
                .drop_nulls("token_id")
                .with_columns(pl.col("token_id").cast(pl.Int64))
            )

            per_token = (
                exploded.group_by("token_id")
                .agg(
                    n_patients=pl.col("hospitalization_id").n_unique(),
                    total_occurrences=pl.len(),
                )
                .join(vocab, on="token_id", how="left")
                .with_columns(
                    token_name=pl.col("token_name").fill_null(
                        pl.lit("<not in vocabulary>")
                    ),
                    family=pl.col("family").fill_null(pl.lit("UNKNOWN")),
                )
            )
            per_family = (
                exploded.join(vocab, on="token_id", how="left")
                .with_columns(family=pl.col("family").fill_null(pl.lit("UNKNOWN")))
                .group_by("family")
                .agg(
                    # a patient is counted once per family however many of its
                    # tokens they emitted, so this cannot be summed from above
                    n_patients=pl.col("hospitalization_id").n_unique(),
                    total_occurrences=pl.len(),
                    # NOT `n_tokens_in_family`: the `n_*` convention in this
                    # pipeline means "a count of patients" and the suppression
                    # check keys on it. This counts vocabulary entries, which
                    # is a property of the tokenizer and of what this site
                    # charts, not of any patient -- blanking it at n<11 would
                    # hide that a family is thinly used here, which is the
                    # finding rather than a disclosure.
                    distinct_tokens=pl.col("token_id").n_unique(),
                )
                .sort("family")
            )

            for name, table in (
                ("token_prevalence", per_token),
                ("token_families", per_family),
            ):
                table = table.with_columns(
                    n_cohort=pl.lit(n_cohort, dtype=pl.Int64),
                    site=pl.lit(site),
                    anchor=pl.lit(anchor),
                )
                # primary suppression, and `total_occurrences` goes with it
                blank = (pl.col("n_patients") > 0) & (pl.col("n_patients") < min_cell)
                table = table.with_columns(
                    n_patients=pl.when(blank).then(None).otherwise(pl.col("n_patients")),
                    total_occurrences=pl.when(blank)
                    .then(None)
                    .otherwise(pl.col("total_occurrences")),
                ).with_columns(prevalence=pl.col("n_patients") / pl.col("n_cohort"))
                suppressed = int(table.select(pl.col("n_patients").is_null().sum()).item())
                contract.validate_counts_frame(table, name, min_cell)
                out = paths.cohort_home / f"{name}_{site}_anchor-{anchor}.csv"
                table.sort(
                    "total_occurrences", descending=True, nulls_last=True
                ).write_csv(out)
                logger.info(
                    f"[{site}/{anchor}] {out.name}: {table.height} rows, "
                    f"{suppressed} suppressed below n={min_cell}"
                )


# ======================================================== stage: baseline


def stage_phenotypes(paths, logger) -> None:
    """
    per-denominator Zweck summaries, from the cohort table alone

    Separate from `cohort` because it needs nothing but cohort.parquet: the
    three fits are already stored there as columns, and regenerating their
    summary tables should not cost a re-read of every site's raw CLIF. The
    report reads these CSVs, so a change to how a summary is presented is a
    seconds-long rerun rather than an hour.
    """
    cohort_file = paths.cohort_home / "cohort.parquet"
    if not cohort_file.is_file():
        logger.error(f"{cohort_file} not found; run --stage cohort first")
        return
    cohort = pl.read_parquet(cohort_file)

    present = [c for c, _ in PHENOTYPE_VARIANTS if c in cohort.columns]
    if not present:
        logger.error(
            "cohort.parquet carries no phenotype columns; it predates the "
            "per-denominator fit. Re-run --stage cohort."
        )
        return

    logger.info("Zweck phenotype variants and their denominators:")
    for column, description in PHENOTYPE_VARIANTS:
        if column not in cohort.columns:
            continue
        labelled = int(cohort[column].is_not_null().sum())
        logger.info(
            f"  {column:<32} {labelled:>7,} patients  ({description})"
        )
        summary = summarize(cohort, logger, column=column)
        if summary.is_empty():
            continue
        summary.write_csv(paths.cohort_home / f"phenotype_summary_{column}.csv")

    # the bare name keeps writing phenotype_summary.csv too, so anything still
    # reading the old filename is not silently left with nothing
    if "cs_phenotype" in cohort.columns:
        base = summarize(cohort, logger, column="cs_phenotype")
        if not base.is_empty():
            base.write_csv(paths.cohort_home / "phenotype_summary.csv")

    agreement = variant_agreement(cohort, logger)
    if not agreement.is_empty():
        agreement.write_csv(paths.cohort_home / "phenotype_variant_agreement.csv")
        logger.info(
            f"wrote {paths.cohort_home / 'phenotype_variant_agreement.csv'}"
        )


def stage_baseline(paths, sites, logger, min_cell: int) -> None:
    """
    median [IQR] and n (%) for each coded shock type

    The table a reader wants before any latent-space claim, and the one that
    says whether "cardiogenic" and "septic" describe clinically different
    patients at this site at all. Physiology is taken at cs_onset_dttm, through
    the same code 09 uses for its grid.

    Per site and never pooled: coding practice differs enough between sites
    that a pooled median is a weighted average over three documentation
    habits. The coordinating site stacks them and compares.
    """
    from clifcs import baseline

    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    for site in sites:
        part = cohort.filter(pl.col("site") == site)
        if part.height == 0:
            continue
        logger.info(f"[{site}] baseline characteristics by shock type")
        # raw_subset, not site_data: the physiology needs the DERIVED tables
        # (clif_medication_admin_continuous_converted, clif_sofa) that the
        # derive stage wrote into scratch, and norepinephrine equivalents
        # cannot be computed without the dose conversion. This is the same
        # directory 09 passes trajectory_state.
        collected = baseline.collect(paths.raw_subset(site), part, logger)
        table = baseline.summarise(collected, site, min_cell)
        if not table.height:
            logger.warning(f"[{site}] baseline table is empty")
            continue
        out = paths.cohort_home / f"baseline_by_shock_type_{site}.csv"
        table.write_csv(out)
        blanked = int(table.select(pl.col("n").is_null().sum()).item())
        logger.info(
            f"  wrote {out.name}: {table.height} rows across "
            f"{table['shock_type'].n_unique()} shock types"
            + (f", {blanked} suppressed below n={min_cell}" if blanked else "")
        )
        for row in table.filter(
            (pl.col("variable") == "MAP at onset (mmHg)")
            & pl.col("median").is_not_null()
        ).iter_rows(named=True):
            logger.info(
                f"    {row['shock_type']:<14} MAP {row['median']:.0f} "
                f"[{row['q1']:.0f}-{row['q3']:.0f}]  n={row['n']}"
            )


# ===================================================== stage: trials, export


def stage_trials(paths, sites, logger, min_cell: int) -> None:
    """
    the trial comparison, as aggregates

    Runs site-side because the eligibility predicates and the baseline variables
    read raw CLIF tables, and produces only shares and medians -- so a site that
    will never share a patient still contributes the whole trial analysis.

    This is the half of 11 that has to happen here. The other half -- fitting
    trial regions in embedding space -- needs the pooled embeddings and stays at
    the coordinating site.
    """
    from clifcs import trial_aggregate
    from clifcs.concordance import add_concordance, attach_procedures, attach_severity
    from clifcs.export import contract
    from clifcs.trials import (
        Region,
        add_eligibility,
        assign_guideline,
        attach_features,
        criteria_frame,
        load_trials,
        rank_and_orphans,
        specificity_from_frame,
        specificity_order,
        table1_position,
    )

    trial_cfg = OmegaConf.load(CONFIG_HOME / "trials.yaml")
    cohort_cfg = load_cohort_cfg()
    trials = load_trials(trial_cfg)
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    logger.info(
        f"{len(trials)} trials: "
        + ", ".join(
            f"{n} {a}"
            for a, n in sorted(
                pl.DataFrame({"a": [t.applies_to for t in trials]})["a"]
                .value_counts()
                .iter_rows()
            )
        )
    )

    for anchor in ANCHORS:
        per_site = []
        for site in sites:
            part = cohort.filter(pl.col("site") == site)
            if part.height == 0:
                continue
            # the anchor only shifts which physiology the criteria read; the
            # 24h anchor uses admission + 24h, onset uses cs_onset_dttm, and
            # attach_features keys off cs_onset_dttm either way
            part = attach_features(paths.site_data[site], part, cohort_cfg, logger)
            part = attach_procedures(paths.site_data[site], part, logger)
            per_site.append(part)
        if not per_site:
            continue

        frame = pl.concat(per_site, how="diagonal_relaxed")
        frame = add_eligibility(frame, trials)
        for trial in trials:
            frame = table1_position(frame, trial, logger)
        frame = attach_severity(paths, frame, logger)
        frame = add_concordance(frame, trials, logger)

        # ----------------------------------------------- coverage and guideline
        # Both were computed at the coordinating site by 11 until now, which
        # meant a site sharing nothing patient-level reported neither -- even
        # though neither reads an embedding. `rank_and_orphans` wants Regions
        # rather than Trials, so it gets unfitted ones: every `inside_*` and
        # `lift_*` column is then absent, `n_regions_inside` is 0 and
        # `evidence_orphan` collapses to `criteria_orphan`, which is exactly
        # the criteria-only reading. 11 still recomputes all of it against the
        # fitted regions when the embeddings are pooled.
        regions = tuple(
            Region(
                key=t.key,
                label=t.label,
                n_eligible=int(frame[f"elig_{t.key}"].sum()),
                fitted=False,
                applies_to=t.applies_to,
            )
            for t in trials
        )
        frame = rank_and_orphans(frame, regions)
        frame = assign_guideline(
            frame, trials, logger, order=specificity_order(trial_cfg, logger)
        )

        # What the ordering WOULD be if ranked on this cohort, logged next to
        # the assignment that used the committed one. At the derivation site
        # this is the line you copy into `specificity_order`; everywhere else
        # it is the check that says how far this site's eligible shares have
        # drifted from the ordering being applied to it.
        local = specificity_from_frame(frame, trials)
        committed = specificity_order(trial_cfg) or []
        logger.info(
            f"[{anchor}] specificity ranked on THIS cohort, most specific "
            "first (the derivation site commits this to "
            "`specificity_order` in trials.yaml):"
        )
        logger.info("  " + " ".join(local))
        if committed and committed != local:
            moved = [
                f"{k}({committed.index(k)}->{local.index(k)})"
                for k in local
                if k in committed and committed.index(k) != local.index(k)
            ]
            logger.info(
                f"  differs from the committed order at {len(moved)} of "
                f"{len(local)} positions: {' '.join(moved)}. The committed "
                "order is the one that was applied, which is what makes this "
                f"site's assignment comparable with every other site's."
            )

        # The per-patient columns, kept for 02 to export under the contract.
        # Written here rather than recomputed there because the predicates read
        # raw CLIF and this stage has already paid for that pass -- an earlier
        # version had `writer.trial_eligibility` do the whole join again, which
        # doubled the slowest site-side read for the same numbers.
        #
        # Only the four `t1_` statistics the contract names travel, plus
        # eligibility and concordance. `t1_maxdist_` and the two `n_` columns
        # stay here: nothing is coloured by them, and every continuous column
        # widens the quasi-identifier the exported row forms.
        per_patient = frame.select(
            "hospitalization_id",
            "site",
            # `sofa_non_cv` rides along because this is the only frame that has
            # it. attach_severity computes it here, from the clif_sofa parquet
            # the derive stage wrote, and it is NOT on the cohort table -- so
            # omitting it from this select, as the first version did, dropped it
            # for good. The contract names it, 07 reports severity-adjusted
            # results against it, and 12 uses it as an adjuster, so losing it
            # silently turned every adjusted comparison into an unadjusted one.
            *[
                c
                for c in ("sofa_non_cv",)
                if c in frame.columns
            ],
            *[
                c
                for c in frame.columns
                if c.startswith(contract.TRIAL_PREFIX)
                or c.startswith(contract.TRIAL_FIT_PREFIXES)
                or c.startswith(contract.CONCORDANCE_PREFIX)
            ],
        )
        out = paths.cohort_home / f"trial_patient_anchor-{anchor}.parquet"
        per_patient.write_parquet(out)
        logger.info(
            f"  wrote {out.name}: {per_patient.height} patients x "
            f"{per_patient.width - 2} trial columns, for 02 to export"
        )

        for site in sites:
            part = frame.filter(pl.col("site") == site)
            if part.height == 0:
                continue
            tables = trial_aggregate.build(
                paths, part, trials, site, anchor, logger, min_cell
            )
            for name, table in tables.items():
                out = paths.cohort_home / f"{name}_{site}_anchor-{anchor}.csv"
                table.write_csv(out)
                logger.info(f"  wrote {out.name}")

    # the published criteria and what can be applied: literature only, no
    # patient data, identical at every site
    criteria = criteria_frame(trial_cfg)
    if criteria.height:
        criteria.write_csv(paths.cohort_home / "trial_criteria.csv")


def stage_export(paths, sites, logger, min_cell: int) -> None:
    """
    assemble the directory this site sends

    Every table is re-checked against `min_cell` on the way in, so a table
    assembled by some other route cannot reach the directory unsuppressed.
    """
    from clifcs.export import contract, writer

    logger.info(contract.describe())
    salt = writer.load_salt(paths.id_salt_file, logger)

    summary = paths.table_home / "model_summary.json"
    embedding_dim = (
        json.loads(summary.read_text(encoding="utf-8")).get("embedding_dim", 0)
        if summary.is_file()
        else 0
    )

    for site in sites:
        bundle = paths.upload_home(site)
        if bundle.exists():
            shutil.rmtree(bundle)
        logger.info(f"=== {site} -> {bundle}")
        writer.write_aggregate(
            paths, site, bundle, salt, min_cell, embedding_dim, logger
        )

    logger.info(
        "the directory holds summary tables only -- every number a count, a "
        f"share or a median over at least {min_cell} patients, re-checked on "
        "the way in."
    )


# =================================================================== driver


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # the docstring is a stage table and two command blocks; the default
        # formatter reflows it into a single paragraph and loses both
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        "--check",
        action="store_true",
        help="verify configuration and input tables, then exit without running "
        "anything. Opens no data file; safe to run first",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="do not run the startup check. Only useful if a check is wrong "
        "about something you know to be fine",
    )
    parser.add_argument(
        "--phenotype-model",
        default=None,
        help="apply a phenotype model fitted elsewhere instead of the one in "
        "models/. Rarely needed: models/phenotype_model.json is applied by "
        "default, which is what a contributing site wants",
    )
    # The sepsis phenotype systems transfer the same way, and for the same
    # reason: a site that refits gets its own four clusters and "beta" stops
    # meaning the same thing in the pooled breakdown.
    parser.add_argument(
        "--seymour-model",
        default=None,
        help="apply a Seymour phenotype model fitted elsewhere "
        "(cohort/seymour_model.json) instead of refitting",
    )
    parser.add_argument(
        "--bhavani-model",
        default=None,
        help="apply a Bhavani phenotype model fitted elsewhere "
        "(cohort/bhavani_model.json) instead of refitting",
    )
    parser.add_argument(
        "--refit-phenotypes",
        action="store_true",
        help="fit the phenotype models on THIS cohort instead of applying the "
        "shared ones. Only correct at the derivation site -- everywhere else "
        "it makes the pooled breakdowns uninterpretable. Required to run the "
        "cohort stage without the three --*-model flags",
    )
    parser.add_argument(
        "--min-cell",
        type=int,
        default=11,
        help="minimum publishable cell size for the exported tables "
        "(default 11, the CMS threshold)",
    )
    args = parser.parse_args()

    # Configuration first, and on its own, because it is the most common thing
    # to be wrong and a bare traceback from inside the loader does not say that
    # it IS configuration. No logger yet -- the log file lives under a path this
    # has to resolve before it can be opened.
    try:
        paths = load_paths()
    except Exception as exc:
        print("=" * 70, file=sys.stderr)
        print("COULD NOT READ THE CONFIGURATION", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "  Configuration lives in code/config/paths.yaml. If you have not "
            "made one yet:",
            file=sys.stderr,
        )
        print(
            "      cp code/config/paths.example.yaml code/config/paths.yaml",
            file=sys.stderr,
        )
        print("  then set the path to this site's CLIF directory.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        raise SystemExit(1) from exc

    paths.mkdirs()
    cfg = load_cohort_cfg()
    sepsis_cfg = OmegaConf.load(CONFIG_HOME / "phenotypes_sepsis.yaml")
    sepsis_models = {
        scheme: path
        for scheme, path in (
            ("seymour", args.seymour_model),
            ("bhavani", args.bhavani_model),
        )
        if path
    }
    logger = get_logger("01_site_tables", paths.log_home / "01_site_tables.log")
    if duckdb_timezone() != "UTC":
        # clifcs pins it on import; clifpy's dose conversion reads timestamps
        # through this connection, so anything else shifts medication times
        raise SystemExit(
            f"DuckDB's session timezone is {duckdb_timezone()!r}, not UTC; "
            "something reset it after clifcs was imported."
        )

    if args.site and args.site not in paths.sites:
        parser.error(
            f"--site {args.site} is not in paths.yaml, which has: "
            f"{', '.join(paths.sites) or '(none)'}"
        )

    # --check: verify and stop, before any of the phenotype bookkeeping below.
    if args.check:
        raise SystemExit(
            0 if preflight(paths, (args.site,) if args.site else paths.sites, logger)
            else 1
        )

    # ------------------------------------------------------------------
    # Refusing to refit by accident.
    #
    # The three --*-model flags are documented as required at a contributing
    # site, and nothing enforced it. At Rush the flags were silently dropped
    # by a shell bug and the cohort stage refit its own clusters; the only
    # sign was the word "refit" in one log line, among warnings that looked
    # like findings about the data -- Seymour agreeing with the published
    # profiles on 32% of patients, mortality not increasing across
    # alpha-to-delta. Those were symptoms of the refit, and a reader could
    # easily have taken them for site characteristics.
    #
    # So refitting is now something you ask for. The derivation site passes
    # --refit-phenotypes and means it; everyone else gets an error naming the
    # files they should be passing instead.
    # Whether the COHORT STAGE will actually run, which is not the same as what
    # --stage says. `--from-stage` overrides `--stage`, but `--stage` keeps its
    # default of "all" -- so the old test (`args.stage in ("all", "cohort")`)
    # was true for every --from-stage invocation and refused
    # `--from-stage subset` for missing phenotype flags that stage never reads.
    if args.from_stage:
        needs_cohort = "cohort" in STAGES[STAGES.index(args.from_stage):]
    else:
        needs_cohort = args.stage in ("all", "cohort")
    transferred = (
        args.phenotype_model or args.seymour_model or args.bhavani_model
    )
    # The shipped models are the default. They are in git, under models/, and
    # applying them is what every contributing site should do -- so making the
    # three flags mandatory bought nothing except an invocation long enough to
    # get mangled, which is exactly what happened at Rush: a shell ate the
    # flags, the cohort stage refit its own clusters, and the only sign was the
    # word "refit" in one log line among warnings that looked like findings.
    #
    # Defaulting to them is strictly safer than erroring. Refitting is still
    # something you have to ask for with --refit-phenotypes.
    models_home = pathlib.Path(__file__).resolve().parent.parent / "models"
    if needs_cohort and not transferred and not args.refit_phenotypes:
        shipped = {
            "phenotype_model": models_home / "phenotype_model.json",
            "seymour_model": models_home / "seymour_model.json",
            "bhavani_model": models_home / "bhavani_model.json",
        }
        missing = [str(p) for p in shipped.values() if not p.is_file()]
        if not missing:
            for attr, path in shipped.items():
                setattr(args, attr, str(path))
            transferred = True
            logger.info(
                "applying the phenotype models shipped in models/ -- pass "
                "--refit-phenotypes to fit new ones on this cohort instead"
            )
    if needs_cohort and not transferred and not args.refit_phenotypes:
        parser.error(
            "the cohort stage would REFIT the phenotype models on this "
            "cohort, which is only correct at the derivation site. Refitting "
            'makes "phenotype II", "beta" and "group A" mean something '
            "different here than at every other site, and the pooled "
            "breakdowns become uninterpretable.\n\n"
            "At a contributing site, apply the shared models:\n"
            f"    --phenotype-model {models_home / 'phenotype_model.json'}\n"
            f"    --seymour-model   {models_home / 'seymour_model.json'}\n"
            f"    --bhavani-model   {models_home / 'bhavani_model.json'}\n\n"
            "At the derivation site, say so explicitly:\n"
            "    --refit-phenotypes"
        )
    if transferred and args.refit_phenotypes:
        parser.error(
            "--refit-phenotypes contradicts the --*-model flags; pass one or "
            "the other"
        )
    if args.refit_phenotypes:
        logger.warning(
            "REFITTING the phenotype models on this cohort. Correct only at "
            "the derivation site -- the labels will not be comparable with "
            "any other site's."
        )


    if args.from_stage:
        todo = STAGES[STAGES.index(args.from_stage) :]
    elif args.stage == "all":
        todo = STAGES
    else:
        todo = (args.stage,)
    sites = (args.site,) if args.site else paths.sites
    logger.info(f"stages: {', '.join(todo)} | sites: {', '.join(sites)}")
    logger.info(f"  output_home   {paths.output_home}")
    logger.info(f"  scratch_home  {paths.scratch_home}")
    logger.info(
        f"  tokenizer     {paths.tokenizer_yaml_or_none or 'NONE -- token stages will be skipped'}"
    )

    # The same check `--check` runs, on every invocation. It costs a second and
    # catches the two failures that otherwise surface hours later from inside
    # `derive` as an error about a column.
    if not args.skip_check and not preflight(paths, sites, logger):
        logger.error(
            "startup check failed, so nothing was run. Fix the lines above, or "
            "pass --skip-check if you are certain the check is wrong."
        )
        raise SystemExit(1)

    # The token stages, or an explanation of why they are not running. One
    # message for the group rather than three, and at WARNING so it survives a
    # long log.
    wanted_token_stages = [stage for stage in TOKEN_STAGES if stage in todo]
    if wanted_token_stages and not paths.has_tokenizer:
        logger.warning(
            f"SKIPPING {', '.join(wanted_token_stages)}: no tokenizer is "
            "configured on this machine. Set `tokenizer_yaml` in paths.yaml to "
            "include them."
        )
        logger.warning(
            "  the output directory will carry no token_prevalence, no "
            "token_families and no token_levels. Everything else -- cohort, "
            "phenotypes, baseline, trials and guidelines -- is unaffected."
        )
        todo = tuple(stage for stage in todo if stage not in TOKEN_STAGES)

    cohort_file = paths.cohort_home / "cohort.parquet"
    cohort = None
    total = len(todo)

    try:
        for position, stage in enumerate(todo, start=1):
            if stage == "cohort":
                cohort = run_stage(
                    stage, position, total, logger,
                    stage_cohort,
                    paths, cfg, logger, args.phenotype_model,
                    sepsis_cfg=sepsis_cfg, sepsis_models=sepsis_models,
                )
            elif stage == "subset":
                if cohort is None:
                    if not cohort_file.is_file():
                        logger.error(
                            f"{cohort_file} does not exist, and the `subset` "
                            "stage reads it. Run the cohort stage first:\n"
                            "    python code/01_site_tables.py --stage cohort"
                        )
                        raise SystemExit(1)
                    cohort = pl.read_parquet(cohort_file)
                run_stage(
                    stage, position, total, logger,
                    stage_subset, paths, cohort, sites, logger, args.force,
                )
            elif stage == "derive":
                run_stage(stage, position, total, logger,
                          stage_derive, paths, sites, logger, args.force)
            elif stage == "collate":
                run_stage(stage, position, total, logger,
                          stage_collate, paths, sites, logger, args.force)
            elif stage == "tokenize":
                run_stage(stage, position, total, logger,
                          stage_tokenize, paths, sites, logger, args.force)
            elif stage == "winnow":
                run_stage(stage, position, total, logger,
                          stage_winnow, paths, sites, logger, args.force)
            elif stage == "tokencounts":
                run_stage(stage, position, total, logger,
                          stage_tokencounts, paths, sites, logger, args.min_cell)
            elif stage == "phenotypes":
                run_stage(stage, position, total, logger,
                          stage_phenotypes, paths, logger)
            elif stage == "baseline":
                run_stage(stage, position, total, logger,
                          stage_baseline, paths, sites, logger, args.min_cell)
            elif stage == "trials":
                run_stage(stage, position, total, logger,
                          stage_trials, paths, sites, logger, args.min_cell)
            elif stage == "export":
                run_stage(stage, position, total, logger,
                          stage_export, paths, sites, logger, args.min_cell)
    except StageFailed as failure:
        report_failure(failure, logger)
        raise SystemExit(1) from failure

    if "export" in todo:
        logger.info("")
        logger.info("=" * 70)
        logger.info("DONE. Send this directory:")
        for site in sites:
            logger.info(f"    {paths.upload_home(site)}")
        logger.info("")
        logger.info(
            "Everything in it is a count, a share or a median over at least "
            f"{args.min_cell} patients."
        )
        logger.info("=" * 70)
    else:
        logger.info(f"done: {', '.join(todo)}")


if __name__ == "__main__":
    main()

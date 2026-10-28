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
import datetime as dt
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
from clifcs import memory
from clifcs.config import CONFIG_HOME
from clifcs.export.writer import put_model, put_table
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


#: one row per stage this invocation attempted, in the order it attempted them
#:
#: Written out as `run_stages.csv` and carried in the bundle, which is the only
#: part of `output/logs/01_site_tables.log` that can travel. The log itself
#: cannot: it prints the attrition ledger and the per-phenotype counts
#: unsuppressed, and it names `data_dir`, `output_home` and `scratch_home` as
#: absolute paths on the site's filesystem. This carries a stage name, a status
#: and a duration -- no counts, no paths, no free text -- and it answers the
#: question the coordinator actually has when a bundle looks thin, which is
#: whether the run finished.
STAGE_LEDGER: list[dict] = []


def write_ledger(paths) -> None:
    """
    the stage ledger as it stands, rewritten after every stage

    Rewritten after each stage rather than once at the end, so a run that dies
    in `derive` still leaves a ledger ending at the stage that failed -- the
    same argument as writing every table as it goes.

    Merged with what is already there rather than replacing it, because a
    stage can be rerun on its own. `--stage export` knows nothing about the
    `cohort` run from two days ago, and overwriting would turn the ledger into
    a record of the last *invocation* when what it has to be is a record of the
    last attempt at each stage -- which is what `output/cohort/` itself is.
    """
    if not STAGE_LEDGER:
        return
    attempted = {row["stage"] for row in STAGE_LEDGER}
    rows = list(STAGE_LEDGER)

    out = paths.cohort_home / "run_stages.csv"
    if out.is_file():
        try:
            previous = pl.read_csv(out)
        except Exception:
            previous = None  # unreadable, so there is nothing to preserve
        if previous is not None and "stage" in previous.columns:
            rows += [
                row
                for row in previous.iter_rows(named=True)
                if row["stage"] not in attempted
            ]

    # One schema across rows from different invocations. A `run_stages.csv`
    # written before `peak_rss_mb` existed has no such key, and polars takes
    # its schema from the first row -- so merging an old file with a new run
    # either drops the column or raises, depending which came first.
    columns = {key: None for row in rows for key in row}
    rows = [{**columns, **row} for row in rows]

    order = {stage: i for i, stage in enumerate(STAGES)}
    rows.sort(key=lambda row: order.get(row["stage"], len(order)))
    paths.cohort_home.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out)


def run_stage(name, position, total, logger, fn, *args, **kwargs):
    """
    run one stage, announcing it and timing it, and label any failure with it

    The announcement is the checkpoint: a run that dies leaves a log whose last
    `started` line names the stage, so the resume point is readable without a
    traceback. The timing is there because `derive` and `collate` take long
    enough that a reader needs to know whether a quiet run is working or hung.

    Peak memory is recorded for the same reason as the timing, one step
    further on. A site whose machine is too small does not get a slow run, it
    gets a killed process and a log that stops mid-stage -- and the figure
    that would have said how far over it went is gone with the process. The
    peak is sampled per stage rather than per process so that `derive`, which
    is where every large site will hit the ceiling, is attributable on its own.
    """
    logger.info(f"=== [{position}/{total}] {name}: {STAGE_PURPOSE.get(name, '')}")
    started = time.monotonic()
    peak = None

    def record(status: str) -> None:
        STAGE_LEDGER.append(
            {
                "stage": name,
                "status": status,
                "seconds": round(time.monotonic() - started, 1),
                # None, not 0, where the platform exposes no counter: a stage
                # that cannot be measured must not look like one that used
                # nothing.
                "peak_rss_mb": peak.megabytes if peak is not None else None,
                "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )

    # Recorded OUTSIDE the block in both cases, including the failure: the OS
    # high-water mark is only read when `watch_peak` exits, and the failing
    # stage is the one whose peak a site most needs.
    failure: Exception | None = None
    with memory.watch_peak() as peak:
        try:
            result = fn(*args, **kwargs)
        except Exception as cause:
            failure = cause
    if failure is not None:
        record("failed")
        raise StageFailed(name, failure) from failure
    record("ok")
    used = peak.megabytes
    logger.info(
        f"=== [{position}/{total}] {name} finished in "
        f"{time.monotonic() - started:,.0f}s"
        + (f", peak {used:,.0f} MB" if used is not None else "")
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
    min_cell: int = 11,
) -> pl.DataFrame:
    """identify cardiogenic shock at each site and assemble the cohort table"""
    paths.cohort_home.mkdir(parents=True, exist_ok=True)
    # Aggregate tables are written once, straight into each site's upload
    # folder. A new cohort starts that folder afresh, so tables from an
    # earlier cohort cannot ship alongside this one.
    for site in paths.sites:
        shutil.rmtree(paths.upload_home(site), ignore_errors=True)
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
        put_model(
            paths, paths.sites,
            variant_model_path(pathlib.Path("phenotype_model.json"), column).name,
            lambda path, model=model: save_phenotype_model(path, **model),
        )
    for column, table in centroid_tables.items():
        put_table(paths, paths.sites, f"phenotype_centroids_{column}.csv", table, min_cell, logger)
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
        put_table(paths, paths.sites, f"{name}.csv", table, min_cell, logger)
    for scheme, model in sepsis_fitted.items():
        put_model(
            paths, paths.sites, f"{scheme}_model.json",
            lambda path, model=model, scheme=scheme: sepsis_phenotype.save_model(path, model, scheme),
        )
    # A scheme that was APPLIED leaves no centroids and no model, so anything
    # of that shape already in the directory is from an earlier run that
    # refit. Removed rather than left: nothing clears cohort_home between
    # runs, and those two files are precisely the evidence that distinguishes
    # a transferred labelling from a local one -- 01_report.py reads them for
    # exactly that, and a derivation site copies `<scheme>_model.json` into
    # models/. Left behind, they describe a fit that no longer governs any
    # label in cohort.parquet.
    #
    # `seymour_agreement.csv` goes with them. It cross-tabulates the local
    # refit against the published Table 2 profiles, so it only exists on the
    # refit branch, and its cells are small enough to fall under `min_cell` --
    # which is why it is not in the export contract and must not be left where
    # a reader could take it for a shippable table.
    for scheme in ("seymour", "bhavani"):
        if scheme in sepsis_fitted:
            continue
        for stale in (
            paths.cohort_home / f"{scheme}_centroids.csv",
            paths.cohort_home / f"{scheme}_model.json",
            paths.cohort_home / f"{scheme}_agreement.csv",
        ):
            if stale.is_file():
                stale.unlink()
                logger.info(
                    f"removed {stale.name}, left by an earlier run that refit "
                    f"{scheme}; this run applied a model fitted elsewhere"
                )

    paths.cohort_home.mkdir(parents=True, exist_ok=True)
    cohort.write_parquet(paths.cohort_home / "cohort.parquet")
    ledger_frame = ledger.to_frame()
    put_table(paths, paths.sites, "filter_ledger.csv", ledger_frame, min_cell, logger)
    if centroids is not None:
        put_table(paths, paths.sites, "phenotype_centroids.csv", centroids, min_cell, logger)
    put_table(paths, paths.sites, "phenotype_summary.csv", phenotype_summary, min_cell, logger)
    put_table(paths, paths.sites, "shock_type_summary.csv", shock_type_summary, min_cell, logger)

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


def _derive_resp(paths, site, data_dir, logger) -> None:
    """the respiratory-support waterfall"""
    from clifpy.tables.respiratory_support import RespiratorySupport

    logger.info(f"[{site}] clifpy: respiratory-support waterfall")
    resp = RespiratorySupport.from_file(
        data_directory=str(data_dir), filetype="parquet", timezone="UTC"
    )
    # `return_dataframe=True` because `waterfall` copies internally and then
    # wraps the result in a second RespiratorySupport. Taking the frame leaves
    # one copy of the table live instead of three.
    processed = resp.waterfall(return_dataframe=True)
    processed.to_parquet(data_dir / "clif_respiratory_support_processed.parquet")


def _derive_doses(paths, site, data_dir, units, logger) -> None:
    """
    the two converted medication tables, each in its own scope

    TWO ORCHESTRATORS, TWO WEIGHT TABLES, and the second one is a bug fix
    rather than a consequence of the scoping.

    `convert_dose_units_for_continuous_meds` loads vitals itself, filtered to
    `vital_category=['weight_kg']` AND to the hospitalizations present in the
    CONTINUOUS medication table, then leaves the table object on the
    orchestrator. The intermittent conversion then reuses whatever is already
    there -- `if (self.vitals is None) or (self.vitals.df is None)`. So a
    patient with intermittent medications but no continuous ones never had a
    weight in this pipeline, and every weight-based intermittent dose they
    received fell back to its original unit. At Rush that is 1,992 patients:
    6,707 rows with no weight becomes 4,715.

    Giving each conversion its own orchestrator fixes it, because each loads
    vitals for its own medication table. It is a real change to the token
    stream -- `collation.yaml` reads `_convert_status` to decide whether a
    MED-INT token carries a dose at all -- and it is being taken in the same
    pass as the SOFA fix, which re-tokenises every site anyway. Doing it later
    would cost a second re-tokenisation for the same result.

    Separate orchestrators are also what frees the continuous medication frame
    before the intermittent one is read: function return is the scope boundary
    that actually releases a pandas-backed clifpy table.
    """
    import clifpy as cpy

    logger.info(f"[{site}] clifpy: dose conversion")
    output_directory = str(paths.log_home / f"clifpy-{site.lower()}")

    def convert(kind: str, preferred) -> None:
        orchestrator = cpy.ClifOrchestrator(
            data_directory=str(data_dir),
            filetype="parquet",
            timezone="UTC",
            output_directory=output_directory,
        )
        method = getattr(orchestrator, f"convert_dose_units_for_{kind}_meds")
        # `show_intermediate=True` stays -- `collation.yaml` reads the `_`
        # columns it adds.
        method(
            preferred_units=preferred,
            show_intermediate=True,
            override=True,
            save_to_table=True,
        )
        converted = getattr(orchestrator, f"medication_admin_{kind}").df_converted
        converted.to_parquet(
            data_dir / f"clif_medication_admin_{kind}_converted.parquet"
        )
        weighed = int(converted["weight_kg"].notna().sum())
        logger.info(
            f"[{site}]   {kind}: {len(converted):,} rows, {weighed:,} with a "
            f"weight ({weighed / max(len(converted), 1):.1%})"
        )

    convert("continuous", OmegaConf.to_container(units.continuous))
    convert("intermittent", OmegaConf.to_container(units.intermittent))


def _derive_sofa(paths, site, data_dir, logger) -> None:
    """
    the whole-stay SOFA score, and the windowed grid beside it

    One wide build serves both, which is what makes them provably consistent:
    the whole-stay worst and the worst in each 24 h window are extremal
    reductions of the same rows. The grid costs about a second per offset on
    top of a build that takes a minute, so it is not worth a separate pass --
    and it finally populates the `sofa_grid.parquet` that 09, 12 and 13 all
    read and nothing wrote.
    """
    from clifcs import sofa_grid
    from clifcs.sofa_grid import SOFA_GRID_OFFSETS

    logger.info(f"[{site}] clifpy: SOFA scoring")
    orchestrator, wide = sofa_grid.build_sofa_wide(paths, site, logger)

    # The whole-stay collapse is `sofa_by_hospitalization` above -- one row
    # per encounter block, merged on the encounter_block COLUMN. Scoring is
    # separate from collapsing, which is the better split: the collapse is a
    # pure function of two frames and can be tested without clifpy.
    scores = orchestrator.compute_sofa_scores(
        wide_df=wide, create_new_wide_df=False
    )
    # read back from the orchestrator: that call merges the encounter mapping
    # into the frame it was given, and `wide` itself has no encounter_block.
    # Projected to the three columns the collapse reads, because the sort is
    # on `event_time` alone and this frame is millions of rows wide.
    stamped = orchestrator.wide_df
    whole_stay = sofa_by_hospitalization(
        stamped[["encounter_block", "event_time", "hospitalization_id"]], scores
    )
    logger.info(
        f"[{site}] SOFA rows for {len(whole_stay)} of "
        f"{stamped['encounter_block'].nunique()} encounter blocks"
    )
    whole_stay.to_parquet(data_dir / "clif_sofa.parquet")

    grid = sofa_grid.sofa_on_grid(
        paths, site, SOFA_GRID_OFFSETS, logger, wide=wide,
        orchestrator=orchestrator,
    )
    if grid.height:
        grid.write_parquet(data_dir / "sofa_grid.parquet")


def stage_derive(paths, sites, logger, force: bool) -> None:
    """
    build the derived tables cocoa's collation.yaml expects

    Reproduces coreopsis recipes/run_clifpy.py then recipes/run_sofa_scoring.py,
    in that order: the first saves the converted medication tables, the second
    re-converts only the four SOFA vasopressors in memory without overwriting
    them.

    SPLIT INTO THREE, and each one its own function rather than three blocks of
    one. Function return is the scope boundary that actually frees a
    pandas-backed clifpy table; `del` inside one long function is the version
    the next edit silently breaks, and at UCMC this stage peaks around 17.5 GB
    with all three sets of tables co-resident. The per-artifact skip checks
    also mean a run that died in SOFA does not redo the waterfall.
    """
    units = OmegaConf.load(CONFIG_HOME / "clifpy_units.yaml")

    artifacts = (
        (("clif_respiratory_support_processed",), _derive_resp, ()),
        (
            (
                "clif_medication_admin_continuous_converted",
                "clif_medication_admin_intermittent_converted",
            ),
            _derive_doses,
            (units,),
        ),
        (("clif_sofa",), _derive_sofa, ()),
    )

    for site in sites:
        data_dir = paths.raw_subset(site)
        for targets, build, extra in artifacts:
            if all((data_dir / f"{t}.parquet").is_file() for t in targets) and not force:
                logger.info(f"[{site}] {targets[0]} exists, skipping")
                continue
            build(paths, site, data_dir, *extra, logger)
        logger.info(f"[{site}] derived tables written to {data_dir}")

    _mirror_sofa_grid(paths, logger)


def _mirror_sofa_grid(paths, logger) -> None:
    """
    assemble the pooled `sofa_grid.parquet` that 09, 12 and 13 read

    ASSEMBLED FROM EVERY SITE, not written by each. The three readers use the
    legacy pooled `patient_level_home` while the canonical copy is per site,
    beside the other derived tables -- so a per-site write into one pooled path
    means the last site to run silently replaces the others, and a
    `--site RUSH` run would leave the pooled file holding Rush alone. Rebuilt
    from whatever each site has on disk instead, which is correct whichever
    subset of sites this invocation touched.

    The readers are corrected to `patient_level_home_for(site)` separately;
    until then this keeps them reading a complete file.
    """
    parts = []
    for site in paths.sites:
        path = paths.raw_subset(site) / "sofa_grid.parquet"
        if path.is_file():
            parts.append(pl.read_parquet(path).with_columns(site=pl.lit(site)))
    if not parts:
        return
    pooled = pl.concat(parts, how="diagonal_relaxed")
    paths.patient_level_home.mkdir(parents=True, exist_ok=True)
    pooled.write_parquet(paths.patient_level_home / "sofa_grid.parquet")
    logger.info(
        f"sofa_grid: {pooled.height:,} rows from {len(parts)} site(s) -> "
        f"{paths.patient_level_home / 'sofa_grid.parquet'}"
    )


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
                # THE SHARED RULE, not a hand-rolled one. The version this
                # replaces blanked `n_patients` in 1..min_cell-1 and stopped
                # there, which covers only half the exposure: a token emitted by
                # all but a handful of the cohort published a `prevalence` that
                # gives those few back as
                # `n_cohort - round(prevalence * n_cohort)`, and published
                # `n_patients` itself, from which the same subtraction works
                # directly. `token_families_anchor-onset.csv` carried exactly
                # one such value and the export-time validator refused it.
                #
                # Nothing new is declared here. The registry already carries
                # both halves: `prevalence` over `n_cohort` with numerator
                # `n_patients`, and the ("n_patients", "n_cohort") complement
                # pair. Calling the shared suppressor is what connects them.
                #
                # `by=None` because these rows are tokens, not a partition -- a
                # patient emits many tokens and is counted in every one of them,
                # so no group of these rows sums to the cohort and the
                # group-wise complementary pass has nothing to protect. The
                # within-row complement pass still runs, and that is the one
                # that matters here.
                from clifcs.export import suppress

                table = table.with_columns(
                    prevalence=pl.col("n_patients") / pl.col("n_cohort")
                )
                table = suppress.suppress_counts(
                    table, min_cell=min_cell, by=None, logger=logger
                )
                # `total_occurrences` goes with a blanked patient count, on the
                # argument `suppress.NOT_PATIENT_STATISTICS` records for it: it
                # is not itself a patient cell, but a token occurring four times
                # in one patient is as identifying as the count. The suppressor
                # leaves it alone by design, so the producer does it.
                table = table.with_columns(
                    total_occurrences=pl.when(pl.col("n_patients").is_null())
                    .then(None)
                    .otherwise(pl.col("total_occurrences"))
                )
                suppressed = int(table.select(pl.col("n_patients").is_null().sum()).item())
                put_table(
                    paths, [site], f"{name}_anchor-{anchor}.csv",
                    table.sort("total_occurrences", descending=True, nulls_last=True),
                    min_cell, logger, suppressed=True,
                )
                logger.info(
                    f"[{site}/{anchor}] {name}: {table.height} rows, "
                    f"{suppressed} suppressed below n={min_cell}"
                )


# ======================================================== stage: baseline


def stage_phenotypes(paths, logger, min_cell: int) -> None:
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
        put_table(paths, paths.sites, f"phenotype_summary_{column}.csv", summary, min_cell, logger)

    # the bare name keeps writing phenotype_summary.csv too, so anything still
    # reading the old filename is not silently left with nothing
    if "cs_phenotype" in cohort.columns:
        base = summarize(cohort, logger, column="cs_phenotype")
        if not base.is_empty():
            put_table(paths, paths.sites, "phenotype_summary.csv", base, min_cell, logger)

    agreement = variant_agreement(cohort, logger)
    if not agreement.is_empty():
        put_table(paths, paths.sites, "phenotype_variant_agreement.csv", agreement, min_cell, logger)


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
        table = baseline.summarise(collected, site)
        if not table.height:
            logger.warning(f"[{site}] baseline table is empty")
            continue
        # `summarise` emits true values; suppression is applied here, by the
        # one module that decides what may be published. Shock types partition
        # the site within a variable, which is the group a single blanked cell
        # would be recoverable from by subtraction.
        #
        # `suppressed=True` then tells `put_table` the builder has already done
        # it, so it validates and writes rather than suppressing again -- which
        # is only a true claim because of the call above. In the version this
        # replaces, the baseline table was written with no suppression at all.
        from clifcs.export import suppress

        table = suppress.suppress_counts(
            table, min_cell=min_cell, by=("site", "variable"), logger=logger
        )
        put_table(
            paths, [site], "baseline_by_shock_type.csv", table, min_cell,
            logger, suppressed=True,
        )
        blanked = int(table.select(pl.col("n").is_null().sum()).item())
        logger.info(
            f"  baseline_by_shock_type.csv: {table.height} rows across "
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

    # ONE FRAME, BOTH ANCHORS, and that is a statement about the anchors
    # rather than an optimisation. Nothing between here and `assign_guideline`
    # reads `anchor`: `attach_features` keys off `cs_onset_dttm` whichever
    # anchor is named, and every column built from it follows. The two
    # `trial_patient_anchor-*.parquet` files this stage writes were verified
    # byte-identical after sorting -- 5,677 x 111, every column equal -- so the
    # loop was computing the same table twice and writing it under two names.
    #
    # Hoisted rather than left: it halves the slowest site-side stage. But it
    # also makes the identity structural, so if the 24h anchor is ever meant to
    # read physiology at admission + 24h, that belongs in `attach_features`,
    # and this hoist has to come back out at the same time.
    per_site = []
    for site in sites:
        part = cohort.filter(pl.col("site") == site)
        if part.height == 0:
            continue
        # the anchor only shifts which physiology the criteria read; the
        # 24h anchor uses admission + 24h, onset uses cs_onset_dttm, and
        # attach_features keys off cs_onset_dttm either way
        #
        # raw_subset, not site_data, for the same reason stage_baseline
        # gives: every table these two read is in REQUIRED_TABLES and so is
        # already in the cohort subset, cut down by an inner join on
        # hospitalization_id. Reading the full extract instead meant seven
        # lazy scans over the untrimmed tables -- 930 MB of clif_labs at
        # Rush against 76 MB subset -- twice per run, once per anchor, for
        # data that is identical either way.
        #
        # It also has to be the subset for a duller reason: at a site whose
        # extract sits on a synced drive, scanning it is a page fault
        # waiting to happen. Rush died in polars' collect_schema on
        # clif_labs with "Windows fatal exception: page error", which is
        # Box handing back a placeholder for a mapped page. Scratch is
        # local by construction.
        part = attach_features(paths.raw_subset(site), part, cohort_cfg, logger)
        part = attach_procedures(paths.raw_subset(site), part, logger)
        per_site.append(part)
    if not per_site:
        logger.warning("no site has any cohort rows; nothing to compare")
        return

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
        "specificity ranked on THIS cohort, most specific first (the "
        "derivation site commits this to `specificity_order` in trials.yaml):"
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
            # `sofa_window` beside it: two values, no patient in either, and
            # it is what tells a reader whether the adjustment came from the
            # 24h-ending-at-onset window or the contaminated whole-stay
            # fallback. A pooled analysis can then refuse a mixture instead of
            # averaging two different quantities under one column name.
            for c in ("sofa_non_cv", "sofa_window")
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

    for anchor in ANCHORS:
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
                put_table(
                    paths, [site], f"{name}_anchor-{anchor}.csv", table, min_cell,
                    logger, folder="trials", suppressed=True,
                )

    # the published criteria and what can be applied: literature only, no
    # patient data, identical at every site
    criteria = criteria_frame(trial_cfg)
    if criteria.height:
        put_table(paths, sites, "trial_criteria.csv", criteria, min_cell, logger, folder="trials", suppressed=True)


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
        logger.info(f"=== {site} -> {bundle}")
        # Only what tier 1 owns. A recursive delete stood here and took
        # patient/, purity/ and counts/token_coverage.csv with it, so
        # re-exporting to pick up one table destroyed the embeddings.
        writer.clear_tier1(bundle, logger)
        writer.write_aggregate(
            paths, site, bundle, salt, min_cell, embedding_dim, logger
        )

    logger.info(
        "the directory holds summary tables only -- every number a count, a "
        f"share or a median over at least {min_cell} patients, re-checked on "
        "the way in."
    )


# =================================================================== driver


SHIPPED_MODELS = {
    "phenotype_model": "phenotype_model.json",
    "seymour_model": "seymour_model.json",
    "bhavani_model": "bhavani_model.json",
}


def resolve_shipped_models(args, models_home):
    """Fill in each --*-model the caller did not name, INDEPENDENTLY.

    Returns ``(named, filled, absent)``: the schemes the caller named on the
    command line, those just filled in from ``models/``, and those that are
    neither -- the shipped file is missing, so the scheme would refit.

    Independence is the whole point. This used to be

        transferred = args.phenotype_model or args.seymour_model or ...

    so naming any ONE model suppressed the shipped defaults for the other two:
    ``--phenotype-model mine.json`` alone left both sepsis flags empty and
    refit Seymour and Bhavani -- the same silent refit the default exists to
    prevent, reached by an invocation that looks more careful rather than
    less.

    Mutates ``args`` because everything downstream reads the flags off it; the
    return value is for the caller to report and for tests to assert on.
    """
    named, filled, absent = [], [], []
    for attr, filename in SHIPPED_MODELS.items():
        if getattr(args, attr):
            named.append(attr)
            continue
        path = models_home / filename
        if path.is_file():
            setattr(args, attr, str(path))
            filled.append(attr)
        else:
            absent.append((attr, path))
    return named, filled, absent


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
    # The shipped models are the default. They are in git, under models/, and
    # applying them is what every contributing site should do -- so making the
    # three flags mandatory bought nothing except an invocation long enough to
    # get mangled, which is what happened at Rush: a shell ate the flags, the
    # cohort stage refit its own clusters, and the only sign was the word
    # "refit" in one log line among warnings that looked like findings.
    #
    # Refitting is still something you have to ask for with --refit-phenotypes.
    models_home = pathlib.Path(__file__).resolve().parent.parent / "models"
    if args.refit_phenotypes:
        if any(getattr(args, attr) for attr in SHIPPED_MODELS):
            parser.error(
                "--refit-phenotypes contradicts the --*-model flags; pass one "
                "or the other"
            )
        logger.warning(
            "REFITTING the phenotype models on this cohort. Correct only at "
            "the derivation site -- the labels will not be comparable with "
            "those of any other site."
        )
    elif needs_cohort:
        named, filled, absent = resolve_shipped_models(args, models_home)
        if absent:
            parser.error(
                "the cohort stage would REFIT "
                + ", ".join(attr for attr, _ in absent)
                + " on this cohort, which is only correct at the derivation "
                "site. Refitting makes the labels mean something different "
                "here than at every other site, and the pooled breakdowns "
                "become uninterpretable.\n\n"
                "These shipped models are missing from models/:\n"
                + "".join(f"    {path}\n" for _, path in absent)
                + "\nEither restore them, or point the flags at your own "
                "copies:\n"
                + "".join(
                    f"    --{attr.replace('_', '-')} <path>\n"
                    for attr, _ in absent
                )
                + "\nAt the derivation site, say so explicitly instead:\n"
                "    --refit-phenotypes"
            )
        if filled:
            logger.info(
                "applying the phenotype models shipped in models/ -- pass "
                "--refit-phenotypes to fit new ones on this cohort instead"
            )
        # NAMED, not summarised, and with the source of each. The line above
        # said "the models" and was true of the Zweck model alone for some
        # time: `sepsis_models` was built from these flags before the block
        # that filled them in, so the two sepsis schemes silently refit while
        # the log said they were being applied. One line per scheme means the
        # next time a model fails to reach a scheme, the log says which -- and
        # now whether it came from a flag or from models/.
        for attr in SHIPPED_MODELS:
            source = "flag   " if attr in named else "shipped"
            logger.info(f"  {attr:<16} {source}  {getattr(args, attr)}")

    # BUILT HERE, not up with the config loading, and that placement is the
    # point. `--seymour-model` and `--bhavani-model` are empty until the
    # shipped-model block above fills them in, so a mapping built any earlier
    # is `{}` on every run that relies on the default -- which is every run at
    # a contributing site. `stage_cohort` then gets no sepsis models and
    # refits both schemes, while the log two screens up says the shipped
    # models are being applied.
    #
    # That happened. It is the same class of failure the shipped-model default
    # was added to prevent, one layer down, and the only sign was the word
    # "refit" in a log line among warnings -- Seymour agreeing with the
    # published profiles on 38.6%, mortality not rising alpha-to-delta -- that
    # read as findings about the cohort rather than as symptoms of a refit.
    #
    # The Zweck model escaped it because `args.phenotype_model` is read at the
    # call site rather than copied into a dict ahead of time.
    sepsis_models = {
        scheme: path
        for scheme, path in (
            ("seymour", args.seymour_model),
            ("bhavani", args.bhavani_model),
        )
        if path
    }

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
                    min_cell=args.min_cell,
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
                          stage_phenotypes, paths, logger, args.min_cell)
            elif stage == "baseline":
                run_stage(stage, position, total, logger,
                          stage_baseline, paths, sites, logger, args.min_cell)
            elif stage == "trials":
                run_stage(stage, position, total, logger,
                          stage_trials, paths, sites, logger, args.min_cell)
            elif stage == "export":
                run_stage(stage, position, total, logger,
                          stage_export, paths, sites, logger, args.min_cell)
            # After the stage, so the ledger on disk never claims a stage
            # finished that did not. The consequence is that the copy `export`
            # puts in the bundle carries every stage up to but not including
            # `export` itself -- which is legible, since a bundle that exists
            # is evidence that export ran.
            write_ledger(paths)
    except StageFailed as failure:
        write_ledger(paths)
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

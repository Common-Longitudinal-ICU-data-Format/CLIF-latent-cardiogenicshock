#!/usr/bin/env python3

"""
01 -- site tier 1: everything a site can send without argument

The first of two scripts that run inside a contributing institution, and the
one every site runs. It builds the cohort and writes the **aggregate** bundle:
patient flow, cohort sizes, the phenotype breakdowns, the coded shock types,
the trial comparison, and how differently this site charts.

Nothing it exports is patient-level. Every number is a count or an average over
at least `--min-cell` patients, so it travels under a consortium's ordinary
terms rather than needing a patient-level data use agreement.

**It does not need a GPU, and it never loads the GEM weights.** It loads the
*tokenizer*, because the token-prevalence table is built from the token stream,
but the model itself is not needed until `02_site_embeddings.py`. A site with no
GPU, or with no agreement permitting patient-level transfer, runs this and
stops -- and that is a complete contribution.

Stages, each rerunnable on its own with --stage, which matters because `derive`
and `collate` are the slow ones:

  cohort       identify shock, print the filter ledger, attach outcomes, coded
               shock type, and the Zweck, Seymour and Bhavani phenotypes
  subset       cut each site's CLIF tables down to the cohort -> scratch/data-raw/
  derive       clifpy: respiratory-support waterfall, dose conversion, SOFA
  collate      cocoa collate                                  -> scratch/processed/
  tokenize     cocoa tokenize, under the model's frozen tokenizer
  winnow       cut each timeline at both anchors (24h, onset)
  tokencounts  per-token and per-family prevalence            -> output/cohort/
  trials       eligibility, IQR position and concordance, as aggregates
  export       assemble the aggregate bundle           -> output/upload_to_box_<SITE>/

Usage
    python code/01_site_tables.py                          # all stages
    python code/01_site_tables.py --stage cohort           # just the cohort
    python code/01_site_tables.py --from-stage tokencounts  # resume
    python code/01_site_tables.py --site NU --force         # redo one site

At a contributing site the three phenotype models are required, so that a label
means the same thing everywhere:

    python code/01_site_tables.py         --phenotype-model <...>/phenotype_model.json         --seymour-model   <...>/seymour_model.json         --bhavani-model   <...>/bhavani_model.json
"""

from __future__ import annotations

import argparse
import json
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
from clifcs.shock_type import shock_type_flags
from clifcs.shock_type import summarize as summarize_shock_type
from clifcs import sepsis_phenotype
from clifcs.phenotype import (
    add_egfr,
    apply_phenotype_model,
    assign_phenotypes,
    nearest_labs,
    save_phenotype_model,
    summarize,
)

STAGES = (
    "cohort",
    "subset",
    "derive",
    "collate",
    "tokenize",
    "winnow",
    # the token stream is tokenizer-only, so the one comparison that needs no
    # model at all -- how differently the sites chart -- belongs in tier 1.
    # 04 draws its tornado from this table, which is why that figure now
    # survives for a site that shares nothing patient-level.
    "tokencounts",
    "trials",
    "export",
)

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
    centroids = None
    if phenotype_model:
        cohort = apply_phenotype_model(cohort, phenotype_model, logger)
    else:
        cohort, centroids, model = assign_phenotypes(cohort, cfg)
        save_phenotype_model(paths.cohort_home / "phenotype_model.json", **model)
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
        add_eligibility,
        attach_features,
        criteria_frame,
        load_trials,
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
    assemble the aggregate bundle each site sends

    No purity here any more. Phenotype coherence in the embedding space needs
    the embeddings, so it moved to 02 along with the rest of the GPU work --
    which means a tier-1-only site no longer reports it. That is a real loss:
    it was the one substantive thing tier 1 said beyond counts. It is the price
    of tier 1 being runnable without a GPU and without an agreement.
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
        "01 done. output/upload_to_box_<SITE>/ holds summary tables only -- "
        "aggregate, suppressed, and safe to upload. No GPU was needed and the "
        "GEM weights were never loaded. Run 02_site_embeddings.py only if your "
        "consortium permits patient-level transfer; without it this bundle is "
        "already a complete contribution."
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

    paths = load_paths()
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
    needs_cohort = args.stage in ("all", "cohort") or args.from_stage == "cohort"
    transferred = (
        args.phenotype_model or args.seymour_model or args.bhavani_model
    )
    if needs_cohort and not transferred and not args.refit_phenotypes:
        models_home = pathlib.Path(__file__).resolve().parent.parent / "models"
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
    if args.site and args.site not in paths.sites:
        parser.error(f"--site {args.site} is not configured; have {paths.sites}")
    sites = (args.site,) if args.site else paths.sites
    logger.info(f"stages: {', '.join(todo)} | sites: {', '.join(sites)}")

    cohort_file = paths.cohort_home / "cohort.parquet"
    cohort = None

    if "cohort" in todo:
        cohort = stage_cohort(
            paths,
            cfg,
            logger,
            args.phenotype_model,
            sepsis_cfg=sepsis_cfg,
            sepsis_models=sepsis_models,
        )
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
    if "tokencounts" in todo:
        stage_tokencounts(paths, sites, logger, args.min_cell)
    if "trials" in todo:
        stage_trials(paths, sites, logger, args.min_cell)
    if "export" in todo:
        stage_export(paths, sites, logger, args.min_cell)
    else:
        logger.info("01 done")


if __name__ == "__main__":
    main()

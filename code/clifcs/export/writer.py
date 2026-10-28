#!/usr/bin/env python3

"""
writing the two tiers of a site bundle

Split in two because the consortium treats them differently, and the split is
the contract rather than a flag on one command:

**Tier 1, the aggregate bundle** (`write_aggregate`) -- patient flow, cohort
sizes, the phenotype breakdowns, and how coherent each phenotype is in this
site's embedding space. All of it is counts and averages over at least
`min_cell` patients. Every site sends this.

**Tier 2, the patient bundle** (`write_patient`) -- one row per patient: the
1024-d embedding, the ordered token bins, the trial eligibility booleans and
the prediction labels. This is **patient-level data and is not
de-identified**; an embedding is a learned function of the patient's whole
recorded history. It moves under the same IRB approval and DUA as patient-level
clinical data, or it does not move. A site whose agreement does not permit it
simply never runs tier 2, and the pooled side degrades rather than failing.

Both write into one bundle directory so a coordinating site sees a single
place per contributor:

    export/<SITE>/
      manifest.json          written by tier 1, amended by tier 2
      token_levels.csv       the legend, derived from the tokenizer alone
      run_stages.csv         tier 1: which stages ran, and how long each took
      counts/                tier 1: aggregate tables, suppressed
      purity/                tier 1: per-phenotype coherence in GEM space
      patient/               tier 2: embeddings and ordered token labels

Tier 1 may rewrite only what it owns -- see `TIER1_CLEARS`. The two tiers share
one directory and often run days apart, so a tier-1 re-export that cleared the
directory wholesale deleted work it did not produce.

What does not go out, and cannot: raw records, timestamps, record numbers, ages
in years, race and ethnicity, the token *sequences*, and anything else not named
in `clifcs.export.contract`. The check is an allowlist and it raises rather than
dropping, so a column added to the cohort frame upstream stops the export
instead of travelling silently.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import secrets
import shutil

import polars as pl
from omegaconf import OmegaConf

from clifcs.config import CONFIG_HOME
from clifcs.export import contract, manifest, suppress
from clifcs.tokenlabels import coverage, last_pre_anchor_ranks, levels
from clifcs.tokens import family_matrix, load_pre_anchor_tokens, vocabulary

#: for each aggregate table, the grouping a published row of it corresponds to
#: -- the columns held CONSTANT across the cells that partition a population.
#:
#: This replaces `_DERIVED`, a tuple of statistic names that was passed to
#: `suppress_counts` as `derived=`. That parameter is gone: a statistic is now
#: registered in `suppress.STATISTICS` against the count it was computed over,
#: keyed by column name alone, so one name has one denominator everywhere
#: rather than one anchor per frame.
#:
#: The `by` groupings used to be built at the call site as "every candidate
#: column present in the table", which swept in the VARYING dimension --
#: `shock_type` for the shock-type breakdown, `cs_phenotype` for a phenotype
#: summary -- leaving one row per group, so complementary suppression never
#: fired on any table here.
#:
#: `()` means the whole table is one group. `None` opts out, for a table whose
#: rows do not partition anything.
#:
#: An unknown name raises rather than defaulting, so adding a table to the
#: bundle forces a decision about it.
_COMPLEMENTARY_BY: dict[str, tuple[str, ...] | None] = {
    # rows are nested subsets, not a partition: subtracting one step from the
    # next IS the attrition ledger, which is the point of the table
    "filter_ledger.csv": None,
    # one distribution over the phenotype labels, which partition the cohort
    "phenotype_summary.csv": (),
    "phenotype_centroids.csv": (),
    # a cross-tab of two labellings of the same patients: the cells partition
    # the cohort, so a blanked cell is recoverable from the others
    "phenotype_variant_agreement.csv": (),
    "seymour_summary.csv": (),
    "bhavani_summary.csv": (),
    "seymour_centroids.csv": (),
    "bhavani_centroids.csv": (),
    "seymour_agreement.csv": (),
    # each site's shock types partition that site
    "shock_type_summary.csv": ("site",),
    # one row per variable; variables do not partition patients
    "seymour_availability.csv": None,
}

#: table families whose names carry a parameter -- the phenotype variants, one
#: file per denominator. Matched by prefix because the suffix is a column name
#: from `phenotype.PHENOTYPE_VARIANTS`, not a fixed string.
_COMPLEMENTARY_PREFIXES: tuple[tuple[str, tuple[str, ...] | None], ...] = (
    ("phenotype_summary_", ()),
    ("phenotype_centroids_", ()),
)


def complementary_by(name: str) -> tuple[str, ...] | None:
    """
    the grouping `name` partitions, or None if its rows partition nothing

    Raises for a table nobody has decided about, which is the one property
    that matters here: a new aggregate table fails the export rather than
    travelling with complementary suppression quietly disabled.
    """
    if name in _COMPLEMENTARY_BY:
        return _COMPLEMENTARY_BY[name]
    for prefix, grouping in _COMPLEMENTARY_PREFIXES:
        if name.startswith(prefix):
            return grouping
    raise KeyError(
        f"{name} has no entry in writer._COMPLEMENTARY_BY, so there is no "
        "declaration of which population a published row of it partitions. "
        "Add one: `()` if the whole table is one group, a tuple of the columns "
        "held constant across a partition, or None if its rows partition "
        "nothing."
    )


#: what tier 1 writes inside a bundle, and therefore what it may clear
#:
#: `shutil.rmtree(bundle)` stood in `stage_export` for this, and it is the
#: right instinct for the wrong unit: a table a previous code version wrote
#: under a name this one no longer produces has to go, or it travels forever.
#: But the bundle is shared. Tier 2 writes patient/, purity/ and
#: counts/token_coverage.csv into the same directory, often days later and from
#: a different machine, so a tier-1 re-export -- which is what picking up one
#: newly produced table means -- silently deleted every embedding the site had
#: computed and left a bundle that `bundles.discover` reads as an incomplete
#: copy.
#:
#: `manifest.json` is deliberately NOT here. `write_aggregate` overwrites it
#: unconditionally, so there is no stale name to remove, and clearing it first
#: destroyed the tier-2 anchors and row counts it reads back out of the
#: existing file to carry forward.
TIER1_CLEARS = (
    "token_levels.csv",
    "run_stages.csv",
    "counts/*.csv",
    "trials/*.csv",
)

#: files that match TIER1_CLEARS but belong to tier 2
#:
#: `token_coverage.csv` is per-variable token availability. It needs the
#: winnowed token stream that 02 holds, so it is written by `write_patient`,
#: but it is an aggregate and so it lives in counts/ with the rest of them --
#: which puts it inside a tier-1 glob without being a tier-1 file.
TIER2_IN_TIER1_PATHS = frozenset({"counts/token_coverage.csv"})


def clear_tier1(bundle: pathlib.Path, logger) -> None:
    """
    remove what tier 1 is about to rewrite, and nothing else

    Not a recursive delete. See TIER1_CLEARS for what that cost.
    """
    if not bundle.is_dir():
        return
    removed = 0
    for pattern in TIER1_CLEARS:
        for stale in sorted(bundle.glob(pattern)):
            if stale.relative_to(bundle).as_posix() in TIER2_IN_TIER1_PATHS:
                continue
            if stale.is_file():
                stale.unlink()
                removed += 1
    kept = [
        name
        for name in ("patient", "purity")
        if (bundle / name).is_dir() and any((bundle / name).iterdir())
    ]
    if removed:
        logger.info(f" cleared {removed} tier-1 file(s) from the existing bundle")
    if kept:
        logger.info(
            f"  left {', '.join(f'{k}/' for k in kept)} in place -- written by "
            "02_site_embeddings.py, not by this script"
        )


def row_keys(frame: pl.DataFrame, salt: str) -> pl.Series:
    """
    the pseudonym that ties a patient's rows together across files

    SHA-256 over salt + record number, truncated to 128 bits -- not polars'
    built-in `.hash()`, which is xxhash and carries no preimage guarantee. The
    claim being made is that a recipient cannot recover the record number, so
    the hash has to be one that actually supports it.
    """
    return pl.Series(
        contract.ROW_KEY,
        [
            hashlib.sha256(f"{salt}:hospitalization_id:{v}".encode()).hexdigest()[:32]
            for v in frame["hospitalization_id"].cast(pl.String).to_list()
        ],
    )


def load_salt(path: pathlib.Path, logger) -> str:
    """
    the site's pseudonymisation salt, created once and never regenerated

    Regeneration is the failure this guards against: the salt lives under
    gitignored `output/`, so a site that cleared its output directory and
    re-exported used to produce an entirely new set of pseudonyms for the same
    patients, which the coordinating site cannot distinguish from a new cohort.
    The manifest carries a fingerprint of the salt so a change is at least
    detectable; this is where it is avoided in the first place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    path.write_text(secrets.token_hex(32), encoding="utf-8")
    logger.warning(
        f"created a new pseudonymisation salt at {path}. It stays on this "
        "machine, it is what lets you re-link a shared pseudonym back to a "
        "local record, and it must be BACKED UP -- losing it means every "
        "future export carries new pseudonyms that cannot be matched to this "
        "one."
    )
    return path.read_text(encoding="utf-8").strip()


def age_bands(frame: pl.DataFrame) -> pl.Expr:
    """
    five-year age bands with a 90+ top code

    Exact age is not exported. A row already carries ten token bins and three
    phenotype labels, and age in years is the quasi-identifier that makes that
    combination re-identifying; every panel drawn against age in this pipeline
    is drawn against a band or a decade anyway.
    """
    age = pl.col("age_at_admission").cast(pl.Float64)
    return (
        pl.when(age >= 90)
        .then(pl.lit("90+"))
        .otherwise(
            ((age / 5).floor() * 5).cast(pl.Int32).cast(pl.String)
            + "-"
            + ((age / 5).floor() * 5 + 4).cast(pl.Int32).cast(pl.String)
        )
        .alias("age_band")
    )


def attach_labels(
    paths,
    cohort: pl.DataFrame,
    site: str,
    anchor: str,
    logger,
    eligibility: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    one row per patient: embedding, ordered token bins, labels and outcomes

    **No redaction.** This is the assembly half of the export, factored out
    because the pooled scripts need the same columns and were not getting them.

    `tok_*`, `fam_n_*`, the trial columns and the downstream labels are all
    joined on here rather than written by 02's embed stage, so a locally
    produced `patient_level_data_<SITE>/embeddings_anchor-*.parquet` does not
    carry any of them. That meant 03 under `--role site` silently skipped every
    panel coloured by MAP, SOFA, trial eligibility, trial fit or concordance --
    the guards are on column presence, so the figures just did not appear.

    Returns the frame and the per-variable token coverage table. Raises if the
    embeddings for this site and anchor have not been produced yet.
    """
    # patient_level_data_<SITE>/<model>/, which is where 02 writes. An earlier
    # version looked in the flat `patient_level_data/` the pooled scripts use
    # and could not find a site-side export at all.
    source = (
        paths.patient_level_home_for(site) / f"embeddings_anchor-{anchor}.parquet"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} not found; run 02_site_embeddings.py --stage embed for "
            f"{site}/{anchor} first"
        )
    embeddings = pl.read_parquet(source)

    # ------------------------------------------------- the ordered token bins
    tokens = load_pre_anchor_tokens(paths, site, anchor)
    ranks = last_pre_anchor_ranks(tokens, str(paths.tokenizer_yaml))
    token_coverage = coverage(ranks).with_columns(site=pl.lit(site), anchor=pl.lit(anchor))

    # ------------------------------------------------- token family counts
    # ~23 integers per patient. These are what 04 regresses the site effect on,
    # and the site effect is the largest finding in this pipeline; without them
    # it cannot be decomposed at the coordinating site at all.
    _, families = vocabulary(str(paths.tokenizer_yaml))
    family_order = sorted(set(families.values()))
    counts = family_matrix(tokens, families, family_order)
    family_frame = tokens.select("hospitalization_id").with_columns(
        [
            pl.Series(f"{contract.FAMILY_PREFIX}{name}", counts[:, i].astype("int32"))
            for i, name in enumerate(family_order)
        ]
    )

    frame = (
        embeddings.join(ranks, on=["hospitalization_id", "site"], how="left")
        .join(family_frame, on="hospitalization_id", how="left")
        .join(
            cohort.select(
                "hospitalization_id",
                "site",
                "sepsis_cohort",
                *[
                    c
                    for c in ("seymour_phenotype", "bhavani_phenotype", "sofa_non_cv")
                    if c in cohort.columns
                ],
            ),
            on=["hospitalization_id", "site"],
            how="left",
        )
    )
    if eligibility is not None and eligibility.height:
        frame = frame.join(eligibility, on="hospitalization_id", how="left")

    # the downstream prediction labels 06 needs, if 06 has been run here. Only
    # the `__label` / `__eligible` pairs travel -- the baseline physiology in
    # the same file (`lab_*`, `vtl_*`, `nee_at_anchor`) is measured values and
    # stays put.
    label_file = paths.cohort_home / f"downstream_labels_anchor-{anchor}.parquet"
    if label_file.is_file():
        labels = pl.read_parquet(label_file)
        wanted = [
            c for c in labels.columns if c.endswith(contract.LABEL_SUFFIXES)
        ]
        if wanted and "hospitalization_id" in labels.columns:
            frame = frame.join(
                labels.select("hospitalization_id", "site", *wanted),
                on=["hospitalization_id", "site"],
                how="left",
            )
            logger.info(
                f"  downstream labels: {len(wanted) // 2} task(s) "
                + ", ".join(
                    sorted(c.removesuffix("__label") for c in wanted if c.endswith("__label"))
                )
            )
    else:
        logger.info(
            f"  no {label_file.name}; run 06_downstream_tasks.py site-side "
            "first if the pooled analysis needs the prediction labels"
        )

    return frame, token_coverage


def build_patient_frame(
    paths,
    cohort: pl.DataFrame,
    site: str,
    anchor: str,
    salt: str,
    logger,
    eligibility: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    the redacted, contract-conformant copy that travels

    `attach_labels` assembles; this redacts. The two are separate so that a
    pooled script can have the labels without also having to pseudonymise and
    then un-pseudonymise its own data.

    Age becomes a five-year band and the record number becomes a salted hash
    here, not in `attach_labels` -- a local analysis wants the real values, and
    only the copy that leaves must not have them.
    """
    frame, token_coverage = attach_labels(
        paths, cohort, site, anchor, logger, eligibility
    )
    frame = frame.with_columns(age_bands(frame), row_keys(frame, salt))

    # Drop everything the contract does not name. Done explicitly and then
    # validated, rather than relying on the selection being right: the
    # validation is what catches a column this list forgot.
    keep = [c for c in frame.columns if contract.is_permitted(c)]
    frame = frame.select(keep)
    contract.validate_patient_frame(frame, f"{site}/{anchor} patient frame")
    return frame, token_coverage


def trial_columns(paths, anchor: str, site: str, logger) -> pl.DataFrame:
    """
    per-trial eligibility, fit and concordance, as the trials stage computed them

    Read from `cohort/trial_patient_anchor-<anchor>.parquet` rather than
    recomputed. The predicates behind these columns read `clif_vitals`,
    `clif_labs`, `clif_medication_admin_*`, `clif_hospital_diagnosis` and
    `clif_patient_procedures` -- which only a site can do, and which 01 has
    already paid a full pass over to produce the aggregate trial tables.
    Recomputing them here, as an earlier version did, doubled the slowest
    site-side read for identical numbers.

    Three families travel, each named in the contract:

      elig_<trial>              would the trial have enrolled this patient
      t1_absdist_/severity_gap_/outside_/frac_compared_<trial>
                                where they sit against the published IQRs, and
                                which way
      concordant_<trial>        was the recommended treatment delivered

    Only the booleans and the four summary statistics. The pressures, lactate
    and per-agent infusion flags they were computed from stay here.
    """
    source = paths.cohort_home / f"trial_patient_anchor-{anchor}.parquet"
    if not source.is_file():
        logger.warning(
            f"  no {source.name}; run 01_site_tables.py --stage trials to add "
            "trial eligibility, fit and concordance to the export. The pooled "
            "trial analysis loses those columns rather than recomputing them."
        )
        return pl.DataFrame({"hospitalization_id": []})

    frame = pl.read_parquet(source).filter(pl.col("site") == site)
    if frame.height == 0:
        return pl.DataFrame({"hospitalization_id": []})

    wanted = [
        c
        for c in frame.columns
        if c.startswith(contract.TRIAL_PREFIX)
        or c.startswith(contract.TRIAL_FIT_PREFIXES)
        or c.startswith(contract.CONCORDANCE_PREFIX)
        # `sofa_non_cv` rides in this parquet because the trials stage is the
        # only place that computes it. Filtering on the three trial prefixes
        # alone dropped it on the way back out -- the same loss as writing it
        # without reading it, one layer down. The contract names it, and 12
        # adjusts on it.
        or c == "sofa_non_cv"
    ]
    eligible = [c for c in wanted if c.startswith(contract.TRIAL_PREFIX)]
    logger.info(
        f"  trial columns: {len(eligible)} trials, {len(wanted)} columns "
        + ", ".join(
            f"{c.removeprefix(contract.TRIAL_PREFIX)} {int(frame[c].sum())}"
            for c in eligible
        )
    )
    return frame.select("hospitalization_id", *wanted)


def put_table(
    paths, sites, name: str, table: pl.DataFrame, min_cell: int, logger,
    *, folder: str = "counts", suppressed: bool = False,
) -> None:
    """
    write one aggregate table into each site's upload folder, its only copy

    Suppressed here unless its builder already did it (the trial, baseline and
    token tables), then checked, so the folder never holds an unchecked file,
    even when a run stops halfway.

    `suppressed=True` is a claim about the BUILDER, and it is only a true claim
    where the builder actually suppresses. The baseline table was passed with
    it while `stage_baseline` wrote raw values; that stage now suppresses
    before calling here, which is what makes the flag honest.

    The grouping comes from `complementary_by(name)` rather than from "every
    candidate column present in the frame". That older rule swept in the
    varying dimension -- `shock_type` for the shock-type breakdown -- so each
    group held exactly one row and complementary suppression never fired.
    """
    if not suppressed:
        table = suppress.suppress_counts(
            table,
            min_cell=min_cell,
            by=complementary_by(name),
            logger=logger,
        )
    # Checked whether or not this function suppressed, because the two-ended
    # design is the point: a table assembled by some other route, or by a
    # builder that believes it suppressed, still cannot reach a bundle
    # unchecked.
    contract.validate_counts_frame(table, name, min_cell)
    for site in sites:
        out = paths.upload_home(site) / folder / name
        out.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(out)
        logger.info(f"  {out.parent.name}/{name} ({table.height} rows)")


def put_model(paths, sites, name: str, write) -> None:
    """
    write a fitted phenotype model into each site's models/ folder

    `write(path)` saves the file; anything beyond contract.MODEL_FIELDS
    (moments and centroids) is refused and the file removed.
    """
    for site in sites:
        out = paths.upload_home(site) / "models" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        write(out)
        unexpected = set(json.loads(out.read_text())) - contract.MODEL_FIELDS
        if unexpected:
            out.unlink()
            raise contract.ContractViolation(
                f"{name}: fields {sorted(unexpected)} are not in contract.MODEL_FIELDS"
            )


def check_bundle(paths, site: str, min_cell: int, logger) -> None:
    """
    re-check everything the stages wrote into this site's upload folder

    Every table is validated again, so a file placed there by any other route
    cannot pass unchecked. Token tables are removed when this machine has no
    tokenizer: they are token ids, comparable across sites only under the
    mapping whose hash the manifest records, and a manifest without that hash
    must not carry them.
    """
    bundle = paths.upload_home(site)
    for table_file in sorted(bundle.glob("*/*.csv")):
        if not paths.has_tokenizer and table_file.name.startswith(("token_prevalence_", "token_families_")):
            table_file.unlink()
            logger.warning(f"  removed {table_file.name}: no tokenizer on this machine")
            continue
        contract.validate_counts_frame(pl.read_csv(table_file), table_file.name, min_cell)
    for model_file in sorted(bundle.glob("models/*.json")):
        unexpected = set(json.loads(model_file.read_text())) - contract.MODEL_FIELDS
        if unexpected:
            raise contract.ContractViolation(
                f"{model_file.name}: fields {sorted(unexpected)} are not in contract.MODEL_FIELDS"
            )


# ============================================================ tier 1: aggregate


def write_aggregate(
    paths,
    site: str,
    bundle: pathlib.Path,
    salt: str,
    min_cell: int,
    embedding_dim: int,
    logger,
) -> pathlib.Path:
    """
    the bundle every site sends: counts, phenotype breakdowns, token prevalence

    No purity. Phenotype coherence in the embedding space needs the embeddings,
    so it moved to tier 2 with the rest of the GPU work and is written by
    `write_purity` when 02 runs. Tier 1 no longer loads the model at all.

    Writes the manifest with `shares_patient_level: false`. Tier 2 amends it if
    and when it runs, so the presence of patient data is recorded in one place
    rather than inferred from a directory listing.
    """
    bundle.mkdir(parents=True, exist_ok=True)

    logger.info(" checking the tables already written here")
    check_bundle(paths, site, min_cell, logger)

    # A BUNDLE WITH NO AGGREGATE TABLES IS A FAILED RUN, NOT A THIN ONE.
    # `check_bundle` validates the files it finds and cannot tell an empty
    # counts/ from a suppressed one, so an export that destroyed or never
    # received the tables looked identical to a clean one. `filter_ledger.csv`
    # is written by `stage_cohort` for every site unconditionally, so its
    # absence means the tables are not here -- either the stages that author
    # them have not run, or something removed them between then and now.
    if not sorted(bundle.glob("counts/*.csv")):
        raise contract.ContractViolation(
            f"{bundle.name} holds no counts/*.csv, so there is nothing to "
            "send. The aggregate tables are written into this folder by the "
            "earlier stages, not by export -- run the cohort stage onward, "
            "and do not clear the folder at export time."
        )

    # the legend, from the tokenizer alone -- no patient data, and identical at
    # every site, so it travels for the reader's convenience. Absent where the
    # site holds no tokenizer, in which case the bundle carries no token tables
    # for it to be a legend to.
    if paths.has_tokenizer:
        levels(str(paths.tokenizer_yaml)).write_csv(bundle / "token_levels.csv")
    else:
        logger.warning(
            " no tokenizer on this machine: token_levels.csv, "
            "token_prevalence_* and token_families_* are not in this bundle. "
            "Everything else is."
        )

    # The stage ledger, if the run left one. This is the only part of
    # output/logs/01_site_tables.log that can cross a site boundary: the log
    # prints the attrition ledger and the per-phenotype counts unsuppressed and
    # names data_dir, output_home and scratch_home as absolute local paths,
    # none of which is in the contract. A stage name, a status and a duration
    # are, and they answer the question a thin bundle actually raises, which is
    # whether the run finished. Copied rather than re-validated because it
    # holds no counts to suppress.
    ledger = paths.cohort_home / "run_stages.csv"
    if ledger.is_file():
        shutil.copyfile(ledger, bundle / "run_stages.csv")
        logger.info(f" run_stages.csv ({pl.read_csv(ledger).height} stages)")
    else:
        logger.info(
            " no run_stages.csv in output/cohort/ -- this bundle carries no "
            "stage ledger, so a reader cannot tell a complete run from a "
            "partial one except by what is in it"
        )

    # What tier 2 recorded, where it has already run. `build` writes
    # `shares_patient_level: false` and empty anchors because tier 1 does not
    # know about embeddings -- correct on a first export, wrong on a re-export
    # over a bundle whose patient/ directory is sitting right there. Left
    # unhandled, re-running tier 1 alone to pick up a table left the embeddings
    # on disk and the manifest denying them, which `bundles.discover` reads as
    # an incomplete copy.
    existing = {}
    if (bundle / manifest.MANIFEST_NAME).is_file() and (bundle / "patient").is_dir():
        try:
            previous = manifest.read(bundle)
        except Exception:
            previous = {}
        if previous.get("shares_patient_level"):
            existing = {
                "anchors": tuple(previous.get("anchors") or ()),
                "row_counts": dict(previous.get("row_counts") or {}),
                "shares_patient_level": True,
            }
            logger.info(
                "  carrying forward the tier-2 fields already in this "
                f"bundle's manifest (anchors {', '.join(existing['anchors'])})"
            )

    manifest.write(
        manifest.build(
            site=site,
            salt=salt,
            model_key=paths.model_key,
            # tier 1 reads the tokenizer, never the weights, so a site asked
            # only for the aggregate tables may legitimately hold no
            # checkpoint. `model_home_or_none` records a null config hash in
            # that case rather than refusing to write the bundle; the
            # tokenizer hash, which is the one the token counts depend on,
            # still travels.
            model_home=paths.model_home_or_none,
            tokenizer_yaml=paths.tokenizer_yaml_or_none,
            anchors=existing.get("anchors", tuple()),
            row_counts=existing.get("row_counts", {}),
            min_cell=min_cell,
            embedding_dim=embedding_dim,
            shares_patient_level=existing.get("shares_patient_level", False),
        ),
        bundle,
    )
    logger.info(f" manifest.json written; {site} aggregate bundle complete")
    return bundle


def write_purity(
    bundle: pathlib.Path,
    purity: pl.DataFrame | None,
    min_cell: int,
    logger,
) -> None:
    """
    add purity/ to a bundle tier 1 already wrote

    Tier 2 rather than tier 1 because it needs the embeddings. It is still an
    aggregate -- one row per scheme per phenotype, holding a neighbour share
    and the share chance would give -- so it is suppressed and checked like any
    other count table, and a site sharing nothing patient-level could in
    principle contribute it if it had a GPU. It does not, under this split,
    because the stage that computes it is the one gated on the agreement.
    """
    if purity is None or not purity.height:
        logger.warning(
            " purity/ omitted: no embeddings available, so phenotype coherence "
            "in GEM space could not be measured. Run --stage embed first."
        )
        return
    home = bundle / "purity"
    home.mkdir(parents=True, exist_ok=True)
    # Suppressed, THEN validated. It was only ever validated, so a site with a
    # phenotype of five patients did not leak -- it hit ContractViolation and
    # the whole tier-2 export died, naming a column and no remedy. Within a
    # scheme the labels partition the cohort, which is the group a single
    # blanked cell is recoverable from.
    purity = suppress.suppress_counts(
        purity, min_cell=min_cell, by=("site", "anchor", "scheme"), logger=logger
    )
    contract.validate_counts_frame(purity, "phenotype purity", min_cell)
    purity.write_csv(home / "phenotype_purity.csv")
    logger.info(
        f" purity/phenotype_purity.csv: {purity.height} rows "
        f"({purity['scheme'].n_unique()} schemes)"
    )


# ============================================================== tier 2: patient


def write_patient(
    paths,
    cohort: pl.DataFrame,
    site: str,
    bundle: pathlib.Path,
    anchors: tuple[str, ...],
    salt: str,
    min_cell: int,
    embedding_dim: int,
    logger,
    *,
    with_eligibility: bool = True,
) -> pathlib.Path:
    """
    the DUA-gated bundle: embeddings and the minimal ordered token labels

    Requires tier 1 to have run, and checks that it ran under the same model and
    the same salt. Not a formality: a tier-2 export whose `row_key`s were
    generated from a different salt cannot be joined to the tier-1 counts, and a
    coordinating site would have no way to notice.
    """
    existing = bundle / manifest.MANIFEST_NAME
    if not existing.is_file():
        raise FileNotFoundError(
            f"{existing} not found. Tier 1 (the aggregate bundle) has to be "
            "written first -- it is what every site sends, and tier 2 only adds "
            "patient-level data to it. Run 01_site_tables.py."
        )
    prior = manifest.read(bundle)
    if prior.get("salt_fingerprint") != manifest.salt_fingerprint(salt):
        raise ValueError(
            f"the tier-1 bundle in {bundle} was written with a different "
            "pseudonymisation salt, so its counts and these row_keys describe "
            "the same patients under different identifiers. Re-run tier 1."
        )
    if prior.get("model", {}).get("key") != paths.model_key:
        raise ValueError(
            f"the tier-1 bundle was written for model "
            f"{prior.get('model', {}).get('key')!r}, not {paths.model_key!r}"
        )

    patient_home = bundle / "patient"
    patient_home.mkdir(parents=True, exist_ok=True)

    row_counts: dict[str, int] = {}
    coverages = []
    for anchor in anchors:
        # Per anchor, not once per site. Eligibility alone is a property of the
        # patient, but the `t1_` fit statistics are not: they are read off the
        # physiology at the anchor, so the 24h and onset frames give different
        # answers. An earlier version hoisted this out of the loop, which under
        # the old eligibility-only export was correct and is now a bug -- it
        # would have written the onset fit against the 24h rows.
        eligibility = (
            trial_columns(paths, anchor, site, logger) if with_eligibility else None
        )
        frame, token_coverage = build_patient_frame(
            paths, cohort, site, anchor, salt, logger, eligibility
        )
        target = patient_home / f"embeddings_anchor-{anchor}.parquet"
        frame.write_parquet(target)
        row_counts[anchor] = frame.height
        coverages.append(token_coverage)
        logger.info(
            f"  {target.name}: {frame.height} rows x {len(frame.columns)} "
            f"columns ({embedding_dim}-d embedding + "
            f"{len(frame.columns) - embedding_dim} labels)"
        )

    # per-variable charting coverage, which is what catches a site missing a
    # variable entirely -- the figure would otherwise look fine, coloured by
    # nulls. Lands in counts/ because it is aggregate.
    if coverages:
        (bundle / "counts").mkdir(parents=True, exist_ok=True)
        # This table went out with no suppression and no validation at all. A
        # variable charted in three patients shipped `n_charted = 3`, and a
        # variable charted in all but nine shipped the nine by subtraction --
        # which is exactly what one Rush row did (5668 of 5677 for heart_rate).
        # One row per variable, so the rows do not partition patients: `None`
        # opts out of the down-column pass, and the (n_charted, n_total)
        # complement pair is what does the work here.
        table = suppress.suppress_counts(
            pl.concat(coverages), min_cell=min_cell, by=None, logger=logger
        )
        contract.validate_counts_frame(table, "token_coverage.csv", min_cell)
        table.write_csv(bundle / "counts" / "token_coverage.csv")

    amended = dict(prior)
    amended["shares_patient_level"] = True
    amended["anchors"] = list(anchors)
    amended["row_counts"] = row_counts
    amended["model"]["embedding_dim"] = embedding_dim
    # Tier 1 may have written a null config hash, which is correct there -- it
    # never loads the weights. Tier 2 did load them, so fill the hashes in now.
    # Without this a bundle carrying embeddings would travel with nothing for
    # `check_agreement` to compare, and pooling two checkpoints' vectors is the
    # exact failure that check exists to prevent.
    amended["model"]["config_sha256"] = manifest.file_digest(
        paths.model_home / "config.json"
    )
    weights = paths.model_home / "model.safetensors"
    if weights.is_file():
        amended["model"]["weights_sha256"] = manifest.file_digest(weights)
    manifest.write(amended, bundle)
    logger.info(f" manifest.json amended; {site} patient bundle complete")
    return bundle


# ========================================================= tier 2: token data


#: labels that travel beside the sequences so the coordinating site can colour
#: and stratify what it re-embeds. Same allowlist as everywhere else -- these
#: are not new permissions, they are the subset that makes sense next to a
#: sequence.
_TOKEN_BUNDLE_LABELS = (
    # cocoa's partition, carried so the coordinating site can reproduce the
    # same train/tuning/held_out split rather than inventing a new one
    "split",
    "cs_phenotype",
    "seymour_phenotype",
    "bhavani_phenotype",
    "shock_type",
    "sepsis_cohort",
    "cardiac_origin",
    "in_hospital_mortality",
    "icu_los_days",
    "sex_category",
    "hours_admission_to_cs_onset",
)

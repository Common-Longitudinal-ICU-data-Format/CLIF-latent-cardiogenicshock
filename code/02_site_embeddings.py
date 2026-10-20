#!/usr/bin/env python3

"""
02 -- site tier 2: the embeddings, which need an agreement

The second of the two scripts that run inside a contributing institution, and
the one a site may decline to run. **This is where the GEM model is loaded**:
01 needs the tokenizer but never the weights, so a site with no GPU, or with no
agreement permitting patient-level transfer, runs 01 and stops.

It adds the embeddings and the phenotype-coherence table to the bundle tier 1
already wrote, so that a site sends exactly **one** directory:

    output/upload_to_box_<SITE>/
        manifest.json
        counts/      tier 1: patient flow, phenotype breakdowns, token prevalence
        trials/      tier 1: eligibility, IQR position and concordance summaries
        purity/      TIER 2: phenotype coherence in this site's GEM space
        patient/     TIER 2: one row per patient per anchor, 1024-d embedding
                     plus the labels needed to colour and stratify it

Stages, each rerunnable on its own with --stage:

  embed    run the frozen model          -> output/patient_level_data_<SITE>/
  purity   per-phenotype neighbour purity in GEM space  -> output/tables/
  export   redact to the contract and add patient/ + purity/ to the bundle

Everything patient-level that the coordinating site needs is in that directory.
The embeddings the model actually produced stay in
`output/patient_level_data_<SITE>/` and are never sent: those carry
`hospitalization_id`, `patient_id` and the onset timestamp, and the copy in
`patient/` is a redaction of them.

WHAT TRAVELS, AND WHAT DOES NOT

Each exported row is keyed by `row_key` -- SHA-256 over a site-local salt and
the record number, truncated to 128 bits -- and carries the 1024-d vector, the
three phenotype labels, the coded shock type, the outcomes, per-trial
eligibility, fit against the published interquartile ranges, treatment
concordance, token-family counts, the last pre-anchor token *bin* for ten
variables, and age as a five-year band topped at 90+.

It does not carry the record number, the patient id, any timestamp, or any
measured value. `clifcs/export/contract.py` is the allowlist, every column on
it has a written argument for being there, and `write_patient` validates
against it and raises rather than dropping.

**Schema 3.0 no longer ships token sequences.** Earlier versions sent
`tokens_past` and `s_elapsed_past` so the coordinating site could re-embed,
which bought re-anchoring, reading another layer, and re-running after a model
update without asking anyone to re-export. They were withdrawn because a
sequence is a re-serialisation of the record and is strictly more disclosive
than the embedding derived from it. Sites embed locally now and ship the
vector; the contract refuses the two columns by name.

WHAT THIS IS NOT

De-identified. A 1024-d embedding is a learned function of the patient's
entire recorded history, and no one has shown that such representations resist
re-identification attack. Treat the export as a limited data set at best and
move it under the IRB approval and data use agreement you would use to share
patient-level clinical data.

BEFORE RUNNING IT

Confirm that approval and that agreement are in place. If they are not, do not
run this. Tier 1 is a complete contribution on its own -- the consortium still
gets patient flow, the phenotype breakdowns, the trial comparison and the
token-prevalence tables 04 decomposes the site effect with -- and the pooled
analysis degrades rather than failing: a tier-1-only site appears in every
count table and in no projection.

Separated from 01 for that reason. A flag on one command invites somebody to
pass it without the question having been asked; a second script does not.

Requires 01 to have run first, and checks that it ran under the same salt -- a
tier-2 export whose row_keys came from a different salt cannot be joined to the
tier-1 counts, and the coordinating site would have no way to notice.

Usage:

    python code/02_site_embeddings.py
    python code/02_site_embeddings.py --site UCMC --anchor 24h
    python code/02_site_embeddings.py --stage embed --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import polars as pl

from clifcs import ANCHORS, get_logger, load_paths
from clifcs.export import contract, suppress, writer

STAGES = ("embed", "purity", "export")


# ==================================================================== embed


def stage_embed(paths, sites, anchors, logger, force: bool, batch_size=None) -> dict:
    """
    run the frozen model over each site x anchor

    Writes to `patient_level_data_<SITE>/`, which is LOCAL ONLY -- these carry
    the record number, the patient id and the onset timestamp. The redacted
    copy that travels is assembled by the export stage.
    """
    from clifcs import embedding

    info = embedding.describe_model(paths, logger)
    paths.table_home.mkdir(parents=True, exist_ok=True)
    (paths.table_home / "model_summary.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    # one row per patient: the embedding plus everything needed to colour it
    covariates = cohort.select(
        [
            c
            for c in (
                "hospitalization_id", "patient_id", "site", "cs_onset_dttm",
                "cs_phenotype", "seymour_phenotype", "bhavani_phenotype",
                "sepsis_cohort", "icu_los_days", "in_hospital_mortality",
                "cardiac_origin", "shock_type", "age_at_admission",
                "sex_category", "hours_admission_to_cs_onset",
            )
            if c in cohort.columns
        ]
    )

    for anchor in anchors:
        for site in sites:
            local = paths.patient_level_home_for(site)
            local.mkdir(parents=True, exist_ok=True)
            out = local / f"embeddings_anchor-{anchor}.parquet"
            if out.is_file() and not force:
                logger.info(f"[{anchor}/{site}] {out.name} exists; skipping")
                continue
            frame = embedding.to_embedding_frame(
                embedding.extract_site_anchor(paths, site, anchor, batch_size, logger),
                info["embedding_dim"],
            )
            frame = frame.join(
                covariates,
                on=["hospitalization_id", "site"],
                how="inner",
                validate="1:1",
            )
            frame.write_parquet(out)
            logger.info(
                f"[{anchor}/{site}] wrote {frame.height} patient embeddings "
                f"({info['embedding_dim']}-d) -> {out.name}"
            )
    return info


# =================================================================== purity


def stage_purity(paths, sites, anchors, logger, min_cell: int) -> pl.DataFrame:
    """
    how coherent each phenotype is in this site's embedding space

    For each phenotype, the share of a patient's 30 nearest neighbours (in
    this site's own 50-d PCA space) carrying the same phenotype, against the share
    chance would give. Computed per site and reported per phenotype, and it
    returns scalars -- so it answers whether the phenotypes correspond to
    anything the model sees, without any embedding moving.

    It lives in tier 2 because it needs the embeddings, which means a
    tier-1-only site does not report it. Under the previous split it sat in
    tier 1 and forced that script to need a GPU.

    Each site fits its own PCA, so the 50 dimensions differ between sites and
    only `lift` is comparable -- it is already normalised by local phenotype
    prevalence. Raw purity is not.
    """
    from clifcs.purity import phenotype_purity

    # The labels are joined from the cohort table rather than read off the
    # embedding file. An embedding file carries whichever covariates were
    # current when it was written, so a scheme added later would be silently
    # absent from the purity table -- which is exactly what happened when the
    # two sepsis phenotypes landed: the measure reported three schemes instead
    # of five and looked complete.
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    label_columns = [
        c
        for c in (
            "cs_phenotype", "seymour_phenotype", "bhavani_phenotype",
            "shock_type", "in_hospital_mortality",
        )
        if c in cohort.columns
    ]
    labels = cohort.select("hospitalization_id", "site", *label_columns)

    frames = []
    for anchor in anchors:
        for site in sites:
            source = (
                paths.patient_level_home_for(site)
                / f"embeddings_anchor-{anchor}.parquet"
            )
            if not source.is_file():
                logger.warning(
                    f"[{site}/{anchor}] no embeddings; run --stage embed first"
                )
                continue
            frame = pl.read_parquet(source).drop(label_columns, strict=False).join(
                labels, on=["hospitalization_id", "site"], how="left"
            )
            table = phenotype_purity(frame, anchor, site, logger, min_cell=min_cell)
            if table.height:
                frames.append(table)

    if not frames:
        return pl.DataFrame()
    purity = pl.concat(frames, how="diagonal_relaxed")
    paths.table_home.mkdir(parents=True, exist_ok=True)
    purity.write_csv(paths.table_home / "phenotype_purity.csv")
    logger.info(f"wrote {paths.table_home / 'phenotype_purity.csv'}")
    return purity


# =================================================================== export


def stage_export(
    paths, sites, anchors, logger, min_cell: int, purity, with_eligibility: bool
) -> None:
    """redact to the contract and add patient/ and purity/ to the tier-1 bundle"""
    logger.info(contract.describe())

    cohort_file = paths.cohort_home / "cohort.parquet"
    if not cohort_file.is_file():
        raise FileNotFoundError(
            f"{cohort_file} not found; run 01_site_tables.py first"
        )
    cohort = pl.read_parquet(cohort_file)

    # The same salt tier 1 used, so the patient rows and the summary tables
    # describe the same patients under the same pseudonyms. write_patient
    # refuses the export if the tier-1 manifest disagrees.
    salt = writer.load_salt(paths.id_salt_file, logger)

    summary = paths.table_home / "model_summary.json"
    embedding_dim = (
        json.loads(summary.read_text(encoding="utf-8")).get("embedding_dim", 0)
        if summary.is_file()
        else 0
    )

    for site in sites:
        bundle = paths.upload_home(site)
        if not bundle.is_dir():
            raise FileNotFoundError(
                f"{bundle} does not exist. 02 adds to the bundle 01 wrote; run "
                "01_site_tables.py first."
            )
        logger.info(f"=== {site} -> {bundle}")
        writer.write_purity(
            bundle,
            purity.filter(pl.col("site") == site) if purity.height else purity,
            min_cell,
            logger,
        )
        writer.write_patient(
            paths,
            cohort,
            site,
            bundle,
            anchors,
            salt,
            min_cell,
            embedding_dim,
            logger,
            with_eligibility=with_eligibility,
        )


# ==================================================================== driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("all", *STAGES), default="all", help="run a single stage"
    )
    parser.add_argument(
        "--from-stage", choices=STAGES, default=None, help="run this stage onward"
    )
    parser.add_argument("--site", default=None, help="one site only")
    parser.add_argument(
        "--anchor", choices=ANCHORS, default=None, help="one anchor only"
    )
    parser.add_argument("--model", default=None, help="which model's embeddings")
    parser.add_argument(
        "--force", action="store_true", help="redo stages whose outputs already exist"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="embedding batch size; 8 fits an 8 GB card, drop to 2-4 on CUDA "
        "OOM (output_hidden_states keeps every layer)",
    )
    parser.add_argument(
        "--min-cell",
        type=int,
        default=suppress.DEFAULT_MIN_CELL,
        help="minimum publishable cell size, matching the tier-1 run",
    )
    parser.add_argument(
        "--no-eligibility",
        action="store_true",
        help=(
            "skip the per-trial eligibility, fit and concordance columns. They "
            "are read from the parquet the trials stage of 01 wrote, so this "
            "only helps if that stage has not run; the pooled trial analysis "
            "loses those trials rather than recomputing them"
        ),
    )
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    logger = get_logger(
        "02_site_embeddings", paths.log_home / "02_site_embeddings.log"
    )

    if args.site and args.site not in paths.sites:
        parser.error(f"--site {args.site} is not configured; have {paths.sites}")
    sites = (args.site,) if args.site else paths.sites
    anchors = (args.anchor,) if args.anchor else ANCHORS

    if args.from_stage:
        todo = STAGES[STAGES.index(args.from_stage) :]
    elif args.stage == "all":
        todo = STAGES
    else:
        todo = (args.stage,)
    logger.info("stages: " + ", ".join(todo) + " | sites: " + ", ".join(sites))

    if "embed" in todo:
        stage_embed(paths, sites, anchors, logger, args.force, args.batch_size)

    # purity feeds the export, so it is recomputed when either stage runs rather
    # than read back from disk -- a stale table would be exported silently
    purity = pl.DataFrame()
    if "purity" in todo or "export" in todo:
        purity = stage_purity(paths, sites, anchors, logger, args.min_cell)
    if "export" in todo:
        stage_export(
            paths,
            sites,
            anchors,
            logger,
            args.min_cell,
            purity,
            with_eligibility=not args.no_eligibility,
        )
        logger.info(
            "02 done. output/upload_to_box_<SITE>/ now carries patient/ and "
            "purity/ as well, and is the ONE directory to send -- everything "
            "the pooled analysis needs is in it. It is NOT de-identified: move "
            "it under the same IRB approval and DUA you would use for "
            "patient-level clinical data. The unredacted embeddings in "
            "output/patient_level_data_<SITE>/ stay here."
        )
    else:
        logger.info("02 done")


if __name__ == "__main__":
    main()

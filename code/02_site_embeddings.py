#!/usr/bin/env python3

"""
02 -- site tier 2: the embeddings, which need an agreement

The second of the two scripts that run inside a contributing institution, and
the one a site may decline to run. It adds the GEM embeddings to the bundle
tier 1 already wrote, so that a site sends exactly **one** directory:

    output/upload_to_box_<SITE>/
        manifest.json
        counts/      tier 1: patient flow, phenotype breakdowns, coverage
        purity/      tier 1: phenotype coherence
        trials/      tier 1: eligibility and IQR-position summaries
        patient/     TIER 2: one row per patient per anchor, 1024-d embedding
                     plus the labels needed to colour and stratify it

Everything patient-level that the coordinating site needs is in that
directory. The embeddings the model actually produced stay in
`output/patient_level_data_<SITE>/` and are never sent: those carry
`hospitalization_id`, `patient_id` and the onset timestamp, and the copy in
`patient/` is a redaction of them.

WHAT TRAVELS, AND WHAT DOES NOT

Each exported row is keyed by `row_key` -- SHA-256 over a site-local salt and
the record number, truncated to 128 bits -- and carries the 1024-d vector, the
three phenotype labels, the coded shock type, the outcomes, trial eligibility
booleans, token-family counts, the last pre-anchor token *bin* for ten
variables, and age as a five-year band topped at 90+.

It does not carry the record number, the patient id, any timestamp, or any
measured value. `clifcs/export/contract.py` is the allowlist, every column on
it has a written argument for being there, and `write_patient` validates
against it and raises rather than dropping.

WHAT THIS IS NOT

De-identified. A 1024-d embedding is a learned function of the patient's
entire recorded history, and no one has shown that such representations resist
re-identification attack. Treat the export as a limited data set at best and
move it under the IRB approval and data use agreement you would use to share
patient-level clinical data.

BEFORE RUNNING IT

Confirm that approval and that agreement are in place. If they are not, do not
run this. Tier 1 is a complete contribution on its own -- the consortium still
gets patient flow, the three phenotype breakdowns and the phenotype-coherence
measure -- and the pooled analysis degrades rather than failing: a tier-1-only
site appears in every count table and in no projection.

Separated from 01 for that reason. A flag on one command invites somebody to
pass it without the question having been asked; a second script does not.

Requires 01 to have run first, and checks that it ran under the same model and
the same salt -- a tier-2 export whose row_keys came from a different salt
cannot be joined to the tier-1 counts, and the coordinating site would have no
way to notice.

Usage:

    python code/02_site_embeddings.py
    python code/02_site_embeddings.py --site UCMC --anchor 24h
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default=None, help="one site only")
    parser.add_argument(
        "--anchor", choices=ANCHORS, default=None, help="one anchor only"
    )
    parser.add_argument("--model", default=None, help="which model's embeddings")
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
            "skip the trial eligibility booleans. They read raw CLIF, which "
            "only a site can do, so the pooled trial analysis loses those "
            "trials rather than recomputing them"
        ),
    )
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    logger = get_logger(
        "02_site_embeddings", paths.log_home / "02_site_embeddings.log"
    )
    logger.info(contract.describe())

    if args.site and args.site not in paths.sites:
        parser.error(f"--site {args.site} is not configured; have {paths.sites}")
    sites = (args.site,) if args.site else paths.sites
    anchors = (args.anchor,) if args.anchor else ANCHORS

    cohort_file = paths.cohort_home / "cohort.parquet"
    if not cohort_file.is_file():
        parser.error(f"{cohort_file} not found; run 01_site_aggregate.py first")
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
        logger.info(f"=== {site} -> {bundle}")
        writer.write_patient(
            paths,
            cohort,
            site,
            bundle,
            anchors,
            salt,
            args.min_cell,
            embedding_dim,
            logger,
            with_eligibility=not args.no_eligibility,
        )

    logger.info(
        "02 done. output/upload_to_box_<SITE>/ now carries patient/ as well, "
        "and is the ONE directory to send -- everything the pooled analysis "
        "needs is in it. It is NOT de-identified: move it under the same IRB "
        "approval and DUA you would use for patient-level clinical data. The "
        "unredacted embeddings in output/patient_level_data_<SITE>/ stay here."
    )


if __name__ == "__main__":
    main()

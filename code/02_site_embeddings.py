#!/usr/bin/env python3

"""
02 -- site tier 2: the patient-level export, which needs an agreement

The second of the two scripts that run inside a contributing institution, and
the one a site may decline to run. It adds the patient-level data to the bundle
01 already wrote: one row per patient per anchor, holding the 1024-d embedding,
the ordered token bins, the trial eligibility booleans and the prediction
labels.

READ THIS BEFORE RUNNING IT

What this writes is **patient-level data and is not de-identified.** A 1024-d
embedding is a learned function of the patient's entire recorded history, and
no one has shown that such representations resist re-identification attack.
Record numbers are replaced by a salted hash, timestamps are dropped and age is
reduced to five-year bands, which narrows what travels -- it does not make the
file anonymous.

Treat the output as a limited data set at best, and move it under the same IRB
approval and data use agreement you would use to share patient-level clinical
data. Confirm that is in place before running this. If it is not, do not run
it: everything in 01 still goes, the consortium still gets patient flow, the
phenotype breakdowns and the phenotype-coherence measure, and the pooled side
degrades rather than failing.

Separated from 01 for exactly that reason. A flag on one command invites
somebody to pass it without the question having been asked; a second script
that has to be invoked deliberately does not.

Requires 01 to have run first, and checks that it ran under the same model and
the same pseudonymisation salt -- otherwise the counts and these row_keys
describe the same patients under different identifiers, and a coordinating site
would have no way to notice.

Usage
    python code/02_site_embeddings.py
    python code/02_site_embeddings.py --site UCMC --anchor 24h
    python code/02_site_embeddings.py --no-eligibility   # skip the trial scan
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import polars as pl

from clifcs import ANCHORS, get_logger, load_paths
from clifcs.export import contract, writer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default=None, help="one site only")
    parser.add_argument(
        "--anchor", choices=ANCHORS, default=None, help="one anchor only"
    )
    parser.add_argument(
        "--no-eligibility",
        action="store_true",
        help="skip the trial eligibility columns. They need a scan of the raw "
        "CLIF tables, so this is the flag to reach for when those are slow or "
        "when 11 is not part of the analysis.",
    )
    parser.add_argument(
        "--min-cell",
        type=int,
        default=11,
        help="minimum publishable cell size, for the coverage table (default 11)",
    )
    parser.add_argument("--model", default=None, help="which model's embeddings")
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    logger = get_logger("02_site_embeddings", paths.log_home / "02_site_embeddings.log")
    logger.info(contract.describe())

    if args.site and args.site not in paths.sites:
        parser.error(f"--site {args.site} is not configured; have {paths.sites}")
    sites = (args.site,) if args.site else paths.sites
    anchors = (args.anchor,) if args.anchor else ANCHORS

    cohort_file = paths.cohort_home / "cohort.parquet"
    if not cohort_file.is_file():
        parser.error(f"{cohort_file} not found; run 01_site_aggregate.py first")
    cohort = pl.read_parquet(cohort_file)
    if "sepsis_cohort" not in cohort.columns:
        logger.warning(
            "the cohort table predates the sepsis phenotypes; re-run "
            "01_site_aggregate.py --stage cohort to populate them"
        )
        cohort = cohort.with_columns(sepsis_cohort=pl.lit(None, dtype=pl.Boolean))

    salt = writer.load_salt(paths.id_salt_file, logger)
    summary = paths.table_home / "model_summary.json"
    embedding_dim = (
        json.loads(summary.read_text(encoding="utf-8")).get("embedding_dim", 0)
        if summary.is_file()
        else 0
    )

    for site in sites:
        bundle = paths.export_home / site
        logger.info(f"=== {site} -> {bundle / 'patient'}")
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
        "02 done. output/export/<SITE>/patient/ is NOT de-identified -- move it "
        "under the same IRB approval and DUA you would use for patient-level "
        "clinical data."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
02 -- site tier 2: the token sequences, which need an agreement

The second of the two scripts that run inside a contributing institution, and
the one a site may decline to run. It writes one row per patient per anchor
holding **the sequence of token ids the model read**, plus the elapsed seconds
at each one.

WHAT THIS IS, AND WHAT IT IS NOT

It is the model's entire input for that patient. It is **not** the raw record:
a token is a decile bin fitted at training time, so the stream says
`LAB-RES//creatinine_Q7` and never "2.3 mg/dL", and the wall-clock timestamps
sit in the same source file and are dropped -- `s_elapsed_past` carries time
relative to the start of the record instead.

It is also not de-identified, and must not be described as such. An embedding
is a lossy projection of a patient's history; a token stream is a
re-serialisation of it, and is the most disclosive artefact this pipeline
produces. Which labs were drawn, in what order, how often, against what clock,
is all still there.

WHY SEQUENCES RATHER THAN EMBEDDINGS

Because a fixed embedding answers one question and the sequence answers the ones
nobody has asked yet. With the sequences and the frozen model, the coordinating
site reproduces the embeddings exactly -- the model is deterministic in eval
mode -- and can additionally re-anchor at a different cut point, read a
different layer, or re-window, none of which a 1024-d vector permits. That is
why the embeddings stay local: they are derivable from what is sent.

BEFORE RUNNING IT

Confirm the IRB approval and data use agreement covering patient-level transfer
are in place. If they are not, do not run this. Tier 1 is a complete
contribution on its own -- the consortium still gets patient flow, the three
phenotype breakdowns and the phenotype-coherence measure, and the pooled
analysis degrades rather than failing.

Separated from 01 for that reason. A flag on one command invites somebody to
pass it without the question having been asked; a second script does not.

Outputs to `output/token_level_data_<SITE>/`.

Usage
    python code/02_site_tokens.py
    python code/02_site_tokens.py --site UCMC --anchor 24h
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--model", default=None, help="which model's tokenizer")
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    logger = get_logger("02_site_tokens", paths.log_home / "02_site_tokens.log")
    logger.info(contract.describe())

    if args.site and args.site not in paths.sites:
        parser.error(f"--site {args.site} is not configured; have {paths.sites}")
    sites = (args.site,) if args.site else paths.sites
    anchors = (args.anchor,) if args.anchor else ANCHORS

    cohort_file = paths.cohort_home / "cohort.parquet"
    if not cohort_file.is_file():
        parser.error(f"{cohort_file} not found; run 01_site_aggregate.py first")
    cohort = pl.read_parquet(cohort_file)

    # The same salt tier 1 used, so a token bundle and the summary tables
    # describe the same patients under the same pseudonyms.
    salt = writer.load_salt(paths.id_salt_file, logger)

    for site in sites:
        logger.info(f"=== {site} -> {paths.token_home(site)}")
        writer.write_tokens(paths, cohort, site, anchors, salt, logger)

    logger.info(
        "02 done. output/token_level_data_<SITE>/ is NOT de-identified -- it "
        "carries the model's full input sequence per patient. Move it under the "
        "same IRB approval and DUA you would use for patient-level clinical "
        "data."
    )


if __name__ == "__main__":
    main()

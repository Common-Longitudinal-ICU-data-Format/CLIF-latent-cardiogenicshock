#!/usr/bin/env python3

"""
12 -- assemble the bundle a site sends to the coordinating site

This is the **only** script that writes anything intended to leave an
institution, and the boundary is deliberately narrow: it reads the local cohort
table, the local embeddings and the local token streams, and writes a directory
whose every column has been checked against `clifcs.export.contract`.

What goes out
-------------
    export/<SITE>/
      manifest.json         schema version, code/model/tokenizer hashes
      counts/               aggregate tables, small-cell suppressed
      patient/              embeddings and the minimal ordered token labels
      token_levels.csv      the legend, derived from the tokenizer alone

`counts/` is aggregate and travels under the consortium's ordinary terms.
`patient/` is **patient-level data and is not de-identified** -- a 1024-d
embedding is a learned function of the patient's whole recorded history. It
moves under the same IRB approval and DUA as patient-level clinical data, or it
does not move; `--counts-only` produces a bundle without it for a site whose
agreement does not permit the transfer.

What does not go out, and cannot
--------------------------------
Raw records, timestamps, record numbers, ages in years, race and ethnicity, the
token *sequences*, and anything else not named in the contract. The check is an
allowlist and it raises rather than dropping, so a column added to the cohort
frame upstream stops this script instead of travelling silently.

Usage
    python code/12_export.py
    python code/12_export.py --counts-only        # no patient-level transfer
    python code/12_export.py --site UCMC --min-cell 20
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import secrets
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import polars as pl
from omegaconf import OmegaConf

from clifcs import ANCHORS, get_logger, load_paths
from clifcs.config import CONFIG_HOME
from clifcs.export import contract, manifest, suppress
from clifcs.tokenlabels import coverage, last_pre_anchor_ranks, levels
from clifcs.tokens import family_matrix, load_pre_anchor_tokens, vocabulary

#: aggregate tables copied from output/cohort/ into counts/, with the column
#: that holds a patient count named so suppression can find it
_COUNT_TABLES = {
    "filter_ledger.csv": ("n_hospitalizations", "n_patients"),
    "phenotype_summary.csv": ("n",),
    "shock_type_summary.csv": ("n",),
    "seymour_summary.csv": ("n",),
    "bhavani_summary.csv": ("n",),
    "seymour_availability.csv": ("n_measured",),
}

#: derived statistics that must be suppressed wherever their count is
_DERIVED = ("mortality", "median_icu_los_days", "share", "availability")


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
    one row per patient: embedding, ordered token bins, labels and outcomes

    Returns the frame and the per-variable token coverage table. Raises if the
    embeddings for this site and anchor have not been produced yet.
    """
    source = paths.patient_level_home / f"embeddings_anchor-{anchor}_{site}.parquet"
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} not found; run 02_model_embedding.py for {site}/{anchor} first"
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
        .with_columns(age_bands(embeddings), row_keys(embeddings, salt))
    )
    if eligibility is not None and eligibility.height:
        frame = frame.join(eligibility, on="hospitalization_id", how="left")

    # Drop everything the contract does not name. Done explicitly and then
    # validated, rather than relying on the selection being right: the
    # validation is what catches a column this list forgot.
    keep = [c for c in frame.columns if contract.is_permitted(c)]
    frame = frame.select(keep)
    contract.validate_patient_frame(frame, f"{site}/{anchor} patient frame")
    return frame, token_coverage


def trial_eligibility(paths, cohort: pl.DataFrame, site: str, logger) -> pl.DataFrame:
    """
    one boolean per trial, computed here because it cannot be computed there

    The eligibility predicates read `clif_vitals`, `clif_labs`,
    `clif_medication_admin_*`, `clif_hospital_diagnosis` and
    `clif_patient_procedures`. The coordinating site has none of those, so this
    is one of the things the federated split moves site-side: 11 used to call
    `attach_features(paths.site_data[site], ...)` itself, which a pooled-role run
    cannot do and should not be able to.

    Only the booleans travel. The features they were computed from -- pressures,
    lactate, the per-agent infusion flags -- stay here.
    """
    from clifcs.concordance import attach_procedures
    from clifcs.trials import add_eligibility, attach_features, load_trials

    trial_cfg = OmegaConf.load(CONFIG_HOME / "trials.yaml")
    cohort_cfg = OmegaConf.load(CONFIG_HOME / "cohort.yaml")
    trials = load_trials(trial_cfg)

    part = cohort.filter(pl.col("site") == site)
    if part.height == 0:
        return pl.DataFrame({"hospitalization_id": []})

    data_dir = paths.site_data[site]
    part = attach_features(data_dir, part, cohort_cfg, logger)
    part = attach_procedures(data_dir, part, logger)
    part = add_eligibility(part, trials)

    columns = [c for c in part.columns if c.startswith(contract.TRIAL_PREFIX)]
    logger.info(
        f"  trial eligibility: {len(columns)} trials "
        + ", ".join(
            f"{c.removeprefix(contract.TRIAL_PREFIX)} {int(part[c].sum())}"
            for c in columns
        )
    )
    return part.select("hospitalization_id", *columns)


def write_counts(paths, bundle: pathlib.Path, min_cell: int, logger) -> None:
    """copy the aggregate tables across, suppressed"""
    out = bundle / "counts"
    out.mkdir(parents=True, exist_ok=True)

    for name, count_columns in _COUNT_TABLES.items():
        source = paths.cohort_home / name
        if not source.is_file():
            logger.info(f"  {name}: not produced at this site, skipping")
            continue
        table = pl.read_csv(source)
        logger.info(f"  {name} ({table.height} rows)")
        table = suppress.suppress_counts(
            table,
            min_cell=min_cell,
            derived=_DERIVED,
            # a published row of these tables is one phenotype across the
            # sites, so complementary suppression works within a phenotype
            by=tuple(
                c
                for c in ("cs_phenotype", "seymour", "bhavani", "shock_type", "site")
                if c in table.columns
            ),
            logger=logger,
        )
        contract.validate_counts_frame(table, name, min_cell)
        table.write_csv(out / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", choices=ANCHORS, default=None, help="one anchor only")
    parser.add_argument("--site", default=None, help="one site only")
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="omit patient/ entirely -- for a site whose DUA does not permit "
        "patient-level transfer. The pooled side degrades rather than failing.",
    )
    parser.add_argument(
        "--min-cell",
        type=int,
        default=suppress.DEFAULT_MIN_CELL,
        help=f"minimum publishable cell size (default {suppress.DEFAULT_MIN_CELL})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="where to write the bundle (default output/export/<SITE>)",
    )
    parser.add_argument(
        "--no-eligibility",
        action="store_true",
        help="skip the trial eligibility columns. They need the raw CLIF "
        "tables, so this is the flag to reach for when those are slow or when "
        "11 is not part of the analysis.",
    )
    parser.add_argument("--model", default=None, help="which model's embeddings")
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    logger = get_logger("12_export", paths.log_home / "12_export.log")
    logger.info(contract.describe())

    if args.site and args.site not in paths.sites:
        parser.error(f"--site {args.site} is not configured; have {paths.sites}")
    sites = (args.site,) if args.site else paths.sites
    anchors = (args.anchor,) if args.anchor else ANCHORS

    cohort_file = paths.cohort_home / "cohort.parquet"
    if not cohort_file.is_file():
        parser.error(f"{cohort_file} not found; run 01_data_processing.py first")
    cohort = pl.read_parquet(cohort_file)
    if "sepsis_cohort" not in cohort.columns:
        logger.warning(
            "the cohort table predates the sepsis phenotypes; re-run "
            "01_data_processing.py --stage cohort to populate them"
        )
        cohort = cohort.with_columns(sepsis_cohort=pl.lit(None, dtype=pl.Boolean))

    salt = load_salt(paths.id_salt_file, logger)
    embedding_dim = sum(
        1 for c in pl.read_parquet_schema(
            paths.patient_level_home
            / f"embeddings_anchor-{anchors[0]}_{sites[0]}.parquet"
        )
        if contract.EMBEDDING_PATTERN.match(c)
    ) if not args.counts_only else 0

    for site in sites:
        bundle = pathlib.Path(
            args.out or (paths.output_home / "export" / site)
        )
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True)
        logger.info(f"=== {site} -> {bundle}")

        # ------------------------------------------------------------ counts
        logger.info(" counts/")
        write_counts(paths, bundle, args.min_cell, logger)

        # ----------------------------------------------------------- patient
        row_counts: dict[str, int] = {}
        if args.counts_only:
            logger.warning(
                " patient/ omitted (--counts-only): this bundle carries no "
                "embeddings, so this site contributes to every count table and "
                "to no projection"
            )
        else:
            patient_home = bundle / "patient"
            patient_home.mkdir()

            # computed once per site, not once per anchor: eligibility is a
            # property of the patient, which is the same argument 11 makes for
            # not recomputing it per anchor
            eligibility = None
            if not args.no_eligibility:
                eligibility = trial_eligibility(paths, cohort, site, logger)

            coverages = []
            for anchor in anchors:
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
            # per-variable charting coverage, which is what catches a site
            # missing a variable entirely -- the figure would otherwise look
            # fine, coloured by nulls
            pl.concat(coverages).write_csv(bundle / "counts" / "token_coverage.csv")

        # the legend, derived from the tokenizer alone -- no patient data, and
        # identical at every site, so it travels for the reader's convenience
        levels(str(paths.tokenizer_yaml)).write_csv(bundle / "token_levels.csv")

        manifest.write(
            manifest.build(
                site=site,
                salt=salt,
                model_key=paths.model_key,
                model_home=paths.model_home,
                tokenizer_yaml=paths.tokenizer_yaml,
                anchors=anchors,
                row_counts=row_counts,
                min_cell=args.min_cell,
                embedding_dim=embedding_dim,
                shares_patient_level=not args.counts_only,
            ),
            bundle,
        )
        logger.info(f" manifest.json written; {site} bundle complete")

    logger.info(
        "12 done. The patient/ directory is NOT de-identified -- move it under "
        "the same IRB approval and DUA you would use for patient-level clinical "
        "data. counts/ is aggregate and suppressed."
    )


if __name__ == "__main__":
    main()

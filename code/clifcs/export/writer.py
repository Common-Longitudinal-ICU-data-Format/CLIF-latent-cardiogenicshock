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
      counts/                tier 1: aggregate tables, suppressed
      purity/                tier 1: per-phenotype coherence in GEM space
      patient/               tier 2: embeddings and ordered token labels

What does not go out, and cannot: raw records, timestamps, record numbers, ages
in years, race and ethnicity, the token *sequences*, and anything else not named
in `clifcs.export.contract`. The check is an allowlist and it raises rather than
dropping, so a column added to the cohort frame upstream stops the export
instead of travelling silently.
"""

from __future__ import annotations

import hashlib
import pathlib
import secrets
import shutil

import polars as pl
from omegaconf import OmegaConf

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
            f"{source} not found; run 01_site_aggregate.py --stage embed for "
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
        .with_columns(age_bands(embeddings), row_keys(embeddings, salt))
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


def write_counts(paths, bundle: pathlib.Path, min_cell: int, logger, site=None) -> None:
    """copy the aggregate tables across, suppressed"""
    out = bundle / "counts"
    out.mkdir(parents=True, exist_ok=True)

    # the trial tables, which are per site and per anchor and so are found by
    # glob rather than named. Already suppressed by clifcs.trial_aggregate;
    # re-checked here rather than re-suppressed, so a table assembled by some
    # other route cannot skip it.
    if site is not None:
        trial_home = out.parent / "trials"
        for source in sorted(paths.cohort_home.glob(f"trial_*_{site}_anchor-*.csv")):
            table = pl.read_csv(source)
            contract.validate_counts_frame(table, source.name, min_cell)
            trial_home.mkdir(parents=True, exist_ok=True)
            table.write_csv(trial_home / source.name.replace(f"_{site}", ""))
            logger.info(f"  trials/{source.name.replace(f'_{site}', '')} ({table.height} rows)")
        criteria = paths.cohort_home / "trial_criteria.csv"
        if criteria.is_file():
            trial_home.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(criteria, trial_home / "trial_criteria.csv")

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


# ============================================================ tier 1: aggregate


def write_aggregate(
    paths,
    site: str,
    bundle: pathlib.Path,
    purity: pl.DataFrame | None,
    salt: str,
    min_cell: int,
    embedding_dim: int,
    logger,
) -> pathlib.Path:
    """
    the bundle every site sends: counts, phenotype breakdowns, purity

    Writes the manifest with `shares_patient_level: false`. Tier 2 amends it if
    and when it runs, so the presence of patient data is recorded in one place
    rather than inferred from a directory listing.
    """
    bundle.mkdir(parents=True, exist_ok=True)

    logger.info(" counts/")
    write_counts(paths, bundle, min_cell, logger, site=site)

    if purity is not None and purity.height:
        home = bundle / "purity"
        home.mkdir(parents=True, exist_ok=True)
        contract.validate_counts_frame(purity, "phenotype purity", min_cell)
        purity.write_csv(home / "phenotype_purity.csv")
        logger.info(
            f" purity/phenotype_purity.csv: {purity.height} rows "
            f"({purity['scheme'].n_unique()} schemes)"
        )
    else:
        logger.warning(
            " purity/ omitted: no embeddings available, so phenotype coherence "
            "in GEM space could not be measured. Run the embed stage first."
        )

    # the legend, from the tokenizer alone -- no patient data, and identical at
    # every site, so it travels for the reader's convenience
    levels(str(paths.tokenizer_yaml)).write_csv(bundle / "token_levels.csv")

    manifest.write(
        manifest.build(
            site=site,
            salt=salt,
            model_key=paths.model_key,
            model_home=paths.model_home,
            tokenizer_yaml=paths.tokenizer_yaml,
            anchors=tuple(),
            row_counts={},
            min_cell=min_cell,
            embedding_dim=embedding_dim,
            shares_patient_level=False,
        ),
        bundle,
    )
    logger.info(f" manifest.json written; {site} aggregate bundle complete")
    return bundle


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
            "patient-level data to it. Run 01_site_aggregate.py."
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

    # computed once per site, not once per anchor: eligibility is a property of
    # the patient, which is the same argument 11 makes for not recomputing it
    eligibility = (
        trial_eligibility(paths, cohort, site, logger) if with_eligibility else None
    )

    row_counts: dict[str, int] = {}
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

    # per-variable charting coverage, which is what catches a site missing a
    # variable entirely -- the figure would otherwise look fine, coloured by
    # nulls. Lands in counts/ because it is aggregate.
    if coverages:
        (bundle / "counts").mkdir(parents=True, exist_ok=True)
        pl.concat(coverages).write_csv(bundle / "counts" / "token_coverage.csv")

    amended = dict(prior)
    amended["shares_patient_level"] = True
    amended["anchors"] = list(anchors)
    amended["row_counts"] = row_counts
    amended["model"]["embedding_dim"] = embedding_dim
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


def write_tokens(
    paths,
    cohort: pl.DataFrame,
    site: str,
    anchors: tuple[str, ...],
    salt: str,
    logger,
) -> pathlib.Path:
    """
    tier 2: the ordered token sequence per patient

    What this is, stated plainly: the model's entire input for each patient, as
    the sequence of token ids it read before the anchor plus the elapsed seconds
    at each one. Not the measured values -- a token is a decile bin fitted at
    training time -- and not the wall-clock times, which sit in the same source
    file and are dropped.

    What it buys the coordinating site is the reason to prefer it over shipping
    embeddings: with the sequences and the frozen model it can reproduce the
    embeddings exactly, re-anchor at a different cut point, or read a different
    layer. A fixed embedding answers one question; the sequence answers the ones
    nobody has asked yet.

    What it costs is that this is the most disclosive artefact the pipeline
    produces. An embedding is a lossy projection of the record; a token stream
    is a re-serialisation of it. It must not be described as de-identified, and
    it moves under a patient-level data use agreement or not at all.
    """
    from clifcs.tokens import load_pre_anchor_tokens

    home = paths.token_home(site)
    home.mkdir(parents=True, exist_ok=True)

    labels = cohort.filter(pl.col("site") == site).select(
        "hospitalization_id",
        *[c for c in _TOKEN_BUNDLE_LABELS if c in cohort.columns],
    )

    row_counts: dict[str, int] = {}
    for anchor in anchors:
        tokens = load_pre_anchor_tokens(paths, site, anchor)
        frame = (
            tokens.join(labels, on="hospitalization_id", how="left")
            .with_columns(
                row_keys(tokens, salt),
                anchor=pl.lit(anchor),
            )
            .drop("hospitalization_id")
        )
        # Dropped explicitly and then validated, rather than trusting the
        # selection: the validation is what catches a column this forgot, and
        # the source file carries `times`, `race_category`, `ethnicity_category`
        # and `age_at_admission` right alongside the sequences.
        frame = frame.select([c for c in frame.columns if contract.is_permitted(c)
                              or c in contract.TOKEN_SEQUENCE_COLUMNS])
        contract.validate_token_frame(frame, f"{site}/{anchor} token bundle")

        out = home / f"tokens_anchor-{anchor}.parquet"
        frame.write_parquet(out)
        row_counts[anchor] = frame.height
        lengths = frame["tokens_past"].list.len()
        logger.info(
            f"  {out.name}: {frame.height} patients, "
            f"{int(lengths.sum()):,} tokens "
            f"(median {int(lengths.median())} per patient, "
            f"max {int(lengths.max())})"
        )

    # the vocabulary the ids index into. Model metadata, no patient data, and
    # the thing without which a bare list of integers is uninterpretable.
    levels(str(paths.tokenizer_yaml)).write_csv(home / "token_levels.csv")

    manifest.write(
        manifest.build(
            site=site,
            salt=salt,
            model_key=paths.model_key,
            model_home=paths.model_home,
            tokenizer_yaml=paths.tokenizer_yaml,
            anchors=anchors,
            row_counts=row_counts,
            min_cell=0,
            embedding_dim=0,
            shares_patient_level=True,
            notes="tier 2: token sequences, not embeddings. Re-embed with the "
            "frozen model at the coordinating site.",
        ),
        home,
    )
    logger.info(f" manifest.json written; {site} token bundle complete")
    return home

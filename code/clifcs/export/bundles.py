#!/usr/bin/env python3

"""
reading what the sites sent

The coordinating site's view of the world is a directory of bundles, one per
contributing site, written by `01_site_aggregate.py` and, where an agreement
permits it, extended by `02_site_embeddings.py`. This module discovers them,
checks them against the contract and against each other, and assembles the
pooled frame the numbered analyses read.

It is deliberately the *only* way the pooled side obtains patient-level data.
`Paths.site_data` raises under the pooled role, so there is no second path that
reaches records, and adding one means editing the contract.

What this checks, and why each check exists
-------------------------------------------
**Schema, model and tokenizer agreement** -- embeddings produced under different
weights or a different token mapping are not in the same space, and pooling them
makes a figure of nothing. Fatal.

**Code version agreement** -- the cohort definition lives in this repository, so
sites on different commits enrolled different populations. A warning, not an
error: it is recoverable by re-running one site, and the coordinator should be
able to look at what arrived.

**Column conformance** -- the same allowlist the sending site applied, applied
again on arrival. Checking only on write would be trusting every site to run the
current code; checking only on read would mean the data had already left.

**Duplicate pseudonyms** -- two bundles from the same site with different salts
describe the same patients under different `row_key`s, and pooling both
double-counts every one of them.
"""

from __future__ import annotations

import pathlib

import polars as pl

from clifcs.export import contract, manifest


#: the directory-name prefixes the two tiers are shipped under
TIER_PREFIXES = {"upload_to_box_": "aggregate", "token_level_data_": "token"}


def discover(received_home: pathlib.Path, logger) -> dict[str, dict[str, pathlib.Path]]:
    """
    the bundles present, grouped by site and then by tier

    A coordinating site drops what it received in verbatim, so the directories
    arrive named the way the sending site wrote them:

        received/
          upload_to_box_UCMC/        tier 1, summary only
          token_level_data_UCMC/     tier 2, token sequences
          upload_to_box_RUSH/        tier 1 only -- RUSH sent no tokens

    Returns `{site: {"aggregate": path, "token": path}}`, with either tier
    possibly absent. A site that sent only tier 1 is a first-class contributor
    here, not a degraded one: it appears in every count table and in the
    phenotype-coherence comparison, and in no projection.

    The site name comes from each bundle's manifest rather than from the
    directory, because a directory can be renamed in transit and the site that
    produced the data is a property of the data.
    """
    received_home = pathlib.Path(received_home)
    if not received_home.is_dir():
        raise FileNotFoundError(
            f"{received_home} does not exist. A coordinating site reads bundles "
            "from there; copy each contributing site's upload_to_box_<SITE>/ "
            "and, where it sent one, token_level_data_<SITE>/ into it."
        )

    found: dict[str, dict[str, pathlib.Path]] = {}
    for candidate in sorted(p for p in received_home.iterdir() if p.is_dir()):
        tier = next(
            (t for prefix, t in TIER_PREFIXES.items() if candidate.name.startswith(prefix)),
            None,
        )
        if tier is None:
            logger.info(
                f"  {candidate.name}/ is not a recognised bundle directory "
                f"(expected one of {', '.join(TIER_PREFIXES)}<SITE>); skipping"
            )
            continue
        if not (candidate / manifest.MANIFEST_NAME).is_file():
            logger.warning(
                f"  {candidate.name}/ has no {manifest.MANIFEST_NAME} and is "
                "being skipped -- a bundle without one cannot be checked for "
                "code, model or tokenizer agreement"
            )
            continue
        site = manifest.read(candidate).get("site") or candidate.name
        if tier in found.get(site, {}):
            raise ValueError(
                f"two {tier} bundles both claim to be site {site!r}: "
                f"{found[site][tier].name} and {candidate.name}. Pooling both "
                "would double-count every patient; keep the newer one."
            )
        found.setdefault(site, {})[tier] = candidate

    if not found:
        raise FileNotFoundError(f"no bundles with a manifest under {received_home}")

    for site, tiers in sorted(found.items()):
        logger.info(f"  {site}: {', '.join(sorted(tiers))}")
    return found


def manifests_of(bundles: dict[str, dict[str, pathlib.Path]]) -> dict[str, dict]:
    """
    one manifest per site, preferring the token bundle's where both exist

    They agree on model, tokenizer and salt by construction -- `check` asserts
    it -- so either will do; the token one is preferred because it carries the
    row counts that the patient-level read-back is validated against.
    """
    out = {}
    for site, tiers in bundles.items():
        home = tiers.get("token") or tiers.get("aggregate")
        out[site] = manifest.read(home)
    return out


def check(
    bundles: dict[str, dict[str, pathlib.Path]], paths, logger
) -> dict[str, dict]:
    """
    validate the bundles against each other and against the local model

    Returns one manifest per site.
    """
    manifests = manifests_of(bundles)
    manifest.check_agreement(manifests, logger)

    # The coordinating site also holds the weights, and should confirm the
    # bundles were made with the ones it is about to interpret them against --
    # otherwise the legend it renders from its own tokenizer describes different
    # bin edges from the ranks the sites exported.
    local_tokenizer = manifest.file_digest(paths.tokenizer_yaml)
    mismatched = [
        site
        for site, m in manifests.items()
        if m.get("tokenizer_sha256") not in (None, local_tokenizer)
    ]
    if mismatched:
        raise ValueError(
            f"bundle(s) from {', '.join(mismatched)} were produced with a "
            "different tokenizer from the one configured here. The exported "
            "token ranks are decile indices under the *sending* tokenizer's bin "
            "edges, so a legend rendered from this one would mislabel every "
            "point."
        )

    aggregate_only = sorted(s for s, tiers in bundles.items() if "token" not in tiers)
    if aggregate_only:
        logger.info(
            f"{', '.join(aggregate_only)} sent tier 1 only (no token transfer "
            "permitted), so they contribute to every count table and to the "
            "phenotype-coherence comparison, and to no projection"
        )
    return manifests


def load_patient_level(
    bundles: dict[str, dict[str, pathlib.Path]],
    manifests: dict[str, dict],
    anchor: str,
    logger,
    paths=None,
    batch_size=None,
) -> pl.DataFrame:
    """
    the pooled patient-level frame for one anchor, re-embedded from tokens

    Tier 2 ships the model's input, not its output, so this runs the frozen
    model over each received token bundle and returns embeddings. The result is
    cached under the coordinating site's own `patient_level_data/`, keyed by
    site and anchor, because re-embedding 34,000 sequences is a GPU job and
    every pooled script would otherwise redo it.

    The cache is invalidated on row count rather than mtime: a re-sent bundle
    with the same number of patients and different contents is the one case
    this misses, and the manifest's `created_utc` is what to check by hand if
    that is suspected.

    Concatenated `how="diagonal_relaxed"` so a site that sent a label another
    did not sits alongside it as null rather than failing the concatenation --
    but every column still has to be on the allowlist, which is what stops that
    leniency from becoming a hole.
    """
    from clifcs import reembed as reembed_module

    if paths is None:
        raise ValueError(
            "load_patient_level needs `paths` to re-embed received token "
            "bundles; pass the Paths object from the calling script"
        )

    frames = []
    for site, tiers in sorted(bundles.items()):
        home = tiers.get("token")
        if home is None:
            continue
        source = home / f"tokens_anchor-{anchor}.parquet"
        if not source.is_file():
            logger.warning(f"  {site}: no token file for anchor {anchor}, skipping")
            continue

        frame = pl.read_parquet(source)
        contract.validate_token_frame(frame, f"{site} token bundle, anchor {anchor}")

        expected = (manifests[site].get("row_counts") or {}).get(anchor)
        if expected is not None and frame.height != int(expected):
            raise ValueError(
                f"{site} anchor {anchor}: file holds {frame.height} rows but "
                f"its manifest says {expected} were exported -- the copy is "
                "incomplete"
            )
        if frame["site"].n_unique() != 1 or frame["site"][0] != site:
            raise ValueError(
                f"{site} bundle carries site column value(s) "
                f"{frame['site'].unique().to_list()}"
            )

        cache = (
            paths.patient_level_home
            / f"reembedded_anchor-{anchor}_{site}.parquet"
        )
        if cache.is_file() and pl.read_parquet_schema(cache) and (
            pl.scan_parquet(cache).select(pl.len()).collect().item() == frame.height
        ):
            logger.info(f"  {site}: reusing {cache.name} ({frame.height} patients)")
            embeddings = pl.read_parquet(cache)
        else:
            embeddings = reembed_module.reembed(
                paths, frame, site, anchor, logger, batch_size
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            embeddings.write_parquet(cache)
            logger.info(f"  {site}: re-embedded -> {cache.name}")

        # the labels travelled beside the sequences; the embeddings are what
        # was just derived from them
        labels = frame.drop("tokens_past", "s_elapsed_past")
        frames.append(labels.join(embeddings, on="row_key", how="inner"))
        logger.info(f"  {site}: {frames[-1].height} patients at anchor {anchor}")

    if not frames:
        raise FileNotFoundError(
            f"no bundle carries token data for anchor {anchor}. Every "
            "contributing site sent tier 1 only, so the projection analyses "
            "cannot run -- the count tables, the phenotype breakdowns and the "
            "phenotype-coherence comparison still can."
        )

    pooled = pl.concat(frames, how="diagonal_relaxed")

    duplicates = pooled.height - pooled[contract.ROW_KEY].n_unique()
    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate {contract.ROW_KEY} value(s) in the pooled "
            "frame. The pseudonym is a salted hash of the local record number, "
            "so a collision across sites is implausible and a repeat within one "
            "means the same bundle was ingested twice."
        )
    logger.info(
        f"pooled anchor {anchor}: {pooled.height} patients from "
        f"{pooled['site'].n_unique()} site(s)"
    )
    return pooled


def load_counts(
    bundles: dict[str, dict[str, pathlib.Path]], name: str, logger
) -> pl.DataFrame | None:
    """
    one aggregate table, stacked across every bundle that carries it

    Returns None when no bundle has it, which is the normal case for a table a
    site could not produce -- a missing phenotype breakdown at one site is a
    fact about that site, not a reason to fail.
    """
    frames = []
    for site, tiers in sorted(bundles.items()):
        home = tiers.get("aggregate")
        if home is None:
            continue
        source = home / "counts" / name
        if not source.is_file():
            continue
        frame = pl.read_csv(source)
        if "site" not in frame.columns:
            frame = frame.with_columns(site=pl.lit(site))
        frames.append(frame)

    if not frames:
        logger.info(f"  {name}: no bundle carries it")
        return None
    return pl.concat(frames, how="diagonal_relaxed")


def load_purity(
    bundles: dict[str, dict[str, pathlib.Path]], logger
) -> pl.DataFrame | None:
    """
    the per-phenotype coherence tables, stacked across bundles

    These come from tier 1, so every bundle has them -- including one from a
    site that sent no patient-level data at all. That is the point of measuring
    purity site-side: a site that cannot share embeddings can still say whether
    the phenotypes mean anything in its own embedding space.

    Not pooled into a single number. Each site fitted its PCA on its own
    patients, so the spaces differ; `lift` is the comparable column because it
    is already normalised by local phenotype prevalence, and even that is a
    per-site statement. See `clifcs.purity` for why.
    """
    frames = []
    for site, tiers in sorted(bundles.items()):
        home = tiers.get("aggregate")
        if home is None:
            continue
        source = home / "purity" / "phenotype_purity.csv"
        if not source.is_file():
            logger.info(f"  {site}: no purity table")
            continue
        frames.append(pl.read_csv(source))
    if not frames:
        return None
    pooled = pl.concat(frames, how="diagonal_relaxed")
    logger.info(
        f"phenotype purity from {pooled['site'].n_unique()} site(s), "
        f"{pooled['scheme'].n_unique()} scheme(s)"
    )
    return pooled


def token_legend(
    bundles: dict[str, dict[str, pathlib.Path]]
) -> pl.DataFrame | None:
    """
    the rank -> value-range legend

    Every bundle carries an identical copy, since it is derived from the shared
    tokenizer and contains no patient data, so the first one found is taken and
    the rest are checked against it.
    """
    legends = {}
    for site, tiers in sorted(bundles.items()):
        for home in (tiers.get("aggregate"), tiers.get("token")):
            if home is not None and (home / "token_levels.csv").is_file():
                legends[site] = pl.read_csv(home / "token_levels.csv")
                break
    if not legends:
        return None
    reference_site, reference = next(iter(legends.items()))
    for site, other in legends.items():
        if not other.equals(reference):
            raise ValueError(
                f"the token legend in {site}'s bundle differs from "
                f"{reference_site}'s. Both are derived from tokenizer.yaml, so "
                "they can only differ if the sites used different tokenizers -- "
                "in which case the exported ranks are not comparable."
            )
    return reference


def load_trial_tables(
    bundles: dict[str, dict[str, pathlib.Path]], anchor: str, logger
) -> dict[str, pl.DataFrame]:
    """
    the aggregate trial tables, stacked across every bundle that carries them

    These come from tier 1, so a site that shared no patient-level data at all
    still contributes its whole trial comparison -- eligibility, position
    against the published interquartile ranges, treatment concordance,
    mortality and ICU length of stay, each broken down by phenotype.

    Not pooled into single figures here. A share is over that site's own
    denominator, and the denominators differ: the sepsis phenotypes rest on a
    diagnosis code used 1.3x more often at NU than at UCMC. Pool by summing
    `n_eligible` against `n_stratum` if a combined share is wanted, and never by
    averaging the shares.
    """
    out: dict[str, pl.DataFrame] = {}
    for name in ("trial_outcomes", "trial_iqr_position"):
        frames = []
        for site, tiers in sorted(bundles.items()):
            home = tiers.get("aggregate")
            if home is None:
                continue
            source = home / "trials" / f"{name}_anchor-{anchor}.csv"
            if not source.is_file():
                continue
            frames.append(pl.read_csv(source))
        if frames:
            out[name] = pl.concat(frames, how="diagonal_relaxed")
            logger.info(
                f"  {name} anchor {anchor}: {out[name].height} rows from "
                f"{out[name]['site'].n_unique()} site(s)"
            )
        else:
            logger.info(f"  {name} anchor {anchor}: no bundle carries it")
    return out

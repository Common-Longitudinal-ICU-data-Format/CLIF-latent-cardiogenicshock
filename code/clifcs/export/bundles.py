#!/usr/bin/env python3

"""
reading what the sites sent

The coordinating site's view of the world is a directory of bundles, one per
contributing site, each produced by `12_export.py`. This module discovers them,
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


def discover(export_home: pathlib.Path, logger) -> dict[str, pathlib.Path]:
    """
    the bundles present, keyed by the site name in their manifest

    Keyed by the manifest rather than the directory name: a directory can be
    renamed in transit, and the site that produced the data is a property of the
    data rather than of where it was filed.
    """
    export_home = pathlib.Path(export_home)
    if not export_home.is_dir():
        raise FileNotFoundError(
            f"{export_home} does not exist. A coordinating site reads bundles "
            "from there; copy each contributing site's export/<SITE>/ directory "
            "into it."
        )

    found: dict[str, pathlib.Path] = {}
    for candidate in sorted(p for p in export_home.iterdir() if p.is_dir()):
        if not (candidate / manifest.MANIFEST_NAME).is_file():
            logger.warning(
                f"  {candidate.name}/ has no {manifest.MANIFEST_NAME} and is "
                "being skipped -- a bundle without one cannot be checked for "
                "code, model or tokenizer agreement"
            )
            continue
        site = manifest.read(candidate).get("site") or candidate.name
        if site in found:
            raise ValueError(
                f"two bundles both claim to be site {site!r}: "
                f"{found[site].name} and {candidate.name}. Pooling both would "
                "double-count every patient; keep the newer one."
            )
        found[site] = candidate

    if not found:
        raise FileNotFoundError(f"no bundles with a manifest under {export_home}")
    return found


def check(bundles: dict[str, pathlib.Path], paths, logger) -> dict[str, dict]:
    """
    validate the bundles against each other and against the local model

    Returns the manifests, keyed by site.
    """
    manifests = {site: manifest.read(home) for site, home in bundles.items()}
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

    counts_only = [s for s, m in manifests.items() if not m.get("shares_patient_level")]
    if counts_only:
        logger.info(
            f"{', '.join(counts_only)} sent counts only (no patient-level "
            "transfer permitted), so they contribute to every count table and "
            "to no projection"
        )
    return manifests


def load_patient_level(
    bundles: dict[str, pathlib.Path], manifests: dict[str, dict], anchor: str, logger
) -> pl.DataFrame:
    """
    the pooled patient-level frame for one anchor

    Concatenated `how="diagonal_relaxed"` so a site that exported a label
    another did not sits alongside it as null rather than failing the
    concatenation -- but every column still has to be on the allowlist, which
    is what stops that leniency from becoming a hole.
    """
    frames = []
    for site, home in sorted(bundles.items()):
        if not manifests[site].get("shares_patient_level"):
            continue
        source = home / "patient" / f"embeddings_anchor-{anchor}.parquet"
        if not source.is_file():
            logger.warning(
                f"  {site}: no patient file for anchor {anchor}, skipping"
            )
            continue
        frame = pl.read_parquet(source)
        contract.validate_patient_frame(frame, f"{site} bundle, anchor {anchor}")

        # the manifest says how many rows were sent; a shorter file means a
        # truncated copy, which is otherwise invisible
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
        frames.append(frame)
        logger.info(f"  {site}: {frame.height} patients at anchor {anchor}")

    if not frames:
        raise FileNotFoundError(
            f"no bundle carries patient-level data for anchor {anchor}. Every "
            "contributing site sent counts only, so the projection analyses "
            "cannot run -- the count tables and phenotype breakdowns still can."
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
    bundles: dict[str, pathlib.Path], name: str, logger
) -> pl.DataFrame | None:
    """
    one aggregate table, stacked across every bundle that carries it

    Returns None when no bundle has it, which is the normal case for a table a
    site could not produce -- a missing phenotype breakdown at one site is a
    fact about that site, not a reason to fail.
    """
    frames = []
    for site, home in sorted(bundles.items()):
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


def token_legend(bundles: dict[str, pathlib.Path]) -> pl.DataFrame | None:
    """
    the rank -> value-range legend

    Every bundle carries an identical copy, since it is derived from the shared
    tokenizer and contains no patient data, so the first one found is taken and
    the rest are checked against it.
    """
    legends = {
        site: pl.read_csv(home / "token_levels.csv")
        for site, home in sorted(bundles.items())
        if (home / "token_levels.csv").is_file()
    }
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

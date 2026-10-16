#!/usr/bin/env python3

"""
how coherent each phenotype is in this site's embedding space

The question a consortium wants answered before anyone argues about whether
embeddings may be shared: do the phenotypes correspond to anything the model
sees? That is answerable without moving a single embedding. Each site computes
the measure over its own patients and reports a handful of scalars.

The measure is the one 03 already uses for the site effect, so the numbers are
directly comparable to the 97% site purity in the README: for each patient, the
share of its k nearest neighbours in 50-d PCA space carrying the same label,
against the share expected if the label were scattered at random.

Reported per phenotype, not as one number per scheme. A pooled figure hides the
thing worth knowing -- that one phenotype is a real region of the space and
another is scattered through it -- and for these schemes that difference is the
result. A scheme-wide row is included alongside for context.

WHAT A SITE SENDS
One row per (scheme, phenotype): n, the observed neighbour purity, the chance
rate, the lift, and the ratio of mean distance to its own centroid against the
other phenotypes' centroid. Nothing per patient, no coordinates, no embeddings.

WHY THIS IS SAFE TO SEND, STATED PRECISELY
Each number is an average over at least min_cell patients, and none can be
inverted to a position: purity is a count of neighbour labels and the centroid
figure is a ratio of mean distances. Rows below the minimum cell size are
suppressed like any other count.

Each scheme is measured inside the population it labels, refitting the PCA
and the neighbour graph on that subset -- see `_scheme_rows` for why, and for
what went wrong when it was not done.

WHY IT IS NOT COMPARABLE ACROSS SITES WITHOUT CARE
The PCA is fitted on this site's patients, so the 50 dimensions differ between
sites, as does the chance rate, which depends on local phenotype prevalence. A
lift of 1.4 at one site and 1.1 at another says the phenotype is more coherent
in the first site's own embedding geometry -- not that the two spaces are the
same shape. The lift is the comparable quantity, because it is already
normalised by the local chance rate; raw purity is not.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#: matches 03 and 04, so the numbers sit beside the site-effect ones
PCA_COMPONENTS = 50
K_NEIGHBOURS = 30

#: the label columns reported on, and the scheme each belongs to
SCHEMES: dict[str, str] = {
    "cs_phenotype": "zweck",
    "seymour_phenotype": "seymour",
    "bhavani_phenotype": "bhavani",
    "shock_type": "coded_shock_type",
    "in_hospital_mortality": "outcome",
}


def embedding_matrix(frame: pl.DataFrame) -> np.ndarray:
    columns = sorted(c for c in frame.columns if c.startswith("emb_"))
    if not columns:
        raise KeyError("no emb_* columns; the embeddings have not been computed")
    return frame.select(columns).to_numpy().astype(np.float64)


def _coordinates(frame: pl.DataFrame, seed: int) -> np.ndarray:
    """the 50-d PCA space the neighbourhoods are measured in"""
    return PCA(
        n_components=min(PCA_COMPONENTS, frame.height - 1), random_state=seed
    ).fit_transform(StandardScaler().fit_transform(embedding_matrix(frame)))


def _centroid_ratio(
    coordinates: np.ndarray, members: np.ndarray, labelled: np.ndarray
) -> float | None:
    """
    mean distance to own centroid over mean distance to the others' centroid

    Below 1 means the phenotype's patients sit closer to their own centre than
    to everyone else's. Reported alongside neighbour purity because the two can
    disagree: a phenotype can be locally clustered in pockets scattered across
    the space (high purity, ratio near 1), or be a broad but genuinely distinct
    region (modest purity, low ratio).
    """
    others = labelled & ~members
    if members.sum() == 0 or others.sum() == 0:
        return None
    own_centre = coordinates[members].mean(axis=0)
    other_centre = coordinates[others].mean(axis=0)
    to_own = float(np.linalg.norm(coordinates[members] - own_centre, axis=1).mean())
    to_other = float(np.linalg.norm(coordinates[members] - other_centre, axis=1).mean())
    return to_own / to_other if to_other > 0 else None


def phenotype_purity(
    frame: pl.DataFrame,
    anchor: str,
    site: str,
    logger,
    *,
    seed: int = 42,
    k: int = K_NEIGHBOURS,
    min_cell: int = 11,
) -> pl.DataFrame:
    """
    per-phenotype neighbourhood purity and centroid separation, site-local

    `frame` is one site's embeddings with the phenotype labels attached.
    Returns one row per (scheme, phenotype) plus one scheme-wide row.
    """
    rows = []
    for column, scheme in SCHEMES.items():
        if column not in frame.columns:
            continue
        rows.extend(_scheme_rows(frame, column, scheme, anchor, site, seed, k, logger))

    table = pl.DataFrame(rows)
    if table.height == 0:
        return table

    # suppressed the same way a count table is: a purity over six patients is
    # a statement about those six
    thin = table.filter((pl.col("n") > 0) & (pl.col("n") < min_cell))
    if thin.height:
        logger.info(
            f"[{site}/{anchor}] suppressing {thin.height} purity row(s) below "
            f"{min_cell} patients: "
            + ", ".join(
                f"{r['scheme']}/{r['label']}" for r in thin.iter_rows(named=True)
            )
        )
    table = table.with_columns(
        [
            pl.when(pl.col("n") < min_cell).then(None).otherwise(pl.col(c)).alias(c)
            for c in ("n", "neighbour_purity", "chance", "lift", "centroid_ratio")
        ]
    )

    _report(table, site, anchor, logger)
    return table


def _report(table: pl.DataFrame, site: str, anchor: str, logger) -> None:
    logger.info(
        f"[{site}/{anchor}] phenotype coherence in this site's embedding space "
        f"(k={K_NEIGHBOURS} neighbours, {PCA_COMPONENTS}-d PCA):"
    )
    for scheme in table["scheme"].unique(maintain_order=True).to_list():
        logger.info(f"  {scheme}")
        for row in table.filter(pl.col("scheme") == scheme).iter_rows(named=True):
            if row["neighbour_purity"] is None:
                logger.info(f"    {row['label']:<24} suppressed (n < minimum)")
                continue
            ratio = (
                f"  centroid {row['centroid_ratio']:.2f}"
                if row["centroid_ratio"] is not None
                else ""
            )
            logger.info(
                f"    {row['label']:<24} n={row['n']:>6}  "
                f"purity {row['neighbour_purity']:>6.1%} vs {row['chance']:>6.1%} "
                f"by chance  lift {row['lift']:.2f}x{ratio}"
            )

    # The interpretive line, which is the point of reporting this at all: a
    # phenotype at lift ~1 is not a region of the embedding space, and no
    # pooled scatterplot coloured by it will show anything but site structure.
    flat = table.filter(
        (pl.col("label") != "(all)")
        & pl.col("lift").is_not_null()
        & (pl.col("lift") < 1.1)
    )
    if flat.height:
        logger.warning(
            f"[{site}/{anchor}] {flat.height} phenotype(s) sit at lift < 1.1 -- "
            "their patients are no more alike in embedding space than chance: "
            + ", ".join(
                f"{r['scheme']}/{r['label']} {r['lift']:.2f}x"
                for r in flat.iter_rows(named=True)
            )
        )


def _scheme_rows(
    frame: pl.DataFrame,
    column: str,
    scheme: str,
    anchor: str,
    site: str,
    seed: int,
    k: int,
    logger,
) -> list[dict]:
    """
    purity for one scheme, measured inside the population that scheme labels

    THE SUBSET IS THE WHOLE POINT. The two sepsis schemes label only the
    coded-septic subset -- at UCMC, 1,860 of 9,135 patients -- so asking
    whether a Seymour-alpha patient is surrounded by other alphas *among
    everyone* is not a question about Seymour. Four fifths of the neighbours
    have no label at all and can never match, so the purity collapses toward
    zero while the chance rate, computed over the labelled patients, does not.

    The first version of this did exactly that and reported every sepsis
    phenotype at a lift of 0.3 to 0.7, which reads as "anti-clustered" and is
    really just two different denominators. Zweck labels everyone, so it was
    unaffected, which is what made the mistake look like a finding about the
    sepsis schemes specifically.

    So the neighbour graph and the PCA are refit on the labelled subset. One
    consequence worth stating: the 50-d space therefore differs between
    schemes, and a Zweck lift is not measured in the same coordinates as a
    Seymour lift. Each is a statement about whether that scheme carves the
    embedding of its own population, which is the question asked.
    """
    from sklearn.neighbors import NearestNeighbors

    subset = frame.filter(pl.col(column).is_not_null())
    if subset.height <= k + 1:
        logger.info(
            f"[{site}/{anchor}] {scheme}: {subset.height} labelled patients, "
            f"too few for {k}-neighbour purity"
        )
        return []

    n_dropped = frame.height - subset.height
    if n_dropped:
        logger.info(
            f"[{site}/{anchor}] {scheme}: measured within the {subset.height} "
            f"patients it labels ({n_dropped} unlabelled excluded)"
        )

    coordinates = _coordinates(subset, seed)
    kk = min(k, subset.height - 1)
    neighbours = (
        NearestNeighbors(n_neighbors=kk + 1, n_jobs=-1)
        .fit(coordinates)
        .kneighbors(coordinates, return_distance=False)[:, 1:]
    )

    names = np.asarray(subset[column].cast(pl.String).to_list())
    classes, codes = np.unique(names, return_inverse=True)
    same = codes[neighbours] == codes[:, None]
    proportions = np.bincount(codes, minlength=len(classes)) / len(codes)

    rows = []
    for index, name in enumerate(classes):
        members = codes == index
        n = int(members.sum())
        if n == 0:
            continue
        observed = float(np.mean(same[members]))
        chance = float(proportions[index])
        rows.append(
            {
                "site": site,
                "anchor": anchor,
                "scheme": scheme,
                "label": name,
                "n": n,
                "n_scheme_population": subset.height,
                "neighbour_purity": observed,
                "chance": chance,
                "lift": observed / chance if chance > 0 else None,
                "centroid_ratio": _centroid_ratio(
                    coordinates, members, np.ones(subset.height, dtype=bool)
                ),
            }
        )

    overall = float(np.mean(same))
    chance_all = float((proportions**2).sum())
    rows.append(
        {
            "site": site,
            "anchor": anchor,
            "scheme": scheme,
            "label": "(all)",
            "n": subset.height,
            "n_scheme_population": subset.height,
            "neighbour_purity": overall,
            "chance": chance_all,
            "lift": overall / chance_all if chance_all > 0 else None,
            "centroid_ratio": None,
        }
    )
    return rows

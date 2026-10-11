#!/usr/bin/env python3

"""
08 -- the model's learned token space

A UMAP of the embedding table itself: one point per vocabulary entry, not per
patient. This asks whether the model organised clinical concepts into coherent
structure during training -- whether labs sit near labs, vasopressors near
vasopressors, and whether dose or value quantiles of the same measurement lie
on an ordered path.

It is the one figure here that needs no patient data at all, and it is a
precondition for reading anything else: if the token space is incoherent, the
patient representations built from it cannot mean much.

Tokens are weighted by how often they actually occur in the cohort, because
the embedding table also holds rows for tokens the model barely saw, whose
positions are close to their random initialisation and would otherwise read as
real structure.

Usage
    python code/08_token_space.py
    python code/08_token_space.py --model GEM-mimic-icu --min-count 300
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from safetensors.numpy import load_file

from clifcs import get_logger, load_paths
from clifcs.palette import CATEGORICAL, GRID, INK_PRIMARY, INK_SECONDARY, SURFACE
from clifcs.tokens import load_pre_anchor_tokens, token_counts_by_site, vocabulary

DPI = 500

#: families grouped for colour; everything else falls into "other"
FAMILY_GROUPS = {
    "labs": ("LAB-ORD", "LAB-RES"),
    "vitals": ("VTL",),
    "medications": ("MED-CTS", "MED-INT"),
    "respiratory": ("RESP",),
    "assessments": ("ASMT",),
    "severity / events": ("SOFA", "LABEL"),
    "location & admin": ("XFR-IN", "XFR-OUT", "ADMN", "DSCG", "CODE"),
    "demographics": ("AGE", "SEX", "RACE", "ETHN"),
}
#: One hue only. Nine groups cannot be told apart by colour in a scatter --
#: the palette validates three slots under the all-pairs gate -- so each group
#: gets its own panel with the rest as grey context, which is the documented
#: answer past three series.
HIGHLIGHT = CATEGORICAL[0]
CONTEXT = "#d9d8d3"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE["light"],
            "axes.facecolor": SURFACE["light"],
            "savefig.facecolor": SURFACE["light"],
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "font.size": 8,
            "legend.frameon": False,
        }
    )


def embedding_table(model_home: pathlib.Path, logger) -> np.ndarray:
    """
    the input embedding matrix, vocab x hidden

    Read straight from the safetensors file rather than by instantiating the
    model: nothing here needs a forward pass, and loading 300 MB of weights to
    read one matrix would be wasteful.
    """
    weights = load_file(str(model_home / "model.safetensors"))
    for key in ("model.embed_tokens.weight", "embed_tokens.weight"):
        if key in weights:
            table = np.asarray(weights[key], dtype=np.float64)
            logger.info(f"embedding table {key}: {table.shape}")
            return table
    raise KeyError(
        f"no embedding matrix in {model_home / 'model.safetensors'}; "
        f"keys include {sorted(weights)[:5]}"
    )


def cohort_token_counts(paths, logger) -> pl.DataFrame:
    """how often each token occurs in this cohort's pre-anchor streams"""
    names, _ = vocabulary(str(paths.tokenizer_yaml))
    frames = []
    for site in paths.sites:
        try:
            frames.append(load_pre_anchor_tokens(paths, site, "24h"))
        except FileNotFoundError:
            logger.warning(f"no winnowed tokens for {site}; skipping")
    if not frames:
        raise FileNotFoundError("no tokenized data; run 01 first")
    tokens = pl.concat(frames)

    # the winnowed stream covers everyone embedded, which is a superset of the
    # cohort once an enrollment criterion is tightened; count only the cohort
    cohort_file = paths.cohort_home / "cohort.parquet"
    if cohort_file.is_file():
        keep = pl.read_parquet(cohort_file).select("hospitalization_id", "site")
        before = tokens.height
        tokens = tokens.join(keep, on=["hospitalization_id", "site"], how="inner")
        logger.info(f"token counts from the cohort: {before} -> {tokens.height} patients")

    counts = token_counts_by_site(tokens, names)
    return counts.group_by("token_id", "token", "family").agg(n=pl.col("n").sum())


def group_of(family: str) -> str:
    for group, members in FAMILY_GROUPS.items():
        if family in members:
            return group
    return "other"


def plot_token_space(
    coords: np.ndarray, meta: pl.DataFrame, out_path: pathlib.Path, title: str, logger
) -> None:
    """
    small multiples of the token space, one panel per clinical family

    All points are drawn in every panel so the overall shape stays visible;
    the panel's own family is lifted out of the grey. Point area follows log
    frequency, so a cluster of common tokens reads as more substantial than a
    cluster of rare ones.
    """
    groups = [g for g in list(FAMILY_GROUPS) + ["other"]
              if (meta["group"] == g).any()]
    columns = 3
    rows = int(np.ceil(len(groups) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.0 * columns, 2.9 * rows), squeeze=False
    )
    sizes = 3 + 22 * (
        np.log1p(meta["n"].to_numpy()) / np.log1p(meta["n"].max())
    ) ** 2
    group_values = np.asarray(meta["group"].to_list())
    counts = meta["n"].to_numpy()

    for position, group in enumerate(groups):
        ax = axes[position // columns][position % columns]
        mask = group_values == group
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=2.0, c=CONTEXT,
                   linewidths=0, rasterized=True)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=sizes[mask], c=HIGHLIGHT,
                   linewidths=0, alpha=0.85, rasterized=True)

        # two labels per panel, the most frequent members, placed apart
        local = np.flatnonzero(mask)
        # alternate the offset: the two most frequent tokens of a family are
        # usually adjacent, so a fixed offset collides them
        for rank, i in enumerate(local[np.argsort(counts[local])[-2:]]):
            ax.annotate(
                meta["token"][int(i)], (coords[i, 0], coords[i, 1]),
                fontsize=5, color=INK_PRIMARY,
                xytext=(4, 4) if rank % 2 else (4, -8),
                textcoords="offset points",
            )
        ax.set_title(f"{group}  ({int(mask.sum())})", loc="left",
                     fontsize=8, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    for position in range(len(groups), rows * columns):
        axes[position // columns][position % columns].axis("off")

    figure.suptitle(title, x=0.02, ha="left", fontsize=10, y=0.995)
    figure.text(0.02, 0.005, "UMAP of the embedding table; grey = all other tokens; "
                "point area follows log frequency in the cohort",
                fontsize=7, color=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0.02, 1, 0.97))
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    logger.info(f"wrote {out_path}")


def quantile_coherence(
    vectors: np.ndarray, coords: np.ndarray, meta: pl.DataFrame, logger
) -> pl.DataFrame:
    """
    do the quantile bins of one measurement lie on an ordered path?

    cocoa bins numeric values into deciles, so `LAB-RES//lactate_Q0` through
    `_Q9` are ten separate tokens the model had to relate on its own. If it
    learned the ordering, consecutive quantiles sit closer than distant ones.
    Reported as the Spearman correlation between |quantile difference| and
    distance between the tokens, per measurement with at least five bins.

    Measured in the native 1024-d space, under the cosine distance the UMAP
    was built on. The projection is reported alongside but is not the claim:
    a correlation computed on 2-d coordinates cannot separate what the model
    learned from how UMAP chose to lay it out, and would overstate the result
    whenever the layout happens to be kind.
    """
    rows = []
    base = (
        meta.with_columns(
            stem=pl.col("token").str.replace(r"_Q\d$", ""),
            quantile=pl.col("token").str.extract(r"_Q(\d)$", 1).cast(pl.Int64),
        )
        .with_row_index("row")
        .filter(pl.col("quantile").is_not_null())
    )
    for stem, part in base.group_by("stem"):
        stem = stem[0] if isinstance(stem, tuple) else stem
        if part.height < 5:
            continue
        index = part["row"].to_numpy()
        q = part["quantile"].to_numpy()
        native = vectors[index]
        native = native / np.linalg.norm(native, axis=1, keepdims=True)
        flat = coords[index]
        dq, d_native, d_flat = [], [], []
        for i in range(len(index)):
            for j in range(i + 1, len(index)):
                dq.append(abs(q[i] - q[j]))
                d_native.append(1.0 - float(native[i] @ native[j]))
                d_flat.append(float(np.linalg.norm(flat[i] - flat[j])))
        if len(dq) < 6:
            continue

        def spearman(a, b):
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            return float(np.corrcoef(ra, rb)[0, 1])

        rows.append({
            "measurement": stem,
            "n_bins": part.height,
            "spearman": spearman(dq, d_native),
            "spearman_umap": spearman(dq, d_flat),
        })

    table = pl.DataFrame(rows).sort("spearman", descending=True)
    if table.height:
        logger.info(
            f"quantile ordering across {table.height} binned measurements, "
            f"in the native 1024-d space (cosine): median Spearman "
            f"{table['spearman'].median():.3f}, "
            f"{(table['spearman'] > 0).mean():.0%} positive"
        )
        logger.info(
            f"    for comparison, in the 2-d UMAP projection: median "
            f"{table['spearman_umap'].median():.3f}, "
            f"{(table['spearman_umap'] > 0).mean():.0%} positive"
        )
        for row in table.head(5).iter_rows(named=True):
            logger.info(
                f"    {row['measurement']:<34} native {row['spearman']:+.3f}  "
                f"umap {row['spearman_umap']:+.3f}"
            )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--min-count",
        type=int,
        default=300,
        help="drop tokens occurring fewer times than this in the cohort; rare "
        "tokens sit near their random initialisation and would read as structure",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger("08_token_space", paths.log_home / "08_token_space.log")
    style()

    import umap

    table = embedding_table(paths.model_home, logger)
    counts = cohort_token_counts(paths, logger)
    names, families = vocabulary(str(paths.tokenizer_yaml))

    meta = (
        counts.filter(pl.col("n") >= args.min_count)
        .with_columns(group=pl.col("family").map_elements(group_of, return_dtype=pl.String))
        .sort("token_id")
    )
    index = meta["token_id"].to_numpy()
    if index.max() >= table.shape[0]:
        raise IndexError(
            f"token id {index.max()} exceeds the embedding table ({table.shape[0]} rows)"
        )
    vectors = table[index]
    logger.info(
        f"{len(index)} of {len(names)} tokens occur >= {args.min_count} times; "
        f"embedding dim {vectors.shape[1]}"
    )

    coords = umap.UMAP(
        n_neighbors=15, min_dist=0.25, n_components=2, metric="cosine",
        random_state=args.seed,
    ).fit_transform(vectors)

    plot_token_space(
        coords,
        meta,
        paths.figure_home / "token_space_umap.png",
        f"{paths.model_key}: learned token embeddings "
        f"(>= {args.min_count} occurrences)",
        logger,
    )

    coherence = quantile_coherence(vectors, coords, meta, logger)
    if coherence.height:
        coherence.write_csv(paths.table_home / "token_quantile_ordering.csv")
    meta.with_columns(
        umap_1=pl.Series(coords[:, 0]), umap_2=pl.Series(coords[:, 1])
    ).write_csv(paths.table_home / "token_space_umap.csv")
    logger.info("08 done")


if __name__ == "__main__":
    main()

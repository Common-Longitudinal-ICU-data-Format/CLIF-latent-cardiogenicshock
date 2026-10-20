#!/usr/bin/env python3

"""
05 -- affine correction of the site effect

Fits each correction on a training split and reports, on held-out patients,
how much site information survives it and how much of the representation it
cost to get there.

  raw                no correction, the reference point
  site-standardized  z-score within each site: removes per-feature location
                     and scale differences and nothing else
  leace              least-squares concept erasure: the affine map that
                     provably removes all linearly-decodable site information
                     while perturbing the representation minimally

Read the output with 04 in mind. The sites separate because each emits token
types the other never does, so an affine map on the embedding closes the
linear channel rather than making the representations exchangeable. The
non-linear probe here is the check on that: if a gradient-boosted tree still
recovers site after LEACE has driven the linear probe to chance, the
information is still present, just no longer linear.

Usage
    python code/05_site_correction.py
    python code/05_site_correction.py --model GEM-mimic-icu --anchor 24h
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from clifcs import ANCHORS, get_logger, load_paths
from clifcs.embedding import site_embedding_files
from clifcs.correction import METHODS, fit

PCA_COMPONENTS = 50
K_NEIGHBOURS = 30


def embedding_matrix(frame: pl.DataFrame) -> np.ndarray:
    columns = sorted(c for c in frame.columns if c.startswith("emb_"))
    return frame.select(columns).to_numpy().astype(np.float64)


def load_embeddings(paths, anchor: str, logger) -> pl.DataFrame:
    sources = site_embedding_files(paths.patient_level_home, anchor)
    if not sources:
        raise FileNotFoundError(f"no embeddings for {anchor} in {paths.patient_level_home}")
    frame = pl.concat([pl.read_parquet(p) for p in sources], how="diagonal_relaxed")

    # Restrict to the current cohort. 02 embeds everyone present in the
    # winnowed data, which is a superset once an enrollment criterion is
    # tightened (cardiac origin, say). Without this join 04 and 05 would keep
    # describing the old cohort while 03 and 06 described the new one.
    cohort_file = paths.cohort_home / "cohort.parquet"
    if cohort_file.is_file():
        keep = pl.read_parquet(cohort_file).select("hospitalization_id", "site")
        before = frame.height
        frame = frame.join(keep, on=["hospitalization_id", "site"], how="inner")
        if frame.height != before:
            logger.info(
                f"[{anchor}] restricted to the cohort: {before} -> {frame.height}"
            )
    logger.info(f"[{anchor}] {frame.height} patients from {len(sources)} site files")
    return frame


def site_purity(x: np.ndarray, sites: np.ndarray) -> tuple[float, float]:
    """share of each patient's neighbours from the same site, and chance"""
    coordinates = PCA(
        n_components=min(PCA_COMPONENTS, x.shape[1], x.shape[0] - 1), random_state=42
    ).fit_transform(StandardScaler().fit_transform(x))
    k = min(K_NEIGHBOURS, len(x) - 1)
    neighbours = (
        NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        .fit(coordinates)
        .kneighbors(coordinates, return_distance=False)[:, 1:]
    )
    _, codes = np.unique(sites, return_inverse=True)
    purity = float(np.mean(codes[neighbours] == codes[:, None]))
    proportions = np.bincount(codes) / len(codes)
    return purity, float((proportions**2).sum())


def evaluate(
    name: str,
    train_x: np.ndarray,
    train_sites: np.ndarray,
    test_x: np.ndarray,
    test_sites: np.ndarray,
    reference: str,
    anchor: str,
    logger,
) -> dict:
    """site recoverability after a correction, linearly and non-linearly"""
    train_y = (train_sites == reference).astype(int)
    test_y = (test_sites == reference).astype(int)

    reducer = PCA(n_components=PCA_COMPONENTS, random_state=42).fit(train_x)
    train_r, test_r = reducer.transform(train_x), reducer.transform(test_x)

    linear = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        train_r, train_y
    )
    linear_auc = roc_auc_score(test_y, linear.decision_function(test_r))

    nonlinear = HistGradientBoostingClassifier(
        max_iter=200, random_state=42
    ).fit(train_r, train_y)
    nonlinear_auc = roc_auc_score(test_y, nonlinear.predict_proba(test_r)[:, 1])

    purity, chance = site_purity(test_x, test_sites)
    logger.info(
        f"[{anchor}] {name:<18} linear-probe AUC {linear_auc:.4f}  "
        f"non-linear AUC {nonlinear_auc:.4f}  "
        f"site purity {purity:.1%} (chance {chance:.1%}, {purity / chance:.2f}x)"
    )
    return {
        "anchor": anchor,
        "correction": name,
        "linear_probe_auc": linear_auc,
        "nonlinear_probe_auc": nonlinear_auc,
        "site_purity": purity,
        "purity_chance": chance,
        "purity_lift": purity / chance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", choices=ANCHORS, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-map",
        action="store_true",
        help="save each fitted affine map to output/patient_level_data/"
        "corrections/anchor-<anchor>_<method>.npz. 09 needs this: it applies "
        "the same map to all 26 grid points, and refitting on the grid would "
        "weight the fit by whose grid is complete",
    )
    parser.add_argument(
        "--write-corrected",
        action="store_true",
        help="also save the corrected embeddings; 06 refits corrections itself, "
        "so this is only needed for inspection",
    )
    parser.add_argument(
        "--role",
        choices=("site", "pooled"),
        default=None,
        help="`pooled` reads received bundles from output/export/ and refuses "
        "to touch raw CLIF data; `site` reads this machine's own files. "
        "Defaults to `role` in paths.yaml, then to `site`.",
    )
    args = parser.parse_args()

    paths = load_paths(model=args.model, role=args.role)
    paths.mkdirs()
    logger = get_logger("05_site_correction", paths.log_home / "05_site_correction.log")

    rows = []
    for anchor in ((args.anchor,) if args.anchor else ANCHORS):
        frame = load_embeddings(paths, anchor, logger)
        sites = np.asarray(frame["site"].to_list())
        present = [s for s in paths.sites if s in set(sites)]
        if len(present) < 2:
            logger.warning(f"[{anchor}] need two sites to correct between")
            continue
        reference = present[0]
        x = embedding_matrix(frame)

        # corrections are fitted on a training split and judged on held-out
        # patients; fitting and scoring on the same rows would let the eraser
        # memorise this sample rather than the site direction
        index_train, index_test = train_test_split(
            np.arange(len(x)), test_size=0.3, random_state=args.seed, stratify=sites
        )
        logger.info(
            f"[{anchor}] model {paths.model_key}, reference site {reference}, "
            f"{len(index_train)} train / {len(index_test)} test"
        )

        for method in METHODS:
            correction = fit(method, x[index_train], sites[index_train], reference)
            rows.append(
                evaluate(
                    correction.name,
                    correction.apply(x[index_train], sites[index_train]),
                    sites[index_train],
                    correction.apply(x[index_test], sites[index_test]),
                    sites[index_test],
                    reference,
                    anchor,
                    logger,
                )
            )
            if args.save_map:
                # The map fitted on the TRAINING split, which is the one the
                # AUCs above describe. Refitting on all rows would use more
                # data but would mean the saved map is not the measured one.
                out = (
                    paths.patient_level_home
                    / "corrections"
                    / f"anchor-{anchor}_{method}.npz"
                )
                correction.save(out)
                logger.info(f"[{anchor}] saved map -> corrections/{out.name}")

            if args.write_corrected and method != "raw":
                corrected = correction.apply(x, sites)
                # A subdirectory, not a sibling. These were first written as
                # `embeddings_anchor-<anchor>_corrected-<method>.parquet`,
                # which matches the `embeddings_anchor-<anchor>_*.parquet`
                # glob that 03, 04, 06, 11 and 13 all use to pool the per-site
                # files -- so writing one silently tripled the apparent cohort
                # the next time 03 ran, and the figures looked entirely
                # plausible at 102,426 patients. Nothing in the pool directory
                # may share that prefix.
                out = (
                    paths.patient_level_home
                    / "corrected"
                    / f"anchor-{anchor}_{method}.parquet"
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                pl.concat(
                    [
                        frame.select(
                            [c for c in frame.columns if not c.startswith("emb_")]
                        ),
                        pl.from_numpy(
                            corrected.astype(np.float32),
                            schema=[f"emb_{i:04d}" for i in range(corrected.shape[1])],
                        ),
                    ],
                    how="horizontal",
                ).write_parquet(out)
                logger.info(f"[{anchor}] wrote {out.name}")

    if rows:
        table = pl.DataFrame(rows)
        table.write_csv(paths.table_home / "site_correction.csv")
        logger.info(f"wrote {paths.table_home / 'site_correction.csv'}")
    logger.info("05 done")


if __name__ == "__main__":
    main()

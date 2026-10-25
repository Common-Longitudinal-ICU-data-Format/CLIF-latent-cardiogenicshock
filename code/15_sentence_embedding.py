#!/usr/bin/env python3

"""
15_sentence_embedding.py -- what the embedding knows that the sentence does not

Two descriptions of the same 48 hours: a BPE sentence over the MAP/pressor
states, and a 1024-d GEM trajectory. This asks what each contributes to
predicting death, and then names what the embedding carries beyond the
sentence.

    python code/15_sentence_embedding.py --subset septic

THE PAIRING IS NOT SYMMETRIC, which is where the text-and-image analogy stops
being exact. In a text-image model neither view is computable from the other.
Here the sentence is a deterministic function of a SUBSET of what the embedding
already read -- GEM consumed the MAP values and the pressor doses, and the
sentence is those same facts thresholded. So "can the embedding predict the
sentence" is near-trivially yes and means nothing. The informative direction is
the residual: what the embedding has left over.

CONFOUNDERS GO IN THE BASELINE, not in a footnote. Log stream length and site
join the sentence-side model, so the embedding's unique contribution is
measured over and above documentation volume and hospital. Position in the
projection explains R2 0.45 of log stream length and site survives the
correction at non-linear AUC 0.855, so without this the "embedding-only" share
would partly be chart density wearing a physiology label.

COMMONALITY ANALYSIS USES PSEUDO-R2, NOT AUC. AUC does not decompose additively,
so unique and shared shares computed from it would not sum to the total. The
partition is taken on McFadden's pseudo-R2 from the cross-validated log loss,
which does.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import types

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.windows import (
    composition_matrix, site_representative_tokens, window_token_counts,
)

OVERLAP = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]
COMPONENTS = 50


def cross_validated(x, y, seed, folds=5):
    """mean AUC and McFadden pseudo-R2 over stratified folds"""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs, losses = [], []
    base_rate = y.mean()
    for train, test in splitter.split(x, y):
        model = LogisticRegression(max_iter=3000, C=1.0).fit(x[train], y[train])
        probability = model.predict_proba(x[test])[:, 1]
        aucs.append(roc_auc_score(y[test], probability))
        losses.append(log_loss(y[test], probability, labels=[0, 1]))
    null = log_loss(y, np.full(len(y), base_rate), labels=[0, 1])
    return float(np.mean(aucs)), float(1.0 - np.mean(losses) / null)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="septic",
                        choices=("septic", "cardiogenic"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-group", type=int, default=8,
                        help="patients sharing an identical sentence needed "
                             "before that sentence forms a matched group")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "15_sentence_embedding", paths.log_home / "15_sentence_embedding.log"
    )

    labels = pl.read_parquet(
        paths.patient_level_home / f"longitudinal_clusters_{args.subset}.parquet"
    )
    if "bpe_sentence_48h" not in labels.columns:
        logger.error("no bpe_sentence_48h; run 11 first")
        return
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet").select(
        "hospitalization_id", "site", "in_hospital_mortality"
    )

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "traj", pathlib.Path(__file__).resolve().parent / "09_trajectories.py"
    )
    traj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(traj)
    grid_args = types.SimpleNamespace(
        subset=args.subset, start_hours=-24.0, end_hours=48.0, step_hours=6.0,
        correction="coral", anchor="onset", force=False, min_tokens=8,
    )
    grid = traj.load_grid(paths, grid_args, logger)
    grid = traj.apply_correction(paths, grid, "coral", "onset", logger)
    grid = grid.filter(pl.col("offset_hours").is_in(OVERLAP))

    emb_columns = sorted(c for c in grid.columns if c.startswith("emb_"))
    per_patient = grid.group_by("hospitalization_id").agg(
        [pl.col(c).mean() for c in emb_columns]
        + [pl.col("last_valid").median().alias("stream"),
           pl.col("site").first()]
    )
    frame = (
        labels.select("hospitalization_id", "bpe_sentence_48h")
        .join(per_patient, on="hospitalization_id", how="inner")
        .join(cohort.drop("site"), on="hospitalization_id", how="inner")
        .filter(pl.col("in_hospital_mortality").is_not_null())
    )
    logger.info(f"{frame.height:,} patients with a sentence, an embedding and an outcome")
    y = frame["in_hospital_mortality"].cast(pl.Int8).to_numpy()

    # ---- sentence side: bag of BPE words, plus the two known confounders
    sentences = [s.split() for s in frame["bpe_sentence_48h"].to_list()]
    vocab = sorted({w for s in sentences for w in s})
    bag = np.zeros((len(sentences), len(vocab)), dtype=np.float32)
    position = {w: i for i, w in enumerate(vocab)}
    for row, words in enumerate(sentences):
        for word in words:
            bag[row, position[word]] += 1.0
    sites = sorted(frame["site"].unique().to_list())
    site_dummies = np.column_stack([
        (frame["site"] == s).cast(pl.Int8).to_numpy() for s in sites[:-1]
    ]) if len(sites) > 1 else np.zeros((frame.height, 0))
    stream = np.log1p(
        frame["stream"].fill_null(0).cast(pl.Float64).to_numpy()
    )[:, None]
    sentence_x = np.column_stack([bag, site_dummies, stream]).astype(np.float64)
    logger.info(
        f"sentence side: {len(vocab)} BPE words + {site_dummies.shape[1]} site "
        "dummies + log stream length"
    )

    # ---- embedding side: PCA of the mean corrected trajectory
    embedding = frame.select(emb_columns).to_numpy().astype(np.float64)
    reducer = PCA(n_components=COMPONENTS, random_state=args.seed)
    embedding_x = reducer.fit_transform(
        (embedding - embedding.mean(axis=0))
        / (embedding.std(axis=0) + 1e-9)
    )
    logger.info(
        f"embedding side: {COMPONENTS} components, "
        f"{reducer.explained_variance_ratio_.sum():.1%} of variance"
    )

    both_x = np.column_stack([sentence_x, embedding_x])
    results = {}
    for name, matrix in (
        ("sentence + confounders", sentence_x),
        ("embedding", embedding_x),
        ("both", both_x),
    ):
        auc, r2 = cross_validated(matrix, y, args.seed)
        results[name] = (auc, r2)
        logger.info(f"  {name:<24} AUC {auc:.4f}   pseudo-R2 {r2:.4f}")

    r2_s = results["sentence + confounders"][1]
    r2_e = results["embedding"][1]
    r2_b = results["both"][1]
    partition = {
        "unique to embedding": r2_b - r2_s,
        "unique to sentence": r2_b - r2_e,
        "shared": r2_s + r2_e - r2_b,
    }
    logger.info("commonality analysis on pseudo-R2:")
    for name, value in partition.items():
        logger.info(f"    {name:<22} {value:+.4f}")

    rows = [
        {"model": name, "auc": auc, "pseudo_r2": r2}
        for name, (auc, r2) in results.items()
    ] + [
        {"model": f"[partition] {name}", "auc": None, "pseudo_r2": value}
        for name, value in partition.items()
    ]
    (paths.output_home / "tables" / paths.model_key).mkdir(
        parents=True, exist_ok=True
    )
    pl.DataFrame(rows).write_csv(
        paths.output_home / "tables" / paths.model_key
        / f"variance_partition_{args.subset}.csv"
    )

    # ---- sentence-conditioned residual
    ridge = Ridge(alpha=10.0).fit(sentence_x, embedding_x)
    residual = embedding_x - ridge.predict(sentence_x)
    kept = 1.0 - (residual.var(axis=0).sum() / embedding_x.var(axis=0).sum())
    logger.info(
        f"the sentence explains {kept:.1%} of the embedding's variance; "
        "the rest is the residual"
    )
    auc_residual, r2_residual = cross_validated(residual, y, args.seed)
    logger.info(
        f"  residual alone predicts death: AUC {auc_residual:.4f} "
        f"(pseudo-R2 {r2_residual:.4f})"
    )

    # within groups of patients sharing an IDENTICAL sentence, which is the
    # strictest version of "same trajectory, different outcome"
    text = frame["bpe_sentence_48h"].to_list()
    groups = {}
    for index, sentence in enumerate(text):
        groups.setdefault(sentence, []).append(index)
    matched = [
        index for members in groups.values() if len(members) >= args.min_group
        for index in members
    ]
    matched = np.array(sorted(matched))
    usable = [
        members for members in groups.values()
        if len(members) >= args.min_group and len(set(y[members])) > 1
    ]
    logger.info(
        f"  matched groups: {len(usable):,} sentences shared by at least "
        f"{args.min_group} patients with both outcomes present, covering "
        f"{len(matched):,} patients"
    )
    auc_matched = float("nan")
    if len(matched) > 200 and len(set(y[matched])) > 1:
        auc_matched, _ = cross_validated(residual[matched], y[matched], args.seed)
        logger.info(
            f"  residual within matched sentences: AUC {auc_matched:.4f}"
        )

    # ---- name the residual direction by what loads on it
    model = LogisticRegression(max_iter=3000, C=1.0).fit(residual, y)
    score = residual @ model.coef_[0]
    counts = window_token_counts(
        paths, OVERLAP, 6.0, frame["hospitalization_id"].to_list(), logger
    )
    token_ids = site_representative_tokens(counts, list(paths.sites), logger)
    from clifcs.tokens import vocabulary as token_vocabulary
    names, _families = token_vocabulary(str(paths.tokenizer_yaml))
    present = (
        counts.filter(pl.col("token_id").is_in(token_ids))
        .select("hospitalization_id", "token_id").unique()
    )
    index_of = {p: i for i, p in enumerate(frame["hospitalization_id"].to_list())}
    carried = {}
    for patient, token in zip(
        present["hospitalization_id"].to_list(), present["token_id"].to_list()
    ):
        if patient in index_of:
            carried.setdefault(int(token), []).append(index_of[patient])

    loadings = []
    standard = (score - score.mean()) / (score.std() + 1e-9)
    for token, rows_with in carried.items():
        if len(rows_with) < 30 or len(rows_with) > len(score) - 30:
            continue
        mask = np.zeros(len(score), dtype=bool)
        mask[rows_with] = True
        difference = standard[mask].mean() - standard[~mask].mean()
        loadings.append({
            "token": names.get(token, str(token)),
            "patients_with_token": int(mask.sum()),
            "residual_score_gap": float(difference),
        })
    table = pl.DataFrame(loadings).sort(
        pl.col("residual_score_gap").abs(), descending=True
    )
    table.write_csv(
        paths.output_home / "tables" / paths.model_key
        / f"residual_tokens_{args.subset}.csv"
    )
    logger.info(
        "tokens most associated with the outcome-predicting residual "
        "(same sentence, worse outcome):"
    )
    for row in table.head(12).iter_rows(named=True):
        logger.info(
            f"    {row['token']:<34} gap {row['residual_score_gap']:+.3f} "
            f"({row['patients_with_token']:,} patients)"
        )
    logger.info("15 done")


if __name__ == "__main__":
    main()

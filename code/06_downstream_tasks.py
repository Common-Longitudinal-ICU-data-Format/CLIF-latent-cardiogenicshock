#!/usr/bin/env python3

"""
06 -- downstream prediction from the embeddings, raw vs site-corrected

Three tasks, each anchored at the moment the model read the patient and asking
what happens in the next 24 hours: in-hospital death, first invasive
ventilation, and escalation of vasopressor dose.

Each is scored three ways, because they answer different questions:

  within-site   train and test inside one hospital -- does the embedding carry
                clinical signal at all, free of any site confounding
  cross-site    train on one hospital, test on the other -- does a model built
                at one site transport to another. This is the headline: it is
                where site correction can help, and the question the Rush
                expansion actually poses
  pooled        train and test on everything -- the number site correction
                should *hurt*, since site genuinely predicts outcome here
                (UCMC mortality 23.8% vs NU 14.0%)

A clinical baseline (age, sex, last labs and vitals before the anchor,
vasopressor burden) runs alongside, because an AUC with nothing to compare it
to is not interpretable.

Usage
    python code/06_downstream_tasks.py
    python code/06_downstream_tasks.py --model GEM-mimic-icu --task mortality_24h
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clifcs import ANCHORS, get_logger, load_cohort_cfg, load_paths
from clifcs.correction import METHODS, fit
from clifcs.labels import TASKS, anchor_times, baseline_features

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PCA_COMPONENTS = 50
N_BOOTSTRAP = 200


def embedding_matrix(frame: pl.DataFrame) -> np.ndarray:
    columns = sorted(c for c in frame.columns if c.startswith("emb_"))
    return frame.select(columns).to_numpy().astype(np.float64)


def load_embeddings(paths, anchor: str) -> pl.DataFrame:
    sources = sorted(
        p
        for p in paths.patient_level_home.glob(f"embeddings_anchor-{anchor}_*.parquet")
        if "_share-safe" not in p.name and "_corrected-" not in p.name
    )
    if not sources:
        raise FileNotFoundError(f"no embeddings for {anchor} in {paths.patient_level_home}")
    return pl.concat([pl.read_parquet(p) for p in sources], how="diagonal_relaxed")


def build_labels(paths, cfg, anchor: str, logger) -> pl.DataFrame:
    """labels and baseline features for every patient at this anchor"""
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    horizon = float(cfg.downstream.horizon_hours)
    parts = []
    for site in paths.sites:
        data_dir = paths.raw_subset(site)
        anchors = anchor_times(paths, site, anchor, cohort)
        frame = anchors.select("hospitalization_id", "site", "anchor_dttm")
        for task in cfg.downstream.tasks:
            labelled = TASKS[task](data_dir, anchors, horizon, cfg).rename(
                {"label": f"{task}__label", "eligible": f"{task}__eligible"}
            )
            frame = frame.join(labelled, on="hospitalization_id", how="left")
        frame = frame.join(
            baseline_features(data_dir, anchors), on="hospitalization_id", how="left"
        )
        parts.append(frame)
        for task in cfg.downstream.tasks:
            eligible = frame.filter(pl.col(f"{task}__eligible"))
            logger.info(
                f"[{anchor}/{site}] {task:<24} eligible {eligible.height:>6}  "
                f"positives {int(eligible[f'{task}__label'].sum()):>5} "
                f"({eligible[f'{task}__label'].mean():.1%})"
            )
    return pl.concat(parts, how="diagonal_relaxed")


def bootstrap_auc(y: np.ndarray, score: np.ndarray, seed: int = 42) -> tuple[float, float]:
    """percentile CI on AUROC, resampling patients"""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(N_BOOTSTRAP):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        values.append(roc_auc_score(y[index], score[index]))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


#: the two heads compared on every feature set
CLASSIFIERS = ("logistic", "gbm")


def classifier(kind: str, head: str):
    """
    a linear and a non-linear head on identical features

    Worth running both because 05 showed site is recoverable non-linearly but
    not linearly: if the same asymmetry holds for outcome, a logistic probe
    would understate what the embedding knows, and the gap between the heads
    is the part of the signal that is not linearly decodable.
    """
    steps = (
        [StandardScaler(), PCA(n_components=PCA_COMPONENTS, random_state=42)]
        if kind == "embedding"
        else [SimpleImputer(strategy="median"), StandardScaler()]
    )
    head_model = (
        LogisticRegression(max_iter=5000, C=0.1 if kind == "embedding" else 1.0)
        if head == "logistic"
        else HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, random_state=42
        )
    )
    return make_pipeline(*steps, head_model)


def score_split(
    name: str,
    kind: str,
    head: str,
    train_x,
    train_y,
    test_x,
    test_y,
    min_positives: int,
) -> dict | None:
    if train_y.sum() < min_positives or test_y.sum() < min_positives:
        return None
    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
        return None
    model = classifier(kind, head).fit(train_x, train_y)
    score = model.predict_proba(test_x)[:, 1]
    low, high = bootstrap_auc(test_y, score)
    return {
        "evaluation": name,
        "classifier": head,
        "auroc": roc_auc_score(test_y, score),
        "auroc_lo": low,
        "auroc_hi": high,
        "auprc": average_precision_score(test_y, score),
        "prevalence": float(test_y.mean()),
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
    }


def run_task(
    paths, frame: pl.DataFrame, task: str, anchor: str, cfg, baseline_cols, logger
) -> list[dict]:
    """every correction x evaluation combination for one task"""
    min_positives = int(cfg.downstream.min_positives)
    eligible = frame.filter(pl.col(f"{task}__eligible").fill_null(False))
    if eligible.height == 0:
        logger.warning(f"[{anchor}] {task}: nobody eligible")
        return []

    y = eligible[f"{task}__label"].fill_null(False).to_numpy().astype(int)
    sites = np.asarray(eligible["site"].to_list())
    x = embedding_matrix(eligible)
    baseline = eligible.select(baseline_cols).to_numpy().astype(np.float64)
    present = [s for s in paths.sites if s in set(sites)]
    reference = present[0]

    # Drawn ONCE and reused for every feature set and correction. Re-drawing
    # per loop would score each arm on a different random test set, so a
    # difference between raw and corrected could be nothing but split noise.
    splits = _splits(sites, present, np.random.default_rng(42))
    rows = []

    # the clinical baseline, the same ways, so it is comparable
    for head in CLASSIFIERS:
        for name, train_mask, test_mask in splits:
            got = score_split(
                name, "baseline", head, baseline[train_mask], y[train_mask],
                baseline[test_mask], y[test_mask], min_positives,
            )
            if got:
                rows.append({**got, "task": task, "anchor": anchor,
                             "features": "clinical baseline", "correction": "-"})

    for method in METHODS:
        for name, train_mask, test_mask in splits:
            # the correction is fitted on this split's training patients only,
            # then reused by both heads so they see identical features
            correction = fit(method, x[train_mask], sites[train_mask], reference)
            train_c = correction.apply(x[train_mask], sites[train_mask])
            test_c = correction.apply(x[test_mask], sites[test_mask])
            for head in CLASSIFIERS:
                got = score_split(
                    name, "embedding", head, train_c, y[train_mask],
                    test_c, y[test_mask], min_positives,
                )
                if got:
                    rows.append({**got, "task": task, "anchor": anchor,
                                 "features": "embedding", "correction": method})
    return rows


def _splits(sites: np.ndarray, present: list[str], rng):
    """(name, train mask, test mask) for within-site, cross-site and pooled"""
    out = []
    for site in present:
        inside = sites == site
        index = np.flatnonzero(inside)
        held = rng.permutation(index)[: max(1, int(0.3 * len(index)))]
        test_mask = np.zeros(len(sites), bool)
        test_mask[held] = True
        out.append((f"within:{site}", inside & ~test_mask, test_mask))

    for source in present:
        for target in present:
            if source != target:
                out.append((f"{source}->{target}", sites == source, sites == target))

    index = np.arange(len(sites))
    held = rng.permutation(index)[: int(0.3 * len(index))]
    test_mask = np.zeros(len(sites), bool)
    test_mask[held] = True
    out.append(("pooled", ~test_mask, test_mask))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", choices=ANCHORS, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    cfg = load_cohort_cfg()
    logger = get_logger("06_downstream_tasks", paths.log_home / "06_downstream_tasks.log")

    tasks = [args.task] if args.task else list(cfg.downstream.tasks)
    rows = []

    for anchor in ((args.anchor,) if args.anchor else ANCHORS):
        logger.info(f"=== {paths.model_key} / anchor {anchor}")
        # The cache is keyed on nothing but the anchor, so a change to the
        # enrollment criteria would silently reuse labels for the old cohort.
        # Compare the patient sets and rebuild when they differ.
        label_file = paths.cohort_home / f"downstream_labels_anchor-{anchor}.parquet"
        cohort_ids = set(
            pl.read_parquet(
                paths.cohort_home / "cohort.parquet", columns=["hospitalization_id"]
            )["hospitalization_id"].to_list()
        )
        labels = None
        if label_file.is_file():
            cached = pl.read_parquet(label_file)
            if set(cached["hospitalization_id"].to_list()) == cohort_ids:
                labels = cached
                logger.info(f"[{anchor}] reusing {label_file.name}")
            else:
                logger.info(
                    f"[{anchor}] {label_file.name} covers {cached.height} patients "
                    f"but the cohort has {len(cohort_ids)}; rebuilding"
                )
        if labels is None:
            labels = build_labels(paths, cfg, anchor, logger)
            labels.write_parquet(label_file)
            logger.info(f"[{anchor}] wrote {label_file.name}")

        embeddings = load_embeddings(paths, anchor)
        frame = embeddings.join(
            labels.drop("site"), on="hospitalization_id", how="inner"
        )
        baseline_cols = [
            c
            for c in labels.columns
            if c.startswith(("lab_", "vtl_")) or c in ("nee_at_anchor", "n_vasopressors")
        ]
        frame = frame.with_columns(
            sex_male=(pl.col("sex_category") == "Male").cast(pl.Int8)
        )
        baseline_cols = baseline_cols + ["age_at_admission", "sex_male"]
        logger.info(f"[{anchor}] baseline features: {len(baseline_cols)}")

        for task in tasks:
            rows.extend(
                run_task(paths, frame, task, anchor, cfg, baseline_cols, logger)
            )

    if not rows:
        logger.warning("no results")
        return

    table = pl.DataFrame(rows).select(
        "anchor", "task", "features", "correction", "classifier", "evaluation",
        "auroc", "auroc_lo", "auroc_hi", "auprc", "prevalence", "n_train", "n_test",
    )
    out = paths.table_home / "downstream_tasks.csv"
    table.write_csv(out)
    logger.info(f"wrote {out}")

    for task in table["task"].unique(maintain_order=True):
        logger.info(f"--- {task}")
        part = table.filter(pl.col("task") == task)
        for row in part.sort("evaluation", "features", "correction", "classifier").iter_rows(named=True):
            label = (
                "clinical baseline"
                if row["features"] != "embedding"
                else f"embedding/{row['correction']}"
            ) + f" [{row['classifier']}]"
            logger.info(
                f"    {row['anchor']:<6} {row['evaluation']:<14} {label:<28} "
                f"AUROC {row['auroc']:.3f} [{row['auroc_lo']:.3f}-{row['auroc_hi']:.3f}]  "
                f"AUPRC {row['auprc']:.3f}  prev {row['prevalence']:.1%}"
            )
    logger.info("06 done")


if __name__ == "__main__":
    main()

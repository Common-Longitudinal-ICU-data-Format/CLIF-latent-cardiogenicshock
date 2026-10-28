#!/usr/bin/env python3

"""
22_pattern_response.py -- same treatment approach, different response

A PrefixSpan pattern is an ordered sequence of BPE words that many patients
share: "6_ 6_ 646_ 0_" is a combination held steady, moved, then stopped. Among
the patients who all followed one such pattern, the treatment approach is
roughly held fixed. This asks whether the GEM embedding then separates their
outcomes -- which is the question of whether there are populations that respond
differently to broadly the same management.

    python code/22_pattern_response.py --subset cardiogenic

WHAT A PATTERN DOES AND DOES NOT HOLD FIXED. It fixes the ORDER of recorded
MAP-and-pressor states, which is a reasonable proxy for the management
approach. It does not fix dose, timing, duration, or anything happening outside
the pressor-and-MAP space -- two patients matching "3_ 3_ 0_" may have had very
different norepinephrine doses for very different lengths of time. So "the same
treatment approach" is a claim about shape, not about equivalence, and the
within-pattern comparison is adjusted-by-matching only to the extent that shape
captures management.

THE TEST IS WITHIN-PATTERN, WHICH IS THE WHOLE POINT. An embedding that
predicts death across the cohort tells us only that it encodes severity, which
is established. An embedding that predicts death AMONG patients whose recorded
pressor course had the same shape is saying something the course does not: that
two patients managed alike were not alike. That is the responder-heterogeneity
claim, and it is why every AUC here is computed inside a pattern and compared
against the same model fitted on the whole cohort.

CONFOUNDING SURVIVES THIS DESIGN. Patients are not assigned to patterns; a
clinician arrives at a management shape partly by reading the patient. So a
within-pattern embedding signal may be the embedding recovering the severity
that drove the management in the first place, rather than identifying patients
who would respond differently to a different approach. Nothing here separates
those, and the counterfactual question -- would this patient have done better
managed otherwise -- is not answerable from observational trajectories.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import types

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.patterns import _contains_in_order

OVERLAP = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]
COMPONENTS = 30
#: patients needed inside a pattern before an AUC is reported for it. Five-fold
#: cross-validation on fewer puts a handful of deaths in each test fold and the
#: estimate swings on individuals.
MIN_PATIENTS = 250
MIN_EVENTS = 40


def auc_within(x, y, seed, folds=5):
    """cross-validated AUC, or NaN when the subgroup cannot support one"""
    labels = np.asarray(y).astype(np.int8)
    if len(labels) < MIN_PATIENTS or labels.sum() < MIN_EVENTS:
        return float("nan")
    if (len(labels) - labels.sum()) < MIN_EVENTS:
        return float("nan")
    scores = []
    for train, test in StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    ).split(x, labels):
        model = LogisticRegression(max_iter=3000).fit(x[train], labels[train])
        scores.append(
            roc_auc_score(labels[test], model.predict_proba(x[test])[:, 1])
        )
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic"))
    parser.add_argument("--max-patterns", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "22_pattern_response", paths.log_home / "22_pattern_response.log"
    )
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    fine = pl.read_parquet(
        paths.patient_level_home / f"fine_state_sequences_{args.subset}.parquet"
    ).filter(pl.col("fine_bpe_sentence").is_not_null())
    labels = pl.read_parquet(
        paths.patient_level_home
        / f"longitudinal_clusters_{args.subset}.parquet"
    )
    patterns_file = tables / f"sentence_patterns_fine_{args.subset}.csv"
    if not patterns_file.is_file():
        logger.error(f"no {patterns_file.name}; run 18 first")
        return
    patterns = pl.read_csv(
        patterns_file, schema_overrides={"pattern": pl.String}
    )

    # ---- the embedding, averaged over the overlap window per patient
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
    )

    outcome_columns = [
        c for c in ("in_hospital_mortality", "new_imv_after", "new_rrt_after")
        if c in labels.columns
    ]
    frame = (
        fine.select("hospitalization_id", "fine_bpe_sentence", "site")
        .join(labels.select(["hospitalization_id"] + outcome_columns),
              on="hospitalization_id", how="inner")
        .join(per_patient, on="hospitalization_id", how="inner")
    )
    logger.info(
        f"{frame.height:,} patients with a sentence, an embedding and an outcome"
    )
    sentences = [s.split() for s in frame["fine_bpe_sentence"].to_list()]

    embedding = frame.select(emb_columns).to_numpy().astype(np.float64)
    reducer = PCA(n_components=COMPONENTS, random_state=args.seed)
    components = reducer.fit_transform(
        (embedding - embedding.mean(axis=0)) / (embedding.std(axis=0) + 1e-9)
    )
    logger.info(
        f"embedding reduced to {COMPONENTS} components "
        f"({reducer.explained_variance_ratio_.sum():.1%} of variance)"
    )

    candidates = (
        patterns.filter(~pl.col("is_a_run") & (pl.col("words") >= 3))
        if "is_a_run" in patterns.columns else patterns
    ).head(args.max_patterns)["pattern"].to_list()
    logger.info(f"testing {len(candidates)} patterns")

    rows = []
    for outcome in outcome_columns:
        y_all = frame[outcome].fill_null(False).cast(pl.Int8).to_numpy()
        baseline = auc_within(components, y_all, args.seed)
        logger.info(
            f"== {outcome}: base rate {y_all.mean():.1%}, "
            f"whole-cohort AUC from the embedding {baseline:.4f}"
        )
        for pattern in candidates:
            words = tuple(pattern.split())
            mask = np.array([
                _contains_in_order(s, words) for s in sentences
            ])
            if mask.sum() < MIN_PATIENTS:
                continue
            inside = auc_within(components[mask], y_all[mask], args.seed)
            if np.isnan(inside):
                continue
            rows.append({
                "outcome": outcome,
                "pattern": pattern,
                "words": len(words),
                "patients_with_pattern": int(mask.sum()),
                "events_with_pattern": int(y_all[mask].sum()),
                "event_rate_with_pattern": float(y_all[mask].mean()),
                "event_rate_without": float(y_all[~mask].mean()),
                "auc_within_pattern": inside,
                "auc_whole_cohort": baseline,
                "auc_change": inside - baseline,
            })
            logger.info(
                f"    {pattern:<34} n={int(mask.sum()):>5,}  "
                f"rate {y_all[mask].mean():>5.1%}  "
                f"AUC within {inside:.4f} ({inside - baseline:+.4f})"
            )

    if not rows:
        logger.warning("no pattern was large enough to test")
        return
    table = pl.DataFrame(rows).sort(
        ["outcome", "auc_within_pattern"], descending=[False, True]
    )
    table.write_csv(tables / f"pattern_response_{args.subset}.csv")

    # The headline comparison, stated once rather than left to the reader:
    # if the within-pattern AUC holds up near the whole-cohort AUC, the
    # embedding is separating outcomes among patients managed alike.
    for outcome in outcome_columns:
        part = table.filter(pl.col("outcome") == outcome)
        if part.is_empty():
            continue
        logger.info(
            f"  {outcome}: within-pattern AUC runs "
            f"{part['auc_within_pattern'].min():.3f} to "
            f"{part['auc_within_pattern'].max():.3f} "
            f"(median {part['auc_within_pattern'].median():.3f}) against "
            f"{part['auc_whole_cohort'][0]:.3f} across the whole cohort"
        )

    # ---- how patients move through the patterns
    # Each patient as the ORDERED set of candidate patterns they carry, which
    # is the "movement through the vocabulary" view: not which words they used
    # but which recognised shapes their week contains.
    carried = []
    for index, words in enumerate(sentences):
        held = [
            p for p in candidates
            if _contains_in_order(words, tuple(p.split()))
        ]
        carried.append(held)
    counts = pl.DataFrame({
        "patterns_carried": [len(c) for c in carried],
    })
    logger.info(
        f"patterns per patient: median {counts['patterns_carried'].median():.0f}, "
        f"{(counts['patterns_carried'] == 0).mean():.1%} carry none of them"
    )
    co_occurrence = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1:]:
            both = sum(1 for c in carried if left in c and right in c)
            if both < 100:
                continue
            only_left = sum(1 for c in carried if left in c and right not in c)
            only_right = sum(1 for c in carried if right in c and left not in c)
            co_occurrence.append({
                "pattern_a": left, "pattern_b": right,
                "both": both, "only_a": only_left, "only_b": only_right,
                "jaccard": both / max(both + only_left + only_right, 1),
            })
    if co_occurrence:
        pl.DataFrame(co_occurrence).sort("jaccard", descending=True).write_csv(
            tables / f"pattern_cooccurrence_{args.subset}.csv"
        )
        logger.info(f"  {len(co_occurrence)} pattern pairs co-occur in 100+ patients")
    logger.info("22 done")


if __name__ == "__main__":
    main()

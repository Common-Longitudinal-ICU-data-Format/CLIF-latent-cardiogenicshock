#!/usr/bin/env python3

"""
33_state_sequences.py -- common courses through the hourly response states,
and whether the states localise in the 1024-d latent space

Three things k-means on composition could not do:

  SEQUENCES. Composition is order-free, so it cannot distinguish a patient who
  deteriorated then recovered from one who recovered then deteriorated. BPE
  over the state string finds recurring courses, which is the structure a
  2-cluster partition of composition was never able to see.

  TRAILING OFF-PRESSOR HOURS REMOVED. 78% of hours are "off pressors" and most
  of them sit at the end of a trajectory, saying repeatedly that a patient who
  recovered is still recovered. The trailing run collapses to ONE symbol;
  off-pressor hours between periods of support are a relapse and are kept in
  full.

  LOCALISATION IN 1024-d, NOT PaCMAP. PaCMAP distances are not metric, so
  "these states overlap on the map" is not a claim about the representation.
  The separability tests here run in the corrected 1024-d space.

    python code/33_state_sequences.py --subset cardiogenic
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import types
from collections import Counter

import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.bpe import apply_merges, learn_merges
from clifcs.config import CONFIG_HOME
from clifcs.regions import (
    HOURLY_REGION_ORDER, STATE_SYMBOL, SYMBOL_STATE, state_sequences,
)

OVERLAP = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]


def spell(word: str) -> str:
    """a BPE word written out as the states it contains"""
    names = [SYMBOL_STATE.get(c, c) for c in word]
    if len(set(names)) == 1 and len(names) > 1:
        return f"{names[0]}  x{len(names)}"
    return " -> ".join(names)


def collapse_runs(sequence: str) -> str:
    """run-length collapse, so a course is its ORDER not its durations"""
    out = []
    for character in sequence:
        if not out or out[-1] != character:
            out.append(character)
    return "".join(out)


#: states grouped by the ACTION that produced them. Within a group the action
#: is identical and only the response differs.
ACTION_GROUPS = {
    "escalation": ("responsive", "nonresponsive", "deteriorating"),
    "wean": ("wean held", "wean with a dip", "failed wean"),
    "hold": ("stable hold", "drifting hold"),
}


def conditional_probe(joined, emb, logger, seed=42, sample=2500):
    """
    within one action, can the embedding tell the responses apart

    THIS IS THE TEST THAT SEPARATES PREDICTION FROM RECONSTRUCTION, and without
    it the unconditional AUCs mean very little. A state is an action plus a
    response. The embedding at hour t was built from tokens at or before t, so
    it has ALREADY SEEN the action -- the dose change ran from t-1 to t -- and
    has NOT seen the response, which is the MAP change from t to t+1.

    So an unconditional AUC of 0.85 for "wean held" mixes recognising that this
    was a wean, which is reconstruction of the model's own input, with
    predicting that the pressure then held, which is not. Restricting to hours
    sharing one action removes the first entirely: every row in a comparison
    had the same thing done to it, so anything the probe recovers is about the
    response it could not see.

    An AUC near 0.5 here, alongside high unconditional AUCs, would mean the
    embedding knows what was done and nothing about what follows.
    """
    from itertools import combinations

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(seed)
    rows = []
    for action, states in ACTION_GROUPS.items():
        for left, right in combinations(states, 2):
            a = joined.filter(pl.col("region") == left)
            b = joined.filter(pl.col("region") == right)
            n = min(sample, a.height, b.height)
            if n < 200:
                logger.info(
                    f"    [{action}] {left} vs {right}: "
                    f"{min(a.height, b.height):,} too few, skipped"
                )
                continue
            x = np.vstack([
                a[rng.choice(a.height, size=n, replace=False)]
                .select(emb).to_numpy(),
                b[rng.choice(b.height, size=n, replace=False)]
                .select(emb).to_numpy(),
            ]).astype(np.float64)
            y = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.int8)
            x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
            scores = []
            for train, test in StratifiedKFold(
                n_splits=4, shuffle=True, random_state=seed
            ).split(x, y):
                model = LogisticRegression(max_iter=2000, C=0.1).fit(
                    x[train], y[train]
                )
                scores.append(roc_auc_score(
                    y[test], model.predict_proba(x[test])[:, 1]
                ))
            auc = float(np.mean(scores))
            rows.append({
                "action": action, "state_a": left, "state_b": right,
                "balanced_n": n, "auc_within_action": auc,
            })
            logger.info(
                f"    [{action}] {left} vs {right}: AUC {auc:.4f} (n={n:,} each)"
            )
    return pl.DataFrame(rows)


def localisation_1024(labelled, paths, subset, logger, seed=42,
                      sample_per_state=3000):
    """
    do the hourly states occupy distinguishable regions of the 1024-d space

    THE TEST IS A PROBE, NOT A DISTANCE. Silhouette in 1024 dimensions is
    dominated by the curse of dimensionality: distances concentrate, so every
    silhouette tends to zero whether or not the classes separate. A supervised
    probe asks the question directly -- given a corrected embedding, can the
    state be recovered -- and an AUC near 0.5 means the state is not written
    anywhere in the representation.

    ONE-VERSUS-REST PER STATE, because a multiclass accuracy would be carried
    entirely by the two largest states. Each state gets its own balanced AUC
    against everything else, which is the per-state version of "does this
    localise".

    WHAT A HIGH AUC WOULD NOT MEAN. The embedding read the MAP and pressor
    tokens, so recovering a state built from MAP and pressors is partly
    recovering its own inputs. A high AUC says the information survived into
    the representation; it does not say the representation discovered anything.
    """
    import importlib.util
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    spec = importlib.util.spec_from_file_location(
        "traj", pathlib.Path(__file__).resolve().parent / "09_trajectories.py"
    )
    traj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(traj)
    grid_args = types.SimpleNamespace(
        subset=subset, start_hours=-24.0, end_hours=48.0, step_hours=6.0,
        correction="coral", anchor="onset", force=False, min_tokens=8,
    )
    grid = traj.load_grid(paths, grid_args, logger)
    grid = traj.apply_correction(paths, grid, "coral", "onset", logger)
    grid = grid.filter(pl.col("offset_hours").is_in(OVERLAP))
    emb = sorted(c for c in grid.columns if c.startswith("emb_"))

    joined = grid.join(
        labelled.select("hospitalization_id", "offset_hours", "region"),
        on=["hospitalization_id", "offset_hours"], how="inner",
    )
    logger.info(
        f"  {joined.height:,} grid points carry a state, in {len(emb)}-d"
    )
    rng = np.random.default_rng(seed)
    rows = []
    for state in HOURLY_REGION_ORDER:
        positive = joined.filter(pl.col("region") == state)
        negative = joined.filter(pl.col("region") != state)
        n = min(sample_per_state, positive.height, negative.height)
        if n < 200:
            logger.info(f"    {state:<20} only {positive.height:,}, skipped")
            continue
        take_p = rng.choice(positive.height, size=n, replace=False)
        take_n = rng.choice(negative.height, size=n, replace=False)
        x = np.vstack([
            positive[take_p].select(emb).to_numpy(),
            negative[take_n].select(emb).to_numpy(),
        ]).astype(np.float64)
        y = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.int8)
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
        scores = []
        for train, test in StratifiedKFold(
            n_splits=4, shuffle=True, random_state=seed
        ).split(x, y):
            model = LogisticRegression(max_iter=2000, C=0.1).fit(
                x[train], y[train]
            )
            scores.append(roc_auc_score(
                y[test], model.predict_proba(x[test])[:, 1]
            ))
        auc = float(np.mean(scores))
        rows.append({
            "state": state, "grid_points": positive.height,
            "balanced_n": n, "auc_1024d": auc,
        })
        logger.info(
            f"    {state:<20} n={positive.height:>7,}  "
            f"balanced AUC {auc:.4f}"
        )
    return pl.DataFrame(rows), joined, emb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--merges", type=int, default=120)
    parser.add_argument("--keep-off", type=int, default=1,
                        help="trailing off-pressor hours retained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "33_state_sequences", paths.log_home / "33_state_sequences.log"
    )
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)
    _ = OmegaConf.load(CONFIG_HOME / "dynamics.yaml")

    labelled = pl.read_parquet(
        paths.patient_level_home / f"hourly_regions_{args.subset}.parquet"
    )
    logger.info(
        f"{labelled.height:,} hourly states, "
        f"{labelled['hospitalization_id'].n_unique():,} patients"
    )

    grouped = state_sequences(labelled, logger, trim=True, keep=args.keep_off)
    sequences = grouped["sequence"].to_list()
    collapsed = [collapse_runs(s) for s in sequences]
    logger.info(
        f"  median course {int(np.median([len(s) for s in sequences]))} hours, "
        f"{int(np.median([len(s) for s in collapsed]))} after collapsing runs"
    )

    # ---- what are the common courses, as orders rather than durations
    logger.info("the commonest collapsed courses (order only, durations gone):")
    common = Counter(collapsed)
    rows = []
    for course, n in common.most_common(25):
        rows.append({
            "course": course,
            "patients": n,
            "share": n / max(len(collapsed), 1),
            "distinct_states": len(set(course)),
            "meaning": spell(course),
        })
        logger.info(
            f"    {course:<14} {n:>5,} ({n / len(collapsed):>5.1%})  "
            f"{spell(course)[:66]}"
        )
    pl.DataFrame(rows).write_csv(
        tables / f"state_courses_{args.subset}.csv"
    )
    logger.info(
        f"  {len(common):,} distinct collapsed courses among "
        f"{len(collapsed):,} patients; "
        f"{sum(1 for n in common.values() if n == 1) / max(len(common), 1):.0%} "
        "occur once"
    )

    # ---- BPE over the hourly states
    logger.info(f"learning {args.merges} BPE merges over the state strings:")
    merges = learn_merges(
        sequences, n_merges=args.merges, boundary="", logger=logger
    )
    segmented = [apply_merges(s, merges) for s in sequences]
    occurrences, carriers = Counter(), Counter()
    for tokens in segmented:
        occurrences.update(tokens)
        carriers.update(set(tokens))
    letters = sum(len(s) for s in sequences)
    words = sum(len(t) for t in segmented)
    logger.info(
        f"  {len(carriers):,} distinct words, compression "
        f"{letters / max(words, 1):.2f} letters per word"
    )
    vocab_rows = []
    for word, n in occurrences.most_common():
        vocab_rows.append({
            "word": word,
            "length": len(word),
            "distinct_states": len(set(word)),
            "is_a_run": len(set(word)) == 1,
            "occurrences": n,
            "patients": carriers[word],
            "share_of_patients": carriers[word] / max(len(sequences), 1),
            "meaning": spell(word),
        })
    vocab = pl.DataFrame(vocab_rows)
    vocab.write_csv(tables / f"state_bpe_vocabulary_{args.subset}.csv")
    logger.info("  commonest words that are NOT a single state repeated:")
    for row in vocab.filter(~pl.col("is_a_run")).head(14).iter_rows(named=True):
        logger.info(
            f"    {row['word']:<10} {row['share_of_patients']:>6.1%} of "
            f"patients   {row['meaning'][:62]}"
        )

    grouped.with_columns(
        collapsed=pl.Series(collapsed),
        bpe_sentence=pl.Series([" ".join(t) for t in segmented]),
        bpe_words=pl.Series([len(t) for t in segmented]),
    ).write_parquet(
        paths.patient_level_home / f"state_sequences_{args.subset}.parquet"
    )

    logger.info("localisation of each state in the corrected 1024-d space:")
    local, joined, emb = localisation_1024(
        labelled, paths, args.subset, logger, seed=args.seed
    )
    if not local.is_empty():
        local.write_csv(
            tables / f"state_localisation_1024_{args.subset}.csv"
        )
    logger.info(
        "within ONE action, can the embedding tell the responses apart "
        "(the action was already visible to it; the response was not):"
    )
    conditional = conditional_probe(joined, emb, logger, seed=args.seed)
    if not conditional.is_empty():
        conditional.write_csv(
            tables / f"state_conditional_probe_{args.subset}.csv"
        )
    logger.info("33 done")


if __name__ == "__main__":
    main()

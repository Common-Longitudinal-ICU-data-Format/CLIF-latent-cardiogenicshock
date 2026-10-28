#!/usr/bin/env python3

"""
16_embedding_to_sentence.py -- why do similar embeddings diverge into
different sentences?

15 fits `embedding ~ sentence` and keeps the residual, which answers the
opposite question: what the model saw that the state description missed. This
fits the MIRROR, `sentence ~ g(embedding)`, so the residual is the part of the
treatment course the embedding did NOT anticipate. Two patients the model
places together whose courses then part company appear here as large residuals
in opposite directions.

    python code/16_embedding_to_sentence.py --subset septic

NONE OF THE THREE VARIANCE COMPONENTS ANSWERS THIS. Unique-to-sentence,
unique-to-embedding and shared are all partitions of what predicts the OUTCOME.
A patient can sit entirely inside the shared component and still have been
treated unusually for where the model placed them. The divergence has to be
measured against a prediction of the sentence, which is what this builds.

THE RESIDUAL IS OUT OF FOLD. An in-sample residual from a 50-component ridge
over several thousand patients is shrunk towards zero wherever the model
happened to interpolate, which would make "unanticipated" partly a statement
about fold membership. Every patient's prediction comes from a model that
never saw them.

THE CONSTRUCTION TURNED OUT TO BE WEAK, AND THE WEAKNESS IS THE FIRST RESULT.
The docstring originally said that predicting the sentence from the embedding
is "near-trivially yes" because GEM read the same MAP values and pressor doses
the sentence thresholds. It is not: out of fold the embedding anticipates only
5.7% (septic) and 6.8% (cardiogenic) of the sentence's variance. A 50-component
linear map cannot say which of 83 sparse BPE words a patient receives, even
having read the underlying numbers.

That matters because it makes the residual nearly equal to the centred bag
itself. "Unanticipated treatment course" and "treatment course" are almost the
same vector here, so `divergence_alone` is not a measure of divergence -- it is
the sentence wearing a different name, which is why it lands at AUC 0.685
beside the 0.739 that 15 reports for the sentence outright. Read the
`increment_over_embedding` column and nothing else; the divergence-alone column
is kept only so the collapse is visible rather than inferred.

WHAT A LARGE RESIDUAL IS NOT. It is not unwarranted variation. Two readings
survive every result below:

  * the clinicians responded to something real that neither the 15-state
    sequence nor the embedding encoded, or
  * the patient was treated differently for reasons that had nothing to do
    with them.

Nothing in the divergence itself separates these, which is why the outcome arm
is here: if the divergence carries no outcome signal, the first reading has
little to stand on; if it does, the direction says which way.

THE SUPPORT OUTCOMES ARE PARTLY CIRCULAR AND TIMED TO LIMIT IT. New
ventilation and new dialysis are counted only when they begin after +48h, past
the window the sentence covers; GEM still read the RESP and CRRT tokens inside
it, so some leak survives. Mortality carries no such leak and stays the
primary. ICU-free days replaces length of stay because dying on day two
produces a stay of two days, which in a mortality-adjacent analysis inverts the
result.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import types

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.perfusion import PERFUSION_STATES, STATE_SYMBOLS

#: the only stretch where the state grid and the embedding grid overlap
OVERLAP = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]
COMPONENTS = 50

#: (column, label, binary?). Mortality first: it is the one outcome the
#: embedding cannot have read directly.
OUTCOMES = (
    ("in_hospital_mortality", "died in hospital", True),
    ("icu_free_days", "ICU-free days to 28", False),
    ("new_imv_after", "ventilation begun after +48h", True),
    ("new_rrt_after", "dialysis begun after +48h", True),
)


def word_direction(word: str) -> str:
    """
    does this BPE word spell an escalation, a wean, or a holding pattern

    Read off the dynamics field of each state the word contains. A word is an
    escalation if more of its letters are rising than falling, and so on; ties
    and all-steady words are "steady". Coarse on purpose -- the question is
    whether the model systematically fails to anticipate one DIRECTION of
    change, and a three-way split answers that without pretending the dynamics
    field is finer than its deadband allows.
    """
    rising = falling = 0
    for character in word:
        index = STATE_SYMBOLS.find(character)
        if not 0 <= index < len(PERFUSION_STATES):
            continue
        name = PERFUSION_STATES[index]
        if name.endswith("rising"):
            rising += 1
        elif name.endswith("falling"):
            falling += 1
    if rising > falling:
        return "escalation"
    if falling > rising:
        return "wean"
    return "steady"


def out_of_fold(x, y, seed, alpha=10.0, folds=5):
    """ridge predictions for every row, from a model that never saw that row"""
    predicted = np.zeros_like(y, dtype=np.float64)
    for train, test in KFold(
        n_splits=folds, shuffle=True, random_state=seed
    ).split(x):
        predicted[test] = Ridge(alpha=alpha).fit(
            x[train], y[train]
        ).predict(x[test])
    return predicted


def scored(x, y, binary, seed, folds=5):
    """AUC for a binary outcome, out-of-fold R2 for a continuous one"""
    if binary:
        labels = np.asarray(y).astype(np.int8)
        if len(set(labels.tolist())) < 2:
            return float("nan")
        scores = []
        for train, test in StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed
        ).split(x, labels):
            model = LogisticRegression(max_iter=3000).fit(
                x[train], labels[train]
            )
            scores.append(roc_auc_score(
                labels[test], model.predict_proba(x[test])[:, 1]
            ))
        return float(np.mean(scores))
    predicted = np.zeros(len(y))
    for train, test in KFold(
        n_splits=folds, shuffle=True, random_state=seed
    ).split(x):
        predicted[test] = LinearRegression().fit(
            x[train], y[train]
        ).predict(x[test])
    return float(
        1.0 - ((y - predicted) ** 2).sum()
        / max(((y - y.mean()) ** 2).sum(), 1e-9)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="septic",
                        choices=("septic", "cardiogenic"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "16_embedding_to_sentence",
        paths.log_home / "16_embedding_to_sentence.log",
    )
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    labels = pl.read_parquet(
        paths.patient_level_home / f"longitudinal_clusters_{args.subset}.parquet"
    )
    if "bpe_sentence_48h" not in labels.columns:
        logger.error("no bpe_sentence_48h; run 11 first")
        return
    available = [c for c, _label, _binary in OUTCOMES if c in labels.columns]
    missing = [c for c, _label, _binary in OUTCOMES if c not in labels.columns]
    if missing:
        logger.warning(
            f"not in the labels parquet, so skipped: {', '.join(missing)} "
            "(re-run 11 to build the outcome panel)"
        )
    logger.info(f"outcomes available: {', '.join(available) or 'none'}")

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
        labels.select(
            ["hospitalization_id", "bpe_sentence_48h"]
            + [c for c in available if c != "site"]
        )
        .join(per_patient, on="hospitalization_id", how="inner")
    )
    logger.info(f"{frame.height:,} patients with a sentence and an embedding")
    if frame.height < 500:
        logger.error("too few patients to fit the mirror")
        return

    # ---- the two sides
    sentences = [s.split() for s in frame["bpe_sentence_48h"].to_list()]
    vocab = sorted({w for s in sentences for w in s})
    bag = np.zeros((len(sentences), len(vocab)), dtype=np.float64)
    position = {w: i for i, w in enumerate(vocab)}
    for row, words in enumerate(sentences):
        for word in words:
            bag[row, position[word]] += 1.0

    embedding = frame.select(emb_columns).to_numpy().astype(np.float64)
    reducer = PCA(n_components=COMPONENTS, random_state=args.seed)
    components = reducer.fit_transform(
        (embedding - embedding.mean(axis=0)) / (embedding.std(axis=0) + 1e-9)
    )
    logger.info(
        f"{len(vocab)} BPE words predicted from {COMPONENTS} components "
        f"({reducer.explained_variance_ratio_.sum():.1%} of embedding variance)"
    )

    predicted = out_of_fold(components, bag, args.seed)
    residual = bag - predicted
    centred = bag - bag.mean(axis=0)
    explained = 1.0 - (
        (residual ** 2).sum(axis=0)
        / np.maximum((centred ** 2).sum(axis=0), 1e-9)
    )
    overall = 1.0 - (residual ** 2).sum() / max((centred ** 2).sum(), 1e-9)
    logger.info(
        f"the embedding anticipates {overall:.1%} of the sentence variance "
        "out of fold; the remainder is the divergence"
    )

    # ---- 1. WHICH WORDS WERE UNPREDICTED
    rows = []
    for word, r2 in zip(vocab, explained):
        column = position[word]
        carriers = int((bag[:, column] > 0).sum())
        rows.append({
            "word": word,
            "length": len(word),
            "direction": word_direction(word),
            "patients": carriers,
            "share_of_patients": carriers / frame.height,
            "variance_anticipated": float(r2),
            "mean_residual_when_present": float(
                residual[bag[:, column] > 0, column].mean()
            ) if carriers else 0.0,
        })
    by_word = pl.DataFrame(rows).sort("variance_anticipated")
    by_word.write_csv(tables / f"sentence_residual_words_{args.subset}.csv")
    logger.info(
        "words the embedding anticipates least, among those in >2% of patients:"
    )
    for row in by_word.filter(
        pl.col("share_of_patients") > 0.02
    ).head(10).iter_rows(named=True):
        logger.info(
            f"    {row['word']:<10} {row['direction']:<10} "
            f"R2 {row['variance_anticipated']:>6.3f}  "
            f"{row['share_of_patients']:>5.1%} of patients"
        )

    # Is the failure one-sided? A model that anticipates weans but not
    # escalations is failing at the clinically consequential end.
    logger.info("variance anticipated, by what the word spells:")
    direction_rows = []
    for direction in ("escalation", "wean", "steady"):
        part = by_word.filter(
            (pl.col("direction") == direction) & (pl.col("length") > 1)
        )
        if part.is_empty():
            continue
        direction_rows.append({
            "direction": direction,
            "words": part.height,
            "median_variance_anticipated": float(
                part["variance_anticipated"].median()
            ),
            "patient_weighted_variance_anticipated": float(
                (part["variance_anticipated"] * part["patients"]).sum()
                / max(part["patients"].sum(), 1)
            ),
        })
        logger.info(
            f"    {direction:<12} {part.height:>3} words, median R2 "
            f"{part['variance_anticipated'].median():>6.3f}"
        )
    pl.DataFrame(direction_rows).write_csv(
        tables / f"sentence_residual_directions_{args.subset}.csv"
    )

    # ---- 2. HOW MUCH OF THE DIVERGENCE IS SITE
    sites = sorted(frame["site"].unique().to_list())
    magnitude = np.linalg.norm(residual, axis=1)[:, None]
    if len(sites) > 1:
        dummies = np.column_stack([
            (frame["site"] == s).cast(pl.Int8).to_numpy() for s in sites[:-1]
        ]).astype(np.float64)
        with_site = bag - out_of_fold(
            np.column_stack([components, dummies]), bag, args.seed
        )
        absorbed = 1.0 - (with_site ** 2).sum() / max(
            (residual ** 2).sum(), 1e-12
        )
        logger.info(
            f"adding site to the predictor absorbs {absorbed:.1%} of the "
            "divergence -- the share of being treated differently that is "
            "being treated at a different hospital"
        )
        # The stronger statement: can the hospital be READ OFF the divergence?
        # If so, the unanticipated part of the treatment course carries a
        # practice signature, whatever else it carries.
        site_index = np.array([sites.index(s) for s in frame["site"].to_list()])
        site_rows = []
        for which, name in enumerate(sites):
            auc = scored(
                residual, (site_index == which).astype(np.int8), True, args.seed
            )
            site_rows.append({
                "site": name,
                "patients": int((site_index == which).sum()),
                "auc_from_divergence": auc,
            })
            logger.info(
                f"    {name} recoverable from the divergence alone: AUC {auc:.4f}"
            )
        pl.DataFrame(site_rows + [{
            "site": "[share of divergence absorbed by site]",
            "patients": frame.height,
            "auc_from_divergence": float(absorbed),
        }]).write_csv(tables / f"sentence_residual_site_{args.subset}.csv")

    # ---- 3. DOES THE DIVERGENCE PREDICT OUTCOME
    # Magnitude alone conflates being treated unusually with being sicker: a
    # patient the model placed in a crowded, mild region who then escalates has
    # both a large residual and a bad outcome, and severity explains both. So
    # every outcome is scored three ways -- embedding alone, divergence alone,
    # and both -- and only the increment over the embedding is evidence that HOW
    # the patient was treated, beyond where the model placed them, mattered.
    escalation = [
        position[w] for w in vocab if word_direction(w) == "escalation"
    ]
    wean = [position[w] for w in vocab if word_direction(w) == "wean"]
    signed = np.column_stack([
        residual[:, escalation].sum(axis=1) if escalation
        else np.zeros(frame.height),
        residual[:, wean].sum(axis=1) if wean else np.zeros(frame.height),
    ])
    divergence = np.column_stack([magnitude, signed])
    logger.info(
        "does the unanticipated part of the treatment course predict outcome?"
    )
    outcome_rows = []
    for column, label, binary in OUTCOMES:
        if column not in frame.columns:
            continue
        present = frame[column].is_not_null().to_numpy()
        if present.sum() < 400:
            logger.warning(f"  {label}: too few non-null values, skipped")
            continue
        y = frame[column].fill_null(0).cast(pl.Float64).to_numpy()[present]
        positive = int(present.sum())
        if binary:
            y = (y > 0.5).astype(np.int8)
            # The rarest outcomes are the ones a five-fold AUC flatters: new
            # dialysis runs around 3% of the cohort, so a fold can hold a few
            # dozen events and the estimate swings on a handful of patients.
            # The count is published beside the AUC rather than inferred from
            # the cohort size, and anything below the floor is left out
            # instead of reported with a caveat nobody reads.
            positive = int(y.sum())
            if positive < 50:
                logger.warning(
                    f"  {label}: only {positive} events, below the floor for "
                    "a five-fold AUC -- skipped"
                )
                continue
        base = scored(components[present], y, binary, args.seed)
        alone = scored(divergence[present], y, binary, args.seed)
        both = scored(
            np.column_stack([components, divergence])[present], y, binary,
            args.seed,
        )
        outcome_rows.append({
            "outcome": label,
            "n": int(present.sum()),
            "n_events": positive if binary else None,
            "metric": "AUC" if binary else "R2",
            "embedding_alone": base,
            "divergence_alone": alone,
            "both": both,
            "increment_over_embedding": both - base,
            "partly_circular": column in ("new_imv_after", "new_rrt_after"),
        })
        logger.info(
            f"  {label:<30} {'AUC' if binary else 'R2 '} "
            f"embedding {base:>6.3f}  divergence {alone:>6.3f}  "
            f"both {both:>6.3f}  ({both - base:+.3f})"
        )
    if outcome_rows:
        pl.DataFrame(outcome_rows).write_csv(
            tables / f"sentence_residual_outcomes_{args.subset}.csv"
        )

    # the per-patient divergence, so other scripts can stratify on it
    pl.DataFrame({
        "hospitalization_id": frame["hospitalization_id"],
        "divergence": magnitude[:, 0],
        "unanticipated_escalation": signed[:, 0],
        "unanticipated_wean": signed[:, 1],
    }).write_parquet(
        paths.patient_level_home / f"sentence_divergence_{args.subset}.parquet"
    )
    logger.info("16 done")


if __name__ == "__main__":
    main()

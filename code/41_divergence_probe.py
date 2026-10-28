#!/usr/bin/env python3

"""
41_divergence_probe.py -- is the within-handbook divergence legible BEFORE it
happens, in the embedding, in the tokens, or in neither

40 established that patients given essentially the same treatment handbook
split into response groups whose mortality differs by up to 17 points, and
that none of the tabular covariates separates those groups at any useful
effect size -- the largest standardised difference was 0.37 and the top hit
was a duration. That is a statement about six columns, not about the model.
This script asks the question those six columns cannot answer: does the GEM
embedding, or the raw token stream, carry the divergence before it shows up
in the response course.

THREE PROBES, AND THEY ARE NOT INTERCHANGEABLE.

  LANDMARK. At a fixed hour -- 6, 12, 18, 24 -- predict the response group and
  the outcome from the embedding at that hour, within handbook. The feature is
  built from tokens at or before the landmark and the label is the future, so
  nothing leaks, and a patient who is not still on record at the landmark is
  simply absent. This is the honest number.

  PRE-DIVERGENCE. Pair patients from opposite response groups inside the same
  handbook, align their hourly response courses, and read the embedding at the
  last grid point STRICTLY BEFORE the pair parted company. Then ask which
  member of the pair goes on to die. This is the question as asked, and it is
  optimistic in a way the landmark is not: the features at that hour are
  clean, but WHICH hour to look at was chosen using the whole future course.
  The gap between this and the landmark probe is the size of that advantage,
  which is why both are reported and neither is reported alone.

  TOKENS. The same landmark, the same patients, the same folds, with the
  windowed token counts in place of the embedding -- and separately, the hour
  at which each token first appears, because "no token separates them" and "no
  token TIMING separates them" are different claims and only the second is
  about trajectory.

WHAT A CHANCE RESULT WOULD AND WOULD NOT MEAN. If the embedding lands at 0.5
within handbook, that does not say the divergence is unpredictable -- it says
this representation, pooled to one vector per timepoint and corrected for
site, does not linearly encode it. The probe is regularised logistic
regression on PCA scores; it is a floor on what is there, not a ceiling.

    python code/41_divergence_probe.py --subset cardiogenic
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import types

import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import (
    apply_handbook_defaults, get_logger, handbook_defaults, load_paths,
)
from clifcs.align import align_many, similarity_matrix
from clifcs.config import CONFIG_HOME
from clifcs.figures import save_table
from clifcs.regions import HOURLY_REGION_ORDER, STATE_SYMBOL
from clifcs.treatment import TREATMENT_SYMBOLS, collapse

RESPONSE_ALPHABET = "".join(
    STATE_SYMBOL[state] for state in HOURLY_REGION_ORDER
)

#: CMS small-cell suppression. Any group smaller than this is not reported.
SUPPRESS = 11

#: token families that must never reach a mortality probe, because they are
#: the outcome rather than a predictor of it. DSCG is the discharge
#: disposition -- it contains `expired` and `hospice` outright -- and EOS is
#: the record ending, which for a patient who died IS the death. A first run
#: of this script scored `DSCG//expired` as a top separator of the mortality
#: groups in handbook 0 at q=0.008, which is the leak announcing itself.
OUTCOME_FAMILIES = ("DSCG", "EOS")

#: code status is handled separately rather than simply dropped. A DNR placed
#: before the landmark predicts death strongly and is not physiology: it is a
#: goals-of-care decision that usually follows a prognosis conversation, so
#: predicting death from it is predicting the outcome from a judgement about
#: the outcome. The project rule is to censor these patients, and `--keep-code
#: -status` exists only to measure how much they were contributing.
CODE_STATUS_TOKENS = ("CODE//dnr", "CODE//dnr/dni")


def sibling(name: str):
    """load a numerically-named sibling script as a module"""
    spec = importlib.util.spec_from_file_location(
        name.split("_")[0] + "_mod",
        pathlib.Path(__file__).resolve().parent / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe(x, y, seed, folds=5, components=32, groups=None):
    """
    cross-validated AUC with everything fitted inside the fold

    The scaler and the PCA are part of the pipeline rather than applied once
    up front, because fitting either on all patients and then splitting leaks
    the test fold's covariance into the training step. With 1,024 features
    against a few hundred patients that leak is not small.

    Returns the AUC of the pooled out-of-fold predictions plus a bootstrap
    interval over patients. Pooling rather than averaging per-fold AUCs keeps
    the interval honest when a fold holds very few events.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y).astype(np.int8)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < folds:
        return None

    if groups is None:
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
        split = list(splitter.split(x, y))
    else:
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
        split = list(splitter.split(x, y, groups=groups))
    if not split:
        return None
    # SIZED ON THE SMALLEST TRAINING FOLD, not on n/folds. Grouped folds are
    # not equal-sized -- keeping both copies of a pair together makes them
    # uneven -- and a component count derived from the average asks PCA for
    # more components than the thinnest fold has rows, which is a hard error
    # rather than a degraded fit.
    smallest = min(len(train) for train, _ in split)
    n_components = int(min(components, x.shape[1], smallest - 1))
    if n_components < 1:
        return None

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=seed)),
        ("fit", LogisticRegression(max_iter=4000, C=0.1)),
    ])
    out = np.full(len(y), np.nan)
    for train, test in split:
        if len(np.unique(y[train])) < 2:
            continue
        model = pipeline.fit(x[train], y[train])
        out[test] = model.predict_proba(x[test])[:, 1]
    keep = ~np.isnan(out)
    if keep.sum() < folds or len(np.unique(y[keep])) < 2:
        return None

    auc = float(roc_auc_score(y[keep], out[keep]))
    rng = np.random.default_rng(seed)
    truth, score = y[keep], out[keep]
    draws = []
    for _ in range(400):
        pick = rng.integers(0, len(truth), len(truth))
        if len(np.unique(truth[pick])) < 2:
            continue
        draws.append(roc_auc_score(truth[pick], score[pick]))
    low, high = (
        (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
        if draws else (float("nan"), float("nan"))
    )
    return {"auc": auc, "low": low, "high": high, "n": int(keep.sum()),
            "events": int(truth.sum()), "features": int(x.shape[1]),
            "components": n_components}


def say(logger, label, result, indent=6):
    """one line per probe, or an explicit note that it could not be run"""
    pad = " " * indent
    if result is None:
        logger.info(f"{pad}{label:<34} not estimable (too few, or one class)")
        return
    logger.info(
        f"{pad}{label:<34} AUC {result['auc']:.3f} "
        f"[{result['low']:.3f}-{result['high']:.3f}]  "
        f"n={result['n']:,} events={result['events']:,} "
        f"p={result['features']:,}"
    )


def load_corrected_grid(paths, subset, step_hours, end_hours, logger):
    """
    the trajectory grid, site-corrected, restricted to the common grid points

    UCMC carries a 2-hour grid and the other two sites a 6-hour one, so the
    intersection is what any pooled probe can use. Running on UCMC's finer grid
    alone is possible and would place the pre-divergence read within two hours
    rather than six, at the cost of two thirds of the cohort.
    """
    traj = sibling("09_trajectories")
    grid_args = types.SimpleNamespace(
        subset=subset, start_hours=0.0, end_hours=float(end_hours),
        step_hours=float(step_hours), correction="coral", anchor="onset",
        force=False, min_tokens=8,
    )
    grid = traj.load_grid(paths, grid_args, logger)
    grid = traj.apply_correction(paths, grid, "coral", "onset", logger)
    present = sorted(grid["offset_hours"].unique().to_list())
    complete = [
        h for h in present
        if grid.filter(pl.col("offset_hours") == h)["site"].n_unique()
        == len(paths.sites)
    ]
    grid = grid.filter(pl.col("offset_hours").is_in(complete))
    logger.info(
        f"  grid points common to all {len(paths.sites)} sites: "
        f"{', '.join(f'{h:.0f}h' for h in complete)}"
    )
    return grid, complete


def embedding_at(grid, hour, columns):
    """one row per patient at a single grid hour"""
    return grid.filter(pl.col("offset_hours") == hour).select(
        ["hospitalization_id", "last_valid"] + columns
    ).unique(subset="hospitalization_id", keep="first")


def rebuild_handbooks(paths, args, logger):
    """
    40's handbooks and response groups, rebuilt with 40's own code and seed

    Re-implementing the split here would let the two scripts drift apart and
    the probe would then be describing groups that are not the ones reported.
    40 is imported instead.
    """
    hb = sibling("40_treatment_handbooks")
    labelled = pl.read_parquet(
        paths.patient_level_home / f"treatment_states_{args.subset}.parquet"
    )
    grouped = hb.build_strings(labelled, logger, args.cap_hours)
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    grouped = grouped.join(
        cohort.select(
            "hospitalization_id", "in_hospital_mortality",
            "age_at_admission", "lactate_max_in_window",
            # hours from shock onset to leaving the hospital. For a patient
            # who died this IS the hour of death, and it is what decides
            # whether a landmark falls inside their record or after it.
            hours_to_discharge=(
                pl.col("discharge_dttm") - pl.col("cs_onset_dttm")
            ).dt.total_seconds() / 3600.0,
        ),
        on="hospitalization_id", how="left",
    )
    try:
        from clifcs.concordance import attach_severity
        severity = attach_severity(paths, cohort, logger)
        grouped = grouped.join(
            severity.select("hospitalization_id", "sofa_non_cv"),
            on="hospitalization_id", how="left",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  non-CV SOFA unavailable: {exc}")

    treatments = grouped["treatment"].to_list()
    responses = grouped["response"].to_list()
    treatment_matrix = similarity_matrix(treatments, TREATMENT_SYMBOLS)
    response_matrix = similarity_matrix(responses, RESPONSE_ALPHABET)

    # distance and expcost come through apply_handbook_defaults for the same
    # reason k and the sample size do: 40 and 41 rebuild the same clustering,
    # and a measure that differed between them would produce a handbook table
    # and a divergence probe that disagree while both saying "handbook 0".
    assignment, medoids, fit = hb.discover_handbooks(
        treatments, TREATMENT_SYMBOLS, treatment_matrix, logger, args.sample,
        args.handbooks, args.seed, args.gap_open, args.gap_extend,
        distance=args.distance, expcost=args.expcost,
    )
    grouped = grouped.with_columns(
        handbook=pl.Series(assignment), handbook_fit=pl.Series(fit),
    ).with_columns(
        handbook=pl.when(pl.col("handbook_fit") >= args.min_fit)
        .then(pl.col("handbook")).otherwise(None)
    )

    parts = []
    for number in sorted(
        n for n in grouped["handbook"].unique().to_list() if n is not None
    ):
        part = grouped.filter(pl.col("handbook") == number)
        if part.height < args.min_group:
            continue
        result = hb.split_responses(
            [collapse(s) for s in part["response"].to_list()],
            RESPONSE_ALPHABET, response_matrix, logger, args.seed,
            args.gap_open, args.gap_extend,
        )
        if result is None:
            continue
        labels, response_medoids = result
        part = part.with_columns(response_group=pl.Series(labels))
        rates = [
            float(part.filter(
                (pl.col("response_group") == g)
                & pl.col("in_hospital_mortality").is_not_null()
            )["in_hospital_mortality"].mean() or 0.0)
            for g in (0, 1)
        ]
        worse = int(np.argmax(rates))
        parts.append(part.with_columns(
            # 1 marks the group with the higher mortality, so the sign of
            # every downstream comparison is fixed and readable
            worse_group=(pl.col("response_group") == worse).cast(pl.Int8),
        ))
        logger.info(
            f"  handbook {number}: n={part.height:>5,}  groups "
            f"{(labels == 0).sum():>4,}/{(labels == 1).sum():<4,}  "
            f"mortality {rates[0]:.1%} vs {rates[1]:.1%}"
        )
    if not parts:
        raise RuntimeError("no handbook survived the group-size floor")
    return pl.concat(parts, how="diagonal_relaxed"), response_matrix, medoids


def disjoint_pairs(part, rng):
    """
    pair patients across the response groups so that no patient is used twice

    Taking every cross-group pair would give far more pairs, and they would
    not be independent -- one patient would appear in hundreds of them, and a
    cross-validated AUC over such pairs is measuring memorisation of that
    patient. A matching costs sample size and buys independence, so the folds
    below can be ordinary stratified folds over pairs.
    """
    left = part.filter(pl.col("worse_group") == 0)
    right = part.filter(pl.col("worse_group") == 1)
    take = min(left.height, right.height)
    if take < SUPPRESS:
        return None
    a = rng.permutation(left.height)[:take].tolist()
    b = rng.permutation(right.height)[:take].tolist()
    return left[a], right[b]


def pre_divergence_rows(part, response_matrix, grid_hours, rng, args, logger):
    """
    for each matched pair, the last grid hour strictly before they parted

    THE ALIGNMENT RUNS ON THE RUN-LENGTH COLLAPSED COURSES and the endpoint is
    mapped back to the hour the first diverging run begins. An earlier version
    aligned the raw hourly strings on the grounds that a collapsed index is a
    step number rather than an hour -- true, but the mapping is one lookup, and
    the hourly alignment was answering the wrong question. Two courses with the
    same order of states and different durations in them disagree hour by hour
    from the moment the durations part, so the "divergence" was substantially a
    difference in length of stay. On the worked example in `collapse_runs` the
    hourly alignment ends in `ran_out` -- censored, no event -- while the
    collapsed one matches at identity 1.00 and parts exactly where the two
    courses genuinely go different ways.

    THE ALIGNMENT IS ALSO LOCAL, so the shared region need not begin at
    onset. A pair's `left_start`/`right_start` say where it did begin, which
    turns the anchoring assumption into a measurement. Pairs matching fewer
    than `--min-runs` runs are reported as having no shared course and do not
    contribute a divergence hour at all -- a local alignment always finds
    something, and two or three runs of coincidental agreement is not a shared
    course.

    What this costs, stated plainly: duration is now discarded inside the
    aligned region, so a patient who held a state for two hours and one who
    held it for twenty are the same course here. The divergence hour is no
    longer "how long they stayed together", and it is no longer measured from
    onset either.
    """
    paired = disjoint_pairs(part, rng)
    if paired is None:
        return None
    left, right = paired
    aligned = align_many(
        left["response"].to_list(), right["response"].to_list(),
        RESPONSE_ALPHABET, response_matrix, args.gap_open, args.gap_extend,
        min_tail=1, collapse=True, local=args.local_align,
        min_aligned=args.min_runs,
    )
    rows = []
    for k, summary in enumerate(aligned):
        if not summary["diverged"]:
            continue
        # the earlier of the two endpoints: a gap can carry one course
        # further along the alignment than the other, and the pair has
        # effectively parted at whichever leaves first
        hour = min(summary["left_hour"], summary["right_hour"])
        earlier = [h for h in grid_hours if h < hour]
        if not earlier:
            continue
        read_hour = float(max(earlier))
        # BOTH members must still be in hospital at the shared read hour,
        # otherwise the pair contrast is partly a contrast between a living
        # patient and a dead one's final tokens
        limits = [left["hours_to_discharge"][k], right["hours_to_discharge"][k]]
        if any(v is not None and v <= read_hour for v in limits):
            continue
        rows.append({
            "pair": k,
            "read_hour": read_hour,
            "divergence_hour": int(hour),
            "shared_from_hour": int(
                max(summary["left_start"], summary["right_start"])
            ),
            "shared_runs": int(summary["matched_columns"]),
            "reason": summary["reason"],
            "left_id": left["hospitalization_id"][k],
            "right_id": right["hospitalization_id"][k],
            "left_death": left["in_hospital_mortality"][k],
            "right_death": right["in_hospital_mortality"][k],
        })
    _ = logger
    return pl.DataFrame(rows, infer_schema_length=None) if rows else None


def paired_probe(pairs, grid, columns, seed):
    """
    within a discordant pair, does the embedding say which one dies

    The feature is the DIFFERENCE of the two embeddings at the shared read
    hour, and each pair enters twice with the sign flipped, so the model
    cannot learn an ordering convention instead of a signal. Both copies of a
    pair share a group id and therefore stay in the same fold.
    """
    discordant = pairs.filter(
        pl.col("left_death").is_not_null()
        & pl.col("right_death").is_not_null()
        & (pl.col("left_death") != pl.col("right_death"))
    )
    if discordant.height < SUPPRESS:
        return None, discordant.height

    features, labels, groups = [], [], []
    for row in discordant.iter_rows(named=True):
        at = grid.filter(pl.col("offset_hours") == row["read_hour"])
        a = at.filter(pl.col("hospitalization_id") == row["left_id"])
        b = at.filter(pl.col("hospitalization_id") == row["right_id"])
        if a.height == 0 or b.height == 0:
            continue
        va = a.select(columns).to_numpy().astype(np.float64)[0]
        vb = b.select(columns).to_numpy().astype(np.float64)[0]
        died_left = int(row["left_death"]) == 1
        features.append(va - vb)
        labels.append(1 if died_left else 0)
        groups.append(row["pair"])
        features.append(vb - va)
        labels.append(0 if died_left else 1)
        groups.append(row["pair"])
    if len(labels) < 2 * SUPPRESS:
        return None, discordant.height
    return (
        probe(np.vstack(features), np.array(labels), seed,
              groups=np.array(groups)),
        discordant.height,
    )


def patient_pre_divergence(pairs, part, grid, columns, seed, components):
    """
    every matched patient once, read at their own pair's pre-divergence hour

    The paired probe above is the stricter test but it spends most of the
    sample: it keeps only pairs whose two members ended differently, which in
    the smaller handbooks is a couple of dozen. This keeps BOTH members of
    every diverging pair and asks the ordinary question of each one -- does
    the embedding at the last hour before they parted predict their own
    outcome, and their own response group. Each patient still appears exactly
    once, because the matching used each patient once.
    """
    wanted = []
    for row in pairs.iter_rows(named=True):
        wanted.append((row["left_id"], row["read_hour"]))
        wanted.append((row["right_id"], row["read_hour"]))
    frame = pl.DataFrame(
        {
            "hospitalization_id": [i for i, _ in wanted],
            "read_hour": [h for _, h in wanted],
        }
    ).unique(subset="hospitalization_id", keep="first")

    collected = []
    for hour in sorted(frame["read_hour"].unique().to_list()):
        ids = frame.filter(pl.col("read_hour") == hour)
        collected.append(
            ids.join(
                grid.filter(pl.col("offset_hours") == hour).select(
                    ["hospitalization_id"] + columns
                ).unique(subset="hospitalization_id", keep="first"),
                on="hospitalization_id", how="inner",
            )
        )
    if not collected:
        return None, None, None, 0
    read = pl.concat(collected, how="diagonal_relaxed").join(
        part.select("hospitalization_id", "in_hospital_mortality",
                    "worse_group", "hours_to_discharge"),
        on="hospitalization_id", how="inner",
    ).filter(
        # the same leak as the landmark probes, in the same place: a read hour
        # past the patient's discharge is a read of their death
        pl.col("hours_to_discharge").is_null()
        | (pl.col("hours_to_discharge") > pl.col("read_hour"))
    )
    if read.height < 2 * SUPPRESS:
        return None, None, None, read.height

    outcome = read.filter(pl.col("in_hospital_mortality").is_not_null())
    death = probe(
        outcome.select(columns).to_numpy(),
        outcome["in_hospital_mortality"].to_numpy(),
        seed, components=components,
    ) if outcome.height >= 2 * SUPPRESS else None
    group = probe(
        read.select(columns).to_numpy(), read["worse_group"].to_numpy(),
        seed, components=components,
    )
    # THE READ HOUR IS NOT CONSTANT and that is a confound this probe has and
    # the paired one does not: pairs part at different times, so a patient
    # read at h24 was on record longer than one read at h6, and time on
    # record predicts dying by itself. The hour alone is therefore scored on
    # the same patients and the same folds, and the embedding row above means
    # nothing except by comparison with it.
    hour_only = probe(
        outcome.select("read_hour").to_numpy(),
        outcome["in_hospital_mortality"].to_numpy(),
        seed, components=1,
    ) if outcome.height >= 2 * SUPPRESS else None
    return death, group, hour_only, read.height


_CODE_STATUS_CACHE: dict = {}


def code_status_before(paths, patients, landmark, step_hours, names, logger):
    """patients carrying a DNR or DNR/DNI token at or before the landmark"""
    from clifcs.windows import window_token_counts

    if landmark in _CODE_STATUS_CACHE:
        return _CODE_STATUS_CACHE[landmark]

    wanted = {
        int(t) for t, name in (names or {}).items()
        if name in CODE_STATUS_TOKENS
    }
    if not wanted:
        _CODE_STATUS_CACHE[landmark] = set()
        return set()
    ends = list(np.arange(step_hours, landmark + 1e-9, step_hours))
    counts = window_token_counts(
        paths, ends, step_hours, set(patients), logger
    ).filter(
        (pl.col("window") <= landmark) & pl.col("token_id").is_in(list(wanted))
    )
    ids = set(counts["hospitalization_id"].to_list())
    logger.info(
        f"  {len(ids):,} patients have a code-status change by "
        f"{landmark:.0f}h and are censored"
    )
    _CODE_STATUS_CACHE[landmark] = ids
    return ids


def token_features(paths, patients, landmark, step_hours, logger,
                   names=None, families=None):
    """
    per-patient token rates and first-appearance hours, up to the landmark

    Two matrices, because "no token separates them" and "no token TIMING
    separates them" are different claims. The rate matrix counts occurrences
    per patient-hour observed; the timing matrix records the first window in
    which each token appeared, with absence left missing rather than coded as
    a large number, which would make absence and lateness the same feature.
    """
    from clifcs.windows import window_token_counts

    ends = list(np.arange(step_hours, landmark + 1e-9, step_hours))
    counts = window_token_counts(
        paths, ends, step_hours, set(patients), logger
    ).filter(pl.col("window") <= landmark)
    if counts.height == 0:
        return None, None, None

    banned = set()
    if families:
        banned |= {
            int(t) for t, family in families.items()
            if family in OUTCOME_FAMILIES
        }
    if names:
        banned |= {
            int(t) for t, name in names.items()
            if name in CODE_STATUS_TOKENS
        }
    dropped = counts.filter(pl.col("token_id").is_in(list(banned)))
    if dropped.height:
        logger.info(
            f"  dropped {dropped.height:,} cells of outcome and code-status "
            f"tokens ({dropped['hospitalization_id'].n_unique():,} patients "
            "touched) before any probe saw them"
        )
    counts = counts.filter(~pl.col("token_id").is_in(list(banned)))
    keep = (
        counts.group_by("token_id")
        .agg(patients=pl.col("hospitalization_id").n_unique())
        .filter(pl.col("patients") >= SUPPRESS)["token_id"]
        .to_list()
    )
    counts = counts.filter(pl.col("token_id").is_in(keep))
    rate = counts.group_by("hospitalization_id", "token_id").agg(
        n=pl.col("n").sum()
    ).pivot(on="token_id", index="hospitalization_id", values="n").fill_null(0)
    timing = counts.group_by("hospitalization_id", "token_id").agg(
        first=pl.col("window").min()
    ).pivot(on="token_id", index="hospitalization_id", values="first")
    logger.info(
        f"  {len(keep):,} tokens seen in at least {SUPPRESS} patients "
        f"within {landmark:.0f}h"
    )
    return rate, timing, keep


def screen(frame, label_column, columns, names, logger, top=8):
    """
    one token at a time against the outcome, with the multiplicity controlled

    A thousand tokens screened at 0.05 returns fifty hits from noise alone, so
    the p-values are Benjamini-Hochberg adjusted and the count of survivors is
    reported before any individual one is named.
    """
    from scipy.stats import mannwhitneyu

    y = frame[label_column].to_numpy()
    rows = []
    for column in columns:
        values = frame[column].to_numpy().astype(np.float64)
        present = ~np.isnan(values)
        a = values[present & (y == 0)]
        b = values[present & (y == 1)]
        if len(a) < SUPPRESS or len(b) < SUPPRESS:
            continue
        try:
            stat = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        spread = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        rows.append({
            "token_id": int(column),
            "token": names.get(int(column), str(column)),
            "n_group_0": int(len(a)), "n_group_1": int(len(b)),
            "median_0": float(np.median(a)), "median_1": float(np.median(b)),
            "smd": float((b.mean() - a.mean()) / spread) if spread > 0 else 0.0,
            "p": float(stat.pvalue),
        })
    if not rows:
        logger.info("      nothing testable at this cell floor")
        return pl.DataFrame()
    table = pl.DataFrame(rows).sort("p")
    m = table.height
    adjusted = np.minimum.accumulate(
        (table["p"].to_numpy() * m / np.arange(1, m + 1))[::-1]
    )[::-1]
    table = table.with_columns(q=pl.Series(np.clip(adjusted, 0, 1)))
    hits = table.filter(pl.col("q") < 0.05)
    logger.info(
        f"      {m:,} tokens tested, {hits.height:,} survive BH at q<0.05"
    )
    for row in hits.head(top).iter_rows(named=True):
        logger.info(
            f"        {row['token'][:38]:<38} "
            f"{row['median_0']:>7.2f} vs {row['median_1']:<7.2f} "
            f"SMD {row['smd']:+.2f}  q {row['q']:.2g}"
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--cap-hours", type=int, default=None)
    parser.add_argument("--handbooks", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--min-group", type=int, default=None)
    parser.add_argument("--min-fit", type=float, default=None)
    parser.add_argument("--distance", default=None,
                        choices=("align", "omspell"))
    parser.add_argument("--expcost", type=float, default=None)
    parser.add_argument("--landmarks", type=float, nargs="+",
                        default=[6.0, 12.0, 24.0])
    parser.add_argument("--step-hours", type=float, default=6.0)
    parser.add_argument("--end-hours", type=float, default=48.0)
    parser.add_argument("--keep-code-status", action="store_true",
                        help="do NOT censor patients whose code status "
                             "changed before the landmark; for measuring how "
                             "much those patients were contributing, not for "
                             "reporting")
    parser.add_argument("--components", type=int, default=32,
                        help="PCA components the probe is allowed; 1,024 raw "
                             "features against a few hundred patients needs a "
                             "hard ceiling, and it is applied inside the fold")
    parser.add_argument("--gap-open", type=float, default=None)
    parser.add_argument("--gap-extend", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--anchored", dest="local_align",
                        action="store_false", default=None,
                        help="anchor the divergence alignment at shock onset "
                             "instead of letting the shared region begin "
                             "anywhere")
    parser.add_argument("--min-runs", type=int, default=None,
                        help="aligned runs below which a pair is reported "
                             "as having no shared course")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    # k and the sample size are per cohort, so they cannot be argparse
    # defaults -- the parser would have committed to one number before
    # anything knew which cohort was running. See dynamics.yaml.
    apply_handbook_defaults(args)
    _div = handbook_defaults(subset=args.subset)
    for _flag, _key in (
        ("local_align", "divergence_local"),
        ("min_runs", "divergence_min_runs"),
    ):
        if getattr(args, _flag) is None:
            setattr(args, _flag, _div[_key])

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "41_divergence_probe", paths.log_home / "41_divergence_probe.log"
    )
    _cfg = OmegaConf.load(CONFIG_HOME / "dynamics.yaml")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    logger.info(
        f"subset {args.subset}; landmarks "
        f"{', '.join(f'{h:.0f}h' for h in args.landmarks)}; "
        f"{args.components} PCA components inside each fold"
    )

    logger.info("rebuilding 40's handbooks and response groups:")
    part_all, response_matrix, _ = rebuild_handbooks(paths, args, logger)

    logger.info("loading the trajectory grid:")
    grid, grid_hours = load_corrected_grid(
        paths, args.subset, args.step_hours, args.end_hours, logger
    )
    emb = sorted(c for c in grid.columns if c.startswith("emb_"))
    logger.info(f"  {len(emb):,} embedding dimensions after CORAL")

    from clifcs.tokens import vocabulary
    try:
        names, families = vocabulary(str(paths.tokenizer_yaml))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  vocabulary unavailable: {exc}")
        names, families = {}, {}

    handbooks = sorted(part_all["handbook"].unique().to_list())
    rows = []

    # ---- section 1: the landmark probes
    logger.info(
        "landmark probes -- embedding at a FIXED hour, label in the future:"
    )
    tabular = [c for c in ("age_at_admission", "sofa_non_cv",
                           "lactate_max_in_window") if c in part_all.columns]
    token_cache = {}
    for landmark in args.landmarks:
        if landmark not in grid_hours:
            logger.info(f"  {landmark:.0f}h: not a common grid point, skipped")
            continue
        at = embedding_at(grid, landmark, emb)
        rate, _timing, token_ids = token_cache.get(landmark, (None, None, None))
        if landmark not in token_cache:
            rate, _timing, token_ids = token_features(
                paths, part_all["hospitalization_id"].to_list(),
                landmark, args.step_hours, logger, names, families,
            )
            token_cache[landmark] = (rate, _timing, token_ids)
        # STILL IN HOSPITAL AT THE LANDMARK. A patient who died at hour 20
        # has their own death in the stream the hour-24 embedding is built
        # from, and is then predicted perfectly. Only 2% of the cohort by
        # h24, but they are 2% of guaranteed-correct answers, and the AUC
        # rising with every later landmark was mostly this.
        eligible = part_all.filter(
            pl.col("hours_to_discharge").is_null()
            | (pl.col("hours_to_discharge") > landmark)
        )
        if not args.keep_code_status:
            censored = code_status_before(
                paths, part_all["hospitalization_id"].to_list(), landmark,
                args.step_hours, names, logger,
            )
            if censored:
                eligible = eligible.filter(
                    ~pl.col("hospitalization_id").is_in(list(censored))
                )
        logger.info(
            f"  at {landmark:.0f}h: {eligible.height:,} of "
            f"{part_all.height:,} patients still in hospital and uncensored"
        )
        for number in handbooks:
            here = eligible.filter(pl.col("handbook") == number).join(
                at, on="hospitalization_id", how="inner"
            )
            if here.height < args.min_group // 2:
                continue
            logger.info(f"    handbook {number} (n={here.height:,} on record):")
            x = here.select(emb).to_numpy()

            outcome = here.filter(pl.col("in_hospital_mortality").is_not_null())
            result_group = probe(
                x, here["worse_group"].to_numpy(), args.seed,
                components=args.components,
            )
            say(logger, "embedding -> response group", result_group)
            result_death = probe(
                outcome.select(emb).to_numpy(),
                outcome["in_hospital_mortality"].to_numpy(),
                args.seed, components=args.components,
            )
            say(logger, "embedding -> mortality", result_death)

            result_tab = None
            if tabular:
                tab = outcome.select(tabular).fill_null(strategy="mean")
                result_tab = probe(
                    tab.to_numpy(), outcome["in_hospital_mortality"].to_numpy(),
                    args.seed, components=len(tabular),
                )
            say(logger, "baseline covariates -> mortality", result_tab)

            result_token = None
            if rate is not None:
                joined = outcome.select(
                    "hospitalization_id", "in_hospital_mortality"
                ).join(rate, on="hospitalization_id", how="inner")
                if joined.height >= args.min_group // 2:
                    result_token = probe(
                        joined.select(
                            [c for c in joined.columns
                             if c not in ("hospitalization_id",
                                          "in_hospital_mortality")]
                        ).fill_null(0).to_numpy(),
                        joined["in_hospital_mortality"].to_numpy(),
                        args.seed, components=args.components,
                    )
            say(logger, "token counts -> mortality", result_token)

            for label, result in (
                ("embedding->group", result_group),
                ("embedding->mortality", result_death),
                ("covariates->mortality", result_tab),
                ("tokens->mortality", result_token),
            ):
                if result is None:
                    continue
                rows.append({
                    "section": "landmark", "handbook": number,
                    "hour": float(landmark), "probe": label, **result,
                })

    # ---- section 2: the last grid point before divergence
    logger.info(
        "pre-divergence probe -- the last grid point before the pair parted:"
    )
    logger.info(
        "  NOTE the read hour is chosen with hindsight; compare against the "
        "landmark rows above, not against chance alone"
    )
    pair_rows = []
    for number in handbooks:
        part = part_all.filter(pl.col("handbook") == number)
        pairs = pre_divergence_rows(
            part, response_matrix, grid_hours, rng, args, logger
        )
        if pairs is None:
            logger.info(f"  handbook {number}: no diverging matched pairs")
            continue
        result, discordant = paired_probe(pairs, grid, emb, args.seed)
        death, group, hour_only, read_n = patient_pre_divergence(
            pairs, part, grid, emb, args.seed, args.components
        )
        logger.info(
            f"  handbook {number}: {pairs.height:,} diverging pairs, "
            f"{discordant:,} with discordant outcomes; median divergence "
            f"h{pairs['divergence_hour'].median():.0f}, median read "
            f"h{pairs['read_hour'].median():.0f}; {read_n:,} patients read"
        )
        say(logger, "read hour alone -> own mortality", hour_only, indent=4)
        say(logger, "embedding -> own mortality", death, indent=4)
        say(logger, "embedding -> own response group", group, indent=4)
        say(logger, "embedding difference -> who dies", result, indent=4)
        for label, item in (
            ("read_hour->mortality", hour_only),
            ("embedding->mortality", death),
            ("embedding->group", group),
            ("embedding_difference->who_dies", result),
        ):
            if item is None:
                continue
            rows.append({
                "section": "pre_divergence", "handbook": number,
                "hour": float(pairs["read_hour"].median()),
                "probe": label, **item,
            })
        pair_rows.append(pairs.with_columns(handbook=pl.lit(number)))

    # ---- section 3: one token at a time, rate and timing
    logger.info("token screen within handbook (rate, then first appearance):")
    usable_landmarks = [h for h in args.landmarks if h in grid_hours]
    landmark = max(usable_landmarks) if usable_landmarks else None
    rate, timing, token_ids = token_cache.get(
        landmark, (None, None, None)
    )
    screens = []
    if rate is None or token_ids is None:
        logger.info("  no windowed token counts available")
    else:
        screen_pool = part_all.filter(
            pl.col("hours_to_discharge").is_null()
            | (pl.col("hours_to_discharge") > landmark)
        )
        if not args.keep_code_status:
            still = code_status_before(
                paths, part_all["hospitalization_id"].to_list(), landmark,
                args.step_hours, names, logger,
            )
            if still:
                screen_pool = screen_pool.filter(
                    ~pl.col("hospitalization_id").is_in(list(still))
                )
        for number in handbooks:
            here = screen_pool.filter(
                (pl.col("handbook") == number)
                & pl.col("in_hospital_mortality").is_not_null()
            ).select("hospitalization_id", "in_hospital_mortality",
                     "worse_group")
            columns = [str(t) for t in token_ids]
            for what, matrix in (("rate", rate), ("first_hour", timing)):
                joined = here.join(
                    matrix, on="hospitalization_id", how="inner"
                )
                usable = [c for c in columns if c in joined.columns]
                if joined.height < args.min_group // 2 or not usable:
                    continue
                logger.info(
                    f"    handbook {number}, token {what}, by mortality:"
                )
                table = screen(
                    joined, "in_hospital_mortality", usable, names, logger
                )
                if table.height:
                    screens.append(table.with_columns(
                        handbook=pl.lit(number), measure=pl.lit(what),
                        contrast=pl.lit("mortality"),
                    ))
                logger.info(
                    f"    handbook {number}, token {what}, by response group:"
                )
                table = screen(joined, "worse_group", usable, names, logger)
                if table.height:
                    screens.append(table.with_columns(
                        handbook=pl.lit(number), measure=pl.lit(what),
                        contrast=pl.lit("response_group"),
                    ))

    # ---- outputs
    if rows:
        save_table(
            pl.DataFrame(rows, infer_schema_length=None),
            tables / f"divergence_probe_{args.subset}.csv",
        )
    if pair_rows:
        save_table(
            pl.concat(pair_rows, how="diagonal_relaxed").drop(
                "left_id", "right_id"
            ),
            tables / f"divergence_probe_pairs_{args.subset}.csv",
        )
    if screens:
        save_table(
            pl.concat(screens, how="diagonal_relaxed"),
            tables / f"divergence_probe_tokens_{args.subset}.csv",
        )
    logger.info("41 done")


if __name__ == "__main__":
    main()

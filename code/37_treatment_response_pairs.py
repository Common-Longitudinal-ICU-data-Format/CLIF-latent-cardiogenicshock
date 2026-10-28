#!/usr/bin/env python3

"""
37_treatment_response_pairs.py -- same treatment, different response, and
whether those patients sit elsewhere in the latent space

Every patient now carries two strings over the same hours: what was done to
them (the treatment cell, one hex digit per hour) and what happened (the
response state, one letter per hour). Two questions follow, and they are not
the same question.

  WHO IS TREATED DIFFERENTLY. Treatment intensity tracks severity almost by
  definition -- mortality runs 24% in the low-norepinephrine cell and 77% in
  high-plus-vasopressin-plus-other at an essentially identical median MAP. A
  subphenotype built on treatment is largely a subphenotype of how sick the
  patient was, which is worth knowing and is not news.

  WHO RESPONDS DIFFERENTLY TO THE SAME TREATMENT. This is the question severity
  cannot answer. Matching on the treatment string holds the plan fixed, so a
  divergence in the response string is a divergence in the patient.

    python code/37_treatment_response_pairs.py --subset cardiogenic

MATCHING IS ON THE COLLAPSED TREATMENT STRING, not the raw one. Raw strings are
near-unique -- 98% of response courses occur once and treatment is no better --
so an exact match finds nobody. Collapsing runs keeps the ORDER of escalation
and de-escalation and discards how long each step lasted, which is the level at
which two patients can be said to have had the same plan.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.bpe import edit_distance
from clifcs.config import CONFIG_HOME
from clifcs.regions import STATE_SYMBOL
from clifcs.treatment import add_treatment_symbol, collapse, treatment_label

MIN_GROUP = 25


def build_strings(labelled, logger):
    """one treatment string and one response string per patient"""
    frame = add_treatment_symbol(labelled).with_columns(
        response_symbol=pl.col("region").replace_strict(
            STATE_SYMBOL, default="?", return_dtype=pl.String
        )
    ).sort(["hospitalization_id", "offset_hours"])
    grouped = frame.group_by("hospitalization_id", maintain_order=True).agg(
        treatment=pl.col("treatment_symbol").str.join(""),
        response=pl.col("response_symbol").str.join(""),
        hours=pl.len(),
    )
    grouped = grouped.with_columns(
        treatment_collapsed=pl.col("treatment").map_elements(
            collapse, return_dtype=pl.String
        ),
        response_collapsed=pl.col("response").map_elements(
            collapse, return_dtype=pl.String
        ),
    )
    logger.info(
        f"  {grouped.height:,} patients; treatment strings "
        f"{grouped['treatment'].n_unique():,} distinct raw, "
        f"{grouped['treatment_collapsed'].n_unique():,} collapsed"
    )
    return grouped


def matched_groups(grouped, logger, min_group=MIN_GROUP):
    """
    patients sharing a collapsed treatment string, with their response spread

    `response_spread` is the mean pairwise normalised edit distance between the
    group's response strings. A group whose members were treated identically
    and whose responses are far apart is the object this script exists to find.
    """
    counts = grouped["treatment_collapsed"].value_counts(sort=True)
    rows = []
    for course, n in zip(
        counts["treatment_collapsed"].to_list(), counts["count"].to_list()
    ):
        if n < min_group:
            continue
        part = grouped.filter(pl.col("treatment_collapsed") == course)
        responses = part["response_collapsed"].to_list()
        # all pairs when the group is small, a capped sample when it is not
        pairs = []
        rng = np.random.default_rng(42)
        if len(responses) <= 60:
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    pairs.append((responses[i], responses[j]))
        else:
            for _ in range(1500):
                i, j = rng.choice(len(responses), size=2, replace=False)
                pairs.append((responses[i], responses[j]))
        spread = float(np.mean([
            edit_distance(list(a), list(b)) / max(len(a), len(b), 1)
            for a, b in pairs
        ])) if pairs else float("nan")
        rows.append({
            "treatment_collapsed": course,
            "patients": n,
            "treatment_steps": len(course),
            "meaning": " -> ".join(treatment_label(c) for c in course[:4]),
            "response_spread": spread,
            "distinct_responses": len(set(responses)),
            "share_unique": len(set(responses)) / max(n, 1),
        })
    table = pl.DataFrame(rows).sort("patients", descending=True)
    logger.info(
        f"  {table.height} collapsed treatment courses shared by at least "
        f"{min_group} patients, covering "
        f"{table['patients'].sum():,} of {grouped.height:,} patients "
        f"({table['patients'].sum() / grouped.height:.0%})"
    )
    return table


def response_split_within_treatment(grouped, courses, labels, paths, subset,
                                    logger, seed=42, top_courses=6):
    """
    inside one treatment course, split patients by response and probe the latent space

    For each shared treatment course the members are split in two on their
    response strings by k-medoids on edit distance -- two groups who had the
    same plan and different courses. Then three questions of each split:
    does it separate mortality, does it separate severity, and can the GEM
    embedding tell the two groups apart.

    THE EMBEDDING TEST IS THE POINT. If patients given the same plan who
    respond differently also sit in different regions of the latent space,
    something about them was legible before the divergence. If the probe lands
    at chance, the divergence is either unmodelled physiology or noise -- and
    either way the embedding does not see it.
    """
    import types

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "traj", pathlib.Path(__file__).resolve().parent / "09_trajectories.py"
    )
    traj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(traj)
    grid_args = types.SimpleNamespace(
        subset=subset, start_hours=-24.0, end_hours=48.0,
        step_hours=6.0, correction="coral", anchor="onset", force=False,
        min_tokens=8,
    )
    grid = traj.load_grid(paths, grid_args, logger)
    grid = traj.apply_correction(paths, grid, "coral", "onset", logger)
    emb = sorted(c for c in grid.columns if c.startswith("emb_"))
    # one vector per patient: the mean of their corrected trajectory, which is
    # where they sat on average rather than at any one hour
    per_patient = grid.group_by("hospitalization_id").agg(
        [pl.col(c).mean() for c in emb]
    )

    rng = np.random.default_rng(seed)
    rows = []
    for course in courses.head(top_courses)["treatment_collapsed"].to_list():
        part = grouped.filter(pl.col("treatment_collapsed") == course).join(
            labels, on="hospitalization_id", how="inner"
        ).join(per_patient, on="hospitalization_id", how="inner")
        if part.height < 60:
            continue
        responses = part["response_collapsed"].to_list()
        # k-medoids with k=2 on edit distance, seeded by the two most distant
        # responses, then a few reassignment passes
        n = len(responses)
        distance = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = edit_distance(list(responses[i]), list(responses[j])) / max(
                    len(responses[i]), len(responses[j]), 1
                )
                distance[i, j] = distance[j, i] = d
        medoids = list(np.unravel_index(np.argmax(distance), distance.shape))
        for _ in range(12):
            assign = np.argmin(distance[:, medoids], axis=1)
            new = []
            for k in (0, 1):
                members = np.where(assign == k)[0]
                if len(members) == 0:
                    new.append(medoids[k])
                    continue
                inner = distance[np.ix_(members, members)].sum(axis=1)
                new.append(int(members[int(np.argmin(inner))]))
            if new == medoids:
                break
            medoids = new
        assign = np.argmin(distance[:, medoids], axis=1)
        if min((assign == 0).sum(), (assign == 1).sum()) < 25:
            continue

        side = part.with_columns(side=pl.Series(assign))
        a = side.filter(pl.col("side") == 0)
        b = side.filter(pl.col("side") == 1)
        x = side.select(emb).to_numpy().astype(np.float64)
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
        y = assign.astype(np.int8)
        scores = []
        for train, test in StratifiedKFold(
            4, shuffle=True, random_state=seed
        ).split(x, y):
            model = LogisticRegression(max_iter=2000, C=0.05).fit(
                x[train], y[train]
            )
            scores.append(roc_auc_score(
                y[test], model.predict_proba(x[test])[:, 1]
            ))
        auc = float(np.mean(scores))
        row = {
            "treatment_collapsed": course,
            "meaning": " -> ".join(treatment_label(c) for c in course[:3]),
            "patients": side.height,
            "group_a": a.height, "group_b": b.height,
            "medoid_a": responses[medoids[0]][:24],
            "medoid_b": responses[medoids[1]][:24],
            "mortality_a": float(a["in_hospital_mortality"].mean()),
            "mortality_b": float(b["in_hospital_mortality"].mean()),
            "mortality_gap": float(
                b["in_hospital_mortality"].mean()
                - a["in_hospital_mortality"].mean()
            ),
            "embedding_auc": auc,
        }
        if "sofa_non_cv" in side.columns:
            row["sofa_a"] = float(a["sofa_non_cv"].median())
            row["sofa_b"] = float(b["sofa_non_cv"].median())
        rows.append(row)
        logger.info(
            f"    {course[:10]:<10} n={side.height:>4,}  "
            f"{a.height:>4}/{b.height:<4}  mortality "
            f"{row['mortality_a']:>5.1%} vs {row['mortality_b']:>5.1%} "
            f"({row['mortality_gap']:+.1%})  embedding AUC {auc:.3f}"
        )
        logger.info(
            f"        medoid A {responses[medoids[0]][:28]}"
        )
        logger.info(
            f"        medoid B {responses[medoids[1]][:28]}"
        )
        _ = rng
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "37_treatment_response_pairs",
        paths.log_home / "37_treatment_response_pairs.log",
    )
    _cfg = OmegaConf.load(CONFIG_HOME / "dynamics.yaml")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    labelled = pl.read_parquet(
        paths.patient_level_home / f"treatment_states_{args.subset}.parquet"
    )
    logger.info(f"{labelled.height:,} patient-hours")
    grouped = build_strings(labelled, logger)

    # ---- question 1: who is treated differently
    logger.info("treatment intensity against outcome (the severity question):")
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet").select(
        "hospitalization_id", "in_hospital_mortality"
    )
    from clifcs.concordance import attach_severity
    try:
        severity = attach_severity(
            paths, pl.read_parquet(paths.cohort_home / "cohort.parquet"), logger
        )
        cohort = cohort.join(
            severity.select("hospitalization_id", "sofa_non_cv"),
            on="hospitalization_id", how="left",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  severity unavailable: {exc}")

    # the highest treatment cell a patient ever reached, as a one-number
    # summary of how hard they were treated
    peak = (
        add_treatment_symbol(labelled)
        .group_by("hospitalization_id")
        .agg(peak_treatment=pl.col("treatment_symbol").max())
    )
    intensity = peak.join(cohort, on="hospitalization_id", how="inner")
    rows = []
    for symbol in sorted(intensity["peak_treatment"].unique().to_list()):
        part = intensity.filter(
            (pl.col("peak_treatment") == symbol)
            & pl.col("in_hospital_mortality").is_not_null()
        )
        if part.height < 50:
            continue
        rows.append({
            "peak_treatment": symbol,
            "meaning": treatment_label(symbol),
            "patients": part.height,
            "mortality": float(part["in_hospital_mortality"].mean()),
            "median_sofa_non_cv": float(part["sofa_non_cv"].median())
            if "sofa_non_cv" in part.columns else None,
        })
        logger.info(
            f"    {treatment_label(symbol):<26} n={part.height:>5,}  "
            f"mortality {rows[-1]['mortality']:>5.1%}  "
            f"non-CV SOFA {rows[-1]['median_sofa_non_cv']}"
        )
    pl.DataFrame(rows, infer_schema_length=None).write_csv(
        tables / f"treatment_intensity_outcome_{args.subset}.csv"
    )

    # ---- question 2: among the similarly treated, who responds differently
    logger.info("shared treatment courses, and how far their responses spread:")
    courses = matched_groups(grouped, logger)
    if courses.is_empty():
        logger.error("no treatment course is shared widely enough")
        return
    courses.write_csv(tables / f"shared_treatment_courses_{args.subset}.csv")
    for row in courses.head(10).iter_rows(named=True):
        logger.info(
            f"    {row['treatment_collapsed'][:12]:<12} "
            f"{row['patients']:>5,} patients  response spread "
            f"{row['response_spread']:.3f}  "
            f"{row['share_unique']:.0%} unique   {row['meaning'][:44]}"
        )

    logger.info(
        "splitting each shared course by response, then probing the embedding:"
    )
    splits = response_split_within_treatment(
        grouped, courses, cohort, paths, args.subset, logger, seed=args.seed
    )
    if not splits.is_empty():
        splits.write_csv(
            tables / f"treatment_response_splits_{args.subset}.csv"
        )
    grouped.write_parquet(
        paths.patient_level_home
        / f"treatment_response_strings_{args.subset}.parquet"
    )
    logger.info("37 done")


if __name__ == "__main__":
    main()

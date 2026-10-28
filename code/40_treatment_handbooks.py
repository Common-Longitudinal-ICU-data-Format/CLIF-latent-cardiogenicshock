#!/usr/bin/env python3

"""
40_treatment_handbooks.py -- groups of patients on essentially the same plan,
and how differently they respond to it

    python code/40_treatment_handbooks.py --subset cardiogenic

WHY THE HANDBOOKS ARE ALIGNED AND NOT MATCHED EXACTLY. Matching on the
identical collapsed treatment string was tried and it finds the wrong thing.
The collapsed courses are dominated by low-dose flicker -- `0` (never on
norepinephrine, 2,114 patients), `10`, `010`, `1010`, `101010` -- which is the
deadband of a single low-dose drug going on and off, and only 21 courses reach
25 patients at all. It is the same failure the BPE words had: the shared
patterns are short, ubiquitous and carry no information, while the courses that
matter are long, varied and therefore unique. So a handbook here is a CLUSTER
of treatment courses under alignment distance, which lets two patients count as
similarly managed without having been managed identically.

THE TWO LAYERS GET THEIR OWN SUBSTITUTION MATRICES, both measured the same way
-- Jensen-Shannon distance between forward and backward neighbour
distributions -- but over different alphabets: 16 treatment cells and 10
response states. Neither is hand-set.

WHAT HOLDING THE HANDBOOK FIXED DOES AND DOES NOT BUY. Within a handbook every
patient received comparable support in comparable order, so a divergence in
response is a divergence in the patient rather than in the plan. It is still
not a randomised comparison: clinicians arrive at a handbook by reading the
patient, so the handbook is selected on the very thing the response measures.
Confounding by indication runs through every number below and no amount of
stratification removes it.

SECTION 3 IS THE ONE TO READ SCEPTICALLY. It asks whether the worse-responding
group contains a subset that does better and is managed differently, which is
four comparisons deep into a dataset that was not designed for it. Every split
is reported with its n, and the treatment differences it finds are differences
in TIMING AND DURATION -- the things the collapse discarded -- because those
are the only management differences that can exist once the collapsed course is
held fixed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

import numpy as np
import polars as pl
from omegaconf import OmegaConf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import apply_handbook_defaults, get_logger, load_paths
from clifcs.align import (
    align_scores, distance_matrix, similarity_matrix,
)
from clifcs.config import CONFIG_HOME
from clifcs.figures import save_table
from clifcs.traminer import omspell_matrix, omspell_to_refs, pam
from clifcs.regions import (
    HOURLY_REGION_ORDER, STATE_SYMBOL, SYMBOL_STATE, trim_trailing_off,
)
from clifcs.treatment import (
    TREATMENT_SYMBOLS, add_treatment_symbol, collapse, treatment_label,
)

RESPONSE_ALPHABET = "".join(
    STATE_SYMBOL[state] for state in HOURLY_REGION_ORDER
)
MATCH = 3.0


def build_strings(labelled, logger, cap_hours):
    """treatment and response strings per patient, over the same hours"""
    frame = add_treatment_symbol(labelled).with_columns(
        response_symbol=pl.col("region").replace_strict(
            STATE_SYMBOL, default="?", return_dtype=pl.String
        )
    ).sort(["hospitalization_id", "offset_hours"])
    grouped = frame.group_by("hospitalization_id", maintain_order=True).agg(
        treatment=pl.col("treatment_symbol").str.join(""),
        response=pl.col("response_symbol").str.join(""),
        peak_ne=pl.col("dose_norepinephrine").max(),
        hours_on_pressors=(pl.col("dose_norepinephrine") > 0).sum(),
        site=pl.col("site").first(),
    )
    # the response is trimmed of its trailing off-pressor run because those
    # hours say the same thing repeatedly; the TREATMENT string is cut to the
    # same hours so the two layers describe the same window
    responses, treatments = [], []
    for response, treatment in zip(
        grouped["response"].to_list(), grouped["treatment"].to_list()
    ):
        kept = trim_trailing_off(response, keep=1)[:cap_hours]
        responses.append(kept)
        treatments.append(treatment[:len(kept)])
    grouped = grouped.with_columns(
        response=pl.Series(responses),
        treatment=pl.Series(treatments),
        hours=pl.Series([len(s) for s in responses]),
    ).filter(pl.col("hours") >= 6)
    logger.info(
        f"  {grouped.height:,} patients with at least 6 hours; median "
        f"{int(grouped['hours'].median())}h"
    )
    return grouped


def describe_treatment(sequence):
    """a collapsed treatment course written out as bands"""
    short = collapse(sequence)
    return " -> ".join(treatment_label(c) for c in short[:4]) + (
        " -> ..." if len(short) > 4 else ""
    )


def _discover_omspell(sequences, subset, take, k, logger, expcost):
    """
    the OMspell path through discovery: same shape, different measure

    Clustering and assignment both go through TraMineR so the medoids are
    defined and used under one metric. The substitution costs are re-estimated
    on the combined set at assignment time, which is why the references are
    passed through with the sequences rather than compared across two seqdef
    objects.

    `fit` is returned as 1 - normalised distance so that larger still means
    "resembles its medoid more", matching the aligner's convention. The NUMBER
    does not match the aligner's: OMspell normalised distance is a metric in
    [0, 1] and the aligner's normalised score is not, so --min-fit must be set
    per measure. config/dynamics.yaml carries both.
    """
    square = omspell_matrix(subset, logger=logger, expcost=expcost,
                            tag="handbook_sample")
    # PAM, not the average-linkage cut the aligner path uses. On OMspell
    # dissimilarities average linkage chains into 511/56/29/2/1/1, four of six
    # handbooks being single patients; see clifcs.traminer.pam.
    _, positions = pam(square.astype(np.float64), k, logger=logger)
    medoids = [subset[i] for i in positions]
    logger.info(
        f"  {len(medoids)} handbook medoids from {take:,} sampled courses "
        "(OMspell, PAM)"
    )

    to_medoid = omspell_to_refs(sequences, medoids, logger=logger,
                               expcost=expcost, tag="handbook_assign")
    assignment = np.argmin(to_medoid, axis=1).astype(np.int32)
    fit = 1.0 - to_medoid[np.arange(len(sequences)), assignment]
    logger.info(
        f"    fit to assigned medoid: median {float(np.median(fit)):.2f}, "
        f"10th centile {float(np.percentile(fit, 10)):.2f}, "
        f"90th {float(np.percentile(fit, 90)):.2f}"
    )
    return assignment, medoids, fit.astype(np.float32)


def discover_handbooks(sequences, alphabet, matrix, logger, sample, k, seed,
                       gap_open, gap_extend, distance="align", expcost=0.5):
    """
    cluster a sample of treatment courses, then assign everyone to a medoid

    Clustering all 9,378 courses would need 44 million alignments. A sample is
    clustered instead, its MEDOIDS are kept as the definition of each handbook,
    and every patient is then assigned by aligning against the k medoids --
    which is k alignments each rather than n. The medoid is used rather than a
    centroid because there is no mean of a sequence.

    `distance` picks how "managed alike" is measured. "align" is this project's
    semi-global aligner, kept as the baseline the change is judged against.
    "omspell" is TraMineR's spell-based optimal matching, which came first of
    fourteen measures on a test set of the phenotypes we expect perturbed the
    way real data is messy, where the aligner came last -- see clifcs.traminer
    for the numbers. The two put `fit` on DIFFERENT scales, so each carries its
    own floor in config; they are not interchangeable thresholds.
    """
    rng = np.random.default_rng(seed)
    take = min(sample, len(sequences))
    chosen = rng.choice(len(sequences), size=take, replace=False)
    subset = [sequences[i] for i in chosen]
    if distance == "omspell":
        return _discover_omspell(sequences, subset, take, k, logger, expcost)
    condensed = distance_matrix(
        subset, alphabet, matrix, gap_open, gap_extend, MATCH, logger
    )
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=k, criterion="maxclust")

    square = np.zeros((take, take), dtype=np.float32)
    index = 0
    for i in range(take):
        for j in range(i + 1, take):
            square[i, j] = square[j, i] = condensed[index]
            index += 1
    medoids = []
    for cluster in range(1, k + 1):
        members = np.where(labels == cluster)[0]
        if len(members) == 0:
            continue
        within = square[np.ix_(members, members)].sum(axis=1)
        medoids.append(subset[members[int(np.argmin(within))]])
    logger.info(f"  {len(medoids)} handbook medoids from {take:,} sampled courses")

    # assign every patient to the medoid its course aligns best against
    best = np.full(len(sequences), -np.inf, dtype=np.float32)
    assignment = np.zeros(len(sequences), dtype=np.int32)
    for number, medoid in enumerate(medoids):
        scores = align_scores(
            sequences, [medoid] * len(sequences), alphabet, matrix,
            gap_open, gap_extend,
        )
        ceiling = MATCH * np.array([
            min(len(s), len(medoid)) for s in sequences
        ], dtype=np.float32)
        normalised = np.where(ceiling > 0, scores / ceiling, 0.0)
        better = normalised > best
        best = np.where(better, normalised, best)
        assignment = np.where(better, number, assignment)
    return assignment, medoids, best


def split_responses(responses, alphabet, matrix, logger, seed, gap_open,
                    gap_extend, cap=400):
    """
    two groups of response course within one handbook, by alignment distance

    Capped at `cap` members because the distance is quadratic; above that a
    random subset is clustered and the rest assigned to the nearer medoid, the
    same compromise the handbooks themselves use.
    """
    rng = np.random.default_rng(seed)
    if len(responses) > cap:
        chosen = rng.choice(len(responses), size=cap, replace=False)
    else:
        chosen = np.arange(len(responses))
    subset = [responses[i] for i in chosen]
    condensed = distance_matrix(
        subset, alphabet, matrix, gap_open, gap_extend, MATCH
    )
    labels = fcluster(linkage(condensed, method="average"), t=2,
                      criterion="maxclust")
    take = len(subset)
    square = np.zeros((take, take), dtype=np.float32)
    index = 0
    for i in range(take):
        for j in range(i + 1, take):
            square[i, j] = square[j, i] = condensed[index]
            index += 1
    medoids = []
    for cluster in (1, 2):
        members = np.where(labels == cluster)[0]
        if len(members) == 0:
            return None
        within = square[np.ix_(members, members)].sum(axis=1)
        medoids.append(subset[members[int(np.argmin(within))]])

    assignment = np.zeros(len(responses), dtype=np.int32)
    best = np.full(len(responses), -np.inf, dtype=np.float32)
    for number, medoid in enumerate(medoids):
        scores = align_scores(
            responses, [medoid] * len(responses), alphabet, matrix,
            gap_open, gap_extend,
        )
        ceiling = MATCH * np.array([
            min(len(s), len(medoid)) for s in responses
        ], dtype=np.float32)
        normalised = np.where(ceiling > 0, scores / ceiling, 0.0)
        better = normalised > best
        best = np.where(better, normalised, best)
        assignment = np.where(better, number, assignment)
    _ = logger
    return assignment, medoids


def composition(sequence):
    """state shares of one response course"""
    counts = Counter(sequence)
    total = max(len(sequence), 1)
    return {
        SYMBOL_STATE.get(s, s): counts.get(s, 0) / total
        for s in RESPONSE_ALPHABET
    }


def timing_features(treatment):
    """
    the management detail a collapsed course throws away

    Once two patients are inside the same handbook their plan is the same at
    the level of WHICH bands in WHICH order. What can still differ is when each
    step happened and how long it lasted, so those are the only management
    differences available to find, and they are what this returns.
    """
    bands = [TREATMENT_SYMBOLS.find(c) % 4 if c in TREATMENT_SYMBOLS else 0
             for c in treatment]
    escalations = [
        i for i in range(1, len(bands)) if bands[i] > bands[i - 1]
    ]
    weans = [i for i in range(1, len(bands)) if bands[i] < bands[i - 1]]
    peak = max(bands) if bands else 0
    return {
        "hours": len(treatment),
        "hours_at_peak_band": sum(1 for b in bands if b == peak and peak > 0),
        "time_to_first_escalation": escalations[0] if escalations else None,
        "time_to_first_wean": weans[0] if weans else None,
        "escalations": len(escalations),
        "weans": len(weans),
        "mean_band": float(np.mean(bands)) if bands else 0.0,
        "hours_on_vasopressin": sum(
            1 for c in treatment
            if c in TREATMENT_SYMBOLS and TREATMENT_SYMBOLS.find(c) & 4
        ),
        "hours_on_other": sum(
            1 for c in treatment
            if c in TREATMENT_SYMBOLS and TREATMENT_SYMBOLS.find(c) & 8
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--cap-hours", type=int, default=None)
    parser.add_argument("--handbooks", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--min-group", type=int, default=None)
    parser.add_argument("--min-fit", type=float, default=None,
                        help="normalised alignment to the handbook medoid "
                             "below which a patient is left unassigned; "
                             "without it the loosest handbooks hold "
                             "members that barely resemble their own "
                             "medoid (median fit ran as low as 0.11)")
    parser.add_argument("--distance", default=None,
                        choices=("align", "omspell"),
                        help="how 'managed alike' is measured. omspell is "
                             "TraMineR spell-based optimal matching, which "
                             "beat the in-house aligner on messy-data "
                             "robustness; align is kept as the baseline. "
                             "Note that --min-fit means a different thing "
                             "under each, so the two floors are separate "
                             "entries in dynamics.yaml")
    parser.add_argument("--expcost", type=float, default=None,
                        help="OMspell cost of a spell's duration. 0 ignores "
                             "duration entirely (pure order, like a run-length "
                             "collapse); large values make duration dominate. "
                             "Only read when --distance omspell")
    parser.add_argument("--landmark-hours", type=int, default=24,
                        help="question 3 is restricted to patients still "
                             "on record at this hour, because dying early "
                             "truncates both the record and the "
                             "opportunity to escalate")
    parser.add_argument("--gap-open", type=float, default=None)
    parser.add_argument("--gap-extend", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    # k and the sample size are per cohort, so they cannot be argparse
    # defaults -- the parser would have committed to one number before
    # anything knew which cohort was running. See dynamics.yaml.
    apply_handbook_defaults(args)

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "40_treatment_handbooks", paths.log_home / "40_treatment_handbooks.log"
    )
    _cfg = OmegaConf.load(CONFIG_HOME / "dynamics.yaml")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"subset {args.subset}; cap {args.cap_hours}h; {args.handbooks} "
        f"handbooks from a {args.sample:,} sample"
    )

    labelled = pl.read_parquet(
        paths.patient_level_home / f"treatment_states_{args.subset}.parquet"
    )
    grouped = build_strings(labelled, logger, args.cap_hours)
    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet").select(
        "hospitalization_id", "in_hospital_mortality", "age_at_admission",
        "lactate_max_in_window", "icu_los_days",
    )
    grouped = grouped.join(cohort, on="hospitalization_id", how="left")
    try:
        from clifcs.concordance import attach_severity
        severity = attach_severity(
            paths, pl.read_parquet(paths.cohort_home / "cohort.parquet"), logger
        )
        grouped = grouped.join(
            severity.select("hospitalization_id", "sofa_non_cv"),
            on="hospitalization_id", how="left",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  non-CV SOFA unavailable: {exc}")

    treatments = grouped["treatment"].to_list()
    responses = grouped["response"].to_list()

    logger.info("treatment-layer substitution matrix:")
    treatment_matrix = similarity_matrix(treatments, TREATMENT_SYMBOLS, logger)
    logger.info("response-layer substitution matrix:")
    response_matrix = similarity_matrix(responses, RESPONSE_ALPHABET, logger)

    # ---- section 1: what the handbooks are
    logger.info(
        f"discovering handbooks from the treatment layer ({args.distance}):"
    )
    assignment, medoids, fit = discover_handbooks(
        treatments, TREATMENT_SYMBOLS, treatment_matrix, logger, args.sample,
        args.handbooks, args.seed, args.gap_open, args.gap_extend,
        distance=args.distance, expcost=args.expcost,
    )
    grouped = grouped.with_columns(
        handbook=pl.Series(assignment),
        handbook_fit=pl.Series(fit),
    )
    before = grouped.height
    grouped = grouped.with_columns(
        handbook=pl.when(pl.col("handbook_fit") >= args.min_fit)
        .then(pl.col("handbook")).otherwise(None)
    )
    # the per-patient assignment, written so that a change of measure can be
    # judged on the cohort rather than on the catalogue summary: whether
    # handbooks separate outcome is a patient-level question and needs
    # patient-level labels. `distance` and `expcost` are carried in the table
    # because two runs of this script now differ in more than the seed.
    save_table(
        grouped.select(
            "hospitalization_id", "handbook", "handbook_fit",
            "in_hospital_mortality",
        ).with_columns(
            distance=pl.lit(args.distance),
            expcost=pl.lit(float(args.expcost)),
            min_fit=pl.lit(float(args.min_fit)),
        ),
        tables / f"handbook_assignment_{args.subset}.csv",
    )
    kept = grouped.filter(pl.col("handbook").is_not_null()).height
    logger.info(
        f"  fit floor {args.min_fit}: {kept:,} of {before:,} patients "
        f"({kept / before:.0%}) resemble their handbook medoid closely enough "
        "to be called similarly managed; the rest are left unassigned"
    )
    rows = []
    for number, medoid in enumerate(medoids):
        part = grouped.filter(pl.col("handbook") == number)
        if part.height == 0:
            continue
        alive = part.filter(pl.col("in_hospital_mortality").is_not_null())
        row = {
            "handbook": number,
            "patients": part.height,
            "medoid_course": collapse(medoid)[:16],
            "meaning": describe_treatment(medoid),
            "median_hours": float(part["hours"].median()),
            "median_peak_ne": float(part["peak_ne"].median()),
            "median_pressor_hours": float(part["hours_on_pressors"].median()),
            "mortality": float(alive["in_hospital_mortality"].mean())
            if alive.height else None,
            "median_fit": float(part["handbook_fit"].median()),
        }
        if "sofa_non_cv" in part.columns:
            row["median_sofa_non_cv"] = float(part["sofa_non_cv"].median())
        rows.append(row)
        logger.info(
            f"  handbook {number}: n={part.height:>5,}  "
            f"peak NE {row['median_peak_ne']:>5.2f}  "
            f"pressor-h {row['median_pressor_hours']:>4.0f}  "
            f"mortality {row['mortality']:>6.1%}  "
            f"non-CV SOFA {row.get('median_sofa_non_cv')}"
        )
        logger.info(f"      {row['meaning']}")
    catalogue = pl.DataFrame(rows, infer_schema_length=None)
    save_table(
        catalogue,
        tables / f"treatment_handbooks_{args.subset}.csv",
    )

    # ---- section 2: how differently patients respond inside one handbook
    logger.info(
        "response groups within each handbook (same plan, different course):"
    )
    split_rows = []
    assignments = {}
    # the fit floor leaves unassigned patients as null, which must be
    # dropped before sorting rather than compared against an int
    for number in sorted(
        n for n in grouped["handbook"].unique().to_list() if n is not None
    ):
        part = grouped.filter(pl.col("handbook") == number)
        if part.height < args.min_group:
            continue
        outcome = part["in_hospital_mortality"].to_list()
        if sum(1 for v in outcome if v is not None) < args.min_group // 2:
            continue
        # CLUSTER THE COLLAPSED COURSE, not the raw one. Clustering raw
        # response strings separated short off-pressor courses from long
        # on-pressor ones -- the medoids were `ooooooooooooDo` against
        # `FHRFSRdNwwHFHHNNNdSHo` and every covariate that separated the groups
        # was a duration, at standardised differences of 0.49 to 0.76. Run-
        # collapsing keeps the ORDER of what happened and discards how long
        # each state lasted, which is the level at which two patients can be
        # said to have had the same response rather than the same length of
        # stay.
        result = split_responses(
            [collapse(s) for s in part["response"].to_list()],
            RESPONSE_ALPHABET, response_matrix,
            logger, args.seed, args.gap_open, args.gap_extend,
        )
        if result is None:
            continue
        labels, response_medoids = result
        part = part.with_columns(response_group=pl.Series(labels))
        assignments[number] = part
        sides = []
        for group in (0, 1):
            side = part.filter(
                (pl.col("response_group") == group)
                & pl.col("in_hospital_mortality").is_not_null()
            )
            sides.append(side)
        if min(s.height for s in sides) < 20:
            logger.info(f"  handbook {number}: split too lopsided to report")
            continue
        rates = [float(s["in_hospital_mortality"].mean()) for s in sides]
        worse = int(np.argmax(rates))
        row = {
            "handbook": number,
            "patients": part.height,
            "group_a": sides[0].height, "group_b": sides[1].height,
            "mortality_a": rates[0], "mortality_b": rates[1],
            "mortality_gap": rates[worse] - rates[1 - worse],
            "medoid_a": response_medoids[0][:28],
            "medoid_b": response_medoids[1][:28],
        }
        for label, side in (("a", sides[0]), ("b", sides[1])):
            if "sofa_non_cv" in side.columns:
                row[f"sofa_{label}"] = float(side["sofa_non_cv"].median())
            row[f"hours_{label}"] = float(side["hours"].median())
            row[f"peak_ne_{label}"] = float(side["peak_ne"].median())
        split_rows.append(row)
        logger.info(
            f"  handbook {number}: {sides[0].height:>4,} vs "
            f"{sides[1].height:<4,}  mortality {rates[0]:>6.1%} vs "
            f"{rates[1]:<6.1%}  gap {row['mortality_gap']:+.1%}"
        )
        logger.info(f"      course A  {response_medoids[0][:34]}")
        logger.info(f"      course B  {response_medoids[1][:34]}")
    save_table(
        pl.DataFrame(split_rows, infer_schema_length=None),
        tables / f"handbook_response_groups_{args.subset}.csv",
    )

    # ---- what differentiates the response groups
    logger.info("what separates the response groups (question 2):")
    candidates = [
        c for c in ("sofa_non_cv", "age_at_admission", "lactate_max_in_window",
                    "peak_ne", "hours_on_pressors", "hours")
        if c in grouped.columns
    ]
    difference_rows = []
    for number, part in assignments.items():
        if "response_group" not in part.columns:
            continue
        a = part.filter(pl.col("response_group") == 0)
        b = part.filter(pl.col("response_group") == 1)
        if min(a.height, b.height) < 20:
            continue
        for column in candidates:
            left = a[column].drop_nulls().cast(pl.Float64).to_numpy()
            right = b[column].drop_nulls().cast(pl.Float64).to_numpy()
            if len(left) < 10 or len(right) < 10:
                continue
            pooled = np.sqrt((left.var() + right.var()) / 2.0)
            # standardised difference, which is comparable across variables on
            # different scales in a way a raw difference is not
            smd = (
                float((right.mean() - left.mean()) / pooled)
                if pooled > 0 else 0.0
            )
            difference_rows.append({
                "handbook": number, "variable": column,
                "median_a": float(np.median(left)),
                "median_b": float(np.median(right)),
                "standardised_difference": smd,
            })
        # the response composition itself, which is what the split was made on
        for group, side in ((0, a), (1, b)):
            shares = {}
            for sequence in side["response"].to_list():
                for state, value in composition(sequence).items():
                    shares.setdefault(state, []).append(value)
            for state, values in shares.items():
                difference_rows.append({
                    "handbook": number, "variable": f"share {state}",
                    "median_a": float(np.median(values)) if group == 0 else None,
                    "median_b": float(np.median(values)) if group == 1 else None,
                    "standardised_difference": None,
                })
    differences = pl.DataFrame(difference_rows, infer_schema_length=None)
    save_table(
        differences,
        tables / f"handbook_group_differences_{args.subset}.csv",
    )
    strong = differences.filter(
        pl.col("standardised_difference").is_not_null()
        & (pl.col("standardised_difference").abs() > 0.2)
    ).sort(pl.col("standardised_difference").abs(), descending=True)
    for row in strong.head(12).iter_rows(named=True):
        logger.info(
            f"  handbook {row['handbook']}  {row['variable']:<22} "
            f"{row['median_a']:>8.2f} vs {row['median_b']:<8.2f} "
            f"SMD {row['standardised_difference']:+.2f}"
        )
    if strong.height == 0:
        logger.info(
            "  nothing reaches a standardised difference of 0.2 -- the "
            "response groups are not separated by any measured covariate"
        )

    # ---- section 3: inside the worse group, a better subset?
    logger.info(
        "inside the worse-responding group, is there a better subset, and is "
        "it managed differently (question 3)?"
    )
    third_rows = []
    for number, part in assignments.items():
        if "response_group" not in part.columns:
            continue
        rates = []
        for group in (0, 1):
            side = part.filter(
                (pl.col("response_group") == group)
                & pl.col("in_hospital_mortality").is_not_null()
            )
            rates.append(
                float(side["in_hospital_mortality"].mean())
                if side.height else 0.0
            )
        worse = int(np.argmax(rates))
        pool = part.filter(
            (pl.col("response_group") == worse)
            & (pl.col("hours") >= args.landmark_hours)
        )
        if pool.height < args.min_group:
            logger.info(
                f"  handbook {number}: worse group has {pool.height:,} "
                f"patients past the {args.landmark_hours}h landmark, too few"
            )
            continue
        result = split_responses(
            [collapse(s) for s in pool["response"].to_list()],
            RESPONSE_ALPHABET, response_matrix,
            logger, args.seed + 1, args.gap_open, args.gap_extend,
        )
        if result is None:
            continue
        labels, _medoids = result
        pool = pool.with_columns(subgroup=pl.Series(labels))
        subs = [
            pool.filter(
                (pl.col("subgroup") == g)
                & pl.col("in_hospital_mortality").is_not_null()
            )
            for g in (0, 1)
        ]
        if min(s.height for s in subs) < 25:
            continue
        sub_rates = [float(s["in_hospital_mortality"].mean()) for s in subs]
        better = int(np.argmin(sub_rates))
        gap = sub_rates[1 - better] - sub_rates[better]
        logger.info(
            f"  handbook {number}, worse group (n={pool.height:,}, mortality "
            f"{rates[worse]:.1%}) splits {subs[better].height:,} at "
            f"{sub_rates[better]:.1%} vs {subs[1 - better].height:,} at "
            f"{sub_rates[1 - better]:.1%}  (gap {gap:+.1%})"
        )
        # are they managed differently? only timing and duration can differ
        for label, side in (("better", subs[better]),
                            ("worse", subs[1 - better])):
            features = [timing_features(t) for t in side["treatment"].to_list()]
            summary = {
                key: float(np.median([
                    f[key] for f in features if f[key] is not None
                ])) if any(f[key] is not None for f in features) else None
                for key in features[0]
            }
            third_rows.append({
                "handbook": number, "subgroup": label,
                "patients": side.height,
                "mortality": float(side["in_hospital_mortality"].mean()),
                **summary,
            })
        a, b = third_rows[-2], third_rows[-1]
        for key in ("time_to_first_escalation", "hours_at_peak_band",
                    "escalations", "weans", "hours_on_vasopressin",
                    "mean_band"):
            if a.get(key) is None or b.get(key) is None:
                continue
            if abs(a[key] - b[key]) > 1e-9:
                logger.info(
                    f"      {key:<26} better {a[key]:>7.2f}   "
                    f"worse {b[key]:>7.2f}"
                )
    save_table(
        pl.DataFrame(third_rows, infer_schema_length=None),
        tables / f"handbook_worse_group_subsets_{args.subset}.csv",
    )
    logger.info("40 done")


if __name__ == "__main__":
    main()

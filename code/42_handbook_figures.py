#!/usr/bin/env python3

"""
42_handbook_figures.py -- the handbooks, drawn

40 found the handbooks, 41 probed whether the divergence inside them is
legible before it happens. Neither drew anything. This produces the figures
and the report-ready tables for both:

  CATALOGUE        who is in each handbook -- age, severity, lactate, support
                   intensity, mortality -- and how its two response groups
                   split. One row per handbook, one block per response group.

  FLOW             one streamplot per handbook on the shared projection, each
                   in its own colour over the grey cloud, so the question is
                   whether handbooks MOVE differently rather than whether they
                   sit in different places.

  DIVERGENCE       for one handbook, matched pairs rendered as aligned
                   strings: the treatment course above, the two response
                   courses with the match line between them, and the hour the
                   alignment stopped paying. This is what a divergence looks
                   like, as opposed to what its summary statistics say.

  OUTCOME          one handbook's patients on the projection, died against
                   survived, to see whether outcome localises at all.

  TOKENS           within that handbook, the tokens whose rate differs most
                   between died and survived, as violins.

THE PROJECTION IS NOT METRIC and three of these figures are drawn on it.
PaCMAP preserves neighbourhoods, not distances, so "these two groups overlap
here" is a claim about neighbourhoods and "this group is twice as far away"
is not a claim at all. The flow figure is the one to trust least and the
token violins -- which touch no projection -- the most.

    python code/42_handbook_figures.py --subset cardiogenic
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import types

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import (
    apply_handbook_defaults, get_logger, handbook_defaults, load_paths,
)
from clifcs.align import align_many, collapse_runs, similarity_matrix
from clifcs.figures import save_figure, save_table, style
from clifcs.palette import CONTEXT, INK_SECONDARY, mortality_colors
from clifcs.regions import HOURLY_REGION_ORDER, STATE_SYMBOL
from clifcs.treatment import TREATMENT_SYMBOLS, treatment_label

RESPONSE_ALPHABET = "".join(
    STATE_SYMBOL[state] for state in HOURLY_REGION_ORDER
)
SUPPRESS = 11


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(
        name.split("_")[0] + "_mod",
        pathlib.Path(__file__).resolve().parent / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def catalogue(part_all, medoids, logger):
    """
    one row per handbook: who is in it, what the plan is, how it ended

    The response-group columns are the point of the table rather than an
    appendix to it. A handbook's mortality is a property of the patients the
    clinicians put in it; the GAP between its two response groups is the part
    that is not explained by the plan, and the two numbers mean opposite
    things when read alone.
    """
    rows = []
    for number in sorted(part_all["handbook"].unique().to_list()):
        part = part_all.filter(pl.col("handbook") == number)
        known = part.filter(pl.col("in_hospital_mortality").is_not_null())
        if part.height < SUPPRESS:
            continue
        groups = []
        for worse in (0, 1):
            side = known.filter(pl.col("worse_group") == worse)
            groups.append(
                (side.height, float(side["in_hospital_mortality"].mean())
                 if side.height else None)
            )
        plan = medoids[number] if number < len(medoids) else ""
        row = {
            "handbook": number,
            "patients": part.height,
            "plan": " to ".join(
                treatment_label(c) for c in _collapse_unique(plan)[:4]
            ),
            "median_age": float(part["age_at_admission"].median()),
            "median_sofa_non_cv": (
                float(part["sofa_non_cv"].median())
                if "sofa_non_cv" in part.columns else None
            ),
            "median_lactate": float(part["lactate_max_in_window"].median()),
            "median_peak_ne": float(part["peak_ne"].median()),
            "median_pressor_hours": float(part["hours_on_pressors"].median()),
            "median_hours": float(part["hours"].median()),
            "mortality": float(known["in_hospital_mortality"].mean())
            if known.height else None,
            "better_group_n": groups[0][0],
            "better_group_mortality": groups[0][1],
            "worse_group_n": groups[1][0],
            "worse_group_mortality": groups[1][1],
        }
        if groups[0][1] is not None and groups[1][1] is not None:
            row["response_gap"] = groups[1][1] - groups[0][1]
        rows.append(row)
        logger.info(
            f"  handbook {number}: n={part.height:>5,}  "
            f"age {row['median_age']:>4.0f}  SOFA {row['median_sofa_non_cv']}  "
            f"lactate {row['median_lactate']:>4.1f}  "
            f"mortality {row['mortality']:>6.1%}  "
            f"groups {groups[0][0]:>4,}/{groups[1][0]:<4,}  "
            f"gap {row.get('response_gap', float('nan')):+.1%}"
        )
        logger.info(f"      {row['plan']}")
    return pl.DataFrame(rows, infer_schema_length=None)


def _collapse_unique(sequence):
    out = []
    for character in sequence:
        if not out or out[-1] != character:
            out.append(character)
    return "".join(out)


def layered_render(path, left_row, right_row, summary, limit=48, tail=14):
    """
    one pair as five lines: the agreed prefix, then where each course WENT

    An earlier version rendered only the aligned columns, which for this data
    meant printing five identical hours and stopping -- the alignment ends AT
    the divergence, so the aligned region is by construction the part where
    nothing happened. The figure showed no divergence at all.

    Each line is therefore the aligned region, a marker, and then the next
    `tail` steps of that course taken from where the alignment stopped. The
    part after the marker is NOT aligned and the two sides are not in register
    there: that is the point, and it is what parting company looks like.

    ONE CHARACTER IS ONE RUN, NOT ONE HOUR. The alignment is now run-length
    collapsed, so `SSS` in the raw record is a single `S` here and the
    treatment character beside it is the treatment at the hour that run began.
    The clock hour of the divergence is reported separately in the row; it
    cannot be counted off these strings and should not be.
    """
    left_states, right_states = [], []
    left_plan, right_plan = [], []
    for a, b, _score in path[:limit]:
        left_states.append(
            left_row["response"][a] if 0 <= a < len(left_row["response"]) else "-"
        )
        right_states.append(
            right_row["response"][b] if 0 <= b < len(right_row["response"]) else "-"
        )
        left_plan.append(
            left_row["treatment"][a]
            if 0 <= a < len(left_row["treatment"]) else "-"
        )
        right_plan.append(
            right_row["treatment"][b]
            if 0 <= b < len(right_row["treatment"]) else "-"
        )
    match = []
    for a, b, score in path[:limit]:
        if a >= 0 and b >= 0:
            same = (
                0 <= a < len(left_row["response"])
                and 0 <= b < len(right_row["response"])
                and left_row["response"][a] == right_row["response"][b]
            )
            match.append("|" if same else ("+" if score and score > 0 else " "))
        else:
            match.append(" ")

    # the endpoint in the SAME units the strings are in: run indices when the
    # alignment was collapsed, hours when it was not
    i = int(summary.get("left_run", summary["left_hour"]))
    j = int(summary.get("right_run", summary["right_hour"]))
    return {
        "treatment_better": "".join(left_plan) + " > "
        + left_row["treatment"][i:i + tail],
        "response_better": "".join(left_states) + " > "
        + left_row["response"][i:i + tail],
        "match": "".join(match) + " ^ ",
        "response_worse": "".join(right_states) + " > "
        + right_row["response"][j:j + tail],
        "treatment_worse": "".join(right_plan) + " > "
        + right_row["treatment"][j:j + tail],
        **_plan_agreement("".join(left_plan), "".join(right_plan)),
    }


def _plan_agreement(left_plan: str, right_plan: str) -> dict:
    """
    how closely two plans match over the region their responses were aligned on

    FOUR MEASURES BECAUSE THE BINARY ONE IS MISLEADING. Asking whether the two
    hourly strings are identical answers almost always no -- 1 of 30 rendered
    pairs -- but a handbook is defined on the ORDER of what was done, not its
    durations, so two patients can be on the same plan all week and differ at
    every hour because one was weaned a few hours sooner. That test condemns
    the handbooks for the thing they were built to allow.

    `band_gap_mean` is the one to read: the mean absolute difference in
    norepinephrine band, per aligned hour, on a 0-to-3 scale. It measures how
    far apart the support actually was rather than whether two strings match.
    """
    bands = [
        (TREATMENT_SYMBOLS.find(a) % 4, TREATMENT_SYMBOLS.find(b) % 4)
        for a, b in zip(left_plan, right_plan)
        if a in TREATMENT_SYMBOLS and b in TREATMENT_SYMBOLS
    ]
    gaps = [abs(a - b) for a, b in bands]
    return {
        "plan_hourly_identical": left_plan == right_plan,
        "plan_course_identical": (
            _collapse_unique(left_plan.replace("-", ""))
            == _collapse_unique(right_plan.replace("-", ""))
        ),
        "plan_same_peak_band": (
            max((a for a, _ in bands), default=-1)
            == max((b for _, b in bands), default=-1)
        ),
        "band_gap_mean": float(np.mean(gaps)) if gaps else None,
    }


def divergence_examples(part, response_matrix, args, logger, wanted=6):
    """
    matched cross-group pairs, rendered, nearest the handbook's median hour

    Chosen near the MEDIAN divergence hour rather than the earliest or the
    most extreme. An example picked for being striking is an argument about
    nothing; one picked for being typical is a description.
    """
    rng = np.random.default_rng(args.seed)
    left = part.filter(pl.col("worse_group") == 0)
    right = part.filter(pl.col("worse_group") == 1)
    take = min(left.height, right.height)
    if take < SUPPRESS:
        return pl.DataFrame(), None
    left = left[rng.permutation(left.height)[:take].tolist()]
    right = right[rng.permutation(right.height)[:take].tolist()]

    aligned, paths = align_many(
        left["response"].to_list(), right["response"].to_list(),
        RESPONSE_ALPHABET, response_matrix, args.gap_open, args.gap_extend,
        keep_paths=True, min_tail=1, collapse=True, local=args.local_align,
        min_aligned=args.min_runs,
    )
    by_index = {k: (lc, rc, path) for k, lc, rc, path in paths}

    def _runs(frame, k):
        """the pair's two layers in run coordinates, one character per run

        The alignment is on collapsed response courses, so an alignment
        position indexes a RUN. The treatment line has to be put in the same
        coordinates or the two layers in a rendered block describe different
        things: it is read at the hour each response run began, which is the
        treatment the patient was on when that run started.
        """
        collapsed, starts, _lengths = collapse_runs(frame["response"][k])
        plan = frame["treatment"][k]
        return {
            "response": collapsed,
            "treatment": "".join(
                plan[h] if h < len(plan) else "?" for h in starts
            ),
        }
    events = [
        (k, s) for k, s in enumerate(aligned) if s["diverged"]
    ]
    if not events:
        logger.info("    no diverging pairs to render")
        return pl.DataFrame(), None
    hours = np.array([min(s["left_hour"], s["right_hour"]) for _, s in events])
    middle = float(np.median(hours))
    order = np.argsort(np.abs(hours - middle))
    logger.info(
        f"    {len(events):,} diverging pairs, median divergence hour "
        f"{middle:.0f}; rendering the {min(wanted, len(events))} nearest it"
    )

    # THE AUDIT RUNS OVER EVERY DIVERGING PAIR, not the handful rendered.
    # Six examples per handbook is an illustration; whether the handbooks are
    # really holding the plan fixed is a population question and has to be
    # answered on the population or it is an anecdote with a decimal point.
    audit = []
    for k, _summary in events:
        left_code, right_code, path = by_index[k]
        left_row, right_row = _runs(left, k), _runs(right, k)
        plan_left, plan_right = [], []
        for a, b, _score in path:
            plan_left.append(
                left_row["treatment"][a]
                if 0 <= a < len(left_row["treatment"]) else "-"
            )
            plan_right.append(
                right_row["treatment"][b]
                if 0 <= b < len(right_row["treatment"]) else "-"
            )
        audit.append(_plan_agreement("".join(plan_left), "".join(plan_right)))
    gaps = [r["band_gap_mean"] for r in audit if r["band_gap_mean"] is not None]
    audit_row = {
        "handbook": int(part["handbook"][0]),
        "diverging_pairs": len(events),
        "band_gap_mean": float(np.mean(gaps)) if gaps else None,
        "same_peak_band": float(
            np.mean([r["plan_same_peak_band"] for r in audit])
        ) if audit else None,
        "course_identical": float(
            np.mean([r["plan_course_identical"] for r in audit])
        ) if audit else None,
        "hourly_identical": float(
            np.mean([r["plan_hourly_identical"] for r in audit])
        ) if audit else None,
    }
    logger.info(
        f"    plan audit over all {len(events):,} pairs: "
        f"{audit_row['band_gap_mean']:.2f} bands apart per aligned run, "
        f"same peak band {audit_row['same_peak_band']:.0%}, "
        f"identical course {audit_row['course_identical']:.0%}"
    )

    rows = []
    for position in order[:wanted]:
        k, summary = events[position]
        left_code, right_code, path = by_index[k]
        drawn = layered_render(
            path, _runs(left, k), _runs(right, k), summary,
        )
        rows.append({
            "handbook": int(part["handbook"][0]),
            "shared_from_hour": int(max(summary["left_start"],
                                        summary["right_start"])),
            "divergence_hour": int(min(summary["left_hour"],
                                       summary["right_hour"])),
            "shared_runs": int(summary["matched_columns"]),
            "reason": summary["reason"],
            "identity": summary["identity"],
            "died_better": left["in_hospital_mortality"][k],
            "died_worse": right["in_hospital_mortality"][k],
            **drawn,
        })
    return pl.DataFrame(rows, infer_schema_length=None), audit_row


#: Tails dropped from each coordinate before the limb axis is fitted. Both
#: projections carry small outlying islands -- cardiogenic has one near x=16
#: holding about 1% of points -- and they dominate a principal axis taken over
#: everything: the untrimmed cardiogenic axis is (+0.99, +0.15), essentially
#: horizontal, which scores travel along the visible diagonal at whatever its x
#: component happens to be. Trimming to the central box recovers the diagonal
#: in both cohorts, (+0.91, +0.41) and (+0.70, +0.71).
LIMB_TRIM = (2.5, 97.5)


def limb_axis(grid, x: str = "pacmap_1", y: str = "pacmap_2"):
    """the axis the two visible limbs lie along, and the cloud's centre"""
    pos = np.column_stack([grid[x].to_numpy(), grid[y].to_numpy()])
    lo_x, hi_x = np.percentile(pos[:, 0], LIMB_TRIM)
    lo_y, hi_y = np.percentile(pos[:, 1], LIMB_TRIM)
    keep = (
        (pos[:, 0] >= lo_x) & (pos[:, 0] <= hi_x)
        & (pos[:, 1] >= lo_y) & (pos[:, 1] <= hi_y)
    )
    core = pos[keep]
    centre = core.mean(axis=0)
    axis = np.linalg.svd(core - centre, full_matrices=False)[2][0]
    if axis[0] < 0:                   # name the upper-right end "upper"
        axis = -axis
    return axis, centre


def travel(grid, axis, centre, x: str = "pacmap_1", y: str = "pacmap_2"):
    """per patient: where they entered along the limb axis, and how far they went"""
    per = grid.sort("hospitalization_id", "offset_hours").group_by(
        "hospitalization_id", maintain_order=True
    ).agg(
        start_x=pl.col(x).first(), start_y=pl.col(y).first(),
        end_x=pl.col(x).last(), end_y=pl.col(y).last(),
        points=pl.len(),
    ).filter(pl.col("points") >= 2)
    start = np.column_stack([
        per["start_x"].to_numpy(), per["start_y"].to_numpy()
    ])
    end = np.column_stack([per["end_x"].to_numpy(), per["end_y"].to_numpy()])
    return per.with_columns(
        start_along=pl.Series((start - centre) @ axis),
        travel_along=pl.Series((end - start) @ axis),
    )


def direction_split(grid, part_all, number: int, out, logger, traj,
                    axis, centre):
    """
    within one handbook, split patients by which way they crossed the space

    The handbook flow figure asks whether the handbooks move differently from
    one another. This is the follow-up that question raises: WITHIN a handbook
    the arrows are not all parallel -- there is an upper-right limb whose
    streamlines run one way and a lower-left limb whose streamlines run the
    other -- so who is in each, and is the difference anything a clinician
    would recognise?

    THE SPLIT IS ON WHERE A PATIENT ENTERS, not on which way they then go, and
    the direction is left to the arrows. Splitting on the sign of travel was
    the first version and it needed a signed axis, which a principal axis does
    not have: orienting it by the population's mean displacement looked
    principled until the mean turned out to have a coherence of 0.07 -- the
    displacements very nearly cancel, there is no prevailing drift to point
    along, and the sign was therefore arbitrary. Entry limb needs no sign
    convention: a patient is in the upper-right limb or the lower-left one,
    and each limb's own field then shows which way that limb travels.

    SO THE FIGURE IS ANSWERING "who is where", AND THE ARROWS ANSWER "which way
    from there". A patient in the upper limb can still travel down and out of
    it; nothing here claims the limbs are absorbing.

    WHAT IT CANNOT BE. Direction of travel is measured over the patient's whole
    record, which for a patient who dies at hour 14 is fourteen hours and for a
    survivor is forty-eight. Any comparison between the groups is therefore
    partly a comparison of how long they were observed, and the hours row in
    the table is there to be read first rather than last.
    """
    here = grid.filter(pl.col("handbook") == number)
    if here["hospitalization_id"].n_unique() < 2 * SUPPRESS:
        logger.info(f"  handbook {number}: too few patients to split")
        return None
    per = travel(here, axis, centre)
    extra = ["sofa_non_cv"] if "sofa_non_cv" in part_all.columns else []
    per = per.with_columns(
        direction=pl.when(pl.col("start_along") > 0)
        .then(pl.lit("upper-right limb"))
        .otherwise(pl.lit("lower-left limb"))
    ).join(
        part_all.select(
            "hospitalization_id", "in_hospital_mortality", "worse_group",
            "peak_ne", "hours_on_pressors", "hours", "age_at_admission",
            "lactate_max_in_window", *extra,
        ),
        on="hospitalization_id", how="left",
    )
    logger.info(
        f"  handbook {number}: "
        + ", ".join(
            f"{d} {int((per['direction'] == d).sum()):,}"
            for d in ("upper-right limb", "lower-left limb")
        )
    )

    numeric = [
        c for c in ("start_along", "travel_along", "hours", "peak_ne",
                    "hours_on_pressors", "lactate_max_in_window",
                    "sofa_non_cv", "age_at_admission")
        if c in per.columns
    ]
    rows = []
    for label in ("upper-right limb", "lower-left limb"):
        side = per.filter(pl.col("direction") == label)
        if side.height < SUPPRESS:
            continue
        known = side.filter(pl.col("in_hospital_mortality").is_not_null())
        row = {
            "handbook": number, "direction": label, "patients": side.height,
            "mortality": float(known["in_hospital_mortality"].mean())
            if known.height else float("nan"),
            "share_worse_response_group": float(
                side["worse_group"].drop_nulls().mean() or 0.0
            ),
        }
        for column in numeric:
            value = side[column].median()
            row[f"median_{column}"] = (
                float(value) if value is not None else float("nan")
            )
        rows.append(row)
    if len(rows) == 2:
        asc = per.filter(pl.col("direction") == "upper-right limb")
        desc = per.filter(pl.col("direction") == "lower-left limb")
        for column in numeric:
            a, b = asc[column], desc[column]
            spread = np.sqrt(((a.var() or 0.0) + (b.var() or 0.0)) / 2.0)
            smd = (
                ((a.mean() or 0.0) - (b.mean() or 0.0)) / spread
                if spread > 0 else float("nan")
            )
            rows[0][f"smd_{column}"] = float(smd)
            logger.info(
                f"      {column:<26} upper {a.median():>8.2f}  "
                f"lower {b.median():>8.2f}   SMD {smd:+.2f}"
            )
    if not rows:
        return None

    palette = {"upper-right limb": "#2a78d6",
               "lower-left limb": "#eb6834"}
    figure, (ax, legend_ax) = plt.subplots(
        1, 2, figsize=(7.4, 5.2),
        gridspec_kw={"width_ratios": [4.4, 1.3], "wspace": 0.04},
    )
    legend_ax.axis("off")
    ax.scatter(grid["pacmap_1"].to_numpy(), grid["pacmap_2"].to_numpy(),
               s=1.0, c=CONTEXT, linewidths=0, alpha=0.55, rasterized=True,
               zorder=1)
    pooled = (
        *np.percentile(grid["pacmap_1"].to_numpy(), [0.5, 99.5]),
        *np.percentile(grid["pacmap_2"].to_numpy(), [0.5, 99.5]),
    )
    handles = []
    for label in ("upper-right limb", "lower-left limb"):
        ids = per.filter(pl.col("direction") == label)["hospitalization_id"]
        if len(ids) < SUPPRESS:
            continue
        side = here.filter(pl.col("hospitalization_id").is_in(ids))
        # Small and faint, because the two directions are thoroughly
        # interleaved -- that IS the finding -- and at full density whichever
        # colour is drawn second simply covers the other.
        ax.scatter(side["pacmap_1"].to_numpy(), side["pacmap_2"].to_numpy(),
                   s=1.2, c=palette[label], linewidths=0, alpha=0.25,
                   rasterized=True, zorder=2)
        cx, cy, vx, vy, counts, n_vectors = traj.displacement_field(
            side, "pacmap_1", "pacmap_2", extent=pooled
        )
        mask = ~traj.flow_support(counts, n_vectors, fraction=0.0)
        with np.errstate(invalid="ignore"):
            ax.streamplot(
                cx, cy, np.where(mask, np.nan, vx), np.where(mask, np.nan, vy),
                color=palette[label], linewidth=0.8, arrowsize=0.8,
                density=1.3, maxlength=1.1, zorder=3,
            )
        known = per.filter(
            (pl.col("direction") == label)
            & pl.col("in_hospital_mortality").is_not_null()
        )
        rate = (
            float(known["in_hospital_mortality"].mean())
            if known.height else float("nan")
        )
        handles.append(plt.Line2D(
            [], [], color=palette[label], linewidth=1.8,
            label=f"{label}  n={len(ids):,}" + "\n"
                  + f"  mortality {rate:.0%}, {n_vectors:,} vectors",
        ))
    legend_ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1),
                     labelcolor=INK_SECONDARY, fontsize=6.5,
                     handletextpad=0.6, borderpad=0, frameon=False,
                     labelspacing=1.2)
    ax.set_title(
        f"handbook {number}: which limb patients enter, and which way each "
        "limb travels", loc="left", pad=8,
    )
    ax.set_xlabel("PaCMAP-1")
    ax.set_ylabel("PaCMAP-2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#e4e3df", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    figure.text(
        0.012, 0.015,
        "split on which side of centre a patient ENTERS along the limb axis "
        f"({axis[0]:+.2f}, {axis[1]:+.2f}), fitted on the central "
        f"{LIMB_TRIM[1] - LIMB_TRIM[0]:.0f}% of points so the outlying islands "
        "do not flatten it; each colour's field is from its own patients",
        fontsize=6.5, color=INK_SECONDARY,
    )
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out}")
    return pl.DataFrame(rows, infer_schema_length=None)


def outcome_scatter(frame, handbook, out, logger):
    """
    one handbook's patient-timepoints, died against survived

    Drawn as two layers with the smaller group on top and both at low alpha,
    because a scatter of 2,700 patients at seven timepoints each is mostly
    overplotting and whichever group is drawn last otherwise looks larger than
    it is.
    """
    style()
    colors = mortality_colors()
    figure, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(frame["pacmap_1"].to_numpy(), frame["pacmap_2"].to_numpy(),
               s=1.0, c=CONTEXT, linewidths=0, alpha=0.5, rasterized=True)

    parts = {
        bool(value): frame.filter(pl.col("in_hospital_mortality") == value)
        for value in (0, 1)
    }
    for died in sorted(parts, key=lambda d: -parts[d].height):
        part = parts[died]
        if part.height == 0:
            continue
        ax.scatter(
            part["pacmap_1"].to_numpy(), part["pacmap_2"].to_numpy(),
            s=4.0, c=colors[died], linewidths=0, alpha=0.45,
            rasterized=True,
            label=f"{'died' if died else 'survived'}  "
                  f"{part['hospitalization_id'].n_unique():,} patients",
        )
    ax.set_xlabel("PaCMAP-1")
    ax.set_ylabel("PaCMAP-2")
    ax.set_title(f"handbook {handbook}: outcome on the projection", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="best", markerscale=2.0)
    ax.text(
        0.0, -0.13,
        "PaCMAP preserves neighbourhoods, not distances. Overlap here means "
        "the two groups share neighbours;\nseparation would mean they do not. "
        "Neither is a distance.",
        transform=ax.transAxes, fontsize=6.0, color=INK_SECONDARY, va="top",
    )
    save_figure(figure, out)
    logger.info(f"    wrote {out.name}")


def outcome_hexbin(frame, handbook, out, logger, gridsize=28, min_count=10):
    """
    the same handbook as a death FRACTION per hexagon, not two scatters

    The scatter above cannot answer the question it invites. Whichever group
    is drawn second sits on top, and here that is the smaller one, so the
    minority looks like the majority wherever the two overlap. Binning and
    colouring by the share who died removes the draw-order artefact entirely
    and turns "deaths seem to be over there" into a number per region.

    Hexagons below `min_count` patient-timepoints are left grey rather than
    given a fraction, because a 2-of-3 cell reads as 67% mortality and is
    noise.
    """
    style()
    x = frame["pacmap_1"].to_numpy()
    y = frame["pacmap_2"].to_numpy()
    died = frame["in_hospital_mortality"].to_numpy().astype(float)

    figure, ax = plt.subplots(figsize=(5.8, 5.0))
    total = ax.hexbin(x, y, gridsize=gridsize, mincnt=1, visible=False)
    deaths = ax.hexbin(x, y, C=died, reduce_C_function=np.sum,
                       gridsize=gridsize, mincnt=1, visible=False)
    counts = total.get_array()
    fraction = np.where(counts >= min_count,
                        deaths.get_array() / np.maximum(counts, 1), np.nan)
    centres = total.get_offsets()

    thin = np.isnan(fraction)
    if thin.any():
        ax.scatter(centres[thin, 0], centres[thin, 1], s=14, c=CONTEXT,
                   marker="h", linewidths=0)
    shown = ~thin
    # CENTRED ON THIS HANDBOOK'S OWN RATE, not on 0.5. A diverging map over
    # [0, 1] puts its midpoint at 50% mortality, so in a handbook that kills
    # 20% every region of genuinely raised risk still renders blue and the
    # figure reads as uniformly safe. Centring on the handbook rate is what
    # makes the colour mean "worse here than this plan's average", which is
    # the only comparison the panel is for.
    overall = float(np.nanmean(died))
    norm = mcolors.TwoSlopeNorm(
        vmin=0.0, vcenter=min(max(overall, 1e-3), 1 - 1e-3), vmax=1.0
    )
    mesh = ax.scatter(
        centres[shown, 0], centres[shown, 1], c=fraction[shown], s=14,
        marker="h", cmap="RdYlBu_r", norm=norm, linewidths=0,
    )
    bar = figure.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02)
    bar.set_label("share who died", fontsize=7)
    bar.ax.axhline(overall, color="#0b0b0b", linewidth=0.9)
    bar.ax.tick_params(labelsize=6)
    ax.set_xlabel("PaCMAP-1")
    ax.set_ylabel("PaCMAP-2")
    ax.set_title(
        f"handbook {handbook}: mortality by region", loc="left"
    )
    ax.text(
        0.0, -0.13,
        f"Grey hexagons hold fewer than {min_count} patient-timepoints and are "
        f"left uncoloured. Handbook mortality is {overall:.1%}; "
        "blue is below "
        "that and red above, so the colour is a local departure from this "
        "handbook's own rate.",
        transform=ax.transAxes, fontsize=6.0, color=INK_SECONDARY, va="top",
    )
    save_figure(figure, out)
    logger.info(
        f"    wrote {out.name} "
        f"({int(shown.sum())} hexagons above the floor, "
        f"{int(thin.sum())} below)"
    )


def token_violins(rate, labels, names, handbook, out, logger, top=8):
    """
    the tokens whose rate differs most between died and survived, as violins

    ON THE LOG SCALE AND ZERO-INFLATED, both of which are visible rather than
    hidden. Most patients record zero of most tokens, so the violins have a
    heavy foot at zero and the interesting part is the upper tail; the median
    and the share of patients with any at all are printed beside each violin
    because a violin alone cannot show a point mass.
    """
    from scipy.stats import mannwhitneyu

    columns = [c for c in rate.columns if c != "hospitalization_id"]
    y = labels.to_numpy()
    scored = []
    for column in columns:
        values = rate[column].to_numpy().astype(np.float64)
        a, b = values[y == 0], values[y == 1]
        # both groups must clear the cell floor on PRESENCE, not on rows, or a
        # token held by three dying patients can lead the ranking
        if (a > 0).sum() < SUPPRESS or (b > 0).sum() < SUPPRESS:
            continue
        spread = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        if spread <= 0:
            continue
        try:
            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            continue
        scored.append((abs((b.mean() - a.mean()) / spread), column, a, b, p))
    if not scored:
        logger.info("    no token clears the cell floor in both groups")
        return pl.DataFrame()
    scored.sort(reverse=True, key=lambda row: row[0])
    chosen = scored[:top]

    style()
    colors = mortality_colors()
    figure, ax = plt.subplots(figsize=(7.2, 0.62 * len(chosen) + 1.6))
    rows = []
    for position, (smd, column, a, b, p) in enumerate(chosen):
        for offset, values, died in (
            (-0.18, a, False), (0.18, b, True)
        ):
            body = ax.violinplot(
                [np.log1p(values)], positions=[position + offset],
                widths=0.32, vert=False, showextrema=False, showmedians=True,
            )
            for patch in body["bodies"]:
                patch.set_facecolor(colors[died])
                patch.set_alpha(0.55)
                patch.set_linewidth(0)
            body["cmedians"].set_color(INK_SECONDARY)
            body["cmedians"].set_linewidth(0.8)
        rows.append({
            "handbook": handbook,
            "token_id": int(column),
            "token": names.get(int(column), str(column)),
            "smd": float(smd),
            "p": p,
            "median_survived": float(np.median(a)),
            "median_died": float(np.median(b)),
            "present_survived": float((a > 0).mean()),
            "present_died": float((b > 0).mean()),
        })
    edge = ax.get_xlim()[1]
    for position, (smd, _column, _a, _b, _p) in enumerate(chosen):
        ax.text(
            edge, position, f"  SMD {smd:.2f}", fontsize=6.0,
            color=INK_SECONDARY, va="center",
        )
    ax.set_yticks(range(len(chosen)))
    ax.set_yticklabels(
        [names.get(int(c), str(c))[:40] for _, c, _, _, _ in chosen],
        fontsize=6.5,
    )
    ax.invert_yaxis()
    ax.set_xlabel("log(1 + token count in the window)")
    ax.set_title(
        f"handbook {handbook}: token use by outcome", loc="left"
    )
    handles = [
        plt.Line2D([], [], color=colors[d], linewidth=6, alpha=0.55,
                   label="died" if d else "survived")
        for d in (False, True)
    ]
    ax.legend(
        handles=handles, frameon=False, fontsize=7, loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.text(
        0.0, -0.10 - 0.25 / max(len(chosen), 1),
        "Most patients record none of most tokens, so each violin has a point "
        "mass at zero that the shape cannot show;\nthe table beside this "
        "figure carries the share of patients with any.",
        transform=ax.transAxes, fontsize=6.0, color=INK_SECONDARY, va="top",
    )
    save_figure(figure, out)
    logger.info(f"    wrote {out.name}")
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--cap-hours", type=int, default=None)
    parser.add_argument("--handbooks", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--min-group", type=int, default=None)
    parser.add_argument("--min-fit", type=float, default=None)
    parser.add_argument("--focus-handbook", type=int, default=None,
                        help="which handbook gets the outcome scatter, the "
                             "token violins and the rendered divergences. "
                             "Defaults to the largest.")
    parser.add_argument("--landmark", type=float, default=24.0)
    parser.add_argument("--step-hours", type=float, default=6.0)
    parser.add_argument("--gap-open", type=float, default=None)
    parser.add_argument("--gap-extend", type=float, default=None)
    parser.add_argument("--keep-code-status", action="store_true")
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
        "42_handbook_figures", paths.log_home / "42_handbook_figures.log"
    )
    figures = paths.output_home / "figures" / paths.model_key
    figures.mkdir(parents=True, exist_ok=True)
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)
    logger.info(f"subset {args.subset}; focus {args.focus_handbook or 'auto'}")

    probe = sibling("41_divergence_probe")
    traj = sibling("09_trajectories")

    logger.info("rebuilding the handbooks:")
    part_all, response_matrix, medoids = probe.rebuild_handbooks(
        paths, args, logger
    )

    logger.info("handbook catalogue:")
    cat = catalogue(part_all, medoids, logger)
    save_table(cat, tables / f"handbook_catalogue_{args.subset}.csv")

    focus = args.focus_handbook
    if focus is None:
        focus = int(cat.sort("patients", descending=True)["handbook"][0])
    logger.info(f"focus handbook: {focus}")

    # ---- the projection, for the three figures drawn on it
    projection = (
        paths.patient_level_home
        / f"trajectory_projections_{args.subset}_corrected-coral.parquet"
    )
    if not projection.is_file():
        logger.warning(f"no projection at {projection}; figures skipped")
        grid = None
    else:
        grid = pl.read_parquet(projection).select(
            "hospitalization_id", "offset_hours", "pacmap_1", "pacmap_2"
        ).join(
            part_all.select("hospitalization_id", "handbook", "worse_group",
                            "in_hospital_mortality"),
            on="hospitalization_id", how="inner",
        )
        offsets = sorted(grid["offset_hours"].unique().to_list())
        spacing = (
            min(b - a for a, b in zip(offsets, offsets[1:]))
            if len(offsets) > 1 else args.step_hours
        )
        logger.info(
            f"  {grid.height:,} grid points carry a handbook, "
            f"{grid['hospitalization_id'].n_unique():,} patients, "
            f"{spacing:.0f}h spacing"
        )

        logger.info("handbook flow overlay:")
        traj.plot_cluster_flow_overlay(
            grid, "pacmap_1", "pacmap_2", "handbook",
            figures / f"handbook_flow_{args.subset}.png",
            f"{args.subset} shock: how each treatment handbook moves",
            logger, step_hours=spacing,
        )

        logger.info("which limb each handbook's patients enter:")
        axis, centre = limb_axis(grid)
        logger.info(
            f"  limb axis ({axis[0]:+.2f}, {axis[1]:+.2f}) through "
            f"({centre[0]:+.2f}, {centre[1]:+.2f}), from the central "
            f"{LIMB_TRIM[1] - LIMB_TRIM[0]:.0f}% of {grid.height:,} points"
        )
        directions = []
        for number in sorted(grid["handbook"].drop_nulls().unique().to_list()):
            frame = direction_split(
                grid, part_all, number,
                figures / f"handbook_direction_hb{number}_{args.subset}.png",
                logger, traj, axis, centre,
            )
            if frame is not None:
                directions.append(frame)
        if directions:
            save_table(
                pl.concat(directions, how="diagonal_relaxed"),
                tables / f"handbook_direction_{args.subset}.csv",
            )

        logger.info(f"outcome scatter for handbook {focus}:")
        here_grid = grid.filter(
            (pl.col("handbook") == focus)
            & pl.col("in_hospital_mortality").is_not_null()
        )
        outcome_scatter(
            here_grid, focus,
            figures / f"handbook_outcome_hb{focus}_{args.subset}.png",
            logger,
        )
        outcome_hexbin(
            here_grid, focus,
            figures / f"handbook_mortality_hex_hb{focus}_{args.subset}.png",
            logger,
        )

    # ---- rendered divergences, every handbook to disk, focus one logged
    logger.info("rendered divergences:")
    rendered, audits = [], []
    for number in sorted(part_all["handbook"].unique().to_list()):
        part = part_all.filter(pl.col("handbook") == number)
        logger.info(f"  handbook {number}:")
        frame, audit_row = divergence_examples(
            part, response_matrix, args, logger
        )
        if frame.height:
            rendered.append(frame)
        if audit_row:
            audits.append(audit_row)
    if audits:
        save_table(
            pl.DataFrame(audits, infer_schema_length=None),
            tables / f"handbook_plan_agreement_{args.subset}.csv",
        )
    if rendered:
        every = pl.concat(rendered, how="diagonal_relaxed")
        save_table(every, tables / f"handbook_divergences_{args.subset}.csv")
        shown = every.filter(pl.col("handbook") == focus).head(3)
        for row in shown.iter_rows(named=True):
            logger.info(f"    diverged at hour {row['divergence_hour']}")
            logger.info(f"      treat {row['treatment_better']}")
            logger.info(f"      resp  {row['response_better']}")
            logger.info(f"            {row['match']}")
            logger.info(f"      resp  {row['response_worse']}")
            logger.info(f"      treat {row['treatment_worse']}")

    # ---- token violins for the focus handbook
    logger.info(f"token use by outcome within handbook {focus}:")
    from clifcs.tokens import vocabulary
    try:
        names, families = vocabulary(str(paths.tokenizer_yaml))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  vocabulary unavailable: {exc}")
        names, families = {}, {}
    here = part_all.filter(
        (pl.col("handbook") == focus)
        & pl.col("in_hospital_mortality").is_not_null()
        & (
            pl.col("hours_to_discharge").is_null()
            | (pl.col("hours_to_discharge") > args.landmark)
        )
    )
    rate, _timing, _ids = probe.token_features(
        paths, here["hospitalization_id"].to_list(), args.landmark,
        args.step_hours, logger, names, families,
    )
    if rate is None:
        logger.warning("  no token counts; violins skipped")
    else:
        joined = here.select(
            "hospitalization_id", "in_hospital_mortality"
        ).join(rate, on="hospitalization_id", how="inner").fill_null(0)
        summary = token_violins(
            joined.drop("in_hospital_mortality"),
            joined["in_hospital_mortality"], names, focus,
            figures / f"handbook_tokens_hb{focus}_{args.subset}.png", logger,
        )
        if summary.height:
            save_table(
                summary,
                tables / f"handbook_token_use_hb{focus}_{args.subset}.csv",
            )
    logger.info("42 done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
34_state_tokens.py -- which record tokens sit alongside each response state,
and which separate a wean that holds from one that fails

    python code/34_state_tokens.py --subset cardiogenic

THE WINDOW IS SIX HOURS AND THE STATE IS ONE HOUR, which is the join this
script has to be honest about. Token counts exist per embedding window --
window t covers (t-6, t] -- while a response state covers one hour. A state at
grid point t is therefore paired with the six hours of record ENDING at t, so a
token "alongside" a state may have been charted up to five hours before it.

WHAT IS CIRCULAR AND FLAGGED AS SUCH. The states are built from MAP and
vasopressor dose, and the token stream contains MAP and vasopressor tokens. Any
`VTL//map` or `MED-CTS//norepinephrine` token found enriched in a state is
partly that state's own definition read back. Those families are marked
constitutive in every table rather than silently reported as findings; the
informative rows are the others.

THE VOLCANO ANSWERS A FORWARD QUESTION. For "failed wean" against "wean held"
the response -- whether the pressure fell -- lies in the hour AFTER the window
the tokens come from. So a token separating the two is associated with an
outcome not yet observed when it was charted, which is a weaker claim than
cause and a stronger one than coincidence.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.regions import HOURLY_REGION_ORDER
from clifcs.tokens import vocabulary
from clifcs.windows import window_token_counts

OVERLAP = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]
STEP_HOURS = 6.0
MIN_CELL = 11

#: token families that help DEFINE the states, so finding them enriched is
#: circular. Reported, but marked.
CONSTITUTIVE = (
    "VTL//map", "VTL//sbp", "VTL//dbp",
    "MED-CTS//norepinephrine", "MED-CTS//epinephrine",
    "MED-CTS//phenylephrine", "MED-CTS//vasopressin",
    "MED-CTS//dopamine", "MED-CTS//angiotensin",
)


def is_constitutive(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in CONSTITUTIVE)


def contrast(present, group_a, group_b, names, logger, label_a, label_b,
             min_cell=MIN_CELL):
    """
    per-token prevalence in one group against another, with a Bonferroni p

    `present` is a set of (patient, window) keys per token. Prevalence is the
    share of that group's patient-windows carrying the token at least once --
    not the count, so a patient charted twice as often does not double their
    contribution.
    """
    from scipy.stats import chi2_contingency

    rows = []
    for token, keys in present.items():
        hit_a = len(keys & group_a)
        hit_b = len(keys & group_b)
        if min(hit_a, hit_b) < min_cell:
            continue
        table = np.array([
            [hit_a, len(group_a) - hit_a],
            [hit_b, len(group_b) - hit_b],
        ])
        if table.min() < 0 or table.sum(axis=0).min() == 0:
            continue
        _chi2, p, _dof, _exp = chi2_contingency(table, correction=False)
        rate_a = hit_a / max(len(group_a), 1)
        rate_b = hit_b / max(len(group_b), 1)
        name = names.get(token, str(token))
        rows.append({
            "token": name,
            "constitutive": is_constitutive(name),
            f"prevalence_{label_a}": rate_a,
            f"prevalence_{label_b}": rate_b,
            "log2_ratio": float(np.log2(max(rate_a, 1e-9) / max(rate_b, 1e-9))),
            "p": float(p),
            f"n_{label_a}": len(group_a),
            f"n_{label_b}": len(group_b),
        })
    if not rows:
        logger.warning(f"  no token cleared the floor for {label_a} vs {label_b}")
        return pl.DataFrame()
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        p_adjusted=(pl.col("p") * frame.height).clip(upper_bound=1.0),
        neg_log10_p=-np.log10(np.maximum(frame["p"].to_numpy(), 1e-300)),
    ).sort("p")


def plot_volcano(frame, paths, subset, logger, label_a="failed_wean",
                 label_b="wean_held", top=18):
    """
    log2 prevalence ratio against significance, for one state pair

    CONSTITUTIVE TOKENS ARE DRAWN AS OPEN MARKERS, not dropped. A
    norepinephrine-dose token separating a failed wean from one that held is
    close to tautological -- the wean is defined by the dose coming down -- but
    removing those points would hide how much of the plot they occupy, which is
    itself worth seeing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from clifcs.figures import save_figure

    figure, axes = plt.subplots(figsize=(10.5, 8.0))
    cut = -np.log10(0.05 / max(frame.height, 1))
    core = frame.filter(~pl.col("constitutive"))
    circular = frame.filter(pl.col("constitutive"))
    axes.scatter(
        core["log2_ratio"], core["neg_log10_p"], s=26, c="#2166ac",
        alpha=0.75, linewidths=0, label="other tokens",
    )
    if not circular.is_empty():
        axes.scatter(
            circular["log2_ratio"], circular["neg_log10_p"], s=36,
            facecolors="none", edgecolors="#b2182b", linewidths=1.1,
            label="constitutive (MAP / pressor dose)",
        )
    axes.axhline(cut, color="#666", ls="--", lw=0.9)
    axes.axvline(0, color="#666", lw=0.8)
    axes.text(
        axes.get_xlim()[0], cut, "  Bonferroni p<0.05", va="bottom",
        ha="left", fontsize=8, color="#555",
    )
    labelled = core.filter(pl.col("neg_log10_p") > cut).sort(
        pl.col("log2_ratio").abs(), descending=True
    ).head(top)
    for row in labelled.iter_rows(named=True):
        axes.annotate(
            row["token"], (row["log2_ratio"], row["neg_log10_p"]),
            fontsize=7, xytext=(4, 3), textcoords="offset points",
        )
    axes.set_xlabel(
        f"log2 prevalence ratio   <- more common when the wean HELD"
        f"        more common when it FAILED ->"
    )
    axes.set_ylabel("-log10 p")
    axes.set_title(
        f"{subset} shock: tokens separating a failed wean from one that held\n"
        "tokens read from the six hours ending at the wean; the outcome lies "
        "in the hour after",
        fontsize=11,
    )
    axes.legend(fontsize=8, frameon=False, loc="upper left")
    figure.tight_layout()
    out = (
        paths.output_home / "figures" / paths.model_key
        / f"wean_token_volcano_{subset}.png"
    )
    save_figure(figure, out, manuscript=True)
    logger.info(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger("34_state_tokens", paths.log_home / "34_state_tokens.log")
    tables = paths.output_home / "tables" / paths.model_key
    tables.mkdir(parents=True, exist_ok=True)

    labelled = pl.read_parquet(
        paths.patient_level_home / f"hourly_regions_{args.subset}.parquet"
    ).filter(pl.col("offset_hours").is_in(OVERLAP))
    logger.info(
        f"{labelled.height:,} hourly states on the embedding grid, "
        f"{labelled['hospitalization_id'].n_unique():,} patients"
    )

    names, _families = vocabulary(str(paths.tokenizer_yaml))
    counts = window_token_counts(
        paths, OVERLAP, STEP_HOURS,
        labelled["hospitalization_id"].unique().to_list(), logger,
    )
    present = {}
    for token, patient, window in zip(
        counts["token_id"].to_list(),
        counts["hospitalization_id"].to_list(),
        counts["window"].to_list(),
    ):
        present.setdefault(int(token), set()).add((patient, float(window)))
    logger.info(f"  {len(present):,} distinct tokens in the windows")

    keys = {
        state: set(
            zip(
                labelled.filter(pl.col("region") == state)
                ["hospitalization_id"].to_list(),
                labelled.filter(pl.col("region") == state)
                ["offset_hours"].to_list(),
            )
        )
        for state in HOURLY_REGION_ORDER
    }
    everything = set().union(*keys.values())

    # ---- each state against everything else
    blocks = []
    for state in HOURLY_REGION_ORDER:
        group = keys[state]
        if len(group) < 200:
            logger.info(f"  {state}: {len(group)} windows, skipped")
            continue
        frame = contrast(
            present, group, everything - group, names, logger,
            "state", "rest",
        )
        if frame.is_empty():
            continue
        frame = frame.with_columns(state=pl.lit(state))
        blocks.append(frame)
        top = frame.filter(
            ~pl.col("constitutive") & (pl.col("p_adjusted") < 0.05)
        ).sort("log2_ratio", descending=True).head(4)
        logger.info(
            f"  {state:<20} ({len(group):,} windows) most enriched "
            "non-constitutive tokens:"
        )
        for row in top.iter_rows(named=True):
            logger.info(
                f"      {row['token']:<36} {row['prevalence_state']:>6.1%} vs "
                f"{row['prevalence_rest']:>6.1%}  log2 {row['log2_ratio']:+.2f}"
            )
    if blocks:
        pl.concat(blocks, how="diagonal_relaxed").write_csv(
            tables / f"state_tokens_{args.subset}.csv"
        )

    # ---- the volcano: failed wean against wean held
    failed, held = keys["failed wean"], keys["wean held"]
    logger.info(
        f"volcano: {len(failed):,} failed-wean windows against "
        f"{len(held):,} that held"
    )
    volcano = contrast(
        present, failed, held, names, logger, "failed_wean", "wean_held"
    )
    if not volcano.is_empty():
        volcano.write_csv(tables / f"wean_token_volcano_{args.subset}.csv")
        significant = volcano.filter(
            (pl.col("p_adjusted") < 0.05) & ~pl.col("constitutive")
        )
        logger.info(
            f"  {volcano.height:,} tokens tested, {significant.height:,} "
            "significant and non-constitutive"
        )
        for row in significant.sort(
            pl.col("log2_ratio").abs(), descending=True
        ).head(12).iter_rows(named=True):
            logger.info(
                f"      {row['token']:<36} failed "
                f"{row['prevalence_failed_wean']:>6.1%} vs held "
                f"{row['prevalence_wean_held']:>6.1%}  "
                f"log2 {row['log2_ratio']:+.2f}"
            )
        plot_volcano(volcano, paths, args.subset, logger)
    logger.info("34 done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
small-cell suppression for the aggregate tables a site sends

A count of 2 in a `seymour_phenotype x shock_type x site` cell is closer to
naming a patient than to being a statistic, and the breakdown tables this
consortium asked for are exactly the ones that produce such cells: four Seymour
groups times four Bhavani groups times six shock types times however many sites
is a lot of cells over a cohort that is not that large once it is divided.

THE RULE, which the four passes below implement:

    No number a bundle publishes, and no number a reader can derive from one
    row of a published table by arithmetic, may resolve to a group of 1 to
    `min_cell - 1` patients.

Four things have to happen, and doing only the first is the usual mistake.

1. **Primary suppression** -- blank any cell holding 1 to `min_cell - 1`.

2. **Complementary suppression** -- if exactly one cell in a group is blanked
   and the total is published, subtracting gives it straight back. So blanking
   one cell forces blanking a second: the smallest positive survivor, which
   loses the least.

3. **Within-row complements** -- the same subtraction works along a row, not
   just down a column. `n_total - n_charted` is a patient cell nobody declared.
   `COMPLEMENT_PAIRS` names those.

4. **Every statistic names its denominator** -- a mortality of 1.000 on n = 3
   names three patients' outcomes whether or not the 3 is printed, and a
   proportion over a published denominator *reveals its numerator*. There is no
   rounding that fixes this at these n: to make a numerator ambiguous across 11
   integers at n = 40 you would have to round to 0.3. So a statistic is
   published only when the count it was computed over survives, and when its
   numerator and that numerator's complement are each either zero or at least
   `min_cell`.

Point 4 is why `STATISTICS` exists and why it is keyed by column name alone,
pipeline-wide: one name, one denominator, everywhere. The predecessor keyed
every statistic in a frame to a single anchor count -- `n` if present, else
whichever count column happened to come first -- so `mortality_concordant`,
computed over the treated arm, was suppressed on `n_eligible`, a superset of a
superset. A row with 40 eligible, 24 with concordance known and a concordant
share of 0.125 published a mortality over three patients, and nothing looked at
it.

`min_cell = 11` is the default because that is the threshold CMS applies to
beneficiary cell sizes and the one most consortium DUAs have been written
against; it is configurable because the right number is a policy decision and
this module should not be the thing that encodes it. It is also stricter than
the CLIF consortium's own n >= 10, so a conformant table here is conformant
there.

Suppressed cells become null rather than zero. A zero is a claim that nobody was
in the cell, which is a different and sometimes very interesting statement --
"no patient at this site was assigned Bhavani D" is a data-quality finding, and
conflating it with "between one and ten were" would hide it. The rule preserves
that exactly: a mortality of 0.000 over n = 40 determines the cells {0, 40},
and neither is in 1 to 10.

KNOWN RESIDUAL. All of this is per row, per table. A reader holding the whole
bundle can still difference `baseline_by_shock_type`'s `n` against
`shock_type_summary`'s. Closing that needs a cross-table pass and is not
attempted here; `suppression_note` says so, and so does the README.
"""

from __future__ import annotations

import dataclasses

import polars as pl

#: CMS cell-size policy, and what most consortium DUAs are written against
DEFAULT_MIN_CELL = 11

#: columns that are a count of patients, by this pipeline's naming convention
COUNT_PREFIX = "n"

#: `n_`-prefixed columns that count something OTHER than patients, and so must
#: not be suppressed at the cell threshold.
#:
#: `n_variables_compared` counts published IQRs, not people. Its value is 4 to 8
#: for every trial -- always below `min_cell` -- so the convention blanked it in
#: every row of every `trial_outcomes` table ever exported, taking with it the
#: denominator a reader needs to interpret `share_inside_all_iqrs` (which falls
#: as 0.5^k whether or not the cohort differs at all). The suppression also
#: protected nothing: `null_inside_all_iqrs` publishes 0.5^k beside it, so k was
#: recoverable by anyone who took a log.
#:
#: The others are per-patient counts of *trials*, used as grouping keys in the
#: coverage tables. A patient eligible for three trials is not a cell of three
#: patients.
NOT_PATIENT_COUNTS = frozenset(
    {
        "n_variables_compared",
        "n_eligible_trials",
        "n_specific_trials",
        "n_trials_in_type",
        "n_specific_trials_in_type",
    }
)


@dataclasses.dataclass(frozen=True)
class Statistic:
    """
    one published statistic, and the count it was computed over

    `over` is the column holding its denominator, and it has to be in the same
    row. A statistic whose denominator cannot be named this way is not
    publishable -- see the module docstring.

    `proportion` says the value times `over` is a patient count. That makes the
    numerator, and the numerator's complement within the denominator, cells in
    their own right. Medians and means are not proportions: they are summaries
    over the denominator and determine no sub-cell.

    `follows` names statistics this one is a transform of, so a blanked parent
    takes its children with it (`ratio_to_null` is meaningless once
    `share_inside_all_iqrs` is gone).
    """

    name: str
    over: str
    proportion: bool = False
    follows: tuple[str, ...] = ()
    #: the count column this statistic's NUMERATOR equals, where the table
    #: exports it. A proportion reconstructs its numerator exactly, so if that
    #: numerator is a column the suppressor blanked, the proportion hands it
    #: straight back -- including when the blanking was complementary rather
    #: than primary, which is the case the first version of this rule missed.
    #: `share` over `n_total` passed every numerator test while recovering a
    #: blanked `n`, and through it the small cell the complementary pass had
    #: been protecting.
    numerator: str = ""
    why: str = ""


#: every statistic column any exported table carries, with the count it rests
#: on. Keyed by name alone and asserted single-valued by
#: `tests/test_small_cell_rule.py`, so one column name cannot mean two
#: denominators in two tables.
STATISTICS: tuple[Statistic, ...] = (
    # ---------------------------------------------------------- trial_outcomes
    Statistic(
        "share_eligible", "n_stratum", True,
        numerator="n_eligible",
        why="share of the stratum this trial would have enrolled",
    ),
    Statistic(
        "mortality_eligible", "n_eligible", True,
        why="in-hospital mortality among the eligible",
    ),
    Statistic(
        "median_icu_los_eligible", "n_eligible",
        why="a median over the eligible, not a proportion of them",
    ),
    Statistic(
        "share_concordant", "n_concordance_known", True,
        numerator="n_concordant",
        why="of those whose treatment is ascertainable, the share treated as "
            "the trial recommends",
    ),
    Statistic(
        "mortality_concordant", "n_concordant", True,
        why="mortality in the treated arm. Rests on the arm size, which is why "
            "n_concordant is exported -- before it was, this rate keyed to "
            "n_eligible and published over groups of three",
    ),
    Statistic(
        "mortality_discordant", "n_discordant", True,
        why="mortality in the untreated arm, same argument",
    ),
    Statistic(
        "share_inside_all_iqrs", "n_comparable", True,
        why="share of comparable patients inside every published IQR",
    ),
    Statistic(
        "ratio_to_null", "n_comparable", follows=("share_inside_all_iqrs",),
        why="that share over the 0.5^k a representative cohort would give",
    ),
    # ------------------------------------------------------ trial_iqr_position
    Statistic(
        "median_here", "n_compared",
        why="this site's median of the variable, against the published one",
    ),
    Statistic("share_below_q1", "n_compared", True, why="position against the published IQR"),
    Statistic("share_inside_iqr", "n_compared", True, why="position against the published IQR"),
    Statistic("share_above_q3", "n_compared", True, why="position against the published IQR"),
    # ------------------------- trial_guideline_assignment and trial_coverage
    Statistic("share_of_stratum", "n_stratum", True,
        numerator="n", why="size of this cell within its stratum"),
    Statistic("mortality", "n", True, why="in-hospital mortality in the cell"),
    Statistic("median_icu_los_days", "n", why="a median over the cell"),
    Statistic("median_age", "n", why="a median over the cell"),
    Statistic("share_female", "n", True, why="sex distribution of the cell"),
    Statistic(
        "median_trials_in_type", "n",
        why="median number of in-type trials applicable to the cell",
    ),
    # --------------------------- phenotype / shock type / sepsis summaries
    Statistic(
        "share", "n_total", True,
        numerator="n",
        why="size of this group within the population the table describes. "
            "n_total is exported for exactly this reason",
    ),
    # ----------------------------------------------------- seymour_availability
    Statistic(
        "availability", "n_total", True,
        numerator="n_measured",
        why="share of patients with the variable measured at all",
    ),
    # -------------------------------------------------- baseline_by_shock_type
    Statistic("median", "n_measured", why="a median over those measured"),
    Statistic("q1", "n_measured", why="a quartile over those measured"),
    Statistic("q3", "n_measured", why="a quartile over those measured"),
    Statistic(
        "share_positive", "n_measured", True,
        numerator="n_positive",
        why="share of those measured who were positive. Named share_positive "
            "rather than share because its denominator is n_measured, and one "
            "column name may only have one denominator",
    ),
    # --------------------------------------------------------- token prevalence
    Statistic(
        "prevalence", "n_cohort", True,
        numerator="n_patients",
        why="share of the cohort that emitted the token at least once",
    ),
    # ------------------------------------------------------------ token_coverage
    Statistic(
        "share_charted", "n_total", True,
        numerator="n_charted",
        why="share of patients with the variable charted",
    ),
    # ---------------------------------------------------------- phenotype_purity
    Statistic(
        "neighbour_purity", "n",
        why="mean share of a patient's neighbours sharing their label. A mean "
            "of per-patient fractions, not a count over a count",
    ),
    Statistic(
        "chance", "n_scheme_population", True,
        numerator="n",
        why="the label's prevalence, which is what purity is compared against",
    ),
    Statistic(
        "lift", "n", follows=("neighbour_purity",),
        why="purity over chance",
    ),
    Statistic(
        "centroid_ratio", "n",
        why="within-label over between-label centroid distance",
    ),
)

#: numeric columns that determine no patient cell at all, each with the reason.
#: Pass 2 of `contract.validate_counts_frame` skips these and REFUSES anything
#: numeric that is in neither this nor `STATISTICS`. That refusal is what makes
#: the export-time re-check real rather than nominal: a table that grows a new
#: rate cannot travel until somebody classifies it.
NOT_PATIENT_STATISTICS: dict[str, str] = {
    "null_inside_all_iqrs": (
        "0.5^k, a property of how many IQRs the trial published and of nothing "
        "about this cohort"
    ),
    "published_median": "the trial's own reported number, from the paper",
    "published_q1": "the trial's own reported number, from the paper",
    "published_q3": "the trial's own reported number, from the paper",
    "total_occurrences": (
        "token occurrences, not patients. The tokencounts stage suppresses it "
        "alongside n_patients on its own argument -- a token occurring four "
        "times in one patient is as identifying as the patient count -- but it "
        "is not itself a patient cell"
    ),
    "distinct_tokens": "vocabulary entries in a family, not patients",
    "token_id": "a vocabulary index",
    "rank": "a decile index from the tokenizer, identical at every site",
    "lo": "a bin edge from the tokenizer",
    "hi": "a bin edge from the tokenizer",
    "seconds": "wall-clock runtime of a pipeline stage",
    "peak_rss_mb": "peak memory of a pipeline stage",
}

#: `(part, whole, why)` where `whole - part` is itself a patient cell, so a
#: small complement has to blank the part. This is the within-row half of
#: complementary suppression; the module used to implement only the down-column
#: half, which left an `n_total - n_charted` of 9 patients in a shipped bundle.
COMPLEMENT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("n_charted", "n_total", "patients who did NOT have the variable charted"),
    ("n_patients", "n_cohort", "patients who never emitted the token"),
    ("n_measured", "n_total", "patients the variable was not measured in"),
    ("n_measured", "n", "patients in the cell the variable was not measured in"),
    ("n_positive", "n_measured", "measured patients who were negative"),
    ("n_eligible", "n_stratum", "stratum patients the trial would not have enrolled"),
    ("n_concordance_known", "n_eligible", "eligible patients whose treatment is unascertainable"),
    ("n_comparable", "n_eligible", "eligible patients with no comparable variable"),
    ("n_concordant", "n_concordance_known", "the untreated arm"),
    ("n_discordant", "n_concordance_known", "the treated arm"),
    ("n", "n_stratum", "the rest of the stratum"),
    ("n", "n_total", "the rest of the population"),
)

_BY_NAME: dict[str, Statistic] = {s.name: s for s in STATISTICS}

_INT_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
)
_NUMERIC_DTYPES = _INT_DTYPES + (pl.Float32, pl.Float64)


def statistic_for(column: str) -> Statistic | None:
    """the registered statistic for a column name, or None if unregistered"""
    return _BY_NAME.get(column)


def count_columns(frame: pl.DataFrame) -> list[str]:
    """the integer columns that hold a patient count"""
    return [
        c
        for c in frame.columns
        if (c == COUNT_PREFIX or c.startswith("n_"))
        and c not in NOT_PATIENT_COUNTS
        and frame.schema[c] in _INT_DTYPES
    ]


def numeric_columns(frame: pl.DataFrame) -> list[str]:
    """every numeric column, counts included"""
    return [c for c in frame.columns if frame.schema[c] in _NUMERIC_DTYPES]


def publishable(stat: Statistic, min_cell: int) -> pl.Expr:
    """
    whether each row may publish `stat`, as a boolean expression

    Shared by the suppressor and the validator so the two cannot drift -- the
    whole point of checking at both ends is that neither assumes the other ran,
    and that only holds if they ask the identical question.

    Exact for a proportion only if the value carries full precision, which is
    why `baseline.summarise` no longer rounds its share to four places: a
    rounded proportion makes `value * over` land between integers and the
    numerator test becomes a guess.
    """
    denominator = pl.col(stat.over)
    ok = denominator.is_not_null() & (denominator >= min_cell)
    if stat.proportion:
        numerator = (pl.col(stat.name) * denominator).round(0)
        complement = denominator - numerator
        ok = (
            ok
            & ((numerator == 0) | (numerator >= min_cell))
            & ((complement == 0) | (complement >= min_cell))
        )
    if stat.numerator:
        # blanked numerator column -> the proportion would restore it
        ok = ok & pl.col(stat.numerator).is_not_null()
    for parent in stat.follows:
        ok = ok & pl.col(parent).is_not_null()
    return ok


def _complement_pass(frame: pl.DataFrame, min_cell: int) -> tuple[pl.DataFrame, int]:
    """blank a part whose complement within its whole is 1 to min_cell-1"""
    out = frame
    blanked = 0
    for part, whole, _why in COMPLEMENT_PAIRS:
        if part not in out.columns or whole not in out.columns:
            continue
        if part in NOT_PATIENT_COUNTS or whole in NOT_PATIENT_COUNTS:
            continue
        gap = pl.col(whole) - pl.col(part)
        exposed = gap.is_not_null() & (gap > 0) & (gap < min_cell)
        blanked += int(out.select(exposed.sum()).item() or 0)
        out = out.with_columns(
            pl.when(exposed).then(None).otherwise(pl.col(part)).alias(part)
        )
    return out, blanked


def suppress_counts(
    frame: pl.DataFrame,
    *,
    min_cell: int = DEFAULT_MIN_CELL,
    by: tuple[str, ...] | None = (),
    logger=None,
) -> pl.DataFrame:
    """
    blank every cell a reader could resolve to fewer than `min_cell` patients

    `by` names the grouping a published row corresponds to -- the columns held
    *constant* across the cells that partition a population. `()` means the
    whole frame is one group, which is right for a single distribution over
    phenotype labels. `None` opts out of the complementary pass for a table
    whose rows do not partition anything (`filter_ledger`, where the rows are
    nested subsets and subtraction between steps IS the attrition ledger).

    There is no `derived` parameter any more. Statistics are found by
    intersecting `STATISTICS` with the frame's columns and each is keyed to
    its own denominator, which is the fix: the old single-anchor scheme
    suppressed `mortality_concordant` on `n_eligible`, two supersets away from
    the arm it was computed over.

    Returns a new frame; does not mutate.
    """
    counts = count_columns(frame)
    out = frame
    n_primary = 0
    n_complementary = 0

    for column in counts:
        small = (pl.col(column) > 0) & (pl.col(column) < min_cell)
        n_primary += int(out.select(small.sum()).item() or 0)
        out = out.with_columns(
            pl.when(small).then(None).otherwise(pl.col(column)).alias(column)
        )

        # -------------------------------------------------- complementary pass
        # One blanked cell in a group whose total is published is recoverable by
        # subtraction, so blank a second. Four things were wrong here and
        # between them they made this fire on no exported table at all:
        #
        #   `by` was built as "every candidate column present", which included
        #   the varying dimension -- ("shock_type", "site") for the shock-type
        #   breakdown -- so every group held exactly one row.
        #
        #   `by=()` was read as "skip" rather than "one group", so a single
        #   distribution over four labels was never protected.
        #
        #   `n_cells > 2` excluded the two-cell case, which is the MOST exposed:
        #   blank one of two and the other is the total minus a blank.
        #
        #   the victim was the smallest survivor including zeros, and blanking a
        #   zero protects nothing -- the remaining sum still gives the
        #   suppressed cell back.
        if by is None:
            continue
        group = list(by)
        counted = (
            out.group_by(group).agg(
                n_suppressed=pl.col(column).is_null().sum(),
                n_cells=pl.len(),
            )
            if group
            else out.select(
                n_suppressed=pl.col(column).is_null().sum(),
                n_cells=pl.len(),
            )
        )
        exposed_groups = counted.filter(
            (pl.col("n_suppressed") == 1) & (pl.col("n_cells") >= 2)
        )
        if exposed_groups.height == 0:
            continue

        if group:
            exposed = out.join(exposed_groups.select(group), on=group, how="semi")
        else:
            exposed = out
        # the smallest POSITIVE survivor; a zero victim would be no victim
        survivors = exposed.filter(
            pl.col(column).is_not_null() & (pl.col(column) > 0)
        )
        if survivors.height == 0:
            # nothing positive left to sacrifice, so this column cannot be
            # published for the group at all
            if group:
                out = (
                    out.join(
                        exposed_groups.select(group).with_columns(
                            _blank=pl.lit(True)
                        ),
                        on=group,
                        how="left",
                    )
                    .with_columns(
                        pl.when(pl.col("_blank").fill_null(False))
                        .then(None)
                        .otherwise(pl.col(column))
                        .alias(column)
                    )
                    .drop("_blank")
                )
            else:
                out = out.with_columns(
                    pl.lit(None, dtype=frame.schema[column]).alias(column)
                )
            n_complementary += 1
            continue

        if group:
            victims = (
                survivors.sort(column)
                .group_by(group, maintain_order=True)
                .head(1)
                .select(*group, **{f"_victim_{column}": pl.col(column)})
            )
            out = out.join(victims, on=group, how="left")
        else:
            victims = survivors.sort(column).head(1).select(
                **{f"_victim_{column}": pl.col(column)}
            )
            out = out.join(victims, how="cross")
        n_complementary += victims.height
        out = out.with_columns(
            pl.when(pl.col(column) == pl.col(f"_victim_{column}"))
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
        ).drop(f"_victim_{column}")

    # -------------------------------------------------- within-row complements
    out, n_complement_pairs = _complement_pass(out, min_cell)

    # --------------------------------------------------------------- statistics
    # each keyed to its OWN denominator rather than to one anchor for the frame
    for stat in STATISTICS:
        if stat.name not in out.columns or stat.over not in out.columns:
            continue
        out = out.with_columns(
            pl.when(publishable(stat, min_cell))
            .then(pl.col(stat.name))
            .otherwise(None)
            .alias(stat.name)
        )

    if logger is not None and (n_primary or n_complementary or n_complement_pairs):
        parts = [f"{n_primary} cell(s) below {min_cell}"]
        if n_complementary:
            parts.append(
                f"{n_complementary} that subtraction down a column would have "
                "recovered"
            )
        if n_complement_pairs:
            parts.append(
                f"{n_complement_pairs} whose row complement was under {min_cell}"
            )
        logger.info("  suppressed " + ", ".join(parts))
    return out


def suppression_note(min_cell: int = DEFAULT_MIN_CELL) -> str:
    """the sentence that has to appear under any table this produced"""
    return (
        f"Cells holding fewer than {min_cell} patients are suppressed (shown "
        "blank, not zero), as is one further cell per group where a single "
        "suppression would otherwise be recoverable by subtraction, any cell "
        "whose complement within a published total is under the threshold, and "
        "any rate or median whose own denominator, numerator or numerator "
        "complement is under it. Suppression is applied per table: a reader "
        "holding the whole bundle may still be able to difference one table "
        "against another."
    )

#!/usr/bin/env python3

"""
the rule: no published number may resolve to a group of 1 to min_cell-1

These test `suppress` and `contract` in isolation -- no I/O, no cohort, no
config -- because the rule is the thing that has to hold, independently of
which producer happens to emit a table.

Each test names a failure that actually shipped. The audit that prompted them
found, in bundles already uploaded: a treated-arm mortality published over
three patients, a per-site share published beside the blanked count it
reconstructed, a token coverage row whose complement named nine patients, and
a baseline table whose declared denominator was really its numerator. All four
got out because `validate_counts_frame` skipped every column not named `n*`,
so no rate was inspected at either end of the pipeline.
"""

from __future__ import annotations

import pathlib
import sys

import polars as pl
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from clifcs.export import contract, suppress  # noqa: E402

MIN_CELL = 11


def refuses(frame: pl.DataFrame) -> bool:
    try:
        contract.validate_counts_frame(frame, "test", MIN_CELL)
    except contract.ContractViolation:
        return True
    return False


# ------------------------------------------------------------ the registry


def test_registry_is_single_valued():
    """one column name, one denominator, pipeline-wide"""
    names = [s.name for s in suppress.STATISTICS]
    assert len(names) == len(set(names)), "a statistic name is declared twice"
    overlap = (
        set(names) & set(suppress.NOT_PATIENT_STATISTICS)
        | set(names) & set(suppress.NOT_PATIENT_COUNTS)
    )
    assert not overlap, f"declared both a statistic and not-a-cell: {overlap}"


def test_every_proportion_names_a_count_as_its_denominator():
    for stat in suppress.STATISTICS:
        assert stat.over.startswith("n"), (
            f"{stat.name} is computed over {stat.over!r}, which is not a count "
            "column by this pipeline's naming convention"
        )


# ------------------------------------------------- what the validator refuses


def test_a_rate_over_a_blanked_count_is_refused():
    assert refuses(
        pl.DataFrame(
            {"n": [None], "mortality": [1.0]},
            schema={"n": pl.Int64, "mortality": pl.Float64},
        )
    )


def test_a_rate_whose_denominator_column_is_absent_is_refused():
    assert refuses(pl.DataFrame({"mortality": [0.5]}))


def test_an_unregistered_numeric_column_is_refused():
    """
    the allowlist property, and the reason the re-check is real

    A table that grows a new rate cannot travel until somebody classifies it.
    Without this the check is nominal: the previous version simply skipped any
    column it did not recognise.
    """
    assert refuses(pl.DataFrame({"n": [50], "unclassified_score": [0.4]}))


def test_the_audited_trial_row_is_refused():
    """40 eligible, 24 ascertainable, 12.5% concordant -> a treated arm of 3"""
    assert refuses(
        pl.DataFrame(
            {
                "n_eligible": [40],
                "n_concordance_known": [24],
                "share_concordant": [0.125],
                "n_concordant": [3],
                "mortality_concordant": [1.0],
            }
        )
    )


def test_validation_does_not_depend_on_suppression_having_run():
    """neither end assumes the other ran -- that is why there are two"""
    raw = pl.DataFrame({"scheme": ["x"], "n": [3], "mortality": [1 / 3]})
    assert refuses(raw)
    assert not refuses(suppress.suppress_counts(raw, min_cell=MIN_CELL, by=("scheme",)))


# --------------------------------------------------- what suppression blanks


@pytest.mark.parametrize(
    "rate,survives,why",
    [
        (0.0, True, "determines the cells {0, 40}; a zero is a claim"),
        (1.0, True, "determines {40, 0}"),
        (0.5, True, "determines {20, 20}"),
        (0.1, False, "numerator of 4"),
        (0.9, False, "complement of 4"),
    ],
)
def test_a_proportion_is_judged_on_numerator_and_complement(rate, survives, why):
    out = suppress.suppress_counts(
        pl.DataFrame({"scheme": ["x"], "n": [40], "mortality": [rate]}),
        min_cell=MIN_CELL,
    )
    assert (out["mortality"][0] is not None) is survives, why


def test_a_median_is_not_a_proportion():
    """a median over the cell determines no sub-cell, so it survives"""
    out = suppress.suppress_counts(
        pl.DataFrame({"scheme": ["x"], "n": [40], "median_icu_los_days": [4.8]}),
        min_cell=MIN_CELL,
    )
    assert out["median_icu_los_days"][0] == 4.8


def test_a_proportion_cannot_restore_a_complementarily_blanked_count():
    """
    the hole the first version of this rule left open

    `share` over `n_total` passed every numerator test -- numerator 143,
    complement 157, both large -- while reconstructing an `n` that the
    COMPLEMENTARY pass had blanked in order to protect a 7-patient cell. The
    share handed back the 143, and 300 - 150 - 143 handed back the 7.
    """
    frame = pl.DataFrame(
        {
            "site": ["S"] * 3,
            "shock_type": ["septic", "cardiogenic", "hypovolemic"],
            "n": [150, 143, 7],
            "n_total": [300] * 3,
            "share": [150 / 300, 143 / 300, 7 / 300],
        }
    )
    out = suppress.suppress_counts(frame, min_cell=MIN_CELL, by=("site",))
    survived = [
        row["share"] * row["n_total"]
        for row in out.iter_rows(named=True)
        if row["share"] is not None and row["n"] is None
    ]
    assert not survived, f"a blanked n was recoverable from its share: {survived}"
    assert not refuses(out)


# ---------------------------------------------------------- token prevalence


def _token_families(n_cohort=9135):
    """
    the tokencounts shape, including the row that failed the export

    LABS is the regression: a family emitted by all but three of the cohort.
    RARE exercises the primary rule, UNIVERSAL the zero-complement case that
    the doctrine deliberately keeps publishable.
    """
    return pl.DataFrame(
        {
            "family": ["VITALS", "LABS", "RARE", "UNIVERSAL"],
            "n_patients": [5000, n_cohort - 3, 4, n_cohort],
            "total_occurrences": [120000, 90000, 9, 300000],
            "distinct_tokens": [40, 55, 2, 7],
            "n_cohort": [n_cohort] * 4,
        }
    ).with_columns(prevalence=pl.col("n_patients") / pl.col("n_cohort"))


def _as_tokencounts_writes_it(frame, min_cell=MIN_CELL):
    """what stage_tokencounts now does, in the order it does it"""
    out = suppress.suppress_counts(frame, min_cell=min_cell, by=None)
    return out.with_columns(
        total_occurrences=pl.when(pl.col("n_patients").is_null())
        .then(None)
        .otherwise(pl.col("total_occurrences"))
    )


def test_a_near_universal_token_does_not_leak_its_non_emitters():
    """
    THE REGRESSION. The hand-rolled blanking in `stage_tokencounts` tested only
    `0 < n_patients < min_cell` and stopped, so a family charted for all but
    three patients published both `n_patients` and a full-precision
    `prevalence`; either one gives the three back by subtraction from
    `n_cohort`. The export-time validator refused
    `token_families_anchor-onset.csv` on exactly this, mid-run.
    """
    out = _as_tokencounts_writes_it(_token_families())
    labs = out.filter(pl.col("family") == "LABS").row(0, named=True)

    assert labs["n_patients"] is None, "the count still gives the gap back"
    assert labs["prevalence"] is None, "the rate still gives the gap back"
    assert not refuses(out)


def test_a_universal_token_still_publishes():
    """
    a complement of zero is publishable, as a numerator of zero is: it resolves
    to {0, n_cohort} and neither lands in 1..min_cell-1. Blanking it would hide
    that a family is charted for everyone, which is a finding about the site
    """
    out = _as_tokencounts_writes_it(_token_families())
    universal = out.filter(pl.col("family") == "UNIVERSAL").row(0, named=True)

    assert universal["prevalence"] == 1.0
    assert universal["n_patients"] == 9135


def test_occurrences_go_with_a_blanked_patient_count():
    """
    `total_occurrences` is registered as not-a-patient-cell, so the suppressor
    leaves it alone and the producer must blank it: a token occurring four
    times in one patient is as identifying as the count of patients
    """
    out = _as_tokencounts_writes_it(_token_families())
    for family in ("LABS", "RARE"):
        row = out.filter(pl.col("family") == family).row(0, named=True)
        assert row["total_occurrences"] is None, family


def test_the_vocabulary_size_survives_suppression():
    """
    `distinct_tokens` counts vocabulary entries, not patients. Blanking it for
    a thinly used family would hide that the family is thinly used here, which
    is the finding rather than the disclosure
    """
    out = _as_tokencounts_writes_it(_token_families())
    rare = out.filter(pl.col("family") == "RARE").row(0, named=True)

    assert rare["n_patients"] is None
    assert rare["distinct_tokens"] == 2

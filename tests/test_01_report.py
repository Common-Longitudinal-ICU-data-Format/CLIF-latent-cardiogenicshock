#!/usr/bin/env python3

"""
tests for the 01 run report

Two things are worth testing here and the tables are not among them.

THE FORMATTING RULE. `01_report.py` decides how a cell is printed from the
NAME of its column, because a count and a share are both numbers and only the
name knows which is which. That rule can be wrong about the dtype that reaches
it -- concatenating one site's table with another's widens an all-null column
to String, so a count arrives as "4" -- and the cases below pin both the rule
and the coercion.

THE DEGRADED PATH. The report's stated contract is that a stage with no output
says so rather than being omitted, so that a reader of a partial run can tell a
stage that found nothing from a stage that never ran. Every section is
therefore run here against an empty output directory, which is the state a
contributing site is in before its first run and the one most likely to be hit
by someone who has never run this before.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import polars as pl
import pytest

CODE = pathlib.Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))


def load_report():
    """01_report.py, whose name starts with a digit and so cannot be imported"""
    spec = importlib.util.spec_from_file_location(
        "report_01_under_test", CODE / "01_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_report()


# --------------------------------------------------------- the naming rule


@pytest.mark.parametrize(
    "column,expected",
    [
        ("n", "int"),
        ("n_patients", "int"),
        ("n_cohort", "int"),
        ("total_occurrences", "int"),
        ("distinct_tokens", "int"),
        ("share", "pct"),
        ("share_eligible", "pct"),
        ("prevalence", "pct"),
        ("availability", "pct"),
        ("mortality", "pct"),
        ("mortality_concordant", "pct"),
        ("median", "num1"),
        ("q1", "num1"),
        ("median_icu_los_days", "num1"),
        ("adjusted_rand", "num3"),
        ("ratio_to_null", "num3"),
        # an identifier that happens to be an integer, and must not be given
        # thousands separators -- token 1,193 is not a number anyone adds up
        ("token_id", "id"),
        # unnamed by the rule, left to the dtype. The attrition ledger is
        # pivoted on site, so its count columns are called UCMC and NU.
        ("UCMC", ""),
        ("variable", ""),
    ],
)
def test_kind_of(column, expected):
    assert report.kind_of(column) == expected


# ----------------------------------------------------------- the coercion


def test_fmt_formats_by_kind():
    assert report.fmt(1193, "int") == "1,193"
    assert report.fmt(0.1531, "pct") == "15.3%"
    assert report.fmt(4.46249, "num1") == "4.5"
    assert report.fmt(0.74910, "num3") == "0.749"


def test_fmt_leaves_identifiers_alone():
    """token 1193 is not a quantity and must not be given a thousands comma"""
    assert report.fmt(1193, "id") == "1193"


def test_fmt_suppressed_cell_is_a_dash_not_a_zero():
    """
    the whole disclosure guarantee rests on this

    A blank is an absence of permission to publish. Rendering it as 0 would
    turn "fewer than eleven patients" into "no patients", which is a false
    statement about the cohort and the one misreading this report exists to
    prevent.
    """
    assert report.fmt(None, "int") == "&mdash;"
    assert report.fmt(None, "pct") == "&mdash;"


def test_fmt_coerces_a_count_that_arrived_as_text():
    """
    `diagonal_relaxed` widens an all-null column to String

    `n_variables_compared` is Int64 at RUSH and an empty String column at NU
    and UCMC, so concatenating the three gives String and RUSH's counts arrive
    here as "4".
    """
    assert report.fmt("4", "int") == "4"
    assert report.fmt("0.25", "pct") == "25.0%"


def test_fmt_falls_back_to_text_when_coercion_fails():
    """named like a number, written as prose -- print it, do not raise"""
    assert report.fmt("not measured", "int") == "not measured"


def test_fmt_groups_thousands_in_unnamed_integer_columns():
    """the attrition ledger is pivoted on site, so its counts are called UCMC"""
    assert report.fmt(166814, "") == "166,814"


def test_fmt_renders_booleans_as_words():
    assert report.fmt(True, "") == "yes"
    assert report.fmt(False, "") == "no"


# ------------------------------------------------------- the suppression audit


def test_blanked_counts_only_suppressible_columns():
    """
    a null in a label column is not a suppressed cell

    `shock_type` being null means the patient carried no code; `n` being null
    means the cell was withheld. Counting the first as suppression would
    inflate the audit with missing data and make the number useless as a
    check on the threshold.
    """
    frame = pl.DataFrame(
        {
            "shock_type": ["septic", None, "mixed"],
            "n": [1200, None, None],
            "share": [0.2, None, 0.1],
        }
    )
    suppressed, suppressible = report.blanked(frame)
    assert suppressible == 6  # two suppressible columns x three rows
    assert suppressed == 3  # two nulls in n, one in share; shock_type ignored


def test_blanked_ignores_a_table_with_nothing_to_suppress():
    """trial_criteria.csv is literature only and has no count column at all"""
    frame = pl.DataFrame({"trial": ["SHOCK"], "criterion": ["age >= 18"]})
    assert report.blanked(frame) == (0, 0)


# ------------------------------------------------------------- the rendering


def test_table_escapes_cell_contents():
    """a token name is <not in vocabulary>, which must not become markup"""
    frame = pl.DataFrame({"token_name": ["<not in vocabulary>"]})
    rendered = report.table(frame)
    assert "&lt;not in vocabulary&gt;" in rendered
    assert "<not in vocabulary>" not in rendered


def test_table_marks_the_rows_a_reader_counts_down_to():
    frame = pl.DataFrame({"step": ["all hospitalizations", "x [CRITERION 1]"]})
    rendered = report.table(frame, rule_rows={"x [CRITERION 1]"})
    assert rendered.count('<tr class="rule">') == 1


def test_table_says_when_it_has_truncated():
    """
    1,108 rows of token prevalence do not belong in a report

    But a silently truncated table is a table a reader will sum, so the head
    has to announce itself as a head.
    """
    frame = pl.DataFrame({"n": list(range(100))})
    rendered = report.table(frame, limit=10)
    assert "10 of 100 rows shown" in rendered


def test_empty_table_is_not_an_empty_string():
    """
    tabs() drops empty panels, so a table rendering to "" would vanish

    A stratum that came out with no rows has to be visible as such; dropping
    its tab would read as a stratum that was never computed.
    """
    assert report.table(pl.DataFrame({"n": []})) != ""
    assert report.table(None) != ""


# -------------------------------------------------------- the degraded path


class EmptyRun:
    """
    a machine where nothing has run yet

    Only the attributes the sections actually reach for. A stub rather than a
    real `Paths`, because the point is to present directories that do not
    exist, which `load_paths` exists to refuse.
    """

    def __init__(self, home: pathlib.Path):
        self.cohort_home = home / "cohort"
        self.cohort_home.mkdir(parents=True)
        self.output_home = home
        self.sites = ("TESTSITE",)
        self.model_key = "GEM-all"
        self.has_tokenizer = False

    def upload_home(self, site):
        return self.output_home / f"upload_to_box_{site}"

    def raw_subset(self, site):
        return self.output_home / "scratch" / site

    def processed(self, site):
        return self.output_home / "scratch" / site

    def anchored(self, site, anchor):
        return self.processed(site) / f"anchor-{anchor}"


class Quiet:
    def info(self, *_):
        pass

    def warning(self, *_):
        pass


@pytest.fixture
def empty(tmp_path):
    """the empty run as a source, which is what the sections now take"""
    return report.LocalRun(EmptyRun(tmp_path), Quiet())


SECTIONS = (
    "section_provenance",
    "section_cohort",
    "section_phenotypes",
    "section_baseline",
    "section_tokens",
    "section_trials",
    "section_export",
    "section_suppression",
)


@pytest.mark.parametrize("name", SECTIONS)
def test_every_section_renders_against_an_empty_run(empty, name):
    """
    a section whose inputs are missing says so instead of raising

    This is the state a contributing site is in before its first run. A
    traceback here would be the first thing they see from this repository.
    """
    out: list[str] = []
    getattr(report, name)(empty, out)
    body = "".join(out)
    assert body, f"{name} rendered nothing at all"
    assert "<h2" in body, f"{name} did not emit its heading"
    # `suppression` is the one section that legitimately finds nothing to say
    # beyond its heading when there are no CSVs; everything else must name
    # what is missing.
    assert "Not present" in body or name == "section_suppression"


def test_stage_ledger_names_every_stage_even_with_no_output(empty):
    """
    the ledger is the spine of the report and is never partial

    A stage missing from the table reads as a stage that does not exist, which
    is the one thing a run report must not do.
    """

    class Driver:
        STAGES = ("cohort", "phenotypes", "subset", "derive", "collate",
                  "tokenize", "winnow", "tokencounts", "baseline", "trials",
                  "export")
        STAGE_PURPOSE = {"cohort": "identifying the shock cohort"}
        TOKEN_STAGES = ("tokenize", "winnow", "tokencounts")

    out: list[str] = []
    report.section_stages(empty, Driver(), out, Quiet())
    body = "".join(out)
    for stage in Driver.STAGES:
        assert f"{stage}</td>" in body or f". {stage}" in body, stage
    assert "11 of 11 stages have no output" in body
    # and it must say why the token stages are empty, rather than leaving a
    # reader to conclude the site charts nothing
    assert "No training-time tokenizer" in body


def test_stage_ledger_separates_stages_that_share_a_directory(empty):
    """
    `subset` and `derive` both write into raw_subset

    Globbing the directory made each report all 52 of the other's files, which
    says nothing about either. The derived tables are named and subtracted.
    """
    home = empty.paths.raw_subset("TESTSITE")
    home.mkdir(parents=True)
    for name in (
        "clif_vitals.parquet",
        "clif_labs.parquet",
        "clif_sofa.parquet",
        "clif_medication_admin_continuous_converted.parquet",
        "clif_respiratory_support_processed.parquet",
    ):
        (home / name).write_bytes(b"")

    subset, _ = empty.stage_files("subset")
    derive, _ = empty.stage_files("derive")
    assert {f.name for f in subset} == {"clif_vitals.parquet", "clif_labs.parquet"}
    assert len(derive) == 3
    assert not set(subset) & set(derive)

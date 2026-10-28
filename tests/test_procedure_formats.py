#!/usr/bin/env python3

"""
tests the per-format coverage warning on clif_patient_procedures

UCMC's table carries 1.72M distinct ids under procedure_code_format "cpt" and
none of them key on hospitalization_id; only the ICD-10-PCS rows do. A code-set
query over that table returns zero from the CPT arm and a plausible number from
the PCS arm, with nothing saying half the search found nothing -- so the check
exists to turn that into a warning.

It used to build a Python set of every distinct id per format and re-collect
the table once per format. These tests pin the behaviour the streaming rewrite
has to preserve: which formats warn, and the exact numbers in the message.
"""

from __future__ import annotations

import pathlib
import sys

import polars as pl
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from clifcs.cohort import _warn_unjoinable_formats  # noqa: E402


class Recorder:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def info(self, *_):
        pass

    def warning(self, message, *_):
        self.warnings.append(message)


def make_procedures(home: pathlib.Path, rows: list[tuple[str, str]]) -> pathlib.Path:
    home.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "hospitalization_id": [r[0] for r in rows],
            "procedure_code_format": [r[1] for r in rows],
            "procedure_code": ["X" for _ in rows],
        }
    ).write_parquet(home / "clif_patient_procedures.parquet")
    return home


def cohort(ids) -> pl.DataFrame:
    return pl.DataFrame({"hospitalization_id": list(ids)})


def test_an_unjoinable_format_warns_with_its_id_count(tmp_path):
    """the UCMC shape: one format keyed on something else entirely"""
    rows = [(f"h{i}", "icd10pcs") for i in range(50)]
    rows += [(f"order{i}", "cpt") for i in range(400)]
    home = make_procedures(tmp_path / "data", rows)

    logger = Recorder()
    _warn_unjoinable_formats(home, cohort(f"h{i}" for i in range(50)), logger)

    assert len(logger.warnings) == 1
    said = logger.warnings[0]
    assert "'cpt'" in said
    assert "has 400 ids" in said
    assert "matches 0.0% of the cohort" in said


def test_a_joinable_format_is_silent(tmp_path):
    rows = [(f"h{i}", "icd10pcs") for i in range(50)]
    home = make_procedures(tmp_path / "data", rows)

    logger = Recorder()
    _warn_unjoinable_formats(home, cohort(f"h{i}" for i in range(50)), logger)

    assert logger.warnings == []


def test_coverage_is_measured_against_the_cohort_not_the_table(tmp_path):
    """
    the denominator is the cohort, so a format covering every id it has still
    warns when the cohort is much larger -- which is the case that matters,
    since a format keyed on orders has plenty of ids and none of them help
    """
    rows = [(f"h{i}", "icd10pcs") for i in range(1)]
    rows += [(f"h{i}", "cpt") for i in range(1)]
    home = make_procedures(tmp_path / "data", rows)

    logger = Recorder()
    _warn_unjoinable_formats(home, cohort(f"h{i}" for i in range(500)), logger)

    # 1 of 500 is 0.2%, under the 1% threshold, for both formats
    assert len(logger.warnings) == 2


def test_the_threshold_is_one_percent(tmp_path):
    """just over and just under, so the boundary cannot drift unnoticed"""
    for matched, expect_warning in ((4, True), (6, False)):
        rows = [(f"h{i}", "icd10pcs") for i in range(matched)]
        home = make_procedures(tmp_path / f"data{matched}", rows)
        logger = Recorder()
        _warn_unjoinable_formats(home, cohort(f"h{i}" for i in range(500)), logger)
        assert bool(logger.warnings) is expect_warning, (matched, logger.warnings)


def test_duplicate_rows_do_not_inflate_the_id_count(tmp_path):
    """
    the count is DISTINCT ids, as the Python-set version measured it -- a
    patient with forty procedures is one id, and counting rows instead would
    make an unjoinable format look larger than it is
    """
    rows = [("h0", "icd10pcs")] * 40 + [(f"order{i}", "cpt") for i in range(30)] * 3
    home = make_procedures(tmp_path / "data", rows)

    logger = Recorder()
    _warn_unjoinable_formats(home, cohort(f"h{i}" for i in range(500)), logger)

    cpt = [m for m in logger.warnings if "'cpt'" in m]
    assert len(cpt) == 1
    assert "has 30 ids" in cpt[0]

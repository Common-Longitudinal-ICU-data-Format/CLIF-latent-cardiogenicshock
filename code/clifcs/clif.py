#!/usr/bin/env python3

"""
thin CLIF-2.1 access layer

The UCMC and NU extracts differ in ways that break naive joins: timestamps carry
different time units (us vs ns) though both are UTC-aware, NU carries a pandas
`__index_level_0__` column, and some tables type hospitalization_id as Int64
while the rest use String. Everything here funnels through `scan` so those
differences are normalized in exactly one place.
"""

from __future__ import annotations

import pathlib

import duckdb
import polars as pl

# ---------------------------------------------------------------------------
# DuckDB reads TIMESTAMPTZ in the LOCAL SYSTEM TIMEZONE unless told otherwise.
#
# clifpy's dose conversion goes through DuckDB and returns pandas via .to_df(),
# and on that path a TIMESTAMPTZ column comes back shifted into whatever
# timezone the machine happens to be set to. Two sites in different timezones
# -- or one cluster whose nodes are set to UTC while a workstation is set to
# local -- then produce medication timestamps that disagree by hours, which
# flows straight into tokenization and into the embeddings as a site effect
# that looks exactly like a difference in practice.
#
# This line was previously carried by a fork of clifpy pinned in
# requirements.txt. That fork has since been deleted from GitHub, so the
# one-line fix lives here instead, where it is visible and attributable. It
# must run before clifpy's unit_converter is imported; clifcs is imported at
# the top of every entry point and clifpy only inside the derive stage, so it
# does.
#
# Upstream clifpy does not set this at all, so nothing resets it afterwards.
#
# Called from clifcs/__init__.py so that it runs on `import clifcs`, which
# every entry point does at module level, long before the derive stage
# imports clifpy. Defining it here and calling it there rather than running
# it at this module's import, because clifcs/__init__ does not otherwise
# import clif.py -- the first version of this fix sat at module level and
# silently never ran.
def set_duckdb_utc() -> None:
    """pin the default DuckDB connection to UTC; see the note above"""
    duckdb.sql("SET TimeZone = 'UTC'")


def duckdb_timezone() -> str:
    """the default connection's timezone, for the startup assertion in 01"""
    return str(duckdb.sql("SELECT current_setting('TimeZone')").fetchone()[0])

#: tables that key on hospitalization_id
#:
#: clif_ecmo_mcs is deliberately absent: it is a concept table at both sites and
#: is not reliably populated (it names 82% of all UCMC hospitalizations and
#: carries no device columns). Mechanical circulatory support comes from
#: clif_patient_procedures instead -- see clifcs.cohort._mcs_procedures.
HOSPITALIZATION_LEVEL = (
    "clif_adt",
    "clif_crrt_therapy",
    "clif_hospital_diagnosis",
    "clif_hospitalization",
    "clif_labs",
    "clif_medication_admin_continuous",
    "clif_medication_admin_intermittent",
    "clif_patient_assessments",
    "clif_patient_procedures",
    "clif_position",
    "clif_respiratory_support",
    "clif_vitals",
)

#: tables that key on patient_id only
PATIENT_LEVEL = ("clif_patient", "clif_code_status")


def utc_naive(frame: pl.LazyFrame, col: str) -> pl.Expr:
    """
    expression casting `col` to a timezone-naive UTC datetime in microseconds

    Mirrors cocoa.collator.Collator.to_utc_naive so that the timestamps used for
    cohort selection are the same instants cocoa later tokenizes. Timezone-naive
    columns are assumed to already be UTC.
    """
    dtype = frame.collect_schema().get(col)
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return (
            pl.col(col)
            .dt.convert_time_zone("UTC")
            .dt.replace_time_zone(None)
            .cast(pl.Datetime("us"))
        )
    return pl.col(col).cast(pl.Datetime("us"))


def table_path(data_dir: pathlib.Path, table: str) -> pathlib.Path:
    return pathlib.Path(data_dir) / f"{table}.parquet"


def has_table(data_dir: pathlib.Path, table: str) -> bool:
    return table_path(data_dir, table).is_file()


def scan(
    data_dir: pathlib.Path,
    table: str,
    *,
    dttm_cols: tuple[str, ...] = (),
) -> pl.LazyFrame:
    """
    lazily scan a CLIF table with cross-site differences normalized:
    drops pandas index columns, casts id columns to String, and converts the
    named datetime columns to naive UTC microseconds
    """
    frame = pl.scan_parquet(table_path(data_dir, table)).drop(
        "__index_level_0__", strict=False
    )
    schema = frame.collect_schema().names()

    casts = {
        c: pl.String for c in ("hospitalization_id", "patient_id") if c in schema
    }
    if casts:
        frame = frame.cast(casts)

    missing = [c for c in dttm_cols if c not in schema]
    if missing:
        raise KeyError(f"{table} is missing datetime column(s) {missing} in {data_dir}")
    if dttm_cols:
        frame = frame.with_columns(utc_naive(frame, c).alias(c) for c in dttm_cols)

    return frame

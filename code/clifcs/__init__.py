"""Shared helpers for the CLIF cardiogenic-shock latent-space pipeline."""

from clifcs.clif import duckdb_timezone, set_duckdb_utc
from clifcs.config import (
    ANCHORS,
    MAX_PLOTTED_SITES,
    ROLES,
    Paths,
    RoleViolation,
    get_logger,
    load_cohort_cfg,
    load_paths,
)

# Runs on `import clifcs`, which happens at the top of every entry point.
# DuckDB otherwise reads TIMESTAMPTZ in the local system timezone, and
# clifpy's dose conversion returns pandas through DuckDB -- so without this
# two sites with different system clocks disagree about when every infusion
# happened. See the note in clifcs/clif.py.
set_duckdb_utc()

__all__ = [
    "ANCHORS",
    "duckdb_timezone",
    "set_duckdb_utc",
    "MAX_PLOTTED_SITES",
    "ROLES",
    "Paths",
    "RoleViolation",
    "get_logger",
    "load_cohort_cfg",
    "load_paths",
]

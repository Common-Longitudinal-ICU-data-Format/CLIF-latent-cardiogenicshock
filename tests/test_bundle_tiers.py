"""
what a tier-1 re-export may and may not touch, and the stage ledger

Both exist because of the same failure: `01_site_tables.py --stage export`
rewrites a directory that `02_site_embeddings.py` also writes into, and used to
clear it with a recursive delete. Re-exporting to pick up one newly produced
table removed every embedding the site had computed, and left a manifest
denying patient-level data beside a patient/ directory holding it.
"""

from __future__ import annotations

import importlib.util
import logging
import pathlib
import sys

import polars as pl
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from clifcs.export import writer  # noqa: E402


@pytest.fixture
def logger():
    return logging.getLogger("test_bundle_tiers")


def driver():
    """01_site_tables.py, whose name is not an identifier"""
    path = ROOT / "code" / "01_site_tables.py"
    spec = importlib.util.spec_from_file_location("clifcs_driver_01_tiers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bundle(home: pathlib.Path) -> pathlib.Path:
    """a bundle holding both tiers, as it looks after 01 and then 02 have run"""
    bundle = home / "upload_to_box_SITE"
    (bundle / "counts").mkdir(parents=True)
    (bundle / "trials").mkdir(parents=True)
    (bundle / "patient").mkdir(parents=True)
    (bundle / "purity").mkdir(parents=True)

    (bundle / "manifest.json").write_text('{"site": "SITE"}', encoding="utf-8")
    (bundle / "token_levels.csv").write_text("rank\n1\n", encoding="utf-8")
    (bundle / "counts/filter_ledger.csv").write_text("step\na\n", encoding="utf-8")
    (bundle / "counts/withdrawn_table.csv").write_text("x\n1\n", encoding="utf-8")
    (bundle / "trials/trial_criteria.csv").write_text("trial\nx\n", encoding="utf-8")

    (bundle / "counts/token_coverage.csv").write_text("v\n1\n", encoding="utf-8")
    (bundle / "patient/embeddings_anchor-24h.parquet").write_bytes(b"PAR1")
    (bundle / "purity/phenotype_purity.csv").write_text("scheme\nzweck\n", encoding="utf-8")
    return bundle


def test_clear_tier1_keeps_the_patient_and_purity_directories(tmp_path, logger):
    bundle = make_bundle(tmp_path)
    writer.clear_tier1(bundle, logger)
    assert (bundle / "patient/embeddings_anchor-24h.parquet").is_file()
    assert (bundle / "purity/phenotype_purity.csv").is_file()


def test_clear_tier1_keeps_token_coverage_despite_living_in_counts(tmp_path, logger):
    # the one tier-2 file that sits inside a tier-1 glob
    bundle = make_bundle(tmp_path)
    writer.clear_tier1(bundle, logger)
    assert (bundle / "counts/token_coverage.csv").is_file()


def test_clear_tier1_keeps_the_manifest_so_tier_2_fields_can_be_read_back(
    tmp_path, logger
):
    bundle = make_bundle(tmp_path)
    writer.clear_tier1(bundle, logger)
    assert (bundle / "manifest.json").is_file()


@pytest.mark.parametrize(
    "relative",
    [
        "token_levels.csv",
        "counts/filter_ledger.csv",
        # the reason a clear happens at all: a table this code no longer
        # produces must not travel forever
        "counts/withdrawn_table.csv",
        "trials/trial_criteria.csv",
    ],
)
def test_clear_tier1_removes_what_tier_1_rewrites(tmp_path, logger, relative):
    bundle = make_bundle(tmp_path)
    writer.clear_tier1(bundle, logger)
    assert not (bundle / relative).exists()


def test_clear_tier1_on_a_bundle_that_does_not_exist_is_not_an_error(
    tmp_path, logger
):
    writer.clear_tier1(tmp_path / "never_written", logger)


def test_token_coverage_is_the_only_tier_2_file_inside_a_tier_1_glob():
    # if 02 starts writing a second aggregate into counts/, this is where the
    # omission shows up rather than in a deleted file at a site
    assert writer.TIER2_IN_TIER1_PATHS == frozenset({"counts/token_coverage.csv"})


# ------------------------------------------------------------ the stage ledger


class FakePaths:
    def __init__(self, home: pathlib.Path):
        self.cohort_home = home


def test_ledger_merges_with_a_previous_invocation(tmp_path, logger):
    module = driver()
    paths = FakePaths(tmp_path)

    module.STAGE_LEDGER[:] = [
        {
            "stage": "cohort",
            "status": "ok",
            "seconds": 94.0,
            "peak_rss_mb": 1024.0,
            "finished_utc": "A",
        },
        {"stage": "baseline", "status": "ok", "seconds": 3.0, "finished_utc": "A"},
    ]
    module.write_ledger(paths)

    # a later `--stage export`, which knows nothing about the cohort run
    module.STAGE_LEDGER[:] = [
        {"stage": "export", "status": "ok", "seconds": 6.0, "finished_utc": "B"},
    ]
    module.write_ledger(paths)

    ledger = pl.read_csv(tmp_path / "run_stages.csv")
    assert ledger["stage"].to_list() == ["cohort", "baseline", "export"]


def test_ledger_replaces_the_row_for_a_rerun_stage(tmp_path, logger):
    module = driver()
    paths = FakePaths(tmp_path)

    module.STAGE_LEDGER[:] = [
        {"stage": "cohort", "status": "failed", "seconds": 2.0, "finished_utc": "A"},
    ]
    module.write_ledger(paths)
    module.STAGE_LEDGER[:] = [
        {"stage": "cohort", "status": "ok", "seconds": 94.0, "finished_utc": "B"},
    ]
    module.write_ledger(paths)

    ledger = pl.read_csv(tmp_path / "run_stages.csv")
    assert ledger.height == 1
    assert ledger["status"].to_list() == ["ok"]


def test_ledger_carries_no_counts_and_no_paths(tmp_path, logger):
    # the whole reason it may travel where the log may not
    module = driver()
    paths = FakePaths(tmp_path)
    module.STAGE_LEDGER[:] = [
        {
            "stage": "cohort",
            "status": "ok",
            "seconds": 94.0,
            "peak_rss_mb": 1024.0,
            "finished_utc": "A",
        },
    ]
    module.write_ledger(paths)
    assert set(pl.read_csv(tmp_path / "run_stages.csv").columns) == {
        "stage",
        "status",
        "seconds",
        "peak_rss_mb",
        "finished_utc",
    }


def test_ledger_merges_a_file_written_before_peak_memory_was_recorded(
    tmp_path, logger
):
    """
    a site that ran an earlier version has a run_stages.csv with four columns

    polars takes its schema from the first row of a list of dicts, so merging
    rows with different keys either drops the new column or raises, depending
    which invocation came first. Every row is widened to the union instead.
    """
    module = driver()
    paths = FakePaths(tmp_path)

    (tmp_path / "run_stages.csv").write_text(
        "stage,status,seconds,finished_utc\ncohort,ok,94.0,A\n", encoding="utf-8"
    )
    module.STAGE_LEDGER[:] = [
        {
            "stage": "export",
            "status": "ok",
            "seconds": 6.0,
            "peak_rss_mb": 812.5,
            "finished_utc": "B",
        },
    ]
    module.write_ledger(paths)

    ledger = pl.read_csv(tmp_path / "run_stages.csv")
    assert ledger["stage"].to_list() == ["cohort", "export"]
    assert ledger.filter(pl.col("stage") == "cohort")["peak_rss_mb"][0] is None
    assert ledger.filter(pl.col("stage") == "export")["peak_rss_mb"][0] == 812.5


# ----------------------------------- export must not clear what it no longer
# ----------------------------------- rewrites


class BundlePaths(FakePaths):
    """enough of Paths for `write_aggregate` to reach its own guard

    `has_tokenizer` is false so the call does not need a real tokenizer.yaml:
    the check under test sits before that branch.
    """

    def __init__(self, home: pathlib.Path, bundle: pathlib.Path):
        super().__init__(home)
        self._bundle = bundle
        self.has_tokenizer = False

    def upload_home(self, site: str) -> pathlib.Path:
        return self._bundle


def test_export_does_not_clear_the_tables_it_no_longer_writes():
    """
    THE REGRESSION, and the reason the five tests above could not catch it.

    They check `clear_tier1` in isolation: it deletes counts/*.csv and
    trials/*.csv and spares tier 2. All true, and all correct while export
    REGENERATED those tables from output/cohort/. Once `put_table` began
    writing the only copy straight into the bundle as each stage produced it,
    and `write_aggregate` merely re-checked what was there, the same call
    became a plain delete: export emptied counts/ and trials/, `check_bundle`
    globbed two empty directories and passed, and the run reported success
    with a three-file bundle.

    Asserted against the source because the SEQUENCE was wrong, not the
    function -- `clear_tier1` is still correct, and still tested above.
    """
    source = (ROOT / "code" / "01_site_tables.py").read_text(encoding="utf-8")
    export = source.split("def stage_export(")[1].split("\ndef ")[0]
    # comments stripped: the fix leaves a comment saying what used to stand
    # here and naming the function, so a bare substring test matches its own
    # explanation and fails on the fixed code
    code = "\n".join(
        line for line in export.splitlines() if not line.lstrip().startswith("#")
    )

    assert "clear_tier1" not in code, (
        "stage_export clears tier-1 tables again. Nothing in export rewrites "
        "them -- put_table authors them during the earlier stages -- so "
        "clearing them here empties the bundle."
    )


def test_a_bundle_with_no_aggregate_tables_is_refused(tmp_path, logger):
    """
    the tripwire that makes the failure above loud instead of silent

    `check_bundle` validates the files it finds and has no notion of what
    ought to be there, so an emptied counts/ is indistinguishable from a clean
    one. `filter_ledger.csv` is written for every site unconditionally, so an
    empty counts/ means the tables are not here.
    """
    from clifcs.export import contract

    bundle = make_bundle(tmp_path)
    for stale in (bundle / "counts").glob("*.csv"):
        stale.unlink()

    with pytest.raises(contract.ContractViolation, match="nothing to send"):
        writer.write_aggregate(
            BundlePaths(tmp_path, bundle), "SITE", bundle, "salt", 11, 0, logger
        )


def test_a_populated_bundle_passes_the_tripwire(tmp_path, logger):
    """
    the guard must not fire on the case it exists to let through -- a bundle
    whose counts/ holds tables gets past it and on to the rest of the export
    """
    from clifcs.export import contract

    bundle = make_bundle(tmp_path)
    try:
        writer.write_aggregate(
            BundlePaths(tmp_path, bundle), "SITE", bundle, "salt", 11, 0, logger
        )
    except contract.ContractViolation as exc:
        assert "nothing to send" not in str(exc), "the tripwire fired on a real bundle"

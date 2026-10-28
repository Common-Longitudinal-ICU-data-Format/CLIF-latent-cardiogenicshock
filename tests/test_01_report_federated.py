#!/usr/bin/env python3

"""
the run report read from bundles rather than from a cohort directory

A coordinating site holds no `output/cohort/`. It holds what the sites
uploaded, and a bundle is not a copy of the cohort directory: the export stage
strips the `_<SITE>` infix out of every filename, sorts the tables into
counts/ and trials/, and leaves behind everything the contract does not carry.

Two failure modes are worth pinning, because both produce a report that looks
fine and is wrong:

  a table the contract does not carry, read as a stage that did not run. The
  five scratch stages are the honest version of this; the phenotype centroids
  are the dangerous one, because their absence is what the report used to take
  as proof that a scheme had been transferred rather than refit.

  a row attributed to the wrong site. Four tables carry no `site` column at
  all, and the directory a bundle arrives in can be renamed in transit, so the
  attribution has to come from the manifest.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import pathlib
import re
import sys

import polars as pl
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def load_report():
    path = ROOT / "code" / "01_report.py"
    spec = importlib.util.spec_from_file_location("report_01_federated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_report()


class Quiet:
    def info(self, *_):
        pass

    def warning(self, *_):
        pass


def write_bundle(home: pathlib.Path, directory: str, site: str, *, tokenizer=True):
    """one tier-1 bundle, shaped the way the export stage writes it"""
    bundle = home / directory
    (bundle / "counts").mkdir(parents=True)
    (bundle / "trials").mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "site": site,
                "tokenizer_sha256": "abc" if tokenizer else None,
                "min_cell": 11,
                "shares_patient_level": False,
            }
        ),
        encoding="utf-8",
    )
    # carries a site column
    pl.DataFrame(
        {
            "site": [site],
            "step": ["all hospitalizations"],
            "n_hospitalizations": [2000],
            "n_patients": [1234],
            "note": [""],
        }
    ).write_csv(bundle / "counts/filter_ledger.csv")
    # carries none -- the case that needs the manifest
    pl.DataFrame({"variable": ["crp"], "n_measured": [900]}).write_csv(
        bundle / "counts/seymour_availability.csv"
    )
    pl.DataFrame(
        {
            "site": [site],
            "anchor": ["24h"],
            "scheme": ["zweck"],
            "level": ["I"],
            "trial": ["SHOCK"],
            "n": [40],
        }
    ).write_csv(bundle / "trials/trial_outcomes_anchor-24h.csv")
    pl.DataFrame(
        {"site": [site], "shock_type": ["septic"], "n": [900],
         "n_total": [3000], "share": [0.3],
         "mortality": [0.3], "median_icu_los_days": [4.0]}
    ).write_csv(bundle / "counts/shock_type_summary.csv")
    # schema 3.1: on a binary row `n_measured` is the DENOMINATOR and
    # `n_positive` the numerator. It used to hold the numerator with the
    # denominator unexported, which is why the share was uncheckable.
    pl.DataFrame(
        {"site": [site, site], "shock_type": ["septic", "septic"],
         "variable": ["Age (years)", "Female"],
         "kind": ["continuous", "binary"], "n": [900, 900],
         "n_measured": [900, 900], "n_positive": [None, 423],
         "median": [62.0, None],
         "q1": [50.0, None], "q3": [71.0, None],
         "share_positive": [None, 423 / 900]}
    ).write_csv(bundle / "counts/baseline_by_shock_type.csv")
    # the three summaries the fit-provenance section keys off
    pl.DataFrame(
        {"cs_phenotype": ["I non-congested"], "n": [600], "mortality": [0.11],
         "median_icu_los_days": [3.3]}
    ).write_csv(bundle / "counts/phenotype_summary.csv")
    for scheme, level in (("seymour", "alpha"), ("bhavani", "A")):
        pl.DataFrame(
            {scheme: [level], "n": [400], "n_total": [1000], "share": [0.4],
             "mortality": [0.3], "median_icu_los_days": [5.0]}
        ).write_csv(bundle / f"counts/{scheme}_summary.csv")
    if tokenizer:
        (bundle / "token_levels.csv").write_text("rank\n1\n", encoding="utf-8")
        pl.DataFrame(
            {
                "site": [site],
                "anchor": ["24h"],
                "family": ["labs"],
                "n_patients": [500],
                "prevalence": [0.5],
                "total_occurrences": [9000],
                "distinct_tokens": [40],
                "n_cohort": [1000],
            }
        ).write_csv(bundle / "counts/token_families_anchor-24h.csv")
    return bundle


@pytest.fixture
def received(tmp_path):
    """
    two bundles, as a coordinator receives them

    Deliberately awkward in the two ways real ones are: the directory name does
    not match the site name, and one site held no tokenizer.
    """
    write_bundle(tmp_path, "upload_to_box_rush", "rush", tokenizer=False)
    write_bundle(tmp_path, "upload_to_box_UCMC", "UCMC", tokenizer=True)
    return report.ReceivedBundles(tmp_path, Quiet())


def test_the_site_comes_from_the_manifest_not_the_directory(received):
    # `upload_to_box_rush` holds `"site": "rush"`; a directory can be renamed in
    # transit and the site that produced the data is a property of the data
    assert set(received.sites) == {"rush", "UCMC"}


def test_a_directory_without_a_manifest_is_skipped(tmp_path):
    write_bundle(tmp_path, "upload_to_box_UCMC", "UCMC")
    (tmp_path / "upload_to_box_NOWHERE").mkdir()
    (tmp_path / "notabundle").mkdir()
    source = report.ReceivedBundles(tmp_path, Quiet())
    assert set(source.sites) == {"UCMC"}


def test_tables_are_stacked_across_bundles(received):
    ledger = received.read("filter_ledger.csv")
    assert ledger.height == 2
    assert set(ledger["site"].to_list()) == {"rush", "UCMC"}


def test_a_table_with_no_site_column_gets_one_from_the_manifest(received):
    # seymour_availability.csv is `variable, n_measured` and nothing else, so
    # without this two bundles stack into rows that cannot be told apart
    availability = received.read("seymour_availability.csv")
    assert "site" in availability.columns
    assert sorted(availability["site"].to_list()) == ["UCMC", "rush"]


def test_the_anchor_is_recovered_from_the_filename_where_a_table_lacks_it(tmp_path):
    bundle = write_bundle(tmp_path, "upload_to_box_UCMC", "UCMC")
    pl.DataFrame({"token_id": [1], "n_patients": [12], "site": ["UCMC"]}).write_csv(
        bundle / "counts/token_prevalence_anchor-onset.csv"
    )
    source = report.ReceivedBundles(tmp_path, Quiet())
    assert source.read("token_prevalence_*.csv")["anchor"].to_list() == ["onset"]


@pytest.mark.parametrize(
    "pattern",
    [
        "phenotype_centroids_*.csv",
        "seymour_centroids.csv",
        "seymour_agreement.csv",
        "phenotype_variant_agreement.csv",
        "phenotype_summary_cs_phenotype.csv",
    ],
)
def test_a_table_the_contract_omits_reads_as_absent_not_as_an_error(
    received, pattern
):
    assert received.read(pattern) is None
    assert received.has(pattern) is False


def test_an_unknown_pattern_raises_rather_than_reading_as_a_missing_stage(received):
    # the point of the guard: a section added later that reads a new table must
    # not silently render "the stage did not run" at a coordinating site
    with pytest.raises(KeyError):
        received.read("something_new.csv")


def test_every_pattern_the_sections_read_is_mapped():
    """
    no literal pattern in the report is missing from both maps

    Scanned out of the source rather than listed, so adding a `source.read`
    without an entry fails here instead of at a coordinating site.
    """
    text = (ROOT / "code" / "01_report.py").read_text(encoding="utf-8")
    literals = set(re.findall(r'source\.(?:read|has)\("([^"{]+)"\)', text))
    assert literals, "the scan found nothing, so it is no longer a guard"
    unmapped = [
        pattern
        for pattern in literals
        if pattern not in report.BUNDLE_LAYOUT
        and not any(fnmatch.fnmatch(pattern, p) for p in report.NOT_SHIPPED)
    ]
    assert not unmapped, f"unmapped in BUNDLE_LAYOUT and NOT_SHIPPED: {unmapped}"


def test_the_tokenizer_split_is_per_site(received):
    with_tokenizer, without = received.tokenizer_sites()
    assert with_tokenizer == ("UCMC",)
    assert without == ("rush",)


def test_the_scratch_stages_report_no_bundle_evidence(received):
    for stage in ("subset", "derive", "collate", "tokenize", "winnow"):
        files, where = received.stage_files(stage)
        assert files == []
        assert "scratch" in where


def test_a_stage_that_did_ship_is_found(received):
    files, where = received.stage_files("cohort")
    assert {f.name for f in files} == {
        "filter_ledger.csv",
        "seymour_availability.csv",
        "shock_type_summary.csv",
        "seymour_summary.csv",
        "bhavani_summary.csv",
    }
    assert where == "upload_to_box_<SITE>"


def test_fit_provenance_does_not_claim_transfer_from_a_bundle(received):
    """
    the regression this section exists to prevent

    The centroid table is the evidence and it never ships, so reading its
    absence as proof of transfer reports every bundle as transferred --
    including one from a site that refit, which is the exact failure the
    section was added to catch.
    """
    out: list[str] = []
    report.section_fit_provenance(received, out)
    body = "".join(out)
    assert "Not answerable from a bundle" in body
    assert "labels came from" not in body, "the verdict table was rendered"


def test_the_suppression_audit_reads_the_copies_that_arrived(received):
    names = {name for name, _ in received.audit()}
    # labelled by site, because two bundles hold files of the same name
    assert "UCMC/counts/filter_ledger.csv" in names
    assert "rush/counts/filter_ledger.csv" in names


@pytest.mark.parametrize(
    "name",
    [
        "section_provenance",
        "section_cohort",
        "section_phenotypes",
        "section_baseline",
        "section_tokens",
        "section_trials",
        "section_export",
        "section_suppression",
    ],
)
def test_every_section_renders_from_bundles(received, name):
    out: list[str] = []
    getattr(report, name)(received, out)
    body = "".join(out)
    assert body, f"{name} rendered nothing at all"
    assert "<h2" in body, f"{name} did not emit its heading"

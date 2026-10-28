#!/usr/bin/env python3

"""
tests that each shipped phenotype model fills in on its own

01 defaults the three `--*-model` flags to the files in `models/`, because a
contributing site is supposed to APPLY the shared models rather than refit --
refitting makes "phenotype II", "beta" and "group A" name a different
partition here than at every other site, and the pooled breakdowns stop
meaning anything.

The default used to be gated on

    transferred = args.phenotype_model or args.seymour_model or ...

which is true as soon as ANY one flag is passed. So naming one model skipped
the defaults for the other two: `--phenotype-model mine.json` on its own left
both sepsis flags empty and refit Seymour and Bhavani. That is the failure the
default was added to prevent, reached by an invocation that looks more
careful, not less -- and the reason these tests assert per scheme rather than
on a single summary flag.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def driver():
    """01_site_tables.py, whose name is not an identifier"""
    path = ROOT / "code" / "01_site_tables.py"
    spec = importlib.util.spec_from_file_location("clifcs_driver_01_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return driver()


@pytest.fixture
def shipped(tmp_path, mod):
    """a models/ directory holding all three files"""
    home = tmp_path / "models"
    home.mkdir()
    for filename in mod.SHIPPED_MODELS.values():
        (home / filename).write_text("{}", encoding="utf-8")
    return home


def args_with(**kwargs):
    base = dict(phenotype_model=None, seymour_model=None, bhavani_model=None)
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_nothing_named_fills_all_three(mod, shipped):
    args = args_with()
    named, filled, absent = mod.resolve_shipped_models(args, shipped)
    assert named == []
    assert filled == list(mod.SHIPPED_MODELS)
    assert absent == []
    for attr in mod.SHIPPED_MODELS:
        assert getattr(args, attr), attr


@pytest.mark.parametrize("own", sorted({"phenotype_model", "seymour_model", "bhavani_model"}))
def test_naming_one_model_still_fills_the_others(mod, shipped, own, tmp_path):
    """the bug: one flag used to suppress the defaults for the other two"""
    mine = tmp_path / "mine.json"
    mine.write_text("{}", encoding="utf-8")
    args = args_with(**{own: str(mine)})

    named, filled, absent = mod.resolve_shipped_models(args, shipped)

    assert named == [own]
    assert absent == []
    assert set(filled) == set(mod.SHIPPED_MODELS) - {own}
    # the named one is left exactly as passed ...
    assert getattr(args, own) == str(mine)
    # ... and the other two now point into models/, so neither refits
    for attr in filled:
        assert pathlib.Path(getattr(args, attr)).parent == shipped


def test_a_missing_shipped_file_is_reported_not_silently_refit(mod, shipped):
    (shipped / mod.SHIPPED_MODELS["bhavani_model"]).unlink()
    args = args_with()

    named, filled, absent = mod.resolve_shipped_models(args, shipped)

    assert [attr for attr, _ in absent] == ["bhavani_model"]
    assert set(filled) == {"phenotype_model", "seymour_model"}
    assert args.bhavani_model is None
    # the caller needs the path to name in the error it raises
    assert absent[0][1] == shipped / mod.SHIPPED_MODELS["bhavani_model"]


def test_a_named_model_does_not_need_a_shipped_file(mod, tmp_path):
    """a site may point the flags at its own copies with no models/ at all"""
    mine = tmp_path / "mine.json"
    mine.write_text("{}", encoding="utf-8")
    args = args_with(
        phenotype_model=str(mine), seymour_model=str(mine), bhavani_model=str(mine)
    )

    named, filled, absent = mod.resolve_shipped_models(args, tmp_path / "absent")

    assert named == list(mod.SHIPPED_MODELS)
    assert filled == []
    assert absent == []


def test_every_scheme_the_cohort_stage_transfers_is_registered(mod):
    """
    SHIPPED_MODELS is the list main() logs and errors from, so a scheme missing
    from it is a scheme that refits with nothing in the log naming it.
    """
    assert set(mod.SHIPPED_MODELS) == {
        "phenotype_model",
        "seymour_model",
        "bhavani_model",
    }
    for filename in mod.SHIPPED_MODELS.values():
        assert (ROOT / "models" / filename).is_file(), filename

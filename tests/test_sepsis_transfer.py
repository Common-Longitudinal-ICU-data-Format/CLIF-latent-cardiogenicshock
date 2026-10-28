#!/usr/bin/env python3

"""
tests that the shipped sepsis phenotype models can actually be transferred

A contributing site is supposed to APPLY `models/seymour_model.json` and
`models/bhavani_model.json` rather than refit, so that "beta" and "group A"
name the same partition everywhere. Two separate things have to hold for that,
and only one of them was ever exercised:

  the model files are complete and applicable -- they name their scheme, their
  features, the transform and the moments, and `apply_model` can place a
  patient with them

  01 actually hands them to the cohort stage

The second failed silently for a while: `sepsis_models` was built from
`--seymour-model` / `--bhavani-model` before the block that fills those flags
in from `models/`, so every run that relied on the default passed an empty
mapping and both schemes refit. The log said the shipped models were being
applied, and the only evidence to the contrary was the word "refit" in one
line among warnings that read as findings about the cohort.

These tests cover the first. The second is now visible in two places that do
not depend on each other: 01 logs one line per model as it applies it, and
`01_report.py` reports a scheme as refit whenever a `*_centroids.csv` -- which
01 writes only on a fit -- is sitting beside the summary.
"""

from __future__ import annotations

import json
import pathlib
import sys

import polars as pl
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from clifcs.sepsis_phenotype import (  # noqa: E402
    BHAVANI_LABELS, SEYMOUR_LABELS, apply_model,
)

MODELS = ROOT / "models"
SCHEMES = (
    ("seymour", MODELS / "seymour_model.json", SEYMOUR_LABELS),
    ("bhavani", MODELS / "bhavani_model.json", BHAVANI_LABELS),
)


class Quiet:
    def info(self, *_):
        pass

    def warning(self, *_):
        pass


def model_of(path):
    return json.loads(path.read_text(encoding="utf-8"))


def features_for(model, rows=3):
    """a frame carrying exactly the columns the model names"""
    return pl.DataFrame(
        {"hospitalization_id": [f"h{i}" for i in range(rows)]}
        | {name: [1.0] * rows for name in model["features"]}
    )


@pytest.mark.parametrize("scheme,path,labels", SCHEMES, ids=lambda v: str(v)[:12])
def test_shipped_model_exists(scheme, path, labels):
    """it is in git, and a site that cannot find it cannot transfer"""
    assert path.is_file(), f"{path} is missing; a site cannot transfer {scheme}"


@pytest.mark.parametrize("scheme,path,labels", SCHEMES, ids=lambda v: str(v)[:12])
def test_shipped_model_carries_everything_apply_needs(scheme, path, labels):
    """
    the fields `apply_model` refuses to default

    `log_transform` especially: absent, there is no way to place a patient in
    the same space as the centroids, and the natural default -- apply no
    transform -- is exactly wrong when the centroids were fitted on logs.
    """
    model = model_of(path)
    assert model["scheme"] == scheme
    for field in ("features", "centre", "scale", "centroids", "labels",
                  "log_transform"):
        assert field in model, f"{path.name} has no {field}"
    assert len(model["centre"]) == len(model["features"])
    assert len(model["scale"]) == len(model["features"])
    assert len(model["labels"]) == len(model["centroids"])
    assert set(model["labels"]) <= set(labels)
    assert set(model["log_transform"]) <= set(model["features"])


@pytest.mark.parametrize("scheme,path,labels", SCHEMES, ids=lambda v: str(v)[:12])
def test_shipped_model_labels_patients(scheme, path, labels):
    model = model_of(path)
    assigned = apply_model(features_for(model), path, scheme, labels, Quiet())
    column = f"{scheme}_phenotype"
    assert assigned.height == 3
    assert set(assigned[column].cast(pl.String).to_list()) <= set(labels)
    assert assigned[column].null_count() == 0


@pytest.mark.parametrize("scheme,path,labels", SCHEMES, ids=lambda v: str(v)[:12])
def test_transfer_refuses_a_missing_feature(scheme, path, labels):
    """
    dropping a variable locally would place patients in a different space

    The centroids were fitted in a space that includes it, so there is no
    honest way to apply the model without it -- and labels produced anyway
    would look exactly like correct ones.
    """
    model = model_of(path)
    short = features_for(model).drop(model["features"][0])
    with pytest.raises(KeyError, match="absent at this site"):
        apply_model(short, path, scheme, labels, Quiet())


@pytest.mark.parametrize("scheme,path,labels", SCHEMES, ids=lambda v: str(v)[:12])
def test_transfer_refuses_the_wrong_scheme(scheme, path, labels):
    """passing the Seymour model where Bhavani is wanted must not silently work"""
    other = "bhavani" if scheme == "seymour" else "seymour"
    with pytest.raises(ValueError, match="model, not"):
        apply_model(features_for(model_of(path)), path, other, labels, Quiet())


def test_transfer_refuses_a_model_with_no_transform_recorded(tmp_path):
    model = model_of(MODELS / "seymour_model.json")
    del model["log_transform"]
    path = tmp_path / "seymour_model.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="log-transformed"):
        apply_model(features_for(model), path, "seymour", SEYMOUR_LABELS, Quiet())


def test_the_shipped_feature_set_matches_what_this_cohort_fitted():
    """
    order may differ, membership may not

    `apply_model` selects by name, so the order `select_available` happened to
    return is irrelevant -- but a genuine difference in membership means the
    shipped model cannot be applied here at all, which is the failure worth
    catching before a site discovers it mid-run.
    """
    local = ROOT / "output" / "cohort" / "seymour_model.json"
    if not local.is_file():
        pytest.skip("no local fit to compare against on this machine")
    assert set(model_of(MODELS / "seymour_model.json")["features"]) == set(
        model_of(local)["features"]
    )


# ===================================================== the Bhavani preparation
#
# `apply_model` on its own is not the model that was fitted. `assign_bhavani`
# applies two steps before the centroids ever see a patient -- an eligibility
# filter that leaves sparse patients NULL, and a carry-forward fill the
# centroids were fitted on top of -- and the transfer path applied neither. A
# patient with nothing charted therefore standardised to all-zero z, which is
# the derivation mean, and came back with a confident label in a column named
# and typed exactly like the one the fit path writes. These tests pin both
# steps.


def bhavani_cfg(**overrides):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ROOT / "code" / "config" / "phenotypes_sepsis.yaml")
    for key, value in overrides.items():
        cfg.bhavani[key] = value
    return cfg


def trajectory_cohort(cfg, n=160, sparse_ids=("sparse", "empty")):
    """
    a synthetic cohort with four separable shapes, plus patients too sparse
    to cluster

    The four shapes exist so the fit has something to find; the sparse ones are
    the point of the test. `sparse` has one charted hour -- below the
    three-block rule -- and `empty` has nothing at all.
    """
    import numpy as np

    bh = cfg.bhavani
    variables = list(bh.variables)
    n_blocks = int(bh.window_hours) // int(bh.block_hours)
    columns = [f"{v}_h{b}" for v in variables for b in range(n_blocks)]

    rng = np.random.default_rng(0)
    shapes = {
        0: {"temp_c": 38.5, "heart_rate": 110, "respiratory_rate": 26, "sbp": 95, "dbp": 55},
        1: {"temp_c": 38.3, "heart_rate": 105, "respiratory_rate": 25, "sbp": 150, "dbp": 85},
        2: {"temp_c": 36.6, "heart_rate": 80, "respiratory_rate": 16, "sbp": 125, "dbp": 72},
        3: {"temp_c": 36.4, "heart_rate": 78, "respiratory_rate": 17, "sbp": 92, "dbp": 52},
    }
    rows, ids = [], []
    for i in range(n):
        shape = shapes[i % 4]
        ids.append(f"h{i}")
        rows.append(
            [shape[c.rsplit("_h", 1)[0]] + rng.normal(0, 0.5) for c in columns]
        )

    frame = pl.DataFrame(
        {"hospitalization_id": ids}
        | {c: [row[j] for row in rows] for j, c in enumerate(columns)}
    )
    counts = pl.DataFrame(
        {
            "hospitalization_id": ids,
            "n_complete_blocks": [n_blocks] * n,
            "n_core_blocks": [n_blocks] * n,
            "n_temp_observations": [n_blocks] * n,
        }
    )

    # the sparse ones: present in the trajectory frame, excluded by the rule
    charted_hours = {"sparse": 1, "empty": 0}
    for hid in sparse_ids:
        charted = charted_hours[hid]
        frame = pl.concat(
            [
                frame,
                pl.DataFrame(
                    {"hospitalization_id": [hid]}
                    | {
                        c: [
                            shapes[2][c.rsplit("_h", 1)[0]]
                            if int(c.rsplit("_h", 1)[1]) < charted
                            else None
                        ]
                        for c in columns
                    }
                ).cast(frame.schema),
            ]
        )
        counts = pl.concat(
            [
                counts,
                pl.DataFrame(
                    {
                        "hospitalization_id": [hid],
                        "n_complete_blocks": [charted],
                        "n_core_blocks": [charted],
                        "n_temp_observations": [charted],
                    }
                ).cast(counts.schema),
            ]
        )
    return frame, counts, columns


@pytest.fixture(scope="module")
def fitted_bhavani(tmp_path_factory):
    """fit on the synthetic cohort and save the model, as a derivation site does"""
    from clifcs.sepsis_phenotype import assign_bhavani, save_model

    cfg = bhavani_cfg()
    frame, counts, _ = trajectory_cohort(cfg)
    labels, _, model = assign_bhavani(frame, counts, cfg, Quiet())
    path = tmp_path_factory.mktemp("models") / "bhavani_model.json"
    save_model(path, model, "bhavani")
    return cfg, frame, counts, labels, path


def test_the_fitted_model_records_its_preparation(fitted_bhavani):
    """
    both steps have to travel, and `save_model` enumerates its payload -- a
    field it does not name does not reach the receiving site however carefully
    the fit put it in the model dict, which is how `log_transform` was lost
    """
    _, _, _, _, path = fitted_bhavani
    model = model_of(path)
    assert model["eligibility"]["min_core_blocks"] == 3
    assert len(model["fill_medians"]) == len(model["features"])


def test_transfer_reproduces_the_fit_including_its_nulls(fitted_bhavani):
    from clifcs.sepsis_phenotype import apply_bhavani

    cfg, frame, counts, labels, path = fitted_bhavani
    transferred = apply_bhavani(frame, counts, path, cfg, Quiet())

    joined = labels.join(
        transferred, on="hospitalization_id", how="inner", suffix="_t"
    )
    assert joined.height == frame.height
    disagree = joined.filter(
        pl.col("bhavani_phenotype").is_null()
        != pl.col("bhavani_phenotype_t").is_null()
    )
    assert disagree.height == 0, disagree
    both = joined.filter(pl.col("bhavani_phenotype").is_not_null())
    assert both.height > 0
    mismatched = both.filter(
        pl.col("bhavani_phenotype") != pl.col("bhavani_phenotype_t")
    )
    assert mismatched.height == 0, mismatched


@pytest.mark.parametrize("hid", ["sparse", "empty"])
def test_a_sparse_patient_is_left_unassigned_not_placed(fitted_bhavani, hid):
    """
    the regression: `apply_model` on raw trajectories labels this patient

    Every unmeasured hour standardises to a non-finite z, which `apply_model`
    replaces with 0.0 -- the derivation mean -- so the patient lands on
    whichever centroid is nearest the origin. The fit path leaves them null,
    because "not enough vitals charted in eight hours" is a real category and
    assigning it to the nearest shape would invent a trajectory.
    """
    from clifcs.sepsis_phenotype import apply_bhavani

    cfg, frame, counts, labels, path = fitted_bhavani

    fit = labels.filter(pl.col("hospitalization_id") == hid)
    assert fit.height == 1 and fit["bhavani_phenotype"][0] is None

    transferred = apply_bhavani(frame, counts, path, cfg, Quiet()).filter(
        pl.col("hospitalization_id") == hid
    )
    assert transferred["bhavani_phenotype"][0] is None

    # and the unprepared path, which is what shipped, does label them
    unprepared = apply_model(frame, path, "bhavani", BHAVANI_LABELS, Quiet()).filter(
        pl.col("hospitalization_id") == hid
    )
    assert unprepared["bhavani_phenotype"][0] is not None


def test_transfer_applies_the_fitted_rule_not_the_local_config(fitted_bhavani):
    """
    a receiving site that configured a stricter rule must still place the
    patients the derivation cohort clustered, or the group sizes stop being
    comparable -- which is the one thing the shared model exists to protect
    """
    from clifcs.sepsis_phenotype import apply_bhavani

    cfg, frame, counts, labels, path = fitted_bhavani
    stricter = bhavani_cfg(min_core_blocks=8, min_temp_observations=8)

    transferred = apply_bhavani(frame, counts, path, stricter, Quiet())
    assert (
        transferred.filter(pl.col("bhavani_phenotype").is_not_null()).height
        == labels.filter(pl.col("bhavani_phenotype").is_not_null()).height
    )


def test_a_model_with_no_rule_recorded_warns_and_uses_the_local_one(fitted_bhavani):
    """
    older files predate the rule travelling. Warned rather than refused: the
    config is shared in git, so the local rule is usually the fitted one -- but
    a site has to be told it is being assumed.
    """
    import pathlib as _pathlib

    from clifcs.sepsis_phenotype import apply_bhavani

    cfg, frame, counts, _, path = fitted_bhavani
    model = model_of(path)
    del model["eligibility"]
    del model["fill_medians"]
    older = _pathlib.Path(str(path).replace(".json", "_older.json"))
    older.write_text(json.dumps(model), encoding="utf-8")

    said = []

    class Loud(Quiet):
        def warning(self, message, *_):
            said.append(message)

    transferred = apply_bhavani(frame, counts, older, cfg, Loud())
    assert any("eligibility rule" in m for m in said)
    assert any("column medians" in m for m in said)
    assert transferred.height == frame.height

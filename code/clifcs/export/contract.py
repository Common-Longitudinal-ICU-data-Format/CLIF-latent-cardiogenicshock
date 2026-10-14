#!/usr/bin/env python3

"""
what a site is allowed to send, enforced in both directions

The consortium policy is that raw records never move and derived patient-level
data is limited to **latent embeddings plus a minimal set of ordered tokens** --
enough to label a point in a projection, not enough to reconstruct a record.
Aggregate count tables move freely subject to small-cell suppression.

This module is the machine-readable form of that policy. Every column that may
cross a site boundary is named here, with the reason it is permitted. Anything
not named is rejected:

  * `validate_patient_frame` runs at the **sending** site, before a file is
    written, so a column added upstream cannot leak by being picked up
    automatically;
  * `read_bundle` runs at the **coordinating** site, so a file that arrives
    carrying more than the contract allows is refused rather than quietly
    analysed.

Validating only on write would be trusting every site to run the current code.
Validating only on read would mean the data had already left. Both ends check,
which is the only arrangement where a disagreement is visible to someone.

Why an allowlist and not a denylist: the failure this guards against is
somebody adding a useful column to the cohort frame and it travelling because
nothing said it could not. A denylist cannot see a column nobody thought of.
"""

from __future__ import annotations

import dataclasses
import re

import polars as pl

#: bumped whenever the permitted column set changes in a way that makes an
#: older bundle non-conformant. The coordinating site compares this against
#: each bundle's manifest and refuses a mismatch rather than guessing.
SCHEMA_VERSION = "1.0"

#: the embedding itself: `emb_0000 .. emb_{dim-1}`.
#:
#: NOT de-identified, and the contract does not pretend otherwise -- a 1024-d
#: vector is a learned function of the patient's entire recorded history. It is
#: permitted because it is the object of study, and it travels under a DUA.
EMBEDDING_PATTERN = re.compile(r"^emb_\d{4}$")

#: the one key tying a patient's rows together across anchors and across the
#: embedding/label file split. A salted SHA-256 of the local record number,
#: truncated to 128 bits; the salt never leaves the sending site.
ROW_KEY = "row_key"


@dataclasses.dataclass(frozen=True)
class Permitted:
    """one allowlisted column and the argument for letting it travel"""

    name: str
    why: str


# ---------------------------------------------------------------------------
# Structural columns. Nothing clinical, and all of it needed to assemble the
# pooled frame at all.
STRUCTURAL: tuple[Permitted, ...] = (
    Permitted(ROW_KEY, "joins a patient's embedding to their labels; salted hash"),
    Permitted("site", "the grouping variable the whole site-effect analysis is about"),
    Permitted("anchor", "which cut point this row was embedded at"),
    Permitted("split", "cocoa train/tuning/held_out partition, for 06"),
)

# ---------------------------------------------------------------------------
# Ordered token labels. This is the "minimal tokenized elements" the policy
# allows, and the form matters: each is the *ordinal rank of the token bin*
# the model actually read, never the measured value.
#
# A rank is what a colour ramp needs and all it needs. Exporting MAP as 58.0
# would ship a measurement; exporting it as bin 3 of 10 ships the thing that
# distinguishes one region of the projection from another, coarsened to the
# resolution the model itself works at. The bin edges are the tokenizer's, so
# they are identical at every site by construction -- which is also what makes
# the ranks comparable across sites without anyone agreeing on anything.
#
# There is no timestamp and no sequence. `tok_*` is the *last* token for that
# variable before the anchor, which is the state the embedding was taken in.
TOKEN_LABEL_PREFIX = "tok_"

#: variables whose last pre-anchor token bin may be exported, and why each is
#: wanted. Deliberately short: every added variable widens the quasi-identifier
#: formed by the whole row, and the justification for each has to be that a
#: projection panel is coloured by it.
TOKEN_LABELS: tuple[Permitted, ...] = (
    Permitted("tok_map", "the panel the shock question is read off"),
    Permitted("tok_sbp", "criterion 1 is defined on it; the trial gap is in it"),
    Permitted("tok_heart_rate", "shock index's other half"),
    Permitted("tok_lactate", "criterion 2 hypoperfusion marker"),
    Permitted("tok_creatinine", "criterion 2 renal marker"),
    Permitted("tok_bicarbonate", "separates Zweck III from II"),
    Permitted("tok_spo2", "distinguishes respiratory from circulatory failure"),
    Permitted("tok_respiratory_rate", "a Bhavani trajectory variable"),
    Permitted("tok_temperature", "the Bhavani defining variable"),
    Permitted("tok_gcs_total", "criterion 2 AMS marker"),
)

# ---------------------------------------------------------------------------
# Phenotype and subgroup labels. All categorical, all assigned by a model or
# codebook identical at every site, so none carries site-specific information
# beyond the assignment itself.
LABELS: tuple[Permitted, ...] = (
    Permitted("cs_phenotype", "Zweck 2021 I/II/III, transferred centroids"),
    Permitted("seymour_phenotype", "Seymour 2016 alpha/beta/gamma/delta"),
    Permitted("bhavani_phenotype", "Bhavani 2022 vital-trajectory group A-D"),
    Permitted("shock_type", "coded mechanism; a labelling, never a filter"),
    Permitted("sepsis_cohort", "denominator flag for the two sepsis phenotypes"),
    Permitted("cardiac_origin", "reported, not applied"),
)

# ---------------------------------------------------------------------------
# Outcomes and the few covariates the figures are stratified on.
#
# `age_band` rather than age in years: a row already carries ten token bins and
# three phenotype labels, and exact age is the quasi-identifier that makes that
# combination re-identifying. Five-year bands with a 90+ top code keep every
# panel that was drawn against age and drop the precision nobody plots.
OUTCOMES: tuple[Permitted, ...] = (
    Permitted("in_hospital_mortality", "the primary outcome"),
    Permitted("icu_los_days", "secondary outcome, and a severity proxy"),
    Permitted("age_band", "5-year bands, 90+ top-coded"),
    Permitted("sex_category", "reported in every trial baseline table"),
    Permitted("hours_admission_to_cs_onset", "relative, so not a HIPAA date"),
    Permitted("sofa_non_cv", "the severity adjustment 07 reports against"),
)

# ---------------------------------------------------------------------------
# Trial eligibility. One boolean per trial, computed site-side from published
# criteria. These have to travel because the eligibility logic reads raw CLIF
# tables the coordinating site no longer has.
TRIAL_PREFIX = "elig_"

# ---------------------------------------------------------------------------
# Token family counts: ~23 integers per patient, the per-family totals of the
# token stream the model read.
#
# Permitted because they are what 04 regresses the site effect on, and the site
# effect is the largest single finding in this pipeline (97.1% neighbour
# purity). Without them, "the sites separate" cannot be decomposed into "because
# they chart differently" at the coordinating site at all. A family count is far
# coarser than the stream it came from: `fam_n_labs = 184` says a patient had
# 184 lab tokens, not which ones or when.
FAMILY_PREFIX = "fam_n_"

# ---------------------------------------------------------------------------
# Downstream prediction labels: `<task>__label` and `<task>__eligible` for each
# task in `clifcs.labels.TASKS` -- 24-hour mortality, new invasive ventilation,
# pressor escalation.
#
# Permitted on the same argument as `in_hospital_mortality`, which is already
# here: they are single booleans about an outcome, derived site-side from
# tables that do not travel. 06 asks whether the embedding carries anything
# clinically predictive and whether a model built at one hospital transfers to
# another, which is a question only a pooled run can answer -- so the labels
# have to reach it or the analysis does not exist federated.
#
# `__eligible` travels with `__label` because it is what makes the label
# interpretable: "did not develop X" and "was already on X at the anchor, so the
# question does not arise" are different, and pooling them would inflate every
# denominator.
LABEL_SUFFIXES = ("__label", "__eligible")


def _names(*groups: tuple[Permitted, ...]) -> frozenset[str]:
    return frozenset(p.name for group in groups for p in group)


#: every non-patterned column a patient-level file may carry
PERMITTED_COLUMNS: frozenset[str] = _names(STRUCTURAL, TOKEN_LABELS, LABELS, OUTCOMES)

#: why each one is allowed, for the error message and for the docs
RATIONALE: dict[str, str] = {
    p.name: p.why
    for group in (STRUCTURAL, TOKEN_LABELS, LABELS, OUTCOMES)
    for p in group
}

#: columns that must be present for a bundle to be usable at all
REQUIRED_COLUMNS: tuple[str, ...] = (ROW_KEY, "site", "anchor")

#: Columns that must NEVER appear, checked by name in addition to the
#: allowlist. Redundant by construction -- the allowlist already excludes them
#: -- and kept because a redundant check costs nothing and this is the list a
#: reviewer actually wants to see spelled out.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "patient_id",
        "hospitalization_id",
        "admission_dttm",
        "discharge_dttm",
        "cs_onset_dttm",
        "anchor_dttm",
        "sbp_onset_dttm",
        "vasoactive_onset_dttm",
        "hemo_onset_dttm",
        "ams_dttm",
        "lactate_dttm",
        "aki_dttm",
        "end_organ_dttm",
        "age_at_admission",
        "race_category",
        "ethnicity_category",
        "peak_troponin_ng_l",
        "tokens_past",
        "s_elapsed_past",
    }
)

#: integer dtypes a count column may have, for the small-cell check
_INT_DTYPES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def is_permitted(column: str) -> bool:
    """does the contract allow this column to cross a site boundary?"""
    if column in FORBIDDEN_COLUMNS:
        return False
    return (
        column in PERMITTED_COLUMNS
        or bool(EMBEDDING_PATTERN.match(column))
        or column.startswith(TRIAL_PREFIX)
        or column.startswith(FAMILY_PREFIX)
        or column.endswith(LABEL_SUFFIXES)
    )


class ContractViolation(Exception):
    """a frame carries something the consortium policy does not permit"""


def validate_patient_frame(frame: pl.DataFrame, origin: str) -> None:
    """
    refuse a patient-level frame carrying anything outside the contract

    `origin` names what is being checked, so the message says whether a site
    tried to send the column or a bundle arrived carrying it.

    Raises rather than dropping. Silently dropping an unpermitted column would
    make the export succeed, which means nobody finds out the cohort frame grew
    a field until it is in another institution's hands.
    """
    offending = sorted(c for c in frame.columns if not is_permitted(c))
    if offending:
        raise ContractViolation(
            f"{origin}: {len(offending)} column(s) not permitted to leave a "
            f"site under schema {SCHEMA_VERSION}: {', '.join(offending)}.\n"
            "If one of these genuinely belongs in a federated analysis, add it "
            "to clifcs/export/contract.py with the argument for why -- which "
            "is a consortium decision, not a code change."
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ContractViolation(f"{origin}: missing required column(s) {missing}")

    if frame.height == 0:
        raise ContractViolation(f"{origin}: no rows")


def validate_counts_frame(frame: pl.DataFrame, origin: str, min_cell: int) -> None:
    """
    refuse an aggregate table that still carries a suppressible cell

    Counts travel more freely than patient rows, but a count of 2 in a
    phenotype x shock-type x site cell is closer to naming a patient than to
    being a statistic. `clifcs.export.suppress` is what produces a conformant
    table; this checks the result, so a table assembled by hand cannot skip it.

    Only columns named `n` or `n_*` are checked, which is the naming convention
    every count table in this pipeline already follows.
    """
    for column in frame.columns:
        if column != "n" and not column.startswith("n_"):
            continue
        if frame.schema[column] not in _INT_DTYPES:
            continue
        values = frame[column].drop_nulls()
        risky = values.filter((values > 0) & (values < min_cell))
        if len(risky):
            raise ContractViolation(
                f"{origin}: column {column!r} holds {len(risky)} cell(s) between "
                f"1 and {min_cell - 1}, which the minimum cell size forbids. "
                "Pass the table through clifcs.export.suppress.suppress_counts "
                "before writing it."
            )


def describe() -> str:
    """the contract as a readable table, for the docs and the export log"""
    lines = [
        f"export contract, schema {SCHEMA_VERSION}",
        "",
        "patient-level columns permitted to cross a site boundary:",
    ]
    for title, group in (
        ("structural", STRUCTURAL),
        ("ordered token bins (ordinal rank, never the value)", TOKEN_LABELS),
        ("phenotype and subgroup labels", LABELS),
        ("outcomes and stratifiers", OUTCOMES),
    ):
        lines.append(f"\n  {title}")
        for p in group:
            lines.append(f"    {p.name:<32} {p.why}")
    lines += [
        "\n  patterned",
        f"    {'emb_NNNN':<32} the latent embedding; NOT de-identified",
        f"    {TRIAL_PREFIX + '<trial>':<32} per-trial eligibility, computed site-side",
        f"    {FAMILY_PREFIX + '<family>':<32} per-family token counts, for the site-effect model",
        f"    {'<task>__label / __eligible':<32} downstream prediction labels, computed site-side",
        "",
        "everything else is refused at both ends.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())

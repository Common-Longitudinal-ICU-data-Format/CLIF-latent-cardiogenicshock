#!/usr/bin/env python3

"""
what a site is allowed to send, enforced in both directions

The consortium policy is that raw records never move and derived patient-level
data is limited to **latent embeddings plus a minimal set of ordered token
bins** -- enough to label a point in a projection, not enough to reconstruct a
record. Aggregate count tables move freely subject to small-cell suppression.

Schema 3.0 withdrew the token *sequences*. Earlier versions shipped
`tokens_past` and `s_elapsed_past` so the coordinating site could re-embed, and
the argument for it was real: a sequence lets you re-anchor, read another
layer, or re-run after a model update without asking anyone to re-export. It
was withdrawn because a sequence is a re-serialisation of the record and is
strictly more disclosive than the embedding derived from it -- which is a
lossy projection and cannot be inverted. Sites now compute the embedding
themselves (`02_site_embeddings.py`) and ship only that. The two sequence
columns are named in `FORBIDDEN_COLUMNS`, so the old path cannot be revived by
accident.

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
import pathlib
import re

import polars as pl

from clifcs.export import suppress

#: bumped whenever the permitted column set changes in a way that makes an
#: older bundle non-conformant. The coordinating site compares this against
#: each bundle's manifest and refuses a mismatch rather than guessing.
# 3.1: every published statistic now names the count it was computed over,
# and an unregistered numeric column refuses the export. Rates that used to
# travel over an unexported or superset denominator are blanked, so a 3.0
# bundle is NOT interchangeable with a 3.1 one.
SCHEMA_VERSION = "3.1"

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
    # Three Zweck variants, one per cardiogenic denominator. The bare name is
    # the all-shock fit, kept for continuity; the two suffixed ones are fitted
    # on the ICD-coded and cardiac-origin cohorts respectively and are null for
    # patients outside those. Null here means "not defined for this patient",
    # not "not assigned", which is why none of them is filled downstream.
    Permitted("cs_phenotype", "Zweck 2021 I/II/III, fitted on all shock"),
    Permitted("cs_phenotype_coded",
              "Zweck I/II/III, fitted on ICD-coded cardiogenic shock only"),
    Permitted("cs_phenotype_cardiac_origin",
              "Zweck I/II/III, fitted on the cardiac-origin cohort only"),
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
    # Which window produced `sofa_non_cv`: "pre24h_onset" (the worst score in
    # the 24h ending at onset, which is the adjuster) or "whole_stay" (the
    # contaminated fallback). Two values, no patient in either -- it is there
    # so a bundle says which definition it used, and so a pooled analysis can
    # refuse a mixture instead of averaging two different quantities under one
    # column name.
    Permitted("sofa_window", "which SOFA window the adjustment came from"),
)

# ---------------------------------------------------------------------------
# Trial eligibility. One boolean per trial, computed site-side from published
# criteria. These have to travel because the eligibility logic reads raw CLIF
# tables the coordinating site no longer has.
TRIAL_PREFIX = "elig_"

# ---------------------------------------------------------------------------
# How far outside the trial's published population a patient sits, and which
# way. Computed site-side by `clifcs.trials.table1_position`, which needs the
# measured values; these four are statistics *over* those values, not the
# values themselves.
#
# Eligibility alone cannot answer the question the guideline-fit analysis is
# about. A trial's criteria are thresholds and admit a far wider range than the
# trial ended up containing, so "would it have enrolled them" and "do they
# resemble who it enrolled" are different questions with different answers.
#
#   t1_absdist_<trial>        mean distance outside the published IQRs, in IQR
#                             units, over the variables this patient has
#   t1_severity_gap_<trial>   the same, SIGNED, so positive is sicker or older
#                             than the trial's median. Direction is the finding:
#                             among ECLS-SHOCK-eligible patients the ones
#                             sicker than the trial run 46.6% mortality against
#                             30.6% for the healthier, and an unsigned distance
#                             calls both groups "outside"
#   t1_outside_<trial>        how many variables sit outside their IQR, so
#                             `== 0` is "inside every published IQR"
#   t1_frac_compared_<trial>  the share of published variables this patient
#                             actually has. Travels because without it the
#                             other three are uninterpretable -- a distance
#                             over one variable is not a distance over seven
#
# Deliberately NOT the whole `t1_` family. `t1_maxdist_` and `t1_n_compared_`
# add nothing a panel is coloured by, and `t1_n_published_` is a property of
# the paper rather than of the patient. Every continuous column widens the
# quasi-identifier the whole row forms, so each one has to earn its place.
TRIAL_FIT_PREFIXES: tuple[str, ...] = (
    "t1_absdist_",
    "t1_severity_gap_",
    "t1_outside_",
    "t1_frac_compared_",
)

# ---------------------------------------------------------------------------
# Was the treatment the trial recommends actually delivered? One boolean per
# trial from `clifcs.concordance`, on the same argument as `elig_`: it reads
# medication and procedure tables the coordinating site does not have.
#
# Read it with the trial's own direction in mind. For the negative device
# trials (IABP-SHOCK II, ECMO-CS, ECLS-SHOCK) "concordant" means the device was
# WITHHELD, which happens most in patients too well or too sick to cannulate.
# Confounding by indication is severe throughout and is not adjustable away.
CONCORDANCE_PREFIX = "concordant_"

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



# ---------------------------------------------------------------------------
# WITHDRAWN in schema 3.0: the token sequences.
#
# `tokens_past` (the ordered token ids the model read) and `s_elapsed_past`
# (seconds elapsed at each one) were together the model's entire input for a
# patient, and schema 2.0 shipped them so the coordinating site could re-embed.
#
# They are now refused. A token stream is a re-serialisation of the record --
# which labs were drawn, in what order, how often, against what clock -- and it
# was the most disclosive artefact this pipeline produced, more so than the
# embedding derived from it. An embedding is a lossy projection of the stream
# and cannot be inverted back to it; the stream is not a lossy anything.
#
# What that cost, stated plainly, because it was not nothing: re-anchoring at a
# different cut point, probing a different layer, changing the pooling, and
# re-running after a model update all now require sites to re-export. Tier 2
# ships an answer where it used to ship the input.
#
# They are in FORBIDDEN_COLUMNS rather than merely absent from the allowlist,
# so a frame carrying them is refused by name and the reviewer sees why.


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
        # withdrawn in schema 3.0 -- see the note above
        "tokens_past",
        "s_elapsed_past",
        # the wall-clock timestamp of every event, which sat in the same source
        # file as the sequences and was never on the allowlist even then
        "times",
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
        or column.startswith(TRIAL_FIT_PREFIXES)
        or column.startswith(CONCORDANCE_PREFIX)
        or column.endswith(LABEL_SUFFIXES)
    )


#: the fields a phenotype model file may carry when a site ships the model it
#: fitted: per-feature moments and cluster centroids, the same disclosure class
#: as the files in models/, and never a patient row
MODEL_FIELDS: frozenset[str] = frozenset(
    {"scheme", "features", "centre", "scale", "centroids", "labels", "log_transform", "z_clip"}
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
    refuse an aggregate table that still carries a recoverable cell

    `clifcs.export.suppress` is what produces a conformant table; this checks
    the result, so a table assembled by hand, or by a producer that forgot,
    cannot skip it. Neither end assumes the other ran -- that is the whole
    point of checking twice -- so this re-tests the thresholds rather than
    trusting that the suppressor was called.

    Two passes, because there are two ways a cell gets out.

    PASS 1, counts. Columns named `n` or `n_*`, minus the handful that count
    something other than patients (`suppress.NOT_PATIENT_COUNTS`).

    PASS 2, statistics. This is the half that did not exist. The old version
    skipped every column not named `n`/`n_*`, so no rate, share, mean or median
    was ever examined at either end of the pipeline -- and a published rate
    hands back its numerator by multiplication. Every numeric column must now be
    either a registered `suppress.Statistic`, whose denominator has to be in the
    same row and whose numerator and complement are re-tested, or named in
    `suppress.NOT_PATIENT_STATISTICS` with the argument for why it determines no
    patient cell.

    An unregistered numeric column REFUSES THE EXPORT. That is the allowlist
    argument this module already makes about patient columns -- "a denylist
    cannot see a column nobody thought of" -- applied to statistics, and it is
    what turns every future omission from a silent leak into a failed export at
    the sending site.
    """
    # ------------------------------------------------------------- pass 1
    for column in frame.columns:
        if column != "n" and not column.startswith("n_"):
            continue
        if column in suppress.NOT_PATIENT_COUNTS:
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

    # ------------------------------------------------------------- pass 2
    counted = {
        c
        for c in frame.columns
        if (c == "n" or c.startswith("n_"))
        and c not in suppress.NOT_PATIENT_COUNTS
        and frame.schema[c] in _INT_DTYPES
    }
    for column in suppress.numeric_columns(frame):
        # NOT_PATIENT_COUNTS as well as NOT_PATIENT_STATISTICS: a column like
        # `n_variables_compared` is excluded from `counted` precisely BECAUSE it
        # is declared not to be a patient cell, so falling through to the
        # unregistered-numeric refusal below would make the two declarations
        # contradict each other and every fit table unexportable.
        if (
            column in counted
            or column in suppress.NOT_PATIENT_STATISTICS
            or column in suppress.NOT_PATIENT_COUNTS
        ):
            continue
        stat = suppress.statistic_for(column)
        if stat is None:
            raise ContractViolation(
                f"{origin}: numeric column {column!r} is not registered. Every "
                "statistic has to name the count it was computed over, because "
                "a published rate hands back its numerator by multiplication. "
                "Add a clifcs.export.suppress.Statistic for it, or name it in "
                "NOT_PATIENT_STATISTICS with the argument for why it determines "
                "no patient cell."
            )
        if stat.over not in frame.columns:
            raise ContractViolation(
                f"{origin}: column {column!r} is computed over {stat.over!r}, "
                f"which this table does not export. A rate whose denominator is "
                "not in the same row cannot be checked by anyone -- export the "
                "denominator as a count column, or do not publish the rate."
            )
        offending = frame.filter(
            pl.col(column).is_not_null()
            & ~suppress.publishable(stat, min_cell)
        )
        if offending.height:
            raise ContractViolation(
                f"{origin}: column {column!r} holds {offending.height} value(s) "
                f"that resolve to a group of 1 to {min_cell - 1} patients over "
                f"{stat.over!r}. Pass the table through "
                "clifcs.export.suppress.suppress_counts before writing it."
            )

    # ------------------------------- within-row complements, re-tested here too
    for part, whole, why in suppress.COMPLEMENT_PAIRS:
        if part not in frame.columns or whole not in frame.columns:
            continue
        if part in suppress.NOT_PATIENT_COUNTS or whole in suppress.NOT_PATIENT_COUNTS:
            continue
        gap = pl.col(whole) - pl.col(part)
        offending = frame.filter(gap.is_not_null() & (gap > 0) & (gap < min_cell))
        if offending.height:
            raise ContractViolation(
                f"{origin}: {whole!r} minus {part!r} leaves "
                f"{offending.height} cell(s) of 1 to {min_cell - 1} patients "
                f"({why}). Suppress {part!r} where its complement is small."
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
        f"    {'t1_absdist_<trial>':<32} distance outside the published IQRs, in IQR units",
        f"    {'t1_severity_gap_<trial>':<32} the same, signed: positive is sicker than the trial",
        f"    {'t1_outside_<trial>':<32} variables outside their IQR; 0 is inside every one",
        f"    {'t1_frac_compared_<trial>':<32} share of published variables this patient has",
        f"    {CONCORDANCE_PREFIX + '<trial>':<32} was the recommended treatment delivered",
        f"    {'<task>__label / __eligible':<32} downstream prediction labels, computed site-side",
        "",
        "WITHDRAWN in 3.0: tokens_past, s_elapsed_past. Sites embed locally and",
        "ship the embedding; the sequence is more disclosive than the vector",
        "derived from it, and is now refused by name.",
        "",
        "everything else is refused at both ends.",
    ]
    return "\n".join(lines)


def audit_bundle(bundle: pathlib.Path, min_cell: int = suppress.DEFAULT_MIN_CELL):
    """
    run both validation passes over every CSV in a bundle, and report

    What a contributing site runs on its own upload directory before sending,
    and what a reviewer runs on one that arrived. Returns a list of
    `(relative path, message)`; empty means nothing in the bundle resolves to a
    group under `min_cell`.

    Reads only the tables; it needs no cohort, no config and no model.
    """
    problems: list[tuple[str, str]] = []
    for path in sorted(bundle.rglob("*.csv")):
        relative = path.relative_to(bundle).as_posix()
        try:
            frame = pl.read_csv(path)
        except Exception as exc:
            problems.append((relative, f"could not be read: {exc}"))
            continue
        if not frame.height:
            continue
        try:
            validate_counts_frame(frame, relative, min_cell)
        except ContractViolation as exc:
            problems.append((relative, str(exc).split(": ", 1)[-1]))
    return problems


def _audit_cli(argv) -> int:
    bundle = pathlib.Path(argv[0])
    if not bundle.is_dir():
        print(f"not a directory: {bundle}")
        return 2
    min_cell = int(argv[1]) if len(argv) > 1 else suppress.DEFAULT_MIN_CELL
    problems = audit_bundle(bundle, min_cell)
    print(f"auditing {bundle} at min_cell={min_cell}")
    tables = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*.csv"))
    print(f"  {len(tables)} table(s) read")
    if not problems:
        print("  no recoverable cell found")
        return 0
    for relative, message in problems:
        print(f"  [LEAK] {relative}")
        print(f"         {message}")
    print(f"  {len(problems)} table(s) carry a recoverable cell")
    return 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--audit":
        raise SystemExit(_audit_cli(sys.argv[2:]))
    print(describe())

#!/usr/bin/env python3

"""
shock type from coded diagnoses

The cohort is selected on physiology -- sustained hypotension or vasoactive
support, plus end-organ damage -- which is mechanism-agnostic by construction
and therefore selects *undifferentiated* shock. This module attaches what the
coders said the mechanism was, so that the embedding can be read against it.

It is a labelling, not a filter. Nothing here changes who is in the cohort; the
label exists so a figure can ask whether the model's latent space separates
septic from cardiogenic shock, and the answer is allowed to be no.

ICD CODES ARE WEAKER EVIDENCE THAN THE REST OF THIS PIPELINE USES, and that is
a deliberate inconsistency. Everywhere else the cohort is built from things that
were done or measured -- an infusion, a procedure, a laboratory value -- on the
argument that a diagnosis code asserts a condition rather than recording an
event. That argument still holds. But the mechanism of shock is a clinical
judgement and there is no measurement that encodes it, so the coded diagnosis is
the only thing available; it is used here only to describe patients already
enrolled, never to select them.

Two consequences worth carrying into any reading of the result:

* **Coding practice differs between sites more than case mix plausibly does.**
  Septic shock (R65.21) is coded in 14.1% of NU hospitalizations against 5.3%
  of UCMC's, a 2.7-fold gap. Cardiogenic shock is much closer (5.6% vs 5.1%).
  Compare shock-type composition *within* a site before reading anything
  across them.
* **A code is not timed to the shock.** `clif_hospital_diagnosis` is
  hospitalization-level, so a diagnosis recorded on day 20 is indistinguishable
  from one recorded on admission. `poa_present` narrows this when it is
  populated, which it is for all UCMC rows but only 61% of NU's, so it is
  reported rather than applied.
"""

from __future__ import annotations

import polars as pl

from clifcs.clif import has_table, scan

#: the mutually exclusive label, in plotting order
#:
#: `mixed` and `none coded` are derived rather than coded: see `_resolve`.
SHOCK_TYPES: tuple[str, ...] = (
    "septic",
    "cardiogenic",
    "hypovolemic",
    "mixed",
    "other shock",
    "none coded",
)

#: the three arms that name a mechanism. A patient carrying more than one of
#: these is `mixed`; "other shock" is not a mechanism and does not count toward
#: it, because a specific code alongside R57.8 is still that specific code.
_SPECIFIC = ("septic", "cardiogenic", "hypovolemic")


def _normalized_code() -> pl.Expr:
    """ICD code with dots and whitespace stripped, upper-cased"""
    return (
        pl.col("diagnosis_code")
        .cast(pl.String)
        .str.replace_all(r"[.\s]", "")
        .str.to_uppercase()
    )


def shock_type_flags(data_dir, cohort: pl.DataFrame, cfg, logger) -> pl.DataFrame:
    """
    one boolean per shock-type arm, plus the resolved `shock_type` label

    Returns a frame keyed on hospitalization_id covering every row of `cohort`,
    so the join in 01 cannot drop patients.

    Matching is by code prefix over both ICD-10-CM and ICD-9-CM without
    consulting `diagnosis_code_format`. The prefixes do not collide across the
    two systems -- ICD-9 shock codes are numeric (785.52) and ICD-10's start
    with a letter (R65.21) -- so the format column adds nothing and relying on
    it would make the match depend on a column NU populates with two values and
    UCMC with one.
    """
    ids = cohort.select("hospitalization_id")
    spec = cfg.shock_type

    empty = ids.with_columns(
        **{f"shock_{arm}": pl.lit(False) for arm in spec.code_groups},
    )
    if not has_table(data_dir, "clif_hospital_diagnosis"):
        logger.warning(
            "clif_hospital_diagnosis is absent; shock type cannot be assigned "
            "and every patient will be labelled 'none coded'"
        )
        return _resolve(empty, spec)

    frame = (
        scan(data_dir, "clif_hospital_diagnosis")
        .join(ids.lazy(), on="hospitalization_id", how="inner")
        .select("hospitalization_id", code=_normalized_code())
    )
    if spec.require_present_on_arrival:
        frame = (
            scan(data_dir, "clif_hospital_diagnosis")
            .join(ids.lazy(), on="hospitalization_id", how="inner")
            .filter(pl.col("poa_present") == 1)
            .select("hospitalization_id", code=_normalized_code())
        )

    hits = (
        frame.group_by("hospitalization_id")
        .agg(
            **{
                f"shock_{arm}": pl.any_horizontal(
                    *(
                        pl.col("code").str.starts_with(str(prefix))
                        for prefix in prefixes
                    )
                ).any()
                for arm, prefixes in spec.code_groups.items()
            }
        )
        .collect()
    )

    out = ids.join(hits, on="hospitalization_id", how="left", validate="1:1")
    out = out.with_columns(
        **{
            f"shock_{arm}": pl.col(f"shock_{arm}").fill_null(False)
            for arm in spec.code_groups
        }
    )
    return _resolve(out, spec)


def _resolve(flags: pl.DataFrame, spec) -> pl.DataFrame:
    """
    collapse the per-arm booleans into one mutually exclusive label

    Order matters: more than one specific mechanism is `mixed`, exactly one is
    that mechanism, otherwise `other shock` if a non-specific shock code is
    present, otherwise `none coded`.
    """
    specific = [f"shock_{arm}" for arm in _SPECIFIC if f"shock_{arm}" in flags.columns]
    n_specific = pl.sum_horizontal(*(pl.col(c).cast(pl.Int32) for c in specific))

    label = pl.when(n_specific >= 2).then(pl.lit("mixed"))
    for arm in _SPECIFIC:
        column = f"shock_{arm}"
        if column in flags.columns:
            label = label.when(pl.col(column)).then(pl.lit(arm))
    if "shock_other" in flags.columns:
        label = label.when(pl.col("shock_other")).then(pl.lit("other shock"))

    return flags.with_columns(
        n_shock_codes=n_specific,
        shock_type=label.otherwise(pl.lit("none coded")).cast(pl.Enum(SHOCK_TYPES)),
    )


def summarize(cohort: pl.DataFrame, logger) -> pl.DataFrame:
    """
    shock-type composition and outcomes, per site and pooled

    Printed per site because the coding-practice gap between the sites is large
    enough that a pooled composition is not a description of either of them.
    """
    rows = []
    for site in [*sorted(set(cohort["site"].to_list())), "pooled"]:
        part = cohort if site == "pooled" else cohort.filter(pl.col("site") == site)
        summary = (
            part.group_by("shock_type")
            .agg(
                n=pl.len(),
                mortality=pl.col("in_hospital_mortality").mean(),
                median_icu_los_days=pl.col("icu_los_days").median(),
            )
            .with_columns(site=pl.lit(site), share=pl.col("n") / part.height)
            .sort("shock_type")
        )
        rows.append(summary)
    out = pl.concat(rows).select(
        "site", "shock_type", "n", "share", "mortality", "median_icu_los_days"
    )

    logger.info("shock type from coded diagnoses (labelling only, not a filter):")
    for site in [*sorted(set(cohort["site"].to_list())), "pooled"]:
        logger.info(f"  -- {site}")
        for row in out.filter(pl.col("site") == site).iter_rows(named=True):
            logger.info(
                f"     {str(row['shock_type']):<13} n={row['n']:<7} "
                f"({row['share']:>5.1%})  mortality {row['mortality']:>5.1%}  "
                f"median ICU LOS {row['median_icu_los_days']:>4.1f}d"
            )

    # the gap this module's docstring warns about, recomputed every run rather
    # than quoted from a one-off check
    sites = sorted(set(cohort["site"].to_list()))
    if len(sites) == 2:
        shares = {
            site: (
                cohort.filter(pl.col("site") == site)["shock_type"] == "septic"
            ).mean()
            for site in sites
        }
        lo, hi = sorted(shares.values())
        if lo > 0 and hi / lo > 1.5:
            logger.warning(
                "septic-shock coding differs "
                f"{hi / lo:.1f}x between sites ("
                + ", ".join(f"{s} {v:.1%}" for s, v in shares.items())
                + "); this is coding practice as much as case mix, so compare "
                "shock-type composition within a site, not across them"
            )
    return out

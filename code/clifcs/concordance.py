#!/usr/bin/env python3

"""
was the patient treated the way the guideline says, and did it matter?

Three things live here, in the order 11 uses them.

  SEVERITY      a score to adjust outcomes by, built to exclude the
                circulation -- see `non_cv_sofa` for why that is not optional.
  CONCORDANCE   whether the treatment the guideline recommends was actually
                delivered, per trial, from procedure and medication records.
  OUTCOMES      mortality and ICU stay for patients inside versus outside the
                trial's published population, among those treated per the
                guideline, adjusted for severity.

THE CENTRAL DIFFICULTY, stated up front because it limits every number below.
Patients fall outside a trial's published interquartile ranges here almost
entirely because their blood pressure is lower, and low blood pressure *is*
severity. So "outside the IQR" and "sicker" are close to the same statement, and
any comparison of their outcomes is confounded by construction. Adjusting for
total SOFA would make it worse rather than better: SOFA's cardiovascular
component is scored from mean arterial pressure and vasopressor dose, so it
absorbs the exposure. What is adjusted for here is severity in the *other* five
organ systems, which is the closest available thing to "how sick is this patient
apart from their circulation". The unadjusted figures are reported beside it.

Even so, this is an observational comparison between groups that differ on the
thing that defines them, and it cannot support a causal reading. It can support
a descriptive one: whether the patients a guideline was not tested on do worse
than the ones it was, among patients who got the recommended treatment.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from clifcs.clif import has_table, scan

#: SOFA components other than cardiovascular
#:
#: The cardiovascular component is scored from MAP and vasopressor dose, which
#: is what separates inside from outside a trial's pressure quartiles. Leaving
#: it in the adjuster would remove the very difference being measured.
NON_CV_SOFA_COMPONENTS = (
    "sofa_coag",
    "sofa_liver",
    "sofa_resp",
    "sofa_cns",
    "sofa_renal",
)

#: procedure code sets for the treatments the guidelines turn on
#:
#: Separate from clifcs.cohort's combined MCS flag, because the guidelines
#: point in opposite directions for different devices: against routine IABP and
#: routine ECMO, and in favour of a microaxial pump. A single "any MCS" flag
#: cannot express that.
PROCEDURE_CODES: dict[str, dict[str, tuple]] = {
    "iabp": {
        "cpt": ("33967", "33970", "33971", "33973"),
        "icd10pcs": ("5A02210", "5A02216", "5A0221D"),
    },
    "ecmo": {
        "cpt": tuple(str(c) for c in range(33946, 33990)),
        "icd10pcs": ("5A15",),
    },
    "microaxial": {  # Impella, TandemHeart and other percutaneous VADs
        "cpt": ("33990", "33991", "33992", "33993", "33995", "33997"),
        "icd10pcs": ("02HA",),
    },
    "revascularisation": {  # PCI or CABG
        "cpt": tuple(str(c) for c in range(92920, 92944))
        + tuple(str(c) for c in range(33510, 33549)),
        "icd10pcs": ("0270", "0271", "0272", "0273", "021"),
    },
}


def attach_severity(paths, cohort: pl.DataFrame, logger) -> pl.DataFrame:
    """
    join the SOFA score the `derive` stage already computed

    Read from the per-site scratch subset rather than recomputed, so the score
    adjusting the outcomes is the same one the model read as tokens. Patients
    clifpy produced no score for keep a null, which propagates into the
    adjusted comparison as an exclusion rather than a zero.
    """
    frames = []
    for site in paths.sites:
        path = paths.raw_subset(site) / "clif_sofa.parquet"
        if not path.is_file():
            logger.warning(
                f"[{site}] clif_sofa.parquet absent; run 01 --stage derive. "
                "Outcomes will be reported unadjusted only."
            )
            continue
        frame = pl.read_parquet(path)
        keep = ["hospitalization_id", "sofa_total", *NON_CV_SOFA_COMPONENTS]
        missing = [c for c in keep if c not in frame.columns]
        if missing:
            logger.warning(f"[{site}] clif_sofa is missing {missing}; skipped")
            continue
        frames.append(
            frame.select(keep).with_columns(
                pl.col("hospitalization_id").cast(pl.String),
                site=pl.lit(site),
            )
        )
    if not frames:
        return cohort.with_columns(
            sofa_total=pl.lit(None, dtype=pl.Int32),
            sofa_non_cv=pl.lit(None, dtype=pl.Int32),
        )

    sofa = pl.concat(frames, how="diagonal_relaxed").unique(
        subset=["hospitalization_id", "site"], keep="first"
    )
    out = cohort.join(sofa, on=["hospitalization_id", "site"], how="left").with_columns(
        sofa_non_cv=pl.sum_horizontal(
            *(pl.col(c).cast(pl.Int32) for c in NON_CV_SOFA_COMPONENTS)
        )
    )
    covered = out["sofa_total"].is_not_null().mean()
    logger.info(
        f"  severity: SOFA joined for {covered:.1%} of the cohort; "
        f"median total {out['sofa_total'].median()}, "
        f"median non-cardiovascular {out['sofa_non_cv'].median()}"
    )
    return out


def attach_procedures(data_dir, cohort: pl.DataFrame, logger) -> pl.DataFrame:
    """
    one flag per device or procedure the guidelines turn on

    UCMC's CPT rows do not key on hospitalization_id -- 1.72M ids, none of
    which match a hospitalization -- so at that site only the ICD-10-PCS arm
    contributes. 01 warns about this already; the consequence here is that any
    CPT-only procedure is undercounted at UCMC, which matters most for
    revascularisation, where the PCI codes are CPT.
    """
    ids = cohort.select("hospitalization_id")
    names = list(PROCEDURE_CODES)
    if not has_table(data_dir, "clif_patient_procedures"):
        logger.warning("clif_patient_procedures absent; no procedure flags")
        return cohort.with_columns(
            **{f"had_{name}": pl.lit(None, dtype=pl.Boolean) for name in names}
        )

    code = pl.col("procedure_code").cast(pl.String).str.replace_all(r"\.", "")
    fmt = (
        pl.col("procedure_code_format")
        .cast(pl.String)
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]", "")
    )
    flags = {}
    for name, spec in PROCEDURE_CODES.items():
        cpt = [str(c) for c in spec.get("cpt", ())]
        pcs = [str(c) for c in spec.get("icd10pcs", ())]
        condition = pl.lit(False)
        if cpt:
            condition = condition | ((fmt == "cpt") & code.is_in(cpt))
        if pcs:
            condition = condition | (
                (fmt == "icd10pcs")
                & pl.any_horizontal(*(code.str.starts_with(p) for p in pcs))
            )
        flags[f"had_{name}"] = condition.any()

    found = (
        scan(data_dir, "clif_patient_procedures")
        .join(ids.lazy(), on="hospitalization_id", how="inner")
        .group_by("hospitalization_id")
        .agg(**flags)
        .collect()
    )
    out = cohort.join(found, on="hospitalization_id", how="left").with_columns(
        **{f"had_{name}": pl.col(f"had_{name}").fill_null(False) for name in names}
    )
    logger.info(
        "  procedures: "
        + ", ".join(f"{n} {out[f'had_{n}'].mean():.1%}" for n in names)
    )
    return out


# ----------------------------------------------------------------- concordance
#
# What each trial's recommendation asks of a clinician, expressed over the
# columns above. `None` means the recommendation is not ascertainable here and
# the trial is reported without a concordance arm rather than guessed at.
#
# Direction matters and is easy to get backwards: four of these trials were
# NEGATIVE, so following them means *not* doing something. IABP-SHOCK II,
# ECMO-CS and ECLS-SHOCK are concordant when the device was withheld; DanGer
# Shock is the only device trial where concordance means the device was used.


def concordance_expressions() -> dict[str, tuple[pl.Expr | None, str]]:
    """per trial: the concordance test, and a sentence describing it"""
    return {
        "shock": (
            pl.col("had_revascularisation"),
            "early revascularisation performed (PCI or CABG coded)",
        ),
        "soap_ii": (
            pl.col("on_norepinephrine") & ~pl.col("on_dopamine"),
            "norepinephrine used and dopamine avoided",
        ),
        "iabp_shock_ii": (
            ~pl.col("had_iabp"),
            "intra-aortic balloon pump withheld (the trial was negative)",
        ),
        "culprit_shock": (
            None,
            "culprit-lesion-only versus multivessel PCI is not ascertainable: "
            "CLIF 2.1 carries no lesion-level detail",
        ),
        "optimacc": (
            pl.col("on_norepinephrine") & ~pl.col("on_epinephrine"),
            "norepinephrine used and epinephrine avoided",
        ),
        "ecmo_cs": (
            ~pl.col("had_ecmo"),
            "immediate VA-ECMO withheld (the trial was negative)",
        ),
        "ecls_shock": (
            ~pl.col("had_ecmo"),
            "routine extracorporeal life support withheld (the trial was negative)",
        ),
        "danger_shock": (
            pl.col("had_microaxial"),
            "microaxial flow pump used (the one positive device trial)",
        ),
        "sepsispam": (
            pl.col("map_achieved") >= 65.0,
            "median MAP over the 24 h after onset at least 65 mmHg "
            "(the pressure achieved, not the nadir that defined the shock)",
        ),
        "vasst": (
            pl.col("on_norepinephrine") & pl.col("on_vasopressin"),
            "vasopressin added to norepinephrine",
        ),
        "adrenal": (
            pl.col("on_hydrocortisone"),
            "hydrocortisone given within the shock window",
        ),
        # Both hypovolaemic trials turn on something this extract does not
        # record. Reported as unascertainable rather than guessed: neither site
        # has a transfusion or blood product table, and tranexamic acid is not
        # a med_category at either.
        "villanueva": (
            None,
            "transfusion threshold is not ascertainable: no blood product "
            "table at either site",
        ),
        "crash_2": (
            None,
            "tranexamic acid is not recorded as a med_category at either site",
        ),
    }


def add_concordance(frame: pl.DataFrame, trials, logger) -> pl.DataFrame:
    """one boolean per trial: was the recommended treatment delivered?"""
    tests = concordance_expressions()
    columns = {}
    notes = {}
    for trial in trials:
        expression, note = tests.get(trial.key, (None, "not defined"))
        notes[trial.key] = note
        columns[f"concordant_{trial.key}"] = (
            expression.fill_null(False)
            if expression is not None
            else pl.lit(None, dtype=pl.Boolean)
        )
    out = frame.with_columns(**columns)

    logger.info("treatment concordance among each trial's eligible patients:")
    for trial in trials:
        column = f"concordant_{trial.key}"
        eligible = out.filter(pl.col(f"elig_{trial.key}"))
        if eligible.height == 0 or eligible[column].null_count() == eligible.height:
            logger.info(f"  {trial.label:<22} not ascertainable -- {notes[trial.key]}")
            continue
        logger.info(
            f"  {trial.label:<22} {eligible[column].mean():>6.1%} of "
            f"{eligible.height:,} eligible  ({notes[trial.key]})"
        )
    return out


# --------------------------------------------------------------------- outcomes


def _standardised_outcome(
    frame: pl.DataFrame, group: str, outcome: str, strata: str, logger
) -> tuple[float | None, float | None]:
    """
    direct standardisation of `outcome` across strata of `strata`

    Chosen over a regression because it needs no functional form and because
    the strata here are a small integer score: a weighted average over strata
    of the cohort is transparent and its failure mode -- an empty stratum -- is
    visible rather than smoothed over.

    Returns the standardised outcome for group False and True.
    """
    usable = frame.drop_nulls([group, outcome, strata])
    if usable.height == 0:
        return None, None
    weights = (
        usable.group_by(strata)
        .agg(weight=pl.len())
        .with_columns(weight=pl.col("weight") / usable.height)
    )
    per_cell = usable.group_by([strata, group]).agg(
        value=pl.col(outcome).cast(pl.Float64).mean(), n=pl.len()
    )
    joined = per_cell.join(weights, on=strata, how="inner")
    out = []
    for flag in (False, True):
        part = joined.filter(pl.col(group) == flag)
        if part.height == 0:
            out.append(None)
            continue
        # renormalise over the strata this group actually occupies, so a group
        # absent from a stratum does not silently inherit the other's rate
        total = part["weight"].sum()
        out.append(
            float((part["value"] * part["weight"]).sum() / total) if total else None
        )
    return out[0], out[1]


def outcomes_by_position(frame: pl.DataFrame, trials, logger) -> pl.DataFrame:
    """
    outcomes inside versus outside each trial's published population

    Restricted to patients who were eligible *and* treated per the guideline,
    which is the population the question is about: among people a guideline
    nominally covers and who got what it recommends, does resembling the trial's
    patients track with doing better?

    Reported three ways -- crude, standardised by non-cardiovascular SOFA, and
    standardised by age band -- because no single one of them is trustworthy on
    its own here. See the module docstring on why the exposure and the
    confounder are close to the same variable.
    """
    rows = []
    frame = frame.with_columns(
        age_band=(pl.col("age_at_admission") / 10).floor().cast(pl.Int32)
    )
    for trial in trials:
        inside_col = f"t1_outside_{trial.key}"
        compared = f"t1_n_compared_{trial.key}"
        concordant = f"concordant_{trial.key}"
        if inside_col not in frame.columns:
            continue
        part = frame.filter(
            pl.col(f"elig_{trial.key}")
            & (pl.col(compared) > 0)
            & pl.col(inside_col).is_not_null()
        )
        if part.height < 50:
            continue
        part = part.with_columns(outside=pl.col(inside_col) > 0)

        for arm, subset in (
            ("all eligible", part),
            (
                "treated per guideline",
                part.filter(pl.col(concordant))
                if concordant in part.columns
                and part[concordant].null_count() < part.height
                else None,
            ),
        ):
            if subset is None or subset.height < 50:
                continue
            crude = subset.group_by("outside").agg(
                n=pl.len(),
                mortality=pl.col("in_hospital_mortality").mean(),
                icu_los=pl.col("icu_los_days").median(),
                sofa_non_cv=pl.col("sofa_non_cv").median(),
            )
            inside_row = crude.filter(~pl.col("outside"))
            outside_row = crude.filter(pl.col("outside"))
            if inside_row.height == 0 or outside_row.height == 0:
                continue
            # Both arms need enough patients for a mortality difference to mean
            # anything. ECLS-SHOCK puts 6 of its concordant patients inside its
            # published IQRs and DanGer Shock 10, which produce differences of
            # +26 and +8 points that are entirely sampling noise. Flagged
            # rather than dropped, so the reader sees the trial was examined.
            reliable = min(int(inside_row["n"][0]), int(outside_row["n"][0])) >= 50

            adj_in, adj_out = _standardised_outcome(
                subset, "outside", "in_hospital_mortality", "sofa_non_cv", logger
            )
            age_in, age_out = _standardised_outcome(
                subset, "outside", "in_hospital_mortality", "age_band", logger
            )
            rows.append(
                {
                    "trial": trial.label,
                    "arm": arm,
                    "reliable": reliable,
                    "n_inside": int(inside_row["n"][0]),
                    "n_outside": int(outside_row["n"][0]),
                    "mortality_inside": float(inside_row["mortality"][0]),
                    "mortality_outside": float(outside_row["mortality"][0]),
                    "mortality_difference": float(outside_row["mortality"][0])
                    - float(inside_row["mortality"][0]),
                    "mortality_inside_sofa_adj": adj_in,
                    "mortality_outside_sofa_adj": adj_out,
                    "mortality_difference_sofa_adj": (
                        adj_out - adj_in
                        if adj_in is not None and adj_out is not None
                        else None
                    ),
                    "mortality_difference_age_adj": (
                        age_out - age_in
                        if age_in is not None and age_out is not None
                        else None
                    ),
                    "icu_los_inside": float(inside_row["icu_los"][0]),
                    "icu_los_outside": float(outside_row["icu_los"][0]),
                    "sofa_non_cv_inside": float(inside_row["sofa_non_cv"][0])
                    if inside_row["sofa_non_cv"][0] is not None
                    else None,
                    "sofa_non_cv_outside": float(outside_row["sofa_non_cv"][0])
                    if outside_row["sofa_non_cv"][0] is not None
                    else None,
                }
            )

    out = pl.DataFrame(rows) if rows else pl.DataFrame()
    if out.height:
        logger.info(
            "mortality inside vs outside the published population "
            "(crude, then standardised by non-cardiovascular SOFA):"
        )
        for row in out.iter_rows(named=True):
            adj = (
                f"{row['mortality_difference_sofa_adj']:+.1%}"
                if row["mortality_difference_sofa_adj"] is not None
                else "   n/a"
            )
            mark = "" if row["reliable"] else "   [too few inside to read]"
            logger.info(
                f"  {row['trial']:<22} {row['arm']:<22} "
                f"inside {row['mortality_inside']:>5.1%} (n={row['n_inside']:,}) vs "
                f"outside {row['mortality_outside']:>5.1%} (n={row['n_outside']:,})  "
                f"crude {row['mortality_difference']:+.1%}, SOFA-adjusted {adj}  "
                f"| non-CV SOFA {row['sofa_non_cv_inside']:.0f} vs "
                f"{row['sofa_non_cv_outside']:.0f}{mark}"
            )
    return out

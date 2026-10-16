#!/usr/bin/env python3

"""
the trial analysis as aggregates, computed site-side

Everything a trial comparison needs to say is a proportion or a median over a
site's own patients:

  * does this patient meet the inclusion criteria -- a count and a share;
  * do they sit inside the published interquartile ranges -- a share, per
    variable and across all of them;
  * were they treated in line with the guideline -- a share;
  * mortality and ICU length of stay -- a rate and a median.

None of that requires a patient to leave the institution. An earlier version of
this pipeline said the published-baseline comparison could not survive
federation, which was wrong: what cannot travel is the measured *value*, and
every quantity above is a statistic over values rather than a value. So this
runs at the site and ships tables.

That matters most for the result the README leads with. The blood-pressure gap
against OptimaCC, ECLS-SHOCK and CULPRIT-SHOCK is "what share of our eligible
patients sit below the trial's first quartile", against the 25% a representative
sample would give. That is one number per trial per variable, and it is
available from a site that will never share a single patient.

STRATIFICATION

Every table is produced overall and broken down by each phenotype system --
Zweck, Seymour, Bhavani -- and by coded shock type. A trial's evidence is about
a population, and the question "is this trial's population like ours" has a
different answer inside each phenotype. The septic-shock trials are in the same
table as the cardiogenic ones for exactly that reason: VASST, SEPSISPAM and
ADRENAL are the evidence base for the septic-coded patients, who are 18% of this
cohort, and reporting only the cardiogenic trials would leave them uncovered by
construction.

SUPPRESSION

Stratifying 13 trials by four phenotype systems produces cells of one or two
patients, so every table goes through the same minimum cell size as the counts.
A share over six patients is a statement about those six.
"""

from __future__ import annotations

import polars as pl

from clifcs.export import suppress
from clifcs.trials import (
    VARIABLE_CONFIDENCE,
    inside_all_null,
    table1_variables,
)

#: how each table is broken down. `None` is the whole cohort.
#:
#: Ordered so the overall row comes first in the output, which is the one a
#: reader wants before any subgroup.
STRATA: dict[str, str | None] = {
    "overall": None,
    "zweck": "cs_phenotype",
    "seymour": "seymour_phenotype",
    "bhavani": "bhavani_phenotype",
    "coded_shock_type": "shock_type",
}


def _strata_frames(frame: pl.DataFrame):
    """yield (scheme, level, subset) for every stratum present"""
    for scheme, column in STRATA.items():
        if column is None:
            yield scheme, "(all)", frame
            continue
        if column not in frame.columns:
            continue
        for level in frame[column].cast(pl.String).unique().drop_nulls().sort():
            yield scheme, level, frame.filter(pl.col(column).cast(pl.String) == level)


def outcomes_table(
    frame: pl.DataFrame, trials, site: str, anchor: str, logger
) -> pl.DataFrame:
    """
    per trial and stratum: who qualifies, how they did, and whether the
    recommended treatment was delivered

    `share_eligible` is of the stratum, not of the cohort, so it answers "what
    fraction of Seymour-delta patients would this trial have enrolled" rather
    than restating the stratum's size.
    """
    rows = []
    for scheme, level, subset in _strata_frames(frame):
        if subset.height == 0:
            continue
        for trial in trials:
            key = trial.key
            elig_col = f"elig_{key}"
            if elig_col not in subset.columns:
                continue
            eligible = subset.filter(pl.col(elig_col).fill_null(False))

            row = {
                "site": site,
                "anchor": anchor,
                "scheme": scheme,
                "level": level,
                "trial": trial.label,
                "trial_key": key,
                "applies_to": trial.applies_to,
                "n_stratum": subset.height,
                "n_eligible": eligible.height,
                "share_eligible": eligible.height / subset.height,
                "mortality_eligible": float(
                    eligible["in_hospital_mortality"].mean()
                )
                if eligible.height
                else None,
                "median_icu_los_eligible": float(eligible["icu_los_days"].median())
                if eligible.height
                else None,
            }

            # treated in line with the guideline, among those who qualified.
            # Null rather than zero where the intervention is not ascertainable
            # here -- "we cannot tell" and "it was never done" are different.
            conc = f"concordant_{key}"
            if conc in eligible.columns and eligible.height:
                known = eligible.filter(pl.col(conc).is_not_null())
                row["n_concordance_known"] = known.height
                row["share_concordant"] = (
                    float(known[conc].mean()) if known.height else None
                )
                treated = known.filter(pl.col(conc))
                untreated = known.filter(~pl.col(conc))
                row["mortality_concordant"] = (
                    float(treated["in_hospital_mortality"].mean())
                    if treated.height
                    else None
                )
                row["mortality_discordant"] = (
                    float(untreated["in_hospital_mortality"].mean())
                    if untreated.height
                    else None
                )
            else:
                row["n_concordance_known"] = 0
                row["share_concordant"] = None
                row["mortality_concordant"] = None
                row["mortality_discordant"] = None

            rows.append(row | _inside_all(eligible, key))

    table = pl.DataFrame(rows)
    if table.height:
        logger.info(
            f"[{site}/{anchor}] trial outcomes: {table.height} trial x stratum "
            f"rows across {len(trials)} trials and {len(STRATA)} schemes"
        )
    return table


def _inside_all(eligible: pl.DataFrame, key: str) -> dict:
    """
    the share of eligible patients inside every published IQR, with its null

    Reported with `null_inside_all_iqrs` beside it because the raw share is
    unreadable alone: it falls as 0.5^k with every variable added whether or not
    the cohort differs at all, so 2.4% across seven variables is three times
    what a representative sample would give, not a tenth.
    """
    outside = f"t1_outside_{key}"
    compared = f"t1_n_compared_{key}"
    if outside not in eligible.columns or eligible.height == 0:
        return {
            "n_comparable": 0,
            "n_variables_compared": None,
            "share_inside_all_iqrs": None,
            "null_inside_all_iqrs": None,
            "ratio_to_null": None,
        }
    usable = eligible.filter(pl.col(compared) > 0)
    if usable.height == 0:
        return {
            "n_comparable": 0,
            "n_variables_compared": None,
            "share_inside_all_iqrs": None,
            "null_inside_all_iqrs": None,
            "ratio_to_null": None,
        }
    k = int(usable[compared].median())
    observed = float((usable[outside] == 0).mean())
    null = inside_all_null(k)
    return {
        "n_comparable": usable.height,
        "n_variables_compared": k,
        "share_inside_all_iqrs": observed,
        "null_inside_all_iqrs": null,
        "ratio_to_null": observed / null if null > 0 else None,
    }


def iqr_position_table(
    frame: pl.DataFrame, trials, site: str, anchor: str, logger
) -> pl.DataFrame:
    """
    per trial, variable and stratum: where our eligible patients sit relative to
    the published interquartile range

    This is the table the blood-pressure result is read off. For each published
    variable it gives the share of eligible patients below the trial's first
    quartile, inside the range, and above the third -- against the 25/50/25 a
    representative sample would give.

    Our median is reported beside the trial's so the direction is legible
    without arithmetic, and a median is a statistic rather than a value, so it
    travels on the same terms as every share here.

    Only `measured` variables. A trial's medical-history rows come from a case
    report form and ours from ICD codes, and they do not measure the same thing
    in either direction -- the filter is explicit so that adding a history
    variable to trials.yaml cannot quietly start driving this.
    """
    rows = []
    for scheme, level, subset in _strata_frames(frame):
        if subset.height == 0:
            continue
        for trial in trials:
            elig_col = f"elig_{trial.key}"
            if elig_col not in subset.columns:
                continue
            eligible = subset.filter(pl.col(elig_col).fill_null(False))
            if eligible.height == 0:
                continue

            spec = table1_variables(trial)
            for name, published in spec.items():
                if "iqr" not in published or name not in eligible.columns:
                    continue
                if VARIABLE_CONFIDENCE.get(name) != "measured":
                    continue
                values = eligible[name].cast(pl.Float64).drop_nulls()
                if values.len() == 0:
                    continue
                q1, q3 = (float(x) for x in published["iqr"])
                rows.append(
                    {
                        "site": site,
                        "anchor": anchor,
                        "scheme": scheme,
                        "level": level,
                        "trial": trial.label,
                        "trial_key": trial.key,
                        "applies_to": trial.applies_to,
                        "variable": name,
                        "n_compared": values.len(),
                        "published_median": float(published.get("median"))
                        if published.get("median") is not None
                        else None,
                        "published_q1": q1,
                        "published_q3": q3,
                        "median_here": float(values.median()),
                        # the three shares a representative sample would put at
                        # 25 / 50 / 25
                        "share_below_q1": float((values < q1).mean()),
                        "share_inside_iqr": float(
                            ((values >= q1) & (values <= q3)).mean()
                        ),
                        "share_above_q3": float((values > q3).mean()),
                    }
                )

    table = pl.DataFrame(rows)
    if table.height:
        _report_shift(table, site, anchor, logger)
    return table


def _report_shift(table: pl.DataFrame, site: str, anchor: str, logger) -> None:
    """log the variables where the eligible patients are most displaced"""
    overall = table.filter(pl.col("scheme") == "overall")
    if overall.height == 0:
        return
    shifted = overall.filter(pl.col("share_below_q1") > 0.5).sort(
        "share_below_q1", descending=True
    )
    logger.info(
        f"[{site}/{anchor}] position against published IQRs "
        f"(null is 25% below Q1):"
    )
    for row in overall.sort("share_below_q1", descending=True).head(8).iter_rows(
        named=True
    ):
        logger.info(
            f"    {row['trial']:<22} {row['variable']:<20} "
            f"ours {row['median_here']:>7.1f} vs {row['published_median'] or float('nan'):>7.1f}  "
            f"below Q1 {row['share_below_q1']:>6.1%}  (n={row['n_compared']})"
        )
    if shifted.height:
        logger.warning(
            f"[{site}/{anchor}] {shifted.height} trial-variable pair(s) have "
            "over half of eligible patients below the published first quartile "
            "-- the result was not tested on patients like these: "
            + ", ".join(
                f"{r['trial']}/{r['variable']} {r['share_below_q1']:.0%}"
                for r in shifted.head(5).iter_rows(named=True)
            )
        )


#: statistics suppressed wherever the count they are computed over is
_DERIVED_OUTCOMES = (
    "share_eligible",
    "mortality_eligible",
    "median_icu_los_eligible",
    "share_concordant",
    "mortality_concordant",
    "mortality_discordant",
    "share_inside_all_iqrs",
    "null_inside_all_iqrs",
    "ratio_to_null",
    "n_variables_compared",
)
_DERIVED_IQR = (
    "median_here",
    "share_below_q1",
    "share_inside_iqr",
    "share_above_q3",
)


def build(
    paths, frame: pl.DataFrame, trials, site: str, anchor: str, logger, min_cell: int
) -> dict[str, pl.DataFrame]:
    """
    the two aggregate trial tables, suppressed and ready to ship

    `frame` must already carry the eligibility booleans, the concordance
    booleans and the `t1_*` position columns -- i.e. it is the frame 11 builds
    site-side, before any of it is reduced.
    """
    tables = {
        "trial_outcomes": outcomes_table(frame, trials, site, anchor, logger),
        "trial_iqr_position": iqr_position_table(frame, trials, site, anchor, logger),
    }

    suppressed = {}
    for name, table in tables.items():
        if table.height == 0:
            logger.warning(f"[{site}/{anchor}] {name}: no rows")
            continue
        derived = _DERIVED_OUTCOMES if name == "trial_outcomes" else _DERIVED_IQR
        # Suppression keyed on the count each statistic is computed over, which
        # differs between the two tables: eligibility shares are over
        # `n_eligible`, IQR shares over `n_compared`.
        count = "n_eligible" if name == "trial_outcomes" else "n_compared"
        table = table.with_columns(
            [
                pl.when(pl.col(count) < min_cell)
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in derived
                if c in table.columns
            ]
        )
        table = suppress.suppress_counts(
            table, min_cell=min_cell, derived=derived, logger=logger
        )
        suppressed[name] = table
        logger.info(f"[{site}/{anchor}] {name}: {table.height} rows")
    return suppressed

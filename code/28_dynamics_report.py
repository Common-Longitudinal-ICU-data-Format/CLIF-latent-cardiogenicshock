#!/usr/bin/env python3

"""
28_dynamics_report.py -- markdown summary of the action-response phases

One section per phase, each carrying its results, the assumptions it rests on,
and what is still open. Sections for phases that have not been built yet say so
rather than being omitted, because a reader of a partial report needs to know
whether a phase is absent because it found nothing or because it has not run.

    python code/28_dynamics_report.py --subset cardiogenic

MARKDOWN, NOT HTML, by choice. The two cohort reports are self-contained HTML
because they carry hundreds of embedded figures. This one is mostly numbers and
prose about method, which is better read in a diff -- the assumptions change as
the phases proceed and the change itself is worth being able to review.

THE CONFIG IS REPRODUCED IN FULL at the end. Every threshold in these results
is a choice, and a reader who cannot see the choices cannot judge the numbers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths

#: phase -> (title, the driver that produces it). Listed whether or not the
#: driver has run, so a missing phase reads as missing rather than as absent.
PHASES = (
    (1, "Action-response state encoding", "23_action_response.py"),
    (2, "Sensitivity trajectory", "24_sensitivity.py"),
    (3, "Absorbing Markov chain and point of no return", "25_absorbing.py"),
    (4, "What drives the drop", "26_leverage.py"),
    (5, "Baseline versus modifiable", "27_baseline_modifiable.py"),
)


def table(frame: pl.DataFrame, limit=None) -> str:
    """a frame as a GitHub-flavoured markdown table"""
    if frame.is_empty():
        return "_(no rows)_\n"
    if limit:
        frame = frame.head(limit)
    columns = frame.columns

    def cell(value):
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:,.4g}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    head = "| " + " | ".join(c.replace("_", " ") for c in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(cell(row[c]) for c in columns) + " |"
        for row in frame.iter_rows(named=True)
    ]
    return "\n".join([head, rule, *body]) + "\n"


def phase_one(paths, subset, tables, logger, out):
    """results, assumptions and open questions for the state encoding"""
    states_file = tables / f"action_response_states_{subset}.csv"
    encoded_file = (
        paths.patient_level_home / f"action_response_{subset}.parquet"
    )
    if not states_file.is_file():
        out.append("_Not run. Produces "
                   "`action_response_states_<subset>.csv`._\n")
        return
    states = pl.read_csv(states_file)
    encoded = pl.read_parquet(encoded_file)
    sequences = pl.read_parquet(
        paths.patient_level_home / f"action_response_sequences_{subset}.parquet"
    )
    decidable = encoded.filter(pl.col("state") != "undecidable")

    out.append(
        f"**{sequences.height:,} patients, {encoded.height:,} decision steps "
        f"within the record, {decidable.height:,} decidable "
        f"({decidable.height / max(encoded.height, 1):.1%}).**\n"
    )
    out.append("### State frequencies\n")
    out.append(table(states))

    escalations = decidable.filter(pl.col("action") == "escalate")
    weans = decidable.filter(pl.col("action") == "de-escalate")
    if escalations.height and weans.height:
        out.append(
            f"\nOf {escalations.height:,} escalations, "
            f"**{(escalations['state'] == 'responsive escalation').mean():.1%} "
            f"were responsive**. Of {weans.height:,} de-escalations, "
            f"**{(weans['state'] == 'tolerated wean').mean():.1%} were "
            "tolerated**. Both are minorities: escalation more often fails to "
            "move the pressure than moves it, and weans fail more often than "
            "they hold. That is the expected shape of confounding by "
            "indication rather than a finding -- a clinician escalates when "
            "the pressure is falling.\n"
        )

    out.append("\n### Titration counts, which gate phase 2\n")
    bands = [(0, 0, "0 (no information about s(t))"), (1, 2, "1-2 (too few)"),
             (3, 5, "3-5 (marginal)"), (6, 10, "6-10"), (11, 10 ** 6, "11+")]
    rows = [
        {
            "titrations": label,
            "patients": sequences.filter(
                pl.col("titrations").is_between(lo, hi)
            ).height,
            "share": sequences.filter(
                pl.col("titrations").is_between(lo, hi)
            ).height / max(sequences.height, 1),
        }
        for lo, hi, label in bands
    ]
    out.append(table(pl.DataFrame(rows)))
    thin = sequences.filter(pl.col("titrations") <= 2).height
    out.append(
        f"\n**{thin / max(sequences.height, 1):.1%} of patients have two or "
        "fewer titrations.** For them a time-varying sensitivity will be the "
        "partially-pooled prior plus a random walk, not an estimate. That is "
        "the correct behaviour of a hierarchical fit and it is also a "
        "limitation: a quarter of the cohort's sensitivity tercile in phase 3 "
        "would be assigned by shrinkage rather than measured, so the flag has "
        "to be carried into the stratification rather than buried in it.\n"
    )

    blind = (
        encoded.group_by("hospitalization_id")
        .agg(bad=(pl.col("state") == "undecidable").mean())
        .filter(pl.col("bad") == 1.0)
    )
    out.append(
        f"\n**{blind.height:,} patients "
        f"({blind.height / max(sequences.height, 1):.1%}) have no usable MAP "
        "at all** -- every step undecidable. They cannot contribute to any "
        "phase and should be excluded up front with a count reported, not "
        "dropped silently mid-pipeline.\n"
    )

    out.append("\n### Assumptions\n")
    for item in (
        "Off-pressor states are decided by MAP **level**, not change: the "
        "question for an unsupported patient is whether the pressure is "
        "adequate, not whether it moved three millimetres.",
        "A wean fails if the dose goes back up, even with MAP held. That is "
        "the clinician's own verdict; excluding it would score a wean as "
        "tolerated precisely when it was reversed fast enough to keep the MAP "
        "from moving.",
        "MAP missing beyond the tolerance is **null**; dose missing is "
        "**zero**. Asymmetric on purpose -- an absent infusion record means no "
        "infusion, an absent MAP means nobody measured.",
        "Undecidable steps are counted, never dropped: dropping them would "
        "make the state distribution one over steps that happened to be "
        "followed by more record.",
        "Vasopressin is **in** the NEE sum per Goradia and also carried as a "
        "flag, with `nee_titratable` published beside it. An earlier version "
        "excluded it on the grounds that it is not titrated, which conflated "
        "'not titrated' with 'not a dose' -- starting it is an escalation "
        "decision and its pressor effect is real.",
        "Dopamine is 1/100 here against 1/150 in `clifcs/labels.py`. A real "
        "disagreement between published tables, not a bug in either. **The "
        "conversion factors are reproduced from the formula as commonly "
        "cited and have not been checked against the paper.**",
    ):
        out.append(f"- {item}\n")

    out.append("\n### Open questions\n")
    for item in (
        "Systolic values are written into the MAP field and a range filter "
        "cannot catch them: among readings above 130 the median MAP and the "
        "median same-timestamp SBP are both 152, and 48% have MAP >= SBP. The "
        "plausibility cross-check drops 2.76% of UCMC readings against 0.38% "
        "at NU and 0.05% at RUSH -- a **55-fold site difference** in a field "
        "every state depends on, and because the error inflates MAP it pushes "
        "states toward 'adequate' unevenly across sites. **The 15-state "
        "perfusion encoding and the BPE vocabularies elsewhere in this project "
        "filter MAP on range alone and carry this contamination.**",
        "The MAP staleness tolerance is a judgement. Tightening it raises the "
        "undecidable share; loosening it manufactures flat responses that read "
        "as stable holds. Not yet sensitivity-tested.",
        "A step and its response window together span longer than one step, so "
        "**adjacent states are not independent**. Phase 3's Markov assumption "
        "has to state this rather than inherit it.",
    ):
        out.append(f"- {item}\n")


def phase_two(paths, subset, tables, logger, out):
    """the sensitivity fit, and why it does not identify a patient-level trait"""
    gain_file = tables / f"sensitivity_gain_summary_{subset}.csv"
    if not gain_file.is_file():
        out.append("_Not run. Produces "
                   "`sensitivity_gain_summary_<subset>.csv`._\n")
        return
    out.append(
        "**This phase returns a negative result, and the negative result is "
        "the finding.** Neither model identifies a patient-level dose "
        "sensitivity in this cohort, and the between-patient variance test "
        "below shows that this is not a shrinkage artefact or a sample-size "
        "problem.\n"
    )

    out.append("\n### The specified gain model\n")
    out.append(
        "`MAP(t) = baseline(t) + s(t) * f(NEE(t))`, both states random walks, "
        "fitted by Kalman smoother with variances pooled across patients by "
        "empirical Bayes. `data_share` is the fraction of each patient's "
        "posterior precision on *s* contributed by their own observations "
        "rather than the prior.\n\n"
    )
    out.append(table(pl.read_csv(gain_file)))
    out.append(
        "\n**Both saturating forms return a median *s* of zero** with over 90% "
        "of patient-hours inside one standard deviation of it. Emax is worse "
        "than log (median sd 19.1 against 12.3), which is expected: it spends "
        "a parameter on a bend point the data cannot place.\n"
        "\n**The fitted hyperparameters say why.** The optimiser chose a "
        "baseline random-walk sd of about **3.8 mmHg per hour** under both "
        "forms. Over a stay of several days that baseline is effectively a "
        "free function of time -- it tracks any pressure trajectory on its "
        "own, leaving the dose term nothing to explain. The baseline walk and "
        "the dose term compete for the same variance and the baseline wins. "
        "Constraining the walk would push variance onto *s*, but that assumes "
        "the answer rather than estimating it.\n"
    )

    direction_file = tables / f"sensitivity_direction_summary_{subset}.csv"
    if direction_file.is_file():
        out.append("\n### The direction model\n")
        out.append(
            "A time-varying probability that the pressure moved the way the "
            "dose was pushed, by exponentially weighted beta-binomial "
            "shrinkage toward the cohort rate. Built because the conditional "
            "means do separate -- MAP rises 1.8 mmHg in the hour after an "
            "escalation and falls 1.4 after a wean -- so direction carries "
            "signal where magnitude does not.\n\n"
        )
        out.append(table(pl.read_csv(direction_file)))
        out.append(
            "\nOf 67,596 titrations with a measured response, **52.4% went the "
            "way the dose was pushed** -- barely above a coin flip. The mean "
            "shift of 2-3 mmHg sits against an observation sd of 6.25 mmHg, so "
            "the distributions overlap almost entirely: the *means* separate "
            "while individual events do not.\n"
        )

    agreement_file = tables / f"sensitivity_agreement_{subset}.csv"
    if agreement_file.is_file():
        out.append("\n### Do the two models agree about who is sensitive?\n\n")
        out.append(table(pl.read_csv(agreement_file)))
        out.append(
            "\n**No.** Spearman -0.045 across all patients and -0.037 among "
            "those whose gain estimate clears the data_share floor. Two "
            "measures of the same supposed trait agreeing at essentially zero "
            "is what two noise series look like.\n"
        )

    out.append("\n### The decisive test: is there any between-patient variation?\n")
    out.append(
        "Shrinkage and thin data would both produce the results above while "
        "leaving a real trait underneath. This separates those cases by asking "
        "whether the RAW, unshrunk response rates vary more than sampling "
        "noise from one shared rate would make them vary.\n\n"
    )
    out.append(
        "| titrations per patient | patients | mean p | sd of p |\n"
        "|---|---|---|---|\n"
        "| 1-4 | 2,624 | 0.547 | 0.352 |\n"
        "| 5-9 | 2,391 | 0.540 | 0.179 |\n"
        "| 10-19 | 1,840 | 0.525 | 0.125 |\n"
        "| 20+ | 765 | 0.506 | 0.086 |\n"
    )
    out.append(
        "\nThe spread shrinks as 1/sqrt(n) -- the signature of one common rate "
        "observed with varying precision, not of a trait measured with varying "
        "precision.\n"
        "\nAmong the 2,605 patients with at least ten titrations the observed "
        "sd of the response rate is **0.115**, against **0.129** expected if "
        "every patient shared the same rate of 0.516. The observed spread is "
        "*smaller* than the binomial expectation, so the estimated "
        "between-patient variance is **zero**. A true spread of the size a "
        "tercile needs -- an sd of 0.10 in true p -- would have produced an "
        "observed sd near 0.163 and been plainly visible.\n"
    )

    out.append("\n### Assumptions\n")
    for item in (
        "ED50 and the log reference dose are **pooled, not per-patient**: a "
        "per-patient bend point cannot be identified when the per-patient gain "
        "cannot be, and fitting one yields a number that moves only because "
        "the optimiser had to put it somewhere.",
        "The slope prior is centred on **zero**, not on a positive expected "
        "response, so a patient with no usable titrations reads as 'no "
        "measured sensitivity' rather than 'average sensitivity'.",
        "Empirical Bayes, not full Bayes: hyperparameters are estimated once "
        "and then treated as known, so the bands on s(t) exclude "
        "hyperparameter uncertainty and are slightly too narrow.",
        "Hyperparameters are fitted on a 400-patient subsample; the likelihood "
        "is a sum over patients and three parameters pin tightly well before "
        "that.",
        "Missing MAP hours are predicted through rather than skipped, so "
        "sparse monitoring widens the bands instead of shortening the series.",
    ):
        out.append(f"- {item}\n")

    out.append("\n### Consequences for phase 3\n")
    out.append(
        "**The sensitivity tercile in the transient-state definition cannot be "
        "built.** Neither model supports it and the variance test says there "
        "is no trait to tercile. Stratifying on it anyway would produce three "
        "groups that always split and never mean anything -- and because "
        "terciles always split, nothing downstream would reveal the variable "
        "was empty.\n"
        "\nWhat this does **not** say: that dose does not raise blood "
        "pressure. It says that *within this cohort, as measured*, patients "
        "are not distinguishable from one another in how far their pressure "
        "moves per unit of dose. Per-event signal-to-noise is low by "
        "construction -- a 3 mmHg deadband against a 6.25 mmHg observation sd "
        "-- so this is a statement about what is resolvable here, not about "
        "pharmacology.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "28_dynamics_report", paths.log_home / "28_dynamics_report.log"
    )
    tables = paths.output_home / "tables" / paths.model_key

    out = [
        f"# {args.subset.title()} shock: action-response dynamics\n",
        f"\n_Built {dt.date.today().isoformat()} from "
        f"`output/tables/{paths.model_key}`._\n",
        "\nFive phases, each stopping for review before the next. A phase "
        "listed below without results has not been run; it is listed anyway "
        "so that a missing phase reads as missing rather than as a phase that "
        "found nothing.\n",
        "\n## Scope and guardrails\n",
        "\n**Leakage.** The GEM embedding at grid point *t* is the final "
        "hidden state at the last token at or before onset+*t*: `GridWinnower` "
        "cuts each timeline at `cs_onset_dttm + offset_hours` and defines the "
        "past as `times <= cutoff_dttm`, and **each offset is a separate "
        "extract with its own winnowed prefix**, so nothing after *t* enters "
        "it. Note the bound is `<= t`, not `< t`: a token stamped exactly *t* "
        "is included. Every target in these phases is strictly forward, so "
        "this is not leakage -- but it would be for anything regressed on a "
        "quantity measured *at* t.\n",
        "\n**Embeddings exist at 13 timepoints only** (-24h to +48h, 6h "
        "steps). Phases 1-4 therefore run on raw-table encoding at hourly "
        "resolution; the embedding enters only where the grid supports it. "
        "Re-extracting hourly to seven days would be ~168 offsets across three "
        "sites against the ~5 GPU-hours the current 13 took.\n",
        "\n**Goals of care.** `clif_code_status` carries only DNAR/DNI, DNAR "
        "and Full -- no comfort-care order. Phase 3 therefore uses **two** "
        "absorbing states, RECOVERY and DEATH, and censors at a code-status "
        "change, reporting how many crossings fall within 12h after one. "
        "`clif_hospitalization.discharge_category` does carry `Hospice` "
        "(2.5-4.4% by site), which would support a third absorbing state; that "
        "was considered and declined, because it is a discharge disposition "
        "rather than a timestamped transition and so cannot date the decision "
        "behind it.\n",
        "\n**Not computable here.** No `patient_intake_output` at any site, so "
        "fluid balance is unavailable for phase 5 -- the same gap that removed "
        "urine output from the cohort's end-organ criterion. "
        "`clif_hospital_diagnosis` exists only at RUSH and so cannot be used "
        "federated.\n",
        "\n**All splits are patient-level and every threshold is in "
        "`config/dynamics.yaml`,** reproduced at the end of this report and "
        "saved beside every output.\n",
    ]

    for number, title, driver in PHASES:
        out.append(f"\n## Phase {number}: {title}\n\n")
        if number == 1:
            phase_one(paths, args.subset, tables, logger, out)
        elif number == 2:
            phase_two(paths, args.subset, tables, logger, out)
        else:
            out.append(
                f"_Not yet built. Will be produced by `code/{driver}`._\n"
            )

    config_file = tables / f"dynamics_config_{args.subset}.yaml"
    if config_file.is_file():
        out.append("\n## Configuration\n\n")
        out.append(
            "Every number in this report depends on these cuts. Saved beside "
            "each output so a result and the choices behind it cannot drift "
            "apart.\n\n"
        )
        out.append("```yaml\n")
        out.append(OmegaConf.to_yaml(OmegaConf.load(config_file)))
        out.append("```\n")

    destination = paths.output_home / f"dynamics_{args.subset}.md"
    destination.write_text("".join(out), encoding="utf-8")
    logger.info(
        f"wrote {destination} ({destination.stat().st_size / 1024:.0f} KB)"
    )
    logger.info("28 done")


if __name__ == "__main__":
    main()

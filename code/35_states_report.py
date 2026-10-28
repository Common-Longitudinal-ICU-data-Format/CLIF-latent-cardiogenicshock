#!/usr/bin/env python3

"""
35_states_report.py -- standalone report for the hourly response states

One object: the ten hourly MAP-and-pressor response states, how often each
occurs, how patients move between them, which courses recur, where they sit in
the embedding, and which record tokens separate a wean that holds from one that
fails.

    python code/35_states_report.py --subset cardiogenic

Reuses 21's helpers rather than restating them, so the two standalone reports
cannot drift apart in formatting. A leading digit makes a module unimportable
by name, hence the importlib load.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import pathlib
import sys

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.regions import HOURLY_REGION_ORDER, STATE_SYMBOL


def _helpers():
    spec = importlib.util.spec_from_file_location(
        "bpe_report",
        pathlib.Path(__file__).resolve().parent / "21_bpe_report.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger("35_states_report", paths.log_home / "35_states_report.log")
    h = _helpers()
    tables = paths.output_home / "tables" / paths.model_key
    figures = paths.output_home / "figures" / paths.model_key

    def read(stem, **kw):
        path = tables / f"{stem}_{args.subset}.csv"
        if not path.is_file():
            logger.warning(f"missing {path.name}")
            return pl.DataFrame()
        return pl.read_csv(path, infer_schema_length=None, **kw)

    out = [f"<h1>{args.subset.title()} shock: hourly response states</h1>"]
    out.append(
        f'<p class="meta">Ten states, one per patient-hour &middot; '
        f"{args.subset} cohort &middot; built {dt.date.today().isoformat()}</p>"
    )
    out.append(h.callout(
        "<b>A state is an action and the response that followed it.</b> The "
        "action is the change in norepinephrine over the hour just ended; the "
        "response is the change in MAP over the hour about to start. The two "
        "windows meet but never overlap.<br><br>"
        "<b>Every hour carries a state</b>, including hours in which nothing "
        "was changed and hours off pressors entirely. An earlier version "
        "labelled only hours in which the dose moved, which dropped 85% of "
        "hours and made a 'transition' span whatever gap separated two "
        "titrations &mdash; 46% one hour, 10% more than six.<br><br>"
        "<b>Nothing here is a dose effect.</b> Clinicians escalate when the "
        "pressure is falling and wean when it is not, so every state is "
        "confounded by the reason it was entered. 'Deteriorating' names an "
        "hour in which the dose went up and the pressure went down; it does "
        "not say the dose failed."
    ))

    # ---- 1. the states
    shares = read("hourly_region_shares")
    out.append("<h2>1. The ten states</h2>")
    if not shares.is_empty():
        shares = shares.with_columns(
            symbol=pl.col("region").replace_strict(
                STATE_SYMBOL, default="?", return_dtype=pl.String
            )
        ).select(["symbol"] + [c for c in shares.columns])
        out.append(h.table(shares, {
            "hours": "int", "share": "pct", "patients": "int",
            "mean_d_ne": "num4", "mean_d_map": "num2",
        }))
    out.append(h.callout(
        "<b>Off pressors is 78% of hours</b>, which is why it is dropped from "
        "the competition in the on-support figures below and from the "
        "sequence work entirely. Most of those hours sit at the end of a "
        "trajectory repeating that a recovered patient is still recovered.",
        kind="warn",
    ))

    # ---- 2. transitions
    out.append("<h2>2. Movement between states</h2>")
    persistence = read("hourly_persistence")
    transitions = read("hourly_transitions")
    out.append(h.callout(
        "<b>Transitions are taken at lag 2 hours, and the reason is an "
        "artefact that invalidated the first attempt.</b> A state at hour "
        "<i>t</i> is built from MAP(<i>t</i>) and MAP(<i>t</i>+1); a state at "
        "<i>t</i>+1 from MAP(<i>t</i>+1) and MAP(<i>t</i>+2). They share "
        "MAP(<i>t</i>+1), so one low reading makes hour <i>t</i> read as "
        "deterioration and hour <i>t</i>+1 as recovery with no physiology "
        "involved. Measured on the adjacent lag the correlation between "
        "consecutive MAP changes was <b>&minus;0.401</b>, against &minus;0.011 "
        "and &minus;0.009 at non-adjacent lags. At lag 2 the windows are "
        "disjoint on both axes.",
        kind="warn",
    ))
    if not persistence.is_empty():
        out.append(h.table(persistence, {
            "pairs": "int", "observed_stay_rate": "pct",
            "shuffle_null_mean": "pct", "shuffle_null_sd": "pct",
            "excess": "pct", "p_exceeds_null": "num4",
        }))
        out.append(
            "<p>The null shuffles each patient's own state sequence, "
            "preserving how much that patient favours each state and "
            "destroying only the order &mdash; so an excess cannot be "
            "explained by composition. Patients do persist, but the headline "
            "figure is inflated by off-pressor hours following one another: "
            "<b>restricted to hours on support, persistence is 26.6% against "
            "a 23.6% null, an excess of 3.0 points</b> (p&nbsp;&lt;&nbsp;0.0001). "
            "Real, and modest.</p>"
        )
    if not transitions.is_empty():
        out.append("<h3>Transition matrix</h3>")
        out.append(
            "<p>Rows sum to one, Dirichlet &alpha;=1. Read the diagonal "
            "against each row's marginal: a state whose most likely successor "
            "is simply the commonest state is not acting as a state.</p>"
        )
        out.append(h.table(transitions, {
            "n": "int",
            **{r: "pct" for r in HOURLY_REGION_ORDER},
        }))

    # ---- 3. sequences
    out.append("<h2>3. Recurring courses</h2>")
    courses = read("state_courses", schema_overrides={"course": pl.String})
    vocab = read("state_bpe_vocabulary", schema_overrides={"word": pl.String})
    out.append(h.callout(
        "<b>Trailing off-pressor hours are collapsed to one symbol.</b> A "
        "patient who recovers contributes one informative hour and then many "
        "repeating it; 66% of all hours are removed this way. Off-pressor "
        "hours sitting <i>between</i> periods of support are a relapse and are "
        "kept in full &mdash; 'came off and stayed off' compresses, 'came off "
        "and went back on' does not.<br><br>"
        "<b>Courses are near-unique.</b> 6,290 distinct collapsed courses "
        "among 9,378 patients, 98% occurring once, so there is no such thing "
        "as the commonest course. 2,455 patients (26%) collapse to a single "
        "'off pressors' and contributed no on-support course at all."
    ))
    if not courses.is_empty():
        out.append("<h3>Commonest collapsed courses</h3>")
        out.append(h.table(courses, {
            "patients": "int", "share": "pct", "distinct_states": "int",
        }, limit=18))
    if not vocab.is_empty():
        moves = vocab.filter(~pl.col("is_a_run"))
        if not moves.is_empty():
            out.append("<h3>BPE motifs: recurring transitions</h3>")
            out.append(
                "<p>Where whole courses are unique, sub-courses are not. "
                "These are the learned units that involve a <b>change</b> of "
                "state rather than a run of one. They are what a clustering on "
                "state composition could not see: composition is order-free, "
                "so it cannot distinguish <code>Sd</code> from "
                "<code>dS</code>.</p>"
            )
            out.append(h.table(moves, {
                "occurrences": "int", "patients": "int",
                "share_of_patients": "pct", "length": "int",
                "distinct_states": "int",
            }, limit=18))

    # ---- 4. localisation
    out.append("<h2>4. Where the states sit in the embedding</h2>")
    local = read("state_localisation_1024")
    conditional = read("state_conditional_probe")
    out.append(h.callout(
        "<b>PaCMAP says the states overlap completely. The 1024-d space says "
        "they do not.</b> A projection's distances are not metric, so overlap "
        "on the map is not a claim about the representation. Probing the "
        "corrected 1024-d embedding directly recovers every state at balanced "
        "AUC 0.80 to 0.97. The hexagon and panel figures below are included "
        "because they were asked for and because the contrast is instructive, "
        "but <b>the table is the answer and the figures are the artefact</b>.",
        kind="warn",
    ))
    if not local.is_empty():
        out.append(h.table(local, {
            "grid_points": "int", "balanced_n": "int", "auc_1024d": "num4",
        }))
    if not conditional.is_empty():
        out.append("<h3>Prediction or reconstruction?</h3>")
        out.append(
            "<p>A state is an action plus a response. The embedding at hour "
            "<i>t</i> was built from tokens at or before <i>t</i>, so it has "
            "<b>already seen the action</b> and has <b>not seen the "
            "response</b>. An unconditional AUC therefore mixes recognising "
            "what was done with predicting what followed. Restricting each "
            "comparison to hours sharing one action removes the first: every "
            "row had the same thing done to it, so whatever the probe recovers "
            "concerns the hour it could not see.</p>"
        )
        out.append(h.table(conditional, {
            "balanced_n": "int", "auc_within_action": "num4",
        }))
        out.append(h.callout(
            "<b>Most of the 0.85 was recognition, but not all of it.</b> "
            "Conditional AUCs run 0.54&ndash;0.71, above chance everywhere and "
            "replicating across both cohorts. The strongest is <b>wean held "
            "against failed wean at 0.68 (cardiogenic) and 0.71 "
            "(septic)</b> &mdash; the embedding carries real information about "
            "whether a wean is about to fail, an hour before it does. That is "
            "the only forward-looking signal in this report."
        ))
    for name, caption in (
        (f"hourly_state_hex_plurality_on_support_{args.subset}.png",
         "Plurality, on-support states only"),
        (f"hourly_state_hex_enrichment_on_support_{args.subset}.png",
         "Enrichment, on-support states only"),
        (f"hourly_state_hex_plurality_on_support_flow_{args.subset}.png",
         "Plurality with the displacement field"),
        (f"hourly_state_panels_on_support_flow_{args.subset}.png",
         "Each state on its own, with flow"),
    ):
        shown = h.figure_tag(figures / name, caption)
        if shown:
            out.append(shown)

    # ---- 5. the volcano
    out.append("<h2>5. Tokens separating a failed wean from one that held</h2>")
    volcano = read("wean_token_volcano")
    shown = h.figure_tag(
        figures / f"wean_token_volcano_{args.subset}.png",
        "Open red markers are constitutive tokens -- MAP and pressor dose -- "
        "whose enrichment restates the state's own definition.",
    )
    if shown:
        out.append(shown)
    out.append(h.callout(
        "<b>This plot is nearly empty, and that is the result.</b> Of 586 "
        "tokens tested in cardiogenic, <b>three</b> were significant and "
        "non-constitutive; of 624 in septic, <b>two</b>. Only one replicates "
        "across both cohorts: <code>LABEL//hypertension_init</code>, present "
        "in 5.7% of failed weans against 2.0% that held (cardiogenic) and 5.1% "
        "against 1.8% (septic) &mdash; log2 ratios of +1.53 and +1.50, which "
        "is a striking agreement for an unplanned comparison.<br><br>"
        "A chronically hypertensive patient has a higher autoregulatory set "
        "point, so a MAP of 65 is relatively lower for them. That is a "
        "mechanism, and it is a hypothesis: the token marks a problem-list "
        "label, nothing was randomised, and one replicated token out of 1,210 "
        "tests is the sort of finding that needs to be sought prospectively "
        "before it is believed.",
        kind="warn",
    ))
    if not volcano.is_empty():
        out.append(h.table(
            volcano.filter(~pl.col("constitutive")).sort("p").head(20),
            {"prevalence_failed_wean": "pct", "prevalence_wean_held": "pct",
             "log2_ratio": "num2", "p": "num4", "p_adjusted": "num3",
             "neg_log10_p": "num2", "n_failed_wean": "int",
             "n_wean_held": "int"},
        ))

    # ---- 6. tokens per state
    out.append("<h2>6. Tokens alongside each state</h2>")
    per_state = read("state_tokens")
    out.append(h.callout(
        "<b>Read this as a severity gradient, not as state-specific "
        "biology.</b> 2,698 of 7,455 tests clear Bonferroni, and what they "
        "mostly say is that the acute states sit beside paralytics "
        "(cisatracurium, rocuronium), sedation (midazolam) and dialysis, while "
        "off-pressor hours sit beside an antihypertensive (nicardipine) and "
        "the highest mobility scores. Token windows are six hours while a "
        "state is one, so a token 'alongside' a state may precede it by five "
        "hours, and states differ systematically in how much record surrounds "
        "them.",
        kind="warn",
    ))
    if not per_state.is_empty():
        top = (
            per_state.filter(
                (pl.col("p_adjusted") < 0.05) & ~pl.col("constitutive")
            )
            .sort(["state", "log2_ratio"], descending=[False, True])
            .group_by("state", maintain_order=True)
            .head(4)
        )
        out.append(h.table(top, {
            "prevalence_state": "pct", "prevalence_rest": "pct",
            "log2_ratio": "num2", "p": "num4", "p_adjusted": "num3",
            "neg_log10_p": "num2", "n_state": "int", "n_rest": "int",
        }))

    body = "\n".join(out)
    document = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{args.subset} shock: hourly response states</title>"
        f"<style>{h.STYLE}</style></head><body>{body}</body></html>"
    )
    destination = paths.output_home / f"report_states_{args.subset}.html"
    destination.write_text(document, encoding="utf-8")
    logger.info(
        f"wrote {destination} ({destination.stat().st_size / 1e6:.1f} MB)"
    )
    logger.info("35 done")


if __name__ == "__main__":
    main()

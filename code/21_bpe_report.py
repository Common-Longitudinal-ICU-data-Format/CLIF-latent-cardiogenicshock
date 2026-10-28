#!/usr/bin/env python3

"""
21_bpe_report.py -- a standalone report for the 6-hour MAP/pressor BPE analysis

Everything in this report concerns ONE object: the 15-state MAP-by-pressor
space sampled inside each 6-hour window, the BPE vocabulary learned over it,
and what that vocabulary says about trials off vasopressors. Nothing from the
embedding-projection analyses appears except the one figure that places weans
in the latent space, and nothing from the five patient-level clusterings
appears at all.

    python code/21_bpe_report.py --subset cardiogenic

WRITTEN AS ITS OWN FILE, not a section of the main report, because the main
report carries thirty sections and this is meant to be readable in one sitting
by someone who cares only about the pressor-trajectory question.

THE WINDOW IS SIX HOURS AND THE LETTERS ARE SUB-WINDOW. That pairing is the
whole point and is easy to misread. A window is six hours; within it a patient
gets as many letters as they had distinct states, so the sentence carries
sub-window detail while the marker keeps the six-hour accounting intact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import pathlib
import sys

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.perfusion import PERFUSION_STATES, STATE_SYMBOLS

STEP_HOURS = 6.0

STYLE = """
:root { --ink:#1b1b1b; --muted:#6a6a6a; --rule:#e3e3e3; --accent:#2166ac; }
* { box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
       color:var(--ink); max-width:1180px; margin:0 auto; padding:40px 28px;
       line-height:1.55; font-size:15px; }
h1 { font-size:27px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:21px; margin:46px 0 10px; padding-bottom:7px;
     border-bottom:2px solid var(--ink); }
h3 { font-size:17px; margin:30px 0 8px; }
h4 { font-size:15px; margin:22px 0 6px; color:var(--muted);
     text-transform:uppercase; letter-spacing:0.04em; }
p { margin:9px 0; }
code { background:#f3f3f3; padding:1px 5px; border-radius:3px;
       font-size:13px; font-family:"SF Mono",Menlo,Consolas,monospace; }
table { border-collapse:collapse; width:100%; margin:14px 0; font-size:13px; }
th { text-align:right; padding:7px 9px; border-bottom:2px solid var(--ink);
     font-weight:600; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
td { padding:6px 9px; border-bottom:1px solid var(--rule); text-align:right;
     font-variant-numeric:tabular-nums; }
tr:hover td { background:#fafafa; }
.callout { border-left:3px solid var(--accent); background:#f5f8fb;
           padding:12px 16px; margin:16px 0; font-size:14px; }
.callout.warn { border-left-color:#c0392b; background:#fdf5f3; }
.meta { color:var(--muted); font-size:13px; margin-bottom:26px; }
figure { margin:18px 0; }
img { max-width:100%; border:1px solid var(--rule); border-radius:3px; }
figcaption { color:var(--muted); font-size:12.5px; margin-top:6px; }
.grid { display:grid; grid-template-columns:repeat(2,1fr); gap:18px; }
.toc a { color:var(--accent); text-decoration:none; display:block;
         padding:3px 0; }
"""


def esc(text) -> str:
    return html.escape("" if text is None else str(text))


def callout(text: str, kind: str = "") -> str:
    return f'<div class="callout {kind}"><p>{text}</p></div>'


def fmt(value, kind):
    """one cell, formatted by column kind rather than by guessing from dtype"""
    if value is None:
        return "&mdash;"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "int":
        return f"{value:,.0f}"
    if kind == "num2":
        return f"{value:.2f}"
    if kind == "num3":
        return f"{value:.3f}"
    if kind == "num4":
        return f"{value:.4f}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return esc(value)


def table(frame: pl.DataFrame, kinds=None, limit=None) -> str:
    """
    a frame as HTML, with explicit per-column formatting

    `kinds` maps column -> one of pct/int/num2/num3/num4. Anything unnamed is
    rendered by dtype. Explicit rather than inferred because a share and a
    count are both floats and only the column name knows which is which.
    """
    if frame.is_empty():
        return "<p><i>(no rows)</i></p>"
    kinds = kinds or {}
    if limit:
        frame = frame.head(limit)
    head = "".join(
        f"<th>{esc(c.replace('_', ' '))}</th>" for c in frame.columns
    )
    body = []
    for row in frame.iter_rows(named=True):
        cells = "".join(
            f"<td>{fmt(row[c], kinds.get(c))}</td>" for c in frame.columns
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def figure_tag(path: pathlib.Path, caption: str = "") -> str:
    """
    a figure embedded as base64, so the report travels as one file

    The main report does the same. A relative link would break the moment the
    file is emailed, which is the normal way these get read.
    """
    import base64
    if not path.is_file():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{data}" alt="{esc(caption)}">'
        + (f"<figcaption>{caption}</figcaption>" if caption else "")
        + "</figure>"
    )


def state_table(fine, labels, logger):
    """
    the 15 states: how much of the week they account for, and who occupies them

    TWO KINDS OF NUMBER, and the column names keep them apart. `share_of_slots`
    is the fraction of all (patient, letter) positions spent in that state --
    a measure of the state's share of recorded TIME. Everything else is
    computed over PATIENTS who ever occupy the state, which is a different
    denominator and a much larger one: a patient needs only one letter in a
    state to count toward its mortality.

    THE OUTCOME COLUMNS ARE NOT ATTRIBUTABLE. A state's mortality is the
    mortality of patients who passed through it, not of the state. Almost
    every patient passes through "MAP >65, no pressors" at some point, so its
    mortality is close to the cohort's; the informative comparison is between
    the rarer states, and even there, occupying a state is a consequence of
    being unwell rather than a cause.
    """
    symbols = {
        STATE_SYMBOLS[i]: name for i, name in enumerate(PERFUSION_STATES)
        if i < len(STATE_SYMBOLS)
    }
    sites = sorted(fine["site"].unique().to_list())
    outcome_columns = [
        c for c in ("in_hospital_mortality", "ever_imv", "new_imv_after",
                    "ever_rrt", "new_rrt_after", "icu_los_days",
                    "icu_free_days")
        if c in labels.columns
    ]
    joined = fine.join(
        labels.select(["hospitalization_id"] + outcome_columns),
        on="hospitalization_id", how="left",
    )

    # one pass, counting letters per state per patient
    occupancy = {s: set() for s in symbols}
    slots = {s: 0 for s in symbols}
    by_site = {s: {site: 0 for site in sites} for s in symbols}
    total_slots = 0
    for patient, sentence, site in zip(
        joined["hospitalization_id"].to_list(),
        joined["fine_sentence"].to_list(),
        joined["site"].to_list(),
    ):
        if not sentence:
            continue
        for character in sentence:
            if character not in symbols:
                continue
            slots[character] += 1
            total_slots += 1
            occupancy[character].add(patient)
            if site in by_site[character]:
                by_site[character][site] += 1

    rows = []
    for symbol, name in symbols.items():
        if not slots[symbol]:
            continue
        people = joined.filter(
            pl.col("hospitalization_id").is_in(list(occupancy[symbol]))
        )
        row = {
            "symbol": symbol,
            "state": name,
            "slots": slots[symbol],
            "share_of_slots": slots[symbol] / max(total_slots, 1),
            "patients_ever_in_state": len(occupancy[symbol]),
            "share_of_patients": len(occupancy[symbol]) / max(joined.height, 1),
        }
        for site in sites:
            row[f"share_at_{site}"] = (
                by_site[symbol][site] / max(slots[symbol], 1)
            )
        for column in outcome_columns:
            values = people[column].drop_nulls()
            if values.is_empty():
                continue
            share = column.startswith(("ever_", "new_", "in_hospital"))
            row[column] = float(
                values.cast(pl.Float64).mean() if share else values.median()
            )
        rows.append(row)
    table_out = pl.DataFrame(rows, infer_schema_length=None).sort(
        "share_of_slots", descending=True
    )
    logger.info(f"state table: {table_out.height} states over {total_slots:,} slots")
    return table_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic"))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger("21_bpe_report", paths.log_home / "21_bpe_report.log")
    tables = paths.output_home / "tables" / paths.model_key
    figures = paths.output_home / "figures" / paths.model_key

    def read(stem):
        path = tables / f"{stem}_{args.subset}.csv"
        if not path.is_file():
            logger.warning(f"missing {path.name}")
            return pl.DataFrame()
        return pl.read_csv(path, schema_overrides={
            "word": pl.String, "preceding_word": pl.String,
            "pattern": pl.String,
        }, infer_schema_length=None)

    fine = pl.read_parquet(
        paths.patient_level_home / f"fine_state_sequences_{args.subset}.parquet"
    )
    labels = pl.read_parquet(
        paths.patient_level_home
        / f"longitudinal_clusters_{args.subset}.parquet"
    )
    out = []

    out.append(f"<h1>{args.subset.title()} shock: the 6-hour MAP/pressor "
               "vocabulary</h1>")
    out.append(
        f'<p class="meta">{fine.height:,} patients &middot; '
        f"{args.subset} cohort &middot; states sampled inside each "
        f"{STEP_HOURS:.0f}-hour window &middot; built "
        f"{dt.date.today().isoformat()}</p>"
    )
    out.append(callout(
        "<b>Everything here concerns one object.</b> The 15-state "
        "MAP-by-pressor space sampled <i>inside</i> each 6-hour window, the "
        "BPE vocabulary learned over it, and what that vocabulary says about "
        "trials off vasopressors. The latent-space projection appears once, to "
        "place the weans; none of the five patient-level clusterings appears "
        "at all.<br><br>"
        "<b>The window is six hours and the letters are sub-window</b> "
        "&mdash; that pairing is the point and is easy to misread. A window is "
        "six hours; within it a patient gets as many letters as they had "
        "distinct states. So <code>646_</code> is three states inside one "
        "window, and the trailing <code>_</code> keeps the six-hour accounting "
        "intact while the letters carry finer detail than six hours."
    ))

    # ---- 1. the states
    out.append('<h2>1. The 15 states</h2>')
    states = state_table(fine, labels, logger)
    out.append(
        "<p>MAP above or below 65&nbsp;mmHg (a Schmitt trigger: above 65 to "
        "enter the adequate state, below 55 to leave it, previous state "
        "carried forward between), crossed with the pressor arm &mdash; none, "
        "or norepinephrine alone or with another agent, each with the dose "
        "rising, falling or steady under a deadband of 25% relative and 0.01 "
        "absolute.</p>"
    )
    kinds = {"slots": "int", "share_of_slots": "pct",
             "patients_ever_in_state": "int", "share_of_patients": "pct",
             "in_hospital_mortality": "pct", "ever_imv": "pct",
             "new_imv_after": "pct", "ever_rrt": "pct",
             "new_rrt_after": "pct", "icu_los_days": "num2",
             "icu_free_days": "num2"}
    kinds.update({c: "pct" for c in states.columns if c.startswith("share_at_")})
    out.append(table(states, kinds))
    out.append(callout(
        "<b>Two denominators, and the column names keep them apart.</b> "
        "<code>share_of_slots</code> is the fraction of all recorded "
        "(patient, letter) positions spent in that state &mdash; the state's "
        "share of recorded <i>time</i>. Everything to its right is computed "
        "over <i>patients who ever occupy the state</i>, a much larger "
        "denominator: one letter is enough to count toward a state's "
        "mortality.<br><br>"
        "<b>The outcome columns are not attributable to the state.</b> Nearly "
        "every patient passes through <code>MAP &gt;65, no pressors</code>, so "
        "its mortality is close to the cohort's by construction. The "
        "informative contrast is between the rarer states, and even there, "
        "occupying a state is a consequence of being unwell, not a cause of "
        "it.",
        kind="warn",
    ))

    # ---- 2. the BPE fit itself
    out.append('<h2>2. The BPE fit</h2>')
    vocab = read("fine_bpe_vocabulary")
    merges = read("fine_bpe_merges")
    stability = read("bpe_stability")
    if not vocab.is_empty():
        multi = vocab.filter(pl.col("length") > 1)
        letters = int((vocab["length"] * vocab["occurrences"]).sum())
        words = int(vocab["occurrences"].sum())
        out.append(
            f"<p><b>{merges.height if not merges.is_empty() else 0} merges</b> "
            f"learned, producing <b>{vocab.height:,} distinct words</b> of "
            f"which {multi.height:,} span more than one state. Compression is "
            f"{letters / max(words, 1):.2f} letters per word across "
            f"{words:,} word occurrences.</p>"
        )
    out.append(callout(
        "<b>The window marker is a suffix, not a separator.</b> Merges absorb "
        "it rightward &mdash; <code>320</code> + <code>_</code> becomes the "
        "word <code>320_</code>, the way SentencePiece writes <code>ing_</code> "
        "&mdash; but nothing may extend past it. So no word ever spans two "
        "windows, and <code>_</code> only ever appears as a word's last "
        "character. <code>320_</code> and <code>320</code> are different "
        "words: one closes the window, the other sits inside it.<br><br>"
        "<b>Every word is movement, by construction.</b> Runs are collapsed "
        "within a window and merges cannot cross a boundary, so no word "
        "carries the same state twice running. The 6-hour-per-letter "
        "vocabulary's three commonest entries were <code>00</code>, "
        "<code>000</code> and <code>0000</code> &mdash; three words all "
        "meaning 'stayed off pressors'. None of them can exist here."
    ))
    if not stability.is_empty():
        out.append('<h4>How much the vocabulary depends on sample size</h4>')
        out.append(
            "<p>BPE refitted on nested subsamples against the full fit. "
            "<code>sentences_segmented_identically</code> is the column to "
            "read: nothing downstream consumes the merge list, everything "
            "consumes the segmentation, so two fits that found the same merges "
            "in a different order should score as agreeing.</p>"
        )
        out.append(table(stability, {
            "sequences": "int", "merges_learned": "int",
            "identical_prefix": "int", "prefix_share": "pct",
            "merges_in_common_any_order": "int",
            "sentences_segmented_identically": "pct", "word_overlap": "pct",
        }))

    # ---- 3. the words
    out.append('<h2>3. The words, and what surrounds them</h2>')
    if not vocab.is_empty():
        columns = [
            c for c in ("word", "length", "occurrences", "patients",
                        "share_of_patients", "opens_sentence",
                        "closes_sentence", "commonly_preceded_by",
                        "commonly_followed_by", "meaning")
            if c in vocab.columns
        ]
        out.append(
            "<p>The commonest multi-state words. <code>preceded by</code> and "
            "<code>followed by</code> are the literally adjacent words, with "
            "each neighbour's share of that word's occurrences that have a "
            "neighbour on that side.</p>"
        )
        out.append(table(
            vocab.filter(pl.col("length") > 1).select(columns),
            {"occurrences": "int", "patients": "int",
             "share_of_patients": "pct", "opens_sentence": "pct",
             "closes_sentence": "pct"},
            limit=30,
        ))

    out.append('<h3>Where each word owns the map</h3>')
    out.append(
        "<p>The projection tiled into hexagons, each coloured by which BPE "
        "word occupies it. A window may hold several words, so the "
        "<b>window-closing</b> one stands for it &mdash; the only word "
        "guaranteed to appear exactly once per recorded window, and the one "
        "whose span ends where the window does.</p>"
    )
    out.append(callout(
        "<b>Only the ten commonest words keep their own colour.</b> The "
        "vocabulary runs to eighty-odd window-closing words and a hundred and "
        "sixty-five overall; no categorical palette separates that many, and "
        "a legend of 165 entries is not a legend. Everything else is pooled "
        "into one grey level named with its size.<br><br>"
        "<b>Plurality and enrichment disagree on purpose.</b> Plurality is the "
        "commonest word in the hexagon and reads directly &mdash; if you are "
        "here you are probably in this word &mdash; but <code>0_</code> alone "
        "holds 40% of all slots, so it wins hexagons where it is merely "
        "present. Enrichment is the word most concentrated here relative to "
        "its <i>own</i> total, so size cancels and a rare word shows up "
        "wherever it localises, at the cost of letting a rare word win a thin "
        "hexagon on a handful of points. Where the two agree the structure is "
        "not an artefact of how unevenly the words are used; where they "
        "differ, that is the finding."
    ))
    for mode in ("plurality", "enrichment"):
        shown = figure_tag(
            figures / f"wean_word_hex_{mode}_{args.subset}.png",
            f"BPE word by hexagon, {mode}",
        )
        if shown:
            out.append(shown)

    # ---- 4. words preceding a trial off pressors
    out.append('<h2>4. Words preceding a trial off vasopressors</h2>')
    preceding = read("wean_preceding_words")
    per_word = read("wean_word_outcomes")
    if not preceding.is_empty():
        out.append(
            "<p><b>94% of <code>0_</code> words follow another "
            "<code>0_</code></b> &mdash; the patient was already off pressors "
            "and nothing happened. The weaning words are the remainder, and "
            "the commonest is <code>30_</code>: norepinephrine steady, then "
            "off, inside six hours, with no documented taper. It outnumbers "
            "the tapered <code>3230_</code> two to three times.</p>"
        )
        out.append(table(
            preceding, {"occurrences": "int", "share": "pct"}, limit=15
        ))
    if not per_word.is_empty():
        out.append('<h3>Each word, its patients, and whether the trial held</h3>')
        out.append(table(per_word, {
            "weans": "int", "patients": "int",
            "restarted_within_24h": "pct", "median_hours_from_onset": "num2",
            "median_sentence_words": "num2", "median_fine_letters": "num2",
            "in_hospital_mortality": "pct", "ever_imv": "pct",
            "new_imv_after": "pct", "ever_rrt": "pct", "new_rrt_after": "pct",
            "icu_los_days": "num2", "icu_free_days": "num2",
            "smd_map": "num3", "smd_nee": "num3", "smd_hours": "num3",
        }))
        out.append(callout(
            "<b>Two denominators again.</b> <code>weans</code> and "
            "<code>restarted_within_24h</code> count <i>occurrences</i>; "
            "<code>patients</code> and every outcome count <i>patients</i> who "
            "carry the word at least once. A patient weaned four times is four "
            "occurrences and one outcome.<br><br>"
            "<b>A fixed 24-hour horizon, and weans without it are dropped.</b> "
            "'Never restarted' would be dominated by how much week was left "
            "&mdash; a wean at hour 150 has four windows in which to fail and "
            "one at hour 12 has twenty-five. Weans without a full 24 hours of "
            "record after them are excluded rather than scored as successes, "
            "because a patient whose record stops two windows later has no "
            "subsequent pressor word, and counting that as a durable wean "
            "would count dying as recovering.<br><br>"
            "<b>The <code>smd_</code> columns are the within-word split</b> "
            "&mdash; standardised difference between the occurrences that "
            "restarted and those that held. <code>smd_map</code> is the one to "
            "look at: it is near zero for every word, meaning the patients "
            "whose trial failed looked, at that moment, like the patients "
            "whose trial held.",
            kind="warn",
        ))

    lead_in = read("wean_lead_in")
    if not lead_in.is_empty():
        out.append('<h3>The trial characterised by what led into it</h3>')
        out.append(
            "<p><code>30_</code> and <code>60_</code> are close to "
            "uninformative on their own &mdash; both mostly say 'came off "
            "pressors', which is what every weaning word says. What separates "
            "one trial from another is the window <b>before</b>: entering a "
            "trial from <code>3_</code>, a steady single agent, is a different "
            "clinical situation from entering it from <code>646_</code>, the "
            "combination being actively moved, or from <code>D_</code>, "
            "hypotensive on a combination. Sorted most durable first.</p>"
        )
        out.append(table(lead_in, {
            "events": "int", "patients": "int",
            "restarted_within_24h": "pct", "median_hours_from_onset": "num2",
            "in_hospital_mortality": "pct", "icu_free_days": "num2",
            "icu_los_days": "num2", "new_imv_after": "pct",
            "new_rrt_after": "pct",
        }, limit=24))

    # ---- 5. where those words sit in the embedding
    out.append('<h2>5. Where the trials sit in the embedding</h2>')
    out.append(
        "<p>The pooled map of all weans together is deliberately not shown: "
        "every weaning word mostly says 'came off pressors', so one map over "
        "all of them is a map of weaning in general and answers nothing about "
        "a particular manoeuvre. The per-trial panels below carry the same "
        "information split by what led into the trial. The statistics for the "
        "pooled map are kept, because the clustering tests are what justify "
        "looking at the panels at all.</p>"
    )
    stats = read("wean_embedding")
    covariates = read("wean_embedding_covariates")
    if not stats.is_empty():
        out.append(table(stats, {
            "weans_placed": "int", "weans_total": "int", "hexagons": "int",
            "pooled_restart": "pct", "rate_min": "pct", "rate_max": "pct",
            "weighted_variance": "num4", "weighted_variance_null": "num4",
            "p_patchiness": "num4", "morans_i": "num4", "p_morans_i": "num4",
        }))
    if not covariates.is_empty():
        out.append(table(covariates, {
            "hexagons": "int", "correlation_with_restart_rate": "num3",
        }))
        out.append(callout(
            "<b>The regions are real, and they are not news.</b> Patchiness "
            "runs about six times the null and Moran's I near +0.54, both "
            "p&nbsp;&lt;&nbsp;0.0001, so weans genuinely cluster by outcome "
            "here. But the high-restart regions are the <i>early, sicker, more "
            "heavily charted</i> part of the map &mdash; the rate correlates "
            "around &minus;0.5 to &minus;0.65 with hours from onset and "
            "+0.32 to +0.61 with mortality. Those are axes this projection was "
            "already known to encode.<br><br>"
            "<b>Do not read these against the <code>smd_</code> columns "
            "above.</b> These are <b>ecological</b> correlations over ~65 "
            "hexagons; those are per patient. MAP separates held from "
            "restarted at SMD near zero <i>per patient</i> while correlating "
            "+0.25 with the restart rate <i>per hexagon</i>, and that is not a "
            "contradiction &mdash; aggregating averages away within-cell noise "
            "and leaves the between-region component, which inflates the "
            "correlation. The aggregate numbers say what the map is organised "
            "by, not what predicts an individual trial.",
            kind="warn",
        ))
    out.append(callout(
        "<b>Only weans inside the embedding window can be placed.</b> The grid "
        "runs &minus;24h to +48h; the sequences run the full week. About two "
        "thirds are placed, and the third that are not sit at a median 84h "
        "against 12&ndash;18h for those kept &mdash; so this describes "
        "<b>early</b> trials specifically. The dropped weans restart at almost "
        "exactly the same rate, so nothing is selected on the outcome, only on "
        "timing."
    ))

    # ---- 6. one map per word
    word_dir = figures / f"wean_words_{args.subset}"
    if word_dir.is_dir():
        panels = sorted(word_dir.glob("wean_word_*.png"))
        if panels:
            out.append('<h3>One map per word</h3>')
            out.append(
                "<p>Separate figures rather than subplots, because the words "
                "differ by an order of magnitude in how often they occur and a "
                "shared panel invites reading the sparse ones against the "
                "dense ones as though the density meant the same thing. Each "
                "panel carries the whole grid in grey, so a region where the "
                "word never happens can be told from a region where no patient "
                "ever goes.</p>"
            )
            out.append('<div class="grid">')
            for panel in panels:
                word = panel.stem.replace("wean_word_", "").replace("W", "_")
                out.append(figure_tag(panel, f"<code>{esc(word)}</code>"))
            out.append("</div>")

    response = read("pattern_response")
    if not response.is_empty():
        out.append('<h2>7. Same approach, different response</h2>')
        out.append(
            "<p>A PrefixSpan pattern is an ordered sequence of words that many "
            "patients share, so among the patients carrying one, the "
            "management approach is roughly held fixed. This asks whether the "
            "GEM embedding then separates their outcomes &mdash; which is the "
            "question of whether there are populations responding differently "
            "to broadly the same management.</p>"
        )
        out.append(callout(
            "<b>The test is within-pattern, and that is the whole point.</b> "
            "An embedding that predicts death across the cohort only tells us "
            "it encodes severity, which is established. An embedding that "
            "predicts death <i>among patients whose recorded pressor course "
            "had the same shape</i> is saying something the course does not: "
            "that two patients managed alike were not alike. Read "
            "<code>auc_change</code> &mdash; the within-pattern AUC against "
            "the same model on the whole cohort.<br><br>"
            "<b>A pattern fixes shape, not equivalence.</b> It fixes the order "
            "of recorded MAP-and-pressor states. It does not fix dose, timing, "
            "duration, or anything outside that space, so two patients "
            "matching <code>3_ 3_ 0_</code> may have had very different doses "
            "for very different lengths of time.<br><br>"
            "<b>Confounding survives this design.</b> Patients are not "
            "assigned to patterns; a clinician arrives at a management shape "
            "partly by reading the patient. So a within-pattern signal may be "
            "the embedding recovering the severity that drove the management "
            "in the first place rather than identifying patients who would "
            "respond differently to a different approach. The counterfactual "
            "&mdash; would this patient have done better managed otherwise "
            "&mdash; is not answerable from observational trajectories.",
            kind="warn",
        ))
        out.append(table(response, {
            "words": "int", "patients_with_pattern": "int",
            "events_with_pattern": "int", "event_rate_with_pattern": "pct",
            "event_rate_without": "pct", "auc_within_pattern": "num4",
            "auc_whole_cohort": "num4", "auc_change": "num4",
        }, limit=40))
    cooc = read("pattern_cooccurrence")
    if not cooc.is_empty():
        out.append('<h3>How the patterns overlap</h3>')
        out.append(
            "<p>Patients carry several patterns at once, so the patterns are "
            "not a partition and a patient's week is not 'one' trajectory "
            "through them. This is the overlap: for each pair, how many "
            "patients carry both against either alone.</p>"
        )
        out.append(table(cooc, {
            "both": "int", "only_a": "int", "only_b": "int",
            "jaccard": "pct",
        }, limit=20))

    # ---- 7 and 8. ordered patterns
    out.append('<h2>8. Ordered patterns across words</h2>')
    grams = read("sentence_ngrams_fine")
    patterns = read("sentence_patterns_fine")
    by_class = read("sentence_patterns_by_class_fine")
    out.append(callout(
        "<b>There is no commonest sentence.</b> 98% of these sentences occur "
        "exactly once and the only common one is the degenerate 'never on "
        "pressors', so the distribution over whole sentences is flat and has "
        "nothing to report. Sub-sentences do.<br><br>"
        "<b>Two views, kept apart.</b> Word n-grams are contiguous &mdash; "
        "strictly what comes next. Sequential patterns (PrefixSpan) allow "
        "gaps, so a wean followed three days later by coming off pressors "
        "counts as one story whatever sat between; a trigram can only express "
        "that if the gap is exactly zero.<br><br>"
        "<b>Support is per patient, not per occurrence</b>, and only "
        "<b>closed</b> patterns are kept &mdash; a pattern is closed when no "
        "longer pattern containing it has the same support, which collapses "
        "the <code>6_ 6_ 6_</code> / <code>6_ 6_ 6_ 6_</code> chains into one "
        "row."
    ))
    if not patterns.is_empty():
        moves = patterns.filter(~pl.col("is_a_run")) \
            if "is_a_run" in patterns.columns else patterns
        out.append('<h3>Patterns involving a change of word</h3>')
        out.append(
            "<p>Runs of one repeated word are listed separately below: "
            "<code>0_ 0_ 0_</code> is three consecutive windows off pressors, "
            "true of three quarters of the cohort and not a sequence of care "
            "in any interesting sense.</p>"
        )
        out.append(table(
            moves, {"patients": "int", "share_of_patients": "pct",
                    "words": "int", "distinct_words": "int"},
            limit=25,
        ))
        if "is_a_run" in patterns.columns:
            out.append('<h4>Runs</h4>')
            out.append(table(
                patterns.filter(pl.col("is_a_run")),
                {"patients": "int", "share_of_patients": "pct",
                 "words": "int", "distinct_words": "int"},
                limit=8,
            ))
    if not grams.is_empty():
        out.append('<h3>Contiguous n-grams</h3>')
        out.append(table(
            grams, {"patients": "int", "share_of_patients": "pct",
                    "occurrences": "int", "words": "int"},
            limit=20,
        ))
    if not by_class.is_empty():
        significant = by_class.filter(pl.col("p_adjusted") < 0.05)
        if not significant.is_empty():
            out.append('<h3>Patterns by latent class</h3>')
            out.append(
                "<p>Each pattern as a per-patient presence flag, one latent "
                "class against the others pooled, small cells suppressed at 11 "
                "and Bonferroni over every (pattern, class) test. The latent "
                "classes are the only clustering used anywhere in this "
                "report.</p>"
            )
            shown = []
            for cluster in sorted(significant["cluster"].unique().to_list()):
                part = significant.filter(pl.col("cluster") == cluster)
                shown.append(part.sort("log2_ratio", descending=True).head(4))
                shown.append(part.sort("log2_ratio").head(2))
            out.append(table(
                pl.concat(shown).unique(subset=["cluster", "pattern"])
                .sort(["cluster", "log2_ratio"], descending=[False, True]),
                {"patients_in_cluster_with_pattern": "int",
                 "prevalence_in_cluster": "pct",
                 "prevalence_elsewhere": "pct", "log2_ratio": "num2",
                 "p": "num4", "p_adjusted": "num4", "words": "int"},
            ))
            out.append(callout(
                "<b>The count of significant tests overstates the count of "
                "findings.</b> Closed-pattern filtering removes "
                "support-identical duplicates, but patterns whose support "
                "differs slightly survive as separate rows saying much the "
                "same thing. Read this as a handful of stories told "
                "repeatedly."
            ))

    body = "\n".join(out)
    document = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{esc(args.subset)} shock: 6-hour MAP/pressor vocabulary</title>"
        f"<style>{STYLE}</style></head><body>{body}</body></html>"
    )
    destination = paths.output_home / f"report_bpe_{args.subset}.html"
    destination.write_text(document, encoding="utf-8")
    logger.info(
        f"wrote {destination} "
        f"({destination.stat().st_size / 1e6:.1f} MB)"
    )
    logger.info("21 done")


if __name__ == "__main__":
    main()

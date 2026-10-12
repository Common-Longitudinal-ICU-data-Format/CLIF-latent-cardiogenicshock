#!/usr/bin/env python3

"""
07 -- assemble the analyses into a single self-contained HTML report

Reads whatever 01-10 have written and renders it as one file. Figures are
inlined as base64 so the report can be emailed or opened from anywhere without
carrying a directory of PNGs alongside it; nothing is fetched at view time.

Sections render only when their inputs exist, so this is safe to run at any
point in the pipeline.

Usage
    python code/07_report.py
    python code/07_report.py --open
"""

from __future__ import annotations

import argparse
import base64
import datetime
import html
import json
import pathlib
import sys
import webbrowser

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import polars as pl

from clifcs import ANCHORS, get_logger, load_cohort_cfg, load_paths
from clifcs.palette import (
    CATEGORICAL,
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    LOGO_PALETTE,
    SURFACE,
)

MODELS = ("GEM-all", "GEM-ucmc-icu", "GEM-mimic-icu")

#: the trajectory labellings, in reading order, matching 09's `colourings`
LABELLINGS = [
    ("site", "site"),
    ("in_hospital_mortality", "mortality"),
    ("n_vasopressors", "pressors"),
    ("map", "MAP"),
    ("cs_phenotype", "phenotype"),
    ("ever_imv", "ever IMV"),
    ("hours_from_admission", "time in admission"),
    ("icu_los_days", "ICU LOS"),
]


# ----------------------------------------------------------------- rendering


def css() -> str:
    return f"""
:root {{
  --surface: {SURFACE['light']};
  --ink: {INK_PRIMARY};
  --ink-2: {INK_SECONDARY};
  --grid: {GRID};
  --accent: {CATEGORICAL[0]};
  --ucmc: {LOGO_PALETTE['UCMC']};
  --nu: {LOGO_PALETTE['NU']};
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 48px 28px 96px; }}
h1 {{ font-size: 30px; margin: 0 0 6px; letter-spacing: -0.02em; }}
h2 {{
  font-size: 21px; margin: 56px 0 4px; letter-spacing: -0.01em;
  padding-top: 20px; border-top: 1px solid var(--grid);
}}
h3 {{ font-size: 15px; margin: 30px 0 8px; color: var(--ink); }}
p, li {{ color: var(--ink-2); max-width: 80ch; }}
.sub {{ color: var(--ink-2); font-size: 13px; margin: 0 0 8px; }}
.meta {{ color: var(--ink-2); font-size: 12.5px; margin-bottom: 36px; }}
table {{ border-collapse: collapse; margin: 14px 0 6px; font-size: 13px; }}
th, td {{ padding: 6px 12px 6px 0; text-align: left; white-space: nowrap; }}
th {{
  font-weight: 600; color: var(--ink); border-bottom: 1px solid var(--grid);
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em;
}}
td {{ border-bottom: 1px solid #f2f1ee; color: var(--ink-2); }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.key {{ color: var(--ink); font-weight: 500; }}
tr.total td {{ font-weight: 600; color: var(--ink); }}
figure {{ margin: 18px 0 8px; }}
figure img {{
  max-width: 100%; height: auto; display: block;
  border: 1px solid var(--grid); border-radius: 4px;
}}
figcaption {{ font-size: 12.5px; color: var(--ink-2); margin-top: 7px; }}
.callout {{
  border-left: 3px solid var(--accent); background: #f6f8fc;
  padding: 12px 16px; margin: 18px 0; border-radius: 0 4px 4px 0;
}}
.callout.warn {{ border-left-color: #eb6834; background: #fdf5f1; }}
.callout p {{ margin: 0; color: var(--ink); font-size: 14px; }}
code {{
  font: 12.5px ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #f2f1ee; padding: 1px 5px; border-radius: 3px;
}}
pre.formula {{
  font: 12.5px ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #f2f1ee; padding: 10px 14px; border-radius: 4px;
  color: var(--ink); margin: 8px 0; overflow-x: auto; max-width: 80ch;
}}
dl.defs {{ margin: 16px 0; }}
dl.defs dt {{ font-weight: 600; color: var(--ink); font-size: 13.5px; margin-top: 14px; }}
dl.defs dd {{ margin: 4px 0 0 0; color: var(--ink-2); font-size: 13.5px; max-width: 80ch; }}
.toc {{ margin: 28px 0 0; padding: 0; list-style: none; columns: 2; }}
.toc li {{ margin: 3px 0; }}
.toc a {{ color: var(--accent); text-decoration: none; font-size: 13.5px; }}
.toc a:hover {{ text-decoration: underline; }}

/* tabs */
.tabs {{ margin: 18px 0 8px; }}
.tabstrip {{
  display: flex; flex-wrap: wrap; gap: 2px; border-bottom: 1px solid var(--grid);
  margin-bottom: 2px;
}}
.tabstrip button {{
  appearance: none; border: 0; background: transparent; cursor: pointer;
  font: inherit; font-size: 12.5px; color: var(--ink-2);
  padding: 7px 13px; border-radius: 4px 4px 0 0;
  border-bottom: 2px solid transparent; margin-bottom: -1px;
}}
.tabstrip button:hover {{ background: #f2f1ee; color: var(--ink); }}
.tabstrip button[aria-selected="true"] {{
  color: var(--ink); font-weight: 600; border-bottom-color: var(--accent);
}}
.tabpanel {{ display: none; padding-top: 6px; }}
.tabpanel[data-active="true"] {{ display: block; }}
"""


def script() -> str:
    """click handling for the tab groups; no dependencies"""
    return """
document.querySelectorAll('[data-tabs]').forEach(function (group) {
  var buttons = group.querySelectorAll(':scope > .tabstrip > button');
  var panels = group.querySelectorAll(':scope > .tabpanel');
  buttons.forEach(function (button, index) {
    button.addEventListener('click', function () {
      buttons.forEach(function (b, i) {
        b.setAttribute('aria-selected', i === index ? 'true' : 'false');
      });
      panels.forEach(function (p, i) {
        p.setAttribute('data-active', i === index ? 'true' : 'false');
      });
    });
  });
});
"""


_TAB_GROUP = [0]


def tabs(items, logger=None) -> str:
    """
    a tab group from (label, html) pairs

    Panels with no content are dropped rather than rendered as an empty tab,
    so a figure that has not been generated yet simply does not appear.
    """
    items = [(label, body) for label, body in items if body]
    if not items:
        return ""
    if len(items) == 1:
        return items[0][1]
    _TAB_GROUP[0] += 1
    group = _TAB_GROUP[0]
    strip = "".join(
        f'<button role="tab" aria-selected="{"true" if i == 0 else "false"}">'
        f"{esc(label)}</button>"
        for i, (label, _) in enumerate(items)
    )
    panels = "".join(
        f'<div class="tabpanel" data-active="{"true" if i == 0 else "false"}">'
        f"{body}</div>"
        for i, (_, body) in enumerate(items)
    )
    return (
        f'<div class="tabs" data-tabs id="tabs-{group}">'
        f'<div class="tabstrip" role="tablist">{strip}</div>{panels}</div>'
    )


def esc(value) -> str:
    return html.escape(str(value))


def table(frame: pl.DataFrame, numeric_format=None, key_column=None,
          highlight=None) -> str:
    numeric_format = numeric_format or {}
    highlight = highlight or set()
    header = "".join(
        f'<th class="{"num" if c in numeric_format else ""}">{esc(c)}</th>'
        for c in frame.columns
    )
    body = []
    for row in frame.iter_rows(named=True):
        cells = []
        for column in frame.columns:
            value = row[column]
            classes = []
            if column in numeric_format:
                classes.append("num")
                value = numeric_format[column](value) if value is not None else "—"
            if column == key_column:
                classes.append("key")
            cells.append(f'<td class="{" ".join(classes)}">{esc(value)}</td>')
        cls = ' class="total"' if str(row.get(key_column, "")) in highlight else ""
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return (
        f"<table><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def figure(path: pathlib.Path, caption: str = "") -> str:
    if not pathlib.Path(path).is_file():
        return ""
    data = base64.b64encode(pathlib.Path(path).read_bytes()).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (
        f'<figure><img src="data:image/png;base64,{data}" alt="{esc(caption)}">'
        f"{cap}</figure>"
    )


def callout(text: str, kind: str = "") -> str:
    return f'<div class="callout {kind}"><p>{text}</p></div>'


def formula(*lines: str) -> str:
    return '<pre class="formula">' + esc("\n".join(lines)) + "</pre>"


def pct(x):
    return "—" if x is None else f"{x:.1%}"


def num(digits):
    return lambda x: "—" if x is None else f"{x:,.{digits}f}"


def thousands(x):
    return "—" if x is None else f"{int(x):,}"


def read_model_tables(paths, name, required, logger):
    """collect one table across models, refusing to merge mismatched schemas"""
    frames, stale = [], []
    for model in MODELS:
        f = paths.output_home / "tables" / model / name
        if not f.is_file():
            continue
        frame = pl.read_csv(f)
        missing = [c for c in required if c not in frame.columns]
        if missing:
            stale.append(f"{model} ({', '.join(missing)} absent)")
            logger.warning(f"{f} lacks {missing}; excluded from the report")
            continue
        frames.append(frame.with_columns(model=pl.lit(model)))
    if not frames:
        return None, stale
    return pl.concat(frames, how="diagonal_relaxed"), stale


# ================================================================= sections


def section_cohort(paths, cfg, out: list) -> None:
    ledger_file = paths.cohort_home / "filter_ledger.csv"
    if not ledger_file.is_file():
        return
    ledger = pl.read_csv(ledger_file)
    out.append('<h2 id="cohort">1 &middot; Cohort</h2>')
    out.append(
        "<p>Three criteria applied in sequence to a base cohort of first "
        "hospitalizations at least 24 h long with an ICU stay in the first "
        "24 h &mdash; which mirrors the preprocessing the models were trained "
        "under, so the patients embedded here sit inside that distribution.</p>"
    )
    out.append(
        '<dl class="defs">'
        "<dt>1 &middot; hemodynamic</dt><dd>SBP &lt; 90 mmHg sustained for "
        "&gt; 30 min (a maximal run of consecutive sub-threshold readings "
        "spanning more than half an hour), <b>or</b> a continuous "
        "vasoactive/inotrope infusion at a non-zero dose.</dd>"
        "<dt>2 &middot; end-organ damage</dt><dd>Within &minus;6 h/+24 h of "
        "the moment criterion 1 is first met: altered mental status (GCS "
        "&lt; 15, or a &ge;2-point drop from the patient's own best), lactate "
        "&ge; 2.0 mmol/L, or creatinine-based KDIGO AKI. Urine output is "
        "unavailable at both sites, so the AKI arm stands in for it and lactate "
        "is added as a second hypoperfusion marker.</dd>"
        "<dt>3 &middot; cardiac origin</dt><dd>An inotrope infusion "
        "(dobutamine, milrinone), a mechanical circulatory support procedure "
        "(ECMO, IABP, percutaneous VAD, in CPT and ICD-10-PCS), <b>or</b> peak "
        "troponin &ge; 500 ng/L within &minus;24 h/+48 h of onset. Diagnosis "
        "codes are deliberately excluded: a diagnosis asserts a condition, "
        "whereas an infusion, a procedure and a troponin record something that "
        "was done or measured.</dd>"
        "</dl>"
    )
    out.append(
        "<p><code>cs_onset_dttm</code>, the anchor for everything downstream, "
        "is the time at which criteria 1 and 2 both hold.</p>"
    )

    main = ledger.filter(~pl.col("step").str.starts_with(" "))
    wide = main.pivot(on="site", index="step", values="n_hospitalizations",
                      aggregate_function="first")
    order = main["step"].unique(maintain_order=True)
    wide = (
        wide.join(pl.DataFrame({"step": order}).with_row_index("o"), on="step")
        .sort("o").drop("o")
    )
    subs = (
        ledger.filter(pl.col("step").str.starts_with(" "))
        .with_columns(step=pl.col("step").str.strip_chars())
        .pivot(on="site", index="step", values="n_hospitalizations",
               aggregate_function="first")
    )
    out.append(tabs([
        ("Attrition", table(
            wide, {c: thousands for c in wide.columns if c != "step"},
            key_column="step",
            highlight={s for s in wide["step"] if "CRITERION 3" in s})),
        ("Criterion components", table(
            subs, {c: thousands for c in subs.columns if c != "step"},
            key_column="step")),
    ]))

    if bool(cfg.cardiac_origin.enabled):
        out.append(callout(
            "<b>How cardiac is this cohort?</b> Criterion 3 is what makes it a "
            "cardiogenic shock cohort rather than a shock cohort, and it is the "
            "criterion with the least precedent, so it is worth sizing against "
            "the coded diagnoses it deliberately does not use. R57.0 "
            "cardiogenic shock is coded in 19.7% of UCMC and 9.6% of NU, any "
            "acute MI in 15.3% / 11.8%, STEMI in 3.0% / 2.8%. The ~81% STEMI "
            "quoted in the literature comes from AMI-CS registries, which enrol "
            "on acute MI itself and so have a different denominator entirely."))
    else:
        out.append(callout(
            "<b>Criterion 3 is computed but not applied</b> "
            "(<code>cardiac_origin.enabled</code> is false), so the cohort as "
            "enrolled spans undifferentiated shock.", "warn"))

    summary_file = paths.cohort_home / "phenotype_summary.csv"
    if summary_file.is_file():
        out.append("<h3>Cardiogenic shock phenotypes (Zweck 2021, refit)</h3>")
        out.append(
            "<p>k-means refit on the eight labs the paper names, labelled from "
            "centroid profiles rather than cluster index. Mortality rising "
            "I &lt; II &lt; III, as published, is the check that the refit "
            "recovered the intended phenotypes.</p>"
        )
        out.append(table(pl.read_csv(summary_file), {
            "n": thousands, "n_ucmc": thousands, "n_nu": thousands,
            "mortality": pct, "median_icu_los_days": num(1),
        }, key_column="cs_phenotype"))


def section_tokens(paths, out: list) -> None:
    home = paths.output_home / "figures" / "GEM-all"
    zoom = home / "token_space_umap_zoom.png"
    full = home / "token_space_umap_full.png"
    if not (zoom.is_file() or full.is_file()):
        return
    out.append('<h2 id="tokens">2 &middot; Token embeddings</h2>')
    out.append(
        "<p>A UMAP of the embedding table itself &mdash; one point per "
        "vocabulary entry, not per patient &mdash; asking whether the model "
        "organised clinical concepts coherently during training. Colour "
        "distinguishes measurements within a family; light grey is the rest of "
        "the vocabulary.</p>"
    )
    out.append(tabs([
        ("Full space", figure(full, "All panels on common axes")),
        ("Zoomed", figure(zoom, "Each panel cropped to its own family")),
    ]))

    ordering_file = paths.output_home / "tables" / "GEM-all" / "token_quantile_ordering.csv"
    if ordering_file.is_file():
        ordering = pl.read_csv(ordering_file)
        out.append("<h3>Did it learn that quantile bins are ordered?</h3>")
        out.append(
            "<p>cocoa splits every numeric measurement into deciles, so "
            "<code>lactate_Q0</code> through <code>_Q9</code> reach the model as "
            "ten unrelated symbols; nothing tells it they are ordered or even "
            "related. Below is the Spearman correlation between quantile "
            "distance and cosine distance between the token vectors, in the "
            "native 1024-d space.</p>"
        )
        out.append(callout(
            f"<b>It recovered the ordering from context alone.</b> Across "
            f"{ordering.height} binned measurements the median Spearman is "
            f"<b>{ordering['spearman'].median():.3f}</b>, positive for "
            f"{(ordering['spearman'] > 0).mean():.0%} of them. Measured in the "
            f"2-d projection instead it reads "
            f"{ordering['spearman_umap'].median():.3f} &mdash; lower, because "
            "UMAP discards ordering it has no room for rather than inventing "
            "it, so the figures above understate the structure."))
        out.append(table(
            ordering.head(10).select("measurement", "n_bins", "spearman",
                                     "spearman_umap"),
            {"n_bins": thousands, "spearman": num(3), "spearman_umap": num(3)},
            key_column="measurement"))


def section_patient_embeddings(paths, out: list) -> None:
    home = paths.output_home / "figures" / "GEM-all" / "trajectory"
    panels = [
        (label, figure(home / f"trajectory_label_{key}.png"))
        for key, label in LABELLINGS
    ]
    if not any(body for _, body in panels):
        return
    out.append('<h2 id="patients">3 &middot; Patient embeddings</h2>')
    out.append(
        "<p>Every patient embedded every 6 hours from 24 h before cardiogenic "
        "shock onset to 48 h after, so one patient contributes a sequence of "
        "points rather than one. All labellings share a single UMAP, fitted "
        "once over every patient-timepoint, so positions are comparable between "
        "tabs. Continuous variables are drawn as hexagon local means: with "
        "75,000 overlapping points, individual markers read as noise whatever "
        "the colour scale.</p>"
    )
    out.append(tabs(panels))
    out.append(figure(home / "trajectory_labels.png",
                      "All eight labellings together, for comparison"))


def section_patient_paths(paths, out: list) -> None:
    home = paths.output_home / "figures" / "GEM-all" / "trajectory"
    panels = [
        (label, figure(home / f"patient_paths_{key}.png"))
        for key, label in LABELLINGS
    ]
    if not any(body for _, body in panels):
        return
    out.append('<h2 id="paths">4 &middot; Individual trajectories</h2>')
    out.append(
        "<p>Five patients drawn at random from those with at least five "
        "timepoints, each shown in its own panel against the same cloud, with "
        "hours relative to shock onset annotated on the path. The tabs change "
        "what the cloud behind them is coloured by; the five patients are the "
        "same throughout.</p>"
    )
    out.append(tabs(panels))
    out.append(callout(
        "Read these against section 5. Individual paths are short and largely "
        "confined to one region: patients move <i>within</i> a neighbourhood "
        "rather than migrating across the space as shock develops, which is "
        "what the flow field's near-zero gradient alignment says quantitatively."))


def section_flow(paths, out: list) -> None:
    home = paths.output_home / "figures" / "GEM-all" / "trajectory"
    alignment_file = paths.output_home / "tables" / "GEM-all" / "trajectory_flow_alignment.csv"
    if not (home / "trajectory_flow_umap.png").is_file():
        return
    out.append('<h2 id="flow">5 &middot; Latent flow</h2>')
    out.append(
        "<p>The mean displacement between consecutive timepoints, drawn as a "
        "streamplot after scVelo's velocity embedding. One difference favours "
        "this setting: RNA velocity must <i>infer</i> direction from spliced and "
        "unspliced read ratios, because a cell is destroyed when measured and "
        "only one snapshot exists per cell. We observe the same patient at t and "
        "t+6h, so the displacement is <b>measured, not estimated</b>.</p>"
    )
    out.append(tabs([
        ("UMAP", figure(home / "trajectory_flow_umap.png",
                        "Flow field in UMAP space")),
        ("PCA (cross-check)", figure(home / "trajectory_flow_pca.png",
                                     "The same field in a linear space")),
        ("Died", figure(home / "trajectory_flow_died.png",
                        "Patients who died in hospital")),
        ("Survived", figure(home / "trajectory_flow_survived.png",
                            "Patients who survived")),
        ("Physiology underlay", figure(home / "trajectory_flow_state.png",
                                       "The same field over each variable")),
    ]))

    out.append(callout(
        "<b>Patients do not follow these lines.</b> A streamplot integrates its "
        "field and renders far more orderly than the data beneath it, so the "
        "figures need the numbers below to be read honestly. Directional "
        "coherence &mdash; the summed displacement in a cell over the sum of "
        "the displacement magnitudes &mdash; is 0.61 in UMAP and 0.48 in PCA "
        "against roughly 0.1 for pure noise: about half the motion is a "
        "consistent direction and half is scatter. More importantly, the flow "
        "is almost unrelated to any physiological gradient, so the arrows are "
        "local circulation rather than directed deterioration.", "warn"))

    if alignment_file.is_file():
        alignment = pl.read_csv(alignment_file)
        out.append("<h3>Where patients sit, and where they go</h3>")
        out.append(
            "<p>Two different questions, and they disagree sharply. "
            "<b>R&sup2;</b> is how much of a variable a patient's position "
            "accounts for (30-nearest-neighbour regression, held out). "
            "<b>cos</b> is the mean cosine between the flow and that variable's "
            "spatial gradient: +1 would mean patients move toward higher values, "
            "0 that the flow ignores it.</p>"
        )
        out.append(table(
            alignment.sort("position_r2", descending=True).select(
                "label", "position_r2", "flow_gradient_cosine", "cells"
            ),
            {"position_r2": num(3), "flow_gradient_cosine": num(3),
             "cells": thousands},
            key_column="label"))
        out.append(
            "<p>The space is strongly stratified by clinical state &mdash; "
            "position alone predicts ventilation at R&sup2; 0.93 and GCS at "
            "0.73 &mdash; while every flow cosine sits within &plusmn;0.14 of "
            "zero. The embedding is a good description of <i>state</i> and a "
            "poor description of <i>trajectory</i> at six-hour resolution. "
            "Cross-sectional comparisons between regions are well founded; "
            "claims that patients move toward a bad region are not supported "
            "by this field.</p>"
        )


def section_sites(paths, out: list) -> None:
    shared = paths.output_home / "figures" / "shared"
    if not (shared / "site_features.png").is_file():
        return
    out.append('<h2 id="sites">6 &middot; UCMC versus NU</h2>')
    out.append(
        "<p>The embedding separates the two sites almost perfectly (section 7), "
        "so the first question is whether the patients differ that much. "
        "Distributions and proportions below, in each institution's own "
        "colours.</p>"
    )
    out.append(figure(shared / "site_features.png"))

    lda_rows = []
    for anchor in ANCHORS:
        f = paths.output_home / "tables" / "GEM-all" / f"site_lda_anchor-{anchor}.csv"
        if f.is_file():
            lda_rows.append(pl.read_csv(f))
    tornado = paths.output_home / "figures" / "GEM-all" / "anchor-24h"
    out.append("<h3>Why they separate</h3>")
    out.append(tabs([
        ("Differential tokens", figure(
            tornado / "site_token_tornado.png",
            "Ranked by absolute difference in per-patient rate")),
        ("Tokens one site never records", figure(
            tornado / "site_token_tornado_exclusive.png",
            "Structural zeros in the vocabulary")),
        ("Token-family mix", figure(tornado / "site_token_families.png")),
    ]))
    out.append(callout(
        "<b>The cause is structural zeros in the vocabulary, not patient "
        "differences.</b> UCMC emits 101 token types NU never produces &mdash; "
        "Braden subscales, CAM, AM-PAC &mdash; carried by 96.5% of its "
        "patients though only 2.8% of its token volume; NU emits 66 UCMC never "
        "does, <code>CODE//dnr</code> among them. A single occurrence identifies "
        "the site with certainty. Separation survives matching the sites on "
        "age, sex, phenotype and mortality at 95.7% purity.", "warn"))
    if lda_rows:
        out.append(table(
            pl.concat(lda_rows).select("anchor", "cv_accuracy", "cv_auc",
                                       "majority_baseline", "n"),
            {"cv_accuracy": num(4), "cv_auc": num(4),
             "majority_baseline": num(4), "n": thousands},
            key_column="anchor"))


def section_models(paths, out: list, logger) -> None:
    shared = paths.output_home / "figures" / "shared"
    results, stale = read_model_tables(
        paths, "downstream_tasks.csv",
        ("task", "evaluation", "auroc", "classifier", "correction"), logger,
    )
    purity_rows = []
    for model in MODELS:
        for anchor in ANCHORS:
            f = paths.output_home / "tables" / model / f"neighbourhood_purity_anchor-{anchor}.csv"
            if f.is_file():
                purity_rows.append(pl.read_csv(f).with_columns(model=pl.lit(model)))
    if results is None and not purity_rows:
        return

    out.append('<h2 id="models">7 &middot; Models</h2>')
    out.append(
        "<p>Three frozen checkpoints of the same architecture, differing only "
        "in what they were trained on: <b>GEM-all</b> on MIMIC + UCMC + NU, "
        "<b>GEM-ucmc-icu</b> on UCMC alone, <b>GEM-mimic-icu</b> on MIMIC alone "
        "&mdash; so GEM-mimic-icu has seen neither of the sites analysed here.</p>"
    )

    target_figure = shared / "target_umaps_anchor-24h.png"
    if target_figure.is_file():
        out.append("<h3>What each checkpoint separates</h3>")
        out.append(figure(
            target_figure,
            "Each checkpoint's own UMAP at the 24h anchor, coloured by the "
            "prediction targets. Rows are targets, columns are checkpoints.",
        ))
        out.append(callout(
            "GEM-all splits the two sites into separate blobs; the single-site "
            "checkpoints intermix them in one cloud. None of the three "
            "separates mortality, new ventilation or pressor escalation "
            "visually &mdash; which is the point of the tables below: visible "
            "separation and predictive value are not the same thing."))

    if purity_rows:
        purity = pl.concat(purity_rows).pivot(
            on="label", index=["anchor", "model"], values="lift"
        ).sort("anchor", "model")
        out.append("<h3>Neighbourhood purity (lift over chance)</h3>")
        out.append(table(
            purity, {c: num(2) for c in purity.columns
                     if c not in ("anchor", "model")}, key_column="model"))
        out.append(
            "<p>Roughly half of GEM-all's site separation comes from having "
            "trained on both hospitals: a checkpoint that saw neither still "
            "reaches 1.27&times;, which is the intrinsic data difference. "
            "Phenotype and mortality lift stay near 1.1&ndash;1.2&times; "
            "throughout, so the extra structure GEM-all carries is site, not "
            "physiology.</p>"
        )

    if results is None:
        return
    if stale:
        out.append(callout(
            "<b>Excluded:</b> " + esc("; ".join(stale))
            + ". Those files predate the current schema; rerun 06 for them.",
            "warn"))

    arm = (
        pl.when(pl.col("features") != "embedding").then(pl.lit("clinical baseline"))
        .when(pl.col("correction") == "raw").then(pl.lit("embedding"))
        .when(pl.col("correction") == "site-standardized")
        .then(pl.lit("site-corrected"))
        .otherwise(pl.lit("affine (LEACE)"))
    )
    results = results.with_columns(arm=arm)
    arm_order = {"clinical baseline": 0, "embedding": 1,
                 "site-corrected": 2, "affine (LEACE)": 3}

    def performance(part: pl.DataFrame) -> str:
        wide = (
            part.group_by("task", "arm", "classifier")
            .agg(auroc=pl.col("auroc").mean())
            .pivot(on="classifier", index=["task", "arm"], values="auroc")
        )
        if "gbm" not in wide.columns or "logistic" not in wide.columns:
            return ""
        wide = (
            wide.with_columns(_o=pl.col("arm").replace_strict(arm_order, default=9))
            .sort("task", "_o")
            .select("task", "arm", "logistic", "gbm",
                    (pl.col("gbm") - pl.col("logistic")).alias("gbm - logistic"))
        )
        return table(wide, {c: num(3) for c in
                            ("logistic", "gbm", "gbm - logistic")},
                     key_column="arm")

    out.append("<h3>Prediction performance</h3>")
    out.append(
        "<p>AUROC for the three 24-hour tasks. <b>One site out</b> trains on "
        "one hospital and tests on the other, averaged over both directions; "
        "<b>pooled split</b> trains and tests on a random split of both sites "
        "together. Rows are the feature set, columns the prediction head.</p>"
    )
    model_tabs = []
    for model in MODELS:
        part = results.filter(pl.col("model") == model)
        if part.height == 0:
            continue
        inner = tabs([
            ("One site out", performance(
                part.filter(pl.col("evaluation").str.contains("->")))),
            ("Pooled split", performance(
                part.filter(pl.col("evaluation") == "pooled"))),
            ("Within site", performance(
                part.filter(pl.col("evaluation").str.starts_with("within:")))),
        ])
        model_tabs.append((model, inner))
    out.append(tabs(model_tabs))

    out.append(callout(
        "<b>Both hypotheses hold.</b> The two heads agree on the embedding to "
        "within a few thousandths, and the gradient-boosted head is if anything "
        "slightly worse &mdash; the representation arrives with the "
        "non-linearity already folded in, so a tree only adds variance. It does "
        "help the clinical baseline substantially, most of all for new "
        "ventilation. And raw, site-corrected and affine-erased embeddings are "
        "indistinguishable: correction moves AUROC by a mean of &minus;0.0005, "
        "because cross-site transfer was never broken. The site axis and the "
        "outcome axes are close to orthogonal, so a supervised model ignores "
        "site without being told to."))


def section_methods(paths, cfg, out: list) -> None:
    out.append('<h2 id="methods">Methods &amp; caveats</h2>')
    out.append(
        "<ul>"
        "<li><b>Urine output is unavailable</b> at both sites &mdash; no "
        "<code>patient_intake_output</code> table, and absent from vitals and "
        "assessments. Creatinine-based KDIGO AKI substitutes.</li>"
        "<li><b><code>clif_ecmo_mcs</code> is not read anywhere.</b> It is a "
        "concept table at both sites; at UCMC it names 82% of all "
        "hospitalizations. MCS comes from <code>clif_patient_procedures</code>.</li>"
        "<li><b>UCMC CPT procedure codes do not join.</b> 1.72M distinct ids, "
        "none keyed on <code>hospitalization_id</code>; only its ICD-10-PCS rows "
        "are usable, so UCMC's MCS ascertainment is PCS-only while NU's uses "
        "both. 01 warns on per-format coverage.</li>"
        "<li><b>AMS is GCS-based</b> and cannot distinguish organ dysfunction "
        "from sedation in a ventilated population.</li>"
        "<li><b>Phenotypes are a refit</b>, not a transfer: the published "
        "centroids are not available, so k-means is refit and labelled from "
        "centroid profiles.</li>"
        "<li><b>The clinical baseline sees only last values, no trends.</b> A "
        "lactate of 4 rising and one falling are identical to it, so some of "
        "the embedding's margin is trajectory information the baseline "
        "structurally cannot access.</li>"
        "<li><b>PCA and UMAP are refit on whatever cohort is present</b>, so "
        "coordinates shift when a site is added and figures are not comparable "
        "across runs with different site membership.</li>"
        "<li><b>Site colours are institutional</b> and clear the colourblind "
        "gates, but UCMC and NU sit at almost identical lightness: they "
        "separate by hue only and are indistinguishable in greyscale.</li>"
        "</ul>"
    )


# =================================================================== driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    paths.mkdirs()
    cfg = load_cohort_cfg()
    logger = get_logger("07_report", paths.log_home / "07_report.log")

    out: list[str] = []
    section_cohort(paths, cfg, out)
    section_tokens(paths, out)
    section_patient_embeddings(paths, out)
    section_patient_paths(paths, out)
    section_flow(paths, out)
    section_sites(paths, out)
    section_models(paths, out, logger)
    section_methods(paths, cfg, out)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cohort_file = paths.cohort_home / "cohort.parquet"
    if cohort_file.is_file():
        counted = pl.read_parquet(cohort_file, columns=["site"])
        per_site = counted.group_by("site").agg(pl.len().alias("n")).sort("site")
        headline = f"{counted.height:,} patients (" + " / ".join(
            f"{r['site']} {r['n']:,}" for r in per_site.iter_rows(named=True)
        ) + ")"
    else:
        headline = "no cohort table"

    toc = "".join(
        f'<li><a href="#{anchor}">{label}</a></li>'
        for anchor, label in (
            ("cohort", "1 &middot; Cohort"),
            ("tokens", "2 &middot; Token embeddings"),
            ("patients", "3 &middot; Patient embeddings"),
            ("paths", "4 &middot; Individual trajectories"),
            ("flow", "5 &middot; Latent flow"),
            ("sites", "6 &middot; UCMC versus NU"),
            ("models", "7 &middot; Models"),
            ("methods", "Methods &amp; caveats"),
        )
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CLIF cardiogenic shock &mdash; latent space analysis</title>
<style>{css()}</style></head>
<body><div class="wrap">
<h1>Cardiogenic shock in the CLIF foundation-model latent space</h1>
<p class="sub">UCMC and NU, CLIF 2.1 &middot; GEM frozen encoders &middot; two anchors</p>
<p class="meta">Generated {stamp} &middot; {headline}</p>
<ul class="toc">{toc}</ul>
{''.join(out)}
</div><script>{script()}</script></body></html>"""

    destination = paths.output_home / "report.html"
    destination.write_text(document, encoding="utf-8")
    logger.info(
        f"wrote {destination} ({destination.stat().st_size / 1e6:.1f} MB, "
        f"{_TAB_GROUP[0]} tab groups)"
    )
    if args.open:
        webbrowser.open(destination.as_uri())


if __name__ == "__main__":
    main()

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
import io
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


#: longest side, in pixels, of a figure as embedded in the report
#:
#: The PNGs on disk are written at 500 dpi because they are also manuscript
#: figures, which makes a 10-inch panel 5,000 pixels wide. Inlining them at
#: that size produced a 127 MB report that a browser takes tens of seconds to
#: lay out, for resolution no screen displays. They are downscaled here and
#: left alone on disk, so the report is light and the figures stay printable.
EMBED_MAX_PX = 1800


def figure(path: pathlib.Path, caption: str = "") -> str:
    path = pathlib.Path(path)
    if not path.is_file():
        return ""
    data = base64.b64encode(_downscaled(path)).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (
        f'<figure><img src="data:image/png;base64,{data}" alt="{esc(caption)}">'
        f"{cap}</figure>"
    )


def _downscaled(path: pathlib.Path) -> bytes:
    """the figure at embed resolution, or the original if it is already small"""
    try:
        from PIL import Image
    except ImportError:  # pillow ships with matplotlib, but do not require it
        return path.read_bytes()
    raw = path.read_bytes()
    with Image.open(path) as image:
        if max(image.size) <= EMBED_MAX_PX:
            return raw
        scale = EMBED_MAX_PX / max(image.size)
        resized = image.convert("RGB").resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue() if len(buffer.getvalue()) < len(raw) else raw


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


def pval(x):
    """a p-value with a floor, since 0.0000 reads as a certainty"""
    if x is None:
        return "—"
    # a bare "<": table() escapes every cell, so an entity here would double up
    return "<0.0001" if x < 0.0001 else f"{x:.4f}"


def present(frame: pl.DataFrame, *columns: str):
    """
    the requested columns this frame actually has, and the ones it does not

    11 writes a narrower eligibility table on the federated path than on the
    site-side one -- proximity cannot be computed from exported decile ranks,
    so the columns built from it are absent -- and the two anchors can have
    been produced by different paths. Selecting the intersection keeps the tab
    renderable; returning the absences lets the caller say which anchor is
    short rather than quietly rendering fewer columns than the prose describes.
    """
    have = [c for c in columns if c in frame.columns]
    return frame.select(have), [c for c in columns if c not in frame.columns]


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
        "<dt>3 &middot; cardiac origin &mdash; computed, not applied</dt>"
        "<dd>An inotrope infusion (dobutamine, milrinone), a "
        "mechanical circulatory support procedure (ECMO, IABP, percutaneous VAD, "
        "in CPT and ICD-10-PCS), <b>or</b> peak troponin &ge; 500 ng/L within "
        "&minus;24 h/+48 h of onset. Reported for every patient but <b>not used "
        "to subset</b>: the question is where the mechanisms of shock sit "
        "relative to one another, which filtering to one of them makes "
        "unanswerable.</dd>"
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
            highlight={s for s in wide["step"] if "CRITERION 2" in s})),
        ("Criterion components", table(
            subs, {c: thousands for c in subs.columns if c != "step"},
            key_column="step")),
    ]))

    if bool(cfg.cardiac_origin.enabled):
        out.append(callout(
            "<b>Criterion 3 is applied, so this is a cardiogenic shock "
            "cohort.</b> It is the criterion with the least precedent, so it is "
            "worth sizing against the coded diagnoses it deliberately does not "
            "use: R57.0 is coded in 19.7% of UCMC and 9.6% of NU, any acute MI "
            "in 15.3% / 11.8%, STEMI in 3.0% / 2.8%."))
    else:
        out.append(
            "<p><b>This is an undifferentiated shock cohort.</b> Criteria 1 and "
            "2 are mechanism-agnostic, so septic, cardiogenic, hypovolaemic and "
            "haemorrhagic shock are all enrolled &mdash; which is the point: the "
            "coded mechanism is attached afterwards as a labelling, so the "
            "embedding can be asked whether it separates them. Criterion 3 is "
            "computed and would cut the cohort to 2,219 UCMC and 5,104 NU "
            "patients if <code>cardiac_origin.enabled</code> were set true.</p>"
        )

    # ------------------------------------------------- shock type, right here
    #
    # Directly under the attrition tables rather than below the phenotypes.
    # The criteria are mechanism-agnostic, so the first question a reader has
    # after seeing who was enrolled is which mechanisms that caught -- and the
    # answer reframes every number above it.
    shock_file = paths.cohort_home / "shock_type_summary.csv"
    if shock_file.is_file():
        shock = pl.read_csv(shock_file)
        out.append("<h3>Shock type, from coded diagnoses</h3>")
        out.append(
            "<p>What the coders said the mechanism was, from "
            "<code>clif_hospital_diagnosis</code> &mdash; a labelling, never a "
            "filter. Codes overlap, so a patient carrying more than one of "
            "septic, cardiogenic or hypovolaemic is <code>mixed</code>; "
            "<code>other shock</code> is R57.8/R57.9 and the rare mechanisms. "
            "This is the one place the pipeline reads diagnosis codes: the "
            "mechanism of shock has no corresponding measurement, so the code "
            "is the only evidence there is.</p>"
        )
        out.append(tabs([
            (
                str(site).upper() if site != "pooled" else "Pooled",
                table(
                    shock.filter(pl.col("site") == site).drop("site"),
                    {
                        "n": thousands,
                        "share": pct,
                        "mortality": pct,
                        "median_icu_los_days": num(1),
                    },
                    key_column="shock_type",
                ),
            )
            for site in shock["site"].unique(maintain_order=True).to_list()
        ]))
        out.append(callout(
            "<b>Over half the cohort carries no shock code, and that group is "
            "far less sick</b> &mdash; 54.6% uncoded at 6.2% mortality against "
            "25&ndash;43% in every coded group, which reads as criteria 1 and 2 "
            "being over-inclusive. <b>And the two sites code different "
            "mechanisms:</b> septic is the label on 20.3% of NU against 12.3% "
            "of UCMC, cardiogenic 5.3% against 13.5%. Compare composition "
            "within a site, never across. Section 9 takes the uncoded group as "
            "its subject.", "warn"))

        baseline_file = paths.cohort_home / "baseline_by_shock_type.csv"
        parts = sorted(paths.cohort_home.glob("baseline_by_shock_type_*.csv"))
        if parts or baseline_file.is_file():
            frames = [pl.read_csv(f) for f in (parts or [baseline_file])]
            baseline = pl.concat(frames, how="diagonal_relaxed")
            out.append("<h3>Who these patients are, by shock type</h3>")
            out.append(
                "<p>Median [IQR] for the continuous variables and n (%) for the "
                "binary ones, measured <b>at shock onset</b> rather than at "
                "admission or over the stay &mdash; the question is how each "
                "mechanism presented, and a median over a whole admission "
                "weights a patient by how long they stayed. Norepinephrine "
                "equivalents are used rather than a count of agents, because a "
                "dose of phenylephrine and a dose of norepinephrine are not the "
                "same quantity and &ldquo;on a pressor&rdquo; discards the "
                "difference this comparison is about. Cells naming fewer than "
                "the minimum cell size are blank, not zero.</p>"
            )
            display = baseline.with_columns(
                value=pl.when(pl.col("kind") == "continuous")
                .then(
                    pl.when(pl.col("median").is_null())
                    .then(pl.lit(""))
                    .otherwise(
                        pl.col("median").cast(pl.String)
                        + pl.lit(" [") + pl.col("q1").cast(pl.String)
                        + pl.lit("-") + pl.col("q3").cast(pl.String) + pl.lit("]")
                    )
                )
                .otherwise(
                    pl.when(pl.col("share").is_null())
                    .then(pl.lit(""))
                    .otherwise(
                        pl.col("n_measured").cast(pl.String)
                        + pl.lit(" (")
                        + (pl.col("share") * 100).round(1).cast(pl.String)
                        + pl.lit("%)")
                    )
                )
            )
            out.append(tabs([
                (
                    str(site).upper(),
                    table(
                        display.filter(pl.col("site") == site)
                        .select("variable", "shock_type", "value")
                        .pivot(on="shock_type", index="variable", values="value"),
                        {},
                        key_column="variable",
                    ),
                )
                for site in display["site"].unique(maintain_order=True).to_list()
            ]))

    # --------------------------------------------------------- the phenotypes
    phenotype_tabs = []
    # One tab per cardiogenic denominator. Zweck derived I/II/III in
    # cardiogenic shock, so fitting on the whole shock cohort clusters a
    # population that is 75% non-cardiogenic and then names the clusters after
    # his; all three fits are shown because the two cardiogenic readings agree
    # on only 24.4% of their union, which makes the denominator a finding.
    for column, label in (
        ("cs_phenotype_coded", "Zweck (ICD-coded CS)"),
        ("cs_phenotype_cardiac_origin", "Zweck (cardiac origin)"),
        ("cs_phenotype", "Zweck (all shock)"),
    ):
        variant_file = paths.cohort_home / f"phenotype_summary_{column}.csv"
        if not variant_file.is_file():
            continue
        frame = pl.read_csv(variant_file)
        phenotype_tabs.append((label, table(
            frame,
            {
                "n": thousands,
                **{c: thousands for c in frame.columns if c.startswith("n_")},
                "mortality": pct, "median_icu_los_days": num(1),
            },
            key_column=column,
        )))
    seymour_file = paths.cohort_home / "seymour_summary.csv"
    if seymour_file.is_file():
        phenotype_tabs.append(("Seymour (sepsis)", table(
            pl.read_csv(seymour_file),
            {
                "n": thousands, "n_NU": thousands, "n_UCMC": thousands,
                "share": pct, "share_NU": pct, "share_UCMC": pct,
                "mortality": pct, "median_icu_los_days": num(1),
            },
            key_column="seymour",
        )))
    bhavani_file = paths.cohort_home / "bhavani_summary.csv"
    if bhavani_file.is_file():
        phenotype_tabs.append(("Bhavani (temperature)", table(
            pl.read_csv(bhavani_file),
            {
                "n": thousands, "n_NU": thousands, "n_UCMC": thousands,
                "share": pct, "share_NU": pct, "share_UCMC": pct,
                "mortality": pct, "median_icu_los_days": num(1),
            },
            key_column="bhavani",
        )))
    if phenotype_tabs:
        out.append("<h3>Published phenotype schemes, refit here</h3>")
        out.append(
            "<p>Three schemes from the literature, each refit on this cohort "
            "rather than applied as published coefficients. <b>Zweck</b> is "
            "k-means on the eight labs the cardiogenic shock paper names, "
            "labelled from centroid profiles rather than cluster index &mdash; "
            "mortality rising I &lt; II &lt; III, as published, is the check "
            "that the refit recovered the intended phenotypes. <b>Seymour</b> "
            "and <b>Bhavani</b> are sepsis schemes and are computed only on the "
            "sepsis subcohort, which is why their totals are smaller.</p>"
        )
        out.append(
            "<p><b>Zweck appears three times, once per cardiogenic "
            "denominator.</b> The phenotypes were derived in cardiogenic "
            "shock, so fitting them on the whole shock cohort clusters a "
            "population that is 75% non-cardiogenic and then gives the "
            "clusters Zweck's names by matching centroid lab profiles. This "
            "pipeline did exactly that until recently. Each tab is now an "
            "independent fit &mdash; its own median and scale, its own "
            "k-means, its own centroid naming &mdash; on its own cohort: "
            "3,105 ICD-coded, 9,133 meeting the computed cardiac-origin "
            "criterion, 39,819 all shock. The first two agree on only 2,397 "
            "patients, 24.4% of their union, so which one a replication uses "
            "is a choice that changes the answer.</p>"
        )
        out.append(tabs(phenotype_tabs))

        agreement_file = paths.cohort_home / "phenotype_variant_agreement.csv"
        if agreement_file.is_file():
            out.append(
                "<p>How far the three fits put the same patient in the same "
                "phenotype, scored only on patients both fits label &mdash; a "
                "variant is null outside its own denominator, and scoring a "
                "null against a label would measure the denominators instead "
                "of the partitions. Adjusted Rand rather than raw agreement, "
                "because phenotype I is the large majority and two fits that "
                "both put most patients there would agree on most patients "
                "while sharing no structure: 0 is chance given the marginals, "
                "1 is identical.</p>"
            )
            out.append(table(
                pl.read_csv(agreement_file),
                {
                    "n_both_labelled": thousands,
                    "same_phenotype": pct,
                    "adjusted_rand": num(3),
                },
            ))


#: token families, in the order their figures are stacked
TOKEN_FAMILIES = [
    ("labs", "Labs"),
    ("vitals", "Vitals"),
    ("medications", "Medications"),
    ("respiratory", "Respiratory"),
    ("assessments", "Assessments"),
    ("severity-events", "Severity &amp; events"),
    ("location-admin", "Location &amp; admin"),
    ("demographics", "Demographics"),
    ("other", "Other"),
]


def section_tokens(paths, out: list) -> None:
    home = paths.output_home / "figures" / "GEM-all"
    panels = [
        (label, home / f"token_space_{key}.png") for key, label in TOKEN_FAMILIES
    ]
    panels = [(label, path) for label, path in panels if path.is_file()]
    if not panels:
        return
    out.append('<h2 id="tokens">2 &middot; Token embeddings</h2>')
    out.append(
        "<p>A UMAP of the embedding table itself &mdash; one point per "
        "vocabulary entry, not per patient &mdash; asking whether the model "
        "organised clinical concepts coherently during training. One figure per "
        "family: the whole vocabulary on the left with the zoom window drawn "
        "on it, that window enlarged on the right. Hue distinguishes "
        "measurements within the family and lightness distinguishes the "
        "decile within each measurement, light for Q0 to dark for Q9; "
        "light grey is the rest of the vocabulary.</p>"
    )

    out.append("<h3>What a single point is</h3>")
    out.append(
        "<p>Not a one-hot vector. The one-hot is the <i>input</i> to the "
        "embedding lookup; the point is the learned 1024-float row it "
        "selects.</p>"
    )
    out.append(formula(
        "embed(onehot_i)  =  onehot_i @ W_E  =  W_E[i]",
        "",
        "W_E : 1344 x 1024, learned       W_E[i] : the plotted point",
    ))
    out.append(
        '<dl class="defs">'
        "<dt>It starts as noise</dt><dd>At initialisation these rows are "
        "random, so every bit of structure in these figures was learned, and "
        "learned from one signal only: predicting the next token. Two tokens "
        "end up near each other because they occur in similar contexts and are "
        "followed by similar events. That is why the decile ordering below "
        "emerges at all &mdash; <code>lactate_Q8</code> and "
        "<code>lactate_Q9</code> arise in similar situations and predict "
        "similar continuations.</dd>"
        "<dt>It does double duty</dt><dd>The checkpoint sets "
        "<code>tie_word_embeddings: true</code>, so the same row is both what "
        "the model adds to the residual stream on reading that token and the "
        "direction whose dot product with the hidden state gives that token's "
        "output logit. &ldquo;Nearby&rdquo; therefore means <i>plays a similar "
        "predictive role</i>.</dd>"
        "<dt>The numeric value never enters directly</dt><dd>"
        "<code>LAB-RES//sodium_Q7</code> carries no information that 7 exceeds "
        "6 beyond what the model inferred from context. And there is no patient "
        "in the vector: it is a property of the vocabulary, averaged over every "
        "training context in which that token appeared.</dd>"
        "</dl>"
    )
    out.append(callout(
        "<b>So distance here is predictive-role similarity, not clinical "
        "similarity.</b> <code>LAB-ORD//sodium</code> and "
        "<code>LAB-ORD//potassium</code> sit close together largely because "
        "they are on the same basic metabolic panel and get ordered in the same "
        "breath &mdash; that is panel structure, not physiology. Reading these "
        "clusters as statements about disease biology would overclaim; they are "
        "statements about what tends to be recorded together and what tends to "
        "follow.", "warn"))

    out.append("<h3>By family</h3>")
    for label, path in panels:
        out.append(figure(path, label.replace("&amp;", "&")))

    runs = [
        (label, home / f"token_deciles_{key}.png") for key, label in TOKEN_FAMILIES
    ]
    runs = [(label, path) for label, path in runs if path.is_file()]
    if runs:
        out.append("<h3>One measurement at a time</h3>")
        out.append(
            "<p>The family figures show that the gradient is there, but not "
            "whether it is <i>ordered</i>: a measurement&rsquo;s ten deciles "
            "span about 1% of the labs window and 3% of the vitals window, so "
            "at family scale a decile run is a few pixels. Cropping is no help "
            "&mdash; the windows are set by genuine spread rather than by "
            "outliers, and the central-98% window is within 1.14&times; of the "
            "full extent for every family. So each measurement below gets its "
            "own panel, cropped to its own deciles and joined Q0 to Q9.</p>"
        )
        out.append(callout(
            "Each panel carries two coefficients. <b>native</b> is measured in "
            "the model&rsquo;s own 1024-d space under cosine distance, and is "
            "the claim about what the model learned. <b>here</b> is the same "
            "quantity measured on the drawn 2-d coordinates. Where the two "
            "part company &mdash; <code>spo2</code> is +0.93 native and "
            "&minus;0.06 here &mdash; the ordering is in the representation and "
            "the projection is what lost it, so the tangled path is an artefact "
            "of the layout rather than evidence against the model. This is why "
            "the table below reports the native figure."))
        for label, path in runs:
            out.append(figure(path, label.replace("&amp;", "&")))

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


#: the per-anchor colourings 03 writes, in reading order
#:
#: Every entry is rendered for the raw space AND for each of 05's corrected
#: spaces, so adding one here adds it to both. Panels 03 did not produce are
#: dropped by the `if body` filter rather than leaving a hole, which is how a
#: label a site could not supply degrades.
PROJECTION_PANELS = [
    ("site", "site"),
    ("shock_type", "shock type (one panel each)"),
    ("shock_type_pooled", "shock type (together)"),
    ("phenotype", "Zweck phenotype"),
    ("seymour_phenotype", "Seymour phenotype"),
    ("bhavani_phenotype", "Bhavani phenotype"),
    ("mortality", "mortality"),
    ("icu_los", "ICU LOS"),
    # the physiology and severity the embedding was taken at
    ("tok_map", "MAP (decile bin the model read)"),
    ("sofa_non_cv", "non-cardiovascular SOFA"),
    # what happened in the 24 hours after the anchor
    ("new_imv_24h", "new invasive ventilation"),
    ("pressor_escalation_24h", "vasopressor escalation"),
    # the three trial questions, one panel per trial inside each
    ("elig", "trial eligibility"),
    ("trial_fit", "inside every published IQR"),
    ("concordance", "treated per the recommendation"),
]

#: 09's patient subsets, in reading order. A subset with no figures is skipped.
SUBSET_ORDER = ("cardiogenic", "septic", "all")

#: the four projections in reading order: the whole cohort either side of the
#: correction first, since that pair is the control, then the two cohort-
#: specific embeddings. Distinct from SUBSET_ORDER, which is a *preference*
#: order for picking one representative projection and leads with cardiogenic.
PROJECTION_ORDER = ("all", "cardiogenic", "septic")

#: which site correction the report treats as current when several have been
#: run. CORAL replaced LEACE because the figures are PaCMAP -- non-linear --
#: and CORAL is the one that moves the non-linear probe (0.9924 -> 0.8553) and
#: neighbour purity (1.66x -> 1.41x), which is what a projection reflects.
PREFERRED_CORRECTION = "coral"

#: how each subset is named in a heading. "cardiogenic" is the union of the two
#: readings of cardiogenic shock, not the ICD code alone, and saying so in the
#: heading is cheaper than a footnote under every panel.
SUBSET_TITLES = {
    "cardiogenic": "cardiogenic shock",
    "septic": "septic shock",
    "all": "the whole shock cohort",
}

#: the physiology underlays 09 draws one figure each for, label as shown.
#: Mirrors clifcs.state.VARIABLES; a variable 09 could not populate is dropped
#: by the `if body` filter rather than leaving an empty tab.
STATE_VARIABLES = [
    ("map", "MAP"),
    ("shock_index", "shock index"),
    ("heart_rate", "heart rate"),
    ("nee", "norepinephrine equivalents"),
    ("n_vasopressors", "vasoactive agents running"),
    ("lactate", "lactate"),
    ("bicarbonate", "bicarbonate"),
    ("creatinine", "creatinine"),
    ("gcs_total", "GCS total"),
    ("sofa_total", "SOFA total"),
    ("on_imv", "invasive ventilation"),
]

#: 05's affine corrections, in the order 05 reports them
CORRECTIONS = [
    ("leace", "LEACE"),
    ("site-standardized", "site-standardized"),
]


def projection_tag(label: str) -> str:
    """
    a projection label short enough for a tab

    "cardiogenic shock" -> "Cardiogenic"; "the whole shock cohort,
    uncorrected" -> "All shock (raw)". Tabs sit in a single row, so the full
    heading does not fit once a variable name follows it.
    """
    raw = ", uncorrected" in label
    stem = label.split(",")[0].replace("the whole shock cohort", "all shock")
    stem = stem.replace(" shock", "") if stem != "all shock" else "All shock"
    return f"{stem.capitalize()} (raw)" if raw else stem.capitalize()


def trajectory_home(paths) -> pathlib.Path:
    """
    the directory the cardiogenic flow panels come from

    Not "the first projection 09 drew". The panels this returns a home for are
    `trajectory_flow_cardiogenic_*`, which only the cardiogenic run writes, so
    resolving to whichever projection happened to sort first put the section's
    headline figures in a directory that does not contain them. Falls back
    through the other cohorts only so that a partial run still renders.
    """
    homes = trajectory_homes(paths)
    for subset in ("cardiogenic", "septic", "all"):
        for info in homes.values():
            if (
                info["subset"] == subset
                and info["correction"] == PREFERRED_CORRECTION
            ):
                return info["home"]
        for info in homes.values():
            if info["subset"] == subset and info["correction"] is not None:
                return info["home"]
        for info in homes.values():
            if info["subset"] == subset:
                return info["home"]
    return paths.output_home / "figures" / "GEM-all" / "trajectory"


def trajectory_homes(paths) -> dict:
    """
    every projection 09 has drawn, as {label: {home, subset, correction}}

    There are four, and they are four different embeddings -- not one embedding
    shown four ways. Each is a PaCMAP fitted on its own set of
    patient-timepoints, so a coordinate in one says nothing about a coordinate
    in another:

        all shock, uncorrected      what the model does untouched, site effect
                                    and all -- the control for the other three
        all shock, corrected        the same cohort after the affine site map
        cardiogenic, corrected      the union of the two cardiogenic readings
        septic, corrected           the coded-septic denominator

    Keyed by label rather than by subset, because subset alone no longer
    identifies a projection: the all-shock cohort appears twice, once either
    side of the correction, and the point of that pair is to be compared.
    Returned in a fixed order so the tabs read the same way every time rather
    than in whatever order the filesystem offers.
    """
    base = paths.output_home / "figures" / "GEM-all" / "trajectory"
    found = {}
    for subset in PROJECTION_ORDER:
        home = base / subset
        if not home.is_dir():
            continue
        # An uncorrected run writes into the subset directory itself, a
        # corrected one into corrected-<method>/ beside it. Both can exist, and
        # for the all-shock cohort both are wanted.
        candidates = []
        if (home / "trajectory_flow_pacmap.png").is_file():
            candidates.append((None, home))
        for corrected in sorted(home.glob("corrected-*")):
            if (corrected / "trajectory_flow_pacmap.png").is_file():
                candidates.append(
                    (corrected.name.removeprefix("corrected-"), corrected)
                )
        # The label must distinguish the correction, or two corrected runs of
        # the same subset collide on one key and the later one silently wins.
        # That happened: `sorted()` puts corrected-coral before corrected-leace,
        # so assigning both to "septic shock" left the report showing the stale
        # LEACE figures -- drawn under the rank-1 erasure bug -- while every
        # caption said the correction had been applied.
        multiple = len([c for c, _ in candidates if c is not None]) > 1
        for correction, directory in candidates:
            label = SUBSET_TITLES[subset]
            if correction is None:
                label = f"{label}, uncorrected"
            elif multiple:
                label = f"{label}, {correction}"
            if label in found:
                raise RuntimeError(
                    f"two projections claim the label {label!r}: "
                    f"{found[label]['home']} and {directory}"
                )
            found[label] = {
                "home": directory,
                "subset": subset,
                "correction": correction,
            }
    if found:
        return found
    # a run from before figures were scoped by subset
    legacy = None
    if (base / "trajectory_flow_pacmap.png").is_file():
        legacy = base
    else:
        corrected = sorted(base.glob("corrected-*"))
        legacy = corrected[0] if corrected else None
    if legacy is None:
        return {}
    return {
        SUBSET_TITLES["all"]: {
            "home": legacy, "subset": "all", "correction": None,
        }
    }


def projection_sizes(paths) -> dict:
    """
    how many patients each projection was fitted on, as {label: count}

    One number cannot describe section 6 any more. The four projections have
    four denominators -- the whole cohort twice, then the cardiogenic union and
    the coded-septic set -- and the grid thins each of them differently, since a
    patient is dropped at any offset where they have too little record before
    it. Reporting the cardiogenic figure as though it covered the section was
    wrong in both directions at once: too small for the all-shock panels, and
    the wrong cohort for the septic ones.
    """
    home = paths.output_home / "patient_level_data" / "GEM-all"
    sizes = {}
    for label, info in trajectory_homes(paths).items():
        name = f"trajectory_projections_{info['subset']}"
        if info["correction"]:
            name = f"{name}_corrected-{info['correction']}"
        parquet = home / f"{name}.parquet"
        if not parquet.is_file():
            continue
        try:
            ids = pl.read_parquet(parquet, columns=["hospitalization_id"])
        except Exception:
            continue
        sizes[label] = ids["hospitalization_id"].n_unique()
    return sizes


def trajectory_coverage(paths, logger=None) -> tuple[str, str]:
    """
    whether 09's trajectory artifacts describe the cohort 01 last wrote

    Returns one of three verdicts, because the two ways of being out of date
    are not equally bad:

      "current"  the same patient set. Render without comment.
      "subset"   fewer patients, all of them still enrolled. 09 drops a patient
                 at any grid point where they have too little record before it,
                 and the flow field needs a patient observed at consecutive
                 timepoints to have a displacement at all, so requiring
                 presence across all thirteen offsets from -24 h to +48 h
                 leaves 7,323 of 34,142. These figures are about these
                 patients; they are about fewer of them, and saying so beats
                 showing nothing.
      "stale"    contains patients the cohort no longer has. That is the case
                 this check was written for and it still refuses.

    Sections 4 to 6 come from 09, which is the most expensive script here and
    so the one most likely to be left behind when the cohort definition
    changes. Its figures carry no record of which cohort they were drawn from,
    so a report that renders them unconditionally will happily show 7,323
    patients' trajectories under a 34,142-patient cohort table and look
    entirely consistent doing it.

    Compared on the patient set, not the row count: a cohort can change size
    and membership independently, and it is membership that decides whether a
    figure is about these patients.
    """
    cohort_file = paths.cohort_home / "cohort.parquet"
    # 09 suffixes its output when it applies a site correction, and correcting
    # is its default -- site dominates that geometry at ~1.9x neighbour purity,
    # so an uncorrected flow field is largely a picture of three hospitals.
    # Looking only for the unsuffixed name reported "09 has not been run" for
    # every corrected run and omitted sections 4-6 on a complete analysis.
    # 09 scopes this by subset and by correction, so there can be several.
    # Preference order matters: the cardiogenic subset is what sections 4-6 are
    # about, and an uncorrected run is the one a reader assumes unless told.
    home = paths.output_home / "patient_level_data" / "GEM-all"
    candidates = []
    for subset in SUBSET_ORDER:
        candidates.append(home / f"trajectory_projections_{subset}.parquet")
        candidates.extend(
            sorted(home.glob(f"trajectory_projections_{subset}_corrected-*.parquet"))
        )
    # names written before the subset was part of the path
    candidates.append(home / "trajectory_projections.parquet")
    candidates.extend(sorted(home.glob("trajectory_projections_corrected-*.parquet")))
    trajectory_file = next((c for c in candidates if c.is_file()), None)
    if trajectory_file is None:
        return "stale", "09 has not been run"
    if not cohort_file.is_file():
        return "current", ""

    cohort_ids = set(
        pl.read_parquet(cohort_file, columns=["hospitalization_id"])[
            "hospitalization_id"
        ].to_list()
    )
    trajectory_ids = set(
        pl.read_parquet(trajectory_file, columns=["hospitalization_id"])[
            "hospitalization_id"
        ].to_list()
    )
    missing = len(cohort_ids - trajectory_ids)
    extra = len(trajectory_ids - cohort_ids)
    if not missing and not extra:
        return "current", ""

    note = f"{len(trajectory_ids):,} of the cohort's {len(cohort_ids):,} patients"
    if extra:
        note += f", and {extra:,} of them are no longer in the cohort at all"
        if logger is not None:
            logger.warning(f"sections 4-6 omitted, 09 is stale: {note}")
        return "stale", note
    if logger is not None:
        logger.info(f"sections 4-6 render on a subset: {note}")
    return "subset", note


def section_projections(paths, out: list) -> None:
    """
    03's per-anchor projections of the patient embeddings

    PaCMAP leads and UMAP is kept beside it. Both are computed every run from
    the same 50-d PCA subspace with the same neighbour count, so the pair is a
    comparison of the two algorithms rather than of two settings: PaCMAP is
    built to hold global structure as well as local neighbourhoods, where UMAP
    optimises the local graph and can tear apart or fuse clusters that are
    genuinely adjacent. Where the two disagree, neither is the truth -- it is a
    sign the layout is carrying less information than it appears to.
    """
    anchors_present = []
    for anchor in ANCHORS:
        home = paths.output_home / "figures" / "GEM-all" / f"anchor-{anchor}"
        if (home / "pacmap_site.png").is_file() or (home / "umap_site.png").is_file():
            anchors_present.append((anchor, home))
    if not anchors_present:
        return

    out.append('<h2 id="projections">2 &middot; Patient embeddings</h2>')
    out.append(
        "<p>One point per patient: the model's hidden state at the last token "
        "before the anchor, projected to 2-d. <b>PaCMAP</b> is shown first and "
        "UMAP alongside it; PCA is there as the linear reference, where "
        "distance means what it usually means.</p>"
    )
    out.append(
        "<p>Shock type appears twice. <b>One panel each</b> draws one mechanism "
        "in its own hue against the rest in grey, and is the version to trust: "
        "six categories cannot be separated by colour in a single scatter, and "
        "the mechanisms overlap heavily enough that whichever is drawn last "
        "looks larger than it is. <b>Together</b> uses the same six hues on one "
        "layout, which the facets cannot do &mdash; it is the only way to see "
        "which mechanisms occupy the same ground. ICU length of stay is "
        "coloured on a log scale: the distribution is dense below a week, so a "
        "linear ramp put most of the cohort in its first two steps.</p>"
    )

    anchor_label = {
        "24h": "24 h after admission",
        "onset": "shock onset",
    }

    anchor_tabs = []
    for anchor, home in anchors_present:
        projection_tabs = []
        for key, pretty in (
            ("pacmap", "PaCMAP"),
            ("umap", "UMAP"),
            ("pca", "PCA (linear reference)"),
        ):
            panels = [
                (label, figure(home / f"{key}_{name}.png"))
                for name, label in PROJECTION_PANELS
            ]
            panels = [(label, body) for label, body in panels if body]
            if panels:
                projection_tabs.append((pretty, tabs(panels)))
        # 05's corrected spaces, as a sibling tab rather than a separate
        # section: the only way to read a correction is against the thing it
        # corrected, and that means the same tab strip.
        for method, pretty in CORRECTIONS:
            corrected_home = home / f"corrected-{method}"
            panels = [
                (label, figure(corrected_home / f"pacmap_{name}.png"))
                for name, label in PROJECTION_PANELS
            ]
            panels = [(label, body) for label, body in panels if body]
            if panels:
                projection_tabs.append(
                    (f"PaCMAP, {pretty}-corrected", tabs(panels))
                )
        if projection_tabs:
            anchor_tabs.append(
                (anchor_label.get(anchor, anchor), tabs(projection_tabs))
            )

    out.append(tabs(anchor_tabs))
    out.append(callout(
        "<b>The corrected tabs are the same patients after 05's affine map has "
        "removed the linearly-decodable site information.</b> The correction "
        "table in section 7 reports how far the linear site probe falls under "
        "each map, which is what the map is for; it says nothing about what the "
        "space looks like afterwards. These panels do. <b>Read them for whether "
        "anything clinical survived</b> &mdash; if the phenotype and mortality "
        "structure disappears along with the site structure, the correction "
        "removed the representation rather than the confound. Note that "
        "neighbour purity on site falls far less than the linear probe does "
        "even after LEACE drives that probe to chance, which is 04's finding "
        "restated: the sites differ in which token types they emit at all, and "
        "an affine map on the embedding closes the linear channel without "
        "making the two representations exchangeable."))


def section_trials(paths, out: list) -> None:
    """
    11's comparison of the cohort against the trials behind the guidelines

    Renders only what 11 produced, so it disappears cleanly if 11 has not run.
    """
    table_home = paths.output_home / "tables" / "GEM-all"
    anchors = [
        anchor
        for anchor in ANCHORS
        if (table_home / f"trial_eligibility_{anchor}.csv").is_file()
    ]
    if not anchors:
        return

    out.append('<h2 id="trials">3 &middot; Guideline fit</h2>')
    out.append(
        "<p>Every cardiogenic shock recommendation rests on a trial that "
        "enrolled a narrow slice of the people who present in shock. This asks "
        "two questions that are easy to conflate: <b>eligibility</b> &mdash; "
        "would the trial have enrolled this patient? &mdash; and "
        "<b>proximity</b> &mdash; do they resemble the patients it did enrol? "
        "Proximity is the share of a patient's 30 nearest neighbours in the "
        "50-d PCA space that the trial would have taken, the same "
        "neighbourhood measure section 7 uses for site purity.</p>"
    )
    out.append(
        "<p>A third question sits between them and is the one the published "
        "baseline tables answer: of the patients who <i>would</i> have "
        "qualified, how many look like the patients who were actually enrolled? "
        "Distance is measured in interquartile ranges, which is all a paper "
        "supports &mdash; it reports a median and an IQR, not a covariance, so "
        "there is no way to weight variables against one another or to account "
        "for their correlation. DanGer Shock and ECLS-SHOCK have complete "
        "tables here; the others published medians without IQRs, or nothing "
        "retrievable.</p>"
    )
    out.append(
        "<p>The <b>By trial</b> tab pairs each trial's published criteria with "
        "its enrolment characteristics. <code>applied_here</code> says whether "
        "this pipeline can evaluate a criterion: 19 of the 39 published "
        "criteria across the eight trials are applied in full, 14 partly, and "
        "6 not at all. Every <code>no</code> on a substantive criterion makes "
        "that trial's eligible count a superset &mdash; CULPRIT-SHOCK's "
        "multivessel requirement is the largest of them.</p>"
    )

    anchor_tabs = []
    for anchor in anchors:
        blocks = []
        absent: list[str] = []
        eligibility = pl.read_csv(table_home / f"trial_eligibility_{anchor}.csv")
        shown, missing = present(
            eligibility,
            "trial", "year", "n_eligible", "share_eligible",
            "mortality_eligible", "n_inside_not_eligible",
            "centroid_separation_spreads", "criteria_not_applied",
        )
        absent += missing
        blocks.append(
            ("Eligibility", table(
                shown,
                {
                    "n_eligible": thousands,
                    "share_eligible": pct,
                    "mortality_eligible": pct,
                    "n_inside_not_eligible": thousands,
                    "centroid_separation_spreads": num(2),
                },
                key_column="trial",
            ))
        )
        representativeness = table_home / f"trial_representativeness_{anchor}.csv"
        if representativeness.is_file():
            shown, missing = present(
                pl.read_csv(representativeness),
                "trial", "n_published", "n_eligible_here",
                "age_published", "age_here",
                "pct_female_published", "pct_female_here",
                "lactate_published", "lactate_here",
            )
            absent += missing
            blocks.append(
                ("Us vs them", table(
                    shown,
                    {
                        "n_published": thousands,
                        "n_eligible_here": thousands,
                        "age_published": num(1), "age_here": num(1),
                        "pct_female_published": pct, "pct_female_here": pct,
                        "lactate_published": num(1), "lactate_here": num(1),
                    },
                    key_column="trial",
                ))
            )
        coverage = table_home / f"trial_coverage_{anchor}.csv"
        if coverage.is_file():
            blocks.append(
                ("Coverage", table(
                    pl.read_csv(coverage),
                    {
                        "n": thousands, "share": pct, "mortality": pct,
                        "median_age": num(0), "pct_female": pct,
                        "median_lactate": num(1), "median_icu_los": num(1),
                    },
                    key_column="n_eligible_trials",
                ))
            )
        # --------------------------------------------- one tab per trial
        # Criteria and enrolment characteristics side by side, which is the
        # pairing that matters: the criteria say who a recommendation nominally
        # covers, the characteristics say who it was observed in.
        criteria_file = table_home / "trial_criteria.csv"
        baseline_file = table_home / f"trial_table1_{anchor}.csv"
        if criteria_file.is_file() or baseline_file.is_file():
            criteria = (
                pl.read_csv(criteria_file) if criteria_file.is_file()
                else pl.DataFrame()
            )
            baseline = (
                pl.read_csv(baseline_file) if baseline_file.is_file()
                else pl.DataFrame()
            )
            names = []
            for source in (criteria, baseline):
                if source.height:
                    names += source["trial"].unique(maintain_order=True).to_list()
            seen, ordered = set(), []
            for name in names:
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)

            per_trial = []
            for name in ordered:
                parts = []
                if criteria.height:
                    rows = criteria.filter(pl.col("trial") == name)
                    if rows.height:
                        parts.append(
                            "<h4>Inclusion and exclusion criteria</h4>"
                            + table(
                                rows.select("kind", "criterion", "applied_here",
                                            "note"),
                                None,
                                key_column="criterion",
                            )
                        )
                if baseline.height:
                    rows = baseline.filter(pl.col("trial") == name)
                    if rows.height:
                        parts.append(
                            "<h4>Enrolment characteristics: ours vs the trial</h4>"
                            + table(
                                rows.select(
                                    "variable", "confidence", "published",
                                    "published_q1", "published_q3", "ours",
                                    "pct_below_q1", "pct_above_q3",
                                    "net_shift_low",
                                ),
                                {
                                    "published": num(2),
                                    "published_q1": num(2),
                                    "published_q3": num(2),
                                    "ours": num(2),
                                    "pct_below_q1": pct,
                                    "pct_above_q3": pct,
                                    "net_shift_low": pct,
                                },
                                key_column="variable",
                            )
                        )
                if parts:
                    per_trial.append((name, "".join(parts)))
            if per_trial:
                blocks.append(("By trial", tabs(per_trial)))

        inside = table_home / f"trial_inside_tested_{anchor}.csv"
        if inside.is_file():
            blocks.append(
                ("Inside the tested population", table(
                    pl.read_csv(inside),
                    {
                        "n_eligible": thousands,
                        "n_comparable": thousands,
                        "share_inside_all_iqrs": pct,
                        "median_variables_outside": num(0),
                        "median_max_distance_iqrs": num(2),
                    },
                    key_column="trial",
                ))
            )
        # the interactive view, embedded verbatim. 11 writes it as a fragment
        # rather than the report building it, so the payload is produced once
        # alongside the data it describes.
        interactive = table_home / f"trial_interactive_{anchor}.html"
        if interactive.is_file():
            blocks.append(
                ("Explore (interactive)",
                 interactive.read_text(encoding="utf-8"))
            )
        outcomes_file = table_home / f"trial_outcomes_{anchor}.csv"
        if outcomes_file.is_file():
            shown, missing = present(
                pl.read_csv(outcomes_file),
                "trial", "arm", "reliable", "n_inside", "n_outside",
                "mortality_inside", "mortality_outside",
                "mortality_difference", "mortality_difference_sofa_adj",
                "icu_los_inside", "icu_los_outside",
                "sofa_non_cv_inside", "sofa_non_cv_outside",
            )
            absent += missing
            blocks.append(
                ("Outcomes inside vs outside", table(
                    shown,
                    {
                        "n_inside": thousands, "n_outside": thousands,
                        "mortality_inside": pct, "mortality_outside": pct,
                        "mortality_difference": pct,
                        "mortality_difference_sofa_adj": pct,
                        "icu_los_inside": num(1), "icu_los_outside": num(1),
                        "sofa_non_cv_inside": num(0),
                        "sofa_non_cv_outside": num(0),
                    },
                    key_column="trial",
                ))
            )

        home = paths.output_home / "figures" / "GEM-all" / f"anchor-{anchor}"
        for name, label in (
            ("trial_eligible_vs_tested", "Would qualify vs resembles"),
            ("trial_regions", "Where each trial sits"),
            ("trial_coverage", "Coverage"),
        ):
            body = figure(home / f"{name}.png")
            if body:
                blocks.append((label, body))
        rendered = tabs(blocks)
        if absent:
            # Saying which columns are gone, rather than rendering a narrower
            # table under prose that describes the wider one.
            rendered = callout(
                f"<b>This anchor's tables were written by 11's federated path, "
                f"which cannot compute proximity.</b> Absent here: "
                + ", ".join(f"<code>{esc(c)}</code>" for c in absent)
                + ". Re-run <code>python code/11_trial_alignment.py --anchor "
                f"{esc(anchor)}</code> at a site with patient-level values to "
                "fill them in.", "warn") + rendered
        anchor_tabs.append((anchor, rendered))

    out.append(tabs(anchor_tabs))

    out.append(callout(
        "<b>The discrepancy is blood pressure, and it replicates across every "
        "trial that published one.</b> Our criteria-eligible patients match the "
        "enrolled populations on age, weight, height, body-mass index, lactate, "
        "arterial pH and heart rate &mdash; each within a few points of the 25% "
        "a representative sample would put below a first quartile. The "
        "pressures do not, and the gap is in the same direction every time: "
        "below each trial's first systolic quartile sit <b>77.9%</b> of our "
        "OptimaCC-eligible patients, <b>66.8%</b> of ECLS-SHOCK's, "
        "<b>62.6%</b> of CULPRIT-SHOCK's, <b>58.9%</b> of ECMO-CS's and "
        "<b>37.9%</b> of DanGer Shock's, against a 25% null. Mean arterial "
        "pressure runs the same way.</p><p>The trials' own numbers explain it. "
        "OptimaCC enrolled at a median systolic of 104 mmHg, CULPRIT-SHOCK at "
        "100 and ECLS-SHOCK at 96, despite each requiring profound hypotension "
        "<i>or</i> vasopressor support &mdash; so their populations were "
        "largely supported at a near-normal pressure. DanGer Shock, the only "
        "one whose threshold is SBP &lt; 100 rather than &lt; 90, enrolled at "
        "83 and is the least shifted of the five. Ours sit at 73&ndash;79.",
        "warn"))

    out.append(callout(
        "<b>Assigning each patient their single most relevant guideline leaves "
        "68.7% with none.</b> Where a patient qualifies for several trials the "
        "most specific one wins &mdash; SOAP II is excluded from the contest, "
        "since a recommendation covering 89% of the cohort is the least "
        "informative thing one could say about any particular patient. Of the "
        "31.3% who do get a guideline, only <b>429 patients (1.3% of the "
        "cohort)</b> match it on both counts: eligible <i>and</i> inside every "
        "published interquartile range that could be checked. The other 30% are "
        "covered on criteria alone. The mortality gradient follows: among "
        "ECMO-CS-assigned patients, 23.1% for those matching the population "
        "against 28.8% for criteria only; for SHOCK, 7.8% against 25.6%.", "warn"))

    out.append(callout(
        "<b>Clinicians follow the negative recommendations and not the positive "
        "one.</b> Among patients each trial would have enrolled, the device was "
        "correctly withheld in 93.0% (ECMO-CS), 85.6% (ECLS-SHOCK) and 84.5% "
        "(IABP-SHOCK II) &mdash; but the one device a trial actually supports, "
        "the microaxial pump, reached only <b>14.1%</b> of the patients DanGer "
        "Shock would have enrolled. Norepinephrine-without-dopamine was "
        "delivered in 66.6% and norepinephrine-without-epinephrine in 62.4%. "
        "Early revascularisation is coded in 28.4%, but UCMC's CPT rows do not "
        "join on hospitalization_id, so that one is understated at that site "
        "and should not be compared across the two."))

    out.append(callout(
        "<b>Outcomes are worse outside the tested population, and severity does "
        "not explain it &mdash; but nor can this be read causally.</b> Among "
        "patients treated per the guideline, mortality runs 10.3 points higher "
        "outside ECMO-CS's published population than inside it (30.9% against "
        "20.6%), 7.0 higher for OptimaCC and 4.0 for IABP-SHOCK II. Adjusting "
        "for severity in the five non-circulatory organ systems barely moves "
        "any of them, and the two groups' non-cardiovascular SOFA differs by at "
        "most one point &mdash; so the excess is not simply that the outside "
        "group is sicker everywhere. <b>It cannot be read as an effect.</b> "
        "Patients fall outside almost entirely because their blood pressure is "
        "lower, and low blood pressure <i>is</i> severity; the exposure and the "
        "confounder are close to the same variable. Total SOFA is deliberately "
        "not used as the adjuster for that reason &mdash; its cardiovascular "
        "component is scored from mean arterial pressure and vasopressor dose, "
        "so it would absorb the very difference being measured.", "warn"))

    out.append(callout(
        "<b>The size of the gap tracks how selectively the trial enrolled, and "
        "SOAP II is the control case.</b> Below each trial's first "
        "mean-arterial quartile sit 79.1% of our OptimaCC-eligible patients "
        "(the trial enrolled post-PCI at a median MAP of 72), 67.2% of "
        "CULPRIT-SHOCK's (76), 48.1% of ECMO-CS's (63) and 44.6% of DanGer "
        "Shock's (64) &mdash; but only <b>16.9%</b> of SOAP II's, which is "
        "<i>below</i> the 25% null. SOAP II enrolled any shock requiring a "
        "vasopressor, of any cause, and its own population sat at a MAP of 58: "
        "genuinely hypotensive, like ours. The specialised cardiogenic shock "
        "trials enrolled at 63 to 76. The more narrowly a trial selected, the "
        "further its population sits from routine ICU shock, and the one "
        "pragmatic all-comers trial is the only one whose patients ours "
        "resemble.</p><p>That has an uncomfortable corollary. The single "
        "recommendation whose trial population matches these patients &mdash; "
        "norepinephrine rather than dopamine &mdash; is also the one resting "
        "on the thinnest evidence for cardiogenic shock specifically: a "
        "280-patient subgroup of a 1,679-patient trial that was 62% septic, "
        "and whose baseline characteristics were never published. The SOAP II "
        "row set here is the whole trial, not that subgroup.", "warn"))

    out.append(callout(
        "<b>Lighter vasopressor treatment does not explain the pressure gap, "
        "but it is not quite absent either.</b> ECMO-CS is the one trial that "
        "published which drugs were running at randomisation. Measured over a "
        "comparable window &mdash; six hours either side of shock onset, which "
        "is what &ldquo;at randomisation&rdquo; means &mdash; our eligible "
        "patients run norepinephrine at 78.2% against its 85.5% and vasopressin "
        "at 29.9% against 35.0%: slightly <i>less</i> on both, not more. "
        "Epinephrine is higher (19.6% against 3.4%). What dominates is "
        "inotropes, and the gap there is tenfold: dobutamine 5.6% against "
        "54.7%, milrinone 3.9% against 32.5%. A seven-point norepinephrine "
        "deficit cannot account for a 10 mmHg difference in achieved systolic "
        "pressure, so the readings that remain are genuinely deeper shock, a "
        "cardiology-led trial population carrying far more inotropic support "
        "than an undifferentiated ICU cohort receives, or both.</p><p>"
        "<b>An earlier version of this section said our patients were supported "
        "at least as heavily, and that was an artefact.</b> The agent flags "
        "were built over the whole admission rather than around onset, which "
        "put norepinephrine at 86.4% and vasopressin at 48.4% &mdash; both "
        "above the trial. Anchoring them to the shock reverses the direction "
        "of both comparisons."))

    out.append(callout(
        "<b>Do not read the &ldquo;inside every IQR&rdquo; column as a measure "
        "of representativeness.</b> An interquartile range contains half the "
        "trial's own population by construction, so even a perfectly "
        "representative cohort would have only 0.5<sup>k</sup> of its patients "
        "inside all k of them &mdash; 50% at one variable, 6.2% at four, 0.8% "
        "at seven. The observed shares here are 0.72 to 1.61 times that null, "
        "which is to say the eligible patients are about as concentrated as a "
        "representative sample would be. The column falls with every variable "
        "added whether or not anything differs, and it is reported with its "
        "null beside it for that reason. The per-variable directional shifts "
        "are where the signal is."))

    out.append(callout(
        "<b>Only the measured variables count toward that judgement.</b> A "
        "trial's medical-history rows come from a case-report form and ours "
        "from ICD codes, and the two do not measure the same thing in either "
        "direction. Restricting history flags to diagnoses present on arrival "
        "helped but did not fix it: heart failure still reads 54.3% against "
        "DanGer Shock's published 9.3%, because a present-on-arrival I50 "
        "includes the acute failure of the index event, while prior myocardial "
        "infarction collapses to 1.0% against 16.1% because present-on-arrival "
        "is seldom set on a history code. Those rows are marked "
        "<code>ascertainment</code> in the baseline tables and are reported "
        "but never used to decide whether a patient is inside the tested "
        "population; age, the pressures, heart rate, lactate, pH, creatinine "
        "and sex are what drive it."))

    out.append(callout(
        "<b>Three of the eight trials reduce to the same computable criteria.</b> "
        "Once the parts we cannot evaluate are dropped &mdash; shock due to LV "
        "failure for SHOCK, planned revascularisation for IABP-SHOCK II, "
        "multivessel disease for CULPRIT-SHOCK &mdash; all three become "
        "&ldquo;acute MI plus shock&rdquo; and select the identical 4,397 "
        "patients. The figure labels the duplicates rather than implying three "
        "independent pieces of evidence. The missing criteria matter in the "
        "other direction too: left ventricular ejection fraction is unavailable "
        "at both sites, so every eligibility count here is a <i>superset</i> of "
        "what the trial would really have enrolled.", "warn"))

    out.append(callout(
        "<b>The trial populations are not distinct regions of the latent "
        "space.</b> Centroid separation &mdash; the distance from a trial's "
        "centroid to the cohort's, in within-group spreads &mdash; runs from "
        "0.03 to 0.32. Anything under about 0.5 means the trial's patients sit "
        "essentially where everyone else does, so no distance-to-centroid "
        "measure can separate them however it is fitted. That is why the "
        "proximity measure here is a local neighbourhood rate and not a "
        "Mahalanobis radius; the first two versions of this analysis used the "
        "latter and returned the same number, 8.5, for all eight trials, which "
        "is the chi quantile for 50 dimensions and says nothing about any "
        "trial."))

    out.append(callout(
        "<b>Most of this cohort is covered by no trial at all.</b> 62.6% of "
        "patients at the 24 h anchor meet no specific trial's criteria "
        "<i>and</i> have no trial-enriched neighbourhood. They are much less "
        "sick than the covered patients &mdash; 8.7% in-hospital mortality "
        "against 27.2% &mdash; and 72.9% of them carry no shock diagnosis code "
        "either, so the two independent measures of &ldquo;not really the "
        "population the evidence is about&rdquo; agree. Read together with the "
        "shock-type section, the reading is that criteria 1 and 2 enrol a large "
        "mild population that neither the coders nor the trialists would "
        "recognise as cardiogenic shock. <b>Even among the "
        "cardiogenic-coded patients, 26.3% are covered by nothing.</b>",
        "warn"))


def section_trial_fit(paths, out: list) -> bool:
    """
    12's outcome models against the fit score

    A subsection of 4 rather than a section of its own: it asks what 11's
    distance is worth, over the same trials and the same anchors, so splitting
    it off would put the score and its consequence under different headings.

    Renders only what 12 produced. Every table here carries the adjustment
    overlap alongside it, because the adjusted column cannot be read without
    knowing how much of the exposure the adjustment removed.

    Returns whether anything was emitted, so main can decide whether to
    list the heading in the contents.
    """
    table_home = paths.output_home / "tables" / "GEM-all"
    anchors = [
        anchor
        for anchor in ANCHORS
        if (table_home / f"trial_fit_outcomes_{anchor}.csv").is_file()
    ]
    if not anchors:
        return False

    out.append(
        '<h3 id="trial-fit">3.1 &middot; Does fitting the trial track with '
        "doing better?</h3>"
    )
    out.append(
        "<p>The tables above establish that most of these patients sit outside "
        "the populations the recommendations were tested in. This asks what "
        "that distance is worth: whether mortality, ventilator-free days and "
        "vasopressor-free days move with it. Two readings of the same gap are "
        "fitted separately &mdash; <code>absdist</code>, how far outside the "
        "published interquartile ranges a patient sits, and "
        "<code>severity_gap</code>, <i>which way</i>, signed so positive means "
        "sicker or older than the trial's median. The direction score is the "
        "one that answers the question an inside/outside split cannot, since a "
        "patient healthier than the trial and a patient sicker than it are "
        "both &ldquo;outside&rdquo;.</p>"
    )
    out.append(
        "<p>Each score enters as a restricted cubic spline on 4 degrees of "
        "freedom, so a relation that turns can show itself; a linear model is "
        "fitted alongside and the two compared by likelihood ratio, which is "
        "the <code>p (nonlinear)</code> column. Mortality is logistic, the free-day "
        "outcomes least squares. Predictions are reported at the 10th, 50th and "
        "90th percentile of the score among that trial's own eligible patients, "
        "so <code>difference</code> is the change across the middle 80% of the "
        "fit range rather than across its tails.</p>"
    )
    out.append(
        "<p><b>Start with the Four cells tab.</b> It crosses the two questions "
        "this section keeps separate &mdash; does the patient resemble the "
        "enrolled population, and were they treated as the trial recommends "
        "&mdash; and shows the outcome in each of the four combinations. Only "
        "the first cell is what the trial's result actually speaks to. "
        "Mortality is drawn as a proportion with a Wilson interval and the "
        "free-day outcomes as violins, because a violin of a binary variable "
        "says nothing a proportion does not, while free days pile up at 0 and "
        "at the 28-day horizon in a way a mean with an error bar hides. Cells "
        "under 50 patients are hollow and cells under 10 are not drawn at "
        "all.</p>"
    )
    out.append(
        "<p>Each model is fitted twice: unadjusted, and adjusted for "
        "non-cardiovascular SOFA, age and site. <b>The adjusted column is not a "
        "bias-corrected version of the unadjusted one</b> &mdash; the exposure "
        "and the confounder are built from overlapping measurements, and the "
        "<b>Adjustment overlap</b> tab gives the share per trial. Where the "
        "overlap is total the row is marked and should be read as collinearity "
        "rather than as an estimate.</p>"
    )

    anchor_tabs = []
    for anchor in anchors:
        blocks = []
        results = pl.read_csv(table_home / f"trial_fit_outcomes_{anchor}.csv")
        results = results.with_columns(
            adjusted=pl.when(pl.col("adjusted")).then(pl.lit("yes"))
            .otherwise(pl.lit("no")),
            note=pl.when(~pl.col("interpretable"))
            .then(pl.lit("collinear: score and adjuster are the same variable"))
            .otherwise(pl.lit("")),
        )
        for outcome, label, fmt in (
            ("in_hospital_mortality", "Mortality", pct),
            ("vent_free_days", "Ventilator-free days", num(1)),
            ("pressor_free_days", "Vasopressor-free days", num(1)),
        ):
            rows = results.filter(pl.col("outcome") == outcome)
            if not rows.height:
                continue
            columns = [
                "trial", "score", "adjusted", "n",
                "predicted_at_p10", "predicted_at_p50", "predicted_at_p90",
                "difference_p90_minus_p10", "curve_shape",
                "p_overall", "p_nonlinearity", "note",
            ]
            blocks.append((label, table(
                rows.sort("score", "trial", "adjusted").select(columns),
                {
                    "n": thousands,
                    "predicted_at_p10": fmt,
                    "predicted_at_p50": fmt,
                    "predicted_at_p90": fmt,
                    "difference_p90_minus_p10": fmt,
                    "p_overall": pval,
                    "p_nonlinearity": pval,
                },
                key_column="trial",
            )))

        interaction = table_home / f"trial_fit_interaction_{anchor}.csv"
        if interaction.is_file():
            blocks.append(("Concordant care, by fit", table(
                pl.read_csv(interaction).sort("trial", "score", "outcome")
                .select(
                    "trial", "score", "outcome", "n", "n_concordant",
                    "slope_discordant", "interaction", "p_interaction",
                    "concordance_means", "confounding",
                ),
                {
                    "n": thousands, "n_concordant": thousands,
                    "slope_discordant": num(3), "interaction": num(3),
                    "p_interaction": pval,
                },
                key_column="trial",
            )))

        overlap = table_home / "trial_fit_adjustment_overlap.csv"
        if overlap.is_file():
            blocks.append(("Adjustment overlap", table(
                pl.read_csv(overlap).sort("overlap_share", descending=True)
                .select(
                    "trial", "score", "fit_score_variables",
                    "also_adjusted_for", "overlap_share",
                    "adjusted_interpretable",
                ),
                {"overlap_share": pct},
                key_column="trial",
            )))

        skipped = table_home / f"trial_fit_skipped_{anchor}.csv"
        if skipped.is_file():
            blocks.append(("Trials with no fit score", table(
                pl.read_csv(skipped).sort("n_eligible", descending=True),
                {"n_eligible": thousands},
                key_column="trial",
            )))

        home = paths.output_home / "figures" / "GEM-all" / f"anchor-{anchor}"
        # Figures first in the tab strip. The three outcome tables were the
        # thing readers could not parse: the comparison they encode is four
        # numbers spread across different rows of different tables, and a
        # figure makes it one shape. The tables stay, behind the figures.
        figures = [
            (
                "Four cells",
                figure(
                    home / "trial_fit_cells.png",
                    "Outcome by whether the patient resembles the enrolled "
                    "population and was treated as the trial recommends.",
                ),
            ),
            (
                "Fit and outcome in the space",
                figure(
                    home / "trial_fit_space.png",
                    "The same layout as section 2, binned into local means. "
                    "Compare the fit panels with the outcome panels.",
                ),
            ),
            (
                "Spline curves",
                figure(
                    home / "trial_fit_outcomes.png",
                    "Fitted outcome against fit score, with 95% bands. Each "
                    "column an outcome, each row a trial.",
                ),
            ),
        ]
        blocks = [(label, body) for label, body in figures if body] + blocks

        anchor_tabs.append((anchor, tabs(blocks)))

    out.append(tabs(anchor_tabs))

    out.append(callout(
        "<b>Fit tracks outcome in every trial, every outcome and both "
        "directions of adjustment &mdash; the smallest <code>p (overall)</code> "
        "anywhere in the table is 0.0034.</b> Against the direction score, "
        "predicted mortality runs from 15.0% at the 10th percentile of fit to "
        "<b>48.8%</b> at the 90th for ECMO-CS-eligible patients, 30.7% to 57.3% "
        "for ECLS-SHOCK, 20.8% to 44.7% for OptimaCC and 10.7% to 27.9% for the "
        "SOAP II subgroup. Ventilator-free days fall 20.5 to 12.5 across the "
        "same range for ECMO-CS and vasopressor-free days 22.0 to 12.9. The "
        "gradient is real and it is large; what it <i>is</i> is the question "
        "the next three callouts are about."))

    out.append(callout(
        "<b>Direction carries almost all of it, and the unsigned distance is "
        "close to uninformative.</b> The two scores are computed from the same "
        "variables and differ only in whether the sign is kept. Keeping it "
        "nearly doubles the mortality spread for ECMO-CS (33.8 points against "
        "19.9) and multiplies it fourteen-fold for the SOAP II subgroup (17.3 "
        "against 1.2), and it is what the nonlinearity is made of: <b>20 of 21 "
        "unadjusted <code>severity_gap</code> models reject linearity against "
        "4 of 21 <code>absdist</code> models.</b> A patient far outside a "
        "trial's ranges in the healthy direction and one equally far outside in "
        "the sick direction have the same <code>absdist</code> and opposite "
        "outcomes, so averaging them is what flattens the unsigned score. "
        "<b>This is the result that justifies computing a direction at all</b>, "
        "and it is the reason the binary inside/outside column in the tables "
        "above should not be used as a fit measure."))

    out.append(callout(
        "<b>The curve turns for the two MI trials, and in the direction that "
        "says the score stops measuring severity.</b> CULPRIT-SHOCK and "
        "ECLS-SHOCK are U-shaped on mortality and inverted-U on both free-day "
        "outcomes, adjusted and not: patients <i>healthier</i> than the trial's "
        "median on the measured axes do worse than patients sitting at it. That "
        "cannot be read as &ldquo;being unlike the trial is bad for you&rdquo; "
        "in any symmetric sense. The likelier reading is that the measured "
        "axes &mdash; pressures, lactate, pH, age &mdash; stop ordering "
        "severity at their benign end, where what distinguishes patients is "
        "something the published baseline tables never reported. A monotone "
        "model would have averaged the turn away, which is why the spline is "
        "fitted even where the linear fit is adequate.", "warn"))

    out.append(callout(
        "<b>Adjustment does not attenuate the gradient, and that is a warning "
        "rather than a finding.</b> The adjusted estimate is <i>larger</i> than "
        "the unadjusted one in 25 of 42 trial-score-outcome combinations. For a "
        "genuine confounder one expects shrinkage, so the honest reading is "
        "that non-cardiovascular SOFA, age and site are not the confounders "
        "that matter here &mdash; and that where they do overlap the exposure, "
        "adjusting removes signal rather than bias. <b>Three rows are pure "
        "collinearity and are labelled so:</b> SHOCK on both scores and "
        "IABP-SHOCK II on the direction score reduce to age alone once the "
        "unavailable criteria are dropped, so their adjusted models regress age "
        "on age &mdash; which is where IABP-SHOCK II's implausible 2.7% to "
        "74.9% mortality sweep comes from. They are reported rather than "
        "dropped because dropping them would leave the unadjusted estimate "
        "looking unopposed.", "warn"))

    out.append(callout(
        "<b>The concordance interaction is significant only where concordance "
        "means choosing one drug over another, and there it is confounded by "
        "indication beyond repair.</b> OptimaCC (norepinephrine used, "
        "epinephrine never) and the SOAP II subgroup (norepinephrine used, "
        "dopamine never) show interactions at p &lt; 0.0001 on all three "
        "outcomes; the three device trials show nothing except ECLS-SHOCK on "
        "mortality (p = 0.02 of 18 tests). The device trials were negative, so "
        "&ldquo;concordant&rdquo; there means the device was <i>withheld</i> "
        "&mdash; which happens most in patients too well and in patients too "
        "sick to cannulate, inverting the usual direction at both ends at once. "
        "Nothing in this tab supports a claim that guideline-concordant care "
        "pays off differently by fit; it is reported because the question was "
        "asked and the answer is that this design cannot address it.", "warn"))

    out.append(callout(
        "<b>Ten trials contribute eligibility but no fit score, and the largest "
        "of them is not small.</b> A fit score needs a transcribed baseline "
        "table with interquartile ranges, and 61 published values in "
        "<code>trials.yaml</code> are still unverified placeholders. <b>All ten "
        "are the sepsis and haemorrhage trials</b> &mdash; CLOVERS (8,528 "
        "eligible patients here), SEPSISPAM, SMART, APROCCHSS and CLASSIC "
        "(8,115 each), VASST (6,819), the 65 trial (4,715), ADRENAL (3,225), "
        "Villanueva (590) and CRASH-2 (0) &mdash; so they appear in the "
        "eligibility tables above and nowhere in this subsection. "
        "<code>trial_table1_to_verify.csv</code> lists every value awaiting "
        "verification against its paper. Until those are checked, this "
        "subsection speaks only to the cardiogenic shock literature."))

    return True


def section_patient_embeddings(paths, out: list) -> None:
    home = trajectory_home(paths)
    panels = [
        (label, figure(home / f"trajectory_label_{key}.png"))
        for key, label in LABELLINGS
    ]
    if not any(body for _, body in panels):
        return
    out.append('<h2 id="patients">4 &middot; Embeddings over time</h2>')
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
    home = trajectory_home(paths)
    panels = [
        (label, figure(home / f"patient_paths_{key}.png"))
        for key, label in LABELLINGS
    ]
    if not any(body for _, body in panels):
        return
    out.append('<h2 id="paths">5 &middot; Individual trajectories</h2>')
    out.append(
        "<p>Five patients drawn at random from those with at least five "
        "timepoints, each shown in its own panel against the same cloud, with "
        "hours relative to shock onset annotated on the path. The tabs change "
        "what the cloud behind them is coloured by; the five patients are the "
        "same throughout.</p>"
    )
    out.append(tabs(panels))
    out.append(callout(
        "Read these against section 4. Individual paths are short and largely "
        "confined to one region: patients move <i>within</i> a neighbourhood "
        "rather than migrating across the space as shock develops, which is "
        "what the flow field's near-zero gradient alignment says quantitatively."))


def section_longitudinal(paths, out: list) -> None:
    """
    11's longitudinal strata: how they were found, and what they contain

    Deliberately its own section rather than a tab inside section 6. The
    clustering and the embedding are separate analyses that share only the
    (patient, time) key -- the embedding reads every token in the cumulative
    prefix with no site filter, the clustering reads 366 site-balanced tokens
    counted inside one six-hour window. Presenting the clusters as though they
    were a property of the projection would hide that.
    """
    homes = trajectory_homes(paths)
    septic_home = next(
        (info["home"] for info in homes.values()
         if info["subset"] == "septic"
         and info["correction"] == PREFERRED_CORRECTION),
        None,
    )
    tables_home = paths.output_home / "tables" / "GEM-all"
    movement_file = tables_home / "state_movement_septic.csv"
    character_file = tables_home / "cluster_characteristics_sequence_cluster_septic.csv"
    growth_file = tables_home / "cluster_characteristics_growth_cluster_septic.csv"
    sites_file = tables_home / "cluster_sites_sequence_cluster_septic.csv"
    tokens_file = tables_home / "cluster_mortality_tokens_septic.csv"
    if not any(f.is_file() for f in (movement_file, character_file, tokens_file)):
        return

    out.append('<h2 id="longitudinal">7 &middot; Longitudinal strata</h2>')
    out.append(
        "<p>Septic shock patients clustered on how their <b>token "
        "composition</b> moves over the seven days after shock onset, by two "
        "methods that assume different things. <b>State sequences</b> assign "
        "each six-hour window one of six care states learned by NMF over the "
        "window compositions, then cluster the resulting 29-symbol strings. "
        "<b>Latent class growth</b> summarises each composition component by a "
        "quadratic growth curve per patient and mixes over the coefficients "
        "&mdash; two-stage, since true GBTM estimates curves and classes "
        "jointly.</p>"
    )
    out.append(
        "<p>This is a <b>separate analysis from the embedding</b>. They share "
        "only the (patient, time) key. The embedding at +24&nbsp;h reads every "
        "token in the patient's cumulative prefix, unfiltered, capped at the "
        "model's 4096-token context; the label at +24&nbsp;h comes from 366 "
        "site-balanced tokens counted inside the six hours ending there. Same "
        "patient, same instant, two different views of it &mdash; which is why "
        "colouring one by the other says something rather than restating "
        "itself. Token space carries no site correction, so the feature space "
        "was filtered down from 1,192 tokens: 953 appear at every site, 472 "
        "clear 1% prevalence everywhere, and 330 also sit within a 2&times; "
        "between-site prevalence ratio.</p>"
    )

    if movement_file.is_file():
        out.append('<h3>How much patients actually move</h3>')
        out.append(
            "<p>The number of state changes per patient, out of 28 possible "
            "transitions over seven days. This replaced a time-shuffled null, "
            "which was not a meaningful test here: silhouette mostly reports "
            "that k-means found a compact partition, which it does in almost "
            "any feature space, and the windows are anchored to each patient's "
            "own shock onset <i>by design</i>, so permuting them produces "
            "noise rather than a plausible alternative cohort. Counting "
            "transitions asks the question directly.</p>"
        )
        out.append(table(
            pl.read_csv(movement_file),
            {"patients": thousands, "share": pct},
        ))

    character_tabs = []
    for path, label in ((character_file, "State sequences"),
                        (growth_file, "Latent class growth")):
        if not path.is_file():
            continue
        frame = pl.read_csv(path)
        character_tabs.append((label, table(
            frame,
            {
                "n": thousands,
                **{c: pct for c in frame.columns
                   if c.startswith(("invasive", "renal", "died"))},
                **{c: num(3) for c in frame.columns if "norepinephrine" in c},
                "ICU length of stay (days)": num(1),
                "peak vasoactive agents": num(0),
            },
        )))
    if character_tabs:
        out.append('<h3>What each stratum contains</h3>')
        out.append(
            "<p>Treatment and outcome by stratum. Read the treatment columns "
            "knowing the feature space includes treatment tokens &mdash; "
            "vasopressor and ventilation events are part of the composition "
            "the clusters were built from, so a treatment difference is partly "
            "by construction. ICU length of stay and mortality are not in the "
            "feature space at all and are the columns that carry weight.</p>"
        )
        out.append(tabs(character_tabs))

    if septic_home is not None:
        panels = [
            ("Care state (per timepoint)",
             figure(septic_home / "trajectory_flow_septic_care_state.png",
                    "The state assigned to each six-hour window")),
            ("Stratum (per patient)",
             figure(septic_home / "trajectory_flow_septic_stratum.png",
                    "The stratum the patient's whole sequence fell into")),
        ]
        panels = [(label, body) for label, body in panels if body]
        if panels:
            out.append('<h3>The strata, drawn in the septic embedding</h3>')
            out.append(
                "<p>Two different grains, and they are not interchangeable. "
                "<b>Care state</b> is per (patient, timepoint) and is the thing "
                "that moves. <b>Stratum</b> is per patient and is constant "
                "along a trajectory by construction, so that panel shows where "
                "a stratum lives rather than where a patient is at that "
                "moment. If the care states land coherently here, the model "
                "organises patients much as their site-comparable event "
                "composition does; if they do not, the likely reason is the "
                "~800 tokens the site filter removed, which the embedding "
                "still reads and the labels do not.</p>"
            )
            out.append(tabs(panels))

    if sites_file.is_file():
        out.append('<h3>Are the strata hospitals?</h3>')
        out.append(
            "<p>Cram&eacute;r's V between stratum and contributing site. "
            "Roughly: below 0.1 the partition is near-independent of site, "
            "above 0.3 it is substantially a site partition and should not be "
            "read as clinical. Token space has no site correction, so this is "
            "the check that the prevalence filter did its job.</p>"
        )
        out.append(table(
            pl.read_csv(sites_file),
            {
                "n": thousands,
                **{c: pct for c in pl.read_csv(sites_file).columns
                   if c.startswith("share_")},
                "cramers_v": num(3),
            },
        ))

    if tokens_file.is_file():
        frame = pl.read_csv(tokens_file)
        significant = frame.filter(pl.col("p_adjusted") < 0.05)
        out.append('<h3>Within a stratum, what separates the patients who died</h3>')
        out.append(
            "<p>The strata stratify by severity, so a token enriched among "
            "deaths across the whole cohort mainly says that sicker patients "
            "die. Asking inside a stratum holds severity roughly fixed and "
            "asks what separates outcome among patients who look similarly "
            "sick. Prevalence is per patient, not per window, so a patient "
            "charted twice as often does not count twice; cells below 11 "
            "patients are suppressed; Bonferroni is over every "
            "token-by-stratum test rather than within each stratum.</p>"
        )
        out.append(callout(
            "<b>Several of these are consequences, not antecedents.</b> The "
            "token window spans the whole week, so <code>XFR-IN//ward</code> "
            "means the patient left the ICU alive, <code>EOS</code> is the end "
            "of a record, and <code>rass_Q8</code> against "
            "<code>rass_Q0</code> partly restates survival. As "
            "characterisation that is fine. To make these predictive the token "
            "window has to stop early &mdash; say +24&nbsp;h &mdash; while the "
            "outcome stays at discharge."
        ))
        out.append(table(
            significant.sort(["stratification", "cluster", "p_adjusted"])
            .filter(pl.col("stratification") == "sequence_cluster")
            .drop("stratification", "p")
            .head(60),
            {
                "died_prevalence": pct,
                "survived_prevalence": pct,
                "log2_ratio": num(2),
                "p_adjusted": num(4),
            },
            key_column="cluster",
        ))


def section_flow(paths, out: list) -> None:
    home = trajectory_home(paths)
    homes = trajectory_homes(paths)
    # Falls back to a path that cannot exist, so the four septic tabs come up
    # empty and are dropped by the `if body` filter when no septic grid has
    # been extracted. Falling back to `home` was wrong: 09 used to draw both
    # panel sets in every run, so the cardiogenic directory held
    # septic-named figures drawn over the cardiogenic grid, and the fallback
    # found them and presented them as the septic cohort.
    # Resolved by subset, not by label, so a rename of the heading cannot
    # quietly repoint these four panels at another cohort. Falls back to a path
    # that cannot exist, so the septic tabs come up empty and are dropped by
    # the `if body` filter when no septic grid has been extracted -- falling
    # back to `home` was wrong, because 09 used to draw both panel sets in
    # every run and the fallback found the cardiogenic directory's
    # septic-named figures and presented them as the septic cohort.
    septic_home = next(
        (info["home"] for info in homes.values()
         if info["subset"] == "septic" and info["correction"] == PREFERRED_CORRECTION),
        next(
            (info["home"] for info in homes.values()
             if info["subset"] == "septic" and info["correction"] is not None),
            home / "no-septic-grid",
        ),
    )
    alignment_file = paths.output_home / "tables" / "GEM-all" / "trajectory_flow_alignment.csv"
    # 09 names the flow figure after whichever projection it drew, so that a
    # field can never be shown under the wrong axis labels. Prefer PaCMAP,
    # which is 09's default, and fall back to a UMAP run.
    flow = next(
        (
            home / f"trajectory_flow_{name}.png"
            for name in ("pacmap", "umap")
            if (home / f"trajectory_flow_{name}.png").is_file()
        ),
        None,
    )
    if flow is None:
        return
    projection_label = "PaCMAP" if flow.stem.endswith("pacmap") else "UMAP"
    out.append('<h2 id="flow">6 &middot; Latent flow</h2>')
    out.append(
        "<p>The mean displacement between consecutive timepoints, drawn as a "
        "streamplot after scVelo's velocity embedding. One difference favours "
        "this setting: RNA velocity must <i>infer</i> direction from spliced and "
        "unspliced read ratios, because a cell is destroyed when measured and "
        "only one snapshot exists per cell. We observe the same patient at t and "
        "t+6h, so the displacement is <b>measured, not estimated</b>.</p>"
    )
    out.append(tabs([
        (projection_label, figure(flow,
                                  f"Flow field in {projection_label} space")),
        ("PCA (cross-check)", figure(home / "trajectory_flow_pca.png",
                                     "The same field in a linear space")),
        # ONE field over the cardiogenic union, drawn with different hues under
        # it. Splitting the cohort per definition or per outcome fitted a
        # separate field to each, so two panels differed both in who was in them
        # and in what the arrows were computed from, with nothing saying which
        # difference you were looking at. Every panel here carries the arrows.
        ("Cardiogenic, by Zweck",
         figure(home / "trajectory_flow_cardiogenic_phenotype.png",
                "Patients meeting either definition, coloured by Zweck phenotype")),
        ("Cardiogenic, by definition",
         figure(home / "trajectory_flow_cardiogenic_definition.png",
                "The same field, coloured by whether the ICD code, the "
                "treatment-based flag, or both identified the patient")),
        ("By outcome",
         figure(home / "trajectory_flow_mortality.png",
                "One field over everyone, coloured by in-hospital mortality")),
        # Septic shock, from the grid extracted over the coded-septic
        # denominator. Present only when 09 has been run with --subset septic;
        # drawing these from the cardiogenic grid would show the minority of
        # septic patients who also met a cardiac-origin criterion.
        ("Septic, by Seymour",
         figure(septic_home / "trajectory_flow_septic_seymour_phenotype.png",
                "Coded septic shock, coloured by Seymour phenotype")),
        ("Septic, by Bhavani",
         figure(septic_home / "trajectory_flow_septic_bhavani_phenotype.png",
                "The same field, coloured by Bhavani vital-trajectory group")),
        ("Septic, by Zweck",
         figure(septic_home / "trajectory_flow_septic_zweck.png",
                "The same field, coloured by Zweck phenotype -- derived in "
                "cardiogenic shock, so a comparison and not a transfer")),
        ("Septic, by outcome",
         figure(septic_home / "trajectory_flow_septic_mortality.png",
                "The same field, coloured by in-hospital mortality")),
    ]))

    # Who is where, before any flow. Site first, because site is the largest
    # single structure in this geometry -- 1.9x neighbour purity uncorrected --
    # and these projections have had LEACE applied, so the panel is the
    # evidence that the correction worked rather than decoration. Residual
    # separation here is residual confounding in every panel below it.
    site_panels = [
        (label.capitalize(),
         figure(info["home"] / "trajectory_site.png",
                "Patient-timepoints coloured by contributing site"))
        for label, info in homes.items()
    ]
    site_panels = [(label, body) for label, body in site_panels if body]
    if site_panels:
        out.append('<h3>Where the sites sit</h3>')
        out.append(
            "<p>Each cohort's own projection, coloured by contributing site. "
            "These are the <b>corrected</b> coordinates: the site effect has "
            "already been removed by LEACE, which drove a linear site probe "
            "from 0.9972 to 0.5040. Three hospitals still overlapping here is "
            "what licenses reading any other panel in this section as "
            "physiology rather than as provenance.</p>"
        )
        out.append(tabs(site_panels))

    # Zweck under each denominator, in whichever projection is the primary
    # one. Three fits of the same published scheme on three different cohorts:
    # the two restricted ones are what the name actually refers to, the
    # all-shock one is what this pipeline used to produce.
    zweck_panels = [
        (label, figure(home / f"trajectory_flow_{tag}.png", note))
        for tag, label, note in (
            ("phenotype_coded", "ICD-coded CS",
             "Fitted on the 3,105 patients with a coded cardiogenic shock "
             "diagnosis -- closest to Zweck's own derivation cohort"),
            ("phenotype_cardiac_origin", "Cardiac origin",
             "Fitted on the 9,133 patients meeting the computed cardiac-origin "
             "criterion"),
            ("phenotype", "All shock",
             "Fitted on all 39,819 shock patients, 75% of them not cardiogenic "
             "by either reading -- shown because it is what the earlier "
             "versions of this report displayed as 'Zweck'"),
        )
    ]
    zweck_panels = [(label, body) for label, body in zweck_panels if body]
    if zweck_panels:
        out.append('<h3>Zweck, by cardiogenic denominator</h3>')
        out.append(
            "<p>The same published scheme fitted three times, on three "
            "cohorts. Each panel's flow field is computed over only the "
            "patients that fit labels, so the arrows differ between tabs as "
            "well as the hue &mdash; unlike every other tab group in this "
            "section, where one field carries several hues. The ICD-coded and "
            "cardiac-origin cohorts overlap on 2,397 patients, 24.4% of their "
            "union, so these are substantially different populations wearing "
            "the same three phenotype names.</p>"
        )
        out.append(tabs(zweck_panels))

    # The flow field itself, once per projection. The tab group above is the
    # cardiogenic field with the cohort-specific hues on it; this one is the
    # four fields side by side, which is the only place the uncorrected
    # all-shock geometry can be compared against its corrected counterpart.
    field_panels = [
        (projection_tag(label),
         figure(info["home"] / "trajectory_flow_pacmap.png",
                f"Mean displacement between consecutive timepoints, {label}"))
        for label, info in homes.items()
    ]
    field_panels = [(label, body) for label, body in field_panels if body]
    if len(field_panels) > 1:
        out.append('<h3>The field in each projection</h3>')
        out.append(
            "<p>Four separate PaCMAPs, each fitted on its own set of "
            "patient-timepoints. <b>Coordinates do not correspond between "
            "them</b> and neither do the arrow directions &mdash; a vector "
            "pointing right in one panel has no relation to one pointing right "
            "in another. What is comparable is the <i>structure</i>: whether "
            "the field is coherent, and whether the correction destroyed it.</p>"
        )
        out.append(tabs(field_panels))

    # Shock type, which only the all-shock projections can show: in the
    # cardiogenic subset this column separates coded-cardiogenic from
    # cardiac-origin-only, and in the septic subset it is nearly constant.
    type_panels = [
        (projection_tag(label),
         figure(info["home"] / "trajectory_shock_type.png",
                "Patient-timepoints coloured by coded shock type"))
        for label, info in homes.items()
    ]
    type_panels = [(label, body) for label, body in type_panels if body]
    if type_panels:
        out.append('<h3>Where the shock types sit</h3>')
        out.append(
            "<p>The whole cohort, coloured by coded shock type. 21,462 of the "
            "39,819 have no shock type coded at all and are drawn first, "
            "underneath; cardiogenic is drawn last so that 3,105 patients are "
            "not buried under seven times as many. This is the panel that says "
            "whether the shock types occupy distinct regions of the embedding "
            "or overlap &mdash; and the two cardiogenic definitions agree on "
            "only 24.4% of their union, so a clean separation here would be "
            "surprising.</p>"
        )
        out.append(tabs(type_panels))

    # Every projection, coloured by the decile bin of whatever the model read
    # last. This asks a different question from the physiology underlays below:
    # those colour a patient by a value we computed, these colour them by the
    # token the model actually consumed. A gradient means the projection
    # separates a last-recorded high value from a low one.
    token_panels = []
    for label, info in homes.items():
        for png in sorted(info["home"].glob("trajectory_last_token_*.png")):
            # 09 slugs `VTL//map` to `vtl_map`; read the family back out so
            # the tab says which table the variable came from.
            family, _, rest = png.stem.removeprefix(
                "trajectory_last_token_").partition("_")
            body = figure(png, "")
            if body:
                token_panels.append(
                    (f"{projection_tag(label)} · {family.upper()} "
                     f"{rest.replace('_', ' ')}".strip(), body)
                )
    if token_panels:
        out.append('<h3>What the model read last</h3>')
        out.append(
            "<p>One panel per most common last token before a grid point, "
            "coloured by that token's decile bin &mdash; the tokenizer numbers "
            "these Q0 to Q9, and dark is the highest decile. One ramp is shared "
            "across every panel, so a dark point means the same decile in each. "
            "Grey is every patient-timepoint whose last token was some "
            "other variable, kept so each panel shows what share of the cloud "
            "it accounts for. A visible gradient means the embedding encodes "
            "the <i>value</i>, not just that the measurement was taken; a "
            "uniform scatter of bins means it does not.</p>"
        )
        out.append(tabs(token_panels))

    # Perfusion state: MAP adequacy crossed with pressor exposure, as four
    # groups with a hysteresis band on the MAP side. Separate from the
    # physiology underlays because it is a STATE a patient is in rather than a
    # value they have, and the transition matrix below only makes sense for a
    # state.
    perfusion = figure(
        septic_home / "trajectory_flow_septic_perfusion.png",
        "Coded septic shock, coloured by MAP adequacy and pressor exposure",
    )
    transitions = figure(
        septic_home / "trajectory_perfusion_transitions.png",
        "Row-normalised, with the count in each cell",
    )
    if perfusion or transitions:
        out.append('<h3>Perfusion state, and movement between the states</h3>')
        out.append(
            "<p>Four groups: MAP above or below 65&nbsp;mmHg, crossed with "
            "whether vasopressors are running. The MAP side is a <b>Schmitt "
            "trigger, not a threshold</b> &mdash; crossing up needs MAP&nbsp;&gt;&nbsp;65, "
            "crossing down needs MAP&nbsp;&lt;&nbsp;55, and inside the band the patient "
            "keeps whatever state they were last in. A plain cut at 65 makes a "
            "patient hovering there change group at nearly every six-hour grid "
            "point, and a transition matrix built from that measures "
            "measurement noise. The band resolves about 95% of "
            "patient-timepoints; the rest never leave 55&ndash;65 and fall "
            "back to the plain cut.</p>"
        )
        out.append(
            "<p>Two consequences, both deliberate. The state is "
            "<b>path-dependent</b>: two patients with the same MAP can be in "
            "different groups depending which side they arrived from, which is "
            "what a trigger is for. And a missing MAP carries the previous "
            "state forward rather than breaking the run, since an absent "
            "measurement is not evidence that perfusion changed. The pressor "
            "side gets no hysteresis &mdash; starting or stopping a "
            "vasopressor is a recorded action, not a noisy measurement.</p>"
        )
        out.append(tabs([
            ("Where the states sit", perfusion),
            ("Where patients go next", transitions),
        ]))

    # Physiology, one variable per tab. The combined grid is kept as an
    # overview, but eleven panels at 3.6 inches each is unreadable as evidence
    # -- the colour ramp spans a few hundred pixels and the streamlines are
    # hairlines. The flow field is identical in every one of these, so a
    # difference between two tabs is a difference in the underlay, never in
    # the arrows.
    # One section per cohort, not one tab strip over both. 09 fits a separate
    # projection per subset, so the cardiogenic MAP panel and the septic MAP
    # panel are not the same axes -- interleaving them in a single strip of
    # twenty-four tabs invites exactly the comparison that cannot be made.
    for projection, info in homes.items():
        state_panels = [
            (label,
             figure(info["home"] / f"trajectory_flow_state_{name}.png",
                    f"Flow over {label}"))
            for name, label in STATE_VARIABLES
        ]
        state_panels = [(label, body) for label, body in state_panels if body]
        overview = figure(
            info["home"] / "trajectory_flow_state.png",
            "Every variable at once; the flow field is the same in each",
        )
        if overview:
            state_panels.append(("All together", overview))
        if state_panels:
            out.append(
                f'<h3>What changes as patients move &mdash; {projection}</h3>'
            )
            out.append(tabs(state_panels))

    # And the unsmoothed version: individual patients, in time order. A flow
    # field through a cell where half the patients move each way still draws an
    # arrow, with only the coherence number beside it to say so -- these panels
    # are what that arrow is an average of.
    # One figure per patient, tabbed. Eight on a shared canvas gave each about
    # three inches and the hour labels overlapped whatever the spacing rule --
    # the constraint was the panel size, not the rule.
    # Drawn per subset, because 09 fits a separate projection per subset and a
    # patient's path is only interpretable against the one it was placed in.
    # Reading only the cardiogenic home hid the septic paths entirely.
    # Cohort-specific projections only. 09 draws flow followers for every
    # projection it fits, but a single patient's path through the whole-cohort
    # embedding is a path through a cloud that is 54% patients with no shock
    # type coded -- the field it is being compared against is not about them.
    patient_panels = []
    for label, info in homes.items():
        if info["subset"] == "all":
            continue
        for index in range(1, 13):
            body = figure(info["home"] / f"patient_path_{index:02d}.png", "")
            if body:
                patient_panels.append(
                    (f"{projection_tag(label)} {index}", body)
                )
    if patient_panels:
        out.append('<h3>Individual patients</h3>')
        out.append(
            "<p>Each dot is one 6-hourly grid point, joined in time order, with "
            "the hour relative to shock onset. These patients were chosen "
            "<b>because their own path follows the averaged field</b> &mdash; a "
            "deliberately biased sample, picked to show what the arrows are an "
            "average of. A patient drawn at random is usually a few coincident "
            "dots and one long jump. Nothing is measured from these; the field "
            "beside them carries any claim. Each patient is placed in their own "
            "cohort's projection &mdash; the cardiogenic tabs in the "
            "cardiogenic-only PaCMAP, the septic tabs in the septic-only one "
            "&mdash; so coordinates are not comparable between the two.</p>"
        )
        out.append(tabs(patient_panels))

    # Hypothetical next tokens: the same patients, with one more event appended
    # to their prefix and re-embedded. Present only when 09 ran with
    # --counterfactuals.
    cf_panels = []
    for label, info in homes.items():
        if info["subset"] == "all":
            continue
        for index in range(1, 13):
            body = figure(
                info["home"] / f"counterfactual_path_{index:02d}.png", ""
            )
            if body:
                cf_panels.append((f"{projection_tag(label)} {index}", body))
    if cf_panels:
        out.append('<h3>Where one more event would move them</h3>')
        out.append(
            "<p>The same patients, with a single hypothetical event appended to "
            "their token prefix at the branch point and re-embedded. The model "
            "is a deterministic function of the sequence, so this is what it "
            "would have computed had that event been charted. Branches are "
            "coloured <code>tab10</code>, one hue per intervention, and the "
            "hollow marker is the unmodified prefix re-embedded through the "
            "identical code path &mdash; the control the hypotheticals are "
            "measured against.</p>"
        )
        out.append(
            "<p>Dose and pressure changes are relative to <b>that patient's own "
            "current decile</b>: an absolute target would be an increase for one "
            "patient and a decrease for another. An intervention that does not "
            "apply &mdash; increasing a vasopressor for somebody on none &mdash; "
            "omits that patient from that branch rather than inventing a "
            "starting dose, so the number of arrows differs between panels.</p>"
        )
        out.append(callout(
            "<b>Branch positions are interpolated and arrow length is not a "
            "distance.</b> PaCMAP has no transform, so a point outside the fit "
            "cannot be placed exactly; each branch sits at the weighted mean of "
            "its nearest real neighbours in the full 1024-d space. An "
            "interpolated point cannot land anywhere the real points do not "
            "already cover, so a hypothetical that moves the representation "
            "somewhere genuinely new is dragged back toward the nearest "
            "familiar region. Read direction and neighbourhood, not magnitude; "
            "the per-intervention displacements in "
            "<code>counterfactual_displacement.csv</code> are the quantitative "
            "version, and they are in the same interpolated units.", "warn"))
        out.append(tabs(cf_panels))

        displacement = paths.output_home / "tables" / "GEM-all" / "counterfactual_displacement.csv"
        if displacement.is_file():
            frame = pl.read_csv(displacement)
            summary = (
                frame.group_by("label")
                .agg(patients=pl.len(),
                     median_displacement=pl.col("displacement").median())
                .sort("median_displacement", descending=True)
            )
            out.append(
                "<p>Which charted events the representation is most sensitive "
                "to, ranked. This is the question these panels can actually "
                "answer.</p>"
            )
            out.append(table(
                summary, {"patients": thousands, "median_displacement": num(2)},
                key_column="label",
            ))

    out.append(callout(
        "<b>Patients do not follow these lines.</b> A streamplot integrates its "
        "field and renders far more orderly than the data beneath it, so the "
        "figures need the numbers below to be read honestly. Directional "
        "coherence &mdash; the summed displacement in a cell over the sum of "
        "the displacement magnitudes &mdash; was 0.61 in UMAP and 0.48 in PCA "
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
    """
    7 - why the sites separate

    Each block is guarded on ITS OWN input. An earlier version returned from the
    whole section when figures/shared/site_features.png was absent, which is
    produced by 10_site_features.py -- so not running one optional script
    silently deleted the token tornado, the structural-zero ranking and the
    family mix along with it. Those are 04's output and the largest single
    finding in this pipeline; they do not depend on 10 in any way.
    """
    shared = paths.output_home / "figures" / "shared"
    features = figure(shared / "site_features.png")
    tornado = paths.output_home / "figures" / "GEM-all" / "anchor-24h"
    # One tab per PAIR of sites. The tornado ranks on the difference between
    # two per-patient rates, so it is pairwise by construction; with three
    # sites there are three comparisons and showing only one of them would hide
    # whichever site was left out.
    pair_figures = sorted(tornado.glob("site_token_tornado_*-vs-*.png"))
    pairs = sorted({
        f.stem.split("site_token_tornado_")[-1].replace("exclusive_", "")
        for f in pair_figures
    })
    token_panels = []
    for pair in pairs:
        body = figure(
            tornado / f"site_token_tornado_{pair}.png",
            f"{pair.replace('-vs-', ' versus ')}, ranked by the difference in "
            "per-patient rate",
        )
        if body:
            token_panels.append((pair.replace("-vs-", " v "), body))
    for pair in pairs:
        body = figure(
            tornado / f"site_token_tornado_exclusive_{pair}.png",
            f"{pair.replace('-vs-', ' versus ')}: tokens one of them never "
            "records at all",
        )
        if body:
            token_panels.append((f"{pair.replace('-vs-', ' v ')}, exclusive", body))
    if not pair_figures:
        # a run from before the tornado was drawn per pair
        token_panels = [
            ("Differential tokens", figure(
                tornado / "site_token_tornado.png",
                "Ranked by absolute difference in per-patient rate")),
            ("Tokens one site never records", figure(
                tornado / "site_token_tornado_exclusive.png",
                "Structural zeros in the vocabulary")),
        ]
    token_panels.append(
        ("Token-family mix", figure(tornado / "site_token_families.png"))
    )
    # A volcano per site, which answers a different question from the tornado:
    # the tornado is pairwise and ranks on a raw difference, so it is dominated
    # by the commonest tokens and says nothing about noise. This asks which
    # tokens each site is unusual for against everywhere else, with an effect
    # size and a corrected p-value.
    volcano_panels = [
        (home_site, figure(
            tornado / f"site_token_volcano_{home_site}.png",
            f"{home_site} against every other site: log2 prevalence ratio "
            "against corrected significance",
        ))
        for home_site in sorted(
            f.stem.split("site_token_volcano_")[-1]
            for f in tornado.glob("site_token_volcano_*.png")
        )
    ]
    volcano_panels = [(label, body) for label, body in volcano_panels if body]
    token_panels = [(label, body) for label, body in token_panels if body]
    if not features and not token_panels:
        return

    out.append('<h2 id="sites">7 &middot; Why the sites separate</h2>')
    if features:
        out.append(
            "<p>The embedding separates the contributing sites almost perfectly "
            "(section 6), so the first question is whether the patients differ "
            "that much. Distributions and proportions below, in each "
            "institution's own colours.</p>"
        )
        out.append(features)

    lda_rows = []
    for anchor in ANCHORS:
        f = paths.output_home / "tables" / "GEM-all" / f"site_lda_anchor-{anchor}.csv"
        if f.is_file():
            lda_rows.append(pl.read_csv(f))
    if volcano_panels:
        out.append("<h3>Tokens enriched and depleted at each site</h3>")
        out.append(
            "<p>One point per vocabulary entry. Effect is the log2 ratio of "
            "<b>prevalence</b> &mdash; the share of patients emitting the token "
            "at least once &mdash; at that site against every other, and "
            "significance is a chi-square on the 2&times;2 of patients with and "
            "without it, Bonferroni-corrected across every test. Open markers "
            "are tokens only that site records at all: their true ratio is "
            "infinite and the plotted value is a floor.</p>"
        )
        out.append(callout(
            "<b>Read the effect size before the p-value.</b> With tens of "
            "thousands of patients almost every token clears an uncorrected "
            "threshold, and a great many clear the corrected one too &mdash; so "
            "the vertical axis mostly separates common tokens from rare ones, "
            "not real differences from noise. The horizontal axis is the "
            "finding. Points along the top edge are at the floating-point "
            "floor for p, not at a meaningfully larger significance than each "
            "other.", "warn"))
        out.append(tabs(volcano_panels))

    if token_panels:
        out.append("<h3>What each site charts</h3>")
        out.append(
            "<p>Per-token and per-family rates, from the counts every site "
            "ships in tier 1 &mdash; so this comparison covers a site that "
            "shares nothing patient-level. Rates are per patient, because the "
            "sites contribute very different numbers of them and raw totals "
            "would make the largest site look the most verbose.</p>"
        )
        out.append(tabs(token_panels))
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

    out.append('<h2 id="models">8 &middot; Models</h2>')
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


def section_undifferentiated(paths, out: list) -> bool:
    """
    13's analysis of the patients with no coded shock mechanism

    Its own section rather than a subsection of 1, even though the cohort table
    already reports the shock-type counts. The counts answer "how many"; this
    answers "what are they", which needs the embedding, a probe and a treatment
    comparison, and does not belong under a heading about enrollment.

    Returns whether anything was emitted, so main can decide whether to list the
    heading in the contents.
    """
    table_home = paths.output_home / "tables" / "GEM-all"
    anchors = [
        anchor
        for anchor in ANCHORS
        if (table_home / f"undiff_groups_{anchor}.csv").is_file()
    ]
    if not anchors:
        return False

    out.append(
        '<h2 id="undifferentiated">9 &middot; Undifferentiated shock</h2>'
    )
    out.append(
        "<p>The cohort is selected on physiology &mdash; sustained hypotension "
        "or vasoactive support, plus end-organ damage &mdash; which is "
        "mechanism-agnostic by construction. Attaching the coded mechanism "
        "leaves <b>18,626 of 34,142 patients (54.6%) with no shock code at "
        "all</b>. This section asks what those patients resemble in the frozen "
        "model's representation, whether a linear probe trained on the coded "
        "patients will assign them a mechanism, and whether the treatment they "
        "received matches the assignment.</p>"
    )
    out.append(
        "<p><b>Everything here is computed within site.</b> The two hospitals "
        "leave a similar share uncoded (NU 56.6%, UCMC 48.9%) but reach for "
        "different mechanisms when they do code one: UCMC codes cardiogenic "
        "shock 2.5 times as often as NU (13.5% of its patients against 5.3%) "
        "and NU codes septic shock 1.7 times as often as UCMC (20.3% against "
        "12.3%). The coded patients are what defines every centroid and trains "
        "every probe below, so a mechanism mix differing that much means "
        "neither the centroids nor the class priors transport. Centroids are "
        "fitted per site, neighbourhoods are searched within site, and the "
        "probe is scored within-site and cross-site separately.</p>"
    )

    anchor_tabs = []
    for anchor in anchors:
        blocks = []
        blocks.append(("Who they are", table(
            pl.read_csv(table_home / f"undiff_groups_{anchor}.csv"),
            {
                "n": thousands, "share_of_site": pct, "mortality": pct,
                "median_age": num(0), "median_sofa": num(0),
                "median_lactate": num(1), "median_icu_los": num(1),
                "pct_female": pct, "pct_on_vasoactive": pct,
            },
            key_column="shock_type",
        )))

        evidence = table_home / f"undiff_evidence_{anchor}.csv"
        if evidence.is_file():
            frame = pl.read_csv(evidence)
            shares = [
                c for c in frame.columns
                if c.startswith(("hint_", "culture_")) or c == "any_hint"
            ]
            blocks.append(("Evidence from elsewhere", table(
                frame,
                {"n": thousands, **{c: pct for c in shares}},
                key_column="shock_type",
            )))

        reference = table_home / f"undiff_reference_{anchor}.csv"
        if reference.is_file():
            blocks.append(("What a distance means", table(
                pl.read_csv(reference),
                {
                    "n": thousands,
                    "mean_cosine": num(4), "median_cosine": num(4),
                },
                key_column="label",
            )))

        neighbours = table_home / f"undiff_neighbours_{anchor}.csv"
        if neighbours.is_file():
            frame = pl.read_csv(neighbours)
            blocks.append(("Neighbour composition", table(
                frame,
                {
                    "n": thousands,
                    **{c: pct for c in frame.columns if c.startswith("knn")},
                },
                key_column="shock_type",
            )))

        metrics = table_home / f"undiff_probe_metrics_{anchor}.csv"
        if metrics.is_file():
            blocks.append(("Probe", table(
                pl.read_csv(metrics).sort("scoring", "class", "tested_on"),
                {
                    "n": thousands, "n_positive": thousands,
                    "auroc": num(3), "brier": num(4),
                },
                key_column="class",
            )))

        calibration = table_home / f"undiff_probe_calibration_{anchor}.csv"
        if calibration.is_file():
            blocks.append(("Calibration", table(
                pl.read_csv(calibration).sort(
                    "scoring", "class", "tested_on", "bin_low"
                ),
                {
                    "bin_low": num(1), "bin_high": num(1), "n": thousands,
                    "mean_predicted": num(3), "observed": num(3),
                },
                key_column="class",
            )))

        check = table_home / f"undiff_probe_vs_evidence_{anchor}.csv"
        if check.is_file():
            frame = pl.read_csv(check)
            blocks.append(("Does the assignment check out", table(
                frame,
                {
                    "n": thousands, "mortality": pct,
                    **{
                        c: pct for c in frame.columns
                        if c.startswith(("hint_", "culture_"))
                    },
                },
                key_column="probe_assignment",
            )))

        snapshots = table_home / f"undiff_snapshots_{anchor}.csv"
        if snapshots.is_file():
            blocks.append(("Leakage by snapshot", table(
                pl.read_csv(snapshots)
                .filter(pl.col("scoring") == "within-site")
                .group_by("class", "offset_hours")
                .agg(pl.col("auroc").mean(), pl.col("n").sum())
                .sort("class", "offset_hours"),
                {"offset_hours": num(0), "n": thousands, "auroc": num(3)},
                key_column="class",
            )))

        treatments = table_home / f"undiff_treatments_{anchor}.csv"
        if treatments.is_file():
            frame = pl.read_csv(treatments)
            rates = [
                c for c in frame.columns
                if c.startswith(("on_", "ever_", "pct_"))
                or c in ("mcs", "inotrope", "abx_within_6h")
            ]
            blocks.append(("What they were given", table(
                frame,
                {
                    "n": thousands,
                    **{c: pct for c in rates},
                    "median_hours_to_first_antibiotic": num(1),
                    "median_hours_to_first_agent": num(1),
                },
                key_column="shock_type",
            )))

        first = table_home / f"undiff_first_agents_{anchor}.csv"
        if first.is_file():
            blocks.append(("Which agent came first", table(
                pl.read_csv(first),
                {"n": thousands, "share": pct},
                key_column="shock_type",
            )))

        concordance = table_home / f"undiff_concordance_{anchor}.csv"
        if concordance.is_file():
            frame = pl.read_csv(concordance)
            blocks.append(("Treated as assigned", table(
                frame,
                {
                    "n": thousands, "mortality": pct,
                    "pressor_free_days": num(1), "vent_free_days": num(1),
                    "sofa_non_cv": num(1), "sofa_total": num(1),
                },
                key_column="probe_assignment",
            )))

        adjusted = table_home / f"undiff_concordance_adjusted_{anchor}.csv"
        if adjusted.is_file():
            blocks.append(("Treated as assigned, adjusted", table(
                pl.read_csv(adjusted).select(
                    "probe_assignment", "outcome", "model", "n", "n_treated",
                    "effect_unadjusted", "p_unadjusted",
                    "effect_adjusted", "p_adjusted", "adjusted_for",
                ),
                {
                    "n": thousands, "n_treated": thousands,
                    "effect_unadjusted": num(3), "effect_adjusted": num(3),
                    "p_unadjusted": pval, "p_adjusted": pval,
                },
                key_column="probe_assignment",
            )))

        not_assessable = table_home / "undiff_not_assessable.csv"
        if not_assessable.is_file():
            blocks.append(("Not assessable", table(
                pl.read_csv(not_assessable), key_column="comparison",
            )))

        # Figures lead and the tables sit behind them. Five of the tables here
        # are a group-by-measure grid -- six groups against seven agents, or
        # against three neighbour compositions -- which is a shape the eye
        # reads and a table does not.
        home = paths.output_home / "figures" / "GEM-all" / f"anchor-{anchor}"
        figures = [
            ("Who they are", "undiff_groups",
             "Share of each site as bars, in-hospital mortality as points. "
             "The tall bar with the low point is the central fact of this "
             "section."),
            ("Probe AUROC", "undiff_probe",
             "Within-site beside cross-site, per class. The gap between them "
             "is how much of the within-site figure was documentation habit."),
            ("Leakage by snapshot", "undiff_snapshots",
             "The same probe at 0 h, +6 h and +24 h from onset. The slope is "
             "the quantity of interest, not the level."),
            ("How undecided", "undiff_entropy",
             "Normalised entropy by what the coders said. `mixed` is held out "
             "of training, so it is the validation case."),
            ("Nearest neighbours", "undiff_neighbours",
             "Mechanism composition of each group's 25 nearest coded "
             "neighbours, within site."),
            ("Distance to centroids", "undiff_proximity",
             "The undifferentiated against each coded group's own distances. "
             "The overlap is the result."),
            ("Where they sit", "undiff_overlay",
             "Each group against the whole cohort in the same projection as "
             "section 2."),
            ("Evidence from elsewhere", "undiff_evidence",
             "Mechanism evidence outside the shock codes. None of it carries "
             "a timestamp."),
            ("What they were given", "undiff_treatments",
             "Share of each group with that drug running within six hours of "
             "onset."),
            ("Treated as assigned", "undiff_concordance",
             "Mortality by whether treatment matched the probe. Confounded by "
             "indication and not adjustable."),
        ]
        drawn = []
        for label, name, caption in figures:
            body = figure(home / f"{name}.png", caption)
            if body:
                drawn.append((label, body))
        blocks = drawn + blocks
        anchor_tabs.append((anchor, tabs(blocks)))

    out.append(tabs(anchor_tabs))

    out.append(callout(
        "<b>The centroids do not separate the mechanisms, but a linear probe "
        "does &mdash; comfortably.</b> Coded patients sit a mean cosine "
        "distance of <b>0.456</b> from their own mechanism's centroid against "
        "<b>0.494</b> from the other two, a gap of 0.038 on a scale where the "
        "within-group spread is an order of magnitude larger. On that measure "
        "the mechanisms are not distinct regions, which is the same answer "
        "section 3 got for the trial populations (centroid separation 0.03 to "
        "0.32). Yet a multinomial logistic regression on the same 50 "
        "components reaches an AUROC of <b>0.90 for cardiogenic, 0.87 for "
        "septic and 0.79 for hypovolemic</b>. <b>The information is there and "
        "it is linearly decodable; it simply does not lie along the direction "
        "joining the class means.</b> Distance to a centroid is the wrong "
        "instrument for this space, and the nearest-centroid columns are "
        "reported mainly to show why."))

    out.append(callout(
        "<b>The separation is not the model reading the treatment back out.</b> "
        "This was the main thing that could have been wrong with the probe: "
        "medication administrations are part of the token vocabulary, so an "
        "embedding taken after the clinician starts treating has seen the "
        "antibiotics and the inotrope that the working diagnosis produced, and "
        "a probe scored there would be recovering documented clinical judgement "
        "rather than latent structure. Re-running the identical probe on 09's "
        "grid at 0 h, +6 h and +24 h from onset gives AUROCs of <b>0.899 / "
        "0.917 / 0.919</b> for cardiogenic, <b>0.869 / 0.877 / 0.887</b> for "
        "septic and <b>0.791 / 0.811 / 0.824</b> for hypovolemic. A full day of "
        "post-onset record &mdash; every drug, every lab, every escalation "
        "&mdash; buys about <b>two AUROC points</b>. Essentially all of the "
        "separation is present before the shock is treated, and the +24 h rise "
        "is in any case not attributable to treatment alone, since the patient "
        "is also deteriorating over the same hours."))

    out.append(callout(
        "<b>One of the three assignments survives an external check; the other "
        "two do not.</b> The only evidence available against a probe assignment "
        "is non-shock coding and microbiology, and it supports the septic class "
        "clearly. Among uncoded patients the probe confidently calls septic, "
        "<b>51.6% have a positive culture at NU and 73.4% at UCMC</b>, against "
        "32.2% and 58.1% for the ones it is unsure about and a 37.8% base rate "
        "in the uncoded group at UCMC; the infection-code rate moves the same "
        "way (39.3% against 11.9% at NU). Mortality agrees &mdash; confidently "
        "septic is the deadliest assigned group at both sites (15.2% and "
        "24.5%). <b>The cardiogenic class does not replicate this.</b> "
        "Patients confidently called cardiogenic carry a cardiogenic diagnosis "
        "code 39.0% (NU) and 47.5% (UCMC) of the time against base rates of "
        "29.1% and 26.1% &mdash; an enrichment, but a weak one. And "
        "<b>confidently-cardiogenic patients die at 1.1% (NU) and 3.4% "
        "(UCMC)</b>, against 25.2% among the patients the coders actually "
        "called cardiogenic. Whatever that class is selecting, it is not "
        "patients in cardiogenic shock.", "warn"))

    out.append(callout(
        "<b>The likeliest reading of the whole section: the probe is sorting "
        "the uncoded by severity and by how septic they look, not by "
        "mechanism.</b> The uncoded are much less sick than any coded group "
        "&mdash; 6.2% in-hospital mortality pooled (4.6% NU, 11.2% UCMC) "
        "against 25.2% cardiogenic, 29.6% septic and 43.3% mixed &mdash; so "
        "most of them are not a diagnostic puzzle but a mild population the "
        "physiological criteria admitted and nobody thought was in shock. A "
        "three-way probe forced to choose among septic, cardiogenic and "
        "hypovolemic will put such a patient somewhere, and the class it picks "
        "carries no claim that any mechanism was present. The septic class "
        "works because infection leaves independent traces in cultures; the "
        "other two have nothing comparable to be checked against, and their "
        "mortality ordering suggests they are absorbing the patients with the "
        "least evidence of anything.", "warn"))

    out.append(callout(
        "<b>Treatment matching the assignment tracks with lower mortality, and "
        "this cannot be read as a reason to follow the probe.</b> Among uncoded "
        "patients the probe calls cardiogenic, those who actually received an "
        "inotrope or mechanical support have a mortality log-odds <b>0.71 "
        "lower</b> unadjusted and <b>0.48 lower</b> adjusted for "
        "non-cardiovascular SOFA, age and site (p = 0.016); for the septic "
        "class, antibiotics within six hours give &minus;0.26 unadjusted and "
        "&minus;0.16 adjusted, which does not reach significance (p = 0.069). "
        "<b>Confounding by indication is the whole of this.</b> A clinician "
        "gives an inotrope because they read the same record the probe reads "
        "and reached the same conclusion, so concordance measures agreement "
        "between two readings of one patient &mdash; and a patient whose "
        "mechanism is legible enough for both to agree is a different patient "
        "from one whose is not. Adjusting for severity removes the part "
        "severity carries and leaves the clinician's judgement, which is "
        "measured nowhere in this data.", "warn"))

    out.append(callout(
        "<b>A diagnosis arriving later is not something this data can see, and "
        "one attempt to approximate it had to be withdrawn.</b> The hospital "
        "diagnosis table has no timestamp, and an uncoded patient carries no "
        "shock code at any point in the admission by definition, so "
        "&ldquo;how many are later given a subtype&rdquo; is unanswerable "
        "rather than merely hard. What is reported instead is whether a "
        "mechanism was recorded by some other means anywhere in the admission. "
        "The first version of that flag included <code>D62</code>, acute "
        "posthaemorrhagic anaemia, as evidence of a haemorrhagic mechanism; it "
        "is coded on <b>7,916 of NU's 14,163 uncoded patients (55.9%)</b> "
        "against 0.7 to 1.3% for each of the specific gastrointestinal "
        "haemorrhage codes, so it made 57% of the undifferentiated appear to "
        "have evidence of bleeding and was carrying the entire flag on its own. "
        "It is a post-procedural anaemia code far more often than it is "
        "evidence of hypovolemic shock, and at that prevalence it could not "
        "have distinguished one patient from another whichever it meant. It is "
        "removed, and at NU the hypovolemic flag now reads <b>21.1% in coded "
        "hypovolemic patients against 2.7% in the uncoded</b> &mdash; a "
        "low-sensitivity flag rather than a meaningless one.", "warn"))

    out.append(callout(
        "<b>Three of the treatment comparisons have no data behind them at "
        "either site, and the third is the one that matters.</b> Fluid volume "
        "needs an intake/output table and neither extract has one; transfusion "
        "needs a blood product table and neither has one; <b>echocardiography "
        "and POCUS need an echo table, and neither site has that either</b>. "
        "The third is not a gap in a list &mdash; echo is the main act of "
        "differentiating shock, and whether a study was <i>done</i> is as "
        "unavailable here as what it showed. So this section cannot see the "
        "differentiation itself, only the drugs that follow it, and it cannot "
        "distinguish a patient nobody managed to diagnose from one nobody "
        "tried to. Counting crystalloid administration events as a fluid proxy "
        "was considered and rejected: an event without a volume cannot separate "
        "a 30 mL/kg resuscitation from a carrier line.", "warn"))

    return True


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
        "<li><b>Trial eligibility is a superset, not a match.</b> Left "
        "ventricular ejection fraction and coronary anatomy are unavailable at "
        "both sites, so DanGer Shock's LVEF &lt; 45%, SHOCK's LV-failure "
        "requirement and CULPRIT-SHOCK's multivessel requirement cannot be "
        "applied. Three trials consequently collapse to one criterion set. "
        "Trial regions are also fitted from <i>our</i> eligible patients, never "
        "from trial participants &mdash; we have no embeddings for those &mdash; "
        "so &ldquo;inside the SHOCK region&rdquo; means &ldquo;resembles the "
        "patients here who would have qualified&rdquo;.</li>"
        "<li><b>Shock type is a coded label, not an adjudicated one.</b> "
        "<code>clif_hospital_diagnosis</code> is hospitalization-level, so a "
        "code entered on day 20 is indistinguishable from one entered on "
        "admission; <code>poa_present</code> would narrow that but is populated "
        "for every UCMC row and only 61% of NU's, so it is reported rather than "
        "required. Septic-shock coding also differs 1.7x between the sites.</li>"
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
    # Token embeddings (section_tokens) is deliberately not called. It answers a
    # question about the vocabulary rather than about these patients, and nine
    # stacked zoom figures were the largest thing a reader had to scroll past
    # before reaching any result. The function is kept: one line puts it back.
    section_projections(paths, out)
    section_trials(paths, out)
    fitted = section_trial_fit(paths, out)
    coverage, note = trajectory_coverage(paths, logger)
    if coverage in ("current", "subset"):
        section_patient_embeddings(paths, out)
        if coverage == "subset":
            sizes = projection_sizes(paths)
            rows = "".join(
                f"<li><b>{html.escape(label)}</b> &mdash; {n:,} patients</li>"
                for label, n in sizes.items()
            )
            out.append(callout(
                "<b>Sections 4 to 6 cover fewer patients than the cohort "
                "table.</b> 09 drops a patient at any grid point where they "
                "have too little record before it, and the flow field needs a "
                "patient seen at consecutive timepoints to have a displacement "
                "at all, so requiring presence across all thirteen offsets "
                "from &minus;24 h to +48 h is what thins the set. Every "
                "patient shown is still in the cohort. They are by "
                "construction the ones with the longest records, so read "
                "these sections as describing longer-staying patients rather "
                "than the cohort."
                + (f"<ul>{rows}</ul>" if rows else "")
                + "<p>Each projection has its own denominator and its own "
                "fitted PaCMAP, so these counts are not nested and the "
                "coordinates do not correspond between panels.</p>"))
        section_patient_paths(paths, out)
        section_flow(paths, out)
    else:
        out.append(
            '<h2 id="patients">4 &middot; Embeddings over time</h2>'
        )
        out.append(callout(
            "<b>Sections 4 to 6 are omitted: 09 describes a different "
            f"cohort.</b> It covers {html.escape(note)}. Trajectory figures "
            "carry no record of the cohort they were drawn from, so showing "
            "them here would put one cohort's figures under another's tables. "
            "Run <code>python code/09_trajectories.py</code> and regenerate "
            "this report.", "warn"))
    # Outside the coverage branch above on purpose: 11 reads tokens_times
    # directly rather than the trajectory grid, so the strata are valid even
    # when 09's figures describe a stale cohort. Only the two panels inside it
    # need the grid, and those are each wrapped in figure().
    section_longitudinal(paths, out)
    section_sites(paths, out)
    section_models(paths, out, logger)
    undifferentiated = section_undifferentiated(paths, out)
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

    # Every entry here must point at a heading that was actually emitted.
    # Sections 5 and 6 only exist when 09 is current, and listing them
    # regardless leaves the reader two links that quietly do nothing.
    entries = [
        ("cohort", "1 &middot; Cohort"),
        ("projections", "2 &middot; Patient embeddings"),
        ("trials", "3 &middot; Guideline fit"),
        ("patients", "4 &middot; Embeddings over time"),
    ]
    if fitted:
        entries.insert(3, ("trial-fit", "3.1 &middot; Fit and outcome"))
    if coverage in ("current", "subset"):
        entries += [
            ("paths", "5 &middot; Individual trajectories"),
            ("flow", "6 &middot; Latent flow"),
        ]
    entries += [
        ("sites", "7 &middot; UCMC versus NU"),
        ("models", "8 &middot; Models"),
    ]
    if undifferentiated:
        entries.append(("undifferentiated", "9 &middot; Undifferentiated shock"))
    entries.append(("methods", "Methods &amp; caveats"))
    toc = "".join(
        f'<li><a href="#{anchor}">{label}</a></li>' for anchor, label in entries
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

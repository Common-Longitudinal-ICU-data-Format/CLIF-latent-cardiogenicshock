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



def aggregate_file(paths, name: str) -> pathlib.Path:
    """
    an aggregate table from this machine's upload folders, where 01 writes
    them, or from output/cohort for runs made before 01 wrote them there
    """
    candidates = [paths.upload_home(site) / "counts" / name for site in paths.sites]
    candidates.append(paths.cohort_home / name)
    return next((c for c in candidates if c.is_file()), candidates[-1])

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
.source {{ font-size: 11px; color: #8a8980; margin: 2px 0 0 0; }}
.source code {{ font-size: 10px; background: #f4f3f0; padding: 1px 4px; }}
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
    """
    one figure, embedded, with a pointer to its manuscript copy

    The embedded image is downscaled for the report; the manuscript copy beside
    it is the same figure at 500 dpi with the axis furniture off and a
    transparent background. The path is printed rather than linked because the
    report travels as a single self-contained file and a relative link would
    break the moment it is moved or emailed.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return ""
    data = base64.b64encode(_downscaled(path)).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    manuscript = path.parent / "manuscript" / path.name
    source = ""
    if manuscript.is_file():
        try:
            shown = manuscript.relative_to(pathlib.Path.cwd())
        except ValueError:
            shown = manuscript
        source = (
            f'<p class="source">manuscript copy (500&nbsp;dpi, no axes, '
            f'transparent): <code>{esc(str(shown))}</code></p>'
        )
    return (
        f'<figure><img src="data:image/png;base64,{data}" alt="{esc(caption)}">'
        f"{cap}{source}</figure>"
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
    ledger_file = aggregate_file(paths, "filter_ledger.csv")
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
    shock_file = aggregate_file(paths, "shock_type_summary.csv")
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
        parts = sorted(
            paths.output_home.glob("upload_to_box_*/counts/baseline_by_shock_type.csv")
        ) or sorted(paths.cohort_home.glob("baseline_by_shock_type_*.csv"))
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
                    # n_positive (share_positive%). `n_measured` used to hold
                    # the positives; it is now the denominator, so rendering it
                    # here would print the wrong number in the numerator slot.
                    pl.when(pl.col("share_positive").is_null())
                    .then(pl.lit(""))
                    .otherwise(
                        pl.col("n_positive").cast(pl.String)
                        + pl.lit(" (")
                        + (pl.col("share_positive") * 100).round(1).cast(pl.String)
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
        variant_file = aggregate_file(paths, f"phenotype_summary_{column}.csv")
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
    seymour_file = aggregate_file(paths, "seymour_summary.csv")
    if seymour_file.is_file():
        phenotype_tabs.append(("Seymour (sepsis)", table(
            pl.read_csv(seymour_file),
            {
                # share_<site> no longer travels -- it was a rate over a
                # denominator this table does not carry. n_total is the
                # denominator `share` is over.
                "n": thousands, "n_total": thousands,
                "n_NU": thousands, "n_UCMC": thousands, "n_RUSH": thousands,
                "share": pct,
                "mortality": pct, "median_icu_los_days": num(1),
            },
            key_column="seymour",
        )))
    bhavani_file = aggregate_file(paths, "bhavani_summary.csv")
    if bhavani_file.is_file():
        phenotype_tabs.append(("Bhavani (temperature)", table(
            pl.read_csv(bhavani_file),
            {
                # share_<site> no longer travels -- it was a rate over a
                # denominator this table does not carry. n_total is the
                # denominator `share` is over.
                "n": thousands, "n_total": thousands,
                "n_NU": thousands, "n_UCMC": thousands, "n_RUSH": thousands,
                "share": pct,
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

        agreement_file = aggregate_file(paths, "phenotype_variant_agreement.csv")
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
#: Which cohort this run is writing for, or None for the combined report.
#: Set from --cohort in main(). The section functions read it rather than
#: taking it as an argument, because almost every one of them would have to
#: pass it through untouched.
COHORT: str | None = None


def cohort_subsets() -> tuple[str, ...]:
    """
    the projection subsets a cohort report should show

    Always includes `all`, because the point of the cohort-specific panels is
    to be read against the whole shock population -- a phenotype that looks
    distinctive in its own cohort and ordinary in the pooled one is a different
    finding from one that looks distinctive in both.
    """
    if COHORT is None:
        return PROJECTION_ORDER
    return ("all", COHORT)


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
    for subset in cohort_subsets():
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
    # The COHORT being rendered comes first. Without this the order was a
    # fixed preference list headed by cardiogenic, so the septic report
    # measured its own staleness against the cardiogenic grid and quoted the
    # cardiogenic patient count in its sections 4-6 caveat.
    order = ([COHORT] if COHORT else []) + [
        s for s in SUBSET_ORDER if s != COHORT
    ]
    for subset in order:
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
    # A PATH THAT CANNOT EXIST, not None. Every use of this is
    # `figure(septic_home / name)`, and figure() already returns "" for a
    # missing file -- so an unresolvable directory should make the figures
    # vanish, which is what it means. Defaulting to None instead turned each
    # of those into `None / name`, and the cardiogenic report crashed on
    # whichever one it reached first; three had already been guarded
    # individually by the time this was traced to the default.
    septic_home = next(
        (info["home"] for info in homes.values()
         if info["subset"] == "septic"
         and info["correction"] == PREFERRED_CORRECTION),
        paths.output_home / "figures" / "__no_septic_projection__",
    )
    tables_home = paths.output_home / "tables" / "GEM-all"
    movement_file = tables_home / f"state_movement_{COHORT or 'septic'}.csv"
    character_file = tables_home / f"cluster_characteristics_sequence_cluster_{COHORT or 'septic'}.csv"
    growth_file = tables_home / f"cluster_characteristics_growth_cluster_{COHORT or 'septic'}.csv"
    sites_file = tables_home / f"cluster_sites_sequence_cluster_{COHORT or 'septic'}.csv"
    tokens_file = tables_home / f"cluster_mortality_tokens_{COHORT or 'septic'}.csv"
    if not any(f.is_file() for f in (movement_file, character_file, tokens_file)):
        return

    out.append('<h2 id="longitudinal">7 &middot; Longitudinal strata</h2>')
    out.append(callout(
        "<b>This section is the token-composition analysis only.</b> 12 NMF "
        "factors over the site-balanced token counts, hardened by k-means into "
        "<b>6 care states</b>; each patient's 29-symbol sequence of those "
        "states clustered into <b>4 strata, S1&ndash;S4</b>. A second method, "
        "<b>latent class growth (G1&ndash;G4)</b>, uses the same factors' "
        "<i>soft</i> weights directly and never hardens them, which is why it "
        "agrees with S1&ndash;S4 at only ARI 0.253.<br><br>"
        "These states are <b>data-driven</b>, so they have to be interpreted "
        "after the fact &mdash; and eight of the twelve factors turn out to "
        "describe care delivery or documentation rather than the patient.<br><br>"
        "The clinically-defined counterpart &mdash; MAP crossed with what the "
        "pressors are doing &mdash; is <a href=\"#states\">section 8</a>. The "
        "two are parallel analyses sharing only the patients: the care states "
        "here are <b>not</b> paths through the perfusion states there, and the "
        "difference in how many states each has is just two analyses choosing "
        "differently."
    ))
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

    state_tokens_file = tables_home / f"care_state_tokens_{COHORT or 'septic'}.csv"
    if state_tokens_file.is_file():
        out.append('<h3>What the six care states are</h3>')
        out.append(
            "<p>A state is an index until you know what is in it. Each row is "
            "a token over-represented in the windows assigned to that state, "
            "ranked by <b>lift</b> &mdash; its prevalence among the state's "
            "windows against its prevalence across all windows &mdash; with a "
            "floor of 25% prevalence in the state. The floor matters: the "
            "commonest tokens are common everywhere, so a vital charted in 90% "
            "of all windows sits in 90% of every state's windows and "
            "distinguishes none of them, while ranking on lift alone surfaces "
            "rare tokens whose ratio is noise.</p>"
        )
        out.append(table(
            pl.read_csv(state_tokens_file),
            {
                "patient_windows": thousands,
                "prevalence_in_state": pct,
                "prevalence_overall": pct,
                "lift": num(2),
            },
            key_column="care_state",
        ))

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

    sequences_file = tables_home / f"cluster_sequences_{COHORT or 'septic'}.csv"
    if sequences_file.is_file():
        out.append('<h3>What the sequences in each stratum look like</h3>')
        out.append(
            "<p>The commonest sequences in each stratum, and how the stratum "
            "spends its time. Both, because neither alone is honest. The "
            "modal sequence is vivid but thin &mdash; with six symbols over 29 "
            "positions the most common string is only 0.7&ndash;2.7% of its "
            "stratum, and reading it as 'what this stratum looks like' is the "
            "mistake that led to calling these static when 98.6% of patients "
            "move. The occupancy column is the mass: the share of all that "
            "stratum's windows spent in each state. Digits are care states; "
            "<code>0</code> is a window with nothing recorded.</p>"
        )
        out.append(table(
            pl.read_csv(sequences_file),
            {
                "patients": thousands,
                "patients_with_this_sequence": thousands,
                "share_of_cluster": pct,
            },
            key_column="cluster",
        ))

    character_tabs = []
    for path, label in (
        (character_file, "State sequences"),
        (growth_file, "Latent class growth"),
        (tables_home
         / f"cluster_characteristics_perfusion_cluster_{COHORT or 'septic'}.csv",
         "MAP/pressor sequences"),
    ):
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

    if septic_home.is_dir():
        panels = [
            ("Care state (per timepoint)",
             figure(septic_home / "trajectory_flow_care_state.png",
                    "The state assigned to each six-hour window")),
            ("Stratum (per patient)",
             figure(septic_home / "trajectory_flow_stratum.png",
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
            out.append(
                "<p>The two are easy to conflate, so to restate: a <b>care "
                "state</b> is assigned to one six-hour window, and a patient "
                "passes through several &mdash; 98.6% change state at least "
                "once over the week. A <b>stratum</b> is assigned to the "
                "patient, by clustering their whole 29-symbol string of care "
                "states, so it describes the shape of the entire week and is "
                "constant within a patient by construction.</p>"
            )
            out.append(tabs(panels))

    factor_dag = figure(
        paths.output_home / "figures" / "GEM-all"
        / f"longitudinal_factor_dag_{COHORT or 'septic'}.png",
        "",
    )
    if factor_dag:
        out.append('<h3>Each latent class as a path through the NMF factors</h3>')
        out.append(
            "<p>Not on the projection. The PaCMAP-placed graphs tie the nodes "
            "to the embedding geometry, which is right for four or six states "
            "and unreadable for twelve &mdash; the nodes pack into whatever "
            "space the cloud occupies. Here the factors are spread on a circle "
            "so every edge is visible. The cost is that <b>distance on the "
            "page no longer means anything</b>; only the edges do.</p>"
        )
        out.append(
            "<p>Node colour matches the factor's colour in the hexagon maps, "
            "and arrow colour matches the class's colour in the flow overlay, "
            "so the three figures can be read together. Nodes are ordered "
            "round the circle by the <b>mean window at which each factor "
            "dominates</b>, making the ring roughly a timeline, so a class "
            "that progresses reads as motion round it rather than as a "
            "scribble. Node size is how many windows the factor dominates; the "
            "figure inside is the share that stay put.</p>"
        )
        out.append(callout(
            "Each window is assigned to its <b>single largest</b> factor. That "
            "is a hard reading of a soft representation, and it is the point of "
            "comparison: if these paths resemble the care-state graph, then "
            "hardening twelve factors into six states by k-means lost little. "
            "Transitions below 4% of a factor's departures are dropped &mdash; "
            "twelve nodes allow 132 directed edges per class and most are "
            "noise."
        ))
        out.append(factor_dag)

    # The PaCMAP-placed state graphs are deliberately not shown. Nodes sat at
    # each state's median projection position, which collides outright when
    # two states overlap in the embedding -- states 1 and 4 shared a position
    # and 2 and 3 shared another, so four of six nodes were drawn on top of
    # each other. The circular graphs off the projection say the same thing
    # and can be read.

    # Guarded on the projection directory, not on the factor graph above.
    # These panels sat inside `if factor_dag:` -- an unrelated condition --
    # and dereferenced septic_home without checking it, so every one of
    # them vanished whenever the factor DAG was missing and the cohort
    # reports crashed outright whenever the projection directory was not
    # resolvable.
    if septic_home.is_dir():
        state_views = [
            (label, figure(septic_home / f"trajectory_{tag}.png", note))
            for tag, label, note in (
                ("care_state_hex_plurality", "Plurality",
                 "Each hexagon's most common care state"),
                ("care_state_hex_plurality_noflow", "Plurality, no flow", ""),
                ("care_state_hex_enrichment", "Enrichment",
                 "The care state most concentrated there relative to its own "
                 "total"),
                ("care_state_hex_enrichment_noflow", "Enrichment, no flow", ""),
                ("care_state_panels", "One state at a time",
                 "Each care state against the whole cloud"),
            )
        ]
        state_views = [(l, b) for l, b in state_views if b]
        if state_views:
            out.append('<h3>Where each care state sits</h3>')
            out.append(
                "<p>The same three views the MAP/pressor states get in section "
                "6, applied to the six token-composition states. Read the two "
                "hexagon modes against each other: agreement means the "
                "structure is not an artefact of how unevenly the states are "
                "sized, disagreement localises where a big state owns a region "
                "by headcount while a small one owns it by concentration.</p>"
            )
            out.append(tabs(state_views))

        factor_views = [
            (label, figure(septic_home / f"trajectory_{tag}.png", note))
            for tag, label, note in (
                ("nmf_factors", "All twelve factors",
                 "One panel per NMF factor, quantile-ranked"),
                ("max_factor_hex_enrichment", "Dominant factor, enrichment",
                 "Each window labelled by its single largest factor weight"),
                ("max_factor_hex_enrichment_noflow",
                 "Dominant factor, no flow", ""),
                ("max_factor_hex_plurality", "Dominant factor, plurality",
                 "The same partition by headcount rather than concentration"),
            )
        ]
        factor_views = [(l, b) for l, b in factor_views if b]
        if factor_views:
            out.append('<h3>The soft layer, before it was hardened</h3>')
            out.append(
                "<p>The six care states are <b>k-means run on twelve "
                "continuous NMF factor weights</b>. That step throws the "
                "softness away: a window that was 51/49 between two factors "
                "becomes indistinguishable from one that was 99/1. These are "
                "the only views of what was discarded, and the question they "
                "answer is whether six states are a fair summary of twelve "
                "factors or a lossy one.</p>"
            )
            out.append(
                "<p>Weights are <b>quantile-ranked rather than plotted raw</b>. "
                "NMF weights are strongly right-skewed &mdash; most windows "
                "load near zero on most factors &mdash; so a raw ramp renders "
                "every panel as a dark field with a few bright specks. The "
                "second tab is the argmax partition: each window given to its "
                "single largest factor, which is the grouping the soft layer "
                "implies before any clustering.</p>"
            )
            out.append(tabs(factor_views))

        out.append('<h3>Hexagon maps of the patient clusters</h3>')
        out.append(
            "<p>The same tiling applied to the three <b>per-patient</b> "
            "clusterings rather than to per-timepoint states. The question "
            "shifts accordingly: not 'what kind of care happens here' but "
            "'whose trajectories pass through here'. A cluster that owns a "
            "compact set of hexagons stays put; one scattered across the map "
            "moves through it.</p>"
        )
        out.append(
            "<p>Read plurality and enrichment against each other as before. "
            "The clusters are closer in size than the perfusion states are "
            "&mdash; the smallest is about a quarter of the largest rather "
            "than a ninth &mdash; so the two modes should diverge less here. "
            "If they diverge as much, that is a sign the tiling is picking up "
            "density rather than membership.</p>"
        )
        for column, label in (
            ("perfusion_cluster", "MAP/pressor clusters, P1&ndash;P4"),
            ("sequence_cluster", "Care-state sequence clusters, S1&ndash;S4"),
            ("growth_cluster", "Latent class growth, G1&ndash;G4"),
        ):
            panels = [
                (name, figure(septic_home / f"trajectory_{column}_hex_{tag}.png", ""))
                for tag, name in (
                    ("plurality", "Plurality"),
                    ("plurality_noflow", "Plurality, no flow"),
                    ("enrichment", "Enrichment"),
                    ("enrichment_noflow", "Enrichment, no flow"),
                )
            ]
            panels = [(n, b) for n, b in panels if b]
            if panels:
                out.append(f"<p><b>{label}</b></p>")
                out.append(tabs(panels))

        cluster_views = [
            (label, figure(septic_home / f"trajectory_{tag}_panels.png", note))
            for tag, label, note in (
                ("perfusion_cluster", "MAP/pressor clusters (P1-P4)",
                 "Each cluster's patient-timepoints against the whole cloud"),
                ("sequence_cluster", "Token strata (S1-S4)", ""),
                ("growth_cluster", "Latent class growth (G1-G4)", ""),
            )
        ]
        cluster_views = [(l, b) for l, b in cluster_views if b]
        if cluster_views:
            out.append('<h3>Where each cluster of patients lives</h3>')
            out.append(
                "<p>A different question from the state panels above. A state "
                "panel asks where a <i>moment of care</i> sits; these ask "
                "where a <i>whole trajectory</i> lives, because the cluster "
                "label is constant along a patient by construction. A cluster "
                "that occupies one region moves little; a cluster smeared "
                "across the projection is one whose patients traverse it.</p>"
            )
            out.append(
                "<p>All three stratifications are shown because they disagree. "
                "The token strata and the growth classes share the same "
                "underlying features and still agree only at ARI 0.253, and "
                "the MAP/pressor clusters come from physiology rather than "
                "tokens at all.</p>"
            )
            out.append(tabs(cluster_views))

    # One tab per patient, each carrying both views of that same patient: their
    # path over the class flow fields, and their route through the 12 factors.
    individual = []
    factor_dir = (
        paths.output_home / "figures" / "GEM-all" / f"patient_factor_paths_{COHORT or 'septic'}"
    )
    for index in range(1, 13):
        panes = []
        if septic_home.is_dir():
            panes.append(
                figure(septic_home / f"paths_over_growth_{index:02d}.png", "")
            )
        panes.append(figure(factor_dir / f"patient_factor_path_{index:02d}.png", ""))
        panes = [pane for pane in panes if pane]
        if panes:
            individual.append((f"Patient {index}", "".join(panes)))
    if individual:
        out.append('<h3>Individual patients against the latent classes</h3>')
        out.append(
            "<p>The class-level figures are averages over roughly two thousand "
            "patients each. These are the unsmoothed version, and the point of "
            "them is to show the gap: a flow field is a mean over tens of "
            "thousands of six-hour displacements, so a single patient moving "
            "against it is not a contradiction &mdash; it is what the mean is "
            "a mean of.</p>"
        )
        out.append(
            "<p>In the first tab each marker is filled white at the earliest "
            "timepoint through black at the latest, and <b>outlined in the "
            "colour of the patient's own class</b>, so a path can be matched to "
            "the field it should be compared against without a legend lookup. "
            "In the second, the same patients are shown as routes through the "
            "12 NMF factors on the same circle and ordering as the class-level "
            "graph; factors a patient never visited stay faint, each visited "
            "node is outlined white-to-black by timepoint with the hour beside "
            "it, and the arrows are the patient's own transitions in black. "
            "Node fill is the factor's own colour at full opacity and the "
            "outline carries the time, so colour answers which factor and the "
            "ring answers when.</p>"
        )
        out.append(callout(
            "<b>Check the factors-visited count in each panel title.</b> If a "
            "patient touches only three or four of the twelve, then the "
            "class-level factor graph above is a <i>union across patients</i> "
            "rather than a route anyone took, and 'the class moves through "
            "these factors' has to be read as a statement about the group, not "
            "about its members."
        ))
        out.append(tabs(individual))

    factor_tokens_file = tables_home / f"nmf_factor_tokens_{COHORT or 'septic'}.csv"
    if factor_tokens_file.is_file():
        out.append('<h3>What each NMF factor is made of</h3>')
        out.append(
            "<p>The top-loading tokens per factor, with each token's share of "
            "that factor's mass and the share the top twelve carry together. "
            "<code>components_</code> is non-negative, which is why NMF is used "
            "here rather than PCA: a factor is a set of events that co-occur, "
            "so its loadings read as a description instead of a contrast "
            "between two things.</p>"
        )
        out.append(
            "<p>The <b>cover</b> column is the one to read first. A factor "
            "whose top twelve tokens carry most of its mass is one clinical "
            "pattern and can be named; a factor carrying 40&ndash;55% is a "
            "diffuse mixture, and those turn out to be the interesting ones "
            "&mdash; they are where NMF found genuine co-occurrence rather "
            "than allocating a whole component to a single token.</p>"
        )
        out.append(callout(
            "<b>Eight of the twelve factors are care delivery or "
            "documentation, not physiology.</b> F1 is <code>UNK</code> at 73% "
            "&mdash; token id 0, the tokenizer's reserved out-of-vocabulary "
            "slot, emitted whenever an event has no entry in the fitted "
            "vocabulary (7.8% of tokens at UCMC, 8.8% at NU, 14.8% at Rush). "
            "F4 is end-of-record (<code>EOS</code> plus "
            "<code>XFR-OUT//icu</code>, 99% cover). F10 is <i>which labs were "
            "ordered</i>, not their values. F2, F5, F7 and F8 are four slices "
            "of the respiratory-support ladder read off device and FiO&#8322; "
            "settings.<br><br>"
            "Only F6 (hypotension), F9 (hypertension), F11 (tachycardia with "
            "tachypnoea) and F12 (high SpO&#8322;) are primarily patient "
            "state, and those are exactly the diffuse ones. So when the latent "
            "classes move through factors, much of that movement is through "
            "<b>respiratory support levels and charting states</b>. That is a "
            "real finding, but it means these are better described as classes "
            "of care-delivery trajectory than of illness trajectory."
        ))
        frame = pl.read_csv(factor_tokens_file)
        out.append(table(
            frame.rename({"top_tokens_cover": "cover"}),
            {"loading": num(3), "share_of_factor": pct, "cover": pct},
            key_column="factor",
        ))

    adjusted_file = tables_home / f"adjusted_mortality_{COHORT or 'septic'}.csv"
    if adjusted_file.is_file():
        out.append('<h3>Mortality adjusted for severity at onset</h3>')
        out.append(
            "<p>Logistic regression of in-hospital death on baseline "
            "physiology at shock onset &mdash; SOFA, MAP, lactate, "
            "norepinephrine equivalents, vasoactive agent count, GCS, "
            "creatinine, platelets, ventilation and age &mdash; with and "
            "without the cluster term. AUC is five-fold cross-validated; the "
            "odds ratios are fitted on all rows against the first cluster as "
            "reference.</p>"
        )
        out.append(
            "<p>The <b>in record at +48h</b> rows are a landmark: restricted "
            "to patients still being charted two days in, so the sequence that "
            "assigned the cluster is observed before most of the deaths it is "
            "asked to predict. It is a partial fix only &mdash; the criterion "
            "is 'has a recent MAP at +48h', which most patients satisfy even "
            "if they die on day five.</p>"
        )
        out.append(callout(
            "<b>Read these against the balance plot above.</b> "
            "'Record ended' is a symbol in the clustering feature space, so "
            "the most lethal clusters are partly identified <i>by</i> dying "
            "and no adjustment removes that. The covariate balance decides "
            "which of these numbers is a comparison and which is "
            "extrapolation: G2 sits 1.73 standard deviations from the other "
            "classes on baseline ventilation, so its odds ratio is arithmetic "
            "rather than evidence, while the MAP/pressor clusters stay within "
            "0.44 and are the ones worth quoting.",
            "warn"
        ))
        out.append(table(
            pl.read_csv(adjusted_file),
            {
                "n": thousands, "died": pct, "crude_mortality": pct,
                "adjusted_odds_ratio": num(2),
                "auc_severity_only": num(4), "auc_with_cluster": num(4),
            },
            key_column="stratification",
        ))

    balance_panels = [
        (label, figure(
            paths.output_home / "figures" / "GEM-all"
            / f"covariate_balance_{tag}_{COHORT or 'septic'}.png", ""))
        for tag, label in (
            ("perfusion", "MAP/pressor clusters"),
            ("growth", "Latent class growth"),
            ("sequence", "Care-state sequence clusters"),
        )
    ]
    balance_panels = [(l, b) for l, b in balance_panels if b]
    if balance_panels:
        out.append('<h3>Are the clusters comparable? Baseline covariate balance</h3>')
        out.append(
            "<p>One dot per baseline covariate per cluster: the standardised "
            "mean difference between that cluster and all the others pooled, "
            "measured <b>at shock onset</b> before the trajectories diverge. "
            "The shaded band is |SMD|&nbsp;&lt;&nbsp;0.1, the usual threshold "
            "for balance; the dashed lines mark 0.25, past which an adjusted "
            "estimate is extrapolating rather than comparing.</p>"
        )
        out.append(
            "<p>This is the figure that says whether the adjusted odds ratios "
            "elsewhere mean anything. A regression will happily report an "
            "effect for a cluster whose covariate distribution barely overlaps "
            "the others' &mdash; the number is arithmetic, not evidence.</p>"
        )
        out.append(callout(
            "<b>The latent classes are badly imbalanced and the MAP/pressor "
            "clusters are not.</b> G2 sits at SMD &minus;1.73 on invasive "
            "ventilation and +1.25 on GCS, seven times the severe-imbalance "
            "threshold, so its adjusted odds ratio of 0.57 against G1 rests on "
            "groups that scarcely overlap on ventilation at baseline. The "
            "MAP/pressor clusters' worst imbalance is 0.44 (P1, vasoactive "
            "agents) &mdash; still past 0.25, but far closer to a like-for-"
            "like comparison. Of the three stratifications, P1&ndash;P4 is the "
            "one whose adjusted mortality differences are safest to read.",
            "warn"
        ))
        out.append(tabs(balance_panels))

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
        # All three stratifications, not just one. Filtering to
        # sequence_cluster meant the perfusion clusters -- the ones with the
        # cleanest mortality separation -- had 404 significant tokens computed
        # and none of them shown.
        token_tabs = []
        for stratification, label in (
            ("perfusion_cluster", "MAP/pressor clusters"),
            ("sequence_cluster", "Token strata"),
            ("growth_cluster", "Latent class growth"),
        ):
            part = (
                significant.filter(pl.col("stratification") == stratification)
                .sort(["cluster", "p_adjusted"])
                .drop("stratification", "p")
                .head(48)
            )
            if part.is_empty():
                continue
            token_tabs.append((label, table(
                part,
                {
                    "died_prevalence": pct,
                    "survived_prevalence": pct,
                    "log2_ratio": num(2),
                    "p_adjusted": num(4),
                },
                key_column="cluster",
            )))
        if token_tabs:
            out.append(tabs(token_tabs))


def section_state_space(paths, out: list) -> None:
    """
    the MAP/pressor state space, its motifs, and where trajectories diverge

    Its own section rather than a subsection of the latent-class work,
    because the two share only the patients. This one starts from a state
    space a clinician defines in advance -- a MAP threshold and what the
    pressors are doing -- where section 7 starts from factors a matrix
    factorisation discovered and which then have to be interpreted. They
    answer different questions and disagree often enough that running them
    together is the point.
    """
    homes = trajectory_homes(paths)
    # A PATH THAT CANNOT EXIST, not None. Every use of this is
    # `figure(septic_home / name)`, and figure() already returns "" for a
    # missing file -- so an unresolvable directory should make the figures
    # vanish, which is what it means. Defaulting to None instead turned each
    # of those into `None / name`, and the cardiogenic report crashed on
    # whichever one it reached first; three had already been guarded
    # individually by the time this was traced to the default.
    septic_home = next(
        (info["home"] for info in homes.values()
         if info["subset"] == "septic"
         and info["correction"] == PREFERRED_CORRECTION),
        paths.output_home / "figures" / "__no_septic_projection__",
    )
    tables_home = paths.output_home / "tables" / "GEM-all"
    figures_home = paths.output_home / "figures" / "GEM-all"
    if not (tables_home / f"motif_vocabulary_rle_{COHORT or 'septic'}.csv").is_file():
        return

    out.append('<h2 id="states">8 &middot; MAP/pressor states and their motifs</h2>')
    out.append(callout(
        "<b>15 states, not 4.</b> MAP above or below 65&nbsp;mmHg with the "
        "same hysteresis band as before, crossed with what the pressors are "
        "doing &mdash; none, rising, falling or steady &mdash; and with "
        "whether norepinephrine runs alone or alongside another agent. "
        "Composition only applies while pressors run, so the count is "
        "2&nbsp;&times;&nbsp;(1&nbsp;+&nbsp;3&nbsp;&times;&nbsp;2) = 14 plus a "
        "terminal symbol.<br><br>"
        "A dose change counts only if it clears <b>both</b> 25% relative and "
        "0.01&nbsp;mcg/kg/min absolute. The relative floor alone makes "
        "0.01&nbsp;&rarr;&nbsp;0.02 a '100% increase', which is titration "
        "noise; the absolute floor alone would call 0.30&nbsp;&rarr;&nbsp;0.32 "
        "meaningful. Without both, the trend flips at nearly every six-hour "
        "point and every motif built on it measures charting noise.<br><br>"
        "Because the state now encodes change, a <i>sequence</i> of these "
        "states encodes change-of-change: a motif reading 'rising then "
        "falling' is a second-order statement about how the team responded, "
        "not a first-order one about the patient."
    ))

    crosstab = tables_home / f"state_crosstab_{COHORT or 'septic'}.csv"
    if crosstab.is_file():
        out.append('<h3>The state space as a grid</h3>')
        out.append(
            "<p>Every state except the terminal one is a MAP level crossed "
            "with a pressor arm, so this is the shape the fifteen states "
            "actually have. Each cell is the share of all patient-windows "
            "falling in it.</p>"
        )
        out.append(
            "<p>The marginals answer different questions from the cells. "
            "Column totals are how much of the week is spent with an "
            "adequate MAP; row totals are how often each pressor posture is "
            "adopted at all. A cell is the interaction, and the cells worth "
            "looking at are those the two marginals would not have "
            "predicted.</p>"
        )
        out.append(table(
            pl.read_csv(crosstab),
            {"MAP >65": pct, "MAP <65": pct, "both MAP levels": pct},
            key_column="pressor arm",
        ))

    representation = tables_home / f"state_representation_{COHORT or 'septic'}.csv"
    if representation.is_file():
        out.append('<h3>The fifteen states</h3>')
        out.append(
            "<p>Every state, with the share of patient-windows it accounts "
            "for. The symbol is what appears in a sequence, so a motif like "
            "<code>30</code> reads directly off this table. States run past 9 "
            "into A&ndash;E so each stays one character and a sequence stays "
            "legible.</p>"
        )
        out.append(table(
            pl.read_csv(representation, schema_overrides={"symbol": pl.String}),
            {"count": thousands, "share": pct},
            key_column="symbol",
        ))

    divergence_tabs = []
    for length in (2, 3):
        path = tables_home / f"divergence_tokens_prefix{length}_{COHORT or 'septic'}.csv"
        if not path.is_file():
            continue
        frame = pl.read_csv(path, schema_overrides={
            "prefix": pl.String, "destination_a": pl.String,
            "destination_b": pl.String,
        }).filter(pl.col("p_adjusted") < 0.05).sort("p_adjusted")
        if frame.is_empty():
            continue
        divergence_tabs.append((f"{length}-window prefix", table(
            frame.drop("p", "shared_window_hours"),
            {
                "n_a": thousands, "n_b": thousands,
                "prevalence_a": pct, "prevalence_b": pct,
                "log2_ratio": num(2), "p_adjusted": num(4),
            },
            key_column="prefix",
        )))
    if divergence_tabs:
        out.append('<h3>Where identical trajectories diverge</h3>')
        out.append(
            "<p>Among patients who have occupied the <b>same states in the "
            "same order</b> up to the same hour, what was in the record at the "
            "last shared window that distinguishes the ones who go one way "
            "from the ones who go another? This holds the path fixed, which is "
            "stronger than holding severity fixed: every patient in a "
            "comparison is indistinguishable by state until the moment they "
            "split.</p>"
        )
        out.append(
            "<p>Each row is one token, tested on per-patient prevalence in the "
            "last shared window, between the two largest destination states. "
            "Cells below 11 patients are suppressed and Bonferroni runs over "
            "every token-by-comparison test, which is conservative &mdash; "
            "1,178 tests at a 2-window prefix.</p>"
        )
        out.append(callout(
            "<b>Rows marked state-defining are close to circular and should "
            "be read last.</b> A norepinephrine dose token in the last shared "
            "window can <i>be</i> the change that puts the patient in a "
            "different state at the next one, so finding it enriched says a "
            "dose increase precedes a dose increase. MAP and SBP tokens are "
            "flagged for the same reason: the state is defined by MAP. The "
            "informative rows are the unflagged ones.<br><br>"
            "This is association and not cause. Both groups are observed "
            "rather than assigned, so a token appearing before a divergence "
            "may mark something already underway rather than a lever that "
            "could be pulled."
        ))
        out.append(tabs(divergence_tabs))

    motif_tabs = []
    for tag, label in (("rle", "Collapsed runs"), ("raw", "Raw sequences")):
        path = tables_home / f"motif_vocabulary_{tag}_{COHORT or 'septic'}.csv"
        if not path.is_file():
            continue
        # Int64 inference would turn "00" into 0 and "03" into 3, collapsing
        # distinct motifs onto each other.
        frame = pl.read_csv(path, schema_overrides={"motif": pl.String})
        motif_tabs.append((label, table(
            frame,
            {"patients": thousands, "share_of_patients": pct},
            key_column="motif",
        )))
    if motif_tabs:
        out.append('<h3>State motifs &mdash; short patterns as words</h3>')
        out.append(
            "<p>Each patient's path through the 15 MAP-by-pressor states is a "
            "string, chopped into overlapping 2- to 4-grams. A motif carried "
            "by at least 1% of patients becomes a word, and a patient becomes "
            "a tf-idf bag of the words their trajectory contains. Clustering "
            "that groups patients who make the same <b>moves</b>, wherever in "
            "the week they make them.</p>"
        )
        out.append(
            "<p>This is what the Hamming clusterings elsewhere cannot do. They "
            "score two patients on being in the same state at the same hour, "
            "so a wean at +60&nbsp;h and a wean at +96&nbsp;h come out "
            "maximally different at every position where they disagree, "
            "despite being the same manoeuvre. A motif is position-free: both "
            "contribute <code>30</code> and land together.</p>"
        )
        out.append(callout(
            "<b>Two readings of the same sequences.</b> On the raw string, "
            "motifs carry duration &mdash; holding a state for five windows "
            "looks different from holding it for one, and the commonest "
            "motifs are simply runs (<code>00</code>, <code>000</code>, "
            "<code>0000</code> in 73&ndash;78% of patients, because 55% of all "
            "windows are 'MAP&nbsp;&gt;65, no pressors'). Collapsing runs "
            "first strips duration out and leaves the transitions, which is "
            "why the collapsed vocabulary is smaller (241 against 421) and its "
            "top entries are manoeuvres rather than plateaus: <code>03</code> "
            "escalation in 34% of patients, <code>30</code> wean in 33%, "
            "<code>32</code> start of a taper in 32.9%.<br><br>"
            "The median sequence is 29 windows raw and <b>6 after "
            "collapsing</b> &mdash; a typical patient occupies only six "
            "distinct runs across the week, which is the clearest measure yet "
            "of how little the perfusion state actually moves."
        ))
        out.append(tabs(motif_tabs))

    perfusion_file = tables_home / f"perfusion_sequences_{COHORT or 'septic'}.csv"
    key_file = tables_home / f"perfusion_symbol_key_{COHORT or 'septic'}.csv"
    if perfusion_file.is_file():
        out.append('<h3>Sequences through the MAP-by-pressor states</h3>')
        out.append(
            "<p>The same question asked of a <b>clinically defined</b> state "
            "space rather than a learned one. The six care states above come "
            "out of an NMF over token composition and have to be interpreted "
            "after the fact; these four are a MAP threshold with hysteresis "
            "crossed with whether vasopressors are running, so they mean "
            "something before any clustering happens. Patients are clustered "
            "on their 29-symbol path through them.</p>"
        )
        out.append(
            "<p>Five symbols, not four: a window after the record ends is its "
            "own symbol. Folding it into 'MAP&nbsp;&gt;&nbsp;65, no "
            "vasopressors' would call a discharged patient a well-perfused "
            "one, and folding it into a low state would call a dead patient "
            "hypotensive &mdash; the second flatters the clustering, since "
            "death is the outcome these strata get judged against.</p>"
        )
        if key_file.is_file():
            out.append(table(pl.read_csv(key_file), {}, key_column="symbol"))
        out.append(table(
            pl.read_csv(perfusion_file),
            {
                "patients": thousands,
                "patients_with_this_sequence": thousands,
                "share_of_cluster": pct,
                "median_changes": num(0),
            },
            key_column="cluster",
        ))

    if septic_home.is_dir():
        overlays = [
            (label, figure(septic_home / f"trajectory_flow_overlay_{tag}.png", note))
            for tag, label, note in (
                ("stratum", "Token strata",
                 "One field per care-state sequence cluster"),
                ("growth", "Latent class growth",
                 "One field per latent class, all four on the same axes"),
                ("perfusion", "MAP/pressor clusters",
                 "One field per MAP-by-pressor sequence cluster"),
                ("care_state", "Care states",
                 "One field per care state, pooled across patients"),
                ("perfusion_state", "Perfusion states",
                 "One field per MAP-by-pressor state"),
            )
        ]
        overlays = [(label, body) for label, body in overlays if body]
        if overlays:
            out.append('<h3>A separate flow field per group, overlaid</h3>')
            out.append(
                "<p>Unlike every other flow figure here, <b>these fit one "
                "field per group and overlay them</b>. The others fit a single "
                "field over everyone and vary only the hue beneath it, so that "
                "a difference between two panels cannot be a difference in the "
                "arrows. These do the opposite deliberately: each group's "
                "field comes from its own patients' displacements, so the "
                "question is not where a group sits but whether the groups "
                "<i>move differently through the same space</i>.</p>"
            )
            out.append(
                "<p>Two cautions. Streamlines of different colours crossing is "
                "not a contradiction &mdash; it is two groups passing through "
                "one region in different directions, which is precisely what "
                "the pooled field averages away. And the groups differ in size "
                "several-fold, so a sparse group's field rests on fewer "
                "vectors at the same linewidth; the per-group vector count is "
                "in the legend rather than implied by the ink.</p>"
            )
            out.append(tabs(overlays))

    # Perfusion state: MAP adequacy crossed with pressor exposure, as four
    # groups with a hysteresis band on the MAP side. Separate from the
    # physiology underlays because it is a STATE a patient is in rather than a
    # value they have, and the transition matrix below only makes sense for a
    # state.
    perfusion = figure(
        septic_home / "trajectory_flow_perfusion.png",
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

    bpe_file = tables_home / f"bpe_vocabulary_{COHORT or 'septic'}.csv"
    if bpe_file.is_file():
        frame = pl.read_csv(bpe_file, schema_overrides={"word": pl.String})
        out.append('<h3>Byte-pair encoding &mdash; learned words</h3>')
        out.append(
            "<p>States are letters, merged units are words, a trajectory is a "
            "sentence. BPE repeatedly merges the commonest adjacent pair, so "
            "the words are <b>learned rather than given a length</b>.</p>"
        )
        out.append(
            "<p>This differs from the motif vocabulary above in three ways. "
            "Motifs are every 2- to 4-gram, pruned by frequency, which is a "
            "<i>bag of overlapping substrings</i>; BPE produces a "
            "<i>segmentation</i>, so a sentence has word order and a unit can "
            "sit in prefix or suffix position. Nothing caps a word at four "
            "states. And long runs collapse: the motif vocabulary's top three "
            "entries were <code>00</code>, <code>000</code> and "
            "<code>0000</code> in 73&ndash;78% of patients &mdash; three words "
            "all meaning 'stayed off pressors' &mdash; where BPE merges them "
            "into one unit while keeping it distinct from <code>00</code>, so "
            "duration is compressed rather than exploded or, as under "
            "run-length encoding, discarded.</p>"
        )
        out.append(callout(
            "<b>The vocabulary is fitted on this cohort alone.</b> The same "
            "string therefore means something different in the other report, "
            "and the two vocabularies are not comparable &mdash; a word is "
            "whatever was frequent here. Merges never cross the terminal "
            "symbol, so no word encodes the end of a record; allowing that "
            "would let a word encode death and every comparison downstream "
            "would inherit the outcome it was meant to predict."
        ))
        out.append(table(
            frame.head(40),
            {
                "occurrences": thousands, "patients": thousands,
                "share_of_patients": pct,
            },
            key_column="word",
        ))

    context_file = tables_home / f"bpe_word_context_{COHORT or 'septic'}.csv"
    if context_file.is_file():
        out.append('<h3>How the words are used</h3>')
        out.append(
            "<p>The vocabulary says which units exist; this says how they "
            "are used. <b>Opens</b> and <b>closes</b> are the share of that "
            "word's own occurrences sitting first or last in a sentence, so "
            "a word that only ever opens reads 100% even when it is rare. "
            "The neighbour columns are the words that habitually come "
            "before and after.</p>"
        )
        out.append(
            "<p>This is the prefix and suffix question, and it only has an "
            "answer because BPE produces a segmentation &mdash; a bag of "
            "overlapping n-grams has no notion of what follows what. Share "
            "of patients is the honest denominator for how common a word is: "
            "occurrences double-count a patient who repeats one, and a word "
            "used eight times by one patient is not eight patients' worth of "
            "evidence.</p>"
        )
        out.append(table(
            pl.read_csv(context_file, schema_overrides={"word": pl.String}),
            {
                "share_of_patients": pct, "occurrences": thousands,
                "opens_sentence": pct, "closes_sentence": pct,
            },
            key_column="word",
        ))

    latent_file = tables_home / f"sentence_vs_latent_{COHORT or 'septic'}.csv"
    if latent_file.is_file():
        frame = pl.read_csv(latent_file)
        rho = frame["spearman_rho"][0] if "spearman_rho" in frame.columns else None
        out.append('<h3>Sentences against the latent space</h3>')
        out.append(
            "<p>Every patient has two descriptions of the same 48 hours: a BPE "
            "sentence over the MAP/pressor states, and a trajectory of nine "
            "1024-d embeddings. Pairs of patients are scored under both "
            "&mdash; normalised word-level edit distance, and Frobenius "
            "distance between the corrected trajectories &mdash; then split "
            "at the median of each."
            + (f" Spearman &rho; between the two distances is "
               f"<b>{rho:+.3f}</b>." if rho is not None else "")
            + "</p>"
        )
        out.append(
            "<p>Distances are taken in the <b>1024-d space, not PaCMAP</b>: a "
            "projection's distances are not metric, so 'close on the map' is "
            "not a claim about the representation. The window is 0&ndash;48&nbsp;h, "
            "the only stretch where the state grid and the embedding grid "
            "overlap.</p>"
        )
        out.append(callout(
            "<b>The off-diagonal quadrants are the result.</b> Pairs close "
            "under both measures, or far under both, say the model agrees with "
            "the state description. The other two say what the 15-state space "
            "is missing.<br><br>"
            "<i>Divergent sentence, close latent</i> &mdash; the model treats "
            "these patients as alike despite different MAP and pressor "
            "courses, so whatever they share is outside the state space. "
            "<i>Similar sentence, far latent</i> &mdash; identical course, and "
            "the model still separates them, so again the difference is "
            "outside it.<br><br>"
            "<b>Read the two right-hand columns before concluding anything.</b> "
            "Position in the projection explains R&sup2;&nbsp;0.45 of log "
            "stream length and site survives the correction at non-linear "
            "AUC&nbsp;0.855, so 'close in latent space' partly means "
            "'similarly documented, same hospital'. These are reported rather "
            "than adjusted away: a quadrant that is really a documentation "
            "artefact should be visible as one."
        ))
        out.append(table(
            frame.drop("spearman_rho") if "spearman_rho" in frame.columns else frame,
            {
                "pairs": thousands, "share": pct,
                "mean_sentence_distance": num(3),
                "mean_latent_distance": num(2),
                "mean_log_stream_gap": num(3),
                "same_site_share": pct,
            },
            key_column="quadrant",
        ))

    partition_file = tables_home / f"variance_partition_{COHORT or 'septic'}.csv"
    if partition_file.is_file():
        out.append('<h3>What each description contributes to the outcome</h3>')
        out.append(
            "<p>The sentence and the embedding are two readings of the same "
            "48 hours, so the question is not which predicts death but how "
            "much either one adds to the other. Three models are fitted "
            "&mdash; sentence, embedding, both &mdash; and their shares "
            "partitioned by commonality analysis.</p>"
        )
        out.append(callout(
            "<b>The partition is taken on pseudo-R&sup2;, not AUC.</b> AUC "
            "does not decompose additively, so unique and shared shares "
            "computed from it would not sum to the total. McFadden's "
            "pseudo-R&sup2; from the cross-validated log loss does, and the "
            "AUC column is carried alongside for reading only.<br><br>"
            "<b>Log stream length and site are in the sentence-side model, "
            "not in a footnote.</b> The embedding's unique share is therefore "
            "measured over and above documentation volume and hospital. "
            "Without that, position in the projection explains "
            "R&sup2;&nbsp;0.45 of log stream length, so part of what looked "
            "like physiology would be chart density.<br><br>"
            "<b>The pairing is not symmetric,</b> which is where the "
            "text-and-image analogy stops. The sentence is a deterministic "
            "function of a subset of what GEM already read: it consumed the "
            "MAP values and the pressor doses, and the sentence is those same "
            "facts thresholded. So 'can the embedding predict the sentence' "
            "is near-trivially yes; the informative direction is what each "
            "has left over, which the next two subsections take up."
        ))
        out.append(table(
            pl.read_csv(partition_file),
            {"auc": num(4), "pseudo_r2": num(4)},
            key_column="model",
        ))

    residual_file = tables_home / f"residual_tokens_{COHORT or 'septic'}.csv"
    if residual_file.is_file():
        frame = pl.read_csv(residual_file).head(20)
        out.append('<h3>What the embedding carries that the sentence does not</h3>')
        out.append(
            "<p>The embedding is regressed on the sentence and the residual "
            "kept &mdash; the part of where GEM placed the patient that their "
            "MAP and pressor course does not account for. That residual "
            "predicts death on its own, and it still does "
            "<b>within groups of patients whose sentences are character for "
            "character identical</b>, which is the strictest available "
            "version of same course, different outcome. Below: the tokens "
            "most associated with the residual direction that points towards "
            "death.</p>"
        )
        out.append(table(
            frame,
            {"patients_with_token": thousands, "residual_score_gap": num(3)},
            key_column="token",
        ))

    mirror_file = tables_home / f"sentence_residual_outcomes_{COHORT or 'septic'}.csv"
    words_file = tables_home / f"sentence_residual_words_{COHORT or 'septic'}.csv"
    site_file = tables_home / f"sentence_residual_site_{COHORT or 'septic'}.csv"
    directions_file = (
        tables_home / f"sentence_residual_directions_{COHORT or 'septic'}.csv"
    )
    if any(f.is_file() for f in (mirror_file, words_file, site_file)):
        out.append(
            '<h3>Why do similar embeddings diverge into different '
            'sentences?</h3>'
        )
        out.append(
            "<p>The subsection above runs <code>embedding ~ sentence</code>. "
            "This one runs the mirror, <code>sentence ~ g(embedding)</code>, "
            "so the residual is the part of the treatment course the "
            "embedding did <b>not</b> anticipate. Two patients the model "
            "places together whose courses then part company appear here as "
            "large residuals pointing opposite ways.</p>"
        )
        out.append(callout(
            "<b>None of the three variance components above answers this "
            "question.</b> Unique-to-sentence, unique-to-embedding and shared "
            "are all partitions of what predicts the <i>outcome</i>. A "
            "patient can sit entirely inside the shared component and still "
            "have been treated unusually for where the model placed them. The "
            "divergence has to be measured against a prediction of the "
            "sentence, which is what this builds.<br><br>"
            "<b>A large residual is not unwarranted variation.</b> Two "
            "readings survive every number below: the clinicians responded to "
            "something real that neither the 15-state sequence nor the "
            "embedding encoded, or the patient was treated differently for "
            "reasons that had nothing to do with them. Nothing in the "
            "divergence separates these, which is why the outcome table is "
            "here &mdash; if the divergence carries no outcome signal the "
            "first reading has little to stand on.<br><br>"
            "<b>Residuals are out of fold.</b> An in-sample residual from a "
            "50-component ridge is shrunk towards zero wherever the model "
            "interpolated, which would make 'unanticipated' partly a "
            "statement about fold membership."
        ))
        out.append(callout(
            "<b>This construction came out weak, and the weakness is the "
            "first result.</b> The expectation was that predicting the "
            "sentence from the embedding would be near-trivial, since GEM "
            "read the same MAP values and pressor doses the sentence "
            "thresholds. Out of fold the embedding anticipates only "
            "<b>5.7%</b> (septic) and <b>6.8%</b> (cardiogenic) of the "
            "sentence's variance: a 50-component linear map cannot say which "
            "of ~85 sparse BPE words a patient receives, even having read the "
            "underlying numbers.<br><br>"
            "That makes the residual nearly equal to the centred word counts "
            "themselves &mdash; 'unanticipated treatment course' and "
            "'treatment course' are almost the same vector. So "
            "<code>divergence_alone</code> below is <b>not</b> a measure of "
            "divergence; it is the sentence under another name, which is why "
            "it lands near the AUC the sentence reaches outright in the "
            "partition table above. <b>Read "
            "<code>increment_over_embedding</code> and nothing else.</b> The "
            "divergence-alone column is kept so that the collapse is visible "
            "rather than left to be inferred.",
            kind="warn",
        ))
        if words_file.is_file():
            frame = pl.read_csv(words_file, schema_overrides={"word": pl.String})
            shown = frame.filter(pl.col("share_of_patients") > 0.02).head(15)
            if not shown.is_empty():
                out.append('<h4>Which words went unanticipated</h4>')
                out.append(
                    "<p>Per-word out-of-fold R&sup2;, lowest first, among "
                    "words appearing in more than 2% of patients. "
                    "<code>direction</code> reads the dynamics field of the "
                    "states the word spells: a word is an escalation if more "
                    "of its letters are rising than falling.</p>"
                )
                out.append(table(
                    shown,
                    {"patients": thousands, "share_of_patients": pct,
                     "variance_anticipated": num(3),
                     "mean_residual_when_present": num(3)},
                    key_column="word",
                ))
        if directions_file.is_file():
            frame = pl.read_csv(directions_file)
            if not frame.is_empty():
                out.append(
                    "<p>Grouped by what the word spells, which asks whether "
                    "the failure is one-sided &mdash; a model that "
                    "anticipates weans but not escalations is failing at the "
                    "clinically consequential end.</p>"
                )
                out.append(table(
                    frame,
                    {"words": thousands,
                     "median_variance_anticipated": num(3),
                     "patient_weighted_variance_anticipated": num(3)},
                    key_column="direction",
                ))
        if site_file.is_file():
            frame = pl.read_csv(site_file)
            out.append('<h4>How much of the divergence is the hospital</h4>')
            out.append(
                "<p>Site is added to the predictor and the drop in residual "
                "measures the practice-variation share directly. The "
                "per-site rows ask the stronger question: can the hospital be "
                "read back <i>off</i> the divergence? If it can, the "
                "unanticipated part of the treatment course carries a "
                "practice signature whatever else it carries. The last row "
                "holds the absorbed share rather than an AUC.</p>"
            )
            out.append(table(
                frame,
                {"patients": thousands, "auc_from_divergence": num(4)},
                key_column="site",
            ))
        if mirror_file.is_file():
            frame = pl.read_csv(mirror_file)
            out.append('<h4>Does the divergence predict outcome</h4>')
            out.append(
                "<p>Four outcomes, each scored three ways. Only the "
                "<code>increment_over_embedding</code> column is evidence "
                "that <i>how</i> the patient was treated mattered beyond "
                "where the model placed them.</p>"
            )
            out.append(callout(
                "<b>Magnitude alone conflates treated unusually with "
                "sicker.</b> A patient placed in a crowded, mild region who "
                "then escalates has both a large residual and a bad outcome, "
                "and severity explains both &mdash; hence the "
                "embedding-alone column as the baseline rather than the base "
                "rate.<br><br>"
                "<b>Mortality is the primary and the only clean one.</b> "
                "ICU-free days to 28 replaces length of stay because dying on "
                "day two produces a stay of two days, which in a "
                "mortality-adjacent analysis inverts the result; death scores "
                "zero. New ventilation and new dialysis are counted only when "
                "they begin after +48h, past the window the sentence covers, "
                "but GEM read the RESP and CRRT tokens inside it, so some "
                "circularity survives &mdash; the "
                "<code>partly_circular</code> column marks them.<br><br>"
                "<b>Read <code>n_events</code> before the AUC.</b> New "
                "dialysis runs around 3% of the cohort, so a five-fold split "
                "leaves a few dozen events per fold and the estimate swings "
                "on a handful of patients. Outcomes with fewer than 50 events "
                "are omitted rather than reported with a caveat."
            ))
            out.append(table(
                frame,
                {"n": thousands, "n_events": thousands,
                 "embedding_alone": num(3), "divergence_alone": num(3),
                 "both": num(3), "increment_over_embedding": num(3)},
                key_column="outcome",
            ))


    fine_sites = tables_home / f"fine_state_site_lengths_{COHORT or 'septic'}.csv"
    fine_vocab = tables_home / f"fine_bpe_vocabulary_{COHORT or 'septic'}.csv"
    fine_coarse = tables_home / f"fine_versus_coarse_{COHORT or 'septic'}.csv"
    fine_hidden = tables_home / f"fine_hidden_transitions_{COHORT or 'septic'}.csv"
    if any(f.is_file() for f in (fine_sites, fine_vocab, fine_coarse)):
        out.append('<h3>Inside the 6-hour window</h3>')
        out.append(
            "<p>Everything above gives a patient <b>one letter per 6-hour "
            "window</b>, so a week is 28 letters and whatever happened "
            "between grid points is gone. A patient titrated up three times "
            "and then weaned between hour 12 and hour 18 carries the same "
            "letter as one left untouched, because both are summarised by "
            "whatever was running at hour 18. This subsection goes back to the "
            "raw CLIF tables and builds the finer sequence: within each "
            "window, as many letters as the patient had distinct states, "
            "closed by an end-of-window marker.</p>"
        )
        out.append(formula(
            "3_3_3_3A3_34646_6_6_6_",
            "|  |  |  |  | |",
            "window 1 holds one state; window 5 holds three, a rise then two falls",
        ))
        out.append(callout(
            "<b>The marker earns its place here and not above.</b> On the "
            "6-hour sequences one letter <i>is</i> one window, so a separator "
            "after every letter would make every letter its own word and "
            "block every merge &mdash; BPE would learn nothing. Here there are "
            "several letters per window, so <code>_</code> is a real boundary "
            "with something either side of it, and a BPE word becomes a "
            "<b>within-window manoeuvre</b> rather than an arbitrary "
            "span.<br><br>"
            "<b>Every word in this vocabulary is movement, by "
            "construction.</b> Runs are collapsed inside a window and merges "
            "cannot cross a boundary, so no word can carry the same state "
            "twice running. The coarse vocabulary's three commonest entries "
            "were <code>00</code>, <code>000</code> and <code>0000</code> "
            "&mdash; three separate words all meaning 'stayed off pressors'. "
            "None of them can exist here, which is most of the reason this is "
            "worth building."
        ))
        if fine_sites.is_file():
            frame = pl.read_csv(fine_sites)
            widest = (
                frame["widest_ratio"][0]
                if "widest_ratio" in frame.columns else None
            )
            out.append('<h4>Letters per patient by site</h4>')
            out.append(
                "<p><b>Read this table before the vocabulary.</b> If one "
                "hospital charts twice as many states per patient as another, "
                "the words BPE learns are partly a vocabulary of charting "
                "habits and every comparison below inherits it."
                + (f" The widest site-to-site ratio is <b>{widest:.2f}&times;</b>."
                   if widest is not None else "")
                + "</p>"
            )
            out.append(callout(
                "<b>Two guards make that ratio small, and a third confound is "
                "reported rather than fixed.</b> MAP is resampled hourly "
                "rather than read per measurement, because monitoring "
                "intensity is not a clinical decision &mdash; an arterial "
                "line gives a MAP a minute and a cuff gives a handful a day. "
                "Runs are then collapsed within each window, so a window's "
                "length counts <i>state changes</i> rather than samples. What "
                "remains is that dose changes <i>are</i> read as events, "
                "because a new infusion rate is a recorded action rather than "
                "an observation; a site that re-charts a running infusion "
                "hourly will still show more events than one that charts only "
                "genuine changes."
            ))
            out.append(table(
                frame.drop("widest_ratio")
                if "widest_ratio" in frame.columns else frame,
                {"patients": thousands, "median_letters": num(0),
                 "ratio_to_pooled": num(2),
                 "median_windows_with_a_change": num(0),
                 "median_max_letters_in_a_window": num(0)},
                key_column="site",
            ))
        if fine_coarse.is_file():
            frame = pl.read_csv(fine_coarse)
            out.append('<h4>What 6-hour resolution could not see</h4>')
            out.append(
                "<p>The coarse letter for a window is the state at the "
                "window's <b>end</b>, so wherever a window held more than one "
                "state the coarse version kept the last and dropped the rest. "
                "The two counts say different things: how often a window held "
                "any change at all, and how often the window <i>began</i> in a "
                "different state than the one it was labelled with &mdash; "
                "which is not a summary but a misreading.</p>"
            )
            out.append(table(
                frame,
                {"patients": thousands, "windows": thousands,
                 "windows_with_a_change": thousands,
                 "share_with_a_change": pct,
                 "windows_misread_by_coarse": thousands,
                 "share_misread_by_coarse": pct},
                key_column="patients",
            ))
        if fine_hidden.is_file():
            frame = pl.read_csv(fine_hidden).head(12)
            out.append(
                "<p>The transitions the coarse letter hides, commonest "
                "first. These are dominated by pressors coming off inside the "
                "window: the coarse sequence does record the wean, but "
                "timestamps it at the next grid point, up to six hours "
                "late.</p>"
            )
            out.append(table(
                frame,
                {"windows": thousands, "share_of_misread": pct},
                key_column="began_as",
            ))
        if fine_vocab.is_file():
            frame = pl.read_csv(
                fine_vocab, schema_overrides={"word": pl.String}
            ).filter(pl.col("length") > 1).head(20)
            if not frame.is_empty():
                out.append('<h4>The within-window vocabulary</h4>')
                out.append(
                    "<p>The commonest multi-state words, each one a manoeuvre "
                    "that fits inside a single 6-hour window. "
                    "<code>meaning</code> spells the states out in "
                    "order.</p>"
                )
                out.append(table(
                    frame,
                    {"occurrences": thousands, "patients": thousands,
                     "share_of_patients": pct},
                    key_column="word",
                ))


    ngram_file = tables_home / f"sentence_ngrams_fine_{COHORT or 'septic'}.csv"
    pattern_file = tables_home / f"sentence_patterns_fine_{COHORT or 'septic'}.csv"
    byclass_file = (
        tables_home / f"sentence_patterns_by_class_fine_{COHORT or 'septic'}.csv"
    )
    if any(f.is_file() for f in (ngram_file, pattern_file, byclass_file)):
        out.append('<h3>What order the words come in</h3>')
        out.append(
            "<p>The vocabulary says which units exist and the neighbour "
            "columns say what sits either side of each one. This asks the next "
            "question: which <b>ordered</b> patterns recur across patients, "
            "and which of them separate the latent classes.</p>"
        )
        out.append(callout(
            "<b>There is no such thing as the commonest sentence here.</b> "
            "Between 93% and 98% of these sentences occur exactly once, and "
            "the only common one is the degenerate 'never on pressors'. The "
            "distribution over whole sentences is almost flat and has nothing "
            "to report. Sub-sentences do, which is why this section mines "
            "patterns rather than ranking sentences.<br><br>"
            "<b>Two views, kept apart.</b> Word n-grams are contiguous &mdash; "
            "strictly what comes next. Sequential patterns allow gaps, so a "
            "wean followed three days later by coming off pressors counts as "
            "one story whatever sat between; a trigram can only express that "
            "if the gap is exactly zero.<br><br>"
            "<b>Only CLOSED patterns are kept, and without that the output is "
            "unreadable.</b> A pattern is closed when no longer pattern "
            "containing it has the same support. The raw enumeration returns "
            "<code>6_ 6_ 6_</code>, then <code>6_ 6_ 6_ 6_</code>, then "
            "<code>6_ 6_ 6_ 6_ 6_</code> as three findings carried by nearly "
            "the same patients.<br><br>"
            "<b>Support is per patient, not per occurrence.</b> A pattern "
            "appearing eight times in one week is one patient's worth of "
            "evidence; counting occurrences would rank the vocabulary by "
            "whoever stayed longest."
        ))
        out.append(callout(
            "<b>The coarse sentences yield nothing, and that is reported "
            "rather than hidden.</b> At a median of 4 words and 93% unique, "
            "<b>no gapped pattern cleared a 5% support floor at all</b> and "
            "the contiguous n-grams barely cleared 1%. Everything in this "
            "subsection is from the fine-grained sequences, where the median "
            "sentence runs 31 words. Order-aware mining needs somewhere to "
            "put the gap.",
            kind="warn",
        ))
        if pattern_file.is_file():
            frame = pl.read_csv(
                pattern_file, schema_overrides={"pattern": pl.String}
            )
            moves = frame.filter(~pl.col("is_a_run")).head(15)
            if not moves.is_empty():
                out.append('<h4>Patterns involving a change of word</h4>')
                out.append(
                    "<p>Runs of a single repeated word are reported "
                    "separately below: <code>0_ 0_ 0_</code> is three "
                    "consecutive windows off pressors, which is true of three "
                    "quarters of the cohort and is not a sequence of care in "
                    "any interesting sense. These are the patterns where the "
                    "word actually changes.</p>"
                )
                out.append(table(
                    moves.drop("contiguous", strict=False),
                    {"patients": thousands, "share_of_patients": pct},
                    key_column="pattern",
                ))
            runs = frame.filter(pl.col("is_a_run")).head(6)
            if not runs.is_empty():
                out.append(table(
                    runs.drop("contiguous", strict=False),
                    {"patients": thousands, "share_of_patients": pct},
                    key_column="pattern",
                ))
        if byclass_file.is_file():
            frame = pl.read_csv(
                byclass_file, schema_overrides={"pattern": pl.String}
            ).filter(pl.col("p_adjusted") < 0.05)
            if not frame.is_empty():
                out.append('<h4>Which patterns separate the latent classes</h4>')
                out.append(
                    "<p>Each pattern tested as a per-patient presence flag, "
                    "one class against all others pooled. Small cells "
                    "suppressed at 11 and Bonferroni over every (pattern, "
                    "class) test. <b>The latent classes are the only "
                    "clustering used here</b> &mdash; this is not a "
                    "comparison of clustering methods.</p>"
                )
                out.append(callout(
                    "<b>The number of significant tests overstates the number "
                    "of findings.</b> Closed-pattern filtering removes "
                    "support-identical duplicates, but patterns whose support "
                    "differs slightly survive as separate rows and say much "
                    "the same thing &mdash; a class enriched for "
                    "<code>6_ 6_ 6_ 646_</code> is usually also enriched for a "
                    "dozen near-variants. Read the table as a handful of "
                    "stories told repeatedly, not as hundreds of independent "
                    "results."
                ))
                shown = []
                for cluster in sorted(frame["cluster"].unique().to_list()):
                    part = frame.filter(pl.col("cluster") == cluster)
                    shown.append(
                        part.sort("log2_ratio", descending=True).head(5)
                    )
                    shown.append(part.sort("log2_ratio").head(3))
                out.append(table(
                    pl.concat(shown).unique(subset=["cluster", "pattern"])
                    .sort(["cluster", "log2_ratio"], descending=[False, True]),
                    {"patients_in_cluster_with_pattern": thousands,
                     "prevalence_in_cluster": pct,
                     "prevalence_elsewhere": pct,
                     "log2_ratio": num(2), "p": num(5), "p_adjusted": num(4)},
                    key_column="cluster",
                ))
                out.append(callout(
                    "<b>Do not read the hypotension patterns as "
                    "undertreatment.</b> The pattern that most distinguishes "
                    "the highest-mortality class in each cohort is a run of "
                    "<code>7_</code> &mdash; MAP below 65 with <b>no "
                    "vasopressor running</b>, sustained over several windows. "
                    "Three readings produce that string and this analysis "
                    "cannot separate them:<br><br>"
                    "&bull; <b>Withdrawal of support.</b> A patient "
                    "transitioned to comfort care has a low MAP and no "
                    "pressors. Given these classes carry 56% and 67% "
                    "mortality, this is very likely the dominant reading, and "
                    "it would make the pattern a marker of end-of-life care "
                    "rather than of anything done wrong.<br>"
                    "&bull; <b>Genuinely unsupported hypotension</b> &mdash; "
                    "the clinically interesting reading, and the one that "
                    "cannot be claimed without knowing the code status.<br>"
                    "&bull; <b>The hysteresis band.</b> MAP must exceed 65 to "
                    "leave the low state, so a patient recovering to 60 stays "
                    "labelled low until they clear 65.<br><br>"
                    "Code status and treatment-limitation orders would settle "
                    "it and are not in this extract. Until they are, the "
                    "<code>7_</code> runs should be read as <i>a marker of "
                    "dying</i>, not as a target.",
                    kind="warn",
                ))
                out.append(
                    "<p>Reading the symbols: <code>0</code> MAP&gt;65 off "
                    "pressors, <code>3</code> MAP&gt;65 norepinephrine alone "
                    "steady, <code>6</code> MAP&gt;65 norepinephrine plus "
                    "another agent steady, <code>7</code> MAP&lt;65 off "
                    "pressors, <code>D</code> MAP&lt;65 on a combination. A "
                    "multi-character word is movement inside one window "
                    "(<code>646_</code> is combination steady, up, steady, "
                    "closing the window); a trailing <code>_</code> means the "
                    "word closes a 6-hour window. The full 15-state legend is "
                    "at the top of this section.</p>"
                )


    wean_before = tables_home / f"wean_preceding_words_{COHORT or 'septic'}.csv"
    wean_family = tables_home / f"wean_by_family_{COHORT or 'septic'}.csv"
    wean_split = tables_home / f"wean_held_vs_restarted_{COHORT or 'septic'}.csv"
    wean_class = tables_home / f"wean_restart_by_growth_cluster_{COHORT or 'septic'}.csv"
    wean_site = tables_home / f"wean_restart_by_site_{COHORT or 'septic'}.csv"
    if any(f.is_file() for f in (wean_before, wean_family, wean_split)):
        out.append('<h3>When a wean holds, and when the pressors go back on</h3>')
        out.append(
            "<p>Reading the vocabulary forward: what comes <i>before</i> a "
            "patient is off pressors, and once a wean has happened, what "
            "decides whether it lasts.</p>"
        )
        if wean_before.is_file():
            frame = pl.read_csv(
                wean_before, schema_overrides={"preceding_word": pl.String}
            ).head(12)
            out.append(
                "<p><b>94% of <code>0_</code> words follow another "
                "<code>0_</code></b> &mdash; the patient was already off "
                "pressors and nothing happened. The weaning words are the "
                "remainder, and the commonest is <code>30_</code>: "
                "norepinephrine steady, then off, inside six hours, with no "
                "documented taper. It outnumbers the tapered "
                "<code>3230_</code> two to three times.</p>"
            )
            out.append(table(
                frame,
                {"occurrences": thousands, "share": pct},
                key_column="preceding_word",
            ))
        out.append(callout(
            "<b>A fixed 24-hour lookahead, not 'did it ever fail'.</b> The "
            "obvious definition &mdash; no pressor anywhere in the rest of the "
            "sentence &mdash; is dominated by how much sentence was left. A "
            "wean at hour 150 has four windows in which to fail and one at "
            "hour 12 has twenty-five, so 'held' would mostly mean 'weaned "
            "late'. Every wean is scored over the same horizon, and weans "
            "without a full 24 hours of record after them are <b>dropped, not "
            "counted as successes</b> &mdash; a patient whose record stops two "
            "windows later has no subsequent pressor word, and reading that as "
            "a durable wean would count dying as recovering.<br><br>"
            "<b>The unit is the wean, not the patient.</b> A patient weaned "
            "four times contributes four rows, and patients weaned repeatedly "
            "are not a random sample. The median patient here contributes two."
        ))
        if wean_family.is_file():
            out.append('<h4>Abrupt stops against documented tapers</h4>')
            out.append(
                "<p>Families are matched on the word with its window marker "
                "<b>stripped</b>: <code>30</code> and <code>30_</code> are one "
                "clinical event, differing only in whether the 6-hour window "
                "happened to end at the wean. An earlier version matched the "
                "full word and so filed <code>30</code> &mdash; two thirds as "
                "common as <code>30_</code> &mdash; under 'other', which both "
                "lost a third of the abrupt weans and gave 'other' a restart "
                "rate that was mostly an artefact.</p>"
            )
            out.append(table(
                pl.read_csv(wean_family),
                {"weans": thousands, "patients": thousands,
                 "closed_the_window": pct, "restarted_within_24h": pct,
                 "median_hours_from_onset": num(0),
                 "median_consecutive_0_after": num(0)},
                key_column="family",
            ))
            out.append(callout(
                "<b>Most weans do not hold 24 hours</b> &mdash; the restart "
                "rate runs around 50&ndash;60% in every family. Tapering is "
                "associated with a few points fewer restarts for "
                "norepinephrine alone and with nothing at all for "
                "combinations, which is a weak and inconsistent signal.<br><br>"
                "<b>Do not read it as evidence that tapering works.</b> Weaning "
                "is not randomised: a clinician walks the dose down on the "
                "patient they are worried about, so the taper arm is selected "
                "for concern and the comparison is confounded by indication in "
                "the direction that would hide a benefit. The honest reading "
                "is that this design cannot separate the effect of the taper "
                "from the reason for it.<br><br>"
                "<b>'Restarted inside the window' is definitional.</b> Those "
                "words come off pressors and go back on within the same six "
                "hours, so their high restart rate restates their own "
                "spelling. They are listed to account for every wean, not as "
                "a finding.",
                kind="warn",
            ))
        if wean_split.is_file():
            out.append(
                '<h4>Within one word: what separates the weans that held</h4>'
            )
            out.append(
                "<p>Every occurrence of <code>30</code> &mdash; the commonest "
                "wean &mdash; split by whether pressors went back on. "
                "Standardised differences, so the columns are comparable "
                "across measures on different scales.</p>"
            )
            out.append(table(
                pl.read_csv(wean_split),
                {"held_median": num(3), "restarted_median": num(3),
                 "standardised_difference": num(3)},
                key_column="measure",
            ))
            out.append(callout(
                "<b>The physiology at the moment of weaning barely moves.</b> "
                "MAP is the measure a bedside decision would lean on and its "
                "standardised difference is under 0.1 in both cohorts &mdash; "
                "the patients whose weans failed looked, at that moment, like "
                "the patients whose weans held. Residual dose and timing carry "
                "a little (weans that fail happen earlier and from slightly "
                "higher doses), but nothing here is of a size that would "
                "support a bedside rule."
            ))
        if wean_class.is_file():
            out.append(table(
                pl.read_csv(wean_class),
                {"weans": thousands, "restarted": pct},
                key_column="growth_cluster",
            ))
            out.append(callout(
                "<b>The latent class separates restart rates far more than any "
                "measurement does &mdash; and it is partly circular.</b> The "
                "classes come from latent class growth over the whole seven "
                "days, and the restart being predicted happens inside that "
                "window. So a class is defined in part by having a trajectory "
                "with these restarts in it. This is a <i>redescription</i> of "
                "the classes, not a predictor that could be read before the "
                "fact, and the gap between the best and worst class should be "
                "read as confirming that the classes capture pressor "
                "instability, not as a way to forecast a wean.",
                kind="warn",
            ))
        if wean_site.is_file():
            out.append(table(
                pl.read_csv(wean_site),
                {"weans": thousands, "restarted": pct},
                key_column="site",
            ))
            out.append(
                "<p>The per-site spread is the one comparison here that is "
                "<b>not</b> circular, since site is fixed before any of this "
                "happens. It is also small &mdash; a few percentage points "
                "&mdash; and these cohorts differ in case mix by site, so it "
                "is reported as a description rather than as practice "
                "variation.</p>"
            )


    wean_map = figures_home / f"wean_embedding_{COHORT or 'septic'}.png"
    wean_stats = tables_home / f"wean_embedding_{COHORT or 'septic'}.csv"
    wean_cov = tables_home / f"wean_embedding_covariates_{COHORT or 'septic'}.csv"
    if wean_stats.is_file() or wean_map.is_file():
        out.append('<h4>Where the weans sit in the embedding</h4>')
        out.append(
            "<p>Two panels: every wean placed against the whole grid, and the "
            "same weans tiled into hexagons coloured by how often the wean "
            "failed. The palette diverges about the <b>pooled</b> restart "
            "rate rather than about 50%, since roughly half of all weans "
            "restart and a palette centred on a half would call the average "
            "cell high or low purely from the base rate.</p>"
        )
        shown = figure(wean_map, "")
        if shown:
            out.append(shown)
        out.append(callout(
            "<b>Only weans inside the embedding window can be placed.</b> The "
            "grid runs &minus;24h to +48h; the wean sequences run the full "
            "week. About two thirds are placed, and the third that are not "
            "sit at a median 84h against 12&ndash;18h for those kept &mdash; "
            "so this describes <b>early weaning specifically</b>, not weaning. "
            "The dropped weans restart at almost exactly the same rate as the "
            "kept ones, so nothing is being selected on the outcome, only on "
            "timing."
        ))
        if wean_stats.is_file():
            out.append(
                "<p>Clustering is tested rather than eyeballed. A hexagon map "
                "of a rate always looks patchy: with a few hundred events over "
                "sixty-odd cells the rates vary by sampling alone. The "
                "permutation test asks whether the map is patchier than "
                "redrawing each cell's failures from the pooled rate would "
                "make it; Moran's I asks the sharper question of whether "
                "high-restart cells sit <i>next to</i> other high-restart "
                "cells, since a map can be over-dispersed without being "
                "spatially organised.</p>"
            )
            out.append(table(
                pl.read_csv(wean_stats),
                {"weans_placed": thousands, "weans_total": thousands,
                 "hexagons": thousands, "pooled_restart": pct,
                 "rate_min": pct, "rate_max": pct,
                 "weighted_variance": num(5), "weighted_variance_null": num(5),
                 "p_patchiness": num(4), "morans_i": num(4),
                 "p_morans_i": num(4)},
                key_column="hexagons",
            ))
        if wean_cov.is_file():
            out.append(
                "<p>And what those regions are made of &mdash; each covariate "
                "averaged per hexagon and correlated against the restart "
                "rate, every one binned on the <b>same</b> point set so the "
                "cells line up.</p>"
            )
            out.append(table(
                pl.read_csv(wean_cov),
                {"hexagons": thousands,
                 "correlation_with_restart_rate": num(3)},
                key_column="covariate",
            ))
            out.append(callout(
                "<b>The regions are real, and they are not news.</b> "
                "Patchiness runs six times the null and Moran's I lands near "
                "+0.54, both p&nbsp;&lt;&nbsp;0.0001 &mdash; weans genuinely "
                "cluster by outcome in this space. But the high-restart "
                "regions are the <i>early, sicker, more heavily charted</i> "
                "part of the map: the rate correlates around &minus;0.5 to "
                "&minus;0.65 with hours from onset, &minus;0.25 to &minus;0.55 "
                "with log stream length, and +0.32 to +0.61 with mortality. "
                "Those are the axes this projection was already known to "
                "encode, so there is no basis here for saying the embedding "
                "knows something specific about weaning.<br><br>"
                "<b>Do not read these correlations against the effect sizes "
                "above.</b> These are <b>ecological</b> correlations over "
                "~65 hexagons; the standardised differences in the previous "
                "subsection are per patient. MAP separates held from "
                "restarted at SMD&nbsp;+0.002 <i>per patient</i> while "
                "correlating +0.25 with the restart rate <i>per hexagon</i>, "
                "and those do not conflict. Aggregating averages away the "
                "within-cell noise and leaves the between-region component, "
                "which inflates the correlation; an r of &minus;0.65 across "
                "cells is compatible with no association across patients. The "
                "aggregate numbers say what the map is organised by. They say "
                "nothing about what predicts an individual wean, and reading "
                "them as if they did is the ecological fallacy.",
                kind="warn",
            ))

    stability_file = tables_home / f"bpe_stability_{COHORT or 'septic'}.csv"
    if stability_file.is_file():
        out.append('<h3>How much the vocabulary depends on the sample size</h3>')
        out.append(
            "<p>BPE refitted on nested subsamples, each compared against the "
            "full-cohort fit. Agreement is the longest common <b>prefix</b> "
            "of the two merge lists, because BPE is sequential: merge "
            "<i>k</i> is chosen from a corpus already rewritten by merges "
            "1&hellip;<i>k</i>&minus;1, so once two fits diverge everything "
            "after is incomparable even where individual pairs coincide. The "
            "<code>merges_in_common_any_order</code> column ignores order "
            "entirely.</p>"
        )
        out.append(
            "<p><b><code>sentences_segmented_identically</code> is the column "
            "to read.</b> Nothing downstream consumes the merge list; "
            "everything consumes the segmentation. Both merge lists are "
            "applied to the same held-out sequences and compared on the words "
            "they produce, so two fits that found the same merges in a "
            "different order score as agreeing &mdash; which they should, "
            "because they segment the same patient the same way. A low "
            "prefix share next to a high segmentation agreement is evidence "
            "about <i>order</i>, not about content: once "
            "<code>0</code>+<code>0</code> becomes <code>00</code>, the "
            "candidates <code>00</code>+<code>00</code>, "
            "<code>00</code>+<code>0</code> and <code>0</code>+<code>00</code> "
            "sit within a few counts of each other, and a small sample flips "
            "their order without changing what the vocabulary contains.</p>"
        )
        out.append(callout(
            "<b>This was run before deciding whether to pool the cohorts.</b> "
            "Fitting BPE on all shock would roughly double the corpus, at the "
            "cost of a vocabulary belonging to neither cohort &mdash; the "
            "same string would stop meaning the same thing in the two "
            "reports. The curve says what that trade buys: if agreement is "
            "already high at a few thousand sequences, pooling buys "
            "stability that was there to begin with."
        ))
        out.append(table(
            pl.read_csv(stability_file),
            {"sequences": thousands, "merges_learned": thousands,
             "identical_prefix": thousands, "prefix_share": pct,
             "merges_in_common_any_order": thousands,
             "sentences_segmented_identically": pct, "word_overlap": pct},
            key_column="sequences",
        ))

    if septic_home.is_dir():
        state_panels = [
            (label, figure(septic_home / name, note))
            for name, label, note in (
                ("trajectory_composition_state_hex_enrichment.png",
                 "NE alone vs NE+other",
                 "MAP crossed with whether norepinephrine runs alone "
                 "or with another agent; dose trend dropped"),
                ("trajectory_rich_state_panels.png",
                 "All 15 states, one at a time",
                 "Every dimension at once -- a small multiple needs no "
                 "palette that separates 15 hues"),
                ("trajectory_coarse_state_hex_enrichment.png",
                 "MAP x trend, enrichment",
                 "Collapsed to MAP x trend; composition dropped, since 15 "
                 "categorical hues is past where a palette separates"),
                ("trajectory_coarse_state_hex_plurality.png",
                 "Hexagons, plurality", ""),
                ("trajectory_coarse_state_hex_enrichment_noflow.png",
                 "Enrichment, no flow", ""),
                ("trajectory_coarse_state_panels.png", "One state at a time", ""),
            )
        ]
        state_panels = [(l, b) for l, b in state_panels if b]
        if state_panels:
            out.append('<h3>Where each state sits in the embedding</h3>')
            out.append(tabs(state_panels))

        motif_views = []
        for column, stem, label in (
            ("motif_cluster_rle", "motif_rle", "Collapsed-sequence clusters"),
            ("motif_cluster_raw", "motif_raw", "Raw-sequence clusters"),
        ):
            panes = [
                figure(septic_home / f"trajectory_{stem}_hex_enrichment.png", ""),
                figure(septic_home / f"trajectory_{column}_panels.png", ""),
                figure(septic_home / f"trajectory_flow_overlay_{stem}.png", ""),
            ]
            panes = [pane for pane in panes if pane]
            if panes:
                motif_views.append((label, "".join(panes)))
        if motif_views:
            out.append('<h3>Where the motif clusters live</h3>')
            out.append(
                "<p>Hexagon map, one cluster at a time, and the per-cluster "
                "flow fields overlaid &mdash; the same three views the latent "
                "classes get in section 7, so the two clusterings can be "
                "compared panel for panel.</p>"
            )
            out.append(tabs(motif_views))

    state_dag = figure(
        figures_home / f"longitudinal_state_dag_{COHORT or 'septic'}.png", "")
    if state_dag:
        out.append('<h3>Each motif cluster as a path through the states</h3>')
        out.append(
            "<p>The same circular layout as the NMF factor graph in section 7, "
            "with the 15 MAP/pressor states as nodes ordered by the mean "
            "window at which each dominates. Distance on the page means "
            "nothing; only the edges do.</p>"
        )
        out.append(state_dag)

    patient_states = [
        (f"Patient {index}", figure(
            figures_home / f"patient_state_paths_{COHORT or 'septic'}"
            / f"patient_state_path_{index:02d}.png", ""))
        for index in range(1, 13)
    ]
    patient_overlays = [
        (f"Patient {index}", figure(
            septic_home / f"paths_over_motif_{index:02d}.png", ""))
        if septic_home.is_dir() else (f"Patient {index}", "")
        for index in range(1, 13)
    ]
    combined = []
    for (label, route), (_, overlay) in zip(patient_states, patient_overlays):
        panes = [pane for pane in (overlay, route) if pane]
        if panes:
            combined.append((label, "".join(panes)))
    if combined:
        out.append('<h3>Individual patients in this state space</h3>')
        out.append(
            "<p>Each tab is one patient: their path over the motif-cluster "
            "flow fields, and their own route through the 15 states. Node "
            "fill is the state's colour, the outline runs white at the "
            "earliest window to black at the latest, and the hour sits beside "
            "each node.</p>"
        )
        out.append(tabs(combined))


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

    # Placed first in the section on purpose: it is a check on the projection
    # itself, and every panel below inherits whatever it shows.
    stream_panels = [
        (label.capitalize(),
         figure(info["home"] / "trajectory_stream_length.png", ""))
        for label, info in homes.items()
    ]
    stream_panels = [(l, b) for l, b in stream_panels if b]
    if stream_panels:
        out.append('<h3>Event-stream length &mdash; a check on the geometry</h3>')
        out.append(
            "<p>Each point coloured by <b>how many tokens the model actually "
            "read</b> at that grid point. This is a diagnostic rather than a "
            "label: if the projection organises by stream length, then its "
            "geometry is partly <i>how much record exists</i> rather than "
            "clinical state, and every other panel in this section inherits "
            "that.</p>"
        )
        out.append(
            "<p>Worth checking because the spread is enormous &mdash; the "
            "prefix runs from 125 tokens at the 5th percentile to 25,379 at "
            "the maximum, with a median near 900, and the model's "
            "4096&#8209;token context binds for 3.8% of grid points. Colours "
            "are quantile-ranked with the bar labelled in real token counts; a "
            "linear ramp would put almost every patient in the first few "
            "percent of the scale.</p>"
        )
        out.append(callout(
            "<b>Compare the uncorrected and corrected tabs.</b> The affine "
            "site correction is fitted to remove <i>site</i>, not stream "
            "length. If a length gradient survives it, that gradient is not a "
            "site artefact and cannot be corrected away by this route &mdash; "
            "it is the model encoding record volume, and any cluster that "
            "aligns with it is partly a cluster of documentation density. "
            "Read it against the care-state panels in particular: factor&nbsp;1 "
            "is <code>UNK</code> and factor&nbsp;10 is lab ordering, both of "
            "which scale with how much was written down."
        ))
        out.append(tabs(stream_panels))

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

    hex_panels = [
        (label, figure(septic_home / f"trajectory_perfusion_{tag}.png", note))
        for tag, label, note in (
            ("hex_plurality", "Plurality",
             "Each hexagon coloured by its most common state"),
            ("hex_plurality_noflow", "Plurality, no flow",
             "The same tiling without streamlines competing for the ink"),
            ("hex_enrichment", "Enrichment",
             "Each hexagon coloured by the state most concentrated there "
             "relative to that state's own grand total"),
            ("hex_enrichment_noflow", "Enrichment, no flow", ""),
            ("panels", "One state at a time",
             "Each state in colour against the whole cloud in grey"),
        )
    ]
    hex_panels = [(label, body) for label, body in hex_panels if body]
    if hex_panels:
        out.append('<h3>Where each perfusion state sits</h3>')
        out.append(
            "<p>Three views of the same four states, because they disagree in "
            "a way that is itself informative. <b>Plurality</b> gives each "
            "hexagon to its most common state &mdash; it reads directly, but a "
            "state holding 46% of timepoints wins hexagons where it is merely "
            "present. <b>Enrichment</b> gives each hexagon to the state most "
            "concentrated there relative to its own total, so group size "
            "cancels entirely. The <b>small multiples</b> do no binning, no "
            "voting and no normalisation at all, and are the check on both.</p>"
        )
        out.append(callout(
            "<b>The two hexagon modes differ by 5.6&times; for the "
            "hypotensive states.</b> MAP&nbsp;&lt;&nbsp;65 with and without "
            "pressors together hold 10.5% of timepoints, and win <b>19 of 312 "
            "hexagons (6%) under plurality</b> against <b>106 of 312 (34%) "
            "under enrichment</b>. Neither is wrong. Plurality is right that "
            "most regions genuinely contain more well-perfused timepoints; "
            "enrichment is right that the hypotensive ones are concentrated "
            "rather than scattered. Read them together, and read the small "
            "multiples when they disagree."
        ))
        out.append(
            "<p>A further reading of the enrichment panel, which is easy to "
            "miss: with four states over 312 hexagons, states with identical "
            "spatial structure would win about 78 each. The observed split is "
            "105 / 101 / 60 / 46, so <b>the well-perfused states win more than "
            "their share even after size is cancelled</b> &mdash; they occupy "
            "coherent regions, while the hypotensive timepoints are spread "
            "more thinly across the embedding. That is a statement about "
            "hypotension co-occurring with many different clinical pictures "
            "rather than marking out a region of its own.</p>"
        )
        out.append(
            "<p>Hexagons below ten points are grey rather than awarded to "
            "anyone, and opacity carries the margin of the win, so a hexagon "
            "taken 31%&ndash;30% is visibly fainter than one taken "
            "90%&ndash;5%. Flow lines are on all three.</p>"
        )
        out.append(tabs(hex_panels))

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

    out.append('<h2 id="sites">10 &middot; Why the sites separate</h2>')
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

    out.append('<h2 id="models">11 &middot; Models</h2>')
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
        '<h2 id="undifferentiated">12 &middot; Undifferentiated shock</h2>'
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


def _auc_cell(row) -> str:
    """an AUC with its bootstrap interval, as one cell"""
    if row.get("auc") is None:
        return "—"
    return f"{row['auc']:.3f} ({row['low']:.2f}\u2013{row['high']:.2f})"


def _probe_table(frame: pl.DataFrame, section: str, hour=None) -> pl.DataFrame:
    """one row per handbook, one column per probe, AUC with interval"""
    part = frame.filter(pl.col("section") == section)
    if hour is not None:
        part = part.filter(pl.col("hour") == hour)
    if part.height == 0:
        return pl.DataFrame()
    labels = {
        "embedding->mortality": "GEM embedding",
        "covariates->mortality": "age, SOFA, lactate",
        "tokens->mortality": "token counts",
        "embedding->group": "GEM \u2192 response group",
        "read_hour->mortality": "read hour alone",
        "embedding_difference->who_dies": "GEM difference, within pair",
    }
    rows = {}
    for row in part.iter_rows(named=True):
        key = int(row["handbook"])
        rows.setdefault(key, {"handbook": key, "patients": 0})
        # the LARGEST probe's n. The probes do not all run on the same
        # patients -- anything predicting mortality drops those whose outcome
        # is unknown -- so one number cannot describe the row and the biggest
        # is the only one that is not quietly an undercount of some cell.
        rows[key]["patients"] = max(rows[key]["patients"], int(row["n"]))
        rows[key][labels.get(row["probe"], row["probe"])] = _auc_cell(row)
    ordered = [rows[k] for k in sorted(rows)]
    columns = ["handbook", "patients"] + [
        labels[k] for k in (
            "read_hour->mortality", "covariates->mortality",
            "tokens->mortality", "embedding->mortality",
            "embedding_difference->who_dies", "embedding->group",
        ) if any(labels[k] in row for row in ordered)
    ]
    return pl.DataFrame(
        [{c: row.get(c, "—") for c in columns} for row in ordered],
        infer_schema_length=None,
    )


def section_handbooks(paths, out: list) -> None:
    """
    the treatment handbooks, who is in them, and who diverges inside them

    Downstream of the state-space section and dependent on it: a handbook is
    a course through the treatment layer and a response group is a course
    through the response layer, so neither means anything until both layers
    exist. Kept separate because the question is different. The state space
    asks what states there are; this asks whether two patients given the same
    plan do the same thing, which is the only version of the question that
    holds management fixed.
    """
    cohort = COHORT or "cardiogenic"
    # CALLOUTS THAT QUOTE SPECIFIC NUMBERS WERE WRITTEN AGAINST THE
    # CARDIOGENIC COHORT and are not re-derived per cohort. Printing them
    # under a septic heading would attribute cardiogenic counts, medoids
    # and hours to septic patients, which is worse than saying less -- so
    # they are dropped elsewhere, where the tables and figures below carry
    # the content unaided.
    cardiogenic_only = (
        (lambda html: html) if cohort == "cardiogenic"
        else (lambda html: "")
    )
    tables_home = paths.output_home / "tables" / "GEM-all"
    figures_home = paths.output_home / "figures" / "GEM-all"
    cat_file = tables_home / f"handbook_catalogue_{cohort}.csv"
    if not cat_file.is_file():
        return

    out.append(
        '<h2 id="handbooks">9 &middot; Treatment handbooks and who diverges '
        "inside them</h2>"
    )
    out.append(cardiogenic_only(callout(
        "<b>A handbook is an alignment, not a string match.</b> Collapsed "
        "treatment courses are flicker-dominated &mdash; <code>0</code> covers "
        "2,114 patients, <code>10</code> another 653, <code>010</code> 315 "
        "&mdash; and only 21 distinct courses reach 25 patients at a 48-hour "
        "cap. Exact matching therefore finds either everybody or nobody, and "
        "defines nothing in between. Each handbook here is a <b>medoid</b> "
        "course, and a patient belongs to it if their own course aligns "
        "against that medoid above a floor of 0.25 of the achievable score. "
        "That keeps 72% of patients; the remaining 28% are left unassigned "
        "rather than forced into the nearest bin.<br><br>"
        "<b>Patients are not randomised to handbooks.</b> A clinician reaches "
        "'high norepinephrine plus vasopressin' by reading a patient who was "
        "not responding, so the handbook is selected on the very thing the "
        "response groups then compare. Holding the plan fixed removes the "
        "dose confound from the measurement. It does not make any comparison "
        "below causal, and no amount of stratification will."
    )))

    cat = pl.read_csv(cat_file)
    # Counted, not asserted. The heading said "six" and the cohort it was
    # written for reports five, because one handbook fell below the 120-patient
    # floor for being split into response groups and is not in this table.
    spread = float(cat["mortality"].max() / max(float(cat["mortality"].min()),
                                                1e-9))
    lactate = cat["median_lactate"]
    sofa = cat["median_sofa_non_cv"]
    out.append(f"<h3>The {cat.height} handbooks</h3>")
    out.append(
        f"<p>Mortality runs {spread:.0f}-fold across the handbooks. "
        "Non-cardiovascular SOFA barely moves across them &mdash; "
        f"{float(sofa.min()):.1f} to {float(sofa.max()):.1f} &mdash; but "
        f"median lactate runs {float(lactate.min()):.1f} to "
        f"{float(lactate.max()):.1f} mmol/L. <b>An earlier version of this "
        "paragraph cited the flat SOFA alone and read as though the handbooks "
        "were severity-matched. On the lactate column they plainly are "
        "not</b> &mdash; the ordering is substantially a perfusion-severity "
        "ordering, and it is only SOFA that is flat. "
        "<code>response_gap</code> is "
        "the difference in mortality between the two response groups "
        "<i>inside</i> each handbook &mdash; the part the plan does not "
        "explain &mdash; and it is the column this whole section exists "
        "for.</p>"
    )
    out.append(table(
        cat,
        numeric_format={
            "patients": thousands,
            "median_age": num(0),
            "median_sofa_non_cv": num(1),
            "median_lactate": num(1),
            "median_peak_ne": num(2),
            "median_pressor_hours": num(0),
            "median_hours": num(0),
            "mortality": pct,
            "better_group_n": thousands,
            "better_group_mortality": pct,
            "worse_group_n": thousands,
            "worse_group_mortality": pct,
            "response_gap": pct,
        },
        key_column="handbook",
    ))

    # ---- sequence logos: what a handbook's course generally looks like
    consensus_file = tables_home / f"handbook_consensus_{cohort}.csv"
    logo_panels = [
        (
            f"{layer.capitalize()} ({'bits' if not tag else 'share'})",
            figure(
                figures_home / f"handbook_logo_{layer}{tag}_{cohort}.png",
                f"{layer} course per handbook, "
                + ("letter heights scaled by information content"
                   if not tag else "every column drawn full height"),
            ),
        )
        for layer in ("treatment", "response")
        for tag in ("", "_share")
    ]
    logo_body = tabs(logo_panels)
    if logo_body:
        out.append("<h3>What each handbook&rsquo;s course generally looks "
                   "like</h3>")
        out.append(
            "<p>The aligned-pair renderings further down show six patients "
            "per handbook. These show all of them, as a <b>sequence logo</b> "
            "&mdash; the same device used to summarise a multiple sequence "
            "alignment in genomics. One stack of letters per hour since shock "
            "onset; each letter&rsquo;s height is how often that state "
            "occupies that hour; the column&rsquo;s total height is how "
            "<i>determined</i> the hour is, as information content: "
            "log<sub>2</sub>(K) minus the entropy of the hour&rsquo;s state "
            "distribution, small-sample corrected.</p>"
            "<p>The <b>share</b> tabs draw every column at full height "
            "instead, which trades the determinacy reading for a legible "
            "composition. Read the bits version for whether an hour is "
            "determined and the share version for what it is made of.</p>"
        )
        out.append(logo_body)
        if consensus_file.is_file():
            cons = pl.read_csv(consensus_file)
            resp = cons.filter(pl.col("layer") == "response")
            treat = cons.filter(pl.col("layer") == "treatment")

            def _runs(text: str) -> int:
                return sum(1 for a, b in zip(text, text[1:]) if a != b)

            flat = [
                r for r in treat.iter_rows(named=True)
                if _runs(r["consensus"]) <= 1
            ]
            out.append(callout(
                "<b>The first thing to read off these is how little an hour is "
                "determined.</b> A response column carries a mean of "
                f"{float(resp['mean_bits'].min()):.2f} to "
                f"{float(resp['mean_bits'].max()):.2f} bits of a possible "
                f"{float(resp['max_bits'][0]):.2f}, so even <i>within</i> one "
                "handbook, knowing the hour tells you little about which state "
                "a patient is in. The treatment logos are more determined "
                f"&mdash; {float(treat['mean_bits'].min()):.2f} to "
                f"{float(treat['mean_bits'].max()):.2f} bits of "
                f"{float(treat['max_bits'][0]):.2f} &mdash; and almost "
                "trivially so: in "
                f"{len(flat)} of {treat.height} handbooks the treatment "
                "consensus changes character at most once across all 48 "
                "hours. That is worth saying plainly: <b>a handbook is mostly "
                "a dose level held flat, not a plan with stages.</b> The "
                "alignment-based clustering was free to find staged courses "
                "and largely did not, which is the single most deflating "
                "result in this section and the one that most constrains what "
                "the word &ldquo;handbook&rdquo; can be taken to mean."))
            out.append(callout(
                "<b>The coverage bar under each logo is the denominator and "
                "it falls by design.</b> Response strings are trimmed of their "
                "trailing off-pressor run, so a patient who comes off pressors "
                "at hour 12 contributes twelve columns and stops. The "
                "right-hand end of every logo is therefore drawn from the "
                "patients slowest to come off pressors, who are sicker than "
                "the handbook they came from &mdash; a late drift toward "
                "<code>o</code> is the survivors leaving, not a state "
                "transition. Columns resting on fewer than 25 patients are "
                "left blank and their bars dimmed. The bars are scaled "
                "<i>per row</i>, because the handbooks differ ten-fold in size "
                "and one scale made the small ones read as empty.", "warn"))
            out.append(table(
                cons,
                numeric_format={
                    "patients": thousands, "mortality": pct,
                    "mean_bits": num(2), "max_bits": num(2),
                },
                key_column="handbook",
            ))

    out.append("<h3>Both layers on the projection</h3>")
    out.append(
        "<p>The same projection twice, tiled into hexagons: what the clinician "
        "was doing, and what the patient was doing. The palettes differ on "
        "purpose. Response states are <b>nominal</b> &mdash; deteriorating is "
        "not more or less than wean held, it is a different thing &mdash; so "
        "they take evenly spaced hues.</p>"
        "<p>Treatment cells are <b>factorial</b>: four norepinephrine bands "
        "crossed with the presence of vasopressin and of any other vasoactive. "
        "So the colour is split to match the variable &mdash; the <b>agent "
        "set picks the hue</b> (blue for norepinephrine alone, orange plus "
        "vasopressin, green plus another agent, purple for all three) and the "
        "<b>dose band picks the shade</b> within it, darkest at high dose. An "
        "earlier version put all sixteen cells on one intensity ramp, which "
        "made <code>NE mid</code> and <code>NE low +vaso +other</code> "
        "neighbouring shades of the same colour: a reader could see how much "
        "support a region was on and could not see what it was made of. The "
        "ordinal reading survives <i>within</i> a hue and is gone "
        "<i>across</i> hues, which is correct &mdash; adding vasopressin is "
        "not a step on the norepinephrine scale.</p>"
        "<p><i>Plurality</i> is the most common state in a hexagon and reads "
        "directly. <i>Enrichment</i> is the state most concentrated there "
        "relative to its own total, which lets a small state appear wherever "
        "it genuinely localises. Where the two agree the structure is not an "
        "artefact of group size; where they disagree, that is the finding.</p>"
    )
    for layer, note in (
        ("response", "what the patient was doing"),
        ("treatment", "what the clinician was doing"),
    ):
        panels = [
            (
                mode.capitalize() + (" + flow" if flow else ""),
                figure(
                    figures_home
                    / f"layer_{layer}_hex_{mode}{'_flow' if flow else ''}"
                    f"_6h_{cohort}.png",
                    f"{layer} layer, {mode}"
                    + (", with the displacement field over it" if flow else ""),
                ),
            )
            for mode in ("plurality", "enrichment")
            for flow in (False, True)
        ]
        panels.append((
            "Each level alone",
            figure(
                figures_home / f"layer_{layer}_panels_6h_{cohort}.png",
                f"each {layer} level on its own axes",
            ),
        ))
        body = tabs(panels)
        if body:
            out.append(f"<h4>{esc(layer.capitalize())} &mdash; {esc(note)}</h4>")
            out.append(body)

    flow_figure = figure(
        figures_home / f"handbook_flow_{cohort}.png",
        "one streamplot per handbook, each from its own patients' "
        "displacements, over the pooled point cloud in grey",
    )
    if flow_figure:
        out.append("<h3>Do the handbooks move differently?</h3>")
        out.append(
            "<p><b>Each handbook&rsquo;s field is fitted from that "
            "handbook&rsquo;s own patients</b> &mdash; its own consecutive "
            "timepoints, its own displacements &mdash; and the five fields "
            "are then overlaid on one shared grid taken from the whole cloud. "
            "Crossing streamlines of different colours are therefore not a "
            "contradiction: they are two groups passing through one region in "
            "different directions, which is exactly what a pooled field "
            "averages away. (Elsewhere in this report a flow figure fits one "
            "field over everyone and varies only the colour underneath, so "
            "that a difference between panels cannot be a difference in the "
            "arrows. This figure is the deliberate exception.)</p>"
            "<p>Two things about the arrows were wrong in an earlier version "
            "and both shortened them. Each field took its <i>own</i> 0.5&ndash;"
            "99.5 percentile box, so a small handbook&rsquo;s arrows stopped "
            "at its own percentile edge rather than at the edge of its "
            "patients; and the support floor was 0.15% of the group&rsquo;s "
            "vectors, which on a cohort-sized frame is about 45 vectors "
            "against a mean cell of 44 and blanked roughly half the occupied "
            "cells. The hexagon maps above carried the same two faults.</p>"
        )
        out.append(flow_figure)
        out.append(cardiogenic_only(callout(
            "<b>The earlier version of this figure said the answer was no, "
            "and that was an artefact of the two faults above.</b> With the "
            "fields reaching as far as their patients do, the handbooks do "
            "differ &mdash; in <i>reach</i> rather than in direction. "
            "Handbooks 0 and 3 never leave the left limb (59 and 76 populated "
            "cells); handbooks 1, 4 and 5 carry arrows well into the "
            "upper-right limb (108, 88 and 143 cells). Those two confined "
            "handbooks are precisely the two high-intensity plans &mdash; "
            "<code>NE high +vaso +other</code> at 69% mortality and "
            "<code>NE mid +vaso +other</code> at 58% &mdash; so the region a "
            "handbook can reach tracks how much support it is built on.<br><br>"
            "What still looks shared is the <b>direction</b>: where two "
            "handbooks overlap, their streamlines run the same way. So the "
            "honest reading is that the handbooks differ in where their "
            "patients can get to and not in which way they travel once there, "
            "and the previous, stronger claim that nothing differed should not "
            "be carried forward.")))
        out.append(cardiogenic_only(callout(
            "<b>The handbooks differ several-fold in size</b>, so a small "
            "handbook's field is drawn from fewer vectors and is noisier at "
            "the same linewidth &mdash; 2,215 vectors for handbook 0 against "
            "25,734 for handbook 5. The per-handbook count is in the legend "
            "rather than implied by the ink, and part of the reach difference "
            "above is simply that a bigger handbook has vectors in more "
            "places. The two confined handbooks are not the two smallest, "
            "which is what keeps the reading above from being only a sample-"
            "size effect. And the projection is PaCMAP: it preserves "
            "neighbourhoods, not distances, so the <i>direction</i> of a "
            "streamline is interpretable and its <i>length</i> is not.",
            "warn")))

    # ---- which way a handbook's patients travel through the projection
    dir_file = tables_home / f"handbook_direction_{cohort}.csv"
    dir_panels = [
        (
            f"Handbook {number}",
            figure(
                figures_home / f"handbook_direction_hb{number}_{cohort}.png",
                f"handbook {number}'s patients split by the direction they "
                "travel along the cloud's main axis, each colour's field from "
                "its own patients",
            ),
        )
        for number in sorted(cat["handbook"].to_list())
    ]
    dir_body = tabs(dir_panels)
    if dir_body:
        out.append("<h3>Within a handbook, who enters which limb?</h3>")
        out.append(
            "<p>The overlay above compares handbooks with each other. It says "
            "nothing about whether a handbook&rsquo;s own arrows are parallel, "
            "and on the flow field they plainly are not: there is an "
            "upper-right limb whose streamlines run one way and a lower-left "
            "limb whose run the other, and they meet near the middle. So the "
            "follow-up question is who is in each.</p>"
            "<p>Patients are split by <b>which side of centre they enter</b> "
            "along the limb axis, and the direction is left to the arrows. "
            "The axis is the first principal direction of the point cloud "
            "fitted on its central 95% &mdash; the untrimmed version came out "
            "at (+0.99,&nbsp;+0.15), essentially horizontal, because a small "
            "outlying island stretches the x spread and flattens it; trimming "
            "recovers the visible diagonal in both cohorts.</p>"
            "<p>An earlier version split on the <i>sign of travel</i> instead, "
            "which needs a signed axis. Orienting it along the "
            "population&rsquo;s mean displacement looked principled until the "
            "mean turned out to have a coherence of <b>0.07</b> &mdash; the "
            "displacements very nearly cancel, there is no prevailing drift "
            "to point along, and the sign was arbitrary. Entry limb needs no "
            "sign convention. It is a <i>position</i>, not a destination: a "
            "patient who enters the upper limb can still travel down and out "
            "of it, and nothing here claims the limbs are absorbing.</p>"
        )
        out.append(dir_body)
        if dir_file.is_file():
            frame = pl.read_csv(dir_file)
            out.append(table(
                frame,
                numeric_format={
                    "patients": thousands, "mortality": pct,
                    "share_worse_response_group": pct,
                    "median_start_along": num(2),
                    "median_travel_along": num(2),
                    "median_hours": num(0),
                    "median_peak_ne": num(2),
                    "median_hours_on_pressors": num(0),
                    "median_lactate_max_in_window": num(1),
                    "median_sofa_non_cv": num(1),
                    "median_age_at_admission": num(0),
                },
                key_column="handbook",
            ))
            # The two group LABELS are read off the data rather than written in.
            # They have already changed once -- the split was on the sign of
            # travel and is now on entry limb -- and a hard-coded
            # "mortality_ascending" would have raised a KeyError against the
            # new table rather than quietly saying something false, but only
            # after a forty-minute figure run.
            labels = sorted(frame["direction"].unique().to_list())
            if len(labels) == 2:
                wide = frame.pivot(
                    on="direction", index="handbook",
                    values=["mortality", "share_worse_response_group"],
                )
                first, second = labels
                gap = (
                    wide[f"mortality_{first}"] - wide[f"mortality_{second}"]
                )
                group_gap = (
                    wide[f"share_worse_response_group_{first}"]
                    - wide[f"share_worse_response_group_{second}"]
                ).abs()
                ahead = int((gap > 0).sum())
                agreeing = max(ahead, len(gap) - ahead)
                worse = first if ahead * 2 > len(gap) else second
                spread = 100 * float(gap.abs().median())
                noise = 100 * float(group_gap.median())
                # THE HEADLINE GRADES ITSELF. Written as a flat "this separates
                # mortality" it read as a finding in both cohorts, and in
                # septic the gap is 2.3 points against a response-group gap of
                # 2.7 -- the same size as the quantity it is being contrasted
                # with. A sentence that says "separates" either way is not
                # reporting the number, it is decorating it.
                strong = spread > 2 * noise and agreeing >= len(gap) - 1
                out.append(callout(
                    (
                        "<b>Which limb a patient enters separates mortality, "
                        "and it is nearly orthogonal to the response-group "
                        "split.</b> "
                        if strong else
                        "<b>Which limb a patient enters barely separates "
                        "mortality in this cohort.</b> "
                    )
                    + f"The <i>{esc(worse)}</i> carries the higher mortality "
                    f"in {agreeing} of {len(gap)} handbooks, by a median of "
                    f"{spread:.1f} points. Over the same handbooks the share "
                    "of each limb sitting in the worse-outcome <i>response</i> "
                    f"group differs by a median of {noise:.1f} points. "
                    + (
                        "The mortality gap is several times that, so the limb "
                        "is carrying outcome information the response "
                        "clustering is not: where a patient enters the "
                        "embedding and which response course they trace are "
                        "close to independent, and both predict."
                        if strong else
                        "<b>Those two are the same size</b>, so there is "
                        "nothing here to distinguish a real geometric effect "
                        "from the response split showing through, and this "
                        "cohort's limbs should be read as uninformative about "
                        "mortality until something with more power says "
                        "otherwise."
                    )))
                out.append(callout(
                    "<b><code>median_travel_along</code> cannot be used as "
                    "evidence that the limbs converge</b>, and it will look "
                    "like it does. Patients entering the upper limb show "
                    "negative travel and those entering the lower limb "
                    "positive travel in almost every handbook &mdash; but the "
                    "groups were <i>defined</i> by the sign of their starting "
                    "position on this same axis, so a patient who starts high "
                    "has more room to move down than up and the reverse for "
                    "one who starts low. That is regression to the mean "
                    "manufactured by the split. The column is reported because "
                    "its <i>magnitude</i> is interpretable within a limb; its "
                    "sign is not a result.", "warn"))
            out.append(callout(
                "<b>Two things this cannot settle.</b> The limb axis is fitted "
                "on each cohort's own PaCMAP, and the two projections are "
                "separate fits with no shared orientation, so the "
                "<i>identity</i> of the worse limb does not transfer between "
                "cohorts even though the existence of a gap does. And the "
                "travel column is measured over a patient's whole record, "
                "which for someone who dies at hour 14 is fourteen hours and "
                "for a survivor is forty-eight &mdash; so read the hours row "
                "before the mortality row. PaCMAP preserves neighbourhoods, "
                "not distances: the <i>direction</i> of an arrow is "
                "interpretable and its <i>length</i> is not.", "warn"))
        out.append(cardiogenic_only(callout(
            "<b>Read the hours row before the mortality row.</b> Direction is "
            "measured over a patient&rsquo;s whole record, which for someone "
            "who dies at hour 14 is fourteen hours and for a survivor is "
            "forty-eight. Any difference between ascending and descending is "
            "therefore partly a difference in how long each group was "
            "observed, and the projection is PaCMAP &mdash; the "
            "<i>direction</i> of travel is interpretable and its "
            "<i>magnitude</i> is not.", "warn")))

    groups_file = tables_home / f"handbook_response_groups_{cohort}.csv"
    if groups_file.is_file():
        out.append("<h3>Two response courses inside one plan</h3>")
        out.append(
            "<p>Within each handbook the hourly response courses are "
            "run-length collapsed and split in two by alignment distance. "
            "<b>Collapsing matters and an earlier version of this table was "
            "wrong without it.</b> Clustering raw strings separated short "
            "off-pressor courses from long on-pressor ones, the medoids came "
            "out as <code>ooooooooooooDo</code> against "
            "<code>FHRFSRdNwwHFHHNNNdSHo</code>, and every covariate that "
            "then separated the groups was a duration at standardised "
            "differences of 0.49 to 0.76. Collapsing keeps the <i>order</i> of "
            "what happened and discards how long each state lasted, which is "
            "the level at which two patients can be said to have had the same "
            "response rather than the same length of stay.</p>"
        )
        groups = pl.read_csv(groups_file)
        out.append(table(
            groups,
            numeric_format={
                "patients": thousands, "group_a": thousands,
                "group_b": thousands, "mortality_a": pct,
                "mortality_b": pct, "mortality_gap": pct,
                "sofa_a": num(1), "sofa_b": num(1),
                "hours_a": num(0), "hours_b": num(0),
                "peak_ne_a": num(2), "peak_ne_b": num(2),
            },
            key_column="handbook",
        ))
        # A cohort-agnostic version of the reading above, computed rather than
        # written, so the section still says something when the cardiogenic
        # prose is withheld.
        ranked = groups.sort("mortality_gap", descending=True)
        widest = ranked.row(0, named=True)
        narrowest = ranked.row(-1, named=True)
        out.append(callout(
            "<b>Where the plan stops explaining the outcome.</b> The gap "
            "between the two response groups' mortality is widest in handbook "
            f"{widest['handbook']} at {100 * widest['mortality_gap']:.1f} "
            f"points ({100 * widest['mortality_a']:.0f}% against "
            f"{100 * widest['mortality_b']:.0f}% on {widest['patients']:,} "
            "patients) and narrowest in handbook "
            f"{narrowest['handbook']} at "
            f"{100 * narrowest['mortality_gap']:.1f} points. That range is "
            "the quantity this section exists to put a number on: it is how "
            "much of the outcome is left over once the treatment course is "
            "held fixed, and it is not an effect of the plan because the plan "
            "is what has been held fixed."))
        out.append(cardiogenic_only(callout(
            "<b>The worse course is the same course in every mid-to-high "
            "intensity handbook.</b> Handbooks 0, 1 and 3 all put the higher "
            "mortality on a medoid of the form "
            "<code>DSdSdSdSdSdSdS</code> &mdash; deteriorate once, then "
            "oscillate between stable hold and drifting hold indefinitely "
            "&mdash; while the better group's course contains "
            "<code>R</code>, <code>W</code> and <code>H</code>: responsive, "
            "wean held, highly responsive. This arrives independently of the "
            "motif work above and agrees with it, where drifting hold was "
            "already the destination of four of the top seven motifs. In "
            "handbook 5, the lowest-intensity plan, the gap is 1.0%: response "
            "pattern only separates outcome once the treatment is substantial."
        )))

    # ---- where the two response groups part, on the population
    div_file = tables_home / f"handbook_logo_divergence_{cohort}.csv"
    div_panels = [
        (
            f"Handbook {number} &mdash; {layer}",
            figure(
                figures_home / (
                    f"handbook_divergence_logo_{layer}_hb{number}_"
                    f"{cohort}.png"
                ),
                f"both response groups' {layer} courses on one clock, with "
                "the hour they part marked",
            ),
        )
        for number in sorted(cat["handbook"].to_list())
        for layer in ("response", "treatment")
    ]
    div_body = tabs(div_panels)
    if div_body:
        out.append("<h3>When do the two response groups actually part?</h3>")
        out.append(
            "<p>The matched-pair alignments below answer &ldquo;when do "
            "<i>these two</i> courses stop matching&rdquo;, one pair at a "
            "time. This answers it for the whole handbook. At every hour the "
            "two response groups&rsquo; state distributions are compared by "
            "<b>Jensen&ndash;Shannon divergence</b>, and the observed curve is "
            "read against a null built by permuting the group labels 400 times "
            "while holding the group sizes fixed. Two groups of unequal size "
            "differ at every hour by sampling alone, and more so once the "
            "cohort has thinned, so the null is not flat and a flat threshold "
            "would have found a parting wherever the denominator fell. The "
            "parting hour is the first hour the curve clears that band and "
            "stays clear for two hours.</p>"
        )
        out.append(div_body)
        if div_file.is_file():
            div = pl.read_csv(div_file)
            out.append(table(
                div,
                numeric_format={
                    "group_a": thousands, "group_b": thousands,
                    "divergence_at_hour": num(3), "null_at_hour": num(3),
                    "peak_divergence": num(3),
                },
                key_column="handbook",
            ))
            apart = (
                int(div["differ_from_start"].sum())
                if "differ_from_start" in div.columns else None
            )
            parted = int(div["divergence_hour"].is_not_null().sum())
            # Indexing a SORTED frame with an argmax taken on the UNSORTED one
            # was the first version. It returned the right handbook in both
            # cohorts by coincidence and would have silently returned the wrong
            # one as soon as the row order changed.
            biggest = (
                div.with_columns(
                    _total=pl.col("group_a") + pl.col("group_b")
                ).sort("_total", descending=True).row(0, named=True)
                if div.height else None
            )
            out.append(callout(
                "<b>Most of these groups were never together, so there is "
                "nothing to part.</b> A parting is reported only where the two "
                "groups are indistinguishable from permuted labels for three "
                "consecutive hours and then separate for two &mdash; and on "
                f"that rule {parted} of {div.height} handbooks have a parting "
                "hour at all"
                + (
                    f", while {apart} are already distinguishable in the "
                    "opening hours"
                    if apart is not None else ""
                )
                + ". <b>An earlier version of this table reported "
                "&ldquo;divergence hour 0&rdquo; for four of five handbooks, "
                "which is not a divergence</b>: hour 0 is the first hour on "
                "record and nothing can have parted by then. It put a number "
                "where the honest answer is that these groups never shared a "
                "course, and a reader skimming the column would have read 0 as "
                "&ldquo;immediately after onset&rdquo; rather than "
                "&ldquo;never together&rdquo;. The phrase this section is "
                "built around &mdash; same plan, then the courses diverge "
                "&mdash; presumes a shared beginning that most handbooks do "
                "not have."))
            if biggest is not None:
                out.append(callout(
                    "<b>&ldquo;Hours above the null&rdquo; is a power "
                    "statistic, not an effect size, and the largest handbook "
                    "shows why.</b> Handbook "
                    f"{biggest['handbook']}, with groups of "
                    f"{biggest['group_a']:,} and {biggest['group_b']:,}, "
                    f"clears the permutation band at "
                    f"{biggest['hours_above_null']} of 48 hours while its peak "
                    f"divergence is only {biggest['peak_divergence']:.3f} "
                    "bits. A difference that small is detectable there and "
                    "would be invisible in a handbook a tenth the size. "
                    "Compare the <code>peak_divergence</code> column across "
                    "handbooks; compare <code>hours_above_null</code> only "
                    "against handbooks of similar size.", "warn"))
        out.append(cardiogenic_only(callout(
            "<b>This overturns the framing the section was built on.</b> "
            "&ldquo;Same plan, then the courses diverge&rdquo; implies a "
            "shared beginning, and four of the five handbooks part at hour "
            "<b>zero</b>: the two response groups are distinguishable from the "
            "first hour on record. Only handbook 0 has a genuine shared "
            "opening, parting at hour 6 with its peak divergence at hour 26. "
            "An earlier version of this figure drew a shared prefix and then "
            "split &mdash; which presupposed the answer, and had to invent a "
            "three-hour prefix in a handbook the test had just said shared "
            "nothing. Both groups now get their whole course on one clock and "
            "the parting hour is a marker, not a seam.", "warn")))
        out.append(cardiogenic_only(callout(
            "<b>Handbook 5 does not diverge at all &mdash; it converges.</b> "
            "Its curve peaks at the start, falls monotonically, and sits below "
            "the permutation band from about hour 18 onward. Its two response "
            "groups begin apart and become indistinguishable. That is "
            "consistent with its 1.0% mortality gap, with the embedding "
            "predicting its response group at 0.584, and with the alignment "
            "result in the same handbook: the 0.845 within-pair AUC at the "
            "step before divergence was reported as risk being legible early, "
            "and this says the same thing from the other side &mdash; in the "
            "lowest-intensity plan, whatever separates the two courses is "
            "present at hour zero and spent by the end of the first day.")))

    rendered = tables_home / f"handbook_divergences_{cohort}.csv"
    if rendered.is_file():
        shown = pl.read_csv(rendered)
        # the SAME handbook 42 drew its figures for -- the largest -- rather
        # than whichever one happens to have rendered the most example pairs.
        # Every handbook renders up to six, so a mode over this column is an
        # arbitrary tie-break and the section would illustrate one handbook
        # while the scatter and the violins below describe another.
        available = set(shown["handbook"].unique().to_list())
        ranked = cat.filter(
            pl.col("handbook").is_in(list(available))
        ).sort("patients", descending=True)
        focus = int(ranked["handbook"][0]) if ranked.height else int(
            shown["handbook"][0]
        )
        block = shown.filter(pl.col("handbook") == focus).head(4)
        if block.height:
            out.append("<h3>What a divergence looks like</h3>")
            out.append(
                f"<p>Four typical pairs from handbook {focus}, chosen nearest "
                "that handbook's <b>median</b> divergence hour rather than for "
                "being striking. Each block is five lines: the treatment "
                "course of the better-outcome patient, their response course, "
                "the match line, the worse patient's response course, and "
                "their treatment course. <code>|</code> marks an identical "
                "state, <code>+</code> a substitutable one, a blank a "
                "mismatch, and <code>-</code> a gap. The alignment is anchored "
                "at the first hour and its endpoint is free, so it extends "
                "from onset for exactly as long as extending is worth points "
                "&mdash; <b>the hour it stops is the divergence</b>, marked "
                "<code>^</code>.</p>"
                "<p>Everything after the <code>&gt;</code> is each course "
                "carrying on from where the alignment stopped. That part is "
                "<b>not</b> aligned and the two sides are not in register "
                "there, deliberately: an earlier version of this figure "
                "printed only the aligned region, which is by construction "
                "the part where nothing happened, and so showed five "
                "identical hours and stopped exactly where the reader needed "
                "to look.</p>"
            )
            for row in block.iter_rows(named=True):
                # None is "outcome unknown", and `not None` is True, so the
                # obvious chain of truthiness tests reports an unknown pair as
                # "both survived"
                fates = {
                    (False, False): "both survived",
                    (True, True): "both died",
                    (False, True): "survived vs died",
                    (True, False): "died vs survived",
                }
                pair = (row["died_better"], row["died_worse"])
                fate = (
                    fates[(bool(pair[0]), bool(pair[1]))]
                    if None not in pair else "outcome unknown"
                )
                identity = row.get("identity")
                shown_identity = (
                    f"{identity:.0%} identity up to that point &middot; "
                    if identity is not None and identity == identity else ""
                )
                out.append(
                    f"<p class=\"meta\">diverged at hour "
                    f"{row['divergence_hour']} &middot; "
                    f"{shown_identity}{esc(fate)}</p>"
                )
                out.append(formula(
                    f"treatment  {row['treatment_better']}",
                    f"response   {row['response_better']}",
                    f"           {row['match']}",
                    f"response   {row['response_worse']}",
                    f"treatment  {row['treatment_worse']}",
                ))
            # written by a later version of 42 than the one that may have
            # produced the CSV on disk; absent means the audit is
            # unavailable, not that the plans disagreed
            # prefer the POPULATION audit over the rendered handful: 42
            # writes agreement over every diverging pair, and the six
            # examples per handbook are an illustration, not a denominator
            population = tables_home / f"handbook_plan_agreement_{cohort}.csv"
            if population.is_file():
                every = pl.read_csv(population)
                gap = float(
                    (every["band_gap_mean"] * every["diverging_pairs"]).sum()
                    / max(every["diverging_pairs"].sum(), 1)
                )
                total = int(every["diverging_pairs"].sum())
                peak = int(round(float(
                    (every["same_peak_band"] * every["diverging_pairs"]).sum()
                )))
                course = int(round(float(
                    (every["course_identical"] * every["diverging_pairs"]).sum()
                )))
                hourly = int(round(float(
                    (every["hourly_identical"] * every["diverging_pairs"]).sum()
                )))
                has_audit = True
            elif "band_gap_mean" in shown.columns:
                gap = float(shown["band_gap_mean"].mean())
                course = int(shown["plan_course_identical"].sum())
                peak = int(shown["plan_same_peak_band"].sum())
                total = shown.height
                hourly = int(shown["plan_hourly_identical"].sum())
                has_audit = True
            else:
                has_audit = False
            if has_audit:
                audit = (
                    "<br><br><b>How fixed is 'the same plan', really?</b> "
                    "Across every diverging matched pair, the two patients "
                    "are a mean "
                    f"of <b>{gap:.2f} norepinephrine bands apart per aligned "
                    "hour</b> on a 0-to-3 scale, share a peak band in "
                    f"{peak} of {total} pairs, and have an identical "
                    f"collapsed course in {course} of {total}, and are "
                    f"identical hour by hour in {hourly} of {total}. That "
                    "last figure is the one to distrust, and it moved most "
                    "when the audit stopped being run on a handful of "
                    "examples: six rendered pairs per handbook put it near "
                    "zero, because they are chosen close to the median "
                    "divergence hour and so are the longer, more intricate "
                    "alignments, while most diverging pairs part after only a "
                    "few hours and those few hours agree trivially. It is "
                    "also the wrong test regardless: a handbook is defined on "
                    "the order of what "
                    "was done, not its durations, so two patients on the same "
                    "plan all week differ at every hour if one was weaned "
                    "three hours sooner. Read the band gap. It says the plans "
                    "are close rather than identical, and it is the honest "
                    "bound on how much of this section is "
                    "response-to-a-fixed-plan. Per handbook it runs from 0.17 "
                    "in handbook 4 to 0.38 in handbook 0 &mdash; and note "
                    "that handbook 0 has the <i>largest</i> gap while 97% of "
                    "its pairs share a peak band, because everyone there is "
                    "at high-dose norepinephrine so peaks agree trivially "
                    "while the hour-to-hour path differs most."
                )
            else:
                audit = ""
            out.append(cardiogenic_only(callout(
                "<b>Read the two treatment lines first.</b> Where they still "
                "agree at the hour the response lines part, the divergence is "
                "a difference in the patient under an identical plan. Where "
                "they have already parted, the handbook floor has let two "
                "different plans into one group and the pair says nothing "
                "about response to a shared plan. The alignment is on the "
                "response layer only, so the treatment lines are carried "
                "along rather than optimised &mdash; they are a check on the "
                "comparison, not part of it." + audit, "warn")))

    probe_file = tables_home / f"divergence_probe_{cohort}.csv"
    if probe_file.is_file():
        probe = pl.read_csv(probe_file)
        out.append("<h3>Is the divergence legible before it happens?</h3>")
        out.append(
            "<p>Cross-validated AUC with a bootstrap interval, every "
            "transformation fitted inside the fold. The embedding is 1,024 "
            "CORAL-corrected dimensions reduced to at most 32 principal "
            "components and fed to L2 logistic regression, so these are "
            "<b>floors</b> on what the representation holds, not ceilings: a "
            "value at chance says this probe found no linear signal, not that "
            "there is none. <code>patients</code> is the largest probe's "
            "denominator; the mortality probes drop patients whose "
            "outcome is unknown and so run on slightly fewer.</p>"
        )
        landmark = _probe_table(probe, "landmark", hour=24.0)
        pre = _probe_table(probe, "pre_divergence")
        out.append(tabs([
            ("At a fixed 24 h landmark", table(
                landmark, key_column="handbook") if landmark.height else ""),
            ("At the step before divergence", table(
                pre, key_column="handbook") if pre.height else ""),
        ]))
        out.append(cardiogenic_only(callout(
            "<b>The two tabs are not comparable and the second is "
            "optimistic.</b> The landmark probe reads the embedding at a hour "
            "fixed in advance, so nothing about the future enters it. The "
            "pre-divergence probe reads it at the last grid point before the "
            "pair actually parted &mdash; the features are still built only "
            "from the past, but <i>which hour to look at</i> was chosen using "
            "the whole future course. The 'read hour alone' column exists to "
            "bound that: it scores the hour by itself, on the same patients "
            "and the same folds."
        )))
        out.append(cardiogenic_only(callout(
            "<b>Two leaks were found and removed here, and naming them is "
            "part of the result.</b> A first run scored "
            "<code>DSCG//expired</code> &mdash; the discharge-disposition "
            "token &mdash; as a top separator of the mortality groups at "
            "q&nbsp;=&nbsp;0.008. A patient who died at hour 20 has their own "
            "death in the stream the hour-24 embedding is built from. Patients "
            "who left hospital before the landmark are now excluded (2% of the "
            "cohort by 24&nbsp;h) and the <code>DSCG</code> and "
            "<code>EOS</code> families are dropped before any probe sees them. "
            "The effect on the AUCs was about 0.01; the effect on the "
            "single-token screen was to cut handbook 0 from 36 hits to 11. "
            "<b>One leak is not closed:</b> code status is censored only from "
            "shock onset onward, so a DNR entered before onset still reaches "
            "the probe &mdash; which is why zero patients were censored for "
            "it, a count that is not credible.", "warn")))

    scatters = sorted(figures_home.glob(f"handbook_outcome_hb*_{cohort}.png"))
    hexes = sorted(
        figures_home.glob(f"handbook_mortality_hex_hb*_{cohort}.png")
    )
    violins = sorted(figures_home.glob(f"handbook_tokens_hb*_{cohort}.png"))
    if scatters or hexes or violins:
        out.append("<h3>Where the outcome sits, and what is charted</h3>")
        out.append(
            "<p>One handbook at a time, because the question is whether "
            "outcome localises <i>within</i> a plan rather than across plans. "
            "The scatter puts every patient-timepoint of that handbook on the "
            "projection, died against survived. The violins leave the "
            "projection behind entirely and compare how often each token is "
            "charted in the first 24&nbsp;hours, which is the same evidence "
            "the embedding is built from but read one dimension at a "
            "time.</p>"
            "<p><b>Read the hexagon panel, not the scatter, for whether "
            "outcome localises.</b> In the scatter whichever group is drawn "
            "second sits on top, and that is the smaller one here, so the "
            "minority looks like the majority wherever the two overlap. The "
            "hexagons colour each region by the share who died and carry no "
            "draw-order artefact. Both are shown because the scatter keeps "
            "individual patients and the binning does not.</p>"
        )
        panels = []
        for path in hexes:
            panels.append((
                f"Mortality by region \u2014 "
                f"{path.stem.split('_hb')[1].split('_')[0]}",
                figure(path, "share of patient-timepoints in each hexagon "
                             "who died, against this handbook's own rate"),
            ))
        for path in scatters:
            panels.append((
                f"Outcome \u2014 {path.stem.split('_hb')[1].split('_')[0]}",
                figure(path, "died against survived, within one handbook"),
            ))
        for path in violins:
            panels.append((
                f"Tokens \u2014 {path.stem.split('_hb')[1].split('_')[0]}",
                figure(path, "the tokens whose rate differs most between "
                             "died and survived, within one handbook"),
            ))
        out.append(tabs(panels))

        use = sorted(tables_home.glob(f"handbook_token_use_hb*_{cohort}.csv"))
        if use:
            frame = pl.read_csv(use[0])
            out.append(
                "<p>The violins cannot show a point mass, and most patients "
                "record none of most tokens, so the share of patients with "
                "any is given here instead.</p>"
            )
            out.append(table(
                frame,
                numeric_format={
                    "smd": num(2), "p": pval,
                    "median_survived": num(1), "median_died": num(1),
                    "present_survived": pct, "present_died": pct,
                },
                key_column="token",
            ))
            out.append(cardiogenic_only(callout(
                "<b>A token count is not a measurement &mdash; it is a "
                "measurement that someone chose to take.</b> The tokens that "
                "separate the outcome groups here are consciousness and "
                "oxygenation rather than haemodynamics: survivors carry more "
                "<code>gcs_motor_Q9</code>, <code>gcs_verbal_Q9</code>, "
                "<code>rass_Q8</code> and <code>nasal_cannula</code>, and "
                "those who died more <code>rass_Q0</code>. Two cautions "
                "follow, and both cut against reading this as physiology. "
                "<b>Frequency confounds state with surveillance:</b> a high "
                "<code>gcs_motor_Q9</code> count partly means the patient had "
                "a recordable GCS repeatedly, so some of this separation is "
                "how often someone was assessed rather than how they were. "
                "And <b>nasal cannula against deep sedation is largely a "
                "ventilation contrast</b>, which is treatment, not response "
                "&mdash; the handbook holds the vasoactive plan fixed and "
                "says nothing about the airway. The share-present columns "
                "above are the partial guard: where presence differs as much "
                "as rate, the signal is not purely charting intensity.",
                "warn")))

    out.append("<h3>What the section does and does not establish</h3>")
    out.append(cardiogenic_only(callout(
        "<b>Groups of patients do respond differently to the same handbook.</b> "
        "Holding the plan fixed, response groups differ in mortality by up to "
        "16.8 points, and the worse course is recognisably the same oscillation "
        "in all three mid-to-high intensity handbooks.<br><br>"
        "<b>Baseline covariates do not explain which group a patient lands "
        "in.</b> The largest standardised difference across age, "
        "non-cardiovascular SOFA, lactate and support intensity is 0.37, and "
        "the top hit is a duration.<br><br>"
        "<b>Neither does the embedding.</b> This is the part worth stating "
        "plainly, because it is a negative result about the model this project "
        "is built on. Within handbook, the embedding predicts <i>death</i> at "
        "0.76 to 0.84 &mdash; well above baseline covariates, which reach only "
        "0.59 to 0.75 &mdash; but it predicts <i>which response course a "
        "patient will trace</i> at 0.52 to 0.64, with intervals that touch "
        "0.5. Risk is legible early. Trajectory shape is not. Those are "
        "different targets and conflating them would overstate what the "
        "representation does.<br><br>"
        "<b>Token timing is not the missing channel either.</b> First-"
        "appearance hour yields no survivor of multiplicity correction in "
        "handbooks 0, 1 or 3. Token <i>rate</i> separates mortality strongly "
        "but never beats the embedding, so the embedding is largely "
        "re-encoding token composition rather than adding physiology on top of "
        "it.<br><br>"
        "<b>The better-subset question remains unanswered.</b> Asking whether "
        "a subset of the worse group does better and is managed differently "
        "gave three different answers in three handbooks, and the direction "
        "reversed when the response clustering was corrected and a 24-hour "
        "landmark applied. The only consistent thread &mdash; that better "
        "subsets received less vasoactive exposure &mdash; is confounding by "
        "indication read backwards, not evidence that less drug helps."
    )))


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
    parser.add_argument(
        "--cohort", choices=("septic", "cardiogenic"), default=None,
        help="write a cohort-specific report instead of the combined one. "
             "The cohort description, the token-difference analysis and the "
             "model provenance are shared and appear in both; the "
             "projections, the guideline fit and the whole state-space and "
             "latent-class analysis are scoped to this cohort.",
    )
    args = parser.parse_args()
    # module-level so the section functions can scope themselves without
    # threading the cohort through a dozen signatures
    global COHORT
    COHORT = args.cohort

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
    section_state_space(paths, out)
    section_handbooks(paths, out)
    section_sites(paths, out)
    section_models(paths, out, logger)
    # Undifferentiated shock dropped from the report. The function is kept
    # -- it is the only description of the 21,462 patients with no coded
    # shock type, who are over half the cohort -- so one line restores it.
    undifferentiated = {}
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
        ("longitudinal", "7 &middot; Longitudinal strata"),
        ("states", "8 &middot; MAP/pressor states and their motifs"),
        ("handbooks", "9 &middot; Treatment handbooks"),
        ("sites", "10 &middot; Why the sites separate"),
        ("models", "11 &middot; Models"),
    ]
    if undifferentiated:
        entries.append(
            ("undifferentiated", "12 &middot; Undifferentiated shock")
        )
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

    name = "report.html" if COHORT is None else f"report_{COHORT}.html"
    destination = paths.output_home / name
    destination.write_text(document, encoding="utf-8")
    logger.info(
        f"wrote {destination} ({destination.stat().st_size / 1e6:.1f} MB, "
        f"{_TAB_GROUP[0]} tab groups)"
    )
    if args.open:
        webbrowser.open(destination.as_uri())


if __name__ == "__main__":
    main()

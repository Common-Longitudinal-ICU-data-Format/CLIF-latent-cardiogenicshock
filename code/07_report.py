#!/usr/bin/env python3

"""
07 -- assemble the analyses into a single self-contained HTML report

Reads whatever 01-06 have written and renders it as one file. Figures are
inlined as base64 so the report can be emailed or opened from anywhere without
carrying a directory of PNGs alongside it; nothing is fetched at view time.

Only sections whose inputs exist are rendered, so this is safe to run at any
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
from clifcs.palette import CATEGORICAL, GRID, INK_PRIMARY, INK_SECONDARY, SURFACE

MODELS = ("GEM-all", "GEM-mimic-icu", "GEM-ucmc-icu")


# ----------------------------------------------------------------- rendering


def css() -> str:
    return f"""
:root {{
  --surface: {SURFACE['light']};
  --ink: {INK_PRIMARY};
  --ink-2: {INK_SECONDARY};
  --grid: {GRID};
  --s1: {CATEGORICAL[0]};
  --s2: {CATEGORICAL[1]};
  --s3: {CATEGORICAL[2]};
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 48px 28px 96px; }}
h1 {{ font-size: 30px; margin: 0 0 6px; letter-spacing: -0.02em; }}
h2 {{
  font-size: 21px; margin: 56px 0 4px; letter-spacing: -0.01em;
  padding-top: 20px; border-top: 1px solid var(--grid);
}}
h3 {{ font-size: 15px; margin: 30px 0 8px; color: var(--ink); }}
p, li {{ color: var(--ink-2); max-width: 78ch; }}
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
  border-left: 3px solid var(--s1); background: #f6f8fc;
  padding: 12px 16px; margin: 18px 0; border-radius: 0 4px 4px 0;
}}
.callout.warn {{ border-left-color: var(--s2); background: #fdf5f1; }}
.callout p {{ margin: 0; color: var(--ink); font-size: 14px; }}
.callout p + p {{ margin-top: 8px; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 820px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
code {{
  font: 12.5px ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #f2f1ee; padding: 1px 5px; border-radius: 3px;
}}
.tag {{
  display: inline-block; font-size: 11px; padding: 2px 7px; border-radius: 3px;
  background: #f2f1ee; color: var(--ink-2); margin-right: 6px;
}}
.toc {{ margin: 28px 0 0; padding: 0; list-style: none; columns: 2; }}
.toc li {{ margin: 3px 0; }}
.toc a {{ color: var(--s1); text-decoration: none; font-size: 13.5px; }}
.toc a:hover {{ text-decoration: underline; }}
"""


def esc(value) -> str:
    return html.escape(str(value))


def table(frame: pl.DataFrame, numeric_format: dict | None = None,
          key_column: str | None = None, highlight: set | None = None) -> str:
    """render a dataframe, right-aligning numbers and marking a key column"""
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
                value = (
                    numeric_format[column](value) if value is not None else "—"
                )
            if column == key_column:
                classes.append("key")
            cells.append(
                f'<td class="{" ".join(classes)}">{esc(value)}</td>'
            )
        cls = ' class="total"' if str(row.get(key_column, "")) in highlight else ""
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def figure(path: pathlib.Path, caption: str) -> str:
    """inline a PNG so the report stays a single portable file"""
    if not path.is_file():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<figure><img src="data:image/png;base64,{data}" alt="{esc(caption)}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def callout(text: str, kind: str = "") -> str:
    return f'<div class="callout {kind}"><p>{text}</p></div>'


def read_model_tables(
    paths, name: str, required: tuple[str, ...], logger
) -> tuple[pl.DataFrame | None, list[str]]:
    """
    collect one table across models, refusing to merge mismatched schemas

    A plain diagonal concat fills a column missing from one file with nulls,
    which renders as "None" in the report and looks like a result rather than
    a stale file. This happened: a report built before a rerun finished showed
    `classifier: None` for two of three models. Any file lacking a required
    column is dropped and named instead.
    """
    frames, stale = [], []
    for model in MODELS:
        f = paths.output_home / "tables" / model / name
        if not f.is_file():
            continue
        frame = pl.read_csv(f)
        missing = [c for c in required if c not in frame.columns]
        if missing:
            stale.append(f"{model} ({', '.join(missing)} absent)")
            logger.warning(
                f"{f} lacks {missing}; excluded from the report. Rerun the "
                "stage that writes it."
            )
            continue
        frames.append(frame.with_columns(model=pl.lit(model)))
    if not frames:
        return None, stale
    return pl.concat(frames, how="diagonal_relaxed"), stale


# ------------------------------------------------------------------ sections


def pct(x):
    return "—" if x is None else f"{x:.1%}"


def num(digits):
    return lambda x: "—" if x is None else f"{x:,.{digits}f}"


def thousands(x):
    return "—" if x is None else f"{int(x):,}"


def section_cohort(paths, cfg, out: list) -> None:
    ledger_file = paths.cohort_home / "filter_ledger.csv"
    if not ledger_file.is_file():
        return
    ledger = pl.read_csv(ledger_file)
    out.append('<h2 id="cohort">1 &middot; Cohort</h2>')
    out.append(
        "<p>Cardiogenic shock defined as sustained hypotension or vasoactive "
        "support, plus evidence of end-organ damage. Urine output is not "
        "available at either site, so creatinine-based KDIGO AKI stands in for "
        "it and lactate is added as a second hypoperfusion marker.</p>"
    )

    wide = (
        ledger.filter(~pl.col("step").str.starts_with("  "))
        .pivot(on="site", index="step", values="n_hospitalizations", aggregate_function="first")
    )
    order = ledger.filter(~pl.col("step").str.starts_with("  "))["step"].unique(maintain_order=True)
    wide = wide.join(pl.DataFrame({"step": order}).with_row_index("o"), on="step").sort("o").drop("o")
    out.append(table(wide, {c: thousands for c in wide.columns if c != "step"},
                     key_column="step",
                     highlight={s for s in wide["step"] if "CRITERION 2" in s}))

    subs = ledger.filter(pl.col("step").str.starts_with("  ")).with_columns(
        step=pl.col("step").str.strip_chars()
    ).pivot(on="site", index="step", values="n_hospitalizations", aggregate_function="first")
    out.append("<h3>Criterion components</h3>")
    out.append(table(subs, {c: thousands for c in subs.columns if c != "step"}, key_column="step"))

    enabled = bool(cfg.cardiac_origin.enabled)
    out.append(callout(
        "<b>A cardiac criterion exists and is computed, but is not applied to "
        "enrollment.</b> <code>cardiac_origin</code> is satisfied by an inotrope "
        "infusion, an MCS procedure, or peak troponin ≥500 ng/L near onset, "
        f"and is currently <b>{'applied' if enabled else 'reported only'}</b> "
        "(<code>cardiac_origin.enabled</code>). It selects 2,219 UCMC (24.3%) and "
        "5,104 NU (20.4%). As enrolled, the cohort therefore spans "
        "undifferentiated shock, including septic, hypovolaemic and haemorrhagic. "
        "For reference, "
        "STEMI is coded in 3.0% / 2.8% and R57.0 cardiogenic shock in 19.7% / "
        "9.6% — against the ~81% STEMI of AMI-CS registries, which enrol on "
        "acute MI and so have a different denominator entirely.", "warn"))

    summary_file = paths.cohort_home / "phenotype_summary.csv"
    if summary_file.is_file():
        summary = pl.read_csv(summary_file)
        out.append("<h3>Cardiogenic shock phenotypes (Zweck 2021, refit)</h3>")
        out.append(
            "<p>k-means refit on the eight labs the paper names, then labelled "
            "from centroid profiles rather than cluster index. Mortality rises "
            "I &lt; II &lt; III as published (10&ndash;28% / 32&ndash;45% / "
            "52&ndash;56%), which is the check that the refit recovered the "
            "published phenotypes.</p>"
        )
        out.append(table(summary, {
            "n": thousands, "n_ucmc": thousands, "n_nu": thousands,
            "mortality": pct, "median_icu_los_days": num(1),
        }, key_column="cs_phenotype"))


def section_model(paths, out: list) -> None:
    rows = []
    for model in MODELS:
        f = paths.output_home / "tables" / model / "model_summary.json"
        if f.is_file():
            info = json.loads(f.read_text())
            rows.append({
                "model": model,
                "base": info.get("base_model", ""),
                "layers": info["num_hidden_layers"],
                "embedding dim": info["embedding_dim"],
                "heads": info["num_attention_heads"],
                "vocab": info["vocab_size"],
                "parameters": info["total_parameters"],
            })
    if not rows:
        return
    out.append('<h2 id="model">2 &middot; Models</h2>')
    out.append(
        "<p>Each checkpoint is used strictly frozen, in eval mode under "
        "<code>torch.inference_mode()</code>. The representation is the final "
        "hidden state &mdash; the last decoder layer after the final RMSNorm, "
        "before <code>lm_head</code> &mdash; taken at the single last token "
        "before the anchor. It is not mean-pooled.</p>"
    )
    out.append(table(pl.DataFrame(rows), {
        "layers": thousands, "embedding dim": thousands, "heads": thousands,
        "vocab": thousands, "parameters": thousands,
    }, key_column="model"))
    out.append(
        "<p>Anchors: <b>24h</b> is 24 hours after admission, matching the "
        "winnowing the models were evaluated under; <b>onset</b> is the moment "
        "cardiogenic shock criteria are first jointly met.</p>"
    )


def section_embedding(paths, out: list) -> None:
    rows = []
    for model in MODELS:
        for anchor in ANCHORS:
            f = paths.output_home / "tables" / model / f"neighbourhood_purity_anchor-{anchor}.csv"
            if f.is_file():
                rows.append(pl.read_csv(f).with_columns(model=pl.lit(model)))
    if not rows:
        return
    purity = pl.concat(rows)
    out.append('<h2 id="embedding">3 &middot; Embedding space</h2>')
    out.append(
        "<p>Neighbourhood purity is the share of each patient's 30 nearest "
        "neighbours (50-d PCA space) carrying the same label, against the share "
        "expected by chance. Lift near 1.0 means the space is not organised by "
        "that label.</p>"
    )
    wide = purity.pivot(on="label", index=["model", "anchor"], values="lift").sort("anchor", "model")
    out.append(table(wide, {c: num(2) for c in wide.columns if c not in ("model", "anchor")},
                     key_column="model"))
    out.append(callout(
        "<b>Site dominates the geometry, and training data explains about half "
        "of it.</b> GEM-all, trained on both hospitals, reaches 1.60&times;; "
        "GEM-mimic-icu, which has seen neither UCMC nor NU, still reaches "
        "1.29&times;. So part of the separation is the model having learned "
        "these sites, and part is intrinsic to the data. Phenotype and mortality "
        "lift stay near 1.1&times; throughout &mdash; the extra structure GEM-all "
        "carries is site, not physiology."))

    base = paths.output_home / "figures" / "GEM-all" / "anchor-24h"
    out.append("<h3>GEM-all, 24h anchor</h3>")
    out.append('<div class="grid2">')
    for name, caption in (
        ("umap_site.png", "UMAP coloured by contributing site"),
        ("umap_phenotype.png", "UMAP coloured by cardiogenic shock phenotype"),
        ("umap_mortality.png", "UMAP coloured by in-hospital mortality"),
        ("umap_icu_los.png", "UMAP coloured by ICU length of stay"),
    ):
        out.append(figure(base / name, caption))
    out.append("</div>")


def section_site_effect(paths, out: list) -> None:
    lda_rows = []
    for anchor in ANCHORS:
        f = paths.output_home / "tables" / "GEM-all" / f"site_lda_anchor-{anchor}.csv"
        if f.is_file():
            lda_rows.append(pl.read_csv(f))
    if not lda_rows:
        return
    out.append('<h2 id="site">4 &middot; Why the sites separate</h2>')
    lda = pl.concat(lda_rows).select(
        "anchor", "cv_accuracy", "cv_accuracy_sd", "cv_auc", "majority_baseline", "n"
    )
    out.append(
        "<p>A shrinkage LDA on the 50-d PCA representation, five-fold "
        "cross-validated so a separable class pair cannot report a free 100%.</p>"
    )
    out.append(table(lda, {
        "cv_accuracy": num(4), "cv_accuracy_sd": num(4), "cv_auc": num(4),
        "majority_baseline": num(4), "n": thousands,
    }, key_column="anchor"))

    out.append(callout(
        "<b>The cause is structural zeros in the vocabulary.</b> UCMC emits 101 "
        "token types NU never produces &mdash; Braden subscales, CAM, AM-PAC "
        "&mdash; and 96.5% of UCMC patients carry at least one, though they are "
        "only 2.8% of its token volume. NU emits 66 UCMC never does, "
        "<code>CODE//dnr</code> among them. A single occurrence identifies the "
        "site with certainty, which is what an LDA at 0.995 accuracy is reading. "
        "Separation survives matching the sites on age, sex, phenotype and "
        "mortality at 95.7% purity, so it is documentation practice and not "
        "patient differences.", "warn"))

    base = paths.output_home / "figures" / "GEM-all" / "anchor-24h"
    out.append(figure(base / "site_token_tornado.png",
                      "Tokens ranked by absolute difference in per-patient rate"))
    out.append(figure(base / "site_token_tornado_exclusive.png",
                      "Tokens one site never records at all"))
    out.append(figure(base / "site_token_families.png",
                      "Token-family mix as a share of each site's stream"))


def section_correction(paths, out: list) -> None:
    rows = []
    for model in MODELS:
        f = paths.output_home / "tables" / model / "site_correction.csv"
        if f.is_file():
            rows.append(pl.read_csv(f).with_columns(model=pl.lit(model)))
    if not rows:
        return
    correction = pl.concat(rows).filter(pl.col("anchor") == "24h")
    out.append('<h2 id="correction">5 &middot; Can the site effect be removed?</h2>')
    out.append(
        "<p>Two affine maps fitted on a training split and judged on held-out "
        "patients. <b>The AUCs below predict site</b> &mdash; which hospital a "
        "patient came from &mdash; not any clinical outcome. LEACE is the "
        "closed-form map that provably removes all linearly-decodable site "
        "information with minimal distortion.</p>"
    )
    show = correction.select(
        "model", "correction", "linear_probe_auc", "nonlinear_probe_auc", "purity_lift"
    ).sort("model", "correction")
    out.append(table(show, {
        "linear_probe_auc": num(3), "nonlinear_probe_auc": num(3), "purity_lift": num(2),
    }, key_column="model"))
    out.append(callout(
        "<b>No affine transformation can clean this space.</b> LEACE drives the "
        "linear probe to exactly chance, as guaranteed &mdash; but a "
        "gradient-boosted tree still recovers site at 0.99. Iterating the "
        "erasure does not help: removing 25 directions instead of one leaves the "
        "non-linear probe at 0.992 and still keeps 97.5% of the variance, "
        "because after the first direction there is no linear site signal left "
        "to find. Site is distributed non-linearly through the representation, "
        "not held in a low-dimensional subspace.", "warn"))
    out.append(
        "<p>For supervised prediction this does not matter (section 6). For "
        "unsupervised discovery &mdash; asking how groups of patients differ "
        "&mdash; it matters a great deal, because the contaminated structure is "
        "exactly what such an analysis reads. Only 370 of 34,142 patients (1.1%) "
        "currently sit in neighbourhoods where both sites are represented.</p>"
    )


def section_downstream(paths, out: list, logger) -> None:
    results, stale = read_model_tables(
        paths,
        "downstream_tasks.csv",
        ("task", "evaluation", "auroc", "classifier"),
        logger,
    )
    if results is None:
        return
    has_head = True
    out.append('<h2 id="downstream">6 &middot; Downstream prediction</h2>')
    out.append(
        "<p>Three tasks, each anchored at the moment the model read the patient "
        "and asking what happens in the next 24 hours. The clinical baseline uses "
        "age, sex, the last labs and vitals recorded before the anchor, and "
        "vasopressor burden &mdash; nothing from the prediction window.</p>"
    )
    if stale:
        out.append(callout(
            "<b>Excluded from this section:</b> " + esc("; ".join(stale))
            + ". Those files predate the current schema; rerun 06 for them.",
            "warn"))

    arm = (
        pl.when(pl.col("features") != "embedding").then(pl.lit("clinical baseline"))
        .otherwise(pl.col("correction"))
    )
    results = results.with_columns(arm=arm)

    for anchor in ANCHORS:
        part = results.filter(pl.col("anchor") == anchor)
        if part.height == 0:
            continue
        out.append(f"<h3>Cross-site transfer &mdash; {anchor} anchor</h3>")
        cross = part.filter(pl.col("evaluation").str.contains("->"))
        index = ["task", "evaluation", "model"] + (["classifier"] if has_head else [])
        wide = cross.pivot(on="arm", index=index, values="auroc").sort(*index)
        value_cols = [c for c in wide.columns if c not in index]
        out.append(table(wide, {c: num(3) for c in value_cols}, key_column="task"))

    out.append(callout(
        "<b>Correction changes nothing, because transfer was never broken.</b> "
        "Across every cell, LEACE moves AUROC by a mean of −0.0005 and "
        "per-site standardisation by +0.0001. Cross-site transfer already "
        "matches within-site performance (mortality NU&rarr;UCMC 0.925 against "
        "within-UCMC 0.919). The site axis and the outcome axes are close to "
        "orthogonal, so a supervised model ignores site without being told to."))

    if has_head:
        arm_label = (
            pl.when(pl.col("features") != "embedding")
            .then(pl.lit("clinical baseline"))
            .when(pl.col("correction") == "raw")
            .then(pl.lit("embedding, uncorrected"))
            .when(pl.col("correction") == "site-standardized")
            .then(pl.lit("embedding, site-standardized"))
            .otherwise(pl.lit("embedding, LEACE"))
        )
        order = {
            "clinical baseline": 0,
            "embedding, uncorrected": 1,
            "embedding, site-standardized": 2,
            "embedding, LEACE": 3,
        }

        def head_table(frame: pl.DataFrame, index: list[str]) -> pl.DataFrame:
            wide = (
                frame.with_columns(arm=arm_label)
                .group_by(*index, "arm", "classifier")
                .agg(auroc=pl.col("auroc").mean())
                .pivot(on="classifier", index=index + ["arm"], values="auroc")
            )
            if "gbm" not in wide.columns or "logistic" not in wide.columns:
                return wide
            # pivot orders columns by first appearance in the data, so the two
            # heads can swap places between tables; pin the order explicitly
            return (
                wide.with_columns(
                    _o=pl.col("arm").replace_strict(order, default=9),
                )
                .sort(*index, "_o")
                .select(
                    *index,
                    "arm",
                    "logistic",
                    "gbm",
                    (pl.col("gbm") - pl.col("logistic")).alias("gbm - logistic"),
                )
            )

        cross = results.filter(pl.col("evaluation").str.contains("->"))
        within = results.filter(pl.col("evaluation").str.starts_with("within:"))

        out.append("<h3>Linear vs non-linear head, by correction</h3>")
        out.append(
            "<p>Mean AUROC over both transfer directions and all three "
            "checkpoints. Rows are the feature set and the correction applied "
            "to it; columns are the prediction head on top.</p>"
        )
        table_cross = head_table(cross, ["task"])
        value_cols = {c: num(3) for c in table_cross.columns if c not in ("task", "arm")}
        out.append(table(table_cross, value_cols, key_column="arm"))

        out.append(callout(
            "<b>Non-linearity helps the clinical baseline and hurts the "
            "embedding.</b> On raw labs and vitals a tree gains a great deal "
            "&mdash; most of all for new intubation, where a linear model on "
            "tabular features is close to useless. On the embedding it is never "
            "better. The reading is that the model has already done the "
            "non-linear work: the representation arrives with the interactions "
            "folded in, so a linear head reads it cleanly and a tree only adds "
            "variance. Note this is the opposite of what holds for site, which "
            "after LEACE is recoverable <i>only</i> non-linearly &mdash; outcome "
            "information in these embeddings is linearly accessible while site "
            "information is not, which is why a linear downstream model ignores "
            "site for free. Correction moves nothing in either column."))

        out.append("<h3>Same, within site</h3>")
        out.append(
            "<p>Train and test inside one hospital, averaged over both sites "
            "and all three checkpoints &mdash; free of any site confounding.</p>"
        )
        table_within = head_table(within, ["task"])
        out.append(table(table_within,
                         {c: num(3) for c in table_within.columns
                          if c not in ("task", "arm")},
                         key_column="arm"))

        out.append("<h3>By checkpoint, cross-site</h3>")
        detail = head_table(cross, ["task", "model"])
        out.append(table(detail,
                         {c: num(3) for c in detail.columns
                          if c not in ("task", "model", "arm")},
                         key_column="arm"))

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
        "from sedation in a ventilated ICU population.</li>"
        "<li><b>Phenotypes are a refit</b>, not a transfer: the published "
        "centroids are not available, so k-means is refit and labelled from "
        "centroid profiles. The recovered mortality ordering matches the paper's."
        "</li>"
        "<li><b>PCA and UMAP are refit on whatever cohort is present</b>, so "
        "coordinates shift when a site is added. Figures are not comparable "
        "across runs with different site membership.</li>"
        "</ul>"
    )


# ------------------------------------------------------------------- driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="open in a browser")
    args = parser.parse_args()

    paths = load_paths()
    paths.mkdirs()
    cfg = load_cohort_cfg()
    logger = get_logger("07_report", paths.log_home / "07_report.log")

    out: list[str] = []
    section_cohort(paths, cfg, out)
    section_model(paths, out)
    section_embedding(paths, out)
    section_site_effect(paths, out)
    section_correction(paths, out)
    section_downstream(paths, out, logger)
    section_methods(paths, cfg, out)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    toc = "".join(
        f'<li><a href="#{anchor}">{label}</a></li>'
        for anchor, label in (
            ("cohort", "1 &middot; Cohort"),
            ("model", "2 &middot; Models"),
            ("embedding", "3 &middot; Embedding space"),
            ("site", "4 &middot; Why the sites separate"),
            ("correction", "5 &middot; Can site be removed?"),
            ("downstream", "6 &middot; Downstream prediction"),
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
<p class="meta">Generated {stamp} &middot; 34,142 patients (UCMC 9,135 / NU 25,007)</p>
<ul class="toc">{toc}</ul>
{''.join(out)}
</div></body></html>"""

    destination = paths.output_home / "report.html"
    destination.write_text(document, encoding="utf-8")
    size = destination.stat().st_size / 1e6
    logger.info(f"wrote {destination} ({size:.1f} MB)")
    if args.open:
        webbrowser.open(destination.as_uri())


if __name__ == "__main__":
    main()

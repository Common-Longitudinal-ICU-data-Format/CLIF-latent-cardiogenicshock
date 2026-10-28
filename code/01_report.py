#!/usr/bin/env python3

"""
01_report.py -- what the last 01 run produced, and nothing else

A single self-contained HTML file that reports back the results of
`01_site_tables.py`: the stage ledger, the cohort attrition, the phenotype and
baseline breakdowns, the token prevalence, the trial analysis, and the bundle
each site ends up sending.

    python code/01_report.py                              # a site, its own run
    python code/01_report.py --bundles output/received    # a coordinator
    python code/01_report.py --open

TWO DEPLOYMENTS, ONE SET OF SECTIONS. A contributing site reads its own
`output/cohort/`, before it sends anything. A coordinating site has no cohort
directory at all -- it has the bundles the sites uploaded -- and reads those
instead, which is the default once `output/cohort/` is absent. The difference
is not cosmetic: a bundle carries only what the export contract permits, so
four of the tables below are missing from it by design, and the report says
which rather than letting them read as stages that did not run. See
`BUNDLE_LAYOUT` and `NOT_SHIPPED`.

DELIBERATELY NARROW. 07_report.py assembles everything 01-13 wrote into one
document and is thirty sections long; this one reads the cohort tables or the
bundles and stops. The reason to want it is that 01 is
the stage a contributing site runs on its own machine, often before any model
weights are on it at all -- so the person who needs to check the run is the
person who cannot build the main report. Nothing here touches a figure, an
embedding or a downstream table, and so nothing here can fail because tier 2
has not run.

READS NO PATIENT RECORDS. Every number comes from a CSV that 01 already
suppressed below `--min-cell`, or from a bundle manifest. `cohort.parquet` sits
in the same directory and is patient-level; it is not opened, so this report
runs under the pooled role and may go anywhere the bundle itself may go. Read
from bundles, it is strictly weaker still: the only files it opens are ones
that have already crossed a site boundary.

A STAGE WITH NO OUTPUT SAYS SO rather than being omitted. A reader of a partial
run has to be able to tell a stage that found nothing from a stage that never
ran, and an absent section says neither.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import html
import importlib.util
import json
import pathlib
import sys
import webbrowser

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import ANCHORS, get_logger, load_paths


def driver():
    """
    01_site_tables.py as a module, for its stage list and stage descriptions

    Imported rather than restated. The stage table is the spine of this report,
    and a copy of it would drift the first time a stage was added -- which is
    exactly when a run report gets read, and exactly the stage it would be
    missing. The module name starts with a digit, so `import` cannot reach it.

    Costs a few seconds, because importing it imports polars and the clifcs
    cohort code. It opens no data file: everything expensive in 01 is inside a
    stage function.
    """
    path = pathlib.Path(__file__).resolve().parent / "01_site_tables.py"
    spec = importlib.util.spec_from_file_location("clifcs_driver_01", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: stage -> the glob patterns it writes under `cohort_home`
#:
#: The five middle stages write into scratch_home instead and are handled
#: separately: their products are gigabytes of intermediate timeline, there is
#: nothing in them to summarise, and they are not meant to outlive the run.
STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "cohort": (
        "cohort.parquet",
        "cs_onset.parquet",
        "filter_ledger.csv",
        "shock_type_summary.csv",
        "phenotype_centroids*.csv",
        "seymour_*.csv",
        "bhavani_*.csv",
        "*_model.json",
    ),
    "phenotypes": (
        "phenotype_summary*.csv",
        "phenotype_variant_agreement.csv",
    ),
    "tokencounts": (
        "token_prevalence_*.csv",
        "token_families_*.csv",
    ),
    "baseline": ("baseline_by_shock_type_*.csv",),
    "trials": (
        "trial_criteria.csv",
        "trial_coverage_*.csv",
        "trial_guideline_assignment_*.csv",
        "trial_iqr_position_*.csv",
        "trial_outcomes_*.csv",
        "trial_patient_anchor-*.parquet",
    ),
}

#: the stages whose work lands in scratch: stage -> (which `Paths` method
#: finds the directory, what to look for in it)
#:
#: The patterns are the files 01 itself checks before deciding to skip a
#: stage, so a stage reads as present here exactly when 01 would log
#: "already done, skipping". Globbing the whole directory instead would make
#: the pairs that share one -- `subset` with `derive`, `collate` with
#: `tokenize` -- each report the other's files, and would count the
#: `features-*.parquet` shards that `cotorra extract` writes into the anchor
#: directories as though 01 had produced them.
SCRATCH_OUTPUTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "subset": ("raw_subset", ("clif_*.parquet",)),
    "derive": ("raw_subset", None),  # DERIVED_TABLES, below
    "collate": ("processed", ("meds.parquet",)),
    "tokenize": ("processed", ("tokens_times.parquet", "tokenizer.yaml")),
    "winnow": (
        "anchored",
        (
            "*_for_inference.parquet",
            "*_tokens_times.parquet",
            "tokens_times.parquet",
            "subject_splits.parquet",
            "tokenizer.yaml",
        ),
    ),
}

#: what `derive` adds to the subset directory that `subset` did not put there
#:
#: Listed rather than inferred, and subtracted from `subset` as well as
#: matched for `derive`. The two stages write into one directory, and a ledger
#: in which both claim all 52 files says nothing about either.
DERIVED_TABLES = (
    "clif_sofa.parquet",
    "clif_*_converted.parquet",
    "clif_respiratory_support_processed.parquet",
)
SCRATCH_OUTPUTS["derive"] = ("raw_subset", DERIVED_TABLES)

#: `STAGE_PURPOSE` in 01 predates the `phenotypes` stage and does not name it
EXTRA_PURPOSE: dict[str, str] = {
    "phenotypes": "summarising the phenotype fits per denominator",
}


STYLE = """
:root { --ink:#1b1b1b; --muted:#6a6a6a; --rule:#e3e3e3; --accent:#2166ac;
        --warn:#c0392b; }
* { box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
       color:var(--ink); max-width:1180px; margin:0 auto; padding:40px 28px;
       line-height:1.55; font-size:15px; }
h1 { font-size:27px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:21px; margin:46px 0 10px; padding-bottom:7px;
     border-bottom:2px solid var(--ink); }
h3 { font-size:17px; margin:30px 0 8px; }
h4 { font-size:13px; margin:22px 0 6px; color:var(--muted);
     text-transform:uppercase; letter-spacing:0.04em; }
p { margin:9px 0; max-width:84ch; }
code { background:#f3f3f3; padding:1px 5px; border-radius:3px;
       font-size:13px; font-family:"SF Mono",Menlo,Consolas,monospace; }
table { border-collapse:collapse; width:100%; margin:14px 0; font-size:13px; }
th { text-align:right; padding:7px 9px; border-bottom:2px solid var(--ink);
     font-weight:600; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
td { padding:6px 9px; border-bottom:1px solid var(--rule); text-align:right;
     font-variant-numeric:tabular-nums; }
td.text, th.text { text-align:left; white-space:normal; }
tr:hover td { background:#fafafa; }
tr.rule td { font-weight:600; color:var(--ink);
             border-bottom:1px solid var(--ink); }
.callout { border-left:3px solid var(--accent); background:#f5f8fb;
           padding:12px 16px; margin:16px 0; font-size:14px; }
.callout.warn { border-left-color:var(--warn); background:#fdf5f3; }
.callout p { margin:0; max-width:none; }
.meta { color:var(--muted); font-size:13px; margin-bottom:26px; }
.absent { color:var(--muted); font-style:italic; }
.toc { columns:2; list-style:none; padding:0; margin:26px 0 0; }
.toc li { margin:3px 0; }
.toc a { color:var(--accent); text-decoration:none; font-size:13.5px; }
.toc a:hover { text-decoration:underline; }
.tabs { margin:16px 0 8px; }
.tabstrip { display:flex; flex-wrap:wrap; gap:2px;
            border-bottom:1px solid var(--rule); margin-bottom:2px; }
.tabstrip button { appearance:none; border:0; background:transparent;
    cursor:pointer; font:inherit; font-size:12.5px; color:var(--muted);
    padding:7px 13px; border-radius:4px 4px 0 0;
    border-bottom:2px solid transparent; margin-bottom:-1px; }
.tabstrip button:hover { background:#f3f3f3; color:var(--ink); }
.tabstrip button[aria-selected="true"] { color:var(--ink); font-weight:600;
    border-bottom-color:var(--accent); }
.tabpanel { display:none; padding-top:6px; }
.tabpanel[data-active="true"] { display:block; }
"""

SCRIPT = """
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


# ----------------------------------------------------------------- rendering


def esc(text) -> str:
    return html.escape("" if text is None else str(text))


def callout(text: str, kind: str = "") -> str:
    return f'<div class="callout {kind}"><p>{text}</p></div>'


def absent(what: str, produced_by: str = "") -> str:
    """a table that is not there, said out loud rather than skipped"""
    tail = f" Produced by {produced_by}." if produced_by else ""
    return f'<p class="absent">Not present: {what}.{tail}</p>'


_TAB_GROUP = [0]


def tabs(items) -> str:
    """a tab group from (label, html) pairs; empty panels are dropped"""
    items = [(label, body) for label, body in items if body]
    if not items:
        return ""
    if len(items) == 1:
        return items[0][1]
    _TAB_GROUP[0] += 1
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
        f'<div class="tabs" data-tabs id="tabs-{_TAB_GROUP[0]}">'
        f'<div class="tabstrip" role="tablist">{strip}</div>{panels}</div>'
    )


def kind_of(column: str) -> str:
    """
    how a column is formatted, by name rather than by dtype

    A count and a share are both numbers and only the name knows which is
    which. This pipeline names them consistently -- `n_*` counts, `share_*`
    shares, `median_*` medians -- so one rule covers every table 01 writes,
    where a per-table map would have to be kept in step with it. Anything
    unmatched falls through to the dtype.
    """
    if column == "token_id":  # an identifier that happens to be an integer
        return "id"
    if column in ("n", "len") or column.startswith("n_"):
        return "int"
    if column in ("total_occurrences", "distinct_tokens"):
        return "int"
    if column in ("prevalence", "availability"):
        return "pct"
    if column.startswith(("share", "mortality", "same_", "null_")):
        return "pct"
    if column.startswith("median_") or column in ("median", "q1", "q3"):
        return "num1"
    if column.startswith(("lift_", "ratio_", "adjusted_")):
        return "num3"
    return ""


#: columns whose content is prose and must wrap rather than stretch the table
TEXT_COLUMNS = frozenset(
    {
        "step", "note", "criterion", "kind", "variable", "token_name", "trial",
        "assigned_guideline", "guideline_fit", "stage", "writes", "file",
        "field", "value", "purpose", "applied_here",
    }
)


def fmt(value, kind: str) -> str:
    """
    one cell, formatted by what its column is rather than by its dtype

    `kind` can be right about the column and wrong about the type that reached
    it. Concatenating one site's table with another's under
    `diagonal_relaxed` widens a column to String whenever a site wrote it
    all-null -- `n_variables_compared` is Int64 at RUSH and an empty String
    column at NU and UCMC -- so a count arrives here as "4". Coerced rather
    than rejected: the alternative is a report that cannot render a table
    because one site had nothing to put in a column.
    """
    if value is None:
        return "&mdash;"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if kind in ("int", "pct", "num1", "num3") and isinstance(value, str):
        try:
            value = float(value)
        except ValueError:  # named like a number, written as text
            return esc(value)
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "int":
        return f"{value:,.0f}"
    if kind == "num1":
        return f"{value:,.1f}"
    if kind == "num3":
        return f"{value:.3f}"
    if kind == "id":
        return esc(value)
    # The dtype fallback, for columns no rule names. It has to group thousands
    # too: the attrition ledger is pivoted on site, so its count columns are
    # called UCMC and NU and no name-based rule can reach them.
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return esc(value)


def table(frame: pl.DataFrame, limit=None, rule_rows=frozenset()) -> str:
    """
    a frame as HTML

    `rule_rows` names values of the first column to underline -- the criterion
    rows in the attrition ledger, which are the lines a reader is counting down
    to and which the component rows between them otherwise bury.
    """
    if frame is None or frame.is_empty():
        return "<p><i>(no rows)</i></p>"
    truncated = limit is not None and frame.height > limit
    shown = frame.head(limit) if truncated else frame
    kinds = {c: kind_of(c) for c in shown.columns}
    head = "".join(
        f'<th class="{"text" if c in TEXT_COLUMNS else ""}">'
        f"{esc(c.replace('_', ' '))}</th>"
        for c in shown.columns
    )
    key = shown.columns[0]
    body = []
    for row in shown.iter_rows(named=True):
        cells = "".join(
            f'<td class="{"text" if c in TEXT_COLUMNS else ""}">'
            f"{fmt(row[c], kinds[c])}</td>"
            for c in shown.columns
        )
        classes = ' class="rule"' if str(row[key]) in rule_rows else ""
        body.append(f"<tr{classes}>{cells}</tr>")
    rendered = (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )
    if truncated:
        rendered += (
            f'<p class="absent">{shown.height:,} of {frame.height:,} rows '
            "shown; the whole table is in the CSV it was read from.</p>"
        )
    return rendered


def read(path: pathlib.Path):
    """a CSV if it is there, else None -- every section branches on this"""
    return pl.read_csv(path) if path.is_file() else None


def kv_table(pairs) -> str:
    """field/value rows, for a manifest"""
    return table(
        pl.DataFrame(
            {
                "field": [str(k) for k, _ in pairs],
                "value": [
                    ""
                    if v is None
                    else "yes"
                    if v is True
                    else "no"
                    if v is False
                    else str(v)
                    for _, v in pairs
                ],
            }
        )
    )


def blanked(frame: pl.DataFrame) -> tuple[int, int]:
    """
    (suppressed cells, suppressible cells) in one of 01's aggregate tables

    A suppressed cell is a null in a count or in a statistic derived from one.
    Counted rather than asserted: 01 blanks below `--min-cell` on the way into
    a table and again on the way into the bundle, and a run that blanked
    nothing anywhere is likelier to have had the threshold misconfigured than
    to have had no small cells.
    """
    columns = [
        c for c in frame.columns if kind_of(c) in ("int", "pct", "num1", "num3")
    ]
    if not columns:
        return 0, 0
    nulls = int(
        frame.select(
            pl.sum_horizontal(pl.col(c).is_null() for c in columns).alias("n")
        )["n"].sum()
    )
    return nulls, frame.height * len(columns)


# ------------------------------------------------- where the tables come from
#
# Two deployments read this report, and they do not hold the same files.
#
# A CONTRIBUTING SITE reads its own `output/cohort/`, before it sends anything.
# That is the case the report was written for, and the one where a stage with
# no output means a stage to rerun.
#
# A COORDINATING SITE has no cohort directory at all. What it has is the
# bundles the sites uploaded, dropped in verbatim -- and a bundle is not a copy
# of `output/cohort/`. The export stage strips the `_<SITE>` infix from every
# filename (one site per bundle makes it redundant), sorts the tables into
# counts/ and trials/, and leaves behind everything the contract does not
# carry. So the coordinator reads the same nine sections over a different and
# smaller set of files, and the sections must not have to know which.
#
# Hence two sources behind one interface. Which one is used is decided in
# `main` and said out loud at the top of the report, because "this stage
# produced nothing" and "this stage's output does not cross a site boundary"
# are different facts and a reader has to be able to tell them apart.


#: local pattern under `cohort_home` -> the same table(s) inside a bundle
#:
#: Written out rather than derived by stripping `_<SITE>`. The correspondence
#: is a contract between two scripts, it is not quite regular -- the token and
#: trial tables keep an `anchor-*` infix that the summaries do not have -- and
#: a rule that guessed would fail silently by finding nothing, which in this
#: report renders as "the stage did not run".
BUNDLE_LAYOUT: dict[str, str] = {
    "filter_ledger.csv": "counts/filter_ledger.csv",
    "shock_type_summary.csv": "counts/shock_type_summary.csv",
    "phenotype_summary.csv": "counts/phenotype_summary.csv",
    "seymour_summary.csv": "counts/seymour_summary.csv",
    "bhavani_summary.csv": "counts/bhavani_summary.csv",
    "seymour_availability.csv": "counts/seymour_availability.csv",
    "baseline_by_shock_type_*.csv": "counts/baseline_by_shock_type.csv",
    "token_prevalence_*.csv": "counts/token_prevalence_anchor-*.csv",
    "token_families_*.csv": "counts/token_families_anchor-*.csv",
    "token_coverage.csv": "counts/token_coverage.csv",
    "trial_criteria.csv": "trials/trial_criteria.csv",
    "trial_outcomes_*.csv": "trials/trial_outcomes_anchor-*.csv",
    "trial_iqr_position_*.csv": "trials/trial_iqr_position_anchor-*.csv",
    "trial_coverage_*.csv": "trials/trial_coverage_anchor-*.csv",
    "trial_guideline_assignment_*.csv": (
        "trials/trial_guideline_assignment_anchor-*.csv"
    ),
    "run_stages.csv": "run_stages.csv",
}

#: what 01 writes that no bundle carries, and the reason in each case
#:
#: Reported as a group at the top of a coordinator's report, so that every
#: "Not present" below it is read correctly. Three of the four are deliberate
#: and one is a gap: `*_centroids.csv` is the only evidence this report has for
#: whether a phenotype scheme was refit or transferred, and withholding it is
#: right -- a centroid is a mean over a cluster whose size nothing checks --
#: but it leaves a coordinator unable to answer the question at all. That
#: belongs in the manifest.
NOT_SHIPPED: dict[str, str] = {
    "phenotype_summary_*.csv": (
        "the Zweck fit per cardiogenic denominator; only the pooled summary is "
        "in the contract"
    ),
    "phenotype_variant_agreement.csv": (
        "how the three Zweck denominators agree with each other"
    ),
    "seymour_agreement.csv": (
        "the refit-against-published cross-tabulation, whose cells fall under "
        "the minimum cell size"
    ),
    "*_centroids*.csv": (
        "the cluster centroids, which are how this report tells a local refit "
        "from a transferred model"
    ),
}

#: stage -> what its output looks like inside a bundle
#:
#: The five scratch stages are absent on purpose: they write gigabytes of
#: intermediate timeline that never leaves the site, so a bundle holds no
#: evidence they ran. `run_stages.csv` is that evidence, which is why the
#: ledger reads it.
BUNDLE_STAGE_FILES: dict[str, tuple[str, ...]] = {
    "cohort": (
        "counts/filter_ledger.csv",
        "counts/shock_type_summary.csv",
        "counts/seymour_summary.csv",
        "counts/bhavani_summary.csv",
        "counts/seymour_availability.csv",
    ),
    "phenotypes": ("counts/phenotype_summary.csv",),
    "tokencounts": (
        "counts/token_prevalence_anchor-*.csv",
        "counts/token_families_anchor-*.csv",
        "token_levels.csv",
    ),
    "baseline": ("counts/baseline_by_shock_type.csv",),
    "trials": ("trials/*.csv",),
}


def _stack(frames: list):
    return pl.concat(frames, how="diagonal_relaxed") if frames else None


class LocalRun:
    """
    the tables as 01 wrote them, in this machine's `output/cohort/`

    The site's own view, read before it sends anything. Everything is here,
    including the files the contract does not carry.
    """

    federated = False

    def __init__(self, paths, logger):
        self.paths = paths
        self.logger = logger
        self.home = paths.cohort_home
        self.where = str(paths.cohort_home)
        self.bundles = {
            site: paths.upload_home(site)
            for site in paths.sites
            if paths.upload_home(site).is_dir()
        }

    @property
    def sites(self) -> tuple:
        return tuple(self.paths.sites)

    @property
    def manifests(self) -> dict:
        return {
            site: json.loads(path.read_text(encoding="utf-8"))
            for site, bundle in self.bundles.items()
            for path in [bundle / "manifest.json"]
            if path.is_file()
        }

    def has(self, pattern: str) -> bool:
        return any(f.is_file() for f in self.home.glob(pattern))

    def read(self, pattern: str):
        files = sorted(f for f in self.home.glob(pattern) if f.is_file())
        return _stack([pl.read_csv(f) for f in files])

    def tokenizer_sites(self) -> tuple:
        """(sites with a tokenizer, sites without) -- machine-wide here"""
        return (self.sites, ()) if self.paths.has_tokenizer else ((), self.sites)

    def audit(self) -> list:
        return [
            (path.name, pl.read_csv(path))
            for path in sorted(self.home.glob("*.csv"))
        ]

    def stage_files(self, stage: str) -> tuple:
        """every file one stage wrote, and the directory it writes into"""
        paths = self.paths
        if stage in STAGE_OUTPUTS:
            files = {
                f
                for pattern in STAGE_OUTPUTS[stage]
                for f in paths.cohort_home.glob(pattern)
                if f.is_file()
            }
            return sorted(files), "output/cohort"
        if stage == "export":
            files = [
                f
                for bundle in self.bundles.values()
                for f in bundle.rglob("*")
                if f.is_file()
            ]
            return sorted(files), "output/upload_to_box_<SITE>"
        resolver, patterns = SCRATCH_OUTPUTS[stage]
        if resolver == "anchored":
            directories = [
                paths.anchored(site, anchor)
                for site in paths.sites
                for anchor in ANCHORS
            ]
        else:
            directories = [getattr(paths, resolver)(site) for site in paths.sites]
        files = {
            f
            for directory in directories
            if directory.is_dir()
            for pattern in patterns
            for f in directory.glob(pattern)
            if f.is_file()
        }
        if stage == "subset":
            files = {
                f
                for f in files
                if not any(fnmatch.fnmatch(f.name, p) for p in DERIVED_TABLES)
            }
        return sorted(files), "scratch_home"


class ReceivedBundles:
    """
    the tables as the sites sent them, one `upload_to_box_<SITE>/` per site

    The site a row belongs to comes from the bundle's manifest, not from its
    directory name and not from its filename: a directory can be renamed in
    transit, and export strips the site out of the filename because one bundle
    is one site. Four tables -- the three phenotype summaries and the Seymour
    availability table -- carry no `site` column either, so it is attached
    here, which is the only reason two bundles can be stacked at all.
    """

    federated = True

    def __init__(self, home, logger):
        self.home = pathlib.Path(home)
        self.logger = logger
        self.where = str(self.home)
        self.paths = None
        self.bundles: dict = {}
        self.manifests: dict = {}
        for candidate in sorted(p for p in self.home.iterdir() if p.is_dir()):
            if not candidate.name.startswith("upload_to_box_"):
                continue
            meta_file = candidate / "manifest.json"
            if not meta_file.is_file():
                logger.warning(
                    f"  {candidate.name}/ has no manifest.json and is being "
                    "skipped -- without one there is no way to tell which site "
                    "it came from or what code built it"
                )
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            site = meta.get("site") or candidate.name
            self.bundles[site] = candidate
            self.manifests[site] = meta
            logger.info(f"  {site}: {candidate.name}/")

    @property
    def sites(self) -> tuple:
        return tuple(self.bundles)

    def _resolve(self, pattern: str):
        target = BUNDLE_LAYOUT.get(pattern)
        if target is not None:
            return target
        if any(fnmatch.fnmatch(pattern, p) for p in NOT_SHIPPED):
            return None
        raise KeyError(
            f"{pattern!r} is read from the cohort directory but has no entry "
            "in BUNDLE_LAYOUT and is not listed in NOT_SHIPPED, so a "
            "coordinator reading bundles would report it as a stage that did "
            "not run. Add it to one of the two."
        )

    def _files(self, pattern: str) -> list:
        target = self._resolve(pattern)
        if target is None:
            return []
        subdirectory, _, leaf = target.rpartition("/")
        found = []
        for site, bundle in self.bundles.items():
            directory = bundle / subdirectory if subdirectory else bundle
            if not directory.is_dir():
                continue
            found += [(site, f) for f in sorted(directory.glob(leaf)) if f.is_file()]
        return found

    def has(self, pattern: str) -> bool:
        return bool(self._files(pattern))

    def read(self, pattern: str):
        frames = []
        for site, path in self._files(pattern):
            frame = pl.read_csv(path)
            if "site" not in frame.columns:
                frame = frame.with_columns(site=pl.lit(site))
            # The anchor is in the filename for the token and trial tables and
            # in a column for most of them. Filled from the name where it is
            # missing, because every per-anchor section tabs on it.
            if "anchor" not in frame.columns and "anchor-" in path.stem:
                frame = frame.with_columns(
                    anchor=pl.lit(path.stem.split("anchor-")[-1])
                )
            frames.append(frame)
        return _stack(frames)

    def tokenizer_sites(self) -> tuple:
        """
        (sites that held a tokenizer, sites that did not)

        Per site rather than per machine. A site with no tokenizer skips the
        three token stages and sends everything else, so its bundle carries no
        token tables and a null `tokenizer_sha256` -- a fact about that site,
        which must not read as a missing file.
        """
        with_tokenizer = tuple(
            site
            for site, meta in self.manifests.items()
            if meta.get("tokenizer_sha256")
        )
        return with_tokenizer, tuple(
            s for s in self.bundles if s not in set(with_tokenizer)
        )

    def audit(self) -> list:
        return [
            (f"{site}/{path.relative_to(bundle).as_posix()}", pl.read_csv(path))
            for site, bundle in self.bundles.items()
            for path in sorted(bundle.rglob("*.csv"))
        ]

    def stage_files(self, stage: str) -> tuple:
        if stage == "export":
            files = [
                f
                for bundle in self.bundles.values()
                for f in bundle.rglob("*")
                if f.is_file()
            ]
            return sorted(files), "the bundle itself"
        patterns = BUNDLE_STAGE_FILES.get(stage)
        if patterns is None:
            return [], "scratch_home, which no bundle carries"
        files = set()
        for pattern in patterns:
            subdirectory, _, leaf = pattern.rpartition("/")
            for bundle in self.bundles.values():
                directory = bundle / subdirectory if subdirectory else bundle
                if directory.is_dir():
                    files |= {f for f in directory.glob(leaf) if f.is_file()}
        return sorted(files), "upload_to_box_<SITE>"


# ------------------------------------------------------------------ sections


def section_stages(source, module, out: list, logger) -> None:
    """
    the run ledger: every stage 01 defines, and how much of it is on disk

    This is the section the report exists for. Everything below is the content
    of the tables; this says which of them the run actually produced, and when
    -- the question anyone asks of a run that was resumed, interrupted, or
    executed one stage at a time.
    """
    out.append('<h2 id="stages">1 &middot; Stage ledger</h2>')
    out.append(
        f"<p>The {len(module.STAGES)} stages "
        "<code>01_site_tables.py</code> defines, in the order it runs them, "
        f"against what is in <code>{esc(source.where)}</code>. "
        "<b>Newest write</b> is the most recent mtime among the outputs of a "
        "stage, so a stage rerun on its own stands out from the stages around "
        "it: timestamps that disagree mean the tables below were not all built "
        "from the same cohort.</p>"
    )
    out.append(
        "<p>The five middle stages write gigabytes of intermediate timeline "
        "into <code>scratch_home</code> rather than tables. They are counted "
        "by the files 01 itself looks for before deciding to skip them, so a "
        "stage reads as present here exactly when a rerun would report it "
        "already done. There is nothing in them to summarise and they are not "
        "meant to outlive the run.</p>"
    )
    out.append(
        "<p><code>cohort.parquet</code> is counted among the "
        "<code>cohort</code> stage's files and is the one patient-level thing "
        "01 writes. This report lists it and never opens it, which is what "
        "lets the report itself be sent wherever the bundle may go.</p>"
    )

    rows, empty = [], []
    for position, stage in enumerate(module.STAGES, start=1):
        files, where = source.stage_files(stage)
        if not files:
            empty.append(stage)
        newest = (
            dt.datetime.fromtimestamp(max(f.stat().st_mtime for f in files))
            if files
            else None
        )
        rows.append(
            {
                "stage": f"{position}. {stage}",
                "purpose": module.STAGE_PURPOSE.get(stage)
                or EXTRA_PURPOSE.get(stage, ""),
                "writes": where,
                "n_files": len(files),
                "size_mb": sum(f.stat().st_size for f in files) / 1e6,
                "newest_write": newest.strftime("%Y-%m-%d %H:%M")
                if newest
                else None,
            }
        )
    out.append(table(pl.DataFrame(rows)))

    if empty:
        out.append(
            callout(
                f"<b>{len(empty)} of {len(rows)} stages have no output on this "
                "machine:</b> "
                + ", ".join(f"<code>{esc(s)}</code>" for s in empty)
                + ". Sections below that depend on them say so in place of "
                "their tables. Resume with <code>python "
                f"code/01_site_tables.py --from-stage {esc(empty[0])}</code>.",
                "warn",
            )
        )
        logger.info(f"stages with no output: {', '.join(empty)}")
    else:
        logger.info(f"all {len(rows)} stages have output on disk")

    _, untokenized = source.tokenizer_sites()
    if untokenized:
        # Per site, not per machine. At a coordinator this is read out of each
        # bundle's manifest: one site holding no tokenizer is a fact about that
        # site, and reporting it machine-wide would either hide it or libel the
        # others.
        out.append(
            callout(
                "<b>No training-time tokenizer at "
                + ", ".join(f"<code>{esc(s)}</code>" for s in untokenized)
                + "</b>, so <code>"
                + "</code>, <code>".join(module.TOKEN_STAGES)
                + "</code> are skipped as a group rather than failed. "
                "Everything else 01 reports &mdash; the cohort, the three "
                "phenotype breakdowns, the baseline tables and the whole trial "
                "and guideline analysis &mdash; is computed from CLIF and the "
                "published literature and never sees a token, so such a run is "
                "still a complete contribution minus token prevalence.",
                "warn",
            )
        )

    # What the run itself recorded, where it travelled. The five scratch
    # stages leave no file a bundle carries, so at a coordinator this ledger
    # is the only thing that says whether they ran -- and it is the only place
    # a FAILED stage is visible at all, since a stage that died still leaves
    # whatever it had written.
    recorded = source.read("run_stages.csv")
    if recorded is not None:
        out.append("<h3>What the run recorded</h3>")
        out.append(
            "<p>Written by 01 after each stage and carried in the bundle. The "
            "table above is inferred from files on disk; this one is the run "
            "saying what it did, which is the only evidence for the five "
            "stages whose output never leaves the site.</p>"
        )
        out.append(table(recorded))
        failed = recorded.filter(pl.col("status") != "ok")
        if failed.height:
            out.append(
                callout(
                    "<b>"
                    + ", ".join(
                        f"<code>{esc(r)}</code>" for r in failed["stage"].to_list()
                    )
                    + " did not finish.</b> A stage that failed still leaves "
                    "whatever it had written by then, so the tables below may "
                    "be partial in a way no file listing reveals.",
                    "warn",
                )
            )
    elif source.federated:
        out.append(
            callout(
                "<b>No <code>run_stages.csv</code> in any bundle.</b> Nothing "
                "here can say whether <code>"
                + "</code>, <code>".join(
                    s for s in module.STAGES if s in SCRATCH_OUTPUTS
                )
                + "</code> ran: they write only into the sending site's "
                "scratch directory. Ask each site to re-run "
                "<code>--stage export</code> with current code, which adds the "
                "ledger to the bundle.",
                "warn",
            )
        )


def section_provenance(source, out: list) -> None:
    """what code, model and tokenizer each bundle was built by"""
    out.append('<h2 id="provenance">2 &middot; Provenance</h2>')
    manifests = source.manifests
    if not manifests:
        out.append(
            absent(
                "no <code>manifest.json</code> for any configured site",
                "the <code>export</code> stage",
            )
        )
        return
    out.append(
        "<p>Read from the <code>manifest.json</code> the export stage writes "
        "beside the tables it sends. The git SHA and the weight and tokenizer "
        "hashes are what make two sites' contributions comparable; a pooled "
        "analysis over bundles built by different code is the failure this "
        "records.</p>"
    )
    out.append(
        tabs(
            [
                (
                    site,
                    kv_table(
                        [
                            ("site", manifest.get("site")),
                            ("created (UTC)", manifest.get("created_utc")),
                            ("schema version", manifest.get("schema_version")),
                            ("git SHA", (manifest.get("code") or {}).get("git_sha")),
                            ("git branch", (manifest.get("code") or {}).get("git_branch")),
                            ("git dirty", (manifest.get("code") or {}).get("git_dirty")),
                            ("model", (manifest.get("model") or {}).get("key")),
                            ("weights sha256", (manifest.get("model") or {}).get("weights_sha256")),
                            ("embedding dim", (manifest.get("model") or {}).get("embedding_dim")),
                            ("tokenizer sha256", manifest.get("tokenizer_sha256")),
                            ("anchors", ", ".join(manifest.get("anchors") or [])),
                            (
                                "rows per anchor",
                                ", ".join(
                                    f"{anchor}: {n:,}"
                                    for anchor, n in (
                                        manifest.get("row_counts") or {}
                                    ).items()
                                ),
                            ),
                            ("minimum cell size", manifest.get("min_cell")),
                            ("shares patient-level data", manifest.get("shares_patient_level")),
                            ("salt fingerprint", manifest.get("salt_fingerprint")),
                        ]
                    ),
                )
                for site, manifest in manifests.items()
            ]
        )
    )

    shas = {(m.get("code") or {}).get("git_sha") for m in manifests.values()}
    if len(shas) > 1:
        out.append(
            callout(
                f"<b>The {len(manifests)} bundles were built by {len(shas)} "
                "different commits.</b> Rebuild the older ones before pooling: "
                "a difference in the cohort definition between two sites reads "
                "downstream as a difference between the sites.",
                "warn",
            )
        )
    dirty = [
        s for s, m in manifests.items() if (m.get("code") or {}).get("git_dirty")
    ]
    if dirty:
        out.append(
            callout(
                "<b>Built from a dirty working tree at "
                + ", ".join(esc(s) for s in dirty)
                + ".</b> The recorded SHA does not fully describe the code "
                "that produced these tables.",
                "warn",
            )
        )
    thresholds = {m.get("min_cell") for m in manifests.values()}
    if len(thresholds) > 1:
        out.append(
            callout(
                "<b>The bundles were suppressed at different cell sizes: "
                + ", ".join(str(t) for t in sorted(thresholds, key=str))
                + ".</b> Pooled tables built from them are then only as "
                "suppressed as the loosest of the three, whichever site that "
                "was.",
                "warn",
            )
        )


def section_cohort(source, out: list) -> None:
    """the attrition ledger and the coded mechanism"""
    out.append('<h2 id="cohort">3 &middot; Cohort</h2>')
    ledger = source.read("filter_ledger.csv")
    if ledger is None:
        out.append(
            absent(
                "<code>filter_ledger.csv</code>", "the <code>cohort</code> stage"
            )
        )
    else:
        out.append(
            "<p>Three criteria applied in sequence to a base cohort of first "
            "hospitalizations at least a day long with an ICU stay in the "
            "first 24&nbsp;h &mdash; which mirrors the preprocessing the "
            "models were trained under. Criterion&nbsp;3, cardiac origin, is "
            "<b>computed and reported but not applied</b> unless "
            "<code>cardiac_origin.enabled</code> is set, so by default these "
            "rows describe an undifferentiated shock cohort and the mechanism "
            "is a labelling attached afterwards.</p>"
        )
        main = ledger.filter(~pl.col("step").str.starts_with(" "))
        order = main["step"].unique(maintain_order=True)
        milestones = {s for s in order if "CRITERION" in s or "BASE COHORT" in s}
        components = ledger.filter(
            pl.col("step").str.starts_with(" ")
        ).with_columns(step=pl.col("step").str.strip_chars())

        def in_ledger_order(frame: pl.DataFrame, steps) -> pl.DataFrame:
            """pivoting loses the sequence, and the sequence is the table"""
            return (
                frame.join(
                    pl.DataFrame({"step": steps}).with_row_index("position"),
                    on="step",
                    how="inner",
                )
                .sort("position")
                .drop("position")
            )

        def by_site(frame: pl.DataFrame, values: str) -> pl.DataFrame:
            return frame.pivot(
                on="site", index="step", values=values, aggregate_function="first"
            )

        out.append(
            tabs(
                [
                    (
                        "Attrition (hospitalizations)",
                        table(
                            in_ledger_order(
                                by_site(main, "n_hospitalizations"), order
                            ),
                            rule_rows=milestones,
                        ),
                    ),
                    (
                        "Attrition (patients)",
                        table(
                            in_ledger_order(by_site(main, "n_patients"), order),
                            rule_rows=milestones,
                        ),
                    ),
                    (
                        "Criterion components",
                        table(
                            in_ledger_order(
                                by_site(components, "n_hospitalizations"),
                                components["step"].unique(maintain_order=True),
                            )
                        ),
                    ),
                ]
            )
        )
        out.append(
            "<p>Rows marked <code>[sub]</code> are the arms of the criterion "
            "they sit under and overlap each other, so they do not sum to it. "
            "<code>[arm]</code> rows are the three routes to cardiac origin. A "
            "<code>note</code> of <i>reported only</i> against the "
            "cardiac-origin row is how the ledger records a criterion that was "
            "computed and not applied.</p>"
        )

    out.append("<h3>Shock type, from coded diagnoses</h3>")
    shock = source.read("shock_type_summary.csv")
    if shock is None:
        out.append(
            absent(
                "<code>shock_type_summary.csv</code>",
                "the <code>cohort</code> stage",
            )
        )
    else:
        out.append(
            "<p>What the coders said the mechanism was, from "
            "<code>clif_hospital_diagnosis</code> &mdash; a labelling, never a "
            "filter. Codes overlap, so a patient carrying more than one of "
            "septic, cardiogenic or hypovolaemic is <code>mixed</code>. This "
            "is the one place in 01 that reads diagnosis codes: the mechanism "
            "of shock has no corresponding measurement, so the code is the "
            "only evidence there is. <b>Compare composition within a site, "
            "never across one</b> &mdash; these are documentation habits as "
            "much as case mixes, which is why the table is never pooled.</p>"
        )
        out.append(
            tabs(
                [
                    (
                        str(site),
                        table(shock.filter(pl.col("site") == site).drop("site")),
                    )
                    for site in shock["site"].unique(maintain_order=True).to_list()
                ]
            )
        )


#: the three cardiogenic denominators Zweck is fitted on, one tab each
ZWECK_VARIANTS = (
    ("cs_phenotype_coded", "Zweck (ICD-coded CS)"),
    ("cs_phenotype_cardiac_origin", "Zweck (cardiac origin)"),
    ("cs_phenotype", "Zweck (all shock)"),
)


#: scheme -> (the summary it writes, the centroid table it writes only on a fit)
#:
#: 01 writes a centroid table and a model file when it FITS a scheme, and
#: neither when it applies one fitted elsewhere -- `apply_model` returns
#: labels only. So a `*_centroids.csv` sitting beside a `*_summary.csv` is the
#: evidence that this site refit rather than transferred.
FIT_EVIDENCE = (
    ("Zweck", "phenotype_summary.csv", "phenotype_centroids_*.csv"),
    ("Seymour", "seymour_summary.csv", "seymour_centroids.csv"),
    ("Bhavani", "bhavani_summary.csv", "bhavani_centroids.csv"),
)


def section_fit_provenance(source, out: list) -> None:
    """
    whether each phenotype scheme was fitted here or transferred

    Worth a table of its own rather than a sentence, because the two cases
    produce the same summary tables and differ only in what the labels mean. A
    scheme refit at a contributing site gives that site its own clusters, and
    then &ldquo;beta&rdquo; names this cohort's second cluster and nothing
    else -- which makes the pooled breakdown a comparison of four different
    partitions that happen to share four names. Nothing downstream can detect
    that, because a refit always produces labels.
    """
    if source.federated:
        # The centroid table is the evidence, and it is not in the contract --
        # correctly, since a centroid is a mean over a cluster whose size
        # nothing checks. So a coordinator cannot answer this question at all,
        # and must say that rather than read an absent file as proof of
        # transfer. Every bundle looks like a transfer from here.
        present = [s for s, summary, _ in FIT_EVIDENCE if source.has(summary)]
        if not present:
            return
        out.append("<h3>Fitted here, or transferred?</h3>")
        out.append(
            callout(
                "<b>Not answerable from a bundle.</b> 01 writes a centroid "
                "table only when it <b>fits</b> a scheme, so that file is the "
                "evidence &mdash; and it is not in the export contract, "
                "because a centroid is a mean over a cluster whose size "
                "nothing checks. "
                + ", ".join(f"<code>{esc(s)}</code>" for s in present)
                + " therefore cannot be distinguished here between labels "
                "refit on the sending cohort and labels from a model fitted "
                "elsewhere, and the two cases produce identical summary "
                "tables above. It matters because a refit makes the labels "
                "local: &ldquo;beta&rdquo; then names that cohort's second "
                "cluster rather than Seymour's, and the breakdown above "
                "silently compares partitions that share nothing but their "
                "names. <b>Each sending site's own report answers it</b>, from "
                "the centroid file in its <code>output/cohort/</code>. The "
                "durable fix is for the export stage to record which schemes "
                "it applied and which it fitted in <code>manifest.json</code>, "
                "which carries no cell counts and already travels.",
                "warn",
            )
        )
        return

    rows, refit = [], []
    for scheme, summary, centroids in FIT_EVIDENCE:
        if not source.has(summary):
            continue
        fitted = source.has(centroids)
        if fitted:
            refit.append(scheme)
        rows.append(
            {
                "scheme": scheme,
                "labels came from": "refit on this cohort"
                if fitted
                else "a model fitted elsewhere",
                "evidence": centroids + (" present" if fitted else " absent"),
            }
        )
    if not rows:
        return
    out.append("<h3>Fitted here, or transferred?</h3>")
    out.append(
        "<p>01 writes a centroid table only when it <b>fits</b> a scheme; "
        "applying a model fitted elsewhere writes labels and nothing else. "
        "That file is therefore the evidence, and it is reported because the "
        "two cases produce identical-looking summary tables above.</p>"
    )
    out.append(table(pl.DataFrame(rows)))
    if refit:
        out.append(
            callout(
                "<b>"
                + ", ".join(esc(s) for s in refit)
                + " "
                + ("was" if len(refit) == 1 else "were")
                + " refit on this cohort.</b> That is correct only at the "
                "derivation site. Everywhere else the labels become local: "
                "&ldquo;beta&rdquo; names this cohort's second cluster rather "
                "than Seymour's, and the pooled breakdown silently compares "
                "partitions that share nothing but their names. Check the run "
                "log for <code>--refit-phenotypes</code>; if it says "
                "<i>applying the phenotype models shipped in models/</i> and a "
                "scheme still shows as refit here, the flag did not reach that "
                "scheme.",
                "warn",
            )
        )


def section_phenotypes(source, out: list) -> None:
    """the published schemes, fitted here or transferred"""
    out.append('<h2 id="phenotypes">4 &middot; Phenotypes</h2>')
    out.append(
        "<p>Three schemes from the literature, each <b>refit on this "
        "cohort</b> rather than applied as published coefficients. Zweck is "
        "k-means on the eight labs the cardiogenic shock paper names, labelled "
        "from centroid profiles rather than cluster index &mdash; mortality "
        "rising I&nbsp;&lt;&nbsp;II&nbsp;&lt;&nbsp;III, as published, is the "
        "check that the refit recovered the intended phenotypes rather than "
        "three arbitrary clusters wearing their names. Seymour and Bhavani are "
        "sepsis schemes, fitted on the coded-septic subset only, which is why "
        "their totals are smaller.</p>"
    )
    out.append(
        "<p><b>Zweck is fitted once per cardiogenic denominator.</b> The "
        "phenotypes were derived in cardiogenic shock, so fitting them on the "
        "whole shock cohort clusters a population that is mostly "
        "non-cardiogenic. Each tab is an independent fit &mdash; its own "
        "scaling, its own k-means, its own centroid naming &mdash; on its own "
        "cohort, so which denominator a replication uses is a choice that "
        "changes the answer.</p>"
    )

    panels, per_denominator = [], False
    for column, label in ZWECK_VARIANTS:
        frame = source.read(f"phenotype_summary_{column}.csv")
        if frame is not None:
            per_denominator = True
            panels.append((label, table(frame)))
    pooled = source.read("phenotype_summary.csv")
    if pooled is not None and not per_denominator:
        panels.append(("Zweck", table(pooled)))
    for name, label in (
        ("seymour", "Seymour (sepsis)"),
        ("bhavani", "Bhavani (temperature)"),
    ):
        frame = source.read(f"{name}_summary.csv")
        if frame is not None:
            panels.append((label, table(frame)))
    if panels:
        out.append(tabs(panels))
    else:
        out.append(
            absent(
                "no phenotype summary table",
                "the <code>cohort</code> and <code>phenotypes</code> stages",
            )
        )

    if pooled is not None and not per_denominator:
        out.append(
            callout(
                "<b>Only the pooled <code>phenotype_summary.csv</code> is "
                "present, not the three per-denominator fits.</b> This cohort "
                "table predates the per-denominator fit, so the Zweck numbers "
                "above come from a single clustering of the whole shock cohort "
                "and carry Zweck's labels on clusters fitted to a mostly "
                "non-cardiogenic population. Re-run <code>--stage cohort</code> "
                "and then <code>--stage phenotypes</code> to get all three.",
                "warn",
            )
        )

    agreement = source.read("phenotype_variant_agreement.csv")
    if agreement is not None:
        out.append("<h3>Do the three Zweck fits agree?</h3>")
        out.append(
            "<p>Scored only on patients both fits label &mdash; a variant is "
            "null outside its own denominator, and scoring a null against a "
            "label would measure the denominators instead of the partitions. "
            "Adjusted Rand rather than raw agreement, because phenotype I is "
            "the large majority and two fits that both put most patients there "
            "would agree on most patients while sharing no structure: 0 is "
            "chance given the marginals, 1 is identical.</p>"
        )
        out.append(table(agreement))

    availability = source.read("seymour_availability.csv")
    if availability is not None:
        out.append("<h3>Variable availability behind the sepsis schemes</h3>")
        out.append(
            "<p>Seymour's clustering names variables this extract does not "
            "always carry. A scheme refit on whoever happens to have been "
            "measured is a scheme fitted on who got tested, so the "
            "availability is reported beside the fit rather than under it.</p>"
        )
        out.append(table(availability))

    transfer = source.read("seymour_agreement.csv")
    if transfer is not None:
        out.append("<h4>Refit here against transferred labels</h4>")
        out.append(
            "<p>The local refit cross-tabulated against the labels a model "
            "fitted elsewhere assigns, which is the check that "
            "&ldquo;alpha&rdquo; means the same thing in both.</p>"
        )
        out.append(table(transfer))

    section_fit_provenance(source, out)


def section_baseline(source, out: list) -> None:
    """median [IQR] and n (%) by coded mechanism"""
    out.append('<h2 id="baseline">5 &middot; Baseline, by coded mechanism</h2>')
    frame = source.read("baseline_by_shock_type_*.csv")
    if frame is None:
        out.append(
            absent(
                "<code>baseline_by_shock_type_&lt;SITE&gt;.csv</code>",
                "the <code>baseline</code> stage",
            )
        )
        return
    out.append(
        "<p>The table a reader wants before any latent-space claim, and the "
        "one that says whether &ldquo;cardiogenic&rdquo; and "
        "&ldquo;septic&rdquo; describe clinically different patients at this "
        "site at all. Median&nbsp;[IQR] for the continuous variables and "
        "n&nbsp;(%) for the binary ones, measured <b>at shock onset</b> rather "
        "than at admission or over the stay &mdash; the question is how each "
        "mechanism presented, and a median over a whole admission weights a "
        "patient by how long they stayed. Norepinephrine equivalents rather "
        "than a count of agents, because a dose of phenylephrine and a dose of "
        "norepinephrine are not the same quantity. <b>Per site and never "
        "pooled:</b> a pooled median is a weighted average over several "
        "documentation habits. Cells over fewer than the minimum cell size are "
        "blank, not zero.</p>"
    )
    display = frame.with_columns(
        value=pl.when(pl.col("kind") == "continuous")
        .then(
            pl.when(pl.col("median").is_null())
            .then(pl.lit(""))
            .otherwise(
                pl.col("median").round(1).cast(pl.String)
                + pl.lit(" [")
                + pl.col("q1").round(1).cast(pl.String)
                + pl.lit(" to ")
                + pl.col("q3").round(1).cast(pl.String)
                + pl.lit("]")
            )
        )
        .otherwise(
            # n_positive (share_positive%). `n_measured` used to hold the
            # positives and the denominator was not exported at all; it is now
            # the denominator, as its name says, so rendering it here would
            # print the wrong number in the numerator slot.
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
    out.append(
        tabs(
            [
                (
                    str(site),
                    table(
                        display.filter(pl.col("site") == site)
                        .select("variable", "shock_type", "value")
                        .pivot(on="shock_type", index="variable", values="value")
                    ),
                )
                for site in display["site"].unique(maintain_order=True).to_list()
            ]
        )
    )

    out.append("<h3>How much of each variable was measured</h3>")
    out.append(
        "<p>A median&nbsp;[IQR] cannot show its own denominator, and for "
        "physiology at onset the denominator is part of the finding: a "
        "variable measured in a minority of the cohort describes whoever was "
        "tested, which is a sicker group than the cohort and a differently "
        "selected one at each site.</p>"
    )
    coverage = (
        frame.filter(pl.col("kind") == "continuous")
        .group_by("site", "variable")
        .agg(n=pl.col("n").sum(), n_measured=pl.col("n_measured").sum())
        .with_columns(share_measured=pl.col("n_measured") / pl.col("n"))
        .sort("site", "share_measured")
    )
    out.append(
        tabs(
            [
                (
                    str(site),
                    table(coverage.filter(pl.col("site") == site).drop("site")),
                )
                for site in coverage["site"].unique(maintain_order=True).to_list()
            ]
        )
    )


def section_tokens(source, out: list) -> None:
    """token prevalence, the one cross-site comparison that needs no model"""
    out.append('<h2 id="tokens">6 &middot; Token prevalence</h2>')
    families = source.read("token_families_*.csv")
    prevalence = source.read("token_prevalence_*.csv")
    if families is None and prevalence is None:
        out.append(
            absent(
                "<code>token_prevalence_*.csv</code> and "
                "<code>token_families_*.csv</code>",
                "the <code>tokencounts</code> stage",
            )
        )
        return
    out.append(
        "<p>How often each vocabulary entry appears at each site, and in how "
        "many patients. <b>The one cross-site comparison that needs no "
        "model:</b> a token is a decile bin fitted at training time, so the "
        "bin edges are identical at every site by construction and the counts "
        "are directly comparable without anyone agreeing on anything. A site "
        "that shares nothing patient-level still contributes this, which is "
        "what lets the site effect be decomposed into &ldquo;because they "
        "chart differently&rdquo;.</p>"
    )
    out.append(
        "<p><code>n_patients</code> is a patient count and is suppressed below "
        "the minimum cell size. <code>total_occurrences</code> is suppressed "
        "with it: it is not a patient count, but a token occurring four times "
        "in one patient is as identifying as the count that would have been "
        "blanked, and publishing it beside a blank cell would hand it back.</p>"
    )

    def per_site_anchor(frame: pl.DataFrame, render) -> str:
        return tabs(
            [
                (
                    f"{site} / {anchor}",
                    render(
                        frame.filter(
                            (pl.col("site") == site) & (pl.col("anchor") == anchor)
                        )
                    ),
                )
                for site in frame["site"].unique(maintain_order=True).to_list()
                for anchor in frame["anchor"].unique(maintain_order=True).to_list()
            ]
        )

    if families is not None:
        frame = families
        out.append("<h3>By family</h3>")
        out.append(
            per_site_anchor(
                frame,
                lambda part: table(
                    part.select(
                        "family",
                        "n_patients",
                        "prevalence",
                        "total_occurrences",
                        "distinct_tokens",
                    ).sort("total_occurrences", descending=True, nulls_last=True)
                ),
            )
        )
        out.append(
            "<p><code>distinct tokens</code> counts vocabulary entries, not "
            "patients, and is never suppressed: it is a property of the "
            "tokenizer and of what this site charts, so blanking it would hide "
            "that a family is thinly used here &mdash; which is the finding "
            "rather than a disclosure. A patient is counted once per family "
            "however many of its tokens they emitted, so these rows cannot be "
            "summed out of the per-token table.</p>"
        )

    if prevalence is not None:
        frame = prevalence
        out.append("<h3>The forty most common tokens</h3>")
        out.append(
            "<p>Ranked by total occurrences within a site and anchor. The full "
            "table is one row per vocabulary entry and stays in "
            "<code>output/cohort/</code>; this is the head of it.</p>"
        )
        out.append(
            per_site_anchor(
                frame,
                lambda part: table(
                    part.select(
                        "token_name",
                        "family",
                        "n_patients",
                        "prevalence",
                        "total_occurrences",
                    ).sort("total_occurrences", descending=True, nulls_last=True),
                    limit=40,
                ),
            )
        )
        unknown = frame.filter(pl.col("token_name") == "<not in vocabulary>")
        if unknown.height:
            out.append(
                callout(
                    f"<b>{unknown.height} token ids in the stream are not in "
                    "the tokenizer vocabulary</b> and are reported as "
                    "<code>&lt;not in vocabulary&gt;</code>. The tokenizer on "
                    "this machine is then not the one the stream was encoded "
                    "with, and every token-level comparison here is against "
                    "the wrong bin edges.",
                    "warn",
                )
            )


#: the four per-site trial tables, with what each one is for
TRIAL_TABLES = (
    (
        "trial_outcomes",
        "Eligibility, concordance and outcomes",
        "One row per trial. <code>share_eligible</code> is of the stratum; "
        "<code>share_concordant</code> is how often the care actually given "
        "matched what the trial found, among the patients for whom that is "
        "knowable. <code>ratio_to_null</code> compares the share of patients "
        "inside every published IQR at once against what independence between "
        "the variables would give, so 1 means no more alike than chance.",
    ),
    (
        "trial_iqr_position",
        "Position against published IQRs",
        "Where this cohort sits against each trial's own Table&nbsp;1, "
        "variable by variable. A median inside the published IQR with most of "
        "the mass outside it is still a different population, which is why all "
        "three shares are reported and not just the median.",
    ),
    (
        "trial_coverage",
        "Coverage: how many trials speak to each stratum",
        "How many trials each stratum is eligible for. The point of the table "
        "is the zero &mdash; a stratum no trial enrols is an evidence orphan "
        "&mdash; and the phenotype schemes disagree about who that is.",
    ),
    (
        "trial_guideline_assignment",
        "Guideline fit",
        "Which guideline each eligibility pattern assigns a patient to, "
        "resolved by the specificity ordering committed in "
        "<code>trials.yaml</code> rather than by this cohort's own ranking "
        "&mdash; which is what makes one site's assignment comparable with "
        "another's. The run log prints what the ranking would have been here, "
        "and how far it has drifted from the committed one.",
    ),
)


def section_trials(source, out: list) -> None:
    """eligibility, position against published IQRs, guideline fit"""
    out.append('<h2 id="trials">7 &middot; Trials and guidelines</h2>')
    out.append(
        "<p>Scored site-side because the eligibility predicates and the "
        "baseline variables read raw CLIF, and emitted as shares and medians "
        "only &mdash; so a site that will never share a patient still "
        "contributes the whole trial analysis. The other half, fitting trial "
        "regions in embedding space, needs the pooled embeddings and stays at "
        "the coordinating site.</p>"
    )

    criteria = source.read("trial_criteria.csv")
    out.append("<h3>Published criteria, and which of them can be applied</h3>")
    if criteria is None:
        out.append(
            absent("<code>trial_criteria.csv</code>", "the <code>trials</code> stage")
        )
    else:
        out.append(
            "<p>Literature only, no patient data, identical at every site. "
            "<code>applied_here</code> is the honest part of this section: a "
            "criterion reading <code>partly</code> or <code>no</code> means "
            "the eligibility share below is for a relaxed version of the "
            "trial, and the note says which part was dropped.</p>"
        )
        out.append(
            tabs(
                [
                    (
                        str(trial),
                        table(
                            criteria.filter(pl.col("trial") == trial).drop("trial")
                        ),
                    )
                    for trial in criteria["trial"]
                    .unique(maintain_order=True)
                    .to_list()
                ]
            )
        )

    for stem, heading, blurb in TRIAL_TABLES:
        frame = source.read(f"{stem}_*.csv")
        out.append(f"<h3>{heading}</h3>")
        out.append(f"<p>{blurb}</p>")
        if frame is None:
            out.append(
                absent(
                    f"<code>{stem}_&lt;SITE&gt;_anchor-&lt;ANCHOR&gt;.csv</code>",
                    "the <code>trials</code> stage",
                )
            )
            continue
        out.append(
            tabs(
                [
                    (
                        f"{site} / {anchor}",
                        table(
                            frame.filter(
                                (pl.col("site") == site)
                                & (pl.col("anchor") == anchor)
                            ).drop("site", "anchor"),
                            limit=120,
                        ),
                    )
                    for site in frame["site"].unique(maintain_order=True).to_list()
                    for anchor in frame["anchor"]
                    .unique(maintain_order=True)
                    .to_list()
                ]
            )
        )
        short = [s for s in source.sites if s not in set(frame["site"].to_list())]
        if short:
            out.append(
                callout(
                    f"<b>No <code>{esc(stem)}</code> rows for "
                    + ", ".join(esc(s) for s in short)
                    + ".</b> The table is written per site, so this is a stage "
                    "that did not finish for those sites rather than a stratum "
                    "that came out empty &mdash; a distinction worth keeping, "
                    "because an absent site reads downstream as a site where "
                    "nobody was eligible.",
                    "warn",
                )
            )


def section_export(source, out: list) -> None:
    """what each site ends up sending"""
    out.append('<h2 id="export">8 &middot; The bundle each site sends</h2>')
    bundles = source.bundles
    if not bundles:
        out.append(
            absent(
                "no <code>upload_to_box_&lt;SITE&gt;/</code> directory",
                "the <code>export</code> stage",
            )
        )
        return
    out.append(
        "<p>Assembled fresh each run: the export stage removes the directory "
        "and rebuilds it, so what is listed here is this run's output and not "
        "an accumulation from earlier ones. Every table is re-checked against "
        "the minimum cell size on the way in, so a table assembled by some "
        "other route cannot reach the directory unsuppressed.</p>"
    )
    out.append(
        "<p>A <code>patient/</code> subdirectory means the bundle also carries "
        "patient-level embeddings &mdash; tier&nbsp;2, written by "
        "<code>02_site_embeddings.py</code> into the same directory so that a "
        "site uploads one thing either way. It is listed because whoever sends "
        "the directory should be able to see which tier they are sending; it "
        "is not 01's output.</p>"
    )

    def inventory(bundle: pathlib.Path) -> pl.DataFrame:
        files = sorted(f for f in bundle.rglob("*") if f.is_file())
        return pl.DataFrame(
            {
                "file": [
                    str(f.relative_to(bundle)).replace("\\", "/") for f in files
                ],
                "size_kb": [f.stat().st_size / 1024 for f in files],
            }
        )

    out.append(
        tabs([(site, table(inventory(bundle))) for site, bundle in bundles.items()])
    )
    tiered = [s for s, b in bundles.items() if (b / "patient").is_dir()]
    if tiered:
        plural = len(tiered) > 1
        out.append(
            callout(
                "<b>"
                + ", ".join(esc(s) for s in tiered)
                + f" carr{'y' if plural else 'ies'} a <code>patient/</code> "
                "directory</b>, so "
                + ("these bundles are" if plural else "this bundle is")
                + " a tier-2 contribution and hold"
                + ("" if plural else "s")
                + " one row per patient. Check that the data use agreement "
                "covers that before uploading."
            )
        )


def section_suppression(source, out: list) -> None:
    """the disclosure guarantee, counted rather than asserted"""
    out.append('<h2 id="suppression">9 &middot; Suppression audit</h2>')
    out.append(
        "<p>Nothing 01 sends is patient-level: every number is a count, a "
        "share or a median over at least the minimum cell size. This counts "
        "the blanks rather than taking that claim on trust. A "
        "<b>suppressible</b> cell is a count or a statistic derived from one; "
        "a <b>suppressed</b> one is a null in such a column, which is how 01 "
        "writes a cell it may not publish.</p>"
    )
    rows = []
    for name, frame in source.audit():
        nulls, total = blanked(frame)
        if not total:
            continue
        rows.append(
            {
                "file": name,
                "n_rows": frame.height,
                "n_suppressible": total,
                "n_suppressed": nulls,
                "share_suppressed": nulls / total,
            }
        )
    if not rows:
        out.append(absent(f"no aggregate CSV under <code>{esc(source.where)}</code>"))
        return
    audit = pl.DataFrame(rows).sort("n_suppressed", descending=True)
    out.append(table(audit))
    total_suppressed = int(audit["n_suppressed"].sum())
    out.append(
        f"<p><b>{total_suppressed:,} cells suppressed across {audit.height} "
        "tables.</b> A blank is an absence of permission to publish and not a "
        "zero, and must not be read as one: a blanked row in a prevalence "
        "table means the token occurred in some patients, fewer than the "
        "threshold.</p>"
    )
    if total_suppressed == 0:
        out.append(
            callout(
                "<b>Nothing was suppressed anywhere.</b> In a cohort this size "
                "that is likelier to mean <code>--min-cell</code> was set to 1 "
                "or 0 than that no table had a small cell. Check the "
                "<code>minimum cell size</code> field in the manifests above "
                "before uploading anything.",
                "warn",
            )
        )


SECTIONS = (
    ("stages", "Stage ledger"),
    ("provenance", "Provenance"),
    ("cohort", "Cohort"),
    ("phenotypes", "Phenotypes"),
    ("baseline", "Baseline, by coded mechanism"),
    ("tokens", "Token prevalence"),
    ("trials", "Trials and guidelines"),
    ("export", "The bundle each site sends"),
    ("suppression", "Suppression audit"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=None, help="which checkpoint's output directory to read"
    )
    parser.add_argument(
        "--role",
        default="pooled",
        choices=("site", "pooled"),
        help="defaults to pooled, because this report opens no patient record",
    )
    parser.add_argument(
        "--bundles",
        default=None,
        metavar="DIR",
        help=(
            "read received bundles from DIR instead of this machine's "
            "output/cohort/. DIR holds one upload_to_box_<SITE>/ per site. "
            "Defaults to output/received/ when there is no cohort directory "
            "here, which is the coordinating site's case."
        ),
    )
    parser.add_argument("--out", default=None, help="where to write the HTML")
    parser.add_argument(
        "--open", action="store_true", help="open the report once it is written"
    )
    args = parser.parse_args()

    paths = load_paths(model=args.model, role=args.role)
    paths.log_home.mkdir(parents=True, exist_ok=True)
    logger = get_logger("01_report", paths.log_home / "01_report.log")

    # Which deployment is reading. A contributing site has its own
    # output/cohort/ and reads that, because it holds everything including the
    # tables the contract does not carry. A coordinating site has only what
    # arrived, so it reads bundles -- and says so at the top of the report,
    # since a table missing from a bundle is not a stage that did not run.
    if args.bundles is not None:
        home = pathlib.Path(args.bundles)
        if not home.is_dir():
            logger.error(f"{home} does not exist")
            raise SystemExit(1)
        logger.info(f"reading received bundles from {home}")
        source = ReceivedBundles(home, logger)
    elif paths.cohort_home.is_dir():
        source = LocalRun(paths, logger)
        logger.info(f"reading this machine's own run from {paths.cohort_home}")
    elif paths.received_home.is_dir():
        logger.info(f"reading received bundles from {paths.received_home}")
        source = ReceivedBundles(paths.received_home, logger)
    else:
        logger.error(
            f"neither {paths.cohort_home} nor {paths.received_home} exists, so "
            "there is nothing to report. A contributing site starts with:"
        )
        logger.error("    python code/01_site_tables.py --check")
        logger.error(
            "A coordinating site puts each received upload_to_box_<SITE>/ "
            f"into {paths.received_home}, or passes --bundles DIR."
        )
        raise SystemExit(1)

    if not source.sites:
        logger.error(
            f"no bundle with a manifest.json under {source.where}. A bundle "
            "without one cannot be attributed to a site."
        )
        raise SystemExit(1)

    module = driver()
    out: list[str] = []
    if source.federated:
        out.append(
            callout(
                "<b>Read from the bundles the sites sent, not from a cohort "
                "directory.</b> A bundle carries what the export contract "
                "permits and no more, so the following are absent from every "
                "section below by design rather than because a stage did not "
                "run: "
                + "; ".join(
                    f"<code>{esc(name)}</code> &mdash; {reason}"
                    for name, reason in NOT_SHIPPED.items()
                )
                + ". The suppression audit below counts the blanks in the "
                "files as they arrived, which is the copy that matters."
            )
        )
    section_stages(source, module, out, logger)
    section_provenance(source, out)
    section_cohort(source, out)
    section_phenotypes(source, out)
    section_baseline(source, out)
    section_tokens(source, out)
    section_trials(source, out)
    section_export(source, out)
    section_suppression(source, out)

    toc = "".join(
        f'<li><a href="#{key}">{position} &middot; {esc(label)}</a></li>'
        for position, (key, label) in enumerate(SECTIONS, start=1)
    )
    document = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>01 run report</title>"
        f"<style>{STYLE}</style></head><body>"
        "<h1>01 run report</h1>"
        '<p class="meta">Everything <code>01_site_tables.py</code> produced, '
        "and nothing else. Built "
        f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M')} from "
        f"<code>{esc(source.where)}</code> &middot; model "
        f"<code>{esc(paths.model_key)}</code> &middot; sites "
        f"{esc(', '.join(source.sites)) or 'none configured'}.</p>"
        f'<ul class="toc">{toc}</ul>'
        f"{''.join(out)}"
        f"<script>{SCRIPT}</script>"
        "</body></html>\n"
    )

    destination = (
        pathlib.Path(args.out) if args.out else paths.output_home / "report_01.html"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    logger.info(f"wrote {destination} ({destination.stat().st_size / 1024:.0f} KB)")
    if args.open:
        webbrowser.open(destination.resolve().as_uri())
    logger.info("01_report done")


if __name__ == "__main__":
    main()

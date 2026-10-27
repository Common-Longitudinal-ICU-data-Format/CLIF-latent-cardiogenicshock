#!/usr/bin/env python3

"""
04 -- why do the contributing sites separate in embedding space?

Step 03 establishes *that* site dominates the geometry. This asks what the
model is reading. Four stages, cheapest and most direct first:

  tokens    what each site actually charts: family mix, UNK rate, vocabulary
            coverage, and the tokens that most distinguish the sites
  axis      fit a site probe on the embeddings, then regress that site score
            on per-patient token statistics -- turns "the sites differ" into
            "the site axis is mostly assessment-token density"
  regions   partition patients by how site-mixed their neighbourhood is, and
            compare case-mix and token composition across the partitions
  matched   re-measure site purity after matching the sites on case-mix; if it
            survives, the separation is documentation and not patients

Usage
    python code/04_site_effect.py                      # default model, both anchors
    python code/04_site_effect.py --model GEM-mimic-icu --anchor 24h
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from clifcs import ANCHORS, get_logger, load_paths
from clifcs import site_embedding_files
from clifcs.figures import save_figure
from clifcs.palette import (
    CATEGORICAL,
    CONTEXT,
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    site_colors,
)
from clifcs.tokens import (
    family_matrix,
    load_pre_anchor_tokens,
    token_counts_by_site,
    vocabulary,
)

DPI = 500
PCA_COMPONENTS = 50
K_NEIGHBOURS = 30


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE["light"],
            "axes.facecolor": SURFACE["light"],
            "savefig.facecolor": SURFACE["light"],
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "font.size": 8,
            "legend.frameon": False,
        }
    )


def embedding_matrix(frame: pl.DataFrame) -> np.ndarray:
    columns = sorted(c for c in frame.columns if c.startswith("emb_"))
    return frame.select(columns).to_numpy().astype(np.float64)


def load_embeddings(paths, anchor: str, logger, batch_size=None) -> pl.DataFrame:
    if paths.role == "pooled":
        from clifcs.export import bundles as bundle_io

        found = bundle_io.discover(paths.received_home, logger)
        manifests = bundle_io.check(found, paths, logger)
        return bundle_io.load_patient_level(
            found, manifests, anchor, logger, paths=paths,
            batch_size=batch_size,
        )

    sources = site_embedding_files(paths.patient_level_home, anchor)
    if not sources:
        raise FileNotFoundError(
            f"no embeddings for anchor {anchor} under {paths.patient_level_home}"
        )
    frame = pl.concat([pl.read_parquet(p) for p in sources], how="diagonal_relaxed")

    # Restrict to the current cohort. 02 embeds everyone present in the
    # winnowed data, which is a superset once an enrollment criterion is
    # tightened (cardiac origin, say). Without this join 04 and 05 would keep
    # describing the old cohort while 03 and 06 described the new one.
    cohort_file = paths.cohort_home / "cohort.parquet"
    if cohort_file.is_file():
        keep = pl.read_parquet(cohort_file).select("hospitalization_id", "site")
        before = frame.height
        frame = frame.join(keep, on=["hospitalization_id", "site"], how="inner")
        if frame.height != before:
            logger.info(
                f"[{anchor}] restricted to the cohort: {before} -> {frame.height}"
            )
    logger.info(f"[{anchor}] {frame.height} patients from {len(sources)} site files")
    return frame


# ============================================================ stage: tokens


def _pooled_token_counts(paths, anchor: str, logger) -> pl.DataFrame:
    """
    per-token occurrence counts, from the tier-1 bundles

    Returns `token`, `site`, `n` -- the same shape the site role builds from
    the streams directly, so `plot_token_tornado` needs no second code path.

    This is what changed in schema 3.0. The per-token comparison used to need
    the token sequences, which travelled under tier 2; the figure therefore
    existed only for sites that had sent them, and 04 skipped it rather than
    drawing a degraded version. The counts are now computed site-side by the
    `tokencounts` stage of 01, suppressed, and shipped in tier 1 -- so the
    comparison covers every contributing site, including one that shares
    nothing patient-level at all.

    A token suppressed at ANY site is dropped from the result entirely, at every
    site. This is the subtle part and it is worth stating, because getting it
    wrong produces a confident false finding rather than a gap.

    Two different things look alike in an exported table:

      * the token has no row at this site  -- it never appeared in the stream,
        a genuine structural zero, and `fill_null(0)` downstream is right;
      * the token has a row holding null   -- between one and `min_cell`
        patients emitted it and the count was suppressed.

    `plot_token_tornado` pivots and then calls `.fill_null(0)`, which cannot
    tell those apart, and its `exclusive` mode ranks precisely the tokens one
    site never emits. So a suppressed token would be promoted to "UCMC emits
    this and NU never does" -- a structural-zero claim manufactured out of a
    small cell. Dropping the token everywhere is the only honest option: we do
    not know its rate at that site, and a rate we do not know cannot be
    differenced.
    """
    from clifcs.export import bundles as bundle_io

    found = bundle_io.discover(paths.received_home, logger)
    table = bundle_io.load_counts(
        found, f"token_prevalence_anchor-{anchor}.csv", logger
    )
    if table is None or table.height == 0:
        logger.warning(
            f"[{anchor}] no token_prevalence table in any bundle, so the "
            "per-token tornado is skipped. It is written by the tokencounts "
            "stage of 01_site_tables.py; re-export with the current pipeline."
        )
        return pl.DataFrame()

    blanked = (
        table.filter(pl.col("total_occurrences").is_null())["token_name"]
        .unique()
        .to_list()
    )
    if blanked:
        logger.info(
            f"[{anchor}] {len(blanked)} token(s) were suppressed at at least "
            "one site and are dropped from the tornado at every site: a "
            "suppressed cell cannot be distinguished from a structural zero "
            "once the pivot fills nulls, and the exclusive ranking is built "
            "out of structural zeros"
        )
    return (
        table.filter(
            pl.col("total_occurrences").is_not_null()
            & ~pl.col("token_name").is_in(blanked)
        )
        .select(
            token=pl.col("token_name"),
            site=pl.col("site"),
            n=pl.col("total_occurrences"),
        )
    )


def stage_tokens_pooled(
    paths, frame: pl.DataFrame, anchor: str, sites, logger
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    the family mix and the per-token counts, from the tier-1 bundles

    The **family** comparison -- "NU charts 2.1x the assessment tokens UCMC
    does" -- comes from the `fam_n_*` columns on the patient rows, so it needs
    tier 2.

    The **per-token** comparison now comes from `counts/token_prevalence_*.csv`,
    which the `tokencounts` stage of 01 writes and every site ships in tier 1.
    It used to require the token sequences and so existed only for sites that
    had sent them; a site sharing nothing patient-level now contributes to it
    too. Since the site effect is the largest single finding here -- 97%
    neighbour purity -- being able to decompose it into "because they chart
    differently" should not have been contingent on a data use agreement.
    """
    _, families = vocabulary(str(paths.tokenizer_yaml))
    all_families = sorted(set(families.values()))
    present_cols = {
        name: f"fam_n_{name}"
        for name in all_families
        if f"fam_n_{name}" in frame.columns
    }
    if not present_cols:
        raise KeyError(
            "the pooled frame carries no fam_n_* columns, so the token-family "
            "comparison cannot run. Re-export the bundles with "
            "02_site_embeddings.py."
        )

    per_site = (
        frame.group_by("site")
        .agg([pl.col(c).fill_null(0).sum().alias(name) for name, c in present_cols.items()])
        .unpivot(index="site", variable_name="family", value_name="n")
    )
    totals = per_site.group_by("site").agg(total=pl.col("n").sum())
    logger.info(
        f"[{anchor}] token totals: {dict(zip(totals['site'], totals['total']))}"
    )

    family_mix = (
        per_site.join(totals, on="site")
        .with_columns(share=pl.col("n") / pl.col("total"))
        .pivot(on="site", index="family", values="share")
        .fill_null(0.0)
    )
    counts = _pooled_token_counts(paths, anchor, logger)

    shown = [s for s in sites if s in family_mix.columns]
    if len(shown) == 2:
        a, b = shown
        family_mix = family_mix.with_columns(
            difference=(pl.col(a) - pl.col(b))
        ).sort("difference", descending=True)

    logger.info(f"[{anchor}] token-family share by site (from exported counts):")
    for row in family_mix.iter_rows(named=True):
        bits = "  ".join(f"{s} {row[s]:6.2%}" for s in shown)
        diff = f"  diff {row['difference']:+.2%}" if "difference" in row else ""
        logger.info(f"    {row['family']:<9} {bits}{diff}")

    # The log-odds ranking of individual tokens still needs the streams: it is
    # a per-patient quantity (how many patients at each site carry the token,
    # against how many do not), and the prevalence table is marginal. The
    # tornado does not need it -- it ranks on a per-patient RATE, which is
    # exactly what the exported counts and the site sizes give.
    logger.info(
        f"[{anchor}] per-token tornado from the tier-1 counts; the log-odds "
        "ranking and the UNK burden still need the streams and are available "
        "site-side only"
    )
    return family_mix, pl.DataFrame(), counts


def stage_tokens(paths, anchor: str, sites, logger) -> tuple[pl.DataFrame, pl.DataFrame]:
    """what each site charts, and which tokens most distinguish them"""
    names, families = vocabulary(str(paths.tokenizer_yaml))
    tokens = pl.concat(
        [load_pre_anchor_tokens(paths, site, anchor) for site in sites]
    )

    counts = token_counts_by_site(tokens, names)
    totals = counts.group_by("site").agg(total=pl.col("n").sum())
    logger.info(f"[{anchor}] token totals: {dict(zip(totals['site'], totals['total']))}")

    # family mix as a share of each site's tokens
    family_mix = (
        counts.group_by("site", "family")
        .agg(n=pl.col("n").sum())
        .join(totals, on="site")
        .with_columns(share=pl.col("n") / pl.col("total"))
        .pivot(on="site", index="family", values="share")
        .fill_null(0.0)
    )
    present = [s for s in sites if s in family_mix.columns]
    if len(present) == 2:
        a, b = present
        family_mix = family_mix.with_columns(
            difference=(pl.col(a) - pl.col(b))
        ).sort("difference", descending=True)

    logger.info(f"[{anchor}] token-family share by site:")
    for row in family_mix.iter_rows(named=True):
        bits = "  ".join(f"{s} {row[s]:6.2%}" for s in present)
        diff = f"  diff {row['difference']:+.2%}" if "difference" in row else ""
        logger.info(f"    {row['family']:<9} {bits}{diff}")

    # per-site vocabulary coverage and UNK burden
    coverage = (
        counts.group_by("site")
        .agg(
            distinct_tokens=pl.col("token_id").n_unique(),
            total_tokens=pl.col("n").sum(),
            unk=pl.col("n").filter(pl.col("token") == "UNK").sum(),
        )
        .with_columns(unk_rate=pl.col("unk") / pl.col("total_tokens"))
        .sort("site")
    )
    logger.info(f"[{anchor}] vocabulary use:")
    for row in coverage.iter_rows(named=True):
        logger.info(
            f"    {row['site']:<6} {row['distinct_tokens']:>5} distinct of "
            f"{len(names)}  UNK {row['unk_rate']:.2%}"
        )

    # tokens most skewed between sites, by smoothed log-odds on within-site share
    discriminative = pl.DataFrame()
    if len(present) == 2:
        a, b = present
        wide = (
            counts.pivot(on="site", index=["token", "family"], values="n")
            .fill_null(0)
            .join(totals.pivot(on="site", index=None, values="total"), how="cross")
        )
        discriminative = (
            wide.with_columns(
                share_a=(pl.col(a) + 1) / (pl.col(f"{a}_right") + len(names)),
                share_b=(pl.col(b) + 1) / (pl.col(f"{b}_right") + len(names)),
            )
            .with_columns(log_odds=(pl.col("share_a") / pl.col("share_b")).log())
            .select("token", "family", a, b, "log_odds")
            .filter((pl.col(a) + pl.col(b)) >= 1000)
            .sort("log_odds", descending=True)
        )
        logger.info(f"[{anchor}] most {a}-skewed tokens:")
        for row in discriminative.head(8).iter_rows(named=True):
            logger.info(
                f"    {row['token']:<42} log-odds {row['log_odds']:+.2f}  "
                f"{a} {row[a]:>9,}  {b} {row[b]:>9,}"
            )
        logger.info(f"[{anchor}] most {b}-skewed tokens:")
        for row in discriminative.tail(8).reverse().iter_rows(named=True):
            logger.info(
                f"    {row['token']:<42} log-odds {row['log_odds']:+.2f}  "
                f"{a} {row[a]:>9,}  {b} {row[b]:>9,}"
            )

    # Structural zeros: tokens one site emits and the other never does. These
    # matter far more than any shaded difference in frequency, because a single
    # occurrence identifies the site with certainty -- which is what a linear
    # probe at AUC 1.0 is picking up. A post-hoc correction on the embeddings
    # cannot undo this; the inputs themselves are disjoint.
    if len(present) == 2:
        a, b = present
        wide = counts.pivot(on="site", index="token", values="n").fill_null(0)
        exclusive = {
            a: set(wide.filter((pl.col(a) > 0) & (pl.col(b) == 0))["token"].to_list()),
            b: set(wide.filter((pl.col(b) > 0) & (pl.col(a) == 0))["token"].to_list()),
        }
        logger.info(f"[{anchor}] structurally site-exclusive tokens:")
        for site in present:
            other = b if site == a else a
            ids = {i for i, n in names.items() if n in exclusive[site]}
            carriers, volume, total = 0, 0, 0
            part = tokens.filter(pl.col("site") == site)
            for seq in part["tokens_past"].to_list():
                arr = np.asarray(seq, dtype=np.int64)
                hits = int(np.isin(arr, list(ids)).sum()) if ids else 0
                carriers += hits > 0
                volume += hits
                total += arr.size
            logger.info(
                f"    {site:<6} emits {len(exclusive[site]):>4} tokens that "
                f"{other} never does; {carriers}/{part.height} "
                f"({carriers / max(part.height, 1):.1%}) of its patients carry at "
                f"least one, making up {volume / max(total, 1):.1%} of its tokens"
            )

    return family_mix, discriminative, counts


# ============================================================== stage: axis


def _token_statistics(
    paths, frame, tokens, families, family_order, anchor: str, logger
):
    """
    per-patient family counts, total tokens and stream duration

    Two sources for the same three quantities, because the site effect is the
    largest single finding in this pipeline and the decomposition that explains
    it has to survive the federated split.

    **Site role** reads the token streams directly.

    **Pooled role** reads the `fam_n_*` columns the bundles carry. A family
    count is far coarser than the stream it came from -- `fam_n_LAB-RES = 184`
    says a patient had 184 lab-result tokens, not which ones or when -- which is
    why it is on the allowlist and the sequence is not.

    Duration is the one input with no direct federated equivalent, since
    `s_elapsed_past` does not travel. It is reconstructed instead: at the 24h
    anchor the pre-anchor window is 24 hours by construction, and at the onset
    anchor it is `hours_admission_to_cs_onset`, which the contract already
    carries because the trial analysis needs it.
    """
    if paths.role != "pooled":
        counts = family_matrix(tokens, families, family_order)
        duration_h = np.asarray(
            [
                (e[-1] / 3600.0) if e is not None and len(e) else np.nan
                for e in tokens["s_elapsed_past"].to_list()
            ]
        )
        return counts, counts.sum(axis=1), duration_h

    wanted = [f"fam_n_{name}" for name in family_order]
    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        raise KeyError(
            f"the pooled frame is missing {len(missing)} token-family count "
            f"column(s), e.g. {missing[:3]}. They are produced by "
            "02_site_embeddings.py; re-export the bundles with the current site "
            "pipeline."
        )
    counts = frame.select(wanted).fill_null(0).to_numpy().astype(float)

    # every family, not just the per-patient-varying ones, so `n_tokens` is the
    # stream length rather than the length of the subset being regressed
    all_families = sorted(set(families.values()))
    present = [f"fam_n_{name}" for name in all_families if f"fam_n_{name}" in frame.columns]
    n_tokens = frame.select(present).fill_null(0).to_numpy().astype(float).sum(axis=1)

    if anchor == "24h":
        duration_h = np.full(frame.height, 24.0)
    elif "hours_admission_to_cs_onset" in frame.columns:
        duration_h = (
            frame["hours_admission_to_cs_onset"].cast(pl.Float64).to_numpy()
        )
    else:
        logger.warning(
            f"[{anchor}] no stream duration available in the bundles; the "
            "tokens/hour predictor is dropped from the regression"
        )
        duration_h = np.full(frame.height, np.nan)

    logger.info(
        f"[{anchor}] token statistics from exported family counts "
        f"({len(family_order)} families; duration reconstructed from "
        + ("the 24 h window" if anchor == "24h" else "hours_admission_to_cs_onset")
        + ")"
    )
    return counts, n_tokens, duration_h


def stage_axis(paths, frame: pl.DataFrame, anchor: str, sites, logger) -> pl.DataFrame:
    """
    find the site-discriminating axis by LDA, and say what drives it

    LDA gives the single direction maximising between-site over within-site
    scatter -- the natural definition of "the site axis". It is fitted on the
    50-d PCA representation with shrinkage, because the raw 1024-d classes are
    linearly separable and an unregularised within-class scatter matrix is
    singular there. Accuracy and AUC are cross-validated, so a separable class
    pair cannot report a free 100%.

    Regressing each patient's position on that axis against countable
    properties of their token stream then says what the model is keying on.
    """
    names, families = vocabulary(str(paths.tokenizer_yaml))
    if paths.role != "pooled":
        tokens = pl.concat([load_pre_anchor_tokens(paths, s, anchor) for s in sites])
        tokens = frame.select("hospitalization_id", "site").join(
            tokens.drop("site"), on="hospitalization_id", how="left"
        )
    else:
        tokens = None

    embeddings = StandardScaler().fit_transform(embedding_matrix(frame))
    is_first_site = (frame["site"] == sites[0]).to_numpy().astype(int)

    reduced = PCA(n_components=PCA_COMPONENTS, random_state=42).fit_transform(embeddings)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy = cross_val_score(lda, reduced, is_first_site, cv=folds, scoring="accuracy")
    auc_cv = cross_val_score(lda, reduced, is_first_site, cv=folds, scoring="roc_auc")
    baseline = max(is_first_site.mean(), 1 - is_first_site.mean())

    lda.fit(reduced, is_first_site)
    site_score = lda.decision_function(reduced)
    auc = roc_auc_score(is_first_site, site_score)

    logger.info(
        f"[{anchor}] LDA site axis: 5-fold accuracy "
        f"{accuracy.mean():.4f} +/- {accuracy.std():.4f} "
        f"(majority-class baseline {baseline:.4f}), AUC {auc_cv.mean():.4f}, "
        f"in-sample AUC {auc:.4f}"
    )
    probe_table = pl.DataFrame(
        {
            "anchor": [anchor],
            "cv_accuracy": [float(accuracy.mean())],
            "cv_accuracy_sd": [float(accuracy.std())],
            "cv_auc": [float(auc_cv.mean())],
            "in_sample_auc": [float(auc)],
            "majority_baseline": [float(baseline)],
            "n": [frame.height],
        }
    )
    probe_table.write_csv(paths.table_home / f"site_lda_anchor-{anchor}.csv")

    # AGE/SEX/RACE/ETHN/ADMN/BOS/EOS occur once per patient, so their share is
    # 1/n_tokens -- a restatement of sequence length that would otherwise
    # dominate the coefficients and hide the families that vary per patient
    once_per_patient = {"AGE", "SEX", "RACE", "ETHN", "ADMN", "BOS", "EOS", "DSCG"}
    family_order = sorted({f for f in families.values()} - once_per_patient)
    counts, n_tokens, duration_h = _token_statistics(
        paths, frame, tokens, families, family_order, anchor, logger
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        shares = counts / np.maximum(n_tokens[:, None], 1)
        rate = n_tokens / np.maximum(duration_h, 1e-6)

    design = np.column_stack(
        [shares, np.log1p(n_tokens), np.log1p(np.nan_to_num(rate, nan=0.0))]
    )
    labels = [f"share:{f}" for f in family_order] + ["log n_tokens", "log tokens/hour"]
    design = np.nan_to_num(design, nan=0.0, posinf=0.0, neginf=0.0)

    scaled = StandardScaler().fit_transform(design)
    target = (site_score - site_score.mean()) / site_score.std()
    coefficients, *_ = np.linalg.lstsq(
        np.column_stack([scaled, np.ones(len(scaled))]), target, rcond=None
    )
    predicted = np.column_stack([scaled, np.ones(len(scaled))]) @ coefficients
    r2 = 1 - ((target - predicted) ** 2).sum() / ((target - target.mean()) ** 2).sum()

    logger.info(
        f"[{anchor}] site axis regressed on token statistics: R^2 = {r2:.3f} "
        f"({len(labels)} predictors)"
    )
    table = (
        pl.DataFrame(
            {
                "anchor": anchor,
                "predictor": labels,
                "std_coefficient": coefficients[:-1],
                "site_probe_auc": auc,
                "r2": r2,
            }
        )
        .with_columns(abs_coefficient=pl.col("std_coefficient").abs())
        .sort("abs_coefficient", descending=True)
    )
    for row in table.head(10).iter_rows(named=True):
        logger.info(
            f"    {row['predictor']:<26} beta {row['std_coefficient']:+.3f}"
        )
    return table


# =========================================================== stage: regions


def stage_regions(
    frame: pl.DataFrame, anchor: str, sites, logger
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    split patients by how site-mixed their embedding neighbourhood is

    Patients sitting in a region where both sites are represented are the
    interesting ones: either they are clinically ordinary and the separation
    elsewhere is documentation, or they are a distinct subgroup. Comparing
    case-mix across the partitions distinguishes those.
    """
    coordinates = PCA(n_components=PCA_COMPONENTS, random_state=42).fit_transform(
        StandardScaler().fit_transform(embedding_matrix(frame))
    )
    neighbours = (
        NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1, n_jobs=-1)
        .fit(coordinates)
        .kneighbors(coordinates, return_distance=False)[:, 1:]
    )
    site_values = np.asarray(frame["site"].to_list())
    codes = (site_values == sites[0]).astype(int)
    neighbour_share = codes[neighbours].mean(axis=1)

    own = np.where(codes == 1, neighbour_share, 1 - neighbour_share)
    region = np.where(own >= 0.9, "site-exclusive", np.where(own >= 0.6, "site-leaning", "mixed"))

    frame = frame.with_columns(
        neighbour_same_site=pl.Series(own), region=pl.Series(region)
    )

    summary = (
        frame.group_by("region", "site")
        .agg(
            n=pl.len(),
            mortality=pl.col("in_hospital_mortality").mean(),
            median_icu_los=pl.col("icu_los_days").median(),
            # exact age does not leave a site, so a pooled run reports the modal
            # five-year band instead of a median. A band cannot be averaged.
            median_age=(
                pl.col("age_at_admission").median()
                if "age_at_admission" in frame.columns
                else pl.lit(None, dtype=pl.Float64)
            ),
            modal_age_band=(
                pl.col("age_band").mode().first()
                if "age_band" in frame.columns
                else pl.lit(None, dtype=pl.String)
            ),
            pct_cardiac=pl.col("cardiac_origin").mean(),
            pct_III=(pl.col("cs_phenotype") == "III cardiometabolic").mean(),
        )
        .sort("region", "site")
    )
    logger.info(f"[{anchor}] neighbourhood regions (k={K_NEIGHBOURS}):")
    for row in summary.iter_rows(named=True):
        logger.info(
            f"    {row['region']:<15} {row['site']:<5} n={row['n']:>6}  "
            f"mortality {row['mortality']:>5.1%}  ICU {row['median_icu_los']:>4.1f}d  "
            f"age {row['median_age'] if row['median_age'] is not None else row['modal_age_band']}  "
            f"cardiac {row['pct_cardiac']:>5.1%}  "
            f"phenoIII {row['pct_III']:>5.1%}"
        )
    return frame, summary


# =========================================================== stage: matched


def stage_matched(frame: pl.DataFrame, anchor: str, sites, logger) -> pl.DataFrame:
    """
    site purity after matching the sites on case-mix

    If UCMC and NU patients who look alike clinically still sit apart in
    embedding space, the separation cannot be explained by who the patients
    are. Exact matching on coarse strata, 1:1 without replacement, seeded.
    """
    strata = frame.with_columns(
        stratum=pl.concat_str(
            [
                # `age_band` under the pooled role: the export replaces exact
                # age with five-year bands, which is a finer stratum than the
                # decade this used, so matching is if anything tighter
                (
                    (pl.col("age_at_admission") // 10).cast(pl.Int64).cast(pl.String)
                    if "age_at_admission" in frame.columns
                    else pl.col("age_band").cast(pl.String)
                ),
                pl.col("sex_category").fill_null("?"),
                pl.col("cs_phenotype").cast(pl.String),
                pl.col("in_hospital_mortality").cast(pl.String),
            ],
            separator="|",
        )
    ).with_row_index("row")

    rng = np.random.default_rng(42)
    keep: list[int] = []
    for _, part in strata.group_by("stratum"):
        per_site = {
            s: part.filter(pl.col("site") == s)["row"].to_list() for s in sites
        }
        if any(len(v) == 0 for v in per_site.values()):
            continue
        take = min(len(v) for v in per_site.values())
        for rows in per_site.values():
            keep.extend(rng.permutation(rows)[:take].tolist())

    matched = strata.filter(pl.col("row").is_in(keep)).drop("row", "stratum")
    logger.info(
        f"[{anchor}] case-mix matched subset: {matched.height} patients "
        f"({dict(zip(*np.unique(matched['site'].to_list(), return_counts=True)))})"
    )
    if matched.height < 200:
        logger.warning(f"[{anchor}] matched subset too small to interpret")
        return pl.DataFrame()

    coordinates = PCA(
        n_components=min(PCA_COMPONENTS, matched.height - 1), random_state=42
    ).fit_transform(StandardScaler().fit_transform(embedding_matrix(matched)))
    k = min(K_NEIGHBOURS, matched.height - 1)
    neighbours = (
        NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        .fit(coordinates)
        .kneighbors(coordinates, return_distance=False)[:, 1:]
    )
    values = np.asarray(matched["site"].to_list())
    _, codes = np.unique(values, return_inverse=True)
    purity = float(np.mean(codes[neighbours] == codes[:, None]))
    proportions = np.bincount(codes) / len(codes)
    chance = float((proportions**2).sum())

    logger.info(
        f"[{anchor}] site purity in the matched subset: {purity:.1%} vs "
        f"{chance:.1%} by chance ({purity / chance:.2f}x)"
    )
    return pl.DataFrame(
        {
            "anchor": [anchor],
            "n_matched": [matched.height],
            "purity": [purity],
            "chance": [chance],
            "lift": [purity / chance],
        }
    )


# =============================================================== plotting


def _pooled_token_prevalence(paths, anchor: str, logger) -> pl.DataFrame:
    """
    the tier-1 token_prevalence tables, stacked, with the patient counts kept

    `_pooled_token_counts` reduces to (token, site, n) because that is all the
    tornado needs. The volcano needs the patient count and the cohort size to
    build a 2x2, so it reads the same tables and keeps more of them.
    """
    from clifcs.export import bundles as bundle_io

    if paths.role == "pooled":
        found = bundle_io.discover(paths.received_home, logger)
        table = bundle_io.load_counts(
            found, f"token_prevalence_anchor-{anchor}.csv", logger
        )
        if table is None:
            return pl.DataFrame()
        return table

    frames = []
    for site in paths.sites:
        path = paths.upload_home(site) / "counts" / f"token_prevalence_anchor-{anchor}.csv"
        if not path.is_file():  # written before 01 put its tables in the upload folder
            path = paths.cohort_home / f"token_prevalence_{site}_anchor-{anchor}.csv"
        if path.is_file():
            frames.append(pl.read_csv(path))
    if not frames:
        logger.info(
            f"[{anchor}] no token_prevalence tables; run 01 --stage tokencounts"
        )
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def token_enrichment(counts_by_site: pl.DataFrame, logger) -> pl.DataFrame:
    """
    per-token enrichment at each site against the pooled others

    The tornado answers "which tokens differ most between these two sites" and
    ranks on the raw difference, so it is dominated by the commonest tokens and
    cannot say whether a difference exceeds sampling noise. This answers a
    different question -- which tokens is THIS site unusual for -- with an
    effect size and a p-value, which is what a volcano plot needs.

    Effect is the log2 ratio of PREVALENCE: the share of patients emitting the
    token at least once, here against everywhere else. Prevalence rather than
    rate, because a proportion has a sampling distribution a two-proportion
    test can use, and because "how many patients does this happen to" is the
    question; a per-patient rate conflates that with how often it repeats in
    the patients it happens to.

    Significance is a two-sided chi-square on the 2x2 of patients with and
    without the token, site against the rest, Bonferroni-corrected across every
    test. With tens of thousands of patients almost everything clears an
    uncorrected threshold, so the correction does real work and the effect size
    is what should be read first.

    Suppressed cells are dropped rather than treated as zero: a blank means
    between one and min_cell patients, which is not none.
    """
    from scipy.stats import chi2_contingency

    frame = counts_by_site.filter(pl.col("n_patients").is_not_null())
    if frame.is_empty():
        return pl.DataFrame()

    cohort = dict(
        frame.group_by("site").agg(n=pl.col("n_cohort").max()).iter_rows()
    )
    total = sum(cohort.values())

    rows = []
    for (token,), group in frame.group_by(["token_name"], maintain_order=True):
        per_site = dict(zip(group["site"].to_list(), group["n_patients"].to_list()))
        everywhere = sum(per_site.values())
        for site, n_site in per_site.items():
            n_rest = everywhere - n_site
            cohort_site = cohort.get(site, 0)
            cohort_rest = total - cohort_site
            if cohort_site == 0 or cohort_rest == 0 or n_site <= 0:
                continue
            # a token nobody else emits has an infinite ratio. Half a patient in
            # the denominator makes it finite so the point lands on the figure,
            # and the marker is open so the value reads as a floor.
            exclusive = n_rest == 0
            p_site = n_site / cohort_site
            p_rest = max(n_rest, 0.5) / cohort_rest
            table = [
                [int(n_site), int(cohort_site - n_site)],
                [int(n_rest), int(cohort_rest - n_rest)],
            ]
            if min(min(r) for r in table) < 0:
                continue
            try:
                _, pvalue, _, _ = chi2_contingency(table)
            except ValueError:
                continue
            rows.append({
                "site": site,
                "token_name": token,
                "family": group["family"][0],
                "n_patients": int(n_site),
                "prevalence_site": p_site,
                "prevalence_rest": n_rest / cohort_rest,
                "log2_ratio": float(np.log2(p_site / p_rest)),
                "p_value": float(pvalue),
                "site_exclusive": bool(exclusive),
            })

    if not rows:
        return pl.DataFrame()
    out = pl.DataFrame(rows)
    tested = out.height
    out = out.with_columns(
        p_adjusted=pl.min_horizontal(pl.col("p_value") * tested, pl.lit(1.0))
    )
    logger.info(
        f"  token enrichment: {tested} site-by-token tests, "
        f"{int((out['p_adjusted'] < 0.05).sum())} significant after Bonferroni"
    )
    return out


def plot_token_volcano(enrichment: pl.DataFrame, site: str, out_path, anchor,
                       logger, top_n: int = 12) -> None:
    """
    volcano of token enrichment for one site against all the others

    Effect on x, significance on y, one point per token. Enriched right,
    depleted left. The labelled points are the largest EFFECTS among the
    significant, not the most significant: with this many patients the smallest
    p-values belong to commonplace tokens with trivial differences, and
    labelling those says nothing.

    Colours are the two all-pairs-validated CATEGORICAL slots rather than the
    red-green a volcano usually wears, which fails the colourblind gate this
    project applies to every other figure.

    Site-exclusive tokens are open markers: their true ratio is infinite and the
    plotted value is a floor.
    """
    part = enrichment.filter(pl.col("site") == site)
    if part.is_empty():
        return

    x = part["log2_ratio"].to_numpy()
    y = -np.log10(np.maximum(part["p_adjusted"].to_numpy(), 1e-300))
    significant = part["p_adjusted"].to_numpy() < 0.05
    exclusive = part["site_exclusive"].to_numpy()
    enriched = significant & (x > 1) & ~exclusive
    depleted = significant & (x < -1) & ~exclusive

    figure, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.scatter(x[~significant], y[~significant], s=7, c=CONTEXT, linewidths=0,
               rasterized=True, label="not significant")
    ax.scatter(x[depleted], y[depleted], s=16, c=CATEGORICAL[0], linewidths=0,
               rasterized=True, label=f"depleted at {site}")
    ax.scatter(x[enriched], y[enriched], s=16, c=CATEGORICAL[1], linewidths=0,
               rasterized=True, label=f"enriched at {site}")
    if exclusive.any():
        ax.scatter(x[exclusive], y[exclusive], s=26, facecolors="none",
                   edgecolors=CATEGORICAL[1], linewidths=0.9,
                   label=f"only {site} records it")

    ax.axvline(0, color=GRID, linewidth=0.6)
    for cut in (-1.0, 1.0):
        ax.axvline(cut, color=GRID, linewidth=0.5, linestyle=":")
    ax.axhline(-np.log10(0.05), color=GRID, linewidth=0.5, linestyle=":")

    labelled = (
        part.with_columns(magnitude=pl.col("log2_ratio").abs())
        .filter(pl.col("p_adjusted") < 0.05)
        .sort("magnitude", descending=True)
        .head(top_n)
    )
    # Collision-aware labelling, in axes fraction so the threshold means the
    # same thing on both axes. Everything that hits the 1e-300 p-value floor
    # lands on one horizontal line, so the biggest effects are exactly the
    # points most likely to overlap each other.
    span_x = max(np.ptp(x), 1e-9)
    span_y = max(np.ptp(y), 1e-9)
    placed: list[tuple[float, float]] = []
    for row in labelled.iter_rows(named=True):
        lx = row["log2_ratio"]
        ly = -np.log10(max(row["p_adjusted"], 1e-300))
        fx, fy = lx / span_x, ly / span_y
        if any(np.hypot(fx - qx, fy - qy) < 0.05 for qx, qy in placed):
            continue
        ax.annotate(
            row["token_name"].split("//")[-1], (lx, ly),
            fontsize=6, color=INK_SECONDARY, xytext=(4, 3),
            textcoords="offset points",
        )
        placed.append((fx, fy))

    ax.set_xlabel(f"log2 prevalence ratio, {site} against every other site")
    ax.set_ylabel("-log10 adjusted p  (chi-square, Bonferroni)")
    ax.set_title(
        f"Tokens enriched and depleted at {site}, {anchor} anchor",
        loc="left", pad=8,
    )
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, markerscale=1.3)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    save_figure(figure, out_path)
    plt.close(figure)
    logger.info(
        f"wrote {out_path.name}: {int(enriched.sum())} enriched, "
        f"{int(depleted.sum())} depleted, {int(exclusive.sum())} exclusive"
    )


def plot_token_tornado(
    counts: pl.DataFrame,
    site_sizes: dict[str, int],
    sites,
    out_path,
    anchor,
    logger,
    top_n: int = 18,
    rank: str = "difference",
) -> None:
    """
    diverging bar chart of the tokens that most distinguish the sites

    Rates per patient, not raw counts: the sites contribute very different
    numbers of patients (9,135 vs 25,007) and raw totals would make the larger
    site look systematically more verbose. Tokens are ranked by the absolute
    difference in per-patient rate, which surfaces both the structural zeros
    and the merely lopsided tokens in one view.
    """
    left, right = sites[0], sites[1]
    wide = (
        counts.pivot(on="site", index="token", values="n")
        .fill_null(0)
        .with_columns(
            left_rate=pl.col(left) / site_sizes[left],
            right_rate=pl.col(right) / site_sizes[right],
        )
        .with_columns(difference=(pl.col("left_rate") - pl.col("right_rate")))
        .filter(pl.col("token") != "UNK")
        .with_columns(magnitude=pl.col("difference").abs())
    )
    if rank == "exclusive":
        # tokens one site never emits at all; ranked by how often the other
        # site uses them, since those are the ones a probe can key on
        wide = wide.filter((pl.col(left) == 0) | (pl.col(right) == 0))
    wide = wide.sort("magnitude", descending=True).head(top_n).sort("difference")

    labels = wide["token"].to_list()
    left_rate = np.asarray(wide["left_rate"].to_list())
    right_rate = np.asarray(wide["right_rate"].to_list())
    y = np.arange(len(labels))
    colors = site_colors(sites)

    figure, ax = plt.subplots(figsize=(7.4, 0.32 * len(labels) + 1.6))
    ax.barh(y, -left_rate, color=colors[left], linewidth=0, height=0.68)
    ax.barh(y, right_rate, color=colors[right], linewidth=0, height=0.68)

    span = max(left_rate.max(), right_rate.max())
    for i, (lv, rv) in enumerate(zip(left_rate, right_rate)):
        if lv > span * 0.06:
            ax.text(-lv / 2, i, f"{lv:.1f}", ha="center", va="center",
                    color="white", fontsize=6.5)
        if rv > span * 0.06:
            ax.text(rv / 2, i, f"{rv:.1f}", ha="center", va="center",
                    color="white", fontsize=6.5)

    ax.set_yticks(y, labels, fontsize=6.5)
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_xlim(-span * 1.12, span * 1.12)
    ax.set_xlabel("mean occurrences per patient before the anchor")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}"))

    ax.text(-span * 0.55, len(labels) - 0.1, left, ha="center", va="bottom",
            color=colors[left], fontsize=9, fontweight="bold")
    ax.text(span * 0.55, len(labels) - 0.1, right, ha="center", va="bottom",
            color=colors[right], fontsize=9, fontweight="bold")
    headline = (
        "Tokens one site never records"
        if rank == "exclusive"
        else "Tokens that most distinguish the sites"
    )
    ax.set_title(f"{headline}, {anchor} anchor", loc="left", pad=22)
    ax.grid(True, axis="x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_ylim(-0.8, len(labels) + 0.2)

    save_figure(figure, out_path)
    plt.close(figure)
    logger.info(f"wrote {out_path}")


def plot_family_mix(family_mix: pl.DataFrame, sites, out_path, anchor, logger) -> None:
    """per-site token-family share, the most direct picture of what differs"""
    present = [s for s in sites if s in family_mix.columns]
    if len(present) < 2:
        return
    data = family_mix.sort(present[0], descending=True).head(14)
    labels = data["family"].to_list()
    y = np.arange(len(labels))
    colors = site_colors(present)

    figure, ax = plt.subplots(figsize=(6.2, 4.4))
    height = 0.38
    for offset, site in zip((-height / 2, height / 2), present):
        ax.barh(
            y + offset,
            data[site].to_list(),
            height=height,
            color=colors[site],
            label=site,
            linewidth=0,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("share of the tokens the model read before the anchor")
    ax.set_title(
        f"Token-family mix by site, {anchor} anchor", loc="left", pad=8
    )
    ax.grid(True, axis="x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", labelcolor=INK_SECONDARY, fontsize=7)
    save_figure(figure, out_path)
    plt.close(figure)
    logger.info(f"wrote {out_path}")


# ================================================================= driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", choices=ANCHORS, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # no longer used: schema 3.0 ships embeddings,
        # so there is nothing to re-embed. Kept so an existing invocation or
        # sbatch script that still passes it does not fail at parse time.
    )
    parser.add_argument(
        "--role",
        choices=("site", "pooled"),
        default=None,
        help="`pooled` reads received bundles from output/export/ and refuses "
        "to touch raw CLIF data; `site` reads this machine's own files. "
        "Defaults to `role` in paths.yaml, then to `site`.",
    )
    args = parser.parse_args()

    paths = load_paths(model=args.model, role=args.role)
    paths.mkdirs()
    logger = get_logger("04_site_effect", paths.log_home / "04_site_effect.log")
    style()

    anchors = (args.anchor,) if args.anchor else ANCHORS
    axis_tables, region_tables, matched_tables = [], [], []

    for anchor in anchors:
        frame = load_embeddings(paths, anchor, logger, getattr(args, 'batch_size', None))
        # paths.yaml order, not alphabetical: it is the order the plots read
        # left-to-right, and the reader expects the index site first
        present = set(frame["site"].unique().to_list())
        sites = [s for s in paths.sites if s in present] + sorted(
            present - set(paths.sites)
        )
        if len(sites) < 2:
            logger.warning(f"[{anchor}] only one site present; nothing to compare")
            continue
        logger.info(f"[{anchor}] model {paths.model_key}, sites {sites}")

        if paths.role == "pooled":
            family_mix, discriminative, counts = stage_tokens_pooled(
                paths, frame, anchor, sites, logger
            )
        else:
            family_mix, discriminative, counts = stage_tokens(
                paths, anchor, sites, logger
            )
        family_mix.write_csv(paths.table_home / f"site_token_families_anchor-{anchor}.csv")
        if discriminative.height:
            discriminative.write_csv(
                paths.table_home / f"site_discriminative_tokens_anchor-{anchor}.csv"
            )
        figures = paths.figure_home / f"anchor-{anchor}"
        figures.mkdir(parents=True, exist_ok=True)
        plot_family_mix(
            family_mix, sites, figures / "site_token_families.png", anchor, logger
        )
        site_sizes = dict(
            frame.group_by("site").agg(pl.len().alias("n")).iter_rows()
        )
        # Drawn under both roles now. The pooled role gets the per-token counts
        # from the tier-1 bundles rather than from the streams, which is why
        # this is no longer conditional on the site role -- the guard is only
        # for a bundle set that predates the tokencounts stage.
        if counts.height:
            # ONE TORNADO PER PAIR OF SITES. plot_token_tornado is inherently
            # pairwise -- it ranks on the difference between two per-patient
            # rates -- and it took sites[0] and sites[1]. With a third site
            # configured that silently compared the first two and dropped the
            # third, while labelling itself as the site comparison. Rush has by
            # far the most distinctive token profile of the three (14.8% UNK
            # against 7.8% and 8.8%), so it was the one panel worth seeing and
            # the one being dropped.
            #
            # The first pair keeps the historical filenames so an existing
            # report or link does not break; every pair also gets a named file.
            pairs = list(itertools.combinations(sites, 2))
            for position, pair in enumerate(pairs):
                tag = "-vs-".join(pair)
                for mode, stem in (
                    ("difference", "site_token_tornado"),
                    ("exclusive", "site_token_tornado_exclusive"),
                ):
                    names = [f"{stem}_{tag}.png"]
                    if position == 0:
                        names.append(f"{stem}.png")
                    for name in names:
                        plot_token_tornado(
                            counts, site_sizes, list(pair), figures / name,
                            anchor, logger, rank=mode,
                        )
            if len(pairs) > 1:
                logger.info(
                    f"[{anchor}] drew {len(pairs)} pairwise tornados: "
                    + ", ".join("-vs-".join(p) for p in pairs)
                )

            # And a volcano per site. The tornado is pairwise and ranks on a raw
            # difference, so it is dominated by common tokens and says nothing
            # about noise; this asks which tokens each site is unusual for,
            # against everywhere else, with an effect size and a p-value.
            prevalence = _pooled_token_prevalence(paths, anchor, logger)
            if prevalence.height:
                enrichment = token_enrichment(prevalence, logger)
                if enrichment.height:
                    enrichment.write_csv(
                        paths.table_home / f"site_token_enrichment_anchor-{anchor}.csv"
                    )
                    for site in sites:
                        plot_token_volcano(
                            enrichment, site,
                            figures / f"site_token_volcano_{site}.png",
                            anchor, logger,
                        )

        axis_tables.append(stage_axis(paths, frame, anchor, sites, logger))
        frame, regions = stage_regions(frame, anchor, sites, logger)
        region_tables.append(regions.with_columns(anchor=pl.lit(anchor)))
        matched = stage_matched(frame, anchor, sites, logger)
        if matched.height:
            matched_tables.append(matched)

    for table, name in (
        (axis_tables, "site_axis_attribution"),
        (region_tables, "site_regions"),
        (matched_tables, "site_purity_matched"),
    ):
        if table:
            pl.concat(table).write_csv(paths.table_home / f"{name}.csv")
            logger.info(f"wrote {paths.table_home / f'{name}.csv'}")

    logger.info("04 done")


if __name__ == "__main__":
    main()

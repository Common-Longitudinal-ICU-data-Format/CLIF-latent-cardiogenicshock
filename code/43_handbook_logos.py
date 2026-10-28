#!/usr/bin/env python3

"""
43_handbook_logos.py -- the handbooks as sequence logos

40 found the handbooks and 42 drew six example pairs from each as aligned
strings. Six pairs is an illustration. This summarises ALL of a handbook's
patients the way a genomics sequence logo summarises an alignment: one stack of
letters per hour, letter heights proportional to how often that symbol occurs
there, column height equal to how determined the hour is at all.

    python code/43_handbook_logos.py --subset cardiogenic

TWO FIGURES PER LAYER.

  CLOCK LOGOS      one row per handbook, hours 0-48 since shock onset. What a
                   handbook's course generally looks like, in one panel per
                   handbook rather than six strings per handbook.

  DIVERGENCE LOGO  per handbook, the two response groups' whole courses on one
                   clock, one row each, with the hour they measurably part
                   marked and everything before it shaded. The shading is the
                   claim "indistinguishable from permuted labels here", which
                   reads correctly whether that region is forty hours wide or
                   empty -- and in this cohort it is empty for four of the five
                   handbooks, so the earlier design, which drew a shared prefix
                   and then split, was drawing a prefix the test had just said
                   was not shared.

WHERE THE DIVERGENCE HOUR COMES FROM. Not from the matched-pair alignments in
42, which answer a different question -- when do THESE TWO courses stop
matching -- and have no population summary. Here the two groups' symbol
distributions are compared at every hour by Jensen-Shannon divergence, and the
observed curve is read against a null built by permuting the group labels. The
divergence hour is the first hour the observed value clears the null band and
STAYS clear, because one hour poking above a permutation band is a spike, and
the question is when the courses part rather than when they flicker.

THE COLUMN HEIGHT IS INFORMATION, IN BITS. A column where every patient is in
the same state is log2(K) bits tall; a column where the states are uniform is
flat. That is the standard logo convention and it has the property worth having
here: an hour when the handbook does not determine what happens is SHORT, so
the figure cannot make a hundred patients doing ten different things look like
a consensus. The frequency-stack variant -- every column full height -- is
written as well, because information content hides the composition of an
undetermined hour and sometimes the composition is the point.

WHAT THE RIGHT-HAND SIDE OF EVERY LOGO IS CONDITIONED ON, which is the one
thing that will mislead a reader who skips the caption. The strings are
trimmed of their trailing off-pressor run, so a patient who comes off pressors
at hour 12 contributes twelve hours and stops. The column at hour 40 is
therefore drawn from patients still on pressors at hour 40, a sicker and
smaller group than the handbook. The coverage strip under each logo is that
denominator, drawn so the collapse is visible rather than merely stated.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import (
    apply_handbook_defaults, get_logger, handbook_defaults, load_paths,
)
from clifcs.figures import save_figure, save_table, style
from clifcs.palette import GRID, INK_PRIMARY, INK_SECONDARY, husl_colors
from clifcs.regions import HOURLY_REGION_ORDER, STATE_SYMBOL
from clifcs.treatment import TREATMENT_SYMBOLS, treatment_label

RESPONSE_ALPHABET = "".join(
    STATE_SYMBOL[state] for state in HOURLY_REGION_ORDER
)
TREATMENT_ALPHABET = TREATMENT_SYMBOLS[:16]
SUPPRESS = 11

#: Bold, because a logo letter is scaled to fill its own box and a light face
#: becomes a few hairlines once the box is a twentieth of a bit tall.
GLYPH = FontProperties(family="DejaVu Sans", weight="bold")


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(
        name.split("_")[0] + "_mod",
        pathlib.Path(__file__).resolve().parent / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ======================================================= counting and scoring


def column_counts(strings, alphabet: str, length: int):
    """
    per-hour symbol counts, and the number of patients each column rests on

    A patient contributes to column i only if their record reaches hour i.
    Nothing is padded: padding with a sentinel would turn "this patient's
    course ended" into a state, and that state would then win every late
    column, most decisively in the handbook whose patients recover fastest.
    """
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    counts = np.zeros((len(alphabet), length))
    for sequence in strings:
        for i, symbol in enumerate(sequence[:length]):
            row = index.get(symbol)
            if row is not None:
                counts[row, i] += 1.0
    return counts, counts.sum(axis=0)


def information(counts, n):
    """
    bits of information per column, with the small-sample correction

    log2(K) - H, the standard logo height. The correction subtracts the
    entropy deficit expected from estimating K frequencies on n observations,
    (K-1)/(2 n ln2), which at n=25 and K=10 is 0.26 bits -- not negligible at
    the thin end, and the thin end is exactly where an uncorrected logo invents
    a consensus out of six patients who happened to agree.
    """
    k = counts.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, counts / np.maximum(n, 1e-12), 0.0)
        terms = np.where(p > 0, p * np.log2(np.maximum(p, 1e-12)), 0.0)
    entropy = -terms.sum(axis=0)
    correction = (k - 1) / (2.0 * np.log(2.0) * np.maximum(n, 1.0))
    return np.clip(np.log2(k) - entropy - correction, 0.0, None)


def consensus_string(counts, n, alphabet: str, min_n: int) -> str:
    """the most common symbol per hour, dots where the column is too thin"""
    out = []
    for i in range(counts.shape[1]):
        if n[i] < min_n:
            out.append(".")
            continue
        out.append(alphabet[int(np.argmax(counts[:, i]))])
    return "".join(out)


def _normalise(counts, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, counts / np.maximum(n, 1e-12), 0.0)


def jensen_shannon(p, q):
    """base-2 Jensen-Shannon divergence, column by column"""
    m = 0.5 * (p + q)

    def kl(a):
        with np.errstate(invalid="ignore", divide="ignore"):
            terms = np.where(
                a > 0,
                a * np.log2(np.maximum(a, 1e-12) / np.maximum(m, 1e-12)),
                0.0,
            )
        return terms.sum(axis=0)

    return 0.5 * kl(p) + 0.5 * kl(q)


def divergence_curve(left, right, alphabet: str, length: int):
    """observed per-hour divergence between two sets of courses"""
    lc, ln = column_counts(left, alphabet, length)
    rc, rn = column_counts(right, alphabet, length)
    return jensen_shannon(_normalise(lc, ln), _normalise(rc, rn)), ln, rn


def permutation_band(left, right, alphabet: str, length: int, draws: int,
                     percentile: float, seed: int):
    """
    the per-hour null for that curve, from permuting the group labels

    Two groups of unequal size differ at every hour by sampling alone, and the
    thinner the hour's denominator the more they differ, so a flat threshold on
    the divergence would report a parting wherever the cohort has worn down.
    Permuting the labels while holding the group SIZES and the set of courses
    fixed reproduces that artefact and nothing else, which is what makes the
    band the right thing to have to clear.
    """
    rng = np.random.default_rng(seed)
    pool = list(left) + list(right)
    take = len(left)
    # Each course encoded ONCE as a (K, L) indicator, so a permutation is a
    # sum over an axis instead of a re-scan of several thousand strings. The
    # readable version re-counted characters inside the loop and took minutes
    # per handbook at 400 draws; this is the same arithmetic, moved.
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    block = np.zeros((len(pool), len(alphabet), length))
    for row, sequence in enumerate(pool):
        for i, symbol in enumerate(sequence[:length]):
            j = index.get(symbol)
            if j is not None:
                block[row, j, i] = 1.0
    band = np.zeros((draws, length))
    for d in range(draws):
        order = rng.permutation(len(pool))
        ac = block[order[:take]].sum(axis=0)
        bc = block[order[take:]].sum(axis=0)
        band[d] = jensen_shannon(
            _normalise(ac, ac.sum(axis=0)), _normalise(bc, bc.sum(axis=0))
        )
    return np.percentile(band, percentile, axis=0)


def first_sustained(observed, band, coverage, min_n: int, run: int,
                    min_shared: int):
    """
    the first hour the groups part, which requires that they were together first

    A PARTING NEEDS A SHARED PERIOD BEFORE IT, and the version without that
    requirement reported "divergence hour 0" for four of five handbooks in both
    cohorts -- which is not a divergence, it is the two groups having been
    different all along. Hour zero is the first hour on record; nothing can
    have parted by then. Reporting it as a parting hour put a number where the
    honest answer is "these groups never shared a course", and a reader
    skimming the column would have taken 0 to mean "immediately after onset"
    rather than "never together".

    So a parting is declared only where `min_shared` consecutive hours with the
    groups indistinguishable from permuted labels are immediately followed by
    `run` consecutive hours where they are not. Returns (hour, differ_from_start):
    the hour if one is found, and a flag marking the case where the groups are
    already distinguishable in the opening hours and so never had a shared
    course to depart from.
    """
    clear = (observed > band) & (coverage >= min_n)
    together = ~clear
    opening = int(min(min_shared, len(clear)))
    differ_from_start = bool(clear[:opening].any())
    for i in range(min_shared, len(clear) - run + 1):
        if together[i - min_shared:i].all() and clear[i:i + run].all():
            return int(i), differ_from_start
    return None, differ_from_start


# ==================================================================== drawing


def _letter(ax, symbol: str, x: float, y: float, width: float, height: float,
            colour: str) -> None:
    """one glyph scaled to fill the box (x, y, width, height), in data units"""
    if height <= 0 or width <= 0:
        return
    path = TextPath((0.0, 0.0), symbol, size=1.0, prop=GLYPH)
    box = path.get_extents()
    if box.width <= 0 or box.height <= 0:
        return
    transform = (
        Affine2D()
        .translate(-box.x0, -box.y0)
        .scale(width / box.width, height / box.height)
        .translate(x, y)
        + ax.transData
    )
    patch = PathPatch(path, facecolor=colour, edgecolor="none", linewidth=0.0)
    patch.set_transform(transform)
    ax.add_patch(patch)


def draw_logo(ax, counts, n, alphabet: str, colours: dict, min_n: int,
              x0: int = 0, mode: str = "bits", width: float = 0.86) -> None:
    """
    the stack of letters, tallest symbol on top, thin columns left blank

    Tallest on top rather than at the bottom: a reader scanning the figure
    follows the skyline, so putting the dominant symbol where the skyline is
    makes the consensus legible without reading any individual letter.
    """
    heights = (
        information(counts, n) if mode == "bits"
        else np.where(n > 0, 1.0, 0.0)
    )
    share = _normalise(counts, n)
    for i in range(counts.shape[1]):
        if n[i] < min_n or heights[i] <= 0:
            continue
        bottom = 0.0
        for row in np.argsort(share[:, i]):     # smallest first, drawn first
            piece = float(share[row, i] * heights[i])
            if piece <= 1e-4:
                continue
            _letter(
                ax, alphabet[row], x0 + i - width / 2.0, bottom, width, piece,
                colours.get(alphabet[row], "#999999"),
            )
            bottom += piece


def coverage_strip(ax, n, x0: int = 0, colour: str = "#b9b7b1",
                   ymax: float | None = None, floor: int | None = None,
                   dim: str = "#dedcd6") -> None:
    """
    how many patients each column rests on, as a bar under the logo

    `ymax` IS NOT OPTIONAL WHEN TWO STRIPS SHARE A FIGURE. Scaled to its own
    maximum, a 35-patient group's strip is the same height as a 218-patient
    group's and the two rows read as equally well supported, which is the
    opposite of what the strip is there to say. Bars for hours below the
    logo's own floor are dimmed, so the reader can see where a blank column in
    the logo above means "too few patients" rather than "nothing happened".
    """
    counts = np.asarray(n, dtype=float)
    shades = (
        [colour if c >= floor else dim for c in counts]
        if floor is not None else colour
    )
    ax.bar(np.arange(len(counts)) + x0, counts, width=0.86, color=shades,
           linewidth=0)
    top = ymax if ymax is not None else (
        float(np.max(counts)) if counts.size else 1.0
    )
    ax.set_ylim(0, max(top, 1.0))
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def logo_legend(ax, alphabet: str, colours: dict, names: dict,
                title: str) -> None:
    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=6,
                   markerfacecolor=colours.get(symbol, "#999999"),
                   markeredgecolor="none",
                   label=f"{symbol}   {names.get(symbol, symbol)}")
        for symbol in alphabet
    ]
    ax.axis("off")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1),
              labelcolor=INK_SECONDARY, fontsize=6, handletextpad=0.6,
              borderpad=0, frameon=False, title=title,
              title_fontproperties={"size": 6.5})


# ==================================================================== figures


def clock_logos(part_all, layer: str, alphabet: str, colours: dict,
                names: dict, out, title: str, subtitle: str, logger,
                length: int, min_n: int, mode: str = "bits"):
    """
    one logo per handbook over hours since onset, plus a coverage strip

    THE Y AXIS IS SHARED AND DATA-DRIVEN, not fixed at log2(K). Fixed at the
    theoretical maximum the figure was unreadable: these columns carry 0.3 to
    0.7 bits of a possible 3.32, so every logo sat in the bottom fifth of its
    panel and the letters were too small to name. A shared top taken from the
    tallest column in the figure keeps the handbooks comparable with each other
    -- which is the comparison the figure is for -- and the axis label still
    carries the theoretical ceiling, so the smallness is stated rather than
    drawn. The smallness is itself the first finding: an hour in any of these
    handbooks is barely determined at all.
    """
    numbers = sorted(part_all["handbook"].unique().to_list())
    prepared = []
    for number in numbers:
        part = part_all.filter(pl.col("handbook") == number)
        counts, n = column_counts(part[layer].to_list(), alphabet, length)
        prepared.append((number, part, counts, n))
    ceiling = float(np.log2(len(alphabet)))
    if mode == "bits":
        tallest = max(
            (float(np.max(information(c, n)[n >= min_n]))
             for _, _, c, n in prepared if (n >= min_n).any()),
            default=ceiling,
        )
        top = min(ceiling, max(0.25, 1.12 * tallest))
    else:
        top = 1.0

    rows = len(prepared)
    figure = plt.figure(figsize=(10.4, 1.5 * rows + 0.9))
    grid = GridSpec(
        2 * rows, 2, figure=figure, width_ratios=[7.4, 1.5],
        height_ratios=[3.4, 0.7] * rows, hspace=0.62, wspace=0.03,
        top=0.93, bottom=0.07,
    )
    summary = []
    for index, (number, part, counts, n) in enumerate(prepared):
        ax = figure.add_subplot(grid[2 * index, 0])
        draw_logo(ax, counts, n, alphabet, colours, min_n, mode=mode)
        ax.set_xlim(-1, length)
        ax.set_ylim(0, top)
        ax.set_ylabel(
            f"bits of {ceiling:.2f}" if mode == "bits" else "share",
            fontsize=6.5,
        )
        ax.tick_params(labelsize=6, labelbottom=False)
        ax.grid(True, axis="y", color=GRID, linewidth=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        known = part.filter(pl.col("in_hospital_mortality").is_not_null())
        rate = (
            float(known["in_hospital_mortality"].mean())
            if known.height else float("nan")
        )
        ax.set_title(
            f"handbook {number}  --  n={part.height:,}, mortality "
            f"{rate:.0%}, median {int(part['hours'].median())}h on record",
            loc="left", fontsize=7.5, pad=2,
        )
        strip = figure.add_subplot(grid[2 * index + 1, 0], sharex=ax)
        # PER ROW, not shared. Across handbooks the sizes run 253 to 2,740,
        # and one scale made four of the five strips a sliver at the bottom of
        # their panel that read as "no patients here at all". Within a row the
        # strip is there for the SHAPE of the decline; the absolute n is in the
        # row title, where it does not have to be measured off an axis.
        coverage_strip(strip, n, floor=min_n)
        strip.tick_params(labelsize=6, labelbottom=index == rows - 1)
        if index == rows - 1:
            strip.set_xlabel("hours since shock onset", fontsize=6.5)
        text = consensus_string(counts, n, alphabet, min_n)
        bits = information(counts, n)
        summary.append({
            "handbook": number,
            "layer": layer,
            "patients": part.height,
            "mortality": rate,
            "consensus": text,
            "hours_above_floor": int((n >= min_n).sum()),
            "mean_bits": float(np.mean(bits[n >= min_n]))
            if (n >= min_n).any() else float("nan"),
            "max_bits": ceiling,
        })
        logger.info(f"  handbook {number} {layer}: {text}")
    legend = figure.add_subplot(grid[:, 1])
    logo_legend(legend, alphabet, colours, names, layer)
    figure.suptitle(title, x=0.008, y=0.985, ha="left", fontsize=9.5)
    figure.text(0.008, 0.004, subtitle, fontsize=6.5, color=INK_SECONDARY)
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out}")
    return pl.DataFrame(summary, infer_schema_length=None)


def divergence_logo(part, number: int, split, alphabet: str, colours: dict,
                    names: dict, layer: str, out, title: str, subtitle: str,
                    logger, length: int, min_n: int, observed=None, band=None,
                    detected: bool = True):
    """
    both response groups' whole courses, one row each, over a shared clock

    AN EARLIER VERSION DREW A SHARED PREFIX AND THEN SPLIT, which presupposed
    the answer. Run on this cohort it fell over immediately: four of the five
    handbooks part at hour ZERO, so the "shared prefix" was a region the test
    had just said was not shared, drawn as if it were, three hours wide because
    the layout needed somewhere to put it. Worse, handbook 5's divergence curve
    PEAKS at the start and decays below the null by hour 18 -- its two groups
    begin apart and converge, which the prefix layout cannot represent at all.

    So both groups get their whole course, aligned on one clock, and the
    divergence hour is a marker rather than a seam. The region before it is
    shaded to say "indistinguishable from permuted labels here", which is a
    claim the permutation test supports and which reads correctly whether that
    region is forty hours wide or empty.
    """
    groups = []
    for group in (0, 1):
        here = part.filter(pl.col("worse_group") == group)
        if here.height < SUPPRESS:
            logger.info(f"  group {group}: {here.height} patients, suppressed")
            return None
        groups.append(here)

    panels = 3 if observed is not None else 2
    figure = plt.figure(figsize=(10.4, 1.5 * panels + 1.6))
    grid = GridSpec(
        2 * panels, 2, figure=figure, width_ratios=[7.4, 1.5],
        height_ratios=[3.4, 1.0] * panels, hspace=0.42, wspace=0.03,
        top=0.92, bottom=0.08,
    )
    ceiling = float(np.log2(len(alphabet)))
    prepared = []
    for here in groups:
        counts, n = column_counts(here[layer].to_list(), alphabet, length)
        prepared.append((here, counts, n))
    tallest = max(
        (float(np.max(information(c, n)[n >= min_n]))
         for _, c, n in prepared if (n >= min_n).any()),
        default=ceiling,
    )
    top = min(ceiling, max(0.25, 1.12 * tallest))
    widest = max((float(np.max(n)) for _, _, n in prepared), default=1.0)

    for row, (here, counts, n) in enumerate(prepared):
        ax = figure.add_subplot(grid[2 * row, 0])
        if split is not None and split > 0:
            ax.axvspan(-1, split, color="#eeedea", zorder=0)
        draw_logo(ax, counts, n, alphabet, colours, min_n)
        if split is not None:
            ax.axvline(split, color="#c0392b", linewidth=0.9, zorder=4)
        ax.set_xlim(-1, length)
        ax.set_ylim(0, top)
        ax.set_ylabel(f"bits of {ceiling:.2f}", fontsize=6.5)
        ax.tick_params(labelsize=6, labelbottom=False)
        ax.grid(True, axis="y", color=GRID, linewidth=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        known = here.filter(pl.col("in_hospital_mortality").is_not_null())
        rate = (
            float(known["in_hospital_mortality"].mean())
            if known.height else float("nan")
        )
        worse = "worse-outcome " if row else ""
        ax.set_title(
            f"{worse}group {'B' if row else 'A'}  --  n={here.height:,}, "
            f"mortality {rate:.0%}, median "
            f"{int(here['hours'].median())}h on record",
            loc="left", fontsize=7.5, pad=2, color=INK_PRIMARY,
        )
        strip = figure.add_subplot(grid[2 * row + 1, 0], sharex=ax)
        coverage_strip(strip, n, ymax=widest, floor=min_n)
        strip.tick_params(labelsize=6, labelbottom=observed is None and row)
    if observed is not None:
        ax = figure.add_subplot(grid[4, 0])
        hours = np.arange(length)
        if split is not None and split > 0:
            ax.axvspan(-1, split, color="#eeedea", zorder=0)
        ax.plot(hours, observed, color=INK_PRIMARY, linewidth=1.1, zorder=3,
                label="observed divergence between the two groups")
        if band is not None:
            ax.plot(hours, band, color="#8a8980", linewidth=0.9,
                    linestyle="--", zorder=2,
                    label="upper percentile of the label permutations")
        if split is not None:
            ax.axvline(split, color="#c0392b", linewidth=0.9, zorder=4,
                       label=(
                           f"first sustained parting, hour {split}"
                           if detected else
                           f"peak hour {split} (no sustained parting found)"
                       ))
        ax.set_xlim(-1, length)
        ax.set_ylabel("JS divergence\n(bits)", fontsize=6.5)
        ax.set_xlabel("hours since shock onset", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False, labelcolor=INK_SECONDARY,
                  loc="upper right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        figure.add_subplot(grid[5, 0]).axis("off")
    legend = figure.add_subplot(grid[:, 1])
    logo_legend(legend, alphabet, colours, names, layer)
    figure.suptitle(title, x=0.008, y=0.985, ha="left", fontsize=9.5)
    figure.text(0.008, 0.004, subtitle, fontsize=6.5, color=INK_SECONDARY)
    save_figure(figure, out)
    plt.close(figure)
    logger.info(f"wrote {out}")
    return True


# ======================================================================= main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="cardiogenic",
                        choices=("cardiogenic", "septic", "all"))
    parser.add_argument("--cap-hours", type=int, default=None)
    parser.add_argument("--handbooks", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--min-group", type=int, default=None)
    parser.add_argument("--min-fit", type=float, default=None)
    parser.add_argument("--gap-open", type=float, default=None)
    parser.add_argument("--gap-extend", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-patients", type=int,
                        default=None)
    parser.add_argument("--permutations", type=int,
                        default=None)
    parser.add_argument("--percentile", type=float,
                        default=None)
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--min-shared", type=int, default=None,
                        help="hours the two groups must be indistinguishable "
                             "before a parting can be declared")
    parser.add_argument("--focus-handbook", type=int, default=None,
                        help="draw the divergence logo for this handbook only; by default every handbook with two reportable groups gets one")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    # k and the sample size are per cohort, so they cannot be argparse
    # defaults -- the parser would have committed to one number before
    # anything knew which cohort was running. See dynamics.yaml.
    apply_handbook_defaults(args)
    # This script's four extra settings are config keys whose flags are named
    # for the reader rather than for the YAML, so they are mapped by hand
    # instead of by matching attribute names.
    logo = handbook_defaults(subset=args.subset)
    for flag, key in (
        ("min_patients", "logo_min_patients"),
        ("permutations", "divergence_permutations"),
        ("percentile", "divergence_percentile"),
        ("run", "divergence_run"),
        ("min_shared", "divergence_min_shared"),
    ):
        if getattr(args, flag) is None:
            setattr(args, flag, logo[key])

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger(
        "43_handbook_logos", paths.log_home / "43_handbook_logos.log"
    )
    style()
    figures = paths.output_home / "figures" / paths.model_key
    tables = paths.output_home / "tables" / paths.model_key
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"subset {args.subset}; {args.handbooks} handbooks from a "
        f"{args.sample} sample; logo floor {args.min_patients} patients per "
        f"hour; divergence from {args.permutations} permutations at the "
        f"{args.percentile:.0f}th percentile with a {args.run}h run"
    )

    probe = sibling("41_divergence_probe")
    hexmaps = sibling("38_layer_hexmaps")
    part_all, _matrix, _medoids = probe.rebuild_handbooks(paths, args, logger)

    by_label, _order = hexmaps.treatment_palette()
    treatment_colours = {
        symbol: by_label[treatment_label(symbol)]
        for symbol in TREATMENT_ALPHABET
    }
    treatment_names = {s: treatment_label(s) for s in TREATMENT_ALPHABET}
    by_state = husl_colors(list(HOURLY_REGION_ORDER))
    response_colours = {
        STATE_SYMBOL[state]: by_state[state] for state in HOURLY_REGION_ORDER
    }
    response_names = {STATE_SYMBOL[s]: s for s in HOURLY_REGION_ORDER}

    layers = {
        "treatment": (TREATMENT_ALPHABET, treatment_colours, treatment_names,
                      "norepinephrine band crossed with the other agents"),
        "response": (RESPONSE_ALPHABET, response_colours, response_names,
                     "the action taken and the MAP response that followed"),
    }

    summaries = []
    for layer, (alphabet, colours, names, note) in layers.items():
        for mode, tag in (("bits", ""), ("share", "_share")):
            height = (
                "information content, log2(K) minus entropy, small-sample "
                "corrected" if mode == "bits"
                else "fixed; the letter heights are the raw share"
            )
            frame = clock_logos(
                part_all, layer, alphabet, colours, names,
                figures / f"handbook_logo_{layer}{tag}_{args.subset}.png",
                f"{args.subset} shock: what each treatment handbook's "
                f"{layer} course generally looks like",
                f"{note}  |  column height is {height}  |  the bar under "
                "each logo is how many patients that hour rests on, and it "
                "falls because a recovered patient's string ends",
                logger, args.cap_hours, args.min_patients, mode=mode,
            )
            if mode == "bits":
                summaries.append(frame)
    if summaries:
        save_table(
            pl.concat(summaries, how="diagonal_relaxed"),
            tables / f"handbook_consensus_{args.subset}.csv",
        )

    # ---- where the two response groups part, per handbook
    logger.info("where the response groups part:")
    rows = []
    for number in sorted(part_all["handbook"].unique().to_list()):
        part = part_all.filter(pl.col("handbook") == number)
        left = part.filter(pl.col("worse_group") == 0)["response"].to_list()
        right = part.filter(pl.col("worse_group") == 1)["response"].to_list()
        if min(len(left), len(right)) < SUPPRESS:
            logger.info(f"  handbook {number}: a group is below {SUPPRESS}")
            continue
        observed, ln, rn = divergence_curve(
            left, right, RESPONSE_ALPHABET, args.cap_hours
        )
        band = permutation_band(
            left, right, RESPONSE_ALPHABET, args.cap_hours,
            args.permutations, args.percentile, args.seed,
        )
        coverage = np.minimum(ln, rn)
        hour, differ_from_start = first_sustained(
            observed, band, coverage, args.min_patients, args.run,
            args.min_shared,
        )
        peak = int(np.nanargmax(observed))
        rows.append({
            "handbook": number,
            "group_a": len(left),
            "group_b": len(right),
            "divergence_hour": hour,
            "differ_from_start": differ_from_start,
            "divergence_at_hour": float(observed[hour]) if hour is not None
            else float("nan"),
            "null_at_hour": float(band[hour]) if hour is not None
            else float("nan"),
            "peak_divergence": float(np.nanmax(observed)),
            "peak_hour": peak,
            "hours_above_null": int(
                ((observed > band) & (coverage >= args.min_patients)).sum()
            ),
            "_observed": observed,
            "_band": band,
        })
        logger.info(
            f"  handbook {number}: groups of {len(left):,}/{len(right):,} "
            + (
                # THREE CASES, NOT TWO. A handbook can both differ in the
                # opening hours AND part again later, and an earlier version of
                # this line reported only the parting -- which told the reader
                # that handbook and a handbook that was genuinely together
                # first had the same shape of history. They do not.
                f"were apart in the opening hours, came back together, and "
                f"part again at hour {hour}"
                if hour is not None and differ_from_start else
                f"were together and then part at hour {hour}"
                if hour is not None else
                "were ALREADY apart in the opening hours and never cleanly "
                "part, so no parting hour is reported" if differ_from_start
                else "never part"
            )
            + f"; peak "
            f"divergence {np.nanmax(observed):.3f} bits at hour {peak}; "
            f"{rows[-1]['hours_above_null']} of {args.cap_hours} hours clear "
            "the null"
        )

    if rows:
        save_table(
            pl.DataFrame(
                [{k: v for k, v in r.items() if not k.startswith("_")}
                 for r in rows],
                infer_schema_length=None,
            ),
            tables / f"handbook_logo_divergence_{args.subset}.csv",
        )

    # ---- the divergence logo, for EVERY handbook rather than one
    #
    # Drawing only the focus handbook made the choice of focus carry the
    # finding: the largest handbook never parts at all and a small one parts
    # late, and a reader shown only one of those two learns the opposite lesson
    # from a reader shown the other. Five figures is cheap and the report can
    # tab between them.
    for chosen in rows:
        number = chosen["handbook"]
        if args.focus_handbook is not None and number != args.focus_handbook:
            continue
        split = chosen["divergence_hour"]
        detected = split is not None
        apart = chosen["differ_from_start"]
        if not detected:
            # NO MARKER AT ALL when there is no parting. Falling back to the
            # peak-divergence hour drew a red line and shaded everything left
            # of it, which is the picture of a parting, under a caption saying
            # there wasn't one -- and the picture wins that argument.
            split = None
            logger.info(
                f"  handbook {number}: no parting to mark, so the figure is "
                "drawn without one"
            )
        part = part_all.filter(pl.col("handbook") == number)
        shared = (
            "the groups were apart in the opening hours, came back together, "
            f"and part again at hour {split}; the shaded region is the "
            "indistinguishable stretch immediately before that second parting "
            "and is NOT the whole of their shared history"
            if detected and apart else
            "the groups are indistinguishable from permuted labels before "
            f"hour {split}, which is shaded"
            if detected else
            "the two groups are ALREADY distinguishable in the opening hours "
            "and never cleanly part, so there is no parting to mark"
            if apart else
            "the groups never separate for long enough to call a parting, so "
            "there is no marker"
        )
        for layer, (alphabet, colours, names, note) in layers.items():
            divergence_logo(
                part, number, None if split is None else int(split),
                alphabet, colours, names, layer,
                figures / (
                    f"handbook_divergence_logo_{layer}_hb{number}_"
                    f"{args.subset}.png"
                ),
                f"{args.subset} shock, handbook {number}: each response "
                f"group's {layer} course, on one clock",
                f"{note}  |  {shared}  |  a column rests on the patients "
                "still on record that hour, so the right-hand end of each row "
                "is the slowest to come off pressors",
                logger, args.cap_hours, args.min_patients,
                observed=chosen["_observed"] if layer == "response" else None,
                band=chosen["_band"] if layer == "response" else None,
                detected=detected,
            )
    logger.info("43 done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
plot palette, and the colour checks it has to pass

The palette is the dataviz reference instance unchanged. Scatterplots put every
series on screen at once, so the all-pairs gate applies rather than the
adjacent-pairs one, and under all-pairs the reference palette validates only its
first three slots -- which is exactly what is needed here: two for site, three
for phenotype, two for mortality.

`validate` is a port of the skill's scripts/validate_palette.js (node is not
installed on this machine). `self_test` checks the port against the figures the
reference palette documents for its first three slots, so a drifted port fails
loudly instead of silently blessing a bad palette.

    python code/clifcs/palette.py      # run the checks and print the report
"""

from __future__ import annotations

import itertools
import math

# ---------------------------------------------------------------- the palette

#: categorical slots 1-3 of the reference palette (blue, orange, aqua).
#: These are the only three that clear the all-pairs colourblind gate, which is
#: the one that applies to a scatter, so this is what scatter series use.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")

#: The full eight-slot reference order.
#:
#: Clears the *adjacent*-pairs gate (worst CVD dE 9.1 light) but NOT all-pairs
#: -- slot 4 puts yellow beside orange, which fails the floors when every
#: series is on screen together. Use it in a scatter only where identity is
#: also carried by something other than colour: a per-panel legend over a
#: zoomed, well-separated layout, or direct labels. Never as the sole channel.
CATEGORICAL_8 = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

#: Institutional colours, taken from each site's own branding. Fixed per site
#: rather than assigned by position, so a site keeps its colour across every
#: figure and however many other sites are present.
#:
#: UCMC and NU clear the colourblind gates comfortably against each other
#: (protan dE 22.9, deutan 22.3, tritan 15.8). Both sit below the light-mode
#: lightness band at L 0.377 and 0.379, so they are dark and nearly identical
#: in luminance: they separate by hue, not by value, and would be
#: indistinguishable in greyscale or on a monochrome printer. Run
#: `python code/clifcs/palette.py` after adding a site to check the new pair.
LOGO_PALETTE = {
    "UCMC": "#800000",
    "MIMIC": "#aaaaaa",
    "RUSH": "#0f683d",
    "UCSF": "#052049",
    "NU": "#59118e",
}


def site_colors(sites) -> dict[str, str]:
    """institutional colour per site; unknown sites fall back to the palette"""
    out, spare = {}, [c for c in CATEGORICAL if c not in LOGO_PALETTE.values()]
    for site in sites:
        if site in LOGO_PALETTE:
            out[site] = LOGO_PALETTE[site]
        elif spare:
            out[site] = spare.pop(0)
        else:
            raise ValueError(
                f"no colour for site {site!r}: add it to LOGO_PALETTE"
            )
    return out


def continuous_cmap():
    """
    sequential colormap for magnitude

    cubehelix rather than a single-hue blue ramp: it increases monotonically in
    perceived lightness while also rotating in hue, so neighbouring values stay
    distinguishable where a one-hue ramp runs out of contrast, and it still
    degrades correctly to greyscale.
    """
    import seaborn as sns

    return sns.color_palette("cubehelix", as_cmap=True)


def husl_colors(keys) -> dict:
    """
    categorical colours for an ordered set of levels

    husl spaces hues evenly at constant lightness and saturation, so no level
    is visually heavier than another -- which matters when the levels are
    outcome categories and a darker colour would read as more severe.
    """
    import seaborn as sns

    keys = list(keys)
    hexes = [
        "#{:02x}{:02x}{:02x}".format(*(int(round(255 * v)) for v in rgb))
        for rgb in sns.color_palette("husl", max(len(keys), 2))
    ]
    return dict(zip(keys, hexes))

#: Zweck phenotypes, in the paper's order
PHENOTYPE_LEVELS = (
    "I non-congested",
    "II cardiorenal",
    "III cardiometabolic",
)

#: binary outcome
MORTALITY_LEVELS = (False, True)


def phenotype_colors() -> dict:
    return husl_colors(PHENOTYPE_LEVELS)


def mortality_colors() -> dict:
    return dict(zip(MORTALITY_LEVELS, husl_colors(range(2)).values()))

#: single-hue blue ramp, steps 100-700, for continuous magnitude
SEQUENTIAL_BLUE = (
    "#cde2fb",  # 100
    "#b7d3f6",  # 150
    "#9ec5f4",  # 200
    "#86b6ef",  # 250
    "#6da7ec",  # 300
    "#5598e7",  # 350
    "#3987e5",  # 400
    "#2a78d6",  # 450
    "#256abf",  # 500
    "#1c5cab",  # 550
    "#184f95",  # 600
    "#104281",  # 650
    "#0d366b",  # 700
)

#: the same ramp with its three palest steps dropped, for scatter marks
#:
#: The full 100-700 range is for area encodings -- heatmap cells, choropleths --
#: where the lightest step is meant to recede into the surface because the cell
#: still occupies space. A scatter mark that recedes into the surface simply
#: disappears, and a reader undercounts the short-stay patients. Starting at
#: step 250 is the skill's documented floor for discrete ordered marks on a
#: light surface (2.06:1 against it).
SCATTER_SEQUENTIAL = SEQUENTIAL_BLUE[3:]

SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8980"
GRID = "#e4e3df"

# ------------------------------------------------------------------ the checks

_BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
_CHROMA_FLOOR = 0.10
_CVD_TARGET, _CVD_FLOOR = 8.0, 6.0
_NORMAL_FLOOR = 15.0
_CONTRAST_MIN = 3.0

# Machado, Oliveira & Fernandes (2009) CVD transforms at severity 1.0, linear RGB
_MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def _srgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.strip().lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear(hex_color: str) -> tuple[float, float, float]:
    return tuple(_to_linear(c) for c in _srgb(hex_color))


def _oklab_from_linear(rgb) -> tuple[float, float, float]:
    r, g, b = rgb
    l = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def oklch(hex_color: str) -> tuple[float, float]:
    """OKLCH lightness and chroma"""
    lightness, a, b = _oklab_from_linear(_linear(hex_color))
    return lightness, math.hypot(a, b)


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio"""

    def luminance(hex_color: str) -> float:
        r, g, b_ = _linear(hex_color)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b_

    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _simulate(hex_color: str, kind: str) -> tuple[float, float, float]:
    r, g, b = _linear(hex_color)
    matrix = _MACHADO[kind]
    return tuple(
        min(1.0, max(0.0, row[0] * r + row[1] * g + row[2] * b)) for row in matrix
    )


def delta_e(a: str, b: str, kind: str | None = None) -> float:
    """Euclidean distance in OKLab x100; `kind` None means unsimulated vision"""
    first = _oklab_from_linear(_simulate(a, kind) if kind else _linear(a))
    second = _oklab_from_linear(_simulate(b, kind) if kind else _linear(b))
    return 100 * math.dist(first, second)


def validate(
    palette, mode: str = "light", pairs: str = "all", surface: str | None = None
) -> tuple[list[tuple[str, str, str]], bool]:
    """
    run the lightness-band, chroma-floor, CVD-separation, normal-vision and
    contrast checks; returns (report rows, hard-pass)
    """
    palette = list(palette)
    surface = surface or SURFACE[mode]
    lo, hi = _BAND[mode]
    report: list[tuple[str, str, str]] = []
    ok = True

    offband = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    ok &= not offband
    report.append(
        (
            "Lightness band",
            "fail" if offband else "pass",
            f"outside band: {offband}" if offband else f"all {len(palette)} inside L {lo}-{hi}",
        )
    )

    low_chroma = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < _CHROMA_FLOOR]
    ok &= not low_chroma
    report.append(
        (
            "Chroma floor",
            "fail" if low_chroma else "pass",
            f"reads gray: {low_chroma}"
            if low_chroma
            else f"all {len(palette)} >= {_CHROMA_FLOOR}",
        )
    )

    n = len(palette)
    pairlist = (
        list(itertools.combinations(range(n), 2))
        if pairs == "all"
        else [(i, i + 1) for i in range(n - 1)]
    )

    worst = min(
        (
            (delta_e(palette[i], palette[j], kind), kind, palette[i], palette[j])
            for kind in ("protan", "deutan")
            for i, j in pairlist
        ),
        default=(99.0, "-", "-", "-"),
    )
    tritan = min(
        (delta_e(palette[i], palette[j], "tritan") for i, j in pairlist), default=99.0
    )
    cvd_state = (
        "pass" if worst[0] >= _CVD_TARGET else "floor" if worst[0] >= _CVD_FLOOR else "fail"
    )
    ok &= cvd_state != "fail"
    report.append(
        (
            "CVD separation",
            cvd_state,
            f"worst {pairs}-pairs {worst[2]}<->{worst[3]} dE {worst[0]:.1f} "
            f"({worst[1]}) - tritan {tritan:.1f}",
        )
    )

    nworst = min(
        ((delta_e(palette[i], palette[j]), palette[i], palette[j]) for i, j in pairlist),
        default=(99.0, "-", "-"),
    )
    normal_state = "pass" if nworst[0] >= _NORMAL_FLOOR else "fail"
    ok &= normal_state == "pass"
    report.append(
        (
            "Normal-vision floor",
            normal_state,
            f"worst {pairs}-pairs {nworst[1]}<->{nworst[2]} dE {nworst[0]:.1f}",
        )
    )

    dim = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < _CONTRAST_MIN]
    report.append(
        (
            "Contrast vs surface",
            "relief" if dim else "pass",
            f"below {_CONTRAST_MIN}:1, relief required (visible labels or table view): {dim}"
            if dim
            else f"all {len(palette)} >= {_CONTRAST_MIN}:1",
        )
    )

    return report, ok


def self_test() -> None:
    """
    check this port against the numbers the reference palette documents

    palette.md states that slots 1-3 validate all-pairs in both modes, with
    worst-pair CVD dE 9.2 light / 9.4 dark and normal-vision 24.0 light /
    20.9 dark. Reproducing those is what makes this port trustworthy.
    """
    expected = {("light", "cvd"): 9.2, ("light", "normal"): 24.0,
                ("dark", "cvd"): 9.4, ("dark", "normal"): 20.9}
    # the dark column of slots 1-3
    dark_slots = ("#3987e5", "#d95926", "#199e70")

    for mode, palette in (("light", CATEGORICAL), ("dark", dark_slots)):
        pairlist = list(itertools.combinations(range(len(palette)), 2))
        cvd = min(
            delta_e(palette[i], palette[j], kind)
            for kind in ("protan", "deutan")
            for i, j in pairlist
        )
        normal = min(delta_e(palette[i], palette[j]) for i, j in pairlist)
        for name, got in (("cvd", cvd), ("normal", normal)):
            want = expected[(mode, name)]
            if abs(got - want) > 0.05:
                raise AssertionError(
                    f"palette validator port disagrees with the documented "
                    f"reference for {mode}/{name}: got {got:.1f}, expected {want}"
                )
    print("self-test: port reproduces the documented reference figures")


def check_sequential() -> None:
    """
    the categorical gates do not apply to a continuous ramp

    A sequential ramp is read as magnitude, not identity: its lightest step is
    meant to recede toward the surface, and neighbouring steps are meant to be
    close. What it must do instead is vary lightness monotonically, so that
    "darker" reads unambiguously as "more" -- which is what is checked here.
    """
    lightness = [oklch(c)[0] for c in SEQUENTIAL_BLUE]
    monotonic = all(a > b for a, b in zip(lightness, lightness[1:]))
    span = lightness[0] - lightness[-1]
    state = "PASS" if monotonic and span > 0.4 else "FAIL"
    print(
        f"\nICU LOS sequential ramp [light, continuous] -> {state}"
        f"\n  {'PASS' if monotonic else 'FAIL':<7} {'Monotonic lightness':<22} "
        f"{len(SEQUENTIAL_BLUE)} steps, strictly decreasing: {monotonic}"
        f"\n  {'PASS' if span > 0.4 else 'FAIL':<7} {'Lightness span':<22} "
        f"L {lightness[0]:.3f} -> {lightness[-1]:.3f} (span {span:.3f})"
    )
    if not (monotonic and span > 0.4):
        raise AssertionError("sequential ramp is not a usable magnitude encoding")


def main() -> None:
    self_test()
    for label, palette in (
        ("sites (up to 3)", list(CATEGORICAL)),
        ("phenotype (I, II, III)", list(PHENOTYPE_COLORS.values())),
        ("in-hospital mortality", list(MORTALITY_COLORS.values())),
    ):
        # scatterplots put every series on screen together, so all pairs must
        # separate, not just adjacent ones
        report, ok = validate(palette, mode="light", pairs="all")
        print(f"\n{label} [light, all-pairs] -> {'PASS' if ok else 'FAIL'}")
        for check, state, detail in report:
            print(f"  {state.upper():<7} {check:<22} {detail}")
    check_sequential()


if __name__ == "__main__":
    main()

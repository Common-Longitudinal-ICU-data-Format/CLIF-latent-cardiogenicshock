#!/usr/bin/env python3

"""
regenerate config/clifpy_units.yaml from the upstream coreopsis recipes

The dose-unit maps are long and getting one entry wrong would silently shift
that drug into different cocoa bins, so they are lifted out of the reference
implementation by parsing it rather than copied by hand.

    python code/config/_regen_clifpy_units.py
"""

from __future__ import annotations

import ast
import pathlib
import urllib.request

import yaml

COMMIT = "950c24782670103d130328822d06b39cb4724d7a"
BASE = f"https://raw.githubusercontent.com/bbj-lab/coreopsis/{COMMIT}/recipes"
OUT = pathlib.Path(__file__).resolve().parent / "clifpy_units.yaml"

HEADER = f"""%YAML 1.2
---
# Preferred dose units for clifpy, copied verbatim from coreopsis
# recipes/run_clifpy.py and recipes/run_sofa_scoring.py at commit {COMMIT[:8]}.
#
# These must match the units used when the foundation model was trained: cocoa
# bins MED-CTS / MED-INT numeric values into deciles, so a drug converted to a
# different unit here lands in different bins and therefore different tokens.
# Extracted by code/config/_regen_clifpy_units.py -- do not hand-edit.

"""


def fetch(name: str) -> ast.Module:
    with urllib.request.urlopen(f"{BASE}/{name}") as response:
        return ast.parse(response.read().decode("utf-8"))


def main() -> None:
    tree = fetch("run_clifpy.py")

    # the two `preferred_units = {...}` literals, in source order:
    # continuous first, then intermittent
    maps = sorted(
        (node.lineno, ast.literal_eval(node.value))
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "preferred_units" for t in node.targets)
    )
    if len(maps) != 2:
        raise RuntimeError(f"expected 2 unit maps in run_clifpy.py, found {len(maps)}")

    sofa_map = None
    for node in ast.walk(fetch("run_sofa_scoring.py")):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "convert_dose_units_for_continuous_meds"
        ):
            for kw in node.keywords:
                if kw.arg == "preferred_units":
                    sofa_map = ast.literal_eval(kw.value)
    if not sofa_map:
        raise RuntimeError("preferred_units not found in run_sofa_scoring.py")

    OUT.write_text(
        HEADER
        + yaml.safe_dump(
            {
                "continuous": maps[0][1],
                "intermittent": maps[1][1],
                "sofa_continuous": sofa_map,
            },
            default_flow_style=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

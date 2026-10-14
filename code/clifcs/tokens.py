#!/usr/bin/env python3

"""
reading the token streams the model actually saw

The embedding is a function of `tokens_past` -- the prefix up to the anchor --
and nothing else. Any explanation of why two sites separate in embedding space
has to be an explanation about these tokens, so this module reads them back and
decodes them to names and families.
"""

from __future__ import annotations

import functools
import pathlib

import numpy as np
import polars as pl
from omegaconf import OmegaConf

SPLITS = ("train", "tuning", "held_out")


@functools.lru_cache(maxsize=4)
def vocabulary(tokenizer_yaml: str) -> tuple[dict[int, str], dict[int, str]]:
    """id -> token name, and id -> family (the prefix before `//`)"""
    lookup = dict(OmegaConf.load(tokenizer_yaml).lookup)
    names = {int(i): str(name) for name, i in lookup.items()}
    families = {
        i: (name.split("//")[0] if "//" in name else name) for i, name in names.items()
    }
    return names, families


def load_pre_anchor_tokens(paths, site: str, anchor: str) -> pl.DataFrame:
    """
    one row per patient: the token ids the model read, and the elapsed seconds

    Concatenates cocoa's three splits, which partition the cohort.

    Site role only. A token stream is a reconstruction of the record -- it is
    the most disclosive thing in this pipeline, more so than the embedding
    derived from it -- so it never leaves a site, and a pooled-role run is
    stopped here rather than finding an empty scratch directory and reporting
    that every site contributed nothing.
    """
    if getattr(paths, "role", "site") == "pooled":
        from clifcs.config import RoleViolation

        raise RoleViolation(
            "token streams do not cross a site boundary: a full `tokens_past` "
            "sequence reconstructs the record it came from.\n"
            "What travels instead is per-patient token *family counts* "
            "(`fam_n_*` in the bundle), which is what the site-effect model in "
            "04 regresses on, plus the last pre-anchor decile bin per "
            "allowlisted variable (`tok_*`). Anything needing the sequence "
            "itself -- 08's token-space geometry -- has to run site-side, with "
            "only its results pooled."
        )
    frames = []
    for split in SPLITS:
        f = paths.anchored(site, anchor) / f"{split}_for_inference.parquet"
        if f.is_file():
            frames.append(
                pl.read_parquet(f, columns=["subject_id", "tokens_past", "s_elapsed_past"])
            )
    if not frames:
        raise FileNotFoundError(
            f"no *_for_inference.parquet under {paths.anchored(site, anchor)}"
        )
    return (
        pl.concat(frames)
        .rename({"subject_id": "hospitalization_id"})
        .with_columns(site=pl.lit(site))
    )


def family_matrix(
    frame: pl.DataFrame, families: dict[int, str], family_order: list[str]
) -> np.ndarray:
    """
    per-patient counts of tokens in each family

    Dense counts rather than a sparse term-document matrix: there are only ~23
    families, and keeping it dense makes the downstream regression trivial.
    """
    index = {f: i for i, f in enumerate(family_order)}
    out = np.zeros((frame.height, len(family_order)), dtype=np.float64)
    max_id = max(families) if families else 0
    codes = np.full(max_id + 1, -1, dtype=np.int64)
    for token_id, fam in families.items():
        if fam in index:
            codes[token_id] = index[fam]

    for row, tokens in enumerate(frame["tokens_past"].to_list()):
        if not len(tokens):
            continue
        mapped = codes[np.asarray(tokens, dtype=np.int64)]
        mapped = mapped[mapped >= 0]
        if mapped.size:
            np.add.at(out[row], mapped, 1.0)
    return out


def token_counts_by_site(
    frame: pl.DataFrame, names: dict[int, str]
) -> pl.DataFrame:
    """total occurrences of every token, per site"""
    rows = []
    for site, part in frame.group_by("site"):
        site = site[0] if isinstance(site, tuple) else site
        flat = np.concatenate(
            [np.asarray(t, dtype=np.int64) for t in part["tokens_past"].to_list() if len(t)]
        )
        ids, counts = np.unique(flat, return_counts=True)
        rows.append(
            pl.DataFrame(
                {
                    "site": [site] * len(ids),
                    "token_id": ids,
                    "token": [names.get(int(i), "?") for i in ids],
                    "n": counts,
                }
            )
        )
    out = pl.concat(rows)
    return out.with_columns(
        family=pl.when(pl.col("token").str.contains("//"))
        .then(pl.col("token").str.split("//").list.first())
        .otherwise(pl.col("token"))
    )

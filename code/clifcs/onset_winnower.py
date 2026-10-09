#!/usr/bin/env python3

"""
onset-anchored winnowing

cocoa's Winnower cuts every timeline at the same place: either a fixed elapsed
duration (`threshold.duration_s`) or the first occurrence of one token
(`threshold.first_occurrence`). Anchoring at cardiogenic shock onset needs a
per-patient timestamp, which neither option expresses, so `run_thresholding` is
overridden here. Everything downstream of the cut -- the past/future split, the
outcome flags, the output schema that `cotorra extract` consumes -- is cocoa's
unchanged, so the two anchors produce interchangeable inputs.
"""

from __future__ import annotations

import pathlib

import polars as pl
from cocoa.winnower import Winnower


class OnsetWinnower(Winnower):
    """cuts each timeline at that patient's `cs_onset_dttm`"""

    def __init__(
        self,
        winnowing_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        onset_parquet: pathlib.Path | str = None,
        **kwargs,
    ):
        super().__init__(
            winnowing_cfg=winnowing_cfg,
            processed_data_home=processed_data_home,
            **kwargs,
        )
        if onset_parquet is None:
            raise ValueError("onset_parquet is required")
        self.onset_parquet = pathlib.Path(onset_parquet).expanduser().resolve()
        if not self.onset_parquet.is_file():
            raise FileNotFoundError(f"onset table not found: {self.onset_parquet}")

        self.onsets = (
            pl.scan_parquet(self.onset_parquet)
            .select(
                subject_id=pl.col("hospitalization_id").cast(pl.String),
                # cocoa stores token times as naive UTC; match that exactly
                cs_onset_dttm=pl.col("cs_onset_dttm").cast(pl.Datetime("us")),
            )
            .unique(subset="subject_id")
        )

    def run_thresholding(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        set `last_valid` to the number of tokens at or before CS onset

        Timelines with too few pre-onset tokens are dropped: a representation
        taken from a one- or two-token prefix carries no patient history and
        would be an artefact of padding rather than a reading of the patient.
        """
        anchor = self.cfg.get("onset_anchor", {})
        inclusive = bool(anchor.get("inclusive", True))
        min_tokens_past = int(anchor.get("min_tokens_past", 2))

        joined = df.join(self.onsets, on="subject_id", how="inner")

        # `times` is chronologically ordered by the collator, which is what lets
        # cocoa treat `last_valid` as a prefix length; so counting the tokens at
        # or before the cutoff gives the cut point. list.eval() cannot reference
        # another column, hence the explode-and-count rather than an in-list
        # comparison.
        at_or_before = (
            pl.col("times") <= pl.col("cs_onset_dttm")
            if inclusive
            else pl.col("times") < pl.col("cs_onset_dttm")
        )
        counts = (
            joined.select("subject_id", "times", "cs_onset_dttm")
            .explode("times")
            .group_by("subject_id")
            .agg(last_valid=at_or_before.sum())
        )

        return joined.join(counts, on="subject_id", how="inner").filter(
            pl.col("last_valid") >= min_tokens_past
        )

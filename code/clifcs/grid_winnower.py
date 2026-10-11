#!/usr/bin/env python3

"""
winnowing a timeline at a grid of timepoints

`OnsetWinnower` cuts each patient's stream once, at their cardiogenic shock
onset. This cuts the same stream repeatedly, at onset plus a fixed set of
offsets, so one patient yields a sequence of representations rather than one.
That sequence is what makes a trajectory.

The cut points are per-patient wall-clock times, which cocoa's own Winnower
cannot express (it offers a global elapsed duration or a global first-occurrence
token), so `run_thresholding` is overridden exactly as in the onset case. The
past/future split, the outcome flags and the output schema stay cocoa's, so
`cotorra extract` consumes these files unchanged.
"""

from __future__ import annotations

import pathlib

import polars as pl
from cocoa.winnower import Winnower


class GridWinnower(Winnower):
    """cuts each timeline at `cs_onset_dttm + offset_hours`"""

    def __init__(
        self,
        winnowing_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        onset_parquet: pathlib.Path | str = None,
        offset_hours: float = 0.0,
        min_tokens_past: int = 8,
        **kwargs,
    ):
        super().__init__(
            winnowing_cfg=winnowing_cfg,
            processed_data_home=processed_data_home,
            **kwargs,
        )
        self.offset_hours = float(offset_hours)
        self.min_tokens_past = int(min_tokens_past)
        if onset_parquet is None:
            raise ValueError("onset_parquet is required")
        path = pathlib.Path(onset_parquet).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"onset table not found: {path}")

        self.cutoffs = (
            pl.scan_parquet(path)
            .select(
                subject_id=pl.col("hospitalization_id").cast(pl.String),
                cutoff_dttm=pl.col("cs_onset_dttm").cast(pl.Datetime("us"))
                + pl.duration(seconds=int(self.offset_hours * 3600)),
            )
            .unique(subset="subject_id")
        )

    def run_thresholding(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        keep the tokens at or before this grid point

        A patient who has died or been discharged before the grid point simply
        has no further tokens, so their prefix stops growing and the trajectory
        ends there rather than being extrapolated. Patients whose stream is
        shorter than `min_tokens_past` at this point are dropped: a
        representation taken from a handful of tokens reflects padding more
        than the patient.
        """
        joined = df.join(self.cutoffs, on="subject_id", how="inner")
        counts = (
            joined.select("subject_id", "times", "cutoff_dttm")
            .explode("times")
            .group_by("subject_id")
            .agg(last_valid=(pl.col("times") <= pl.col("cutoff_dttm")).sum())
        )
        return joined.join(counts, on="subject_id", how="inner").filter(
            pl.col("last_valid") >= self.min_tokens_past
        )

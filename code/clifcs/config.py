#!/usr/bin/env python3

"""
path resolution, config loading, and logging shared by all numbered scripts
"""

from __future__ import annotations

import dataclasses
import glob
import logging
import pathlib
import sys

from omegaconf import DictConfig, OmegaConf

CODE_HOME = pathlib.Path(__file__).resolve().parent.parent
CONFIG_HOME = CODE_HOME / "config"

# clifpy's progress messages contain emoji, which raise UnicodeEncodeError on a
# Windows console defaulting to cp1252 and abort the run. Force UTF-8 on the
# standard streams at import so a log line can never kill the pipeline.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

#: embedding anchors; each produces its own set of representations
ANCHORS: tuple[str, ...] = ("24h", "onset")

#: Scatterplots put every site on screen at once, so the all-pairs colour gate
#: applies, and the reference palette validates only its first three slots under
#: that gate. Three is therefore a hard cap on sites plotted in one panel.
MAX_PLOTTED_SITES = 3


@dataclasses.dataclass(frozen=True)
class Paths:
    """resolved input and output locations"""

    site_data: dict[str, pathlib.Path]
    model_home: pathlib.Path
    tokenizer_yaml: pathlib.Path
    output_home: pathlib.Path
    scratch_home: pathlib.Path

    @property
    def sites(self) -> tuple[str, ...]:
        """
        the sites this machine holds data for, in paths.yaml order

        Not a fixed constant: each institution runs the pipeline over only the
        data it has, and the pooled analysis in 03 works from whatever
        per-site embedding files are present rather than from this list.
        """
        return tuple(self.site_data)

    # ---------------------------------------------------------------- outputs
    @property
    def cohort_home(self) -> pathlib.Path:
        """cohort tables and the filter-count ledger"""
        return self.output_home / "cohort"

    @property
    def patient_level_home(self) -> pathlib.Path:
        """one row per patient: embedding location plus covariates"""
        return self.output_home / "patient_level_data"

    @property
    def figure_home(self) -> pathlib.Path:
        return self.output_home / "figures"

    @property
    def table_home(self) -> pathlib.Path:
        return self.output_home / "tables"

    @property
    def log_home(self) -> pathlib.Path:
        return self.output_home / "logs"

    # --------------------------------------------------------------- scratch
    def raw_subset(self, site: str) -> pathlib.Path:
        """cohort-restricted copy of a site's CLIF tables, plus derived tables"""
        return self.scratch_home / "data-raw" / f"{site.lower()}-cs"

    def processed(self, site: str) -> pathlib.Path:
        """collated + tokenized timelines for a site"""
        return self.scratch_home / "processed" / f"{site.lower()}-cs"

    def anchored(self, site: str, anchor: str) -> pathlib.Path:
        """per-anchor view of a site's timelines, ready for `cotorra extract`"""
        return self.processed(site) / f"anchor-{anchor}"

    def mkdirs(self) -> None:
        for p in (
            self.cohort_home,
            self.patient_level_home,
            self.figure_home,
            self.table_home,
            self.log_home,
            self.scratch_home,
        ):
            p.mkdir(parents=True, exist_ok=True)


def load_paths(config: pathlib.Path | str | None = None) -> Paths:
    """resolve paths.yaml into absolute, existence-checked locations"""
    path = pathlib.Path(config or CONFIG_HOME / "paths.yaml")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. It is machine-specific and deliberately not in "
            f"git; copy {CONFIG_HOME / 'paths.example.yaml'} to it and fill in "
            "the locations for this machine."
        )
    cfg = OmegaConf.load(path)

    site_data = {}
    for site in cfg.sites:
        d = pathlib.Path(cfg.sites[site].data_dir).expanduser().resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"{site} data_dir not found: {d}")
        site_data[site] = d
    if not site_data:
        raise ValueError(f"no sites configured in {path}")

    # the model directory name contains a non-ASCII character; glob for it so the
    # literal byte sequence never has to be reproduced in a config file
    matches = sorted(glob.glob(str(cfg.model.home_glob)))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"model.home_glob matched {len(matches)} directories "
            f"(expected exactly 1): {cfg.model.home_glob} -> {matches}"
        )
    model_home = pathlib.Path(matches[0]).resolve()
    if not (model_home / "config.json").is_file():
        raise FileNotFoundError(f"no config.json under {model_home}")

    tokenizer_yaml = pathlib.Path(cfg.model.tokenizer_yaml).expanduser().resolve()
    if not tokenizer_yaml.is_file():
        raise FileNotFoundError(
            "the training-time cocoa tokenizer is required to make the model's "
            f"token embeddings meaningful, but was not found: {tokenizer_yaml}"
        )

    return Paths(
        site_data=site_data,
        model_home=model_home,
        tokenizer_yaml=tokenizer_yaml,
        output_home=pathlib.Path(cfg.output_home).expanduser().resolve(),
        scratch_home=pathlib.Path(cfg.scratch_home).expanduser().resolve(),
    )


def load_cohort_cfg(config: pathlib.Path | str | None = None) -> DictConfig:
    """load the cardiogenic-shock cohort definition"""
    return OmegaConf.load(pathlib.Path(config or CONFIG_HOME / "cohort.yaml"))


def get_logger(name: str, log_file: pathlib.Path | None = None) -> logging.Logger:
    """stdout logger, optionally mirrored to `log_file`"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    # clifpy installs a rich handler on the root logger; without this every
    # record would be printed twice, once plain and once rich-wrapped
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger

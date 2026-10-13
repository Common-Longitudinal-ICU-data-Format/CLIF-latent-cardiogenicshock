#!/usr/bin/env python3

"""
path resolution, config loading, and logging shared by all numbered scripts
"""

from __future__ import annotations

import dataclasses
import glob
import logging
import os
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


def _redirect_hf_cache() -> None:
    """
    send HuggingFace `datasets` scratch into scratch_home, not ~/.cache

    cotorra loads the winnowed token streams through
    `datasets.load_dataset("parquet", ...)`, which caches the Arrow conversion
    and every `.map()` output under HF_DATASETS_CACHE -- by default
    ~/.cache/huggingface. Nothing is downloaded; the files there are *our*
    tokenized patient data.

    Two reasons that default is wrong here, and both bit:

    * It is unbounded. The transform cotorra maps is a closure, so `datasets`
      cannot fingerprint it, warns that it "will consider it to be different
      from the previous calls and recompute everything", and writes a fresh
      build every run. 02 makes four per run and 09 twenty-six. 172 builds
      accumulated over four days and reached 119 GB, which filled the system
      disk and killed 09 at grid point 23 of 26.
    * It puts patient-derived data somewhere this project does not document or
      manage. The README promises that large regenerable intermediates live in
      scratch_home, deliberately off the Box-synced tree; a cache in the home
      directory is a third location holding the same class of data, and it is
      the one nobody thinks to clear or exclude from a backup.

    Set at import of this module, which every numbered script reaches before it
    imports cotorra or datasets -- `datasets.config` reads these variables once,
    at its own import, so setting them later would have no effect. An
    HF_DATASETS_CACHE already in the environment is left alone.
    """
    if os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_HOME"):
        return
    config = CONFIG_HOME / "paths.yaml"
    if not config.is_file():
        return
    try:
        scratch = OmegaConf.load(config).get("scratch_home")
    except Exception:  # a malformed paths.yaml is load_paths' problem to report
        return
    if not scratch:
        return
    cache = pathlib.Path(str(scratch)).expanduser() / "hf-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")


_redirect_hf_cache()

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
    model_key: str
    model_home: pathlib.Path
    available_models: tuple[str, ...]
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
        return self.output_home / "patient_level_data" / self.model_key

    @property
    def figure_home(self) -> pathlib.Path:
        return self.output_home / "figures" / self.model_key

    @property
    def table_home(self) -> pathlib.Path:
        return self.output_home / "tables" / self.model_key

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

    @property
    def id_salt_file(self) -> pathlib.Path:
        """
        salt for pseudonymizing record numbers on export

        Deliberately above the per-model directories: a patient must get the
        same pseudonym whichever model embedded them, or the same person
        cannot be matched across models in a pooled analysis.
        """
        return self.output_home / "patient_level_data" / ".id_salt"

    def features_home(self, site: str, anchor: str) -> pathlib.Path:
        """
        where `cotorra extract` writes its raw feature shards

        Model-scoped, because cotorra names those files after
        `model_home.name` -- which is "mdl-cotorra" for every GEM checkpoint.
        Left in the shared anchor directory they would overwrite each other.
        The tokenized inputs stay shared: the models read identical data.
        """
        return self.anchored(site, anchor) / self.model_key

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


def load_paths(
    config: pathlib.Path | str | None = None, model: str | None = None
) -> Paths:
    """
    resolve paths.yaml into absolute, existence-checked locations

    `model` selects which checkpoint's outputs this run reads and writes;
    it defaults to `default_model` in the config.
    """
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

    available = tuple(cfg.models)
    if not available:
        raise ValueError(f"no models configured in {path}")
    key = model or cfg.get("default_model") or available[0]
    if key not in available:
        raise KeyError(f"model {key!r} not in paths.yaml; have {available}")

    # the weights directory names contain a non-ASCII character, so they are
    # globbed rather than written out literally
    pattern = str(cfg.models[key].home_glob)
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"models.{key}.home_glob matched {len(matches)} directories "
            f"(expected exactly 1): {pattern} -> {matches}"
        )
    model_home = pathlib.Path(matches[0]).resolve()
    if not (model_home / "config.json").is_file():
        raise FileNotFoundError(f"no config.json under {model_home}")

    tokenizer_yaml = pathlib.Path(cfg.tokenizer_yaml).expanduser().resolve()
    if not tokenizer_yaml.is_file():
        raise FileNotFoundError(
            "the training-time cocoa tokenizer is required to make the model's "
            f"token embeddings meaningful, but was not found: {tokenizer_yaml}"
        )

    return Paths(
        site_data=site_data,
        model_key=key,
        model_home=model_home,
        available_models=available,
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

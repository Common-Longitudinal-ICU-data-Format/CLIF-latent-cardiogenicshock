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


#: the two roles a machine runs this pipeline in.
#:
#: `site` holds raw CLIF data and may read it. `pooled` holds only the bundles
#: contributing sites sent, and may not -- the whole point of the federated
#: split is that the coordinating analysis cannot depend on record-level data,
#: and the way to keep that true is to make the attempt fail rather than to
#: document that it should not happen.
ROLES: tuple[str, ...] = ("site", "pooled")


class RoleViolation(Exception):
    """a pooled-role run tried to read raw patient records"""


@dataclasses.dataclass(frozen=True)
class Paths:
    """resolved input and output locations"""

    raw_site_data: dict[str, pathlib.Path]
    model_key: str
    model_home: pathlib.Path
    available_models: tuple[str, ...]
    tokenizer_yaml: pathlib.Path
    output_home: pathlib.Path
    scratch_home: pathlib.Path
    role: str = "site"

    @property
    def site_data(self) -> dict[str, pathlib.Path]:
        """
        where this machine's raw CLIF tables live -- site role only

        Raises under the pooled role rather than returning an empty mapping.
        An empty mapping would make a pooled script that still reads raw data
        silently produce nothing for every site, which looks like "this site
        contributed no patients" and is the single most plausible way for the
        federated split to be quietly wrong.
        """
        if self.role == "pooled":
            raise RoleViolation(
                "this run is in the `pooled` role and must not read raw CLIF "
                "data. The coordinating site holds only the bundles that "
                "contributing sites sent; anything derived from records has to "
                "be computed site-side and exported under the contract in "
                "clifcs/export/contract.py.\n"
                "If this is actually a site-role analysis, load paths with "
                "load_paths(role='site')."
            )
        return self.raw_site_data

    @property
    def received_home(self) -> pathlib.Path:
        """
        the coordinating site's inbox

        Received bundles are dropped in verbatim, keeping the directory
        names the sending site wrote -- `upload_to_box_<SITE>/`, one per
        site whichever tier it sent. Whether a bundle carries patient-level
        data is visible from a `patient/` subdirectory inside it, and is
        recorded in its manifest as `shares_patient_level`.
        """
        return self.output_home / "received"

    @property
    def sites(self) -> tuple[str, ...]:
        """
        the sites this machine holds data for, in paths.yaml order

        Not a fixed constant: each institution runs the pipeline over only the
        data it has, and the pooled analysis in 03 works from whatever
        per-site embedding files are present rather than from this list.

        Reads `raw_site_data` directly rather than going through `site_data`,
        so it does not trip the pooled-role guard: knowing which sites are
        *configured* is not the same as reading their records, and a pooled run
        legitimately wants the former. What a pooled run must not do is take
        this list as the set of contributing sites -- that comes from the
        bundles actually present, via `clifcs.export.bundles.discover`.
        """
        return tuple(self.raw_site_data)

    # ---------------------------------------------------------------- outputs
    @property
    def cohort_home(self) -> pathlib.Path:
        """cohort tables and the filter-count ledger"""
        return self.output_home / "cohort"

    # ------------------------------------------------- the three output tiers
    #
    # Named so that what may leave the institution is obvious from the
    # directory, not from reading the code:
    #
    #   patient_level_data_<SITE>   LOCAL ONLY. Embeddings and covariates.
    #                               Never shared.
    #   upload_to_box_<SITE>        TIER 1. Summary tables only -- the classic
    #                               CLIF federated contribution.
    #                               TIER 2 adds patient/ to this SAME
    #                               directory, where the consortium permits
    #                               it, so a site uploads one thing either way.
    # Each is per site rather than one pooled directory, because a site sends
    # its own and nothing else, and because the person doing the sending should
    # be able to see at a glance which directory is which.

    def patient_level_home_for(self, site: str) -> pathlib.Path:
        """
        one row per patient: embedding plus covariates -- LOCAL ONLY

        Model-scoped inside the site directory, since the same cohort can be
        embedded by several checkpoints without their outputs colliding.
        """
        return self.output_home / f"patient_level_data_{site}" / self.model_key

    def upload_home(self, site: str) -> pathlib.Path:
        """tier 1: the summary bundle a site uploads"""
        return self.output_home / f"upload_to_box_{site}"

    @property
    def patient_level_home(self) -> pathlib.Path:
        """
        the legacy single-directory location

        Kept because the pooled scripts read whatever the coordinating site
        assembled locally, which is not per-contributor. Site-side code should
        use `patient_level_home_for(site)`.
        """
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


def site_embedding_files(patient_level_home, anchor: str) -> list:
    """
    the per-site embedding files for one anchor, and nothing else

    `embeddings_anchor-<anchor>_<SITE>.parquet` is how a site contributes, and
    every pooling script globs for it. The glob is wider than the contract:
    anything dropped in that directory whose name starts the same way gets
    pooled as though it were another hospital.

    That is not hypothetical. 05's corrected embeddings were briefly written as
    `embeddings_anchor-onset_corrected-leace.parquet`, and the next run of 03
    pooled the raw file with both corrected ones and reported 102,426 patients
    against a 34,142-patient cohort -- with figures that looked entirely
    reasonable, because tripling a cohort with affine images of itself does not
    distort a projection in any obvious way. The corrected files now live in a
    subdirectory; this function is the second line of defence, so the next
    thing to land there fails loudly instead.

    A share-safe copy is a redaction of a file that is usually already in the
    list, and reading both double-counts that site -- so it is read only when
    its unredacted twin is absent, which is the case where it came from a
    collaborating site rather than from this machine. 03 already did this; the
    other four scripts dropped share-safe files unconditionally and so would
    have ignored a remote site that could only ship the redacted form.
    """
    import pathlib

    found = sorted(
        pathlib.Path(patient_level_home).glob(
            f"embeddings_anchor-{anchor}_*.parquet"
        )
    )
    prefix = f"embeddings_anchor-{anchor}_"
    rejected = []
    for path in found:
        tag = path.stem.split(prefix, 1)[-1].replace("_share-safe", "")
        # a site tag is a bare name; a hyphen means a role prefix, not a site
        if "-" in tag:
            rejected.append(path.name)
    if rejected:
        raise RuntimeError(
            "these files match the per-site embedding pattern but are not site "
            "files: " + ", ".join(rejected) + ". Move them out of "
            f"{patient_level_home} -- pooling them would silently inflate the "
            "cohort, which is how a 34,142-patient cohort once came back as "
            "102,426."
        )
    unredacted = {p.name for p in found if "_share-safe" not in p.name}
    return [
        p
        for p in found
        if "_share-safe" not in p.name
        or p.name.replace("_share-safe", "") not in unredacted
    ]


def load_paths(
    config: pathlib.Path | str | None = None,
    model: str | None = None,
    role: str | None = None,
) -> Paths:
    """
    resolve paths.yaml into absolute, existence-checked locations

    `model` selects which checkpoint's outputs this run reads and writes;
    it defaults to `default_model` in the config.

    `role` is `site` or `pooled`. Under `pooled`, `paths.site_data` raises --
    see `Paths.site_data`. It defaults to `role` in paths.yaml, then to `site`,
    so an existing config keeps working unchanged.

    A pooled-role run still needs paths.yaml for the model and tokenizer (it
    has to check that the bundles it received were produced under the weights
    it holds), but its `sites` block may be empty.
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
    if not site_data and (role or cfg.get("role")) != "pooled":
        raise ValueError(
            f"no sites configured in {path}. A coordinating machine that holds "
            "no raw data should set `role: pooled` there, which makes an empty "
            "`sites` block legitimate and raw-data access an error."
        )

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

    resolved_role = role or cfg.get("role") or "site"
    if resolved_role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, not {resolved_role!r}")

    return Paths(
        raw_site_data=site_data,
        role=resolved_role,
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

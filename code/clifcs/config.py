#!/usr/bin/env python3

"""
path resolution, config loading, and logging shared by all numbered scripts
"""

from __future__ import annotations

import dataclasses
import functools
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


def _is_placeholder(value) -> bool:
    """an unedited `/path/to/...` value from paths.example.yaml"""
    text = str(value) if value is not None else ""
    return text.startswith("/path/to") or text == "/path"


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
    # a template placeholder counts as unset, as it does in load_paths
    if not scratch or _is_placeholder(scratch):
        return
    # PER PROCESS. Two runs sharing one cache directory destroy each other:
    # both clear it between grid points, so one deletes an arrow file the other
    # is still reading. That killed a septic grid extract twice -- first as a
    # PermissionError in the process doing the deleting, and then, after the
    # deleter was made not to raise, as a FileNotFoundError in the process doing
    # the reading. Making the deletion forgiving moved the failure rather than
    # removing it.
    #
    # Per-process costs nothing because this cache is never read back: cotorra
    # maps a closure, `datasets` cannot fingerprint it, and every lookup misses
    # by construction. There is no reuse to lose.
    #
    # A crashed run leaves its directory behind, bounded at roughly one grid
    # point. Nothing sweeps those, deliberately: deleting another process's
    # directory is the bug this is fixing.
    cache = (
        pathlib.Path(str(scratch)).expanduser() / "hf-cache" / f"pid-{os.getpid()}"
    )
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
#: used for output subdirectories and the manifest when no `models:` block
#: is configured. Matches `default_model` in paths.example.yaml so that a site
#: which omits the block and one which fills it in agree.
DEFAULT_MODEL_KEY = "GEM-all"

ROLES: tuple[str, ...] = ("site", "pooled")


class RoleViolation(Exception):
    """a pooled-role run tried to read raw patient records"""


@dataclasses.dataclass(frozen=True)
class Paths:
    """resolved input and output locations"""

    raw_site_data: dict[str, pathlib.Path]
    model_key: str
    model_home_glob: str
    available_models: tuple[str, ...]
    tokenizer_yaml_path: pathlib.Path | None
    output_home: pathlib.Path
    scratch_home: pathlib.Path
    role: str = "site"

    @functools.cached_property
    def model_home(self) -> pathlib.Path:
        """
        the resolved checkpoint directory -- resolved on access, not on load

        Tier 1 (`01_site_tables.py`) never loads the GEM weights. It reads the
        tokenizer, which is a different file. Resolving the glob eagerly in
        `load_paths` therefore made the weights a hard prerequisite for a run
        that does not use them, and a site asked only for the aggregate tables
        had to fetch a checkpoint tree to satisfy a check.

        So the glob is resolved here instead, and raises only once something
        actually reaches for the weights -- which is 02, 08, and the tier-2
        manifest. A tier-1 site needs `tokenizer.yaml` and nothing else.

        Cached, because 09 and `embedding.py` read `model_home.name` inside
        per-split loops and the weights usually sit on a networked filesystem.
        A raised FileNotFoundError is not cached, so a site that fixes its glob
        mid-session does not have to reload paths.
        """
        if not self.model_home_glob:
            raise FileNotFoundError(
                "no `models:` block is configured in paths.yaml, so there is no "
                "checkpoint to resolve. 01_site_tables.py does not need one."
            )
        matches = sorted(glob.glob(self.model_home_glob))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"models.{self.model_key}.home_glob matched {len(matches)} "
                f"directories (expected exactly 1): {self.model_home_glob} -> "
                f"{matches}. "
                "The GEM weights are needed for tier 2 (02_site_embeddings.py) "
                "and 08_token_space.py only. Tier 1 reads the tokenizer, so if "
                "you are running 01_site_tables.py this path need not resolve."
            )
        home = pathlib.Path(matches[0]).resolve()
        if not (home / "config.json").is_file():
            raise FileNotFoundError(f"no config.json under {home}")
        return home

    @property
    def model_home_or_none(self) -> pathlib.Path | None:
        """`model_home` if the weights are on this machine, else None"""
        try:
            return self.model_home
        except FileNotFoundError:
            return None

    @property
    def tokenizer_yaml(self) -> pathlib.Path:
        """
        the training-time cocoa tokenizer -- checked on access, not on load

        Raises when it is not on this machine, which is deliberate for the
        things that genuinely cannot proceed without it: tokenizing, the
        token-prevalence tables, and anything that reads the model's embedding
        table by token id.

        It is NOT required to run 01 as a whole. The cohort, the phenotype
        breakdowns, the baseline tables and the entire trial and guideline
        analysis are computed from CLIF and the published literature and never
        see a token, so a site with no tokenizer skips the token stages and
        sends everything else. Use `has_tokenizer` to branch rather than
        catching this.
        """
        if self.tokenizer_yaml_path is None:
            raise FileNotFoundError(
                "no `tokenizer_yaml` is configured in paths.yaml. It is needed "
                "to tokenize and to compare token prevalence across sites, and "
                "must be the tokenizer the model was TRAINED with. The rest of "
                "01_site_tables.py does not use it."
            )
        if not self.tokenizer_yaml_path.is_file():
            raise FileNotFoundError(
                "the training-time cocoa tokenizer is required to make the "
                "model's token embeddings meaningful, but was not found: "
                f"{self.tokenizer_yaml_path}. The cohort, phenotype, baseline "
                "and trial tables in 01_site_tables.py do not need it; the "
                "token stages and all of tier 2 do."
            )
        return self.tokenizer_yaml_path

    @property
    def tokenizer_yaml_or_none(self) -> pathlib.Path | None:
        """`tokenizer_yaml` if it is on this machine, else None"""
        try:
            return self.tokenizer_yaml
        except FileNotFoundError:
            return None

    @property
    def has_tokenizer(self) -> bool:
        """whether the token stages can run at all"""
        return self.tokenizer_yaml_or_none is not None

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

    TWO LAYOUTS ARE SEARCHED, because the pipeline writes one and the pooling
    scripts were written against the other:

      patient_level_data/<model>/embeddings_anchor-<a>_<SITE>.parquet
          the flat pool. How a received bundle is staged, and what every
          pooling script globbed for.

      patient_level_data_<SITE>/<model>/embeddings_anchor-<a>.parquet
          where 02_site_embeddings.py actually writes, one directory per site
          so that what may leave the institution is obvious from the name.

    Searching only the first meant that on a machine which had just run 02 --
    a contributing site that also pools, which is the normal case for the
    coordinating institution -- 03 found nothing, warned, and returned an empty
    frame. An empty frame reads exactly like "no site has contributed yet".
    """
    import pathlib

    home = pathlib.Path(patient_level_home)
    # `output/patient_level_data/<model>` -> `output/` and `<model>`
    output_home, model_key = home.parent.parent, home.name
    per_site = sorted(
        output_home.glob(
            f"patient_level_data_*/{model_key}/embeddings_anchor-{anchor}.parquet"
        )
    )

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
    flat = [
        p
        for p in found
        if "_share-safe" not in p.name
        or p.name.replace("_share-safe", "") not in unredacted
    ]

    # A site present in both layouts is the same site twice. The flat copy wins:
    # it is how a received bundle is staged, so if both exist the flat one is
    # the deliberate act and the per-site one is this machine's own working
    # copy of the same patients.
    flat_sites = {
        p.stem.split(f"embeddings_anchor-{anchor}_", 1)[-1].replace("_share-safe", "")
        for p in flat
    }
    return flat + [
        p
        for p in per_site
        # patient_level_data_<SITE> -> <SITE>
        if p.parent.parent.name.split("patient_level_data_", 1)[-1] not in flat_sites
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

    The checkpoint glob is recorded but not resolved: `paths.model_home`
    resolves it on access and raises there. Tier 1 never touches the weights,
    so a site running only `01_site_tables.py` needs the tokenizer alone.
    """
    path = pathlib.Path(config or CONFIG_HOME / "paths.yaml")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. It is machine-specific and deliberately not in "
            f"git; copy {CONFIG_HOME / 'paths.example.yaml'} to it and fill in "
            "the locations for this machine."
        )
    cfg = OmegaConf.load(path)

    # ------------------------------------------------------------------
    # Unedited placeholders, caught by name.
    #
    # paths.example.yaml ships `/path/to/...` in every location. Left in, the
    # first thing that touches one fails somewhere deep in pathlib or glob and
    # the error is whatever the filesystem said about `/path` -- at Rush that
    # was `[Errno 13] Permission denied: '/path'`, which names neither the file
    # to edit nor the key that is wrong, and sends the reader looking for a
    # permissions problem they do not have.
    #
    # Checked before anything is resolved, so the message is about the config
    # rather than about the filesystem's opinion of a path nobody meant.
    # ONLY `sites.<name>.data_dir` is fatal, because it is the only key with no
    # sane fallback -- there is nothing to analyse without it. Everything else
    # either degrades or defaults:
    #
    #   models.<key>.home_glob  tier 2 only; checked by `Paths.model_home`
    #   tokenizer_yaml          token stages only; checked by `Paths.tokenizer_yaml`
    #   scratch_home            defaults to ./scratch
    #   output_home             defaults to ./output
    #
    # Checking any of those here would re-impose a prerequisite that the lazy
    # resolution and the defaults exist to remove, and would stop a site that
    # could have contributed almost everything from contributing anything.
    placeholders = [
        f"    {key}: {value}"
        for key, value in (
            (f"sites.{site}.data_dir", cfg.sites[site].get("data_dir"))
            for site in (cfg.get("sites") or {})
        )
        if _is_placeholder(value)
    ]
    if placeholders:
        raise ValueError(
            f"{path} still holds the example placeholders:"
            + chr(10)
            + chr(10).join(placeholders)
            + chr(10) * 2
            + "Point it at this machine's CLIF-2.1 directory. That is the "
            "only key 01_site_tables.py cannot do without: `scratch_home` and "
            "`output_home` default to ./scratch and ./output, and "
            "`tokenizer_yaml` and `models.<key>.home_glob` may be left pointing "
            "at nothing -- without the tokenizer 01 skips the token stages and "
            "exports everything else."
        )

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

    # The `models:` block is optional. 01 never resolves it, and a paths.yaml
    # that omits it entirely is the minimal working configuration.
    #
    # `model_key` still has to be SOMETHING, because it names output
    # subdirectories and is recorded in the manifest, where bundles are checked
    # against each other for agreement. It defaults to DEFAULT_MODEL_KEY rather
    # than to a placeholder so that a site which omits the block and a site
    # which configures it land on the same key and still pool.
    configured_models = cfg.get("models") or {}
    available = tuple(configured_models)
    key = model or cfg.get("default_model") or (
        available[0] if available else DEFAULT_MODEL_KEY
    )
    if available and key not in available:
        raise KeyError(
            f"model {key!r} is not in {path}; it has: {', '.join(available)}"
        )

    # the weights directory names contain a non-ASCII character, so they are
    # globbed rather than written out literally. The glob is NOT resolved here
    # -- see `Paths.model_home`, which resolves it on access so that a tier-1
    # site, which never loads the weights, is not blocked by their absence.
    pattern = (
        str(configured_models[key].home_glob)
        if key in configured_models and configured_models[key].get("home_glob")
        else ""
    )

    # Recorded, not checked -- see `Paths.tokenizer_yaml`, which checks it on
    # access. A missing tokenizer stops the token stages and all of tier 2; it
    # does not stop the cohort, the phenotype breakdowns, the baseline tables or
    # the trial and guideline analysis, none of which read a token. Requiring it
    # here made a site that could have contributed all of those contribute
    # nothing. May also be absent from paths.yaml entirely.
    configured_tokenizer = cfg.get("tokenizer_yaml")
    tokenizer_yaml = (
        pathlib.Path(str(configured_tokenizer)).expanduser().resolve()
        if configured_tokenizer
        else None
    )

    resolved_role = role or cfg.get("role") or "site"
    if resolved_role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, not {resolved_role!r}")

    # `output_home` and `scratch_home` default to the working directory, so
    # neither has to be set to get a run out of the box. An unedited
    # placeholder counts as unset rather than as an error: falling back to a
    # working default beats refusing to start over a path nobody chose.
    #
    # `./scratch` is a DEFAULT, not a recommendation. Intermediates are
    # cohort-restricted CLIF, token streams and feature shards -- hundreds of GB
    # for a large cohort, fully regenerable, and ruinous on a synced drive
    # (Box, Dropbox, OneDrive will both replicate them and hold locks on them).
    # Point `scratch_home` at real scratch before running a big cohort; it is
    # logged on every run so the location is never a surprise.
    def _home(key: str, default: str) -> pathlib.Path:
        value = cfg.get(key)
        if not value or _is_placeholder(value):
            value = default
        return pathlib.Path(str(value)).expanduser().resolve()

    return Paths(
        raw_site_data=site_data,
        role=resolved_role,
        model_key=key,
        model_home_glob=pattern,
        available_models=available,
        tokenizer_yaml_path=tokenizer_yaml,
        output_home=_home("output_home", "./output"),
        scratch_home=_home("scratch_home", "./scratch"),
    )


def load_cohort_cfg(config: pathlib.Path | str | None = None) -> DictConfig:
    """load the cardiogenic-shock cohort definition"""
    return OmegaConf.load(pathlib.Path(config or CONFIG_HOME / "cohort.yaml"))


def load_dynamics_cfg(config=None):
    """the action-response dynamics config, thresholds and bins included"""
    return OmegaConf.load(pathlib.Path(config or CONFIG_HOME / "dynamics.yaml"))


def _per_subset(value, subset: str | None) -> int:
    """a config entry that may be one number or one number per cohort"""
    if isinstance(value, (int, float)):
        return int(value)
    if subset is not None and subset in value:
        return int(value[subset])
    # No cohort given -- take the largest, so a caller that forgot to pass
    # one over-clusters rather than silently under-clustering a cohort that
    # needed more handbooks than the smallest entry allows.
    return int(max(int(v) for v in value.values()))


def handbook_defaults(config=None, subset: str | None = None) -> dict:
    """
    the handbook-clustering settings, as argparse defaults

    Four scripts rebuild the same clustering -- 40 discovers it, 41 probes it,
    42 draws it, 43 writes its logos -- and for a while each carried its own
    copy of k and the sample size in its own argparse block. They had drifted
    by a factor of two, which does not raise an error anywhere: it produces a
    handbook table and a handbook figure that describe different clusterings of
    the same patients and agree on the word "handbook 0". Returning the block
    as a dict of argparse defaults means a flag still overrides it, but the
    unflagged run is the same run in all four.

    `subset` matters for k and the sample size, which are per cohort: see the
    note in dynamics.yaml for why. Scripts therefore have to resolve these
    AFTER argparse knows which cohort is being run, which is what
    `apply_handbook_defaults` is for.
    """
    block = load_dynamics_cfg(config).handbooks
    return {
        "handbooks": _per_subset(block.count, subset),
        "sample": _per_subset(block.sample, subset),
        "min_group": int(block.min_group),
        "min_fit": float(block.min_fit),
        "min_fit_omspell": float(block.min_fit_omspell),
        "distance": str(block.distance),
        "expcost": float(block.expcost),
        "cap_hours": int(block.cap_hours),
        "gap_open": float(block.gap_open),
        "gap_extend": float(block.gap_extend),
        "seed": int(block.seed),
        "logo_min_patients": int(block.logo_min_patients),
        "divergence_permutations": int(block.divergence_permutations),
        "divergence_percentile": float(block.divergence_percentile),
        "divergence_run": int(block.divergence_run),
        "divergence_min_shared": int(block.divergence_min_shared),
        "divergence_local": bool(block.divergence_local),
        "divergence_min_runs": int(block.divergence_min_runs),
    }


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


#: Handbook settings that depend on the cohort and so cannot be baked into an
#: argparse default at parser-construction time.
HANDBOOK_ARGS = (
    "handbooks", "sample", "min_group", "min_fit", "cap_hours", "gap_open",
    "gap_extend", "seed", "distance", "expcost",
)


def apply_handbook_defaults(args, config=None, extra=()):
    """
    fill any unset handbook argument from the config, now that --subset is known

    Every handbook flag is declared with `default=None` so that "the user did
    not pass this" is distinguishable from "the user passed the value that
    happens to be the default". Without that distinction a per-cohort default
    cannot exist: the parser would already have committed to one number before
    anything knew which cohort was being run.
    """
    resolved = handbook_defaults(config, getattr(args, "subset", None))
    # `distance` first, and whether --min-fit was passed read BEFORE anything is
    # filled in: the fit floor depends on which measure is in use, and once the
    # default has been written in there is no way to tell it from a flag.
    wants_default_fit = getattr(args, "min_fit", None) is None
    for name in tuple(HANDBOOK_ARGS) + tuple(extra):
        if getattr(args, name, None) is None:
            setattr(args, name, resolved[name])
    if wants_default_fit and getattr(args, "distance", None) == "omspell":
        args.min_fit = resolved["min_fit_omspell"]
    return args

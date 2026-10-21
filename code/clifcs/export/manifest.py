#!/usr/bin/env python3

"""
what a bundle says about itself

Every export carries a manifest, and the coordinating site refuses a bundle
without one. It exists to make three specific failures loud instead of silent.

**Sites running different code.** The cohort definition lives in this
repository, so a site on last month's commit enrols a different population and
nothing about the parquet it sends would reveal that. The manifest carries the
git SHA and whether the tree was dirty.

**Sites running a different model or tokenizer.** The embedding is only
comparable across sites if the weights and the token->id mapping are identical.
`02` already asserts the tokenizer matches the model locally; the manifest
carries a hash of both so the coordinator can assert it matches *across sites*,
which is the question that actually matters in a federated run.

**A regenerated pseudonym salt.** This one bit the design rather than a person:
the salt is created on demand if absent and lives under gitignored `output/`, so
a site that clears its output directory and re-exports produces an entirely new
set of `row_key`s for the same patients. The coordinator cannot tell the resend
from a new cohort, and would double-count every patient if both files were
present. The manifest carries a *fingerprint* of the salt -- a hash of the salt,
which reveals nothing about it -- so a change is detectable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import subprocess

from clifcs.export.contract import SCHEMA_VERSION

MANIFEST_NAME = "manifest.json"


def _git(*args: str) -> str | None:
    """run a git command in this repository, or None if it is unavailable"""
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=pathlib.Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_version() -> dict:
    """
    the commit this export was produced by, and whether the tree was clean

    A dirty tree is recorded rather than refused: a site may legitimately carry
    a local patch for its own data quirks, and the coordinator should be able to
    see that it did, not be prevented from receiving the data.
    """
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_sha": sha,
        "git_dirty": bool(status) if status is not None else None,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def file_digest(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    """sha256 of a file, read in chunks so a 300 MB checkpoint is fine"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def salt_fingerprint(salt: str) -> str:
    """
    a stable identifier for a salt that does not reveal it

    Double-hashed with a fixed domain separator: the `row_key`s in the bundle
    are `sha256(salt:column:value)`, so publishing `sha256(salt)` alongside them
    would hand an attacker the one input they are missing for a dictionary
    attack over record numbers. Hashing a distinct string keeps the fingerprint
    useless for that while still changing whenever the salt does.
    """
    return hashlib.sha256(f"clifcs-salt-fingerprint:{salt}".encode()).hexdigest()[:16]


def build(
    *,
    site: str,
    salt: str,
    model_key: str,
    model_home: pathlib.Path | None,
    tokenizer_yaml: pathlib.Path | None,
    anchors: tuple[str, ...],
    row_counts: dict[str, int],
    min_cell: int,
    embedding_dim: int,
    shares_patient_level: bool,
    notes: str = "",
) -> dict:
    """assemble the manifest for one site's bundle"""
    return {
        "schema_version": SCHEMA_VERSION,
        "site": site,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code": code_version(),
        # `model_home` is None for a tier-1 bundle produced on a machine that
        # holds no weights, which is legitimate: tier 1 reads the tokenizer and
        # never loads the model, so there is nothing for the coordinator to
        # check the weights against. The tokenizer hash still travels, and it
        # is the hash that matters for aggregate tables -- the token ids in
        # `counts/` are only comparable across sites under one tokenizer.
        # A tier-2 bundle always carries both, and `check_agreement`
        # skips nulls rather than reading them as disagreement.
        "model": {
            "key": model_key,
            "config_sha256": file_digest(model_home / "config.json")
            if model_home is not None
            else None,
            "weights_sha256": file_digest(model_home / "model.safetensors")
            if model_home is not None
            and (model_home / "model.safetensors").is_file()
            else None,
            "embedding_dim": embedding_dim,
        },
        # None where the site holds no tokenizer. Legitimate: without one, 01
        # skips the token stages, so the bundle carries no token ids and there
        # is nothing for the coordinator to check a mapping against. A bundle
        # that DOES carry token counts always carries this hash, which is what
        # makes those counts comparable across sites.
        "tokenizer_sha256": file_digest(tokenizer_yaml)
        if tokenizer_yaml is not None
        else None,
        "anchors": list(anchors),
        "row_counts": dict(row_counts),
        # lets the coordinator spot a resend whose pseudonyms no longer line up
        # with the previous bundle from the same site
        "salt_fingerprint": salt_fingerprint(salt),
        "min_cell": min_cell,
        # a site whose DUA forbids patient-level transfer sends counts only;
        # the pooled side degrades rather than failing
        "shares_patient_level": shares_patient_level,
        "notes": notes,
    }


def write(manifest: dict, bundle_home: pathlib.Path) -> pathlib.Path:
    path = bundle_home / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read(bundle_home: pathlib.Path) -> dict:
    path = bundle_home / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{bundle_home.name} has no {MANIFEST_NAME}. A bundle without one "
            "cannot be checked for code, model or tokenizer agreement, so it is "
            "not accepted. Re-export it with the current site pipeline."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def check_agreement(manifests: dict[str, dict], logger) -> None:
    """
    assert the bundles were produced comparably, and say so when they were not

    Schema mismatch is fatal: the column set differs, so the pooled frame would
    be wrong rather than merely inconsistent. Model and tokenizer mismatch are
    fatal for the same reason -- embeddings from two checkpoints are not in the
    same space and pooling them produces a figure of nothing.

    Code mismatch warns rather than raises. It means the cohorts were defined
    differently, which is serious, but it is recoverable by re-running one site
    and it should not block a coordinator from inspecting what arrived.
    """
    if not manifests:
        raise ValueError("no bundles to check")

    def collect(path: tuple[str, ...]) -> dict[str, object]:
        out = {}
        for site, m in manifests.items():
            node: object = m
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            out[site] = node
        return out

    for path, label in (
        (("schema_version",), "export schema version"),
        (("model", "key"), "model key"),
        (("model", "config_sha256"), "model config"),
        (("model", "weights_sha256"), "model weights"),
        (("tokenizer_sha256",), "tokenizer"),
    ):
        values = collect(path)
        distinct = {v for v in values.values() if v is not None}
        if len(distinct) > 1:
            detail = ", ".join(f"{s}={str(v)[:12]}" for s, v in values.items())
            raise ValueError(
                f"bundles disagree on {label}: {detail}. Embeddings produced "
                "under different weights or a different token mapping are not "
                "in the same space and must not be pooled."
            )

    shas = collect(("code", "git_sha"))
    if len({v for v in shas.values() if v}) > 1:
        logger.warning(
            "bundles were produced by different commits of this repository: "
            + ", ".join(f"{s}={str(v)[:8]}" for s, v in shas.items())
            + " -- the cohort definition may differ between sites. Re-run the "
            "site pipeline from one commit before publishing anything pooled."
        )
    dirty = [s for s, m in manifests.items() if m.get("code", {}).get("git_dirty")]
    if dirty:
        logger.warning(
            f"bundle(s) from {', '.join(dirty)} were produced from a modified "
            "working tree, so the git SHA does not fully describe the code that "
            "made them"
        )

    cells = collect(("min_cell",))
    if len({v for v in cells.values() if v is not None}) > 1:
        logger.warning(
            "sites applied different minimum cell sizes to their count tables: "
            + ", ".join(f"{s}={v}" for s, v in cells.items())
            + " -- pooled tables inherit the weakest of them"
        )

    # A bundle with neither hash has had nothing checked. That is legitimate
    # for a tier-1 site holding no weights and no tokenizer -- it shipped no
    # embeddings and no token ids, so there is nothing a hash would protect --
    # but it must not read as "verified".
    unverifiable = sorted(
        site
        for site, m in manifests.items()
        if not m.get("tokenizer_sha256")
        and not m.get("model", {}).get("config_sha256")
    )
    if unverifiable:
        logger.warning(
            f"bundle(s) from {', '.join(unverifiable)} carry neither a model "
            "nor a tokenizer hash, so nothing in them was checked for "
            "compatibility. Expected for a site that holds neither and so "
            "shipped no embeddings and no token counts; verify that is what "
            "happened rather than a broken export."
        )

    logger.info(
        f"{len(manifests)} bundle(s) agree on schema, model and tokenizer: "
        + ", ".join(sorted(manifests))
    )

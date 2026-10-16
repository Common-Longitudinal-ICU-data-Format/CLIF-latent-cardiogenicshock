#!/usr/bin/env python3

"""
reproducing the embeddings at the coordinating site, from received token streams

Tier 2 ships the model's *input* rather than its output. This is what turns that
input back into embeddings, so that 03 onward have something to read.

WHY THIS IS EXACT AND NOT AN APPROXIMATION

The model is used frozen, in eval mode, under torch.inference_mode(). Given the
same token ids and the same elapsed times it is a deterministic function, so an
embedding computed here is the same vector the contributing site would have
computed -- not a re-estimate of it. `verify` below checks that claim against a
site's own embeddings whenever both are available, which for the coordinating
site's own data they are.

WHAT IT BUYS OVER RECEIVING EMBEDDINGS

Re-anchoring, a different layer, different pooling, a longer or shorter window
-- all of which need the sequence and none of which a fixed 1024-d vector
permits. It also means the coordinating site can re-run everything after a model
update without asking anyone to re-export.

HOW IT WORKS

cotorra's Loader.for_inference reads {split}_for_inference.parquet from a
processed_data_home and selects exactly two columns from it, tokens_past and
s_elapsed_past, renaming the first to input_ids. So a received bundle can be
staged into a directory shaped the way cotorra expects and the unmodified
Extractor run over it. Nothing about the extraction path is reimplemented here;
only the directory it reads is reconstructed.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from clifcs.config import CONFIG_HOME

#: cotorra writes one feature shard per split; these are the names it uses
SPLITS = ("train", "tuning", "held_out")


def stage_bundle(
    paths, frame: pl.DataFrame, site: str, anchor: str, logger
) -> pathlib.Path:
    """
    lay a received token bundle out the way cotorra's Loader expects

    Writes one {split}_for_inference.parquet per split present, carrying the two
    columns the Loader selects plus subject_id -- which here is the pseudonymous
    row_key, since the record number never left the site. The positional join
    that recovers it after extraction is the same one
    clifcs.embedding.extract_site_anchor relies on.
    """
    home = paths.scratch_home / "received" / f"{site.lower()}-cs" / f"anchor-{anchor}"
    if home.exists():
        shutil.rmtree(home)
    home.mkdir(parents=True, exist_ok=True)

    # the Extractor loads this from processed_data_home at construction
    shutil.copyfile(paths.tokenizer_yaml, home / "tokenizer.yaml")

    present = (
        set(frame["split"].unique().to_list())
        if "split" in frame.columns
        else {"held_out"}
    )
    for split in present:
        part = (
            frame.filter(pl.col("split") == split)
            if "split" in frame.columns
            else frame
        )
        part.select(
            pl.col("row_key").alias("subject_id"),
            "tokens_past",
            "s_elapsed_past",
        ).write_parquet(home / f"{split}_for_inference.parquet")
        logger.info(f"  staged {split}: {part.height} sequences")

    _satisfy_loader(home, frame, logger)
    return home


def _satisfy_loader(home: pathlib.Path, frame: pl.DataFrame, logger) -> None:
    """
    write the files cotorra's Loader asserts on but `extract` never reads

    `Loader.__init__` requires `tokens_times.parquet` and either
    `subject_splits.parquet` or a pre-built `<split>_tokens_times.parquet` per
    split, and builds `self.dataset` from them. `Extractor.extract` then uses
    `self.loader.for_inference` and never touches `self.dataset` -- so these
    exist to let construction succeed, not to be consumed.

    They are built from the received sequences rather than faked: `tokens` and
    `s_elapsed` are `tokens_past` and `s_elapsed_past`, which is the prefix the
    bundle carries. The full stream beyond the anchor is not in the bundle and
    is not needed, since nothing reads it.

    Ordering matters. The Loader regenerates the per-split files whenever
    `tokens_times.parquet` is newer than any of them, which would send it
    looking for `subject_splits.parquet`. Writing the aggregate first and the
    per-split files after leaves the per-split files newer, so that branch is
    not taken.
    """
    tokens_times = home / "tokens_times.parquet"
    frame.select(
        pl.col("row_key").alias("subject_id"),
        pl.col("tokens_past").alias("tokens"),
        pl.col("s_elapsed_past").alias("s_elapsed"),
    ).write_parquet(tokens_times)

    for split in SPLITS:
        part = (
            frame.filter(pl.col("split") == split)
            if "split" in frame.columns
            else frame
        )
        # an absent split still needs a typed, readable file: the Loader builds
        # a dataset over all three names unconditionally
        part.select(
            pl.col("row_key").alias("subject_id"),
            pl.col("tokens_past").alias("tokens"),
            pl.col("s_elapsed_past").alias("s_elapsed"),
        ).write_parquet(home / f"{split}_tokens_times.parquet")

    logger.info(
        "  staged tokens_times.parquet and the per-split copies cotorra's "
        "Loader constructs from (never read by extract)"
    )


def reembed(
    paths, frame: pl.DataFrame, site: str, anchor: str, logger, batch_size=None
) -> pl.DataFrame:
    """
    run the frozen model over a received token bundle

    Returns one row per patient: row_key plus emb_0000 .. emb_{dim-1}, which is
    what 03 and everything after it read.
    """
    from clifcs import embedding

    info = embedding.describe_model(paths, logger)
    dim = info["embedding_dim"]
    processed = stage_bundle(paths, frame, site, anchor, logger)

    features_home = processed / paths.model_key
    features_home.mkdir(parents=True, exist_ok=True)
    extraction_cfg = OmegaConf.load(CONFIG_HOME / "extraction.yaml")
    if batch_size is not None:
        extraction_cfg.extract.batch_size = int(batch_size)
    cfg_path = features_home / "extraction-resolved.yaml"
    OmegaConf.save(extraction_cfg, cfg_path)

    # the same overflow policy the sending site used, or the two disagree for
    # exactly the long-stay patients the onset anchor exists to capture
    keep = embedding.OVERFLOW_POLICY[anchor]
    extractor = embedding.build_extractor(
        paths, processed, features_home, cfg_path, keep
    )
    logger.info(
        f"[{site}/{anchor}] re-embedding {frame.height} received sequences on "
        f"{extractor.device} (overflow policy: keep {keep})"
    )
    extractor.extract()

    frames = []
    for split in SPLITS:
        shards = sorted(
            features_home.glob(f"features-{split}*-{paths.model_home.name}.parquet")
        )
        if not shards:
            continue
        features = pl.concat(
            [pl.read_parquet(f, columns=["features"]) for f in shards]
        )
        keys = pl.read_parquet(
            processed / f"{split}_for_inference.parquet", columns=["subject_id"]
        )
        if features.height != keys.height:
            raise RuntimeError(
                f"[{site}/{anchor}/{split}] {features.height} feature rows vs "
                f"{keys.height} staged sequences; the positional join that "
                "recovers row_key is not valid"
            )
        frames.append(keys.with_columns(features["features"]))

    if not frames:
        raise RuntimeError(f"[{site}/{anchor}] extraction produced no features")

    out = pl.concat(frames).rename({"subject_id": "row_key"})
    matrix = np.vstack(out["features"].to_list()).astype(np.float32)
    if matrix.shape[1] != dim:
        raise RuntimeError(
            f"re-embedded features have width {matrix.shape[1]}, expected the "
            f"model's hidden size {dim}"
        )
    return out.drop("features").hstack(
        pl.from_numpy(matrix, schema=[f"emb_{i:04d}" for i in range(dim)])
    )


def verify(
    reembedded: pl.DataFrame, reference: pl.DataFrame, logger, tolerance: float = 1e-2
) -> float:
    """
    check re-embedded vectors against a site's own, where both are available

    Only possible for the coordinating site's own data, where it holds both the
    token bundle and the embeddings it computed directly. Worth doing once per
    model: it is the evidence that shipping sequences loses nothing, and if it
    ever fails the cause is a disagreement about the overflow policy or the
    extraction config rather than anything subtle.

    Returns the maximum absolute difference. cotorra stores features at half
    precision, so agreement to ~1e-2 is exact rather than merely close.
    """
    columns = sorted(c for c in reembedded.columns if c.startswith("emb_"))
    joined = reembedded.join(reference, on="row_key", how="inner", suffix="_ref")
    if joined.height == 0:
        logger.warning("no overlapping row_key; cannot verify re-embedding")
        return float("nan")

    a = joined.select(columns).to_numpy()
    b = joined.select([f"{c}_ref" for c in columns]).to_numpy()
    delta = float(np.nanmax(np.abs(a - b)))
    if delta <= tolerance:
        logger.info(
            f"re-embedding verified against {joined.height} locally computed "
            f"vectors: max |difference| = {delta:.2e}, at or below the "
            f"{tolerance:.0e} half-precision tolerance"
        )
    else:
        logger.error(
            f"re-embedded vectors differ from the locally computed ones by up "
            f"to {delta:.2e}, above the {tolerance:.0e} tolerance. The likely "
            "cause is a different overflow policy or extraction config, not "
            "numerical noise -- the model is deterministic in eval mode."
        )
    return delta

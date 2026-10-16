#!/usr/bin/env python3

"""
running the frozen foundation model over a site's tokenized timelines

Was `02_model_embedding.py`; folded into the site aggregate script because the
phenotype-purity measure it reports needs the embeddings, so a site that can
only send aggregates still has to compute them. The embeddings themselves stay
on the machine unless `02_site_embeddings.py` is run as well.

The model is used strictly as a frozen encoder: weights are loaded in eval mode
under torch.inference_mode() and never updated. The representation is the final
hidden state at the last token before the anchor -- i.e. the model's state after
reading everything known about that patient up to that point.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

from clifcs.config import CONFIG_HOME

SPLITS = ("train", "tuning", "held_out")

#: What to do when a patient's pre-anchor history exceeds the model's max_len.
#:
#: cotorra keeps the first max_len tokens, which is right for the 24h anchor --
#: the prefix is short (median 783 tokens; 2 of 9135 UCMC patients overflow) and
#: keeping the head matches how the model was evaluated. It is wrong for the
#: onset anchor: 3.4% of patients develop shock late in a long stay, and keeping
#: the head would read them at hour ~10 of a two-week admission instead of at
#: onset. Those get the *last* max_len tokens before onset, so the
#: representation really is the model's state at the event.
#:
#: Keeping the tail does not re-base elapsed time, and does not need to: cotorra
#: trains by concatenating patients into one stream and chopping it into max_len
#: chunks, so chunks whose position ids start far from zero are what the model
#: saw during training.
OVERFLOW_POLICY = {"24h": "first", "onset": "last"}


# ======================================================== model description


def describe_model(paths, logger) -> dict:
    """
    report the architecture and embedding dimension, and write it alongside the
    embeddings so the provenance travels with the data
    """
    config = AutoConfig.from_pretrained(paths.model_home)
    model = AutoModelForCausalLM.from_pretrained(paths.model_home)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainer_state = paths.model_home / "trainer_state.json"
    state = json.loads(trainer_state.read_text()) if trainer_state.is_file() else {}

    # AutoConfig rewrites _name_or_path to whatever directory it loaded from, so
    # the base model it was derived from has to come from the file on disk
    raw_config = json.loads((paths.model_home / "config.json").read_text())

    tokenizer_cfg = OmegaConf.load(paths.tokenizer_yaml)
    vocab = len(tokenizer_cfg.lookup)

    info = {
        "model_key": paths.model_key,
        "model_home": str(paths.model_home),
        "base_model": raw_config.get("_name_or_path"),
        "architecture": config.architectures[0] if config.architectures else None,
        "model_type": config.model_type,
        "embedding_dim": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        "intermediate_size": int(config.intermediate_size),
        "vocab_size": int(config.vocab_size),
        "tokenizer_vocab_size": vocab,
        "max_position_embeddings": int(config.max_position_embeddings),
        "rope_theta": float(config.rope_theta),
        "tie_word_embeddings": bool(config.tie_word_embeddings),
        "torch_dtype": str(config.torch_dtype),
        "total_parameters": int(n_params),
        "trainable_parameters": int(n_trainable),
        "training_best_eval_loss": state.get("best_metric"),
        "training_epochs": state.get("epoch"),
        "training_global_step": state.get("global_step"),
        "tokenizer_yaml": str(paths.tokenizer_yaml),
        "tokenizer_created": str(tokenizer_cfg.get("created_dttm", "")),
        "cocoa_version": str(tokenizer_cfg.get("cocoa_version", "")),
    }

    if config.vocab_size != vocab:
        raise RuntimeError(
            f"model vocab_size ({config.vocab_size}) does not match the "
            f"tokenizer's lookup ({vocab}); the embedding table and the token "
            "ids are not the same vocabulary, so the representations would be "
            "meaningless"
        )

    logger.info("=" * 78)
    logger.info("FROZEN FOUNDATION MODEL")
    logger.info("=" * 78)
    logger.info(f"  checkpoint            {paths.model_home}")
    logger.info(f"  base model            {info['base_model']}")
    logger.info(f"  architecture          {info['architecture']} ({info['model_type']})")
    logger.info(f"  EMBEDDING DIMENSION   {info['embedding_dim']}")
    logger.info(
        f"  layers / heads        {info['num_hidden_layers']} layers, "
        f"{info['num_attention_heads']} heads (kv {info['num_key_value_heads']}), "
        f"head_dim {info['head_dim']}"
    )
    logger.info(f"  MLP intermediate      {info['intermediate_size']}")
    logger.info(
        f"  vocabulary            {info['vocab_size']} tokens "
        f"(tokenizer lookup {vocab}: match)"
    )
    logger.info(
        f"  parameters            {n_params:,} total ({n_params / 1e6:.1f}M), "
        f"{n_trainable:,} trainable"
    )
    logger.info(
        f"  position embeddings   max {info['max_position_embeddings']}, "
        f"rope_theta {info['rope_theta']:g}, tied embeddings "
        f"{info['tie_word_embeddings']}"
    )
    logger.info(
        f"  training              {info['training_epochs']} epoch(s), "
        f"{info['training_global_step']} steps, best eval loss "
        f"{info['training_best_eval_loss']}"
    )
    logger.info("")
    logger.info("  module tree:")
    for line in str(model).splitlines():
        logger.info(f"    {line}")
    logger.info("=" * 78)

    del model
    return info


# ============================================================== extraction


def report_overflow(processed: pathlib.Path, site, anchor, max_len, keep, logger) -> None:
    """say how many patients are longer than the model can read, and which end is kept"""
    lengths = pl.concat(
        [
            pl.scan_parquet(processed / f"{split}_for_inference.parquet").select(
                "last_valid"
            )
            for split in SPLITS
            if (processed / f"{split}_for_inference.parquet").is_file()
        ]
    ).collect()["last_valid"]
    over = int((lengths > max_len).sum())
    logger.info(
        f"[{site}/{anchor}] pre-anchor length: median {lengths.median():.0f}, "
        f"p90 {lengths.quantile(0.9):.0f}, max {lengths.max()} tokens; "
        f"{over}/{len(lengths)} ({over / max(len(lengths), 1):.1%}) exceed "
        f"max_len {max_len} and are truncated keeping the {keep} tokens"
    )


def build_extractor(paths, processed, features_home, cfg_path, keep):
    """
    cotorra's Extractor, with a configurable overflow policy

    Extractor.collate_fn hard-codes `x[:max_len]`. Pre-trimming the batch to the
    tail before delegating leaves the rest of its logic -- padding, time-based
    rope position ids, locating the last token before EOS -- exactly as cotorra
    wrote it.
    """
    from cotorra.extractor import Extractor

    class AnchorExtractor(Extractor):
        def collate_fn(self, batch):
            if keep == "last":
                max_len = int(self.cfg.get("extract", {}).get("max_len", 4096))
                batch = {
                    key: (
                        [x[-max_len:] for x in value]
                        if key in ("input_ids", "s_elapsed_past")
                        else value
                    )
                    for key, value in batch.items()
                }
            return super().collate_fn(batch)

    return AnchorExtractor(
        extraction_cfg=cfg_path,
        processed_data_home=processed,
        model_home=paths.model_home,
        output_home=features_home,
    )


def extract_site_anchor(paths, site, anchor, batch_size, logger) -> pl.DataFrame:
    """
    run cotorra's Extractor for one site and anchor, then put the patient
    identifiers back on the features

    cotorra drops subject_id on the way through (its Loader selects only
    input_ids and s_elapsed_past), so the features parquet carries no key. It
    does preserve row order within each split, which is how cotorra's own
    RepBasedScorer rejoins labels, so the identifiers are taken positionally
    from {split}_for_inference.parquet -- with the row counts asserted, because
    a silent length mismatch here would mislabel every patient.
    """
    processed = paths.anchored(site, anchor)
    features_home = paths.features_home(site, anchor)
    features_home.mkdir(parents=True, exist_ok=True)
    extraction_cfg = OmegaConf.load(CONFIG_HOME / "extraction.yaml")
    if batch_size is not None:
        extraction_cfg.extract.batch_size = int(batch_size)

    cfg_path = features_home / "extraction-resolved.yaml"
    OmegaConf.save(extraction_cfg, cfg_path)

    keep = OVERFLOW_POLICY[anchor]
    report_overflow(
        processed, site, anchor, int(extraction_cfg.extract.max_len), keep, logger
    )

    logger.info(
        f"[{site}/{anchor}] extracting (batch_size="
        f"{extraction_cfg.extract.batch_size}, max_len={extraction_cfg.extract.max_len})"
    )
    t0 = time.perf_counter()
    extractor = build_extractor(paths, processed, features_home, cfg_path, keep)
    logger.info(f"[{site}/{anchor}] model {paths.model_key}, device {extractor.device}")
    extractor.extract()
    logger.info(f"[{site}/{anchor}] extraction took {time.perf_counter() - t0:.1f}s")

    frames = []
    for split in SPLITS:
        feature_files = sorted(
            features_home.glob(f"features-{split}*-{paths.model_home.name}.parquet")
        )
        if not feature_files:
            logger.warning(f"[{site}/{anchor}] no features for split {split}")
            continue

        features = pl.concat(
            [pl.read_parquet(f, columns=["features"]) for f in feature_files]
        )
        keys = pl.read_parquet(
            processed / f"{split}_for_inference.parquet", columns=["subject_id"]
        )
        if features.height != keys.height:
            raise RuntimeError(
                f"[{site}/{anchor}/{split}] {features.height} feature rows vs "
                f"{keys.height} inference rows; the positional join that "
                "recovers subject_id is not valid"
            )
        frames.append(
            keys.with_columns(
                features["features"],
                split=pl.lit(split),
            )
        )

    out = pl.concat(frames).rename({"subject_id": "hospitalization_id"})
    return out.with_columns(site=pl.lit(site), anchor=pl.lit(anchor))


def to_embedding_frame(frame: pl.DataFrame, dim: int) -> pl.DataFrame:
    """
    materialize the list-of-float feature column as `emb_0000 .. emb_{dim-1}`

    Stored as flat columns rather than a list so the parquet can be read by
    anything (pandas, R, DuckDB) without unnesting, and so a column-subset read
    is possible.
    """
    matrix = np.vstack(frame["features"].to_list()).astype(np.float32)
    if matrix.shape[1] != dim:
        raise RuntimeError(
            f"extracted features have width {matrix.shape[1]}, expected the "
            f"model's hidden size {dim}"
        )
    embedding = pl.from_numpy(matrix, schema=[f"emb_{i:04d}" for i in range(dim)])
    return frame.drop("features").hstack(embedding)

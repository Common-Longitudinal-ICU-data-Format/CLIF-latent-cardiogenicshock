#!/usr/bin/env python3

"""
02 -- embed the cohort with the frozen foundation model

Reports the model's structure and embedding dimension, then runs
`cotorra.Extractor` over each site x anchor and writes one row per patient to
output/patient_level_data/.

The model is used strictly as a frozen encoder: weights are loaded in eval mode
under torch.inference_mode() and never updated. The representation is the final
hidden state at the last token before the anchor -- i.e. the model's state after
reading everything known about that patient up to that point.

Usage
    python code/02_model_embedding.py
    python code/02_model_embedding.py --anchor 24h --batch-size 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import secrets
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

from clifcs import ANCHORS, get_logger, load_paths
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


# ============================================================ sharing export


#: columns stripped from a share-safe export, and why
_DIRECT_IDENTIFIERS = ("patient_id", "hospitalization_id")
_DATE_COLUMNS = ("cs_onset_dttm",)


def write_share_safe(
    frame: pl.DataFrame, source: pathlib.Path, salt_file: pathlib.Path, logger
) -> pathlib.Path:
    """
    write a copy with direct identifiers and raw dates removed

    This reduces what travels; it does not make the file anonymous, and it is
    not a substitute for a DUA or IRB approval:

      * local record numbers are replaced by a salted hash, so they cannot be
        reversed without the salt -- which stays on the sending machine;
      * `cs_onset_dttm` is dropped, because a timestamp finer than a year is a
        HIPAA identifier. The relative `hours_admission_to_cs_onset` is kept,
        which is what the analysis actually uses;
      * ages over 89 are capped, per Safe Harbor.

    The 1024-d embedding itself is NOT a de-identified summary. It is a
    learned function of the patient's whole recorded history, and no one has
    shown such representations are robust to re-identification attack. Treat
    the export as limited-data-set-grade at best, and move it under the same
    agreement you would use for patient-level clinical data.
    """
    salt_file.parent.mkdir(parents=True, exist_ok=True)
    if not salt_file.is_file():
        salt_file.write_text(secrets.token_hex(32), encoding="utf-8")
        logger.info(
            f"created {salt_file.name}; it stays on this machine and is what "
            "lets you re-link a shared pseudonym back to a local record"
        )
    salt = salt_file.read_text(encoding="utf-8").strip()

    def pseudonymize(column: str) -> pl.Series:
        # SHA-256, not polars' built-in .hash(): that is xxhash, which is a
        # fast non-cryptographic hash and carries no preimage guarantee. The
        # claim being made here is that a recipient cannot recover the record
        # number, so the hash has to be one that actually supports it.
        return pl.Series(
            f"{column}_pseudonym",
            [
                hashlib.sha256(f"{salt}:{column}:{value}".encode()).hexdigest()[:32]
                for value in frame[column].cast(pl.String).to_list()
            ],
        )

    present_ids = [c for c in _DIRECT_IDENTIFIERS if c in frame.columns]
    safe = (
        frame.with_columns([pseudonymize(c) for c in present_ids])
        .drop(*present_ids, *(c for c in _DATE_COLUMNS if c in frame.columns))
        .with_columns(
            age_at_admission=pl.when(pl.col("age_at_admission") > 89)
            .then(90)
            .otherwise(pl.col("age_at_admission"))
        )
    )
    out = source.with_name(source.stem + "_share-safe.parquet")
    safe.write_parquet(out)
    logger.info(
        f"  share-safe copy -> {out.name} "
        f"(dropped {', '.join(present_ids)}, {', '.join(_DATE_COLUMNS)}; ages capped at 90)"
    )
    return out


# =================================================================== driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", choices=ANCHORS, default=None, help="one anchor only")
    parser.add_argument("--site", default=None, help="one site only")
    parser.add_argument(
        "--model",
        default=None,
        help="which checkpoint from paths.yaml to embed with "
        "(default: default_model). Outputs are written under that name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="sequences per forward pass; lower this if the GPU runs out of "
        "memory (output_hidden_states keeps every layer)",
    )
    parser.add_argument(
        "--share-safe-only",
        action="store_true",
        help="regenerate the share-safe copies from the per-site files already "
        "on disk, without re-running the model",
    )
    parser.add_argument(
        "--share-safe",
        action="store_true",
        help="also write a *_share-safe.parquet per site with direct identifiers "
        "and raw timestamps removed, for transfer to a collaborating site",
    )
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger("02_model_embedding", paths.log_home / "02_model_embedding.log")

    if args.share_safe_only:
        regenerated = []
        for source in sorted(paths.patient_level_home.glob("embeddings_anchor-*.parquet")):
            if "_share-safe" in source.name:
                continue
            regenerated.append(
                write_share_safe(pl.read_parquet(source), source, paths.id_salt_file, logger)
            )
        logger.info(
            "02 done; regenerated " + ", ".join(p.name for p in regenerated)
        )
        return

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; extraction will run on CPU and be slow")

    logger.info(f"model: {paths.model_key}  (available: {', '.join(paths.available_models)})")
    info = describe_model(paths, logger)
    paths.table_home.mkdir(parents=True, exist_ok=True)
    (paths.table_home / "model_summary.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    if args.site and args.site not in paths.sites:
        parser.error(f"--site {args.site} is not configured; have {paths.sites}")

    cohort = pl.read_parquet(paths.cohort_home / "cohort.parquet")
    anchors = (args.anchor,) if args.anchor else ANCHORS
    sites = (args.site,) if args.site else paths.sites

    # one row per patient: the embedding plus everything needed to colour it
    covariates = cohort.select(
        "hospitalization_id",
        "patient_id",
        "site",
        "cs_onset_dttm",
        "cs_phenotype",
        "icu_los_days",
        "in_hospital_mortality",
        "cardiac_origin",
        "shock_type",
        "age_at_admission",
        "sex_category",
        "hours_admission_to_cs_onset",
    )

    written = []
    for anchor in anchors:
        for site in sites:
            frame = to_embedding_frame(
                extract_site_anchor(paths, site, anchor, args.batch_size, logger),
                info["embedding_dim"],
            )
            frame = frame.join(
                covariates,
                on=["hospitalization_id", "site"],
                how="inner",
                validate="1:1",
            )

            # One file per site per anchor, rather than one pooled file. This is
            # what lets a collaborating site contribute: it runs 01 and 02 on
            # its own data and sends back its two files, which 03 picks up by
            # glob without anything having to be merged by hand.
            out = (
                paths.patient_level_home
                / f"embeddings_anchor-{anchor}_{site}.parquet"
            )
            frame.write_parquet(out)
            written.append(out)
            logger.info(
                f"[{anchor}/{site}] wrote {frame.height} patient embeddings "
                f"({info['embedding_dim']}-d) -> {out.name}"
            )

            if args.share_safe:
                safe = write_share_safe(frame, out, paths.id_salt_file, logger)
                written.append(safe)

    logger.info("02 done; wrote " + ", ".join(p.name for p in written))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
14_sentence_latent.py -- do similar sentences sit together in the latent space?

Each patient has two descriptions of the same 48 hours: a BPE sentence over the
MAP/pressor states, and a trajectory of nine 1024-d GEM embeddings. This
measures the distance between patients under each and asks where the two
disagree.

    python code/14_sentence_latent.py --subset septic

THE DISAGREEMENTS ARE THE POINT. Pairs that are close under both, or far under
both, tell us the model agrees with the state description. The off-diagonal
pairs measure what the 15-state space is MISSING:

  divergent sentence, close latent
      the model treats these patients as alike despite different MAP and
      pressor courses, so whatever they share is outside the state space

  similar sentence, far latent
      identical MAP and pressor course, and the model still separates them,
      so the difference is outside the state space

WINDOW. Restricted to 0-48h, nine grid points, the only stretch where the state
sequence and the embedding grid overlap. A sentence built over seven days
cannot be compared pairwise against an embedding that stops at day two.

CONFOUNDERS ARE REPORTED, NOT REMOVED. Position in the projection explains
R2 0.45 of log stream length, and site survives CORAL at non-linear AUC 0.855.
So "close in latent space" partly means "similarly documented, same hospital".
Every quadrant carries its mean stream-length gap and same-site share, so a
quadrant that is really a documentation artefact is visible as one.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clifcs import get_logger, load_paths
from clifcs.bpe import edit_distance

#: the overlap between the state grid and the embedding grid
OVERLAP = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]


def load_latent(paths, subset, logger):
    """the corrected 1024-d trajectory over the overlap window, per patient"""
    candidates = sorted(
        paths.patient_level_home.glob(
            f"trajectory_projections_{subset}_corrected-*.parquet"
        )
    ) or sorted(
        paths.patient_level_home.glob(f"trajectory_projections_{subset}.parquet")
    )
    if not candidates:
        logger.error(f"no trajectory projections for {subset}")
        return None, None, None
    logger.info(f"reading {candidates[0].name}")
    frame = pl.read_parquet(candidates[0])
    if "last_valid" not in frame.columns:
        logger.warning("projections carry no last_valid; stream length omitted")
    return frame, candidates[0].name, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="septic",
                        choices=("septic", "cardiogenic"))
    parser.add_argument("--pairs", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paths = load_paths(model=args.model)
    paths.mkdirs()
    logger = get_logger("14_sentence_latent", paths.log_home / "14_sentence_latent.log")

    labels = pl.read_parquet(
        paths.patient_level_home / f"longitudinal_clusters_{args.subset}.parquet"
    )
    if "bpe_sentence_48h" not in labels.columns:
        logger.error(
            "no bpe_sentence_48h column; run 11 after the BPE change first"
        )
        return

    # The embeddings themselves, not the projection: PaCMAP distances are not
    # metric, so a 2-d "close" is not a claim about the representation.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "traj", pathlib.Path(__file__).resolve().parent / "09_trajectories.py"
    )
    traj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(traj)
    import types

    grid_args = types.SimpleNamespace(
        subset=args.subset, start_hours=-24.0, end_hours=48.0, step_hours=6.0,
        correction="coral", anchor="onset", force=False, min_tokens=8,
    )
    frame = traj.load_grid(paths, grid_args, logger)
    frame = traj.apply_correction(paths, frame, "coral", "onset", logger)
    frame = frame.filter(pl.col("offset_hours").is_in(OVERLAP))

    columns = sorted(c for c in frame.columns if c.startswith("emb_"))
    complete = (
        frame.group_by("hospitalization_id").len()
        .filter(pl.col("len") == len(OVERLAP))["hospitalization_id"]
    )
    frame = frame.filter(
        pl.col("hospitalization_id").is_in(complete)
    ).sort(["hospitalization_id", "offset_hours"])
    logger.info(
        f"{frame['hospitalization_id'].n_unique():,} patients complete over "
        f"{len(OVERLAP)} windows"
    )

    sentences = dict(zip(
        labels["hospitalization_id"].to_list(),
        labels["bpe_sentence_48h"].to_list(),
    ))
    # the raw 9-letter state string too: the word-level distance saturates
    letters = dict(zip(
        labels["hospitalization_id"].to_list(),
        [(s or "")[:len(OVERLAP)] for s in labels["sequence_raw"].to_list()],
    ))
    patients = [
        p for p in frame["hospitalization_id"].unique().sort().to_list()
        if p in sentences and sentences[p]
    ]
    index = {p: i for i, p in enumerate(patients)}
    keep = frame.filter(pl.col("hospitalization_id").is_in(patients))
    matrix = (
        keep.select(columns).to_numpy().astype(np.float32)
        .reshape(len(patients), len(OVERLAP) * len(columns))
    )
    logger.info(f"latent trajectories: {matrix.shape}")

    words = {p: sentences[p].split() for p in patients}
    site = dict(zip(
        keep["hospitalization_id"].to_list(), keep["site"].to_list()
    ))
    length = {}
    if "last_valid" in keep.columns:
        per = keep.group_by("hospitalization_id").agg(
            stream=pl.col("last_valid").median()
        )
        length = dict(zip(
            per["hospitalization_id"].to_list(), per["stream"].to_list()
        ))

    rng = np.random.default_rng(args.seed)
    left = rng.integers(0, len(patients), size=args.pairs)
    right = rng.integers(0, len(patients), size=args.pairs)
    ok = left != right
    left, right = left[ok], right[ok]
    logger.info(f"scoring {len(left):,} patient pairs")

    latent = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), 4000):
        stop = min(start + 4000, len(left))
        latent[start:stop] = np.linalg.norm(
            matrix[left[start:stop]] - matrix[right[start:stop]], axis=1
        )

    # TWO sentence distances, because the obvious one is unusable.
    #
    # Normalised WORD-level edit distance saturates: two sentences that share
    # no BPE word score exactly 1.0, and with a 114-word vocabulary most pairs
    # do, so the median is 1.0 and a median split puts every pair on one side.
    # That is what happened on the first run -- both "divergent sentence"
    # quadrants came out empty.
    #
    # The letter-level distance over the nine state symbols is well behaved: it
    # is the fraction of the nine windows that differ, so it takes ten distinct
    # values and splits cleanly. Word-level Jaccard is kept beside it because
    # it is the one that answers "do these trajectories share vocabulary",
    # which is the question BPE was built for.
    sentence = np.empty(len(left), dtype=np.float32)
    jaccard = np.empty(len(left), dtype=np.float32)
    gap = np.zeros(len(left), dtype=np.float32)
    same_site = np.zeros(len(left), dtype=bool)
    for k, (i, j) in enumerate(zip(left, right)):
        a, b = patients[i], patients[j]
        wa, wb = words[a], words[b]
        set_a, set_b = set(wa), set(wb)
        union = set_a | set_b
        jaccard[k] = (
            1.0 - len(set_a & set_b) / len(union) if union else 0.0
        )
        la, lb = letters.get(a, ""), letters.get(b, "")
        sentence[k] = (
            edit_distance(list(la), list(lb)) / max(len(la), len(lb), 1)
            if la and lb else 1.0
        )
        if length:
            gap[k] = abs(
                np.log1p(length.get(a, 0)) - np.log1p(length.get(b, 0))
            )
        same_site[k] = site.get(a) == site.get(b)

    from scipy.stats import spearmanr
    rho = spearmanr(sentence, latent).statistic
    rho_words = spearmanr(jaccard, latent).statistic
    logger.info(
        f"latent distance vs state-letter distance: Spearman rho {rho:+.3f}; "
        f"vs BPE-word Jaccard: {rho_words:+.3f}"
    )
    logger.info(
        f"  state-letter distance: median {np.median(sentence):.3f}, "
        f"{len(np.unique(sentence))} distinct values; "
        f"word Jaccard median {np.median(jaccard):.3f}"
    )

    s_cut = np.median(sentence)
    l_cut = np.median(latent)
    # A median equal to the maximum leaves one side of the split empty. Fall
    # back to a lower quantile and say so rather than reporting two quadrants
    # as though the other two did not exist.
    if s_cut >= sentence.max():
        s_cut = float(np.quantile(sentence, 0.25))
        logger.warning(
            "sentence distance is degenerate at the median; splitting at the "
            f"25th percentile ({s_cut:.3f}) instead"
        )
    quadrants = {
        "similar sentence, close latent": (sentence <= s_cut) & (latent <= l_cut),
        "similar sentence, FAR latent": (sentence <= s_cut) & (latent > l_cut),
        "divergent sentence, CLOSE latent": (sentence > s_cut) & (latent <= l_cut),
        "divergent sentence, far latent": (sentence > s_cut) & (latent > l_cut),
    }
    rows = []
    logger.info("quadrants (median split on each axis):")
    for name, mask in quadrants.items():
        if not mask.any():
            continue
        rows.append({
            "quadrant": name,
            "pairs": int(mask.sum()),
            "share": float(mask.mean()),
            "mean_sentence_distance": float(sentence[mask].mean()),
            "mean_word_jaccard": float(jaccard[mask].mean()),
            "mean_latent_distance": float(latent[mask].mean()),
            "mean_log_stream_gap": float(gap[mask].mean()),
            "same_site_share": float(same_site[mask].mean()),
        })
        logger.info(
            f"    {name:<34} {int(mask.sum()):>7,} pairs  "
            f"stream gap {gap[mask].mean():.3f}  "
            f"same site {same_site[mask].mean():.1%}"
        )

    table = pl.DataFrame(rows).with_columns(spearman_rho=pl.lit(float(rho)))
    out = (
        paths.output_home / "tables" / paths.model_key
        / f"sentence_vs_latent_{args.subset}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out)
    logger.info(f"wrote {out}")
    logger.info("14 done")


if __name__ == "__main__":
    main()

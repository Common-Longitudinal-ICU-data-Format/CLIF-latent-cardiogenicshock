#!/usr/bin/env python3

"""
longitudinal clustering of septic shock, in token space

Two methods, deliberately different in what they assume:

  state sequences   Each window is assigned one of a small number of discrete
                    care states, learned from the window compositions. A
                    patient becomes a string of nine symbols and patients are
                    clustered on those strings. Assumes the trajectory is well
                    described by which state you are in, and says nothing about
                    rate of change.

  latent class      Each component of the composition is summarised per patient
  growth            by a growth curve -- intercept, slope, curvature -- and
                    patients are clustered on those coefficients. Assumes
                    smooth change and is sensitive to direction and rate, but
                    cannot represent a patient who oscillates.

Agreement between the two is worth more than either alone. A structure both
find, under assumptions that disagree, is more likely to be in the data than in
the method.

THE WINDOW IS +0h TO +48h. The -24h grid point was dropped on purpose: to have
a -24h point at all a patient must already have been admitted a day before
shock onset, so that panel is a longer-staying population -- median prefix 799
tokens against 238 at onset, and the 4096-token context binds four times as
often. Clustering across it would partly be clustering admission timing.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture


def care_states(matrix, n_states: int, seed: int, logger):
    """
    discretise each patient-window into one of `n_states` care states

    NMF first, because a composition over ~hundreds of tokens is sparse and
    non-negative and NMF's parts-based factorisation gives components that read
    as co-occurring groups of events rather than as contrasts between them,
    which is what a PCA of compositional data gives. The states are then
    k-means centroids in that topic space.

    Fitted over ALL patient-windows pooled, not per window position, so a state
    means the same thing at +6h as at +48h -- that is what makes the symbol
    sequence comparable along its own length.
    """
    n_patients, n_windows, n_tokens = matrix.shape
    flat = matrix.reshape(n_patients * n_windows, n_tokens)
    occupied = flat.sum(axis=1) > 0

    components = min(12, max(2, n_tokens // 20))
    model = NMF(
        n_components=components, init="nndsvda", random_state=seed,
        max_iter=400, tol=1e-4,
    )
    topics = np.zeros((flat.shape[0], components), dtype=np.float32)
    topics[occupied] = model.fit_transform(flat[occupied])
    logger.info(
        f"NMF: {components} topics over {int(occupied.sum()):,} occupied "
        f"patient-windows, reconstruction error {model.reconstruction_err_:.4f}"
    )

    kmeans = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
    states = np.full(flat.shape[0], -1, dtype=int)
    states[occupied] = kmeans.fit_predict(topics[occupied])
    # an empty window is its own state rather than being forced into a
    # composition cluster it has no composition for
    states = states + 1  # 0 now means "nothing recorded"

    counts = np.bincount(states, minlength=n_states + 1)
    logger.info("care states over patient-windows:")
    for s, n in enumerate(counts):
        label = "nothing recorded" if s == 0 else f"state {s}"
        logger.info(f"    {label:<20} {n:>8,}  {n / len(states):>6.1%}")
    return states.reshape(n_patients, n_windows), topics.reshape(
        n_patients, n_windows, components
    ), model


def sequence_clusters(states, n_clusters: int, seed: int, logger):
    """
    cluster the nine-symbol state sequences

    One-hot per position, then k-means -- which on a one-hot encoding is
    clustering by how many positions two sequences disagree on, i.e. Hamming
    distance over the aligned sequence.

    Aligned, not warped, and that is the choice. Dynamic time warping would let
    a patient who deteriorates over 12 h match one who takes 36 h, but every
    sequence here is already anchored to the patient's own shock onset and has
    exactly nine positions six hours apart. Warping would trade away the one
    thing the anchor bought -- that position k means the same elapsed time for
    everyone -- to buy tolerance to rate. Rate differences are a finding here,
    not a nuisance to normalise away.
    """
    n_patients, n_windows = states.shape
    n_symbols = int(states.max()) + 1
    one_hot = np.zeros((n_patients, n_windows * n_symbols), dtype=np.float32)
    for w in range(n_windows):
        one_hot[np.arange(n_patients), w * n_symbols + states[:, w]] = 1.0

    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = model.fit_predict(one_hot)
    logger.info(
        f"sequence clusters: k={n_clusters}, sizes "
        f"{np.bincount(labels, minlength=n_clusters).tolist()}"
    )
    return labels, one_hot


def growth_coefficients(topics, logger):
    """
    per patient, a quadratic growth curve per topic

    Two-stage latent class growth: fit the curves, then mix. True GBTM
    estimates both jointly by EM, which is better and much heavier; the
    two-stage form is the standard practical approximation and is named as such
    rather than passed off as the full thing.

    Time is centred and scaled so the intercept is the value at the MIDDLE of
    the window rather than at +0h. With an uncentred basis the intercept and
    slope are strongly correlated, and a mixture over them finds that
    correlation before it finds any patients.
    """
    n_patients, n_windows, n_topics = topics.shape
    t = np.linspace(-1.0, 1.0, n_windows)
    basis = np.column_stack([np.ones_like(t), t, t**2 - t.var()])
    pseudo = np.linalg.pinv(basis)

    coefficients = np.einsum("cw,pwk->pkc", pseudo, topics)
    flat = coefficients.reshape(n_patients, n_topics * 3)
    logger.info(
        f"growth curves: {n_topics} topics x 3 coefficients "
        f"(level, slope, curvature) = {flat.shape[1]} features per patient"
    )
    return flat.astype(np.float64)


def growth_clusters(features, n_clusters: int, seed: int, logger):
    """latent classes over the growth coefficients"""
    scaled = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
    model = GaussianMixture(
        n_components=n_clusters, covariance_type="diag", random_state=seed,
        n_init=5, reg_covar=1e-4,
    )
    labels = model.fit_predict(scaled)
    logger.info(
        f"growth classes: k={n_clusters}, BIC {model.bic(scaled):,.0f}, sizes "
        f"{np.bincount(labels, minlength=n_clusters).tolist()}"
    )
    return labels, scaled, model.bic(scaled)


# ================================================================ validation


def split_half_stability(features, n_clusters, seed, fit, logger, rounds: int = 5):
    """
    how reproducible the partition is

    Each round splits the patients in half, clusters each half independently,
    and assigns every patient in the other half to its nearest centroid. The
    ARI between "clustered in my half" and "assigned from the other half" says
    whether the partition is a property of the cohort or of the sample.

    Reported instead of silhouette because silhouette asks whether the clusters
    are compact, which a k-means objective optimises directly and will therefore
    always look respectable. Stability asks whether they are the same clusters
    twice, which is the question.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(rounds):
        order = rng.permutation(len(features))
        a, b = order[: len(order) // 2], order[len(order) // 2 :]
        labels_a, centroids_a = fit(features[a], n_clusters, seed)
        labels_b, centroids_b = fit(features[b], n_clusters, seed)
        # label each half by the OTHER half's centroids
        cross_a = np.argmin(
            ((features[a][:, None, :] - centroids_b[None]) ** 2).sum(axis=2), axis=1
        )
        cross_b = np.argmin(
            ((features[b][:, None, :] - centroids_a[None]) ** 2).sum(axis=2), axis=1
        )
        scores.append(
            0.5 * (adjusted_rand_score(labels_a, cross_a)
                   + adjusted_rand_score(labels_b, cross_b))
        )
    mean = float(np.mean(scores))
    logger.info(
        f"    split-half stability ARI {mean:.3f} "
        f"(rounds {', '.join(f'{s:.3f}' for s in scores)})"
    )
    return mean


def state_movement(states, labels, logger) -> dict:
    """
    how much patients actually move between states

    This replaced a time-shuffled null, which was not a meaningful test here.
    Two things were wrong with it. Silhouette mostly reports that k-means found
    a compact partition, which it will in almost any feature space, so a high
    score on shuffled data is not evidence that the real partition is empty.
    And the windows are anchored to the patient's own shock onset on purpose --
    that alignment is the design -- so permuting them does not produce a
    plausible alternative cohort to compare against, it produces noise.

    Counting transitions asks the question directly and the answer is
    interpretable: if most patients never left their first state then order
    cannot be carrying much, whatever a silhouette says.
    """
    changes = (states[:, 1:] != states[:, :-1]).sum(axis=1)
    distinct = np.array([len(set(row.tolist())) for row in states])
    static = float((changes == 0).mean())
    logger.info(
        f"  state movement: {int((changes > 0).sum()):,} of {len(changes):,} "
        f"patients ({1 - static:.1%}) change state at least once; "
        f"median {np.median(changes):.0f} of {states.shape[1] - 1} possible "
        f"transitions, mean {changes.mean():.2f}"
    )
    logger.info(
        f"    distinct states visited: "
        + ", ".join(
            f"{k} in {int((distinct == k).sum()):,}"
            for k in range(1, int(distinct.max()) + 1)
        )
    )
    for cluster in sorted(set(labels.tolist())):
        mask = labels == cluster
        logger.info(
            f"    cluster {cluster}: n={int(mask.sum()):,}  "
            f"static {float((changes[mask] == 0).mean()):.1%}  "
            f"mean changes {changes[mask].mean():.2f}"
        )
    return {"static_share": static, "mean_changes": float(changes.mean())}


#: Hypothesised meanings for the 12 NMF factors, each paired with the token we
#: expect to dominate it.
#:
#: THESE ARE TIED TO ONE FIT. NMF numbers its components arbitrarily, so a
#: refit -- a different seed, a different n_components, a changed cohort or a
#: changed site filter -- will permute them and silently leave every label
#: attached to the wrong factor. The anchor token is what makes that
#: detectable: `check_factor_labels` compares each factor's actual top loading
#: against the expected one and warns on mismatch rather than letting a
#: mislabelled figure out.
#:
#: Eight of the twelve describe care delivery or documentation rather than the
#: patient. That is a property of the token vocabulary, not a defect of the
#: factorisation, and it is why these are "care-delivery trajectories" rather
#: than "illness trajectories" wherever they are described.
FACTOR_LABELS = {
    "factor 01": ("Uninterpretable (UNK)", "UNK"),
    "factor 02": ("Low-flow oxygen", "RESP//nasal_cannula"),
    "factor 03": ("Ventilated & sedated", "RESP//imv"),
    "factor 04": ("Record ends / ICU exit", "EOS"),
    "factor 05": ("Ventilated, moderate FiO2", "RESP//fio2_set_Q5"),
    "factor 06": ("Hypotension", "VTL//sbp_Q0"),
    "factor 07": ("Ventilated, high FiO2", "RESP//fio2_set_Q8"),
    "factor 08": ("Severe hypoxaemia", "RESP//fio2_set_Q9"),
    "factor 09": ("Hypertension", "VTL//sbp_Q9"),
    "factor 10": ("Routine lab ordering", "LAB-ORD//bicarbonate"),
    "factor 11": ("Tachycardia & tachypnoea", "VTL//heart_rate_Q9"),
    "factor 12": ("Well oxygenated", "VTL//spo2_Q9"),
}


def factor_label(key: str) -> str:
    """the hypothesised meaning, falling back to the bare index"""
    entry = FACTOR_LABELS.get(key)
    return f"{key.replace('factor ', 'F')} {entry[0]}" if entry else key


def check_factor_labels(model, token_ids, names, logger) -> bool:
    """
    warn if the fit no longer matches the labels

    Returns True when every labelled factor's top-loading token is the one the
    label was written against. A False here means the figures are captioned
    from a different factorisation than the one they are drawn from, which is
    worse than having no labels at all.
    """
    agree = True
    for index in range(model.components_.shape[0]):
        key = f"factor {index + 1:02d}"
        entry = FACTOR_LABELS.get(key)
        if entry is None:
            continue
        top = int(np.argmax(model.components_[index]))
        actual = names.get(int(token_ids[top]), str(token_ids[top]))
        if actual != entry[1]:
            logger.warning(
                f"{key} labelled {entry[0]!r} expecting {entry[1]!r} but its "
                f"top token is {actual!r} -- the NMF has been refit and the "
                "labels no longer correspond"
            )
            agree = False
    if agree:
        logger.info("factor labels check out against the current fit")
    return agree

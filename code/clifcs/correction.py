#!/usr/bin/env python3

"""
affine corrections for the site effect

Both methods here are affine maps on the embedding, fitted on a training split
and applied unchanged to held-out data. Neither touches the tokens or the
model; they act only on the 1024-d representation.

What they can and cannot do is worth stating plainly. 04 shows the sites are
separable at CV accuracy 0.995 because each emits token types the other never
does -- 96.5% of UCMC patients carry at least one token NU never produces. An
affine map on the output can remove the directions along which that difference
is *linearly* readable, and LEACE does so provably. It cannot make two
representations equal when the inputs that produced them were different. So
treat a drop in site-probe AUC as evidence the linear channel is closed, not
as evidence the representations have been made exchangeable.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np


@dataclasses.dataclass
class AffineCorrection:
    """x -> shift + scale * (x - centre), optionally per site"""

    name: str
    centre: np.ndarray
    transform: np.ndarray | None = None
    per_site: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    def apply(self, x: np.ndarray, sites: np.ndarray | None = None) -> np.ndarray:
        if self.per_site is not None:
            if sites is None:
                raise ValueError(f"{self.name} needs site labels at apply time")
            out = np.empty_like(x, dtype=np.float64)
            for site, (mean, scale) in self.per_site.items():
                mask = sites == site
                if mask.any():
                    out[mask] = (x[mask] - mean) / scale
            unseen = ~np.isin(sites, list(self.per_site))
            if unseen.any():
                # a site absent from the fit has no statistics of its own; fall
                # back to the pooled centre rather than silently leaving it on a
                # different scale from everything else
                out[unseen] = x[unseen] - self.centre
            return out
        return (x - self.centre) @ self.transform.T + self.centre

    # ------------------------------------------------------------ persistence
    #
    # 09 needs the map, not the corrected anchor embeddings. It embeds every
    # patient at 26 grid points around onset and has to put all of them through
    # the SAME correction -- so the map is fitted once, on the anchor rows, and
    # applied to the grid.
    #
    # Fitting on the grid instead would be wrong twice over: each patient
    # appears up to 26 times, so the fit would be weighted by whose grid is
    # complete, and the map would change whenever the grid did, which would
    # make two runs of 09 incomparable for a reason that has nothing to do with
    # the patients.
    #
    # npz rather than json: `transform` is 1024x1024 float64, which is 8 MB of
    # numbers that json would store as decimal text.

    def save(self, path: pathlib.Path) -> None:
        """write the fitted map so another script can apply the identical one"""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"centre": self.centre}
        if self.transform is not None:
            arrays["transform"] = self.transform
        if self.per_site is not None:
            # one pair of arrays per site, with the site name in the key. The
            # names are recovered from the keys on load, so a site absent from
            # the fit stays absent rather than being silently given the pooled
            # centre by a schema that could not represent it.
            for site, (mean, scale) in self.per_site.items():
                arrays[f"per_site::{site}::mean"] = mean
                arrays[f"per_site::{site}::scale"] = scale
        np.savez(path, name=np.array(self.name), **arrays)

    @classmethod
    def load(cls, path: pathlib.Path) -> "AffineCorrection":
        """read a map written by `save`"""
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            per_site = None
            sites = sorted(
                {k.split("::")[1] for k in keys if k.startswith("per_site::")}
            )
            if sites:
                per_site = {
                    site: (
                        data[f"per_site::{site}::mean"],
                        data[f"per_site::{site}::scale"],
                    )
                    for site in sites
                }
            return cls(
                name=str(data["name"]),
                centre=data["centre"],
                transform=data["transform"] if "transform" in keys else None,
                per_site=per_site,
            )


def fit_identity(x: np.ndarray) -> AffineCorrection:
    """no-op, so "raw" can travel through the same code path as the corrections"""
    return AffineCorrection(
        name="raw", centre=np.zeros(x.shape[1]), transform=np.eye(x.shape[1])
    )


def fit_site_standardize(
    x: np.ndarray, sites: np.ndarray, eps: float = 1e-8
) -> AffineCorrection:
    """
    z-score within each site

    The crude baseline: it removes any difference in per-feature location and
    scale between sites, and nothing else. Any site structure that lives in the
    correlations between features survives it, which is most of the point of
    comparing it against LEACE.
    """
    per_site = {}
    for site in np.unique(sites):
        part = x[sites == site]
        per_site[site] = (part.mean(axis=0), part.std(axis=0) + eps)
    return AffineCorrection(
        name="site-standardized", centre=x.mean(axis=0), per_site=per_site
    )


def fit_leace(x: np.ndarray, z: np.ndarray, eps: float = 1e-6) -> AffineCorrection:
    """
    LEACE: least-squares concept erasure (Belrose et al., 2023)

    Returns the affine map that removes all linearly-decodable information
    about `z` while changing the representation as little as possible in the
    least-squares sense. The erased subspace has the rank of `z`, found in the
    whitened space so that the deletion is orthogonal with respect to the
    data's own covariance rather than to the raw axes. For three sites `z` must
    be the rank-2 one-hot -- see `site_indicator`, which is where getting this
    wrong cost us a correction that erased one contrast of three.

        r(x) = x - Sxx^(1/2) u u' Sxx^(-1/2) (x - mu),  u = unit(Sxx^(-1/2) Sxz)

    Guarantee: after this, Cov(r(X), z) = 0, so no linear probe can recover the
    site above chance. Non-linear probes can still succeed, and routinely do.
    """
    n, d = x.shape
    centre = x.mean(axis=0)
    xc = x - centre
    zc = z.astype(np.float64)
    zc = zc - zc.mean(axis=0)
    if zc.ndim == 1:
        zc = zc[:, None]

    covariance = (xc.T @ xc) / (n - 1)
    cross = (xc.T @ zc) / (n - 1)

    # symmetric whitening, with the spectrum floored so a rank-deficient
    # covariance cannot blow up the inverse square root
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values, eps, None)
    whiten = vectors @ np.diag(values**-0.5) @ vectors.T
    unwhiten = vectors @ np.diag(values**0.5) @ vectors.T

    basis = whiten @ cross
    q, _ = np.linalg.qr(basis)
    projection = q @ q.T

    transform = np.eye(d) - unwhiten @ projection @ whiten
    return AffineCorrection(name="leace", centre=centre, transform=transform)


def site_indicator(sites: np.ndarray, order) -> np.ndarray:
    """
    the concept matrix for "which site is this", as one-hot minus one column

    A k-level categorical spans k-1 dimensions, and LEACE erases exactly the
    rank of the concept it is given. Handing it a single binary column
    `sites == reference` therefore erased ONE contrast -- the reference site
    against the other two pooled -- and left the contrast between those two
    fully intact. Measured on the cardiogenic grid, with the map fitted on the
    grid so that fitting is not the confound:

        contrast          uncorrected   rank-1 binary   rank-2 one-hot
        UCMC vs rest           0.9985          0.5106           0.5119
        NU vs rest             0.9975          0.8875           0.5039
        RUSH vs rest           0.9894          0.9901           0.4782
        NU vs RUSH             0.9970          0.9968           0.4659
        3-way accuracy         0.9688          0.7670           0.5340

    The last column is chance: the 3-way majority share is 0.5486, so 0.5340 is
    no better than always guessing the largest site. The rank-1 column is not a
    weaker correction, it is a correction of one third of the problem.

    The dropped column is redundant rather than discarded -- the k columns sum
    to one, so k-1 of them span the same subspace and the QR basis comes out
    rank k-1 either way. Dropping it keeps the concept matrix full rank, which
    is what the whitened cross-covariance wants.
    """
    present = [s for s in order if (sites == s).any()]
    if len(present) < 2:
        raise ValueError("need two sites to correct between")
    return np.column_stack(
        [(sites == s).astype(np.float64) for s in present[:-1]]
    )


def fit(
    method: str,
    x: np.ndarray,
    sites: np.ndarray,
    reference: str,
    order=None,
) -> AffineCorrection:
    """
    dispatch by name

    `reference` is the site treated as the positive class, which only
    `site-standardized` still uses to pick its target moments; LEACE erases
    every site contrast at once and so has no privileged site. `order` fixes
    which level is dropped from the one-hot, so the same data always yields the
    same map; it defaults to the sorted levels present.
    """
    if method == "raw":
        return fit_identity(x)
    if method == "site-standardized":
        return fit_site_standardize(x, sites)
    if method == "leace":
        levels = order if order is not None else sorted(set(np.unique(sites)))
        return fit_leace(x, site_indicator(sites, levels))
    raise ValueError(f"unknown correction {method!r}")


METHODS = ("raw", "site-standardized", "leace")

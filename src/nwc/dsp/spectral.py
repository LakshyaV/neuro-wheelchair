"""Cross-spectral frames and the geometry of covariance matrices.

One analysis window is reduced to its multitaper cross-spectrum, summed into
1 Hz bins and stored as the real part of each channel pair. That single array
answers every later question:

* the covariance of the signal in any frequency band is a weighted sum over bins,
  so band layouts are chosen (and compared) after the fact, without filtering again;
* the diagonal is band power, the off-diagonal terms are how channels move together;
* broadband amplitude, the measure of electrode contact, is a sum over bins too,
  and the bins around the power line are simply left out of every sum.

Multitaper estimation (Slepian tapers) uses the whole window with nearly uniform
weight and suppresses leakage from the large slow components of dry-electrode EEG.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from scipy.signal import windows

Band = tuple[float, float]


# ------------------------------------------------------------------ packing
def pair_index(n_channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of every channel pair ``i <= j``, row-major."""
    return np.triu_indices(n_channels)


def n_pairs(n_channels: int) -> int:
    return n_channels * (n_channels + 1) // 2


def channels_from_pairs(pairs: int) -> int:
    n = round((np.sqrt(8 * pairs + 1) - 1) / 2)
    if n_pairs(n) != pairs:
        raise ValueError(f"{pairs} is not a number of channel pairs")
    return n


def pack(matrices: np.ndarray) -> np.ndarray:
    """Symmetric ``[..., C, C]`` to the upper triangle ``[..., P]``."""
    rows, cols = pair_index(matrices.shape[-1])
    return matrices[..., rows, cols]


def unpack(packed: np.ndarray) -> np.ndarray:
    """Upper triangle ``[..., P]`` back to symmetric ``[..., C, C]``."""
    n = channels_from_pairs(packed.shape[-1])
    rows, cols = pair_index(n)
    out = np.empty((*packed.shape[:-1], n, n), dtype=packed.dtype)
    out[..., rows, cols] = packed
    out[..., cols, rows] = packed
    return out


# ------------------------------------------------------------------ estimation
class Multitaper:
    """Cross-spectrum of one window, real part, in 1 Hz bins.

    Bin ``f`` covers ``[f, f + 1)`` Hz. Summing the diagonal over all bins gives the
    variance of the window, so band sums are covariances in physical units.
    """

    def __init__(self, n_samples: int, fs: float, time_bandwidth: float = 3.0, n_tapers: int = 5):
        self.n_samples = int(n_samples)
        self.fs = float(fs)
        self.n_bins = int(self.fs // 2)
        self.tapers = np.asarray(windows.dpss(self.n_samples, time_bandwidth, n_tapers), dtype=np.float64)
        freqs = np.fft.rfftfreq(self.n_samples, 1.0 / self.fs)
        bins = np.floor(freqs + 1e-9).astype(np.int64)
        used = bins < self.n_bins
        if len(np.unique(bins[used])) != self.n_bins:
            raise ValueError("the analysis window must be at least one second long to resolve 1 Hz bins")
        weights = np.full(len(freqs), 2.0)
        weights[0] = 1.0
        if self.n_samples % 2 == 0:
            weights[-1] = 1.0
        # One-sided scaling per FFT bin, and where each 1 Hz bin starts among the FFT bins.
        self._weights = (weights / (self.n_samples * len(self.tapers)))[used]
        self._used = int(used.sum())
        self._starts = np.searchsorted(bins[used], np.arange(self.n_bins))

    def __call__(self, window: np.ndarray) -> np.ndarray:
        """``[N, C]`` to ``[F, P]``."""
        return self.batch(window[None])[0]

    def batch(self, stacked: np.ndarray) -> np.ndarray:
        """``[n, N, C]`` to ``[n, F, P]``.

        Every window is reduced with the same sequence of floating-point operations
        whatever the batch size, so frames do not depend on how the stream was chunked.
        """
        centred = stacked - stacked.mean(axis=1, keepdims=True)
        rows, cols = pair_index(stacked.shape[-1])
        out = np.empty((len(stacked), self.n_bins, len(rows)))
        for i, window in enumerate(centred):
            spectra = np.fft.rfft(self.tapers[:, :, None] * window[None], axis=1)[:, : self._used]  # [K, f, C]
            cross = (spectra[..., rows] * spectra[..., cols].conj()).real.sum(axis=0)  # [f, P]
            out[i] = np.add.reduceat(cross * self._weights[:, None], self._starts, axis=0)
        return out


# ------------------------------------------------------------------ bands
def uniform_bands(low: float, high: float, width: float) -> tuple[Band, ...]:
    edges = np.arange(low, high + 1e-9, width)
    return tuple((float(a), float(b)) for a, b in itertools.pairwise(edges))


def band_weights(bands: Sequence[Band], n_bins: int, exclude: Sequence[Band] = ()) -> np.ndarray:
    """``[B, F]`` weights that turn binned spectra into band sums.

    A bin contributes in proportion to its overlap with the band. Bins that touch an
    excluded range (mains pickup) contribute nothing.
    """
    lower = np.arange(n_bins, dtype=np.float64)
    upper = lower + 1.0
    blocked = np.zeros(n_bins, dtype=bool)
    for lo, hi in exclude:
        blocked |= (upper > lo) & (lower < hi)
    out = np.zeros((len(bands), n_bins))
    for b, (lo, hi) in enumerate(bands):
        if not 0 <= lo < hi:
            raise ValueError(f"invalid band {(lo, hi)}")
        out[b] = np.clip(np.minimum(upper, hi) - np.maximum(lower, lo), 0.0, 1.0)
        out[b, blocked] = 0.0
        if not out[b].any():
            raise ValueError(f"band {(lo, hi)} is empty after excluding {tuple(exclude)}")
    return out


#: Power-line frequencies of the world's grids. Both are always treated as present, so nothing
#: has to be set for the country the chair is in, and nothing can be set wrongly.
MAINS_HZ = (50.0, 60.0)


def mains_harmonics(low: float, high: float) -> tuple[float, ...]:
    """Every multiple of a power-line frequency inside ``(low, high)``, ascending."""
    found = {k * base for base in MAINS_HZ for k in range(1, int(high // base) + 1) if low < k * base < high}
    return tuple(sorted(found))


def mains_exclusion(n_bins: int = 128, half_width: float = 2.0) -> tuple[Band, ...]:
    """Power-line frequencies and their harmonics below ``n_bins`` Hz, each with a guard band."""
    return tuple((f - half_width, f + half_width) for f in mains_harmonics(0.0, float(n_bins)))


def band_covariances(cosp: np.ndarray, weights: np.ndarray, shrink: float = 1e-3) -> np.ndarray:
    """Binned cross-spectra ``[..., F, P]`` to band covariances ``[..., B, C, C]``.

    ``shrink`` pulls each matrix slightly towards its own diagonal, which keeps it
    positive definite when a narrow band leaves few degrees of freedom. Shrinking
    towards the diagonal, not towards a multiple of the identity, commutes with a
    change of channel gains, so features taken relative to a baseline stay exactly
    independent of how well each electrode happens to sit.
    """
    packed = np.einsum("bf,...fp->...bp", weights, cosp)
    covs = unpack(packed)
    n = covs.shape[-1]
    diagonal = np.einsum("...cc->...c", covs)
    covs = (1.0 - shrink) * covs
    covs[..., np.arange(n), np.arange(n)] += shrink * diagonal
    return covs


# ------------------------------------------------------------------ geometry
def _spectral_map(matrices: np.ndarray, function) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrices)
    values = np.maximum(values, np.finfo(np.float64).tiny)
    return (vectors * function(values)[..., None, :]) @ np.swapaxes(vectors, -1, -2)


def logm(matrices: np.ndarray) -> np.ndarray:
    return _spectral_map(matrices, np.log)


def expm(matrices: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrices)
    return (vectors * np.exp(values)[..., None, :]) @ np.swapaxes(vectors, -1, -2)


def invsqrtm(matrices: np.ndarray) -> np.ndarray:
    return _spectral_map(matrices, lambda v: 1.0 / np.sqrt(v))


def whiten(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Express ``covs`` relative to ``reference``, band by band, so that the reference becomes the identity.

    Each channel is first divided by its level in the reference, and the
    correlation that remains between channels is then removed. Any whitening
    yields the same distances; this one also yields the same coordinates when an
    electrode's gain changes, because gains cancel in the first step. Contact
    quality, and with it gain, is what differs most between two sittings with
    dry electrodes.
    """
    level = 1.0 / np.sqrt(np.diagonal(reference, axis1=-2, axis2=-1))
    scaled = covs * level[..., :, None] * level[..., None, :]
    correlation = reference * level[..., :, None] * level[..., None, :]
    root = invsqrtm(correlation)
    return root @ scaled @ root


def tangent_features(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Tangent-space vectors at ``reference``: ``[n, B, C, C]`` to ``[n, B * P]``.

    The matrix logarithm of the whitened covariance is the direction from the
    reference to the observation under the affine-invariant metric. Off-diagonal
    terms are weighted by sqrt(2) so the Euclidean norm of the vector equals the
    Riemannian one.
    """
    logs = logm(whiten(covs, reference))
    rows, cols = pair_index(covs.shape[-1])
    weights = np.where(rows == cols, 1.0, np.sqrt(2.0))
    return (logs[..., rows, cols] * weights).reshape(len(covs), -1)


def power_features(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Log band power per channel relative to the reference: ``[n, B * C]``."""
    power = np.log(np.diagonal(covs, axis1=-2, axis2=-1)) - np.log(np.diagonal(reference, axis1=-2, axis2=-1))
    return power.reshape(len(covs), -1)


def riemann_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Affine-invariant distance between covariance matrices (broadcasts)."""
    values = np.linalg.eigvalsh(whiten(a, b))
    return np.sqrt((np.log(np.maximum(values, np.finfo(np.float64).tiny)) ** 2).sum(axis=-1))

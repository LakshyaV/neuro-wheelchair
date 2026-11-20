"""From cross-spectral frames to feature vectors.

Features are always relative to a reference: the session's baseline. A band
covariance ``C`` becomes ``R^-1/2 C R^-1/2`` with ``R`` the baseline covariance
of the same band, so the vector describes how the brain's rhythms have changed
since the wearer sat relaxed a few minutes ago, not how well the electrodes
happen to sit today. On 30 people of public research EEG, expressing each person
relative to their own reference took a model that had never seen them from 79 %
to 92 % (eyes closed against open, Muse-like electrode positions).

Two kinds of vector are built from the whitened covariances:

``power``    log band power per channel: the diagonal only
``tangent``  the full matrix logarithm: power plus how channels co-vary

Which serves a given wearer better is decided by cross-validation on their data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsp import spectral
from ..dsp.spectral import Band

BAND_LAYOUTS: dict[str, tuple[Band, ...]] = {
    # 2 Hz resolution: resolves narrow effects; the default.
    "fine": spectral.uniform_bands(1.0, 45.0, 2.0),
    # 4 Hz resolution: half the features, for short calibrations.
    "medium": spectral.uniform_bands(1.0, 45.0, 4.0),
    # The textbook rhythms.
    "classic": ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 45.0)),
}


@dataclass(frozen=True)
class FeatureSpec:
    kind: str = "power"
    layout: str = "fine"
    bands: tuple[Band, ...] = ()
    shrink: float = 1e-3

    def __post_init__(self) -> None:
        if self.kind not in ("power", "tangent"):
            raise ValueError(f"feature kind must be 'power' or 'tangent', not {self.kind!r}")
        if not self.bands and self.layout not in BAND_LAYOUTS:
            raise ValueError(f"unknown band layout {self.layout!r}; choose from {sorted(BAND_LAYOUTS)} or give bands")

    @property
    def resolved_bands(self) -> tuple[Band, ...]:
        return tuple(tuple(b) for b in self.bands) if self.bands else BAND_LAYOUTS[self.layout]  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return f"{self.kind}/{self.layout if not self.bands else f'{len(self.bands)} bands'}"


class Featurizer:
    """Applies a :class:`FeatureSpec` to frames. Stateless apart from precomputed band weights."""

    def __init__(self, spec: FeatureSpec, n_bins: int, n_channels: int):
        self.spec = spec
        self.n_channels = n_channels
        exclude = spectral.mains_exclusion(n_bins)
        self._weights = spectral.band_weights(spec.resolved_bands, n_bins, exclude)

    @property
    def size(self) -> int:
        per_band = self.n_channels if self.spec.kind == "power" else spectral.n_pairs(self.n_channels)
        return len(self._weights) * per_band

    @property
    def names(self) -> list[str]:
        rows, cols = spectral.pair_index(self.n_channels)
        out = []
        for lo, hi in self.spec.resolved_bands:
            if self.spec.kind == "power":
                out += [f"{lo:g}-{hi:g}Hz/ch{c}" for c in range(self.n_channels)]
            else:
                out += [f"{lo:g}-{hi:g}Hz/ch{r}xch{c}" for r, c in zip(rows, cols)]
        return out

    def covariances(self, cosp: np.ndarray) -> np.ndarray:
        return spectral.band_covariances(np.asarray(cosp, dtype=np.float64), self._weights, self.spec.shrink)

    def __call__(self, cosp: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Frames ``[n, F, P]`` and a reference ``[F, P]`` to vectors ``[n, D]``."""
        covs = self.covariances(cosp)
        ref = self.covariances(reference[None])[0]
        if self.spec.kind == "power":
            return spectral.power_features(covs, ref)
        return spectral.tangent_features(covs, ref)


def mean_reference(cosp: np.ndarray) -> np.ndarray:
    """The reference of a stretch of frames: their mean cross-spectrum ``[F, P]``.

    Averaging spectra (rather than taking a geometric mean of covariances) keeps the
    reference independent of the band layout, so it is computed once per session.
    """
    if len(cosp) == 0:
        raise ValueError("no frames to build a reference from")
    return np.asarray(cosp, dtype=np.float64).mean(axis=0)

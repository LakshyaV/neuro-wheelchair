"""Signal quality: which frames can be trusted, and whether an electrode has lost contact.

The limits are in microvolts and were set from real Muse recordings. In good
contact the 1-45 Hz amplitude of a channel has a median of 4 to 8 uV over two
seconds; a forehead electrode that has lifted off reads 70 uV and more, for as
long as it stays off. Single frames above the limit are common in good
recordings as well (blinks, a head movement), so two things are kept apart:

* an *artefact* is a frame that is too large or swings too far at low frequency.
  It carries no usable evidence and is skipped;
* *lost contact* is a channel that has been out of range for most of the last few
  seconds. That is a fault, and the supervisor stops the chair on it.

Power-line pickup is not used as a sign of contact. In real recordings that were
otherwise equally good the line stood anywhere from 16 to 56 dB above its
surroundings, depending on the room. It is kept out of every measure instead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .frontend import Frames, FrontendConfig
from .spectral import band_weights, mains_exclusion, pair_index


@dataclass(frozen=True)
class QualityConfig:
    rms_min_uv: float = 0.7
    rms_max_uv: float = 60.0
    swing_max_uv: float = 150.0
    band_hz: tuple[float, float] = (1.0, 45.0)
    lost_after_s: float = 3.0
    lost_share: float = 0.75


@dataclass
class FrameQuality:
    rms: np.ndarray  # [n, C] uV, 1-45 Hz
    out_of_range: np.ndarray  # [n, C] bool
    artefact: np.ndarray  # [n] bool
    usable: np.ndarray  # [n] bool: received in full, settled and free of artefacts

    def __len__(self) -> int:
        return len(self.usable)


class QualityMeter:
    """Turns frames into quality figures; the weights are built once."""

    def __init__(self, frontend: FrontendConfig, config: QualityConfig | None = None):
        self.config = config or QualityConfig()
        self.frontend = frontend
        rows, cols = pair_index(frontend.n_channels)
        self._diagonal = np.flatnonzero(rows == cols)
        exclude = mains_exclusion(frontend.n_bins)
        self._broad = band_weights((self.config.band_hz,), frontend.n_bins, exclude)[0]

    def __call__(self, frames: Frames) -> FrameQuality:
        cfg = self.config
        power = frames.cosp[:, :, self._diagonal].astype(np.float64)  # [n, F, C]
        rms = np.sqrt(np.maximum(np.einsum("nfc,f->nc", power, self._broad), 0.0))
        out_of_range = (rms < cfg.rms_min_uv) | (rms > cfg.rms_max_uv)
        artefact = out_of_range.any(axis=1) | (frames.swing > cfg.swing_max_uv).any(axis=1)
        return FrameQuality(rms, out_of_range, artefact, frames.valid & ~artefact)


class ContactMonitor:
    """Tracks, per channel, whether it has been out of range for most of the recent past."""

    def __init__(self, frontend: FrontendConfig, config: QualityConfig | None = None):
        self.config = config or QualityConfig()
        self.channels = frontend.channels
        self._length = max(1, round(self.config.lost_after_s * frontend.frame_rate))
        self._history: deque[np.ndarray] = deque(maxlen=self._length)
        self.lost = np.zeros(len(self.channels), dtype=bool)

    def push(self, quality: FrameQuality, valid: np.ndarray) -> np.ndarray:
        """Feed frames in order; returns the per-channel lost-contact flags after the last one.

        Frames that were not received in full say nothing about the electrodes and are skipped.
        """
        for flags, ok in zip(quality.out_of_range, valid):
            if not ok:
                continue
            self._history.append(flags)
            if len(self._history) == self._length:
                self.lost = np.mean(self._history, axis=0) >= self.config.lost_share
            else:
                self.lost = np.zeros(len(self.channels), dtype=bool)
        return self.lost

    @property
    def lost_channels(self) -> list[str]:
        return [name for name, flag in zip(self.channels, self.lost) if flag]

    def reset(self) -> None:
        self._history.clear()
        self.lost = np.zeros(len(self.channels), dtype=bool)

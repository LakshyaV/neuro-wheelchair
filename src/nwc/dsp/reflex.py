"""The stop reflex: a jaw clench, detected from muscle activity at the temporal electrodes.

A clench drives the jaw muscles under the TP9 and TP10 electrodes and raises the
30-110 Hz power there far above its resting level within a tenth of a second. It
is the clearest deliberate signal a Muse can pick up, so it is reserved for the
command where a miss matters most: stop.

The detector is deliberately plain. Power on both sides must exceed the resting
level by ``on_db`` for ``hold`` consecutive fast frames; it releases when either
side falls below ``off_db`` for ``release`` frames. The resting level is measured
during the baseline of every session. On 33 minutes of real Muse recordings in
which nobody was asked to clench, the detector fired once at this threshold.
A false trigger costs a stop and nothing else. How far a wearer's clench stands
above their rest is measured from their calibration clenches (:func:`fit_threshold`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frontend import Envelope


@dataclass(frozen=True)
class ClenchConfig:
    channels: tuple[str, ...] = ("TP9", "TP10")
    on_db: float = 15.0
    off_db: float = 8.0
    hold: int = 3
    release: int = 4


@dataclass(frozen=True)
class ClenchEvent:
    frame: int  # fast-frame index at which the state changed
    pressed: bool


def resting_level(envelope: Envelope) -> np.ndarray:
    """Median muscle-band level per channel (dB) over the valid frames of a quiet stretch."""
    level = envelope.level_db[envelope.valid]
    if len(level) == 0:
        raise ValueError("no valid frames to measure the resting muscle level from")
    return np.median(level, axis=0)


class ClenchDetector:
    def __init__(self, all_channels: tuple[str, ...], resting_db: np.ndarray, config: ClenchConfig | None = None):
        self.config = config or ClenchConfig()
        missing = [c for c in self.config.channels if c not in all_channels]
        if missing:
            raise ValueError(f"clench channels {missing} are not in the stream {all_channels}")
        self._columns = [all_channels.index(c) for c in self.config.channels]
        self.resting_db = np.asarray(resting_db, dtype=np.float64)
        self.pressed = False
        self._above = 0
        self._below = 0
        self.excess_db = 0.0  # latest level above rest on the weaker side

    def push(self, envelope: Envelope) -> list[ClenchEvent]:
        """Feed fast frames in order; returns the state changes they caused."""
        if len(envelope) == 0:
            return []
        cfg = self.config
        excess = (envelope.level_db - self.resting_db)[:, self._columns].min(axis=1)
        events: list[ClenchEvent] = []
        for frame, value, valid in zip(envelope.index, excess, envelope.valid):
            if not valid:
                continue  # samples were missing: no evidence either way
            self.excess_db = float(value)
            if not self.pressed:
                self._above = self._above + 1 if value >= cfg.on_db else 0
                if self._above >= cfg.hold:
                    self.pressed, self._above, self._below = True, 0, 0
                    events.append(ClenchEvent(int(frame), True))
            else:
                self._below = self._below + 1 if value < cfg.off_db else 0
                if self._below >= cfg.release:
                    self.pressed, self._above, self._below = False, 0, 0
                    events.append(ClenchEvent(int(frame), False))
        return events


def fit_threshold(
    resting_excess_db: np.ndarray, clench_excess_db: np.ndarray, config: ClenchConfig | None = None
) -> tuple[ClenchConfig, dict[str, float]]:
    """Place ``on_db`` between a person's resting excursions and their calibration clenches.

    Both inputs are per-frame levels above rest on the weaker side. The threshold
    sits halfway (in dB) between the largest resting excursions and a weak clench,
    never below 10 dB, and the release level follows at just over half of it.
    """
    config = config or ClenchConfig()
    rest_top = float(np.percentile(resting_excess_db, 99.9))
    clench_low = float(np.percentile(clench_excess_db, 20))
    margin = clench_low - rest_top
    on_db = float(np.clip((rest_top + clench_low) / 2.0, 10.0, 30.0)) if margin > 0 else config.on_db
    fitted = ClenchConfig(config.channels, on_db, max(on_db * 0.55, 5.0), config.hold, config.release)
    return fitted, {"rest_p99_9_db": rest_top, "clench_p20_db": clench_low, "margin_db": margin, "on_db": on_db}

"""From a stream of class probabilities to one steady command.

A classifier's opinion changes four times a second, and on a single two-second
window an honest classifier is not very sure of anything. A wheelchair must be
sure before it moves. The arbiter stands between them:

* evidence accumulates. The log-probabilities of successive frames are added up
  with a leak, so a state that is held for a few seconds builds up to a confident
  posterior while an isolated odd frame fades away. ``integration_s`` is the
  memory of the accumulator and ``gain`` discounts each frame for the samples it
  shares with its neighbours;
* a command switches on only after its posterior has stayed above ``on`` for
  ``dwell_s``, and off only after it has stayed below ``off`` for ``release_s``.
  The gap between the two thresholds keeps a command from flickering;
* commands never hand over to each other directly. The way from one command to
  another leads through rest, and a pause follows every command;
* frames without usable signal are bridged, briefly. One blink spoils two
  seconds of analysis windows, and on real recordings one window in five is
  spoiled by something, so a command that ended on every such stretch would be
  useless. Evidence is therefore held unchanged for up to ``hold_s``. Beyond
  that it drains towards rest and the command ends. Unusable frames can never
  start a command, and they prolong one by ``hold_s`` at most.

The numbers are not guessed: :mod:`nwc.control.tune` replays held-out calibration
runs through this class and picks them for a target false-start rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REST = "rest"


@dataclass(frozen=True)
class ArbiterConfig:
    integration_s: float = 2.0
    gain: float = 0.5
    on: float = 0.8
    off: float = 0.5
    dwell_s: float = 0.5
    release_s: float = 0.5
    refractory_s: float = 1.0
    hold_s: float = 2.5  # how long evidence is kept unchanged while no usable frame arrives

    def __post_init__(self) -> None:
        if not 0.0 < self.off < self.on <= 1.0:
            raise ValueError("need 0 < off < on <= 1")
        if self.integration_s <= 0 or self.gain <= 0:
            raise ValueError("integration_s and gain must be positive")


class Arbiter:
    def __init__(self, classes: tuple[str, ...], config: ArbiterConfig | None = None, frame_rate: float = 4.0):
        if classes[0] != REST:
            raise ValueError(f"the first class must be {REST!r}, got {classes}")
        self.classes = tuple(classes)
        self.config = config or ArbiterConfig()
        self.frame_rate = float(frame_rate)
        cfg = self.config
        self._keep = float(np.exp(-1.0 / (cfg.integration_s * frame_rate)))
        self._dwell = max(1, round(cfg.dwell_s * frame_rate))
        self._release = max(1, round(cfg.release_s * frame_rate))
        self._refractory = round(cfg.refractory_s * frame_rate)
        self._hold = round(cfg.hold_s * frame_rate)
        self._drain = np.full(len(classes), -2.0)  # log-odds against every command, per frame, once the hold is over
        self._drain[0] = 0.0
        self.reset()

    def reset(self) -> None:
        """Forget everything: evidence returns to neutral and no command is active."""
        self._log_evidence = np.zeros(len(self.classes))
        self.posterior = np.full(len(self.classes), 1.0 / len(self.classes))
        self.active = REST
        self._candidate = 0
        self._above = 0
        self._below = 0
        self._wait = 0
        self._unusable = 0

    def push(self, log_proba: np.ndarray | None) -> str:
        """Feed the log-probabilities of one frame (None: no usable signal); returns the active command."""
        cfg = self.config
        if log_proba is None:
            self._unusable += 1
            if self._unusable > self._hold:
                self._log_evidence = self._keep * self._log_evidence + self._drain
            # within the hold nothing changes: neither the evidence nor the counters below advance
            elif self.active != REST or self._wait == 0:
                return self.active
        else:
            self._unusable = 0
            self._log_evidence = self._keep * self._log_evidence + cfg.gain * np.asarray(log_proba, dtype=np.float64)
        shifted = self._log_evidence - self._log_evidence.max()
        self.posterior = np.exp(shifted) / np.exp(shifted).sum()
        if self.active != REST:
            k = self.classes.index(self.active)
            self._below = self._below + 1 if self.posterior[k] < cfg.off else 0
            if self._below >= self._release:
                self.active, self._below, self._above, self._wait = REST, 0, 0, self._refractory
            return self.active
        if self._wait > 0:
            self._wait -= 1
            return self.active
        k = 1 + int(np.argmax(self.posterior[1:])) if len(self.classes) > 1 else 0
        if k and self.posterior[k] >= cfg.on:
            self._above = self._above + 1 if k == self._candidate else 1
            self._candidate = k
            if self._above >= self._dwell:
                self.active, self._above, self._below = self.classes[k], 0, 0
        else:
            self._above = 0
        return self.active


def replay(
    classes: tuple[str, ...], log_proba: np.ndarray, usable: np.ndarray, config: ArbiterConfig, frame_rate: float
) -> np.ndarray:
    """Run consecutive frames through a fresh arbiter; returns the active class index per frame."""
    arbiter = Arbiter(classes, config, frame_rate)
    out = np.zeros(len(log_proba), dtype=np.int64)
    for i in range(len(log_proba)):
        out[i] = classes.index(arbiter.push(log_proba[i] if usable[i] else None))
    return out

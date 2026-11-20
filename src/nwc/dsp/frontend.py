"""The streaming front-end: samples in, analysis frames out.

One implementation serves recording analysis, training, evaluation and live
driving. It is causal and chunk-invariant: feeding a recording sample by sample,
in packets of 12 or in one piece produces identical frames. A model can therefore
only ever see features computed the way it was trained on, and
:attr:`FrontendConfig.fingerprint` is stored with every model to enforce it.

Two frame rates come out:

* slow frames, one per ``hop_s``, each summarising the last ``window_s`` as a
  cross-spectrum (:mod:`nwc.dsp.spectral`) plus the swing of the low-frequency
  path, which is where eye and movement artefacts live;
* fast frames, one per ``emg_hop_s``, with the power of the muscle band. A jaw
  clench has to stop a wheelchair in a fraction of a second, long before a
  two-second window has filled with it.

A slow frame is ``valid`` if the filters had settled after the last break in the
stream and at most ``max_filled`` of its samples had to be interpolated. One
lost Bluetooth packet is 2 % of a two-second window and barely moves a spectrum;
demanding complete windows would discard nine in ten of them at 5 % packet loss.
A fast frame is three packets long, so it is valid only if it is complete.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..config import fingerprint
from ..io.stream import Block, StreamInfo
from .spectral import Multitaper, mains_harmonics, n_pairs


@dataclass(frozen=True)
class FrontendConfig:
    fs: float = 256.0
    channels: tuple[str, ...] = ("TP9", "AF7", "AF8", "TP10")
    highpass_hz: float = 1.0
    window_s: float = 2.0
    hop_s: float = 0.25
    time_bandwidth: float = 3.0
    tapers: int = 5
    slow_hz: float = 15.0
    emg_band_hz: tuple[float, float] = (30.0, 110.0)
    emg_window_s: float = 0.125
    emg_hop_s: float = 0.0625
    settle_s: float = 1.0
    max_filled: float = 0.1

    def __post_init__(self) -> None:
        if self.emg_band_hz[1] >= self.fs / 2:
            raise ValueError(f"muscle band {self.emg_band_hz} must lie below the Nyquist frequency {self.fs / 2:g} Hz")
        if self.hop_s <= 0 or self.window_s < self.hop_s:
            raise ValueError("need 0 < hop_s <= window_s")

    @property
    def window(self) -> int:
        return round(self.window_s * self.fs)

    @property
    def hop(self) -> int:
        return round(self.hop_s * self.fs)

    @property
    def emg_window(self) -> int:
        return round(self.emg_window_s * self.fs)

    @property
    def emg_hop(self) -> int:
        return round(self.emg_hop_s * self.fs)

    @property
    def n_bins(self) -> int:
        return int(self.fs // 2)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def frame_rate(self) -> float:
        return self.fs / self.hop

    @property
    def fingerprint(self) -> str:
        return fingerprint(self)

    def frame_time(self, index: np.ndarray | int) -> np.ndarray | float:
        """Seconds since the start of the stream at which a slow frame's window ends."""
        return (np.asarray(index) * self.hop + self.window) / self.fs

    def frame_span(self, index: int) -> tuple[int, int]:
        """First and one-past-last sample of a slow frame's window."""
        return index * self.hop, index * self.hop + self.window

    def matches(self, info: StreamInfo) -> None:
        if tuple(info.channels) != tuple(self.channels):
            raise ValueError(f"stream channels {info.channels} do not match the front-end's {self.channels}")
        if abs(info.fs - self.fs) > 1e-6:
            raise ValueError(f"stream sample rate {info.fs:g} Hz does not match the front-end's {self.fs:g} Hz")


@dataclass
class Frames:
    """Slow frames. Frame ``k`` covers samples ``[k * hop, k * hop + window)``."""

    index: np.ndarray  # [n] int64
    cosp: np.ndarray  # [n, F, P] float32, uV^2 per 1 Hz bin and channel pair
    swing: np.ndarray  # [n, C] float32, peak-to-peak of the low-frequency path, uV
    valid: np.ndarray  # [n] bool

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item) -> Frames:
        return Frames(self.index[item], self.cosp[item], self.swing[item], self.valid[item])

    @staticmethod
    def empty(config: FrontendConfig) -> Frames:
        return Frames(
            np.zeros(0, dtype=np.int64),
            np.zeros((0, config.n_bins, n_pairs(config.n_channels)), dtype=np.float32),
            np.zeros((0, config.n_channels), dtype=np.float32),
            np.zeros(0, dtype=bool),
        )

    @staticmethod
    def concatenate(parts: list[Frames], config: FrontendConfig) -> Frames:
        parts = [p for p in parts if len(p)]
        if not parts:
            return Frames.empty(config)
        return Frames(
            *(np.concatenate([getattr(p, name) for p in parts]) for name in ("index", "cosp", "swing", "valid"))
        )


@dataclass
class Envelope:
    """Fast frames. Frame ``j`` covers samples ``[j * emg_hop, j * emg_hop + emg_window)``."""

    index: np.ndarray  # [m] int64
    power: np.ndarray  # [m, C] float32, mean square of the muscle band, uV^2
    valid: np.ndarray  # [m] bool

    def __len__(self) -> int:
        return len(self.index)

    @property
    def level_db(self) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(self.power, 1e-6))

    @staticmethod
    def empty(config: FrontendConfig) -> Envelope:
        return Envelope(
            np.zeros(0, dtype=np.int64), np.zeros((0, config.n_channels), dtype=np.float32), np.zeros(0, dtype=bool)
        )

    @staticmethod
    def concatenate(parts: list[Envelope], config: FrontendConfig) -> Envelope:
        parts = [p for p in parts if len(p)]
        if not parts:
            return Envelope.empty(config)
        return Envelope(*(np.concatenate([getattr(p, name) for p in parts]) for name in ("index", "power", "valid")))


class _Filter:
    """A causal second-order-section filter that keeps its state between calls."""

    def __init__(self, sos: np.ndarray, n_channels: int):
        self.sos = np.atleast_2d(sos)
        self._steady = signal.sosfilt_zi(self.sos)  # [sections, 2]
        self._state = np.zeros((len(self.sos), 2, n_channels))

    def reset(self, first_sample: np.ndarray) -> None:
        """Start as if the input had been at ``first_sample`` forever: no step transient."""
        self._state = self._steady[:, :, None] * first_sample[None, None, :]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self._state = signal.sosfilt(self.sos, x, axis=0, zi=self._state)
        return y


#: Width of the notch at every power-line harmonic, wide enough for a grid that runs a little off.
NOTCH_WIDTH_HZ = 6.0


def _muscle_band(config: FrontendConfig) -> np.ndarray:
    """The muscle band with the power line taken out.

    Both the 50 Hz and the 60 Hz family are removed whatever the country. A line
    left in this band raises the resting level, on real recordings by 13 dB in
    the median and by up to 33 dB, and a jaw clench then stands out that much
    less: the stop reflex must not depend on a setting somebody has to remember.
    """
    low, high = config.emg_band_hz
    sections = [signal.butter(4, config.emg_band_hz, "bandpass", fs=config.fs, output="sos")]
    for harmonic in mains_harmonics(low - 5.0, min(high + 15.0, config.fs / 2 - 2.0)):
        sections.append(signal.tf2sos(*signal.iirnotch(harmonic, harmonic / NOTCH_WIDTH_HZ, fs=config.fs)))
    return np.vstack(sections)


class Frontend:
    """Stateful front-end for one stream."""

    #: Frames per FFT batch when processing a long block at once.
    BATCH = 256

    def __init__(self, config: FrontendConfig | None = None):
        self.config = config or FrontendConfig()
        cfg = self.config
        n = cfg.n_channels
        self._highpass = _Filter(signal.butter(2, cfg.highpass_hz, "highpass", fs=cfg.fs, output="sos"), n)
        self._lowpass = _Filter(signal.butter(2, cfg.slow_hz, "lowpass", fs=cfg.fs, output="sos"), n)
        self._muscle = _Filter(_muscle_band(cfg), n)
        self._taper = Multitaper(cfg.window, cfg.fs, cfg.time_bandwidth, cfg.tapers)
        self._settle = round(cfg.settle_s * cfg.fs)
        self._keep = max(cfg.window, cfg.emg_window)
        self._base = 0  # stream index of the first sample held in the buffers
        self._count = 0  # samples consumed
        self._started = False
        self._bad_until = 0  # samples before this index are still settling
        self._broad = np.zeros((0, n))
        self._slow = np.zeros((0, n))
        self._square = np.zeros((0, n))
        self._bad = np.zeros(0, dtype=bool)  # samples within the settling time after a break
        self._filled = np.zeros(0, dtype=bool)  # samples that were interpolated
        self._max_filled = int(cfg.max_filled * cfg.window)
        self._next_frame = 0
        self._next_fast = 0

    @property
    def count(self) -> int:
        return self._count

    def push(self, block: Block) -> tuple[Frames, Envelope]:
        """Consume a block of assembled samples; return the frames it completes."""
        cfg = self.config
        if len(block) == 0:
            return Frames.empty(cfg), Envelope.empty(cfg)
        if block.start != self._count:
            raise ValueError(f"block starts at sample {block.start}, expected {self._count}")
        x = block.x
        filled = block.filled
        bad = self._count + np.arange(len(x)) < self._bad_until  # still settling from an earlier block
        edges = [0, *[int(b) for b in block.breaks if 0 < b < len(x)], len(x)]
        restart_first = (not self._started) or (len(block.breaks) > 0 and int(block.breaks[0]) == 0)
        broad, slow, square = [], [], []
        for piece, (a, b) in enumerate(itertools.pairwise(edges)):
            if piece > 0 or restart_first:
                # Continuity is lost here: restart the filters at rest on the new level and
                # distrust everything until they have settled.
                self._highpass.reset(x[a])
                self._lowpass.reset(np.zeros(cfg.n_channels))
                self._muscle.reset(x[a])
                bad[a : a + self._settle] = True
                self._bad_until = self._count + a + self._settle
            part = self._highpass(x[a:b])
            broad.append(part)
            slow.append(self._lowpass(part))
            square.append(self._muscle(x[a:b]) ** 2)
        self._started = True

        self._broad = np.concatenate((self._broad, *broad))
        self._slow = np.concatenate((self._slow, *slow))
        self._square = np.concatenate((self._square, *square))
        self._bad = np.concatenate((self._bad, bad))
        self._filled = np.concatenate((self._filled, filled))
        self._count += len(x)

        frames = self._slow_frames()
        envelope = self._fast_frames()
        self._trim()
        return frames, envelope

    # ------------------------------------------------------------------ frames
    def _slow_frames(self) -> Frames:
        cfg = self.config
        last = (self._count - cfg.window) // cfg.hop  # highest complete frame
        if last < self._next_frame:
            return Frames.empty(cfg)
        index = np.arange(self._next_frame, last + 1, dtype=np.int64)
        starts = index * cfg.hop - self._base
        cosp = np.empty((len(index), cfg.n_bins, n_pairs(cfg.n_channels)), dtype=np.float32)
        swing = np.empty((len(index), cfg.n_channels), dtype=np.float32)
        valid = np.empty(len(index), dtype=bool)
        offsets = np.arange(cfg.window)
        for a in range(0, len(index), self.BATCH):
            rows = starts[a : a + self.BATCH, None] + offsets[None, :]
            cosp[a : a + self.BATCH] = self._taper.batch(self._broad[rows])
            slow = self._slow[rows]
            swing[a : a + self.BATCH] = slow.max(axis=1) - slow.min(axis=1)
            valid[a : a + self.BATCH] = ~self._bad[rows].any(axis=1) & (
                self._filled[rows].sum(axis=1) <= self._max_filled
            )
        self._next_frame = last + 1
        return Frames(index, cosp, swing, valid)

    def _fast_frames(self) -> Envelope:
        cfg = self.config
        last = (self._count - cfg.emg_window) // cfg.emg_hop
        if last < self._next_fast:
            return Envelope.empty(cfg)
        index = np.arange(self._next_fast, last + 1, dtype=np.int64)
        rows = (index * cfg.emg_hop - self._base)[:, None] + np.arange(cfg.emg_window)[None, :]
        power = self._square[rows].mean(axis=1).astype(np.float32)
        valid = ~(self._bad[rows] | self._filled[rows]).any(axis=1)
        self._next_fast = last + 1
        return Envelope(index, power, valid)

    def _trim(self) -> None:
        # Every frame still to come starts within the last ``_keep`` samples.
        drop = self._count - self._keep - self._base
        if drop > 0:
            self._broad = self._broad[drop:]
            self._slow = self._slow[drop:]
            self._square = self._square[drop:]
            self._bad = self._bad[drop:]
            self._filled = self._filled[drop:]
            self._base += drop


def process(block: Block, config: FrontendConfig | None = None) -> tuple[Frames, Envelope]:
    """Run a complete assembled recording through a fresh front-end."""
    return Frontend(config).push(block)

"""Synthetic Muse EEG.

The simulator exists so that every part of the system can run, and be tested,
without a headset: recording, training, evaluation, the stop reflex, the safety
supervisor and the motor link. Its levels are matched to real Muse recordings
(amplitude per channel, the muscle-band floor, blink size, packet timing), and
it reproduces the things that make real sessions hard: blinks, incidental muscle
activity, electrode shifts, mains pickup, slow drift, session-to-session changes
in gain and rhythm strength, jittered packet arrival and packet loss.

What it cannot tell anyone is how separable a real person's mental commands are.
Each command changes the strength of a few rhythm generators by a few decibels,
scaled by ``effect``; that is a stand-in chosen to exercise the decoder, not a
model of the brain. Figures obtained on simulated sessions describe the software.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class Generator:
    """A rhythm generator: band-limited noise that reaches the electrodes with fixed weights."""

    name: str
    band: tuple[float, float]
    projection: tuple[float, ...]  # TP9, AF7, AF8, TP10
    amplitude: float  # uV RMS at the electrode with weight 1


GENERATORS = (
    Generator("alpha_posterior", (8.0, 12.0), (1.0, 0.3, 0.3, 1.0), 4.0),
    Generator("alpha_left", (8.0, 12.0), (1.0, 0.4, 0.0, 0.1), 2.2),
    Generator("alpha_right", (8.0, 12.0), (0.1, 0.0, 0.4, 1.0), 2.2),
    Generator("theta_frontal", (4.0, 7.0), (0.2, 1.0, 1.0, 0.2), 2.4),
    Generator("beta_left", (14.0, 24.0), (0.8, 0.8, 0.1, 0.1), 1.5),
    Generator("beta_right", (14.0, 24.0), (0.1, 0.1, 0.8, 0.8), 1.5),
    Generator("beta_frontal", (16.0, 28.0), (0.1, 1.0, 1.0, 0.1), 1.2),
    Generator("gamma_frontal", (30.0, 42.0), (0.1, 0.7, 0.7, 0.1), 0.6),
)

#: Change in generator strength (dB) while a command is held, at ``effect = 1``.
SIGNATURES: dict[str, dict[str, float]] = {
    "rest": {},
    "forward": {
        "alpha_posterior": -4.0,
        "alpha_left": -2.0,
        "alpha_right": -2.0,
        "theta_frontal": 3.0,
        "beta_frontal": 3.0,
        "gamma_frontal": 2.0,
    },
    "left": {"alpha_right": -5.0, "beta_right": -2.5, "beta_left": 2.0},
    "right": {"alpha_left": -5.0, "beta_left": -2.5, "beta_right": 2.0},
    "backward": {"alpha_posterior": 5.0, "beta_frontal": -2.5, "theta_frontal": -1.5},
}


@dataclass(frozen=True)
class SimConfig:
    fs: float = 256.0
    channels: tuple[str, ...] = ("TP9", "AF7", "AF8", "TP10")
    mains_hz: float = 60.0
    effect: float = 1.0
    trial_spread: float = 0.35  # log-normal spread of the effect from trial to trial
    onset_s: float = 0.5  # time constant with which a command takes hold and fades
    background_uv: tuple[float, ...] = (4.7, 3.0, 3.0, 4.7)  # 1/f background, RMS in 1-45 Hz
    sensor_uv: tuple[float, ...] = (5.4, 2.6, 4.2, 4.0)  # white sensor noise, RMS over the full band
    blink_rate_hz: float = 0.25
    blink_uv: float = 90.0
    clench_uv: tuple[float, ...] = (70.0, 18.0, 18.0, 70.0)
    twitch_rate_hz: float = 0.02  # swallowing, talking: brief, weaker, often one-sided muscle activity
    shift_rate_hz: float = 0.01  # an electrode moving on the skin
    mains_uv: float = 3.0
    drift_db: float = 0.5  # slow wander of each generator's strength within a session
    session_spread_db: float = 1.5  # differences between sessions (re-seating the headband)
    gain_spread: float = 0.15
    lifted_uv: float = 200.0  # noise on an electrode that has lost contact, RMS over the full band
    lsb_uv: float = 0.48828125

    @property
    def n_channels(self) -> int:
        return len(self.channels)


def _band_rms(sos: np.ndarray, fs: float, low: float = 0.0, high: float | None = None) -> float:
    """RMS of unit-variance white noise after ``sos``, optionally within a band."""
    freqs, response = signal.sosfreqz(sos, worN=8192, fs=fs)
    power = np.abs(response) ** 2
    mask = (freqs >= low) & (freqs <= (high if high is not None else fs / 2))
    return float(np.sqrt(power[mask].sum() / len(freqs)))


class _Shaped:
    """White noise through a stateful filter, scaled to a wanted RMS."""

    def __init__(self, sos: np.ndarray, columns: int, gain: float):
        self.sos = sos
        self.gain = gain
        self._state = np.zeros((len(sos), 2, columns))

    def __call__(self, white: np.ndarray) -> np.ndarray:
        out, self._state = signal.sosfilt(self.sos, white, axis=0, zi=self._state)
        return out * self.gain


#: A three-pole filter that turns white noise into 1/f noise (the pink-noise design given by
#: J. O. Smith in "Spectral Audio Signal Processing").
_PINK = signal.tf2sos(
    [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1.0, -2.494956002, 2.017265875, -0.522189400]
)


@dataclass
class _Session:
    """What differs from one sitting to the next."""

    gain: np.ndarray
    generator_db: np.ndarray
    background_db: np.ndarray
    sensor_db: np.ndarray
    mains_uv: np.ndarray
    mains_phase: np.ndarray
    offset_uv: np.ndarray


@dataclass
class EEGSynth:
    """A stateful generator of Muse-like EEG. Produces the same samples however it is chunked."""

    config: SimConfig = field(default_factory=SimConfig)
    seed: int = 0
    session_seed: int | None = None

    def __post_init__(self) -> None:
        cfg = self.config
        c, g = cfg.n_channels, len(GENERATORS)
        streams = np.random.SeedSequence([self.seed, 0xEE6]).spawn(6)
        self._noise, self._trial, self._blink, self._twitch, self._shift, self._drift = (
            np.random.default_rng(s) for s in streams
        )
        s_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed if self.session_seed is None else self.session_seed, 0x5E55])
        )
        self._session = _Session(
            gain=np.exp(s_rng.normal(0.0, cfg.gain_spread, c)),
            generator_db=s_rng.normal(0.0, cfg.session_spread_db, g),
            background_db=s_rng.normal(0.0, 1.0, c),
            sensor_db=s_rng.normal(0.0, 1.0, c),
            mains_uv=np.clip(cfg.mains_uv * np.exp(s_rng.normal(0.0, 1.0, c)), 0.3, 40.0),
            mains_phase=s_rng.uniform(0, 2 * np.pi, c),
            offset_uv=s_rng.uniform(-300.0, 300.0, c),
        )
        self._projection = np.array([gen.projection for gen in GENERATORS]).T  # [C, G]
        self._rhythms = []
        for gen in GENERATORS:
            sos = signal.butter(2, gen.band, "bandpass", fs=cfg.fs, output="sos")
            self._rhythms.append(_Shaped(sos, 1, gen.amplitude / _band_rms(sos, cfg.fs)))
        self._background = _Shaped(_PINK, c, 1.0 / _band_rms(_PINK, cfg.fs, 1.0, 45.0))
        emg_sos = signal.butter(2, 20.0, "highpass", fs=cfg.fs, output="sos")
        self._emg = _Shaped(emg_sos, c, 1.0 / _band_rms(emg_sos, cfg.fs))
        self._twitch_emg = _Shaped(emg_sos, c, 1.0 / _band_rms(emg_sos, cfg.fs))
        drift_sos = signal.butter(1, 1.0 / 60.0, "lowpass", fs=cfg.fs, output="sos")
        self._wander = _Shaped(drift_sos, g, cfg.drift_db / max(_band_rms(drift_sos, cfg.fs), 1e-9))
        slow_sos = signal.butter(1, 0.05, "lowpass", fs=cfg.fs, output="sos")
        self._slow = _Shaped(slow_sos, c, 20.0 / max(_band_rms(slow_sos, cfg.fs), 1e-9))

        self._n = 0  # samples generated
        self._target_db = np.zeros(g)
        self._level_db = np.zeros(g)
        self._clench_target = 0.0
        self._clench_level = 0.0
        self._clench_scale = 1.0
        self._contact = np.ones(c)
        self._overlay = np.zeros((0, c))  # transient waveforms that extend past the current call
        self._twitch_gain = np.zeros((0, c))
        self._next = {
            "blink": self._interval(self._blink, cfg.blink_rate_hz),
            "twitch": self._interval(self._twitch, cfg.twitch_rate_hz),
            "shift": self._interval(self._shift, cfg.shift_rate_hz),
        }
        self.intent = "rest"

    # ------------------------------------------------------------------ control
    def _interval(self, rng: np.random.Generator, rate_hz: float) -> int:
        if rate_hz <= 0:
            return np.iinfo(np.int64).max
        return self._n_or_zero() + 1 + int(rng.exponential(1.0 / rate_hz) * self.config.fs)

    def _n_or_zero(self) -> int:
        return getattr(self, "_n", 0)

    def set_intent(self, label: str | None, strength: float | None = None) -> None:
        """The wearer starts holding ``label`` (None or "rest": lets go) from the next sample."""
        label = label or "rest"
        if label not in SIGNATURES:
            raise ValueError(f"the simulator has no signature for {label!r}; known: {sorted(SIGNATURES)}")
        if strength is None:
            strength = float(np.exp(self._trial.normal(0.0, self.config.trial_spread)))
        target = np.zeros(len(GENERATORS))
        for name, change in SIGNATURES[label].items():
            target[[gen.name for gen in GENERATORS].index(name)] = change * self.config.effect * strength
        self._target_db = target
        self.intent = label

    def set_clench(self, on: bool) -> None:
        if on and self._clench_target == 0.0:
            self._clench_scale = float(np.exp(self._trial.normal(0.0, 0.25)))
        self._clench_target = 1.0 if on else 0.0

    def set_contact(self, channel: str, ok: bool) -> None:
        """Lift an electrode off the skin (or put it back)."""
        self._contact[self.config.channels.index(channel)] = 1.0 if ok else 0.0

    @property
    def position(self) -> int:
        return self._n

    # ------------------------------------------------------------------ transients
    def _add(self, buffer: np.ndarray, start: int, wave: np.ndarray) -> np.ndarray:
        end = start + len(wave)
        if end > len(buffer):
            buffer = np.concatenate((buffer, np.zeros((end - len(buffer), buffer.shape[1]))))
        buffer[start:end] += wave
        return buffer

    def _schedule(self, n: int) -> None:
        cfg, fs = self.config, self.config.fs
        stop = self._n + n
        while self._next["blink"] < stop:
            at = self._next["blink"] - self._n
            amplitude = float(
                np.clip(self._blink.normal(cfg.blink_uv, 0.28 * cfg.blink_uv), 0.4 * cfg.blink_uv, 2.0 * cfg.blink_uv)
            )
            t = np.arange(int(0.5 * fs)) / fs - 0.15
            bump = amplitude * np.exp(-0.5 * (t / np.where(t < 0, 0.05, 0.09)) ** 2)
            # The forehead electrodes see the eyelid; the reference at Fpz carries a little of it to the others.
            self._overlay = self._add(
                self._overlay, at, bump[:, None] * np.array([-0.15, 1.0, 1.0, -0.15])[: cfg.n_channels]
            )
            self._next["blink"] += 1 + int(self._blink.exponential(1.0 / cfg.blink_rate_hz) * fs)
        while self._next["shift"] < stop:
            at = self._next["shift"] - self._n
            channel = int(self._shift.integers(cfg.n_channels))
            step = float(self._shift.uniform(150.0, 400.0) * self._shift.choice((-1.0, 1.0)))
            wave = np.zeros((int(1.5 * fs), cfg.n_channels))
            wave[:, channel] = step * np.exp(-np.arange(len(wave)) / (0.3 * fs))
            self._overlay = self._add(self._overlay, at, wave)
            self._next["shift"] += 1 + int(self._shift.exponential(1.0 / cfg.shift_rate_hz) * fs)
        while self._next["twitch"] < stop:
            at = self._next["twitch"] - self._n
            length = int(self._twitch.uniform(0.3, 0.9) * fs)
            level = self._twitch.uniform(10.0, 25.0)
            sides = np.array([1.0, 0.3, 0.3, 1.0])[: cfg.n_channels]
            if self._twitch.random() < 0.5:  # one-sided: must not trigger the stop reflex
                sides = (
                    sides
                    * (
                        np.array([1.0, 1.0, 0.1, 0.05])
                        if self._twitch.random() < 0.5
                        else np.array([0.05, 0.1, 1.0, 1.0])
                    )[: cfg.n_channels]
                )
            envelope = np.hanning(length)[:, None] * level * sides
            self._twitch_gain = self._add(self._twitch_gain, at, envelope)
            self._next["twitch"] += 1 + int(self._twitch.exponential(1.0 / cfg.twitch_rate_hz) * fs)

    @staticmethod
    def _take(buffer: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        head = buffer[:n]
        if len(head) < n:
            head = np.concatenate((head, np.zeros((n - len(head), buffer.shape[1]))))
        return head, buffer[n:]

    # ------------------------------------------------------------------ samples
    def generate(self, n: int) -> np.ndarray:
        """The next ``n`` samples, ``[n, C]`` in microvolts as the headset would report them."""
        cfg, session = self.config, self._session
        c, g = cfg.n_channels, len(GENERATORS)
        if n <= 0:
            return np.zeros((0, c))
        white = self._noise.standard_normal((n, g + 4 * c))
        drift = self._drift.standard_normal((n, g + c))
        self._schedule(n)

        # Generator strengths follow the commanded change with a first-order lag.
        k = 1.0 - np.exp(-1.0 / (cfg.onset_s * cfg.fs))
        steps = np.arange(1, n + 1)[:, None]
        level_db = self._target_db + (self._level_db - self._target_db) * (1.0 - k) ** steps
        self._level_db = level_db[-1]
        strength = 10.0 ** ((level_db + session.generator_db + self._wander(drift[:, :g])) / 20.0)  # [n, G]
        rhythms = np.column_stack([shape(white[:, [i]])[:, 0] for i, shape in enumerate(self._rhythms)]) * strength
        x = rhythms @ self._projection.T

        x += (
            self._background(white[:, g : g + c])
            * np.asarray(cfg.background_uv)
            * 10.0 ** (session.background_db / 20.0)
        )
        x *= session.gain

        clench_k = 1.0 - np.exp(-1.0 / (0.04 * cfg.fs))
        clench = self._clench_target + (self._clench_level - self._clench_target) * (1.0 - clench_k) ** steps[:, 0]
        self._clench_level = float(clench[-1])
        x += self._emg(white[:, g + c : g + 2 * c]) * clench[:, None] * np.asarray(cfg.clench_uv) * self._clench_scale
        twitch, self._twitch_gain = self._take(self._twitch_gain, n)
        x += self._twitch_emg(white[:, g + 2 * c : g + 3 * c]) * twitch
        overlay, self._overlay = self._take(self._overlay, n)
        x += overlay

        # An electrode that has lifted off stops seeing the head and picks up noise instead. On real
        # recordings such a channel reads 70 to 150 uV in the 1-45 Hz band for as long as it is off.
        off = 1.0 - self._contact
        sensor = white[:, g + 3 * c :] * (
            np.asarray(cfg.sensor_uv) * 10.0 ** (session.sensor_db / 20.0) * self._contact + cfg.lifted_uv * off
        )
        x = x * self._contact + sensor

        t = (self._n + np.arange(n)) / cfg.fs
        if cfg.mains_hz:
            x += (
                session.mains_uv
                * (1.0 + 8.0 * off)
                * np.sin(2 * np.pi * cfg.mains_hz * t[:, None] + session.mains_phase)
            )
        x += session.offset_uv + self._slow(drift[:, g:])
        self._n += n
        return np.clip(np.rint(x / cfg.lsb_uv) * cfg.lsb_uv, -1000.0, 1000.0)

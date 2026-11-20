"""The simulator as a stream source, including the Bluetooth transport.

Packets of twelve samples leave the virtual headset on a regular clock and reach
the host with a variable delay. Samples are stamped the way older stream bridges
stamp them, backwards from the moment their packet arrived, so the timestamps
show the same jumps at packet boundaries that real recordings do. Packets can
be lost, and the device's sample counter can be included or withheld.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from ..io.sources import Source
from ..io.stream import Chunk, StreamInfo
from .eeg import EEGSynth, SimConfig


class Transport:
    """Turns device samples into packets with arrival times, jitter and loss."""

    def __init__(
        self,
        fs: float,
        *,
        packet: int = 12,
        latency_ms: float = 40.0,
        jitter_ms: float = 8.0,
        loss: float = 0.0,
        counted: bool = False,
        seed: int = 0,
        start: float = 1_700_000_000.0,
    ):
        self.fs, self.packet = float(fs), int(packet)
        self.latency, self.jitter, self.loss, self.counted = latency_ms / 1e3, jitter_ms / 1e3, float(loss), counted
        self.start = float(start)
        self._rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7A]))
        self._sent = 0  # device samples packed so far
        self._last_arrival = -np.inf
        self._pending = np.zeros((0, 0))

    def push(self, samples: np.ndarray) -> list[tuple[float, int, Chunk]]:
        """Device samples in; ``(arrival time, first device sample, chunk)`` for every packet that survives."""
        if self._pending.size:
            samples = np.concatenate((self._pending, samples))
        whole = len(samples) // self.packet
        out: list[tuple[float, int, Chunk]] = []
        for k in range(whole):
            first = self._sent
            self._sent += self.packet
            delay = self.latency + (self._rng.exponential(self.jitter) if self.jitter > 0 else 0.0)
            arrival = max(self.start + self._sent / self.fs + delay, self._last_arrival + 1e-4)
            lost = first > 0 and self._rng.random() < self.loss
            if lost:
                continue
            self._last_arrival = arrival
            stamps = arrival - (self.packet - 1 - np.arange(self.packet)) / self.fs
            seq = first + np.arange(self.packet, dtype=np.int64) if self.counted else None
            out.append((arrival, first, Chunk(stamps, samples[k * self.packet : (k + 1) * self.packet], seq)))
        self._pending = samples[whole * self.packet :]
        return out

    @property
    def sent(self) -> int:
        return self._sent


class SimulatedSource(Source):
    """A simulated headset. Fast by default; ``realtime=True`` paces it like the real thing."""

    def __init__(
        self,
        synth: EEGSynth | None = None,
        *,
        realtime: bool = False,
        duration_s: float | None = None,
        jitter_ms: float = 8.0,
        loss: float = 0.0,
        counted: bool = False,
        seed: int = 0,
    ):
        self.synth = synth or EEGSynth(seed=seed)
        cfg = self.synth.config
        self.info = StreamInfo(cfg.channels, cfg.fs, source="sim", device="simulated Muse")
        self._transport = Transport(
            cfg.fs,
            jitter_ms=jitter_ms,
            loss=loss,
            counted=counted,
            seed=seed,
            start=time.time() if realtime else 1_700_000_000.0,
        )
        self._realtime = realtime
        self._limit = None if duration_s is None else round(duration_s * cfg.fs)
        self._wall_start: float | None = None
        self._queue: list[tuple[float, int, Chunk]] = []
        self._script: list[tuple[int, str, object]] = []

    @classmethod
    def from_spec(
        cls, rest: str = "", *, channels: Sequence[str] = (), fs: float = 256.0, realtime: bool = False, **options
    ) -> SimulatedSource:
        seed = int(rest) if rest else 0
        config = SimConfig(
            fs=fs, channels=tuple(channels) or SimConfig().channels, effect=float(options.get("effect", 1.0))
        )
        return cls(
            EEGSynth(config, seed=seed),
            realtime=realtime,
            seed=seed,
            duration_s=options.get("duration_s"),
            loss=float(options.get("loss", 0.0)),
            counted=bool(options.get("counted", False)),
        )

    # -- what the virtual wearer does
    def at(self, seconds: float, action: str, value: object = None) -> SimulatedSource:
        """Schedule ``intent`` (a label, or ``(label, strength)``), ``clench`` (bool) or ``contact`` (``(channel, ok)``)."""
        self._script.append((round(seconds * self.synth.config.fs), action, value))
        self._script.sort(key=lambda item: item[0])
        return self

    def _apply(self, action: str, value: object) -> None:
        if action == "intent":  # a label, or (label, strength) to fix how clearly it shows
            if isinstance(value, tuple):
                self.synth.set_intent(*value)
            else:
                self.synth.set_intent(value)  # type: ignore[arg-type]
        elif action == "clench":
            self.synth.set_clench(bool(value))
        elif action == "contact":
            channel, ok = value  # type: ignore[misc]
            self.synth.set_contact(channel, ok)
        else:
            raise ValueError(f"unknown scripted action {action!r}")

    @property
    def exhausted(self) -> bool:  # type: ignore[override]
        return self._limit is not None and self.synth.position >= self._limit and not self._queue

    def _produce(self, n: int) -> None:
        """Generate ``n`` device samples, honouring the script at exact sample positions."""
        while n > 0:
            position = self.synth.position
            while self._script and self._script[0][0] <= position:
                _, action, value = self._script.pop(0)
                self._apply(action, value)
            step = n if not self._script else min(n, self._script[0][0] - position)
            self._queue.extend(self._transport.push(self.synth.generate(step)))
            n -= step

    def read(self, timeout: float = 0.05) -> Chunk | None:
        fs = self.synth.config.fs
        if self._realtime:
            if self._wall_start is None:
                self._wall_start = time.monotonic()
            due = round((time.monotonic() - self._wall_start) * fs)
            if self._limit is not None:
                due = min(due, self._limit)
            if due > self.synth.position:
                self._produce(due - self.synth.position)
            now = time.time()
            ready = [chunk for arrival, _, chunk in self._queue if arrival <= now]
            self._queue = [item for item in self._queue if item[0] > now]
            if not ready:
                time.sleep(min(timeout, 12 / fs))
                return None
        else:
            if not self._queue:
                remaining = 12 if self._limit is None else min(12, self._limit - self.synth.position)
                if remaining > 0:
                    self._produce(remaining)
            ready = [chunk for _, _, chunk in self._queue[:1]]
            self._queue = self._queue[1:]
            if not ready:
                return None
        t = np.concatenate([c.t for c in ready])
        x = np.concatenate([c.x for c in ready])
        seq = None if ready[0].seq is None else np.concatenate([c.seq for c in ready])
        return Chunk(t, x, seq)

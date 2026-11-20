"""Stream assembly: from samples as they arrive to a uniform sample grid.

A headset samples on its own crystal and ships packets over Bluetooth. Two
things follow, and both were measured on real Muse recordings before this was
written:

* Timestamps are not a sample clock. Some stream bridges stamp samples by the
  arrival of their packet: in real Muse recordings made that way, consecutive
  stamps differ by anything from -15 to +33 ms and three packet boundaries in
  ten are off by a sample or more, while the samples themselves are perfectly
  regular. Other bridges regularise the stamps, which hides lost packets
  altogether. Placing samples by their stamps would damage a healthy stream in
  the first case and miss a damaged one in the second, so it is never done.
* Loss is real but only visible in the device's own sample counter. Where a
  transport carries one (``Chunk.seq``), holes are located exactly: a short hole
  is interpolated and flagged, a long one is a break. Where it does not, samples
  are taken as consecutive and only a stall long enough to be unmistakable is
  treated as a break.

Health is measured by counting: samples received over time elapsed. A typical
interval between samples says nothing about the ones that never came.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StreamInfo:
    """What a source delivers."""

    channels: tuple[str, ...]
    fs: float
    source: str = ""
    device: str = ""
    unit: str = "uV"

    @property
    def n_channels(self) -> int:
        return len(self.channels)


@dataclass
class Chunk:
    """Samples as a source hands them over.

    ``t`` is in seconds on the source's clock and is treated as the time of
    arrival. ``seq`` is the device's running sample number when the transport
    carries one.
    """

    t: np.ndarray
    x: np.ndarray
    seq: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.t)


@dataclass
class Block:
    """Consecutive samples on the uniform grid.

    ``start`` is the grid index of the first sample and ``t`` is grid time: the first
    arrival plus index over sample rate, free of transport jitter. ``row`` gives, for
    every grid sample, its position in the order of arrival, or -1 for an
    interpolated one. ``breaks`` lists positions in this block where continuity was
    lost; the sample at such a position is the first one after the hole.
    """

    start: int
    t: np.ndarray
    x: np.ndarray
    row: np.ndarray
    breaks: np.ndarray

    def __len__(self) -> int:
        return len(self.t)

    @property
    def stop(self) -> int:
        return self.start + len(self.t)

    @property
    def filled(self) -> np.ndarray:
        return self.row < 0


@dataclass
class StreamStats:
    received: int = 0
    filled: int = 0
    breaks: int = 0
    lost_in_breaks: int = 0
    stalled_s: float = 0.0  # time during which nothing arrived (uncounted streams)
    first_t: float | None = None
    last_t: float | None = None
    nominal_hz: float = 0.0
    counted: bool = False  # True when the transport carries the device's sample counter

    @property
    def elapsed(self) -> float:
        if self.first_t is None or self.last_t is None:
            return 0.0
        return self.last_t - self.first_t

    @property
    def delivered_hz(self) -> float:
        """Samples that arrived per second while the stream was flowing."""
        flowing = self.elapsed - self.stalled_s
        return (self.received - 1) / flowing if self.received > 1 and flowing > 0 else 0.0

    @property
    def loss(self) -> float:
        """Share of the device's samples that did not arrive.

        Exact when the stream is counted. Otherwise it is the shortfall of the
        delivered rate against the nominal one while the stream was flowing, which
        needs tens of seconds to mean anything; stalls are reported separately.
        """
        if self.counted:
            missing = self.filled + self.lost_in_breaks
            total = self.received + missing
            return missing / total if total else 0.0
        if not self.nominal_hz or self.received < 2:
            return 0.0
        return float(max(0.0, 1.0 - self.delivered_hz / self.nominal_hz))

    def as_dict(self) -> dict[str, float]:
        return {
            "received": self.received,
            "delivered_hz": round(self.delivered_hz, 3),
            "loss": round(self.loss, 5),
            "filled": self.filled,
            "breaks": self.breaks,
            "stalled_s": round(self.stalled_s, 3),
            "elapsed_s": round(self.elapsed, 3),
            "counted": self.counted,
        }


class StreamAssembler:
    """Places an arriving stream on the grid ``k / fs``."""

    def __init__(self, info: StreamInfo, max_fill_s: float = 0.25, stall_s: float = 0.25):
        self.info = info
        self.fs = float(info.fs)
        self.max_fill = round(max_fill_s * self.fs)
        self.stall_s = float(stall_s)
        self.stats = StreamStats(nominal_hz=self.fs)
        self._count = 0  # grid samples produced
        self._rows = 0  # samples received
        self._last_t: float | None = None
        self._last_x: np.ndarray | None = None
        self._last_seq: int | None = None

    @property
    def count(self) -> int:
        """Samples placed on the grid so far."""
        return self._count

    def _empty(self) -> Block:
        return Block(
            self._count,
            np.zeros(0),
            np.zeros((0, self.info.n_channels)),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    def push(self, chunk: Chunk | None) -> Block:
        if chunk is None or len(chunk) == 0:
            return self._empty()
        t = np.asarray(chunk.t, dtype=np.float64)
        x = np.asarray(chunk.x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape != (len(t), self.info.n_channels):
            raise ValueError(f"expected samples of shape (n, {self.info.n_channels}), got {x.shape}")
        rows = self._rows + np.arange(len(t), dtype=np.int64)

        if chunk.seq is not None:
            block = self._counted(t, x, rows, np.asarray(chunk.seq, dtype=np.int64))
        else:
            block = self._uncounted(t, x, rows)

        self._count += len(block)
        self._rows += len(t)
        self._last_t, self._last_x = float(t[-1]), x[-1].copy()
        self.stats.received += len(t)
        if self.stats.first_t is None:
            self.stats.first_t = float(t[0])
        self.stats.last_t = max(float(t.max()), self.stats.last_t or -np.inf)
        return block

    # ------------------------------------------------------------------ without a counter
    def _uncounted(self, t: np.ndarray, x: np.ndarray, rows: np.ndarray) -> Block:
        previous = np.concatenate(([t[0] if self._last_t is None else self._last_t], t[:-1]))
        stalls = np.flatnonzero(t - previous > self.stall_s)
        for i in stalls:
            self.stats.breaks += 1
            self.stats.stalled_s += float(t[i] - previous[i]) - 1.0 / self.fs
            self.stats.lost_in_breaks += max(round(float(t[i] - previous[i]) * self.fs) - 1, 0)
        # Samples are consecutive by definition here.
        return Block(self._count, self._grid_time(t, len(t)), x, rows, stalls.astype(np.int64))

    def _grid_time(self, t: np.ndarray, n: int) -> np.ndarray:
        anchor = float(t[0]) if self.stats.first_t is None else self.stats.first_t
        return anchor + (self._count + np.arange(n)) / self.fs

    # ------------------------------------------------------------------ with the device's counter
    def _counted(self, t: np.ndarray, x: np.ndarray, rows: np.ndarray, seq: np.ndarray) -> Block:
        self.stats.counted = True
        previous = np.concatenate(([seq[0] - 1 if self._last_seq is None else self._last_seq], seq[:-1]))
        step = seq - previous  # 1 is the normal case; more means samples are missing; less, a restart
        out_x: list[np.ndarray] = []
        out_row: list[np.ndarray] = []
        breaks: list[int] = []
        produced = 0
        cursor = 0
        last_x = self._last_x
        for i in np.flatnonzero(step != 1):
            if i > cursor:
                out_x.append(x[cursor:i])
                out_row.append(rows[cursor:i])
                produced += i - cursor
                last_x = x[i - 1]
            missing = int(step[i]) - 1
            if last_x is not None and 0 < missing <= self.max_fill:
                weights = np.arange(1, missing + 1) / (missing + 1)
                out_x.append(last_x + weights[:, None] * (x[i] - last_x))
                out_row.append(np.full(missing, -1, dtype=np.int64))
                produced += missing
                self.stats.filled += missing
            else:
                breaks.append(produced)
                self.stats.breaks += 1
                self.stats.lost_in_breaks += max(missing, 0)
            cursor = i
        out_x.append(x[cursor:])
        out_row.append(rows[cursor:])
        self._last_seq = int(seq[-1])
        samples = np.concatenate(out_x)
        return Block(
            self._count,
            self._grid_time(t, len(samples)),
            samples,
            np.concatenate(out_row),
            np.asarray(breaks, dtype=np.int64),
        )


def assemble(
    t: np.ndarray, x: np.ndarray, info: StreamInfo, seq: np.ndarray | None = None
) -> tuple[Block, StreamStats]:
    """Assemble a complete recording in one call."""
    assembler = StreamAssembler(info)
    block = assembler.push(Chunk(t, x, seq))
    return block, assembler.stats


def grid_of_rows(block: Block) -> np.ndarray:
    """Grid index of every received sample, in order of arrival."""
    return block.start + np.flatnonzero(block.row >= 0)

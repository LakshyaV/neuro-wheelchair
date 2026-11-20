"""Recorded sessions on disk.

A session is a directory::

    data/<subject>/<session>/
        session.json    device, channels, sample rate, protocol
        eeg.csv         samples as received: timestamps,<channel>,...[,seq]
        events.jsonl    one JSON object per line: baseline, cues, pauses, runs

``eeg.csv`` uses the column layout of muselsl recordings, so files written here
open in other Muse tools and recordings made with them can be imported. The
optional last column ``seq`` is the device's sample counter, kept when the
transport provides it so that lost samples can be located again later.

Samples are stored exactly as they arrived. Assembly onto the sample grid is
repeated at processing time and is therefore always the current implementation.
Events refer to samples by their position in the order of arrival (``start`` and
``stop`` rows), never by clock time: packet timestamps jitter, positions do not.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from .stream import Block, Chunk, StreamInfo, StreamStats, assemble, grid_of_rows

FORMAT = 1
SAMPLES = "eeg.csv"
EVENTS = "events.jsonl"
MANIFEST = "session.json"

#: Event kinds written by the recorder.
BASELINE, CUE, PAUSE, RUN, NOTE = "baseline", "cue", "pause", "run", "note"


@dataclass
class Event:
    """An interval of a recording: rows ``[start, stop)`` in order of arrival."""

    kind: str
    start: int
    stop: int
    label: str = ""
    run: int = 0
    trial: int = -1
    extra: dict[str, Any] = field(default_factory=dict)


class SessionWriter:
    """Append-only writer. Samples reach the disk as they arrive, so a crash loses seconds, not the session."""

    def __init__(
        self,
        directory: str | Path,
        info: StreamInfo,
        *,
        subject: str,
        session: str,
        meta: dict[str, Any] | None = None,
    ):
        self.path = Path(directory)
        if (self.path / SAMPLES).exists():
            raise FileExistsError(f"{self.path} already holds a recording")
        self.path.mkdir(parents=True, exist_ok=True)
        self.info = info
        self._manifest: dict[str, Any] = {
            "format": FORMAT,
            "subject": subject,
            "session": session,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "software": f"nwc {__version__}",
            "info": asdict(info),
            "complete": False,
            **(meta or {}),
        }
        self._write_manifest()
        self._samples = (self.path / SAMPLES).open("w", encoding="ascii", newline="\n")
        self._events = (self.path / EVENTS).open("w", encoding="utf-8", newline="\n")
        self._counted: bool | None = None
        self._format = ""
        self.samples = 0
        self._last_flush = time.monotonic()

    def _write_manifest(self) -> None:
        temporary = self.path / (MANIFEST + ".tmp")
        temporary.write_text(json.dumps(self._manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path / MANIFEST)

    def _header(self, counted: bool) -> None:
        self._counted = counted
        self._samples.write("timestamps," + ",".join(self.info.channels) + (",seq" if counted else "") + "\n")
        self._format = "%.4f," + ",".join(["%.3f"] * self.info.n_channels) + (",%d" if counted else "")

    def write(self, chunk: Chunk | None) -> None:
        if chunk is None or len(chunk) == 0:
            return
        if self._counted is None:
            self._header(chunk.seq is not None)
        columns = [np.asarray(chunk.t, dtype=np.float64), np.asarray(chunk.x, dtype=np.float64)]
        if self._counted:
            if chunk.seq is None:
                raise ValueError("this recording stores the device's sample counter, but a chunk came without one")
            columns.append(np.asarray(chunk.seq, dtype=np.float64))
        np.savetxt(self._samples, np.column_stack(columns), fmt=self._format)
        self.samples += len(chunk)
        if time.monotonic() - self._last_flush > 1.0:
            self.flush()

    def event(
        self, kind: str, start: int, stop: int, label: str = "", run: int = 0, trial: int = -1, **extra: Any
    ) -> Event:
        event = Event(kind, int(start), int(stop), label, int(run), int(trial), dict(extra))
        self._events.write(json.dumps(asdict(event), separators=(",", ":")) + "\n")
        self._events.flush()
        return event

    def flush(self) -> None:
        self._samples.flush()
        self._last_flush = time.monotonic()

    def close(self, **meta: Any) -> None:
        if self._samples.closed:
            return
        if self._counted is None:
            self._header(False)
        self._samples.close()
        self._events.close()
        self._manifest.update(meta)
        self._manifest["samples"] = self.samples
        self._manifest["complete"] = True
        self._write_manifest()

    def __enter__(self) -> SessionWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class Session:
    """A recorded session opened for reading."""

    def __init__(self, directory: str | Path):
        self.path = Path(directory)
        manifest = self.path / MANIFEST
        if not manifest.exists():
            raise FileNotFoundError(f"{self.path} is not a session (no {MANIFEST})")
        self.meta: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{self.path}: unsupported session format {self.meta.get('format')!r}")
        info = self.meta["info"]
        self.info = StreamInfo(
            tuple(info["channels"]),
            float(info["fs"]),
            info.get("source", ""),
            info.get("device", ""),
            info.get("unit", "uV"),
        )
        self.subject: str = self.meta.get("subject", self.path.parent.name)
        self.name: str = self.meta.get("session", self.path.name)
        self._raw: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None

    @property
    def key(self) -> str:
        return f"{self.subject}/{self.name}"

    def raw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Arrival times ``[n]``, samples ``[n, C]`` and the device counter (or None), as received."""
        if self._raw is None:
            path = self.path / SAMPLES
            with path.open("r", encoding="ascii") as handle:
                header = handle.readline().strip().split(",")
            counted = header[-1] == "seq"
            data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            n = self.info.n_channels
            if data.shape[1] != n + 1 + counted:
                raise ValueError(f"{path}: expected {n + 1 + counted} columns, found {data.shape[1]}")
            seq = np.rint(data[:, -1]).astype(np.int64) if counted else None
            self._raw = data[:, 0].copy(), data[:, 1 : n + 1].copy(), seq
        return self._raw

    def block(self) -> tuple[Block, StreamStats]:
        """The recording on the uniform sample grid, with fills and breaks marked."""
        t, x, seq = self.raw()
        return assemble(t, x, self.info, seq)

    def events(self, kind: str | None = None) -> list[Event]:
        path = self.path / EVENTS
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = Event(**json.loads(line))
                if kind is None or event.kind == kind:
                    out.append(event)
        return out

    def grid_events(self, block: Block, kind: str | None = None) -> list[Event]:
        """Events with ``start``/``stop`` translated from rows of arrival to grid samples."""
        grid = grid_of_rows(block)
        if len(grid) == 0:
            return []
        out = []
        for event in self.events(kind):
            start = int(grid[min(max(event.start, 0), len(grid) - 1)])
            stop = int(grid[event.stop]) if event.stop < len(grid) else int(grid[-1]) + 1
            out.append(Event(event.kind, start, stop, event.label, event.run, event.trial, event.extra))
        return out

    @property
    def labels(self) -> list[str]:
        """Cue labels in order of first appearance."""
        return list(dict.fromkeys(e.label for e in self.events(CUE)))

    @property
    def duration(self) -> float:
        return float(self.meta.get("samples", len(self.raw()[0]))) / self.info.fs

    def __repr__(self) -> str:
        return f"Session({self.key!r}, {self.info.fs:g} Hz, {len(self.info.channels)} channels)"


def find_sessions(root: str | Path) -> list[Session]:
    """All sessions below ``root`` (which may itself be one), ordered by path."""
    root = Path(root)
    if (root / MANIFEST).exists():
        return [Session(root)]
    return [Session(path.parent) for path in sorted(root.rglob(MANIFEST))]


def iter_chunks(t: np.ndarray, x: np.ndarray, seq: np.ndarray | None = None, size: int = 12) -> Iterator[Chunk]:
    """Cut a recording into packets of ``size`` samples, as a headset delivers them."""
    for start in range(0, len(t), size):
        yield Chunk(
            t[start : start + size], x[start : start + size], None if seq is None else seq[start : start + size]
        )

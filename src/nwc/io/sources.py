"""Where samples come from.

Every source hands over :class:`~nwc.io.stream.Chunk` objects through the same
small interface, so recording, monitoring and driving do not care whether the
EEG arrives over OSC, over Lab Streaming Layer, from a recording being replayed
or from the simulator.

Source specifications accepted by :func:`open_source`:

``osc[:[host:]port]`` Petal Metrics (``/PetalStream/eeg``) or Mind Monitor (``/muse/eeg``) over UDP.
                      Listens on this computer only unless a host is given: ``osc:0.0.0.0:5000``
                      accepts a stream sent from a phone.
``lsl[:name]``        an EEG stream on Lab Streaming Layer (muselsl, BlueMuse, Petal Metrics)
``replay:<dir>``      a recorded session, as fast as it can be read or in real time
``sim[:seed]``        the built-in simulator
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .session import Session
from .stream import Chunk, StreamInfo

MUSE_CHANNELS = ("TP9", "AF7", "AF8", "TP10")
PETAL_PORT = 14739
PETAL_ADDRESS = "/PetalStream/eeg"
MINDMONITOR_ADDRESS = "/muse/eeg"


class SourceError(RuntimeError):
    """A source could not be opened or stopped delivering."""


class Source:
    """Base class. ``read`` returns the samples that arrived since the last call, or None."""

    info: StreamInfo
    exhausted: bool = False

    def open(self) -> None:  # pragma: no cover - trivial default
        pass

    def read(self, timeout: float = 0.05) -> Chunk | None:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial default
        pass

    def __enter__(self) -> Source:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------- replay
class ReplaySource(Source):
    """Replays samples in packets, either immediately or paced like the original stream."""

    def __init__(
        self,
        t: np.ndarray,
        x: np.ndarray,
        info: StreamInfo,
        seq: np.ndarray | None = None,
        *,
        realtime: bool = False,
        packet: int = 12,
    ):
        self.info = info
        self._t, self._x, self._seq = np.asarray(t, dtype=np.float64), np.asarray(x), seq
        self._cursor = 0
        self._packet = int(packet)
        self._realtime = realtime
        self._wall_start: float | None = None

    @classmethod
    def from_session(cls, session: Session | str | Path, **kwargs) -> ReplaySource:
        session = session if isinstance(session, Session) else Session(session)
        t, x, seq = session.raw()
        return cls(t, x, session.info, seq, **kwargs)

    @property
    def exhausted(self) -> bool:  # type: ignore[override]
        return self._cursor >= len(self._t)

    @property
    def position(self) -> int:
        """Samples handed over so far."""
        return self._cursor

    def read(self, timeout: float = 0.05) -> Chunk | None:
        if self.exhausted:
            return None
        if self._realtime:
            if self._wall_start is None:
                self._wall_start = time.monotonic()
            due = int((time.monotonic() - self._wall_start) * self.info.fs) // self._packet * self._packet
            if due <= self._cursor:
                time.sleep(min(timeout, self._packet / self.info.fs))
                return None
            stop = min(due, len(self._t))
        else:
            stop = min(self._cursor + self._packet, len(self._t))
        a, self._cursor = self._cursor, stop
        return Chunk(self._t[a:stop], self._x[a:stop], None if self._seq is None else self._seq[a:stop])


# ---------------------------------------------------------------------- OSC
class OSCSource(Source):
    """EEG over OSC/UDP.

    Two message layouts are understood and told apart by their address:

    * Petal Metrics, ``/PetalStream/eeg``: sample id, unix time (whole and
      fractional part), LSL time (whole and fractional part), then one float per
      channel. The sample id is used as the device counter once it has proven to
      advance by one per sample.
    * Mind Monitor, ``/muse/eeg``: one float per channel, stamped on arrival.
    """

    #: Samples inspected before deciding whether the sample id is a usable counter.
    PROBE = 128

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = PETAL_PORT,
        address: str = PETAL_ADDRESS,
        channels: Sequence[str] = MUSE_CHANNELS,
        fs: float = 256.0,
        use_counter: bool | None = None,
    ):
        self.host, self.port, self.address = host, int(port), address
        self.info = StreamInfo(tuple(channels), float(fs), source=f"osc://{host}:{port}{address}")
        self._use_counter = use_counter
        self._queue: deque[tuple[float, int | None, tuple[float, ...]]] = deque()
        self._lock = threading.Lock()
        self._server = None
        self._thread: threading.Thread | None = None
        self._probe: list[tuple[float, int | None, tuple[float, ...]]] = []
        self._decided = use_counter is not None
        self.malformed = 0

    # -- message handling (runs in the server thread)
    def _on_message(self, address: str, *args: float) -> None:
        n = self.info.n_channels
        if address == MINDMONITOR_ADDRESS or len(args) < n + 5:
            if len(args) < n:
                self.malformed += 1
                return
            item = (time.time(), None, tuple(float(v) for v in args[:n]))
        else:
            try:
                stamp = float(args[1]) + float(args[2])
                item = (stamp, int(args[0]), tuple(float(v) for v in args[5 : 5 + n]))
            except (TypeError, ValueError):
                self.malformed += 1
                return
        with self._lock:
            self._queue.append(item)

    def open(self) -> None:
        try:
            from pythonosc.dispatcher import Dispatcher
            from pythonosc.osc_server import ThreadingOSCUDPServer
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise SourceError('OSC input needs python-osc: pip install "neuro-wheelchair[osc]"') from error
        dispatcher = Dispatcher()
        dispatcher.map(self.address, self._on_message)
        if self.address != MINDMONITOR_ADDRESS:
            dispatcher.map(MINDMONITOR_ADDRESS, self._on_message)
        try:
            self._server = ThreadingOSCUDPServer((self.host, self.port), dispatcher)
        except OSError as error:
            raise SourceError(f"cannot listen on udp://{self.host}:{self.port}: {error}") from error
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="nwc-osc", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _decide(self) -> None:
        """Use the sample id as a counter only if it behaves like one."""
        ids = [item[1] for item in self._probe]
        if any(i is None for i in ids):
            self._use_counter = False
        else:
            steps = np.diff(np.asarray(ids, dtype=np.int64))
            self._use_counter = bool(len(steps)) and float(np.mean(steps == 1)) >= 0.95
        self._decided = True

    def read(self, timeout: float = 0.05) -> Chunk | None:
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
        if not self._decided:
            self._probe.extend(items)
            # Samples without an id settle the question at once; otherwise watch the ids for a while.
            if len(self._probe) < self.PROBE and all(item[1] is not None for item in self._probe):
                if not items:
                    time.sleep(timeout)
                return None
            self._decide()
            items, self._probe = self._probe, []
        if not items:
            time.sleep(timeout)
            return None
        t = np.array([item[0] for item in items])
        x = np.array([item[2] for item in items])
        seq = np.array([item[1] for item in items], dtype=np.int64) if self._use_counter else None
        return Chunk(t, x, seq)


# ---------------------------------------------------------------------- LSL
class LSLSource(Source):
    """An EEG stream on Lab Streaming Layer, found by name or, failing that, by type."""

    def __init__(self, name: str | None = None, channels: Sequence[str] = MUSE_CHANNELS, timeout: float = 10.0):
        self.name = name
        self._wanted = tuple(channels)
        self._timeout = timeout
        self._inlet = None
        self._columns: list[int] = []
        self.info = StreamInfo(self._wanted, 256.0, source=f"lsl:{name or 'EEG'}")

    def open(self) -> None:
        try:
            import pylsl
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise SourceError('LSL input needs pylsl: pip install "neuro-wheelchair[lsl]"') from error
        except Exception as error:  # pragma: no cover - pylsl raises when its native library is missing
            raise SourceError(
                "pylsl is installed but its native library, liblsl, was not found. Install it from "
                "https://github.com/sccn/liblsl/releases or with `conda install -c conda-forge liblsl`"
            ) from error
        key, value = ("name", self.name) if self.name else ("type", "EEG")
        streams = pylsl.resolve_byprop(key, value, timeout=self._timeout)
        if not streams:
            raise SourceError(f"no LSL stream with {key} {value!r} within {self._timeout:g} s")
        self._inlet = pylsl.StreamInlet(streams[0], max_chunklen=12)
        self._inlet.open_stream(timeout=self._timeout)  # subscribe now: an inlet otherwise connects on its first read
        info = self._inlet.info()
        labels = []
        node = info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            labels.append(node.child_value("label"))
            node = node.next_sibling()
        if all(name in labels for name in self._wanted):
            self._columns = [labels.index(name) for name in self._wanted]
        elif info.channel_count() >= len(self._wanted):
            self._columns = list(range(len(self._wanted)))  # unlabelled stream: Muse order
        else:
            raise SourceError(f"LSL stream has {info.channel_count()} channels, need {len(self._wanted)}")
        self.info = StreamInfo(
            self._wanted, float(info.nominal_srate()), source=f"lsl:{info.name()}", device=info.name()
        )

    def read(self, timeout: float = 0.05) -> Chunk | None:
        if self._inlet is None:
            raise SourceError("source is not open")
        samples, stamps = self._inlet.pull_chunk(timeout=timeout, max_samples=256)
        if not stamps:
            return None
        return Chunk(np.asarray(stamps, dtype=np.float64), np.asarray(samples, dtype=np.float64)[:, self._columns])

    def close(self) -> None:
        if self._inlet is not None:
            self._inlet.close_stream()
            self._inlet = None


# ---------------------------------------------------------------------- factory
def open_source(
    spec: str, *, channels: Sequence[str] = MUSE_CHANNELS, fs: float = 256.0, realtime: bool = False, **options
) -> Source:
    """Build a source from a specification string (see the module docstring). The caller opens it."""
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    if kind == "osc":
        host, port = "127.0.0.1", PETAL_PORT
        if rest:
            head, _, tail = rest.rpartition(":")
            host, port = (head or host), int(tail)
        return OSCSource(host, port, options.get("address", PETAL_ADDRESS), channels, fs, options.get("use_counter"))
    if kind == "lsl":
        return LSLSource(rest or None, channels, timeout=options.get("timeout", 10.0))
    if kind == "replay":
        if not rest:
            raise SourceError("replay needs a session directory: replay:<dir>")
        return ReplaySource.from_session(rest, realtime=realtime)
    if kind == "sim":
        from ..sim.source import SimulatedSource

        return SimulatedSource.from_spec(rest, channels=channels, fs=fs, realtime=realtime, **options)
    raise SourceError(f"unknown source {spec!r}: use osc[:[host:]port], lsl[:name], replay:<dir> or sim[:seed]")

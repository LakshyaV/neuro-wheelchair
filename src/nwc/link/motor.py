"""The link to the motor controller.

All links share the protocol and the bookkeeping: numbering commands, parsing
what comes back, and judging whether the controller is alive. They differ only
in where the bytes go: a USB serial port, the virtual controller of the
simulator, or nowhere at all (a dry run).

A link is healthy while telemetry keeps arriving. Whether it is safe to drive is
not decided here; the supervisor asks.
"""

from __future__ import annotations

from collections.abc import Callable

from . import protocol
from .controller import VirtualController
from .protocol import Info, Telemetry


class LinkError(RuntimeError):
    """The motor controller could not be reached."""


class MotorLink:
    """Base class: protocol, sequence numbers, telemetry."""

    #: Without telemetry for this long the controller counts as gone.
    TIMEOUT_S = 0.5

    def __init__(self) -> None:
        self._seq = 0
        self._lines = protocol.LineBuffer()
        self.telemetry: Telemetry | None = None
        self.info: Info | None = None
        self._last_heard: float | None = None
        self.damaged = 0
        self.last_command: tuple[int, int] = (0, 0)

    # -- to be provided by subclasses
    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def _write(self, data: bytes, now: float) -> None:
        raise NotImplementedError

    def _read(self, now: float) -> bytes:
        raise NotImplementedError

    # -- commands
    def _next(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFF
        return self._seq

    def drive(self, left: float, right: float, now: float) -> None:
        """Ask for wheel speeds as fractions of full scale, -1..1."""
        command = (round(255 * max(-1.0, min(1.0, left))), round(255 * max(-1.0, min(1.0, right))))
        self.last_command = command
        self._write(protocol.drive(self._next(), *command), now)

    def stop(self, now: float) -> None:
        self.last_command = (0, 0)
        self._write(protocol.stop(self._next()), now)

    def release(self, now: float) -> None:
        """Ask the controller to leave its emergency stop."""
        self._write(protocol.release(self._next()), now)

    def query(self, now: float) -> None:
        self._write(protocol.query(self._next()), now)

    # -- reports
    def poll(self, now: float) -> None:
        """Take in whatever the controller has sent."""
        for line in self._lines.feed(self._read(now)):
            report = protocol.parse_report(line)
            if report is None:
                self.damaged += 1
            elif isinstance(report, Info):
                self.info = report
                self._last_heard = now
            else:
                self.telemetry = report
                self._last_heard = now

    def healthy(self, now: float) -> bool:
        return self._last_heard is not None and now - self._last_heard <= self.TIMEOUT_S

    @property
    def state(self) -> str:
        return self.telemetry.state if self.telemetry else "?"

    @property
    def emergency(self) -> bool:
        """The controller's own emergency stop is latched."""
        return self.telemetry is not None and self.telemetry.state == protocol.EMERGENCY

    def describe(self) -> str:
        if self.telemetry is None:
            return "no telemetry"
        t = self.telemetry
        return f"{t.state_name}, wheels {t.left:+d}/{t.right:+d}"

    def __enter__(self) -> MotorLink:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class SerialLink(MotorLink):
    """The controller on a USB serial port."""

    def __init__(self, port: str, baud: int = 115200):
        super().__init__()
        self.port, self.baud = port, baud
        self._serial = None

    def open(self) -> None:
        try:
            import serial
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise LinkError('the serial link needs pyserial: pip install "neuro-wheelchair[serial]"') from error
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=0, write_timeout=0.2)
        except (OSError, serial.SerialException) as error:
            raise LinkError(f"cannot open {self.port}: {error}") from error

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.write(protocol.stop(self._next()))
                self._serial.flush()
            except Exception:  # the port may already be gone; the controller's watchdog covers it
                pass
            self._serial.close()
            self._serial = None

    def _write(self, data: bytes, now: float) -> None:
        if self._serial is None:
            raise LinkError("the serial link is not open")
        try:
            self._serial.write(data)
        except Exception as error:
            raise LinkError(f"write to {self.port} failed: {error}") from error

    def _read(self, now: float) -> bytes:
        if self._serial is None:
            return b""
        try:
            waiting = self._serial.in_waiting
            return self._serial.read(waiting) if waiting else b""
        except Exception:
            return b""  # reported through the missing telemetry


class VirtualLink(MotorLink):
    """An in-process controller. ``on_step`` is called after every advance, e.g. to move a simulated chair."""

    def __init__(
        self, controller: VirtualController | None = None, on_step: Callable[[VirtualController], None] | None = None
    ):
        super().__init__()
        self.controller = controller or VirtualController()
        self._on_step = on_step
        self._origin: float | None = None

    def _ms(self, now: float) -> int:
        if self._origin is None:
            self._origin = now
        return round((now - self._origin) * 1000.0)

    def _advance(self, now: float, data: bytes = b"") -> None:
        """Bring the controller to ``now`` in steps of at most 10 ms, telling the listener after each."""
        target = max(self._ms(now), self.controller.now_ms)
        while True:
            self.controller.advance(min(target, self.controller.now_ms + 10), data)
            data = b""
            if self._on_step is not None:
                self._on_step(self.controller)
            if self.controller.now_ms >= target:
                break

    def _write(self, data: bytes, now: float) -> None:
        self._advance(now, data)

    def _read(self, now: float) -> bytes:
        self._advance(now)
        return self.controller.transmit()


class NullLink(MotorLink):
    """No controller: commands are accepted and reported back as if carried out. For dry runs."""

    def _write(self, data: bytes, now: float) -> None:
        pass

    def _read(self, now: float) -> bytes:
        left, right = self.last_command
        state = protocol.DRIVING if (left or right) else protocol.IDLE
        return protocol.telemetry(self._seq, state, left, right, 0, 0)


def open_link(spec: str) -> MotorLink:
    """``none`` (dry run), ``sim`` (virtual controller) or a serial port such as ``/dev/cu.usbmodem1101`` or ``COM3``."""
    kind = spec.strip()
    if kind.lower() in ("none", "dry", "null"):
        return NullLink()
    if kind.lower() == "sim":
        return VirtualLink()
    return SerialLink(kind)

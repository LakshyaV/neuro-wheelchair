"""The driving loop: a source, an engine, a keyboard and a status line.

Keys for the operator (the person next to the wearer)::

    a        arm: allow the chair to move (after the baseline)
    space    disarm: stop and stay stopped
    b        take the baseline again (after re-seating the headband); disarms
    q        quit: stop and leave

The loop itself is a few lines. Everything that matters happens in
:class:`~nwc.runtime.engine.Engine`.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

from ..io.sources import Source
from .engine import Engine, Status


class Keys:
    """Single key presses without Enter, where a terminal allows it. Otherwise no keys are seen."""

    def __init__(self) -> None:
        self._restore: Callable[[], None] | None = None
        self._read: Callable[[], str] = lambda: ""

    def __enter__(self) -> Keys:
        if not sys.stdin.isatty():
            return self
        try:
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self._restore = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            self._read = lambda: sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else ""
        except ImportError:  # Windows
            import msvcrt

            self._read = lambda: msvcrt.getwch() if msvcrt.kbhit() else ""  # type: ignore[attr-defined]
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._restore is not None:
            self._restore()

    def poll(self) -> str:
        return self._read()


def stream_clock(engine: Engine) -> Callable[[], float]:
    """Time measured in samples received: makes replays and simulations exactly repeatable."""
    return lambda: engine.assembler.stats.received / engine.frontend_config.fs


def drive(
    engine: Engine,
    source: Source,
    *,
    clock: Callable[[], float] = time.monotonic,
    keys: Keys | None = None,
    show: Callable[[Status], None] | None = None,
    duration_s: float | None = None,
    auto_arm: bool = False,
    on_step: Callable[[float, Status], None] | None = None,
) -> Status:
    """Run until the source ends, ``duration_s`` has passed or the operator quits. Always ends with a stop."""
    started = clock()
    status = engine.status
    armed_once = False
    try:
        while not source.exhausted:
            chunk = source.read()
            now = clock()
            status = engine.feed(chunk, now)
            if auto_arm and not armed_once and status.phase == "ready":
                armed_once = engine.arm(now)
            key = keys.poll() if keys is not None else ""
            if key == "a":
                engine.arm(now)
            elif key == " ":
                engine.disarm(now)
            elif key == "b":
                engine.rebaseline(now)
            elif key == "q":
                break
            if show is not None:
                show(status)
            if on_step is not None:
                on_step(now, status)
            if duration_s is not None and now - started >= duration_s:
                break
    finally:
        engine.close(clock())
    return status


class StatusLine:
    """Rewrites one terminal line a few times a second; prints changes of state on their own lines."""

    def __init__(self, stream=None, interval_s: float = 0.2):
        self.stream = stream or sys.stdout
        self.interval = interval_s
        self._last = 0.0
        self._state = ("", "", "")
        self._tty = self.stream.isatty()

    def __call__(self, status: Status) -> None:
        state = (status.phase, status.state, status.motion)
        changed = state != self._state
        if changed:
            self._state = state
            if not self._tty:
                print(status.line(), file=self.stream, flush=True)
                return
        now = time.monotonic()
        if self._tty and (changed or now - self._last >= self.interval):
            self._last = now
            self.stream.write("\r\x1b[2K" + status.line()[:160] + ("\n" if changed else ""))
            self.stream.flush()

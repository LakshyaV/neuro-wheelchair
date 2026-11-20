"""The motor controller's behaviour in Python.

:class:`Drive` is a line-for-line port of ``firmware/wheelchair_controller/drive.h``
with the same integer arithmetic, and :class:`VirtualController` is the sketch's
main loop around it. Together they stand in for the Arduino in simulation and in
tests, and a test runs the compiled firmware and this port through the same
script to make sure they stay identical.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import protocol

SCALE = 256
FIRMWARE_VERSION = 2


@dataclass(frozen=True)
class DriveConfig:
    watchdog_ms: int = 400
    ramp_up_ms: int = 600
    ramp_down_ms: int = 200


def _trunc_div(a: int, b: int) -> int:
    """Integer division that truncates towards zero, as C does."""
    return abs(a) // b * (1 if a >= 0 else -1)


class Drive:
    def __init__(self, config: DriveConfig | None = None):
        self.config = config or DriveConfig()
        self.state = protocol.IDLE
        self._left = self._right = self._target_left = self._target_right = 0
        self._last_command_ms = self._last_update_ms = 0
        self._ever_commanded = self._started = self._input_active = False

    @staticmethod
    def _clamp(value: int) -> int:
        return max(-255, min(255, value)) * SCALE

    def command(self, left: int, right: int, now_ms: int) -> None:
        self._last_command_ms = now_ms
        self._ever_commanded = True
        if self.state == protocol.EMERGENCY:
            return
        self._target_left, self._target_right = self._clamp(left), self._clamp(right)
        if self.state == protocol.WATCHDOG:
            self.state = protocol.IDLE

    def stop(self) -> None:
        self._target_left = self._target_right = 0

    def emergency_input(self, active: bool) -> None:
        self._input_active = active
        if active:
            self.state = protocol.EMERGENCY
            self._target_left = self._target_right = self._left = self._right = 0

    def release(self) -> bool:
        if self.state != protocol.EMERGENCY or self._input_active:
            return False
        self.state = protocol.IDLE
        return True

    def _approach(self, current: int, target: int, elapsed_ms: int) -> int:
        if current == target:
            return current
        same_sign = (current >= 0 and target >= 0) or (current <= 0 and target <= 0)
        goal = target if same_sign else 0
        slowing = abs(goal) < abs(current)
        ramp_ms = self.config.ramp_down_ms if slowing else self.config.ramp_up_ms
        step = 255 * SCALE if ramp_ms == 0 else (255 * SCALE * min(elapsed_ms, 1000)) // ramp_ms
        if goal > current:
            return goal if goal - current <= step else current + step
        return goal if current - goal <= step else current - step

    def update(self, now_ms: int) -> None:
        elapsed = now_ms - self._last_update_ms if self._started else 0
        self._last_update_ms, self._started = now_ms, True
        if self.state == protocol.EMERGENCY:
            return
        if self._ever_commanded and now_ms - self._last_command_ms > self.config.watchdog_ms:
            self._target_left = self._target_right = 0
            self.state = protocol.WATCHDOG
        self._left = self._approach(self._left, self._target_left, elapsed)
        self._right = self._approach(self._right, self._target_right, elapsed)
        if self.state != protocol.WATCHDOG:
            moving = self._left or self._right or self._target_left or self._target_right
            self.state = protocol.DRIVING if moving else protocol.IDLE

    @property
    def left(self) -> int:
        return _trunc_div(self._left, SCALE)

    @property
    def right(self) -> int:
        return _trunc_div(self._right, SCALE)

    def command_age(self, now_ms: int) -> int:
        return min(now_ms - self._last_command_ms, 9999) if self._ever_commanded else 9999


class VirtualController:
    """The sketch around :class:`Drive`: one :meth:`step` is one pass of its ``loop()``."""

    TELEMETRY_MS = 100

    def __init__(self, config: DriveConfig | None = None, *, emergency_input: bool = False):
        self.drive = Drive(config)
        self._lines = protocol.LineBuffer()
        self._last_seq = 0
        self._last_telemetry_ms = 0
        self._now_ms = 0
        self._has_input = emergency_input  # the sketch was built with an emergency-stop pin
        self._input_active = False
        self.rejected = 0
        cfg = self.drive.config
        self._out = bytearray(protocol.info(FIRMWARE_VERSION, cfg.watchdog_ms, cfg.ramp_up_ms, cfg.ramp_down_ms))

    def _handle(self, line: bytes) -> None:
        command = protocol.parse_command(line)
        if command is None:
            self.rejected = (self.rejected + 1) & 0xFFFF
            return
        self._last_seq = command.seq
        if command.type == "D":
            self.drive.command(command.left, command.right, self._now_ms)
        elif command.type == "S":
            self.drive.stop()
        elif command.type == "R":
            self.drive.release()
        elif command.type == "Q":
            cfg = self.drive.config
            self._out += protocol.info(FIRMWARE_VERSION, cfg.watchdog_ms, cfg.ramp_up_ms, cfg.ramp_down_ms)

    def step(self, now_ms: int, received: bytes = b"") -> None:
        """One pass of the loop at ``now_ms``: check the input, read what arrived, update, report."""
        self._now_ms = now_ms
        if self._has_input:
            self.drive.emergency_input(self._input_active)
        for line in self._lines.feed(received):
            self._handle(line)
        self.drive.update(now_ms)
        if now_ms - self._last_telemetry_ms >= self.TELEMETRY_MS:
            self._last_telemetry_ms = now_ms
            d = self.drive
            self._out += protocol.telemetry(
                self._last_seq, d.state, d.left, d.right, d.command_age(now_ms), self.rejected
            )

    def advance(self, now_ms: int, received: bytes = b"") -> None:
        """Run the loop every millisecond up to ``now_ms``; ``received`` is read in the first pass."""
        if now_ms <= self._now_ms:
            self.step(self._now_ms, received)
            return
        for t in range(self._now_ms + 1, now_ms + 1):
            self.step(t, received)
            received = b""

    def emergency_input(self, active: bool) -> None:
        """Level of the emergency-stop input (True: the switch is open)."""
        self._has_input = True
        self._input_active = active

    def transmit(self) -> bytes:
        """Everything the controller has sent since the last call."""
        data, self._out = bytes(self._out), bytearray()
        return data

    @property
    def now_ms(self) -> int:
        return self._now_ms

    @property
    def outputs(self) -> tuple[int, int]:
        return self.drive.left, self.drive.right

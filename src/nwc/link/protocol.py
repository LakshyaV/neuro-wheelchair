"""The serial protocol between the host and the motor controller, host side.

One line of ASCII per message::

    $<payload>*<HH>\\n

with ``HH`` the CRC-8 (polynomial 0x07) of the payload in hex. See
``firmware/wheelchair_controller/protocol.h`` for the message list. The firmware
parses with the same strictness as :func:`parse_command` here, and a test feeds
both the same bytes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PROTOCOL_VERSION = 1
MAX_LINE = 48

IDLE, DRIVING, WATCHDOG, EMERGENCY = "I", "D", "W", "E"
STATE_NAMES = {IDLE: "idle", DRIVING: "driving", WATCHDOG: "watchdog", EMERGENCY: "emergency stop"}


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def frame(payload: str) -> bytes:
    data = payload.encode("ascii")
    return b"$" + data + b"*" + f"{crc8(data):02X}".encode("ascii") + b"\n"


def unframe(line: bytes) -> str | None:
    """The payload of a line if it is well-formed and its checksum matches, else None."""
    line = line.rstrip(b"\r\n")
    if not 7 <= len(line) <= MAX_LINE or line[:1] != b"$" or line[-3:-2] != b"*":
        return None
    payload = line[1:-3]
    try:
        if int(line[-2:], 16) != crc8(payload) or not re.fullmatch(rb"[0-9A-Fa-f]{2}", line[-2:]):
            return None
        return payload.decode("ascii")
    except (ValueError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------- host to controller
def drive(seq: int, left: int, right: int) -> bytes:
    """A drive command. Speeds are clipped to -255..255."""
    left, right = (max(-255, min(255, round(v))) for v in (left, right))
    return frame(f"D,{seq & 0xFFFF},{left},{right}")


def stop(seq: int) -> bytes:
    return frame(f"S,{seq & 0xFFFF}")


def release(seq: int) -> bytes:
    return frame(f"R,{seq & 0xFFFF}")


def query(seq: int) -> bytes:
    return frame(f"Q,{seq & 0xFFFF}")


@dataclass(frozen=True)
class Command:
    type: str
    seq: int
    left: int = 0
    right: int = 0


_FIELD = r"-?[0-9]{1,5}"
_DRIVE = re.compile(rf"D,({_FIELD}),({_FIELD}),({_FIELD})")
_OTHER = re.compile(rf"([SRQ]),({_FIELD})")


def parse_command(line: bytes) -> Command | None:
    """What the controller accepts. Mirrors ``parseLine`` in the firmware."""
    payload = unframe(line)
    if payload is None:
        return None
    match = _DRIVE.fullmatch(payload)
    if match:
        seq, left, right = (int(v) for v in match.groups())
        if 0 <= seq <= 65535 and -255 <= left <= 255 and -255 <= right <= 255:
            return Command("D", seq, left, right)
        return None
    match = _OTHER.fullmatch(payload)
    if match and 0 <= int(match.group(2)) <= 65535:
        return Command(match.group(1), int(match.group(2)))
    return None


# ---------------------------------------------------------------------- controller to host
@dataclass(frozen=True)
class Telemetry:
    seq: int
    state: str
    left: int
    right: int
    age_ms: int
    rejected: int

    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(self.state, self.state)


@dataclass(frozen=True)
class Info:
    protocol: int
    firmware: int
    watchdog_ms: int
    ramp_up_ms: int
    ramp_down_ms: int


def telemetry(seq: int, state: str, left: int, right: int, age_ms: int, rejected: int) -> bytes:
    return frame(f"T,{seq},{state},{left},{right},{age_ms},{rejected}")


def info(firmware: int, watchdog_ms: int, ramp_up_ms: int, ramp_down_ms: int) -> bytes:
    return frame(f"I,{PROTOCOL_VERSION},{firmware},{watchdog_ms},{ramp_up_ms},{ramp_down_ms}")


def parse_report(line: bytes) -> Telemetry | Info | None:
    """A message from the controller, or None if the line is damaged or unknown."""
    payload = unframe(line)
    if payload is None:
        return None
    fields = payload.split(",")
    try:
        if fields[0] == "T" and len(fields) == 7 and fields[2] in STATE_NAMES:
            return Telemetry(int(fields[1]), fields[2], int(fields[3]), int(fields[4]), int(fields[5]), int(fields[6]))
        if fields[0] == "I" and len(fields) == 6:
            return Info(*(int(v) for v in fields[1:]))
    except ValueError:
        return None
    return None


class LineBuffer:
    """Collects received bytes into complete lines, the way the firmware's reader does.

    Carriage returns are dropped, an empty line is nothing, and a line that grows
    past any sensible length is cut off (it will fail its checksum).
    """

    def __init__(self) -> None:
        self._pending = b""

    def feed(self, data: bytes) -> list[bytes]:
        self._pending += data.replace(b"\r", b"")
        *lines, self._pending = self._pending.split(b"\n")
        if len(self._pending) > 4 * MAX_LINE:
            self._pending = self._pending[: MAX_LINE + 1]
        return [line for line in lines if line]

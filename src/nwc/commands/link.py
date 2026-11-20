"""Test the motor controller on its own, without any EEG.

    nwc link --port /dev/cu.usbmodem1101            check: identify, telemetry, watchdog
    nwc link --port /dev/cu.usbmodem1101 --keys     drive from the keyboard (wheels off the ground)

The check moves nothing. It asks the controller who it is, sends zero-speed
commands, then goes quiet and confirms that the controller reports its watchdog.
With --keys: w forward, x backward, a left, d right, space stop, q quit. A motion
lasts one and a half seconds unless the key is pressed again.
"""

from __future__ import annotations

import argparse

NAME = "link"
HELP = "test the motor controller"
ORDER = 45

SPEEDS = {"w": (0.5, 0.5), "x": (-0.4, -0.4), "a": (-0.35, 0.35), "d": (0.35, -0.35), " ": (0.0, 0.0)}


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", required=True, help="serial port of the controller, or `sim`")
    parser.add_argument("--keys", action="store_true", help="drive from the keyboard")


def _settle(link, seconds: float, speeds=(0.0, 0.0), send: bool = True) -> None:
    import time

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        now = time.monotonic()
        if send:
            link.drive(*speeds, now)
        link.poll(now)
        time.sleep(0.05)


def run(args: argparse.Namespace) -> int:
    import time

    from ..link import protocol
    from ..link.motor import LinkError, open_link
    from ..runtime.live import Keys

    try:
        link = open_link(args.port)
        link.open()
    except LinkError as error:
        print(f"error: {error}")
        return 1
    try:
        # An Arduino resets when its port is opened; give it time to start.
        deadline = time.monotonic() + 4.0
        while link.info is None and time.monotonic() < deadline:
            now = time.monotonic()
            link.query(now)
            link.poll(now)
            time.sleep(0.1)
        if link.info is None:
            print("error: no answer. Is this the right port, and is the wheelchair_controller sketch on the board?")
            return 1
        info = link.info
        print(
            f"controller: protocol {info.protocol}, firmware {info.firmware}, watchdog {info.watchdog_ms} ms, ramps {info.ramp_up_ms}/{info.ramp_down_ms} ms"
        )
        if info.protocol != protocol.PROTOCOL_VERSION:
            print(
                f"error: this program speaks protocol {protocol.PROTOCOL_VERSION}. Upload the sketch from firmware/wheelchair_controller."
            )
            return 1
        if args.keys:
            print("w forward, x backward, a left, d right, space stop, q quit")
            speeds, until = (0.0, 0.0), 0.0
            with Keys() as keys:
                while True:
                    now = time.monotonic()
                    key = keys.poll()
                    if key == "q":
                        break
                    if key in SPEEDS:
                        speeds, until = SPEEDS[key], now + 1.5
                    link.drive(*(speeds if now < until else (0.0, 0.0)), now)
                    link.poll(now)
                    print(f"\r\x1b[2K{link.describe()}", end="", flush=True)
                    time.sleep(0.05)
            link.stop(time.monotonic())
            print()
            return 0
        _settle(link, 1.0)
        if not link.healthy(time.monotonic()) or link.telemetry is None:
            print("error: no telemetry while sending commands")
            return 1
        print(
            f"telemetry: {link.describe()}, {link.telemetry.rejected} lines rejected, {link.damaged} damaged on the way back"
        )
        if link.emergency:
            print("the emergency stop is latched: close the switch, then arm from `nwc drive` or send a release")
            return 1
        _settle(link, info.watchdog_ms / 1000.0 + 0.4, send=False)
        watchdog = link.telemetry.state == protocol.WATCHDOG
        print(
            f"watchdog: {'reported after the commands stopped' if watchdog else 'NOT reported after the commands stopped'}"
        )
        _settle(link, 0.3)
        print(f"after resuming: {link.describe()}")
        return 0 if watchdog else 1
    except LinkError as error:
        print(f"error: {error}")
        return 1
    finally:
        link.close()

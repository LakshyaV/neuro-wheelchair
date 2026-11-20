from __future__ import annotations

import itertools

import pytest

from nwc.link import protocol
from nwc.link.controller import Drive, DriveConfig, VirtualController
from nwc.link.motor import LinkError, MotorLink, NullLink, SerialLink, VirtualLink, open_link
from nwc.sim.chair import Chair


# ---------------------------------------------------------------------- protocol
def test_frames_are_checksummed_lines():
    assert protocol.crc8(b"") == 0 and protocol.crc8(b"123456789") == 0xF4  # the standard check value of CRC-8
    line = protocol.drive(7, 120, -40)
    assert line == b"$D,7,120,-40*" + f"{protocol.crc8(b'D,7,120,-40'):02X}".encode() + b"\n"
    assert protocol.unframe(line) == "D,7,120,-40" and protocol.unframe(line.rstrip(b"\n") + b"\r\n") == "D,7,120,-40"
    assert protocol.drive(70000, 999.4, -999) == protocol.drive(70000 & 0xFFFF, 255, -255)  # clipped, sequence wraps
    for damaged in (
        line[:-1].replace(b"120", b"121"),
        line[1:],
        b"$D,7,120,-40\n",
        b"$*00\n",
        b"$" + b"x" * 60 + b"*00\n",
        b"\xff\xfe",
    ):
        assert protocol.unframe(damaged) is None


def test_the_controller_accepts_exactly_the_documented_commands():
    assert protocol.parse_command(protocol.drive(1, 255, -255)) == protocol.Command("D", 1, 255, -255)
    assert protocol.parse_command(protocol.stop(2)) == protocol.Command("S", 2)
    assert (
        protocol.parse_command(protocol.release(3)).type == "R"
        and protocol.parse_command(protocol.query(4)).type == "Q"
    )
    for payload in (
        "D,1,256,0",
        "D,1,0,-256",
        "D,65536,0,0",
        "D,1,0",
        "D,1,0,0,0",
        "D,1,0,0,",
        "D, 1,0,0",
        "D,+1,0,0",
        "D,1,0.5,0",
        "D,1,a,0",
        "X,1",
        "S",
        "S,",
        "S,1,2",
        "d,1,0,0",
        "D,123456,0,0",
    ):
        assert protocol.parse_command(protocol.frame(payload)) is None, payload
    assert protocol.parse_command(protocol.frame("D,00001,-0,007")) == protocol.Command(
        "D", 1, 0, 7
    )  # leading zeros are numbers too


def test_reports_from_the_controller():
    report = protocol.parse_report(protocol.telemetry(9, protocol.DRIVING, 100, -100, 12, 3))
    assert report == protocol.Telemetry(9, "D", 100, -100, 12, 3) and report.state_name == "driving"
    info = protocol.parse_report(protocol.info(2, 400, 600, 200))
    assert info == protocol.Info(protocol.PROTOCOL_VERSION, 2, 400, 600, 200)
    for payload in ("T,1,Z,0,0,0,0", "T,1,D,0,0,0", "T,x,D,0,0,0,0", "I,1,2,3", "Q,1"):
        assert protocol.parse_report(protocol.frame(payload)) is None
    buffer = protocol.LineBuffer()
    assert buffer.feed(b"$T,1,") == [] and buffer.feed(b"I,0,0,0,0*00\r\n\n$next") == [b"$T,1,I,0,0,0,0*00"]
    assert buffer.feed(b"x" * 500 + b"\n") != [] and buffer.feed(b"ok\n") == [b"ok"]  # garbage does not accumulate


# ---------------------------------------------------------------------- drive logic
def run(
    drive: Drive, start: int, stop: int, command: tuple[int, int] | None = None, every: int = 100
) -> list[tuple[int, int]]:
    out = []
    for t in range(start, stop):
        if command is not None and (t - start) % every == 0:
            drive.command(*command, t)
        drive.update(t)
        out.append((drive.left, drive.right))
    return out


def test_outputs_ramp_and_never_jump():
    drive = Drive(DriveConfig(watchdog_ms=400, ramp_up_ms=600, ramp_down_ms=200))
    trace = run(drive, 0, 1000, (255, 128))
    left = [v[0] for v in trace]
    assert left[0] == 0 and left[299] == pytest.approx(127, abs=2) and left[620] == 255  # full speed after 600 ms
    assert max(b - a for a, b in itertools.pairwise(left)) <= 1 and trace[-1] == (255, 128)
    assert drive.state == protocol.DRIVING
    down = [v[0] for v in run(drive, 1000, 1400, (0, 0))]
    assert down[100] == pytest.approx(127, abs=3) and down[210] == 0  # down in a third of the time
    assert drive.state == protocol.IDLE


def test_reversing_goes_through_standstill():
    drive = Drive()
    run(drive, 0, 800, (200, 200))
    trace = [v[0] for v in run(drive, 800, 1800, (-200, -200))]
    zero = trace.index(0)
    assert 140 <= zero <= 180 and min(trace[:zero]) >= 0  # first to a stop on the fast ramp...
    assert trace[-1] == -200 and all(
        b <= a for a, b in itertools.pairwise(trace)
    )  # ...then away from it on the slow one


def test_watchdog_stops_when_commands_stop():
    drive = Drive(DriveConfig(watchdog_ms=400))
    run(drive, 0, 1000, (200, 200))
    silent = run(drive, 1000, 1800)  # the host has gone quiet
    assert silent[300] == (200, 200) and drive.state == protocol.WATCHDOG
    stopped = next(i for i, v in enumerate(silent) if v == (0, 0))
    assert 400 <= stopped + 100 <= 400 + 200 + 110  # watchdog time after the last command (at 900), plus the ramp down
    assert drive.command_age(1800) == 900
    run(drive, 1800, 2400, (100, 100))
    assert drive.state == protocol.DRIVING and drive.left == 100  # commands again: it carries on
    assert Drive().command_age(5) == 9999 and Drive().state == protocol.IDLE


def test_emergency_input_cuts_at_once_and_latches():
    drive = Drive()
    run(drive, 0, 800, (200, -200))
    drive.emergency_input(True)
    assert (drive.left, drive.right) == (0, 0) and drive.state == protocol.EMERGENCY  # no ramp
    run(drive, 800, 1200, (200, 200))
    assert (drive.left, drive.right) == (0, 0)  # commands are ignored
    assert not drive.release()  # not while the switch is still open
    drive.emergency_input(False)
    run(drive, 1200, 1400, (200, 200))
    assert (drive.left, drive.right) == (0, 0) and drive.state == protocol.EMERGENCY  # closing the switch is not enough
    assert drive.release() and not drive.release()
    run(drive, 1400, 2200, (200, 200))
    assert drive.left == 200


def test_virtual_controller_speaks_the_protocol():
    controller = VirtualController()
    assert protocol.parse_report(controller.transmit().splitlines()[0]) == protocol.Info(
        1, 2, 400, 600, 200
    )  # announces itself
    controller.advance(50, protocol.drive(1, 100, 100) + b"$D,2,300,0*00\n" + protocol.frame("D,3,300,0"))
    controller.advance(700)
    reports = [protocol.parse_report(line) for line in controller.transmit().splitlines()]
    assert (
        len(reports) == 7
        and reports[-1].seq == 1
        and reports[-1].rejected == 2
        and reports[-1].state == protocol.WATCHDOG
    )
    assert controller.outputs == (0, 0) and controller.now_ms == 700
    controller.advance(710, protocol.query(9))
    assert isinstance(protocol.parse_report(controller.transmit().splitlines()[0]), protocol.Info)


# ---------------------------------------------------------------------- links
def test_virtual_link_health_and_emergency():
    chair = Chair()
    link = chair.link()
    assert not link.healthy(0.0) and link.describe() == "no telemetry" and link.state == "?"
    t = 0.0
    while t < 2.0:
        link.drive(0.5, 0.5, t)
        link.poll(t)
        t += 0.05
    assert link.healthy(t) and link.info is not None and link.telemetry.state == protocol.DRIVING
    assert link.last_command == (128, 128) and chair.controller.outputs == (128, 128) and chair.moving
    assert chair.x > 0.4 and abs(chair.y) < 1e-6 and chair.distance == pytest.approx(chair.x)  # straight ahead
    chair.controller.emergency_input(True)
    link.poll(t + 0.2)
    assert link.emergency and not chair.moving and "emergency" in link.describe()
    chair.controller.emergency_input(False)
    link.release(t + 0.3)
    link.poll(t + 0.5)
    assert not link.emergency
    assert not link.healthy(t + 5.0)  # nothing heard for five seconds


def test_the_chair_turns_and_stops_when_the_host_goes_away():
    chair = Chair()
    link = chair.link()
    for i in range(40):
        link.drive(-0.35, 0.35, i * 0.05)  # turn left on the spot
        link.poll(i * 0.05)
    assert chair.heading > 0.5 and abs(chair.x) < 0.02 and abs(chair.y) < 0.02
    link.poll(2.0 + 1.0)  # no command for a second: the controller's watchdog has stopped the wheels
    assert not chair.moving and link.telemetry.state == protocol.WATCHDOG
    assert "heading" in chair.describe()


def test_null_link_and_link_factory():
    link = open_link("none")
    assert isinstance(link, NullLink)
    link.drive(1.0, -1.0, 0.0)
    link.poll(0.0)
    assert link.healthy(0.1) and link.telemetry.left == 255 and link.telemetry.right == -255 and not link.emergency
    link.stop(0.2)
    link.poll(0.2)
    assert link.telemetry.state == protocol.IDLE
    assert isinstance(open_link("sim"), VirtualLink) and isinstance(open_link("/dev/null-port"), SerialLink)
    with pytest.raises(NotImplementedError):
        MotorLink().drive(0, 0, 0.0)
    with open_link("none") as entered:
        assert isinstance(entered, NullLink)


@pytest.mark.skipif(not hasattr(__import__("os"), "openpty"), reason="needs pseudo-terminals")
def test_serial_link_talks_to_a_controller_over_a_real_serial_device():
    """pyserial on one end of a pseudo-terminal, the controller's logic on the other."""
    pytest.importorskip("serial")
    import os
    import threading
    import time

    master, slave = os.openpty()
    os.set_blocking(master, False)
    link = SerialLink(os.ttyname(slave))
    link.open()  # puts the terminal into raw mode; before that it would echo the controller's own greeting back to it
    controller = VirtualController()
    running = threading.Event()
    running.set()
    started = time.monotonic()

    def board() -> None:
        while running.is_set():
            try:
                received = os.read(master, 4096)
            except BlockingIOError:
                received = b""
            controller.step(round((time.monotonic() - started) * 1000), received)
            outgoing = controller.transmit()
            if outgoing:
                os.write(master, outgoing)
            time.sleep(0.002)

    thread = threading.Thread(target=board, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and (
            link.info is None or link.telemetry is None or link.telemetry.state != protocol.DRIVING
        ):
            now = time.monotonic()
            link.query(now)
            link.drive(0.5, -0.5, now)
            link.poll(now)
            time.sleep(0.02)
        assert link.info == protocol.Info(1, 2, 400, 600, 200) and link.healthy(time.monotonic())
        assert link.telemetry.state == protocol.DRIVING and link.telemetry.rejected == 0 and link.damaged == 0
        # Stop sending: the controller's watchdog must report, and the wheels must come to rest.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            link.poll(time.monotonic())
            if link.telemetry.state == protocol.WATCHDOG and (link.telemetry.left, link.telemetry.right) == (0, 0):
                break
            time.sleep(0.02)
        assert link.telemetry.state == protocol.WATCHDOG and (link.telemetry.left, link.telemetry.right) == (0, 0)
        assert link.telemetry.seq > 0  # it had been receiving our commands
    finally:
        link.close()
        running.clear()
        thread.join(timeout=1.0)
        os.close(master)
        os.close(slave)
    with pytest.raises(LinkError):
        SerialLink("/dev/no-such-port-here").open()
    with pytest.raises(LinkError):
        SerialLink("/dev/no-such-port-here").drive(0.0, 0.0, 0.0)

"""The sketch itself, compiled for the host against a mock of the Arduino API and run.

Nothing in the sketch is changed or stubbed for these tests: ``harness.cpp``
includes ``wheelchair_controller.ino`` as it is. What runs here is the code that
runs on the board, minus the board.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
from pathlib import Path

import pytest

from nwc.link import protocol
from nwc.link.controller import DriveConfig, VirtualController

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "tests" / "firmware" / "harness.cpp"
COMPILER = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")

pytestmark = pytest.mark.skipif(COMPILER is None, reason="no C++ compiler on this machine")


def build(tmp_path_factory, name: str, *defines: str) -> Path:
    binary = tmp_path_factory.mktemp("firmware") / name
    command = [
        COMPILER,
        "-std=gnu++11",  # what the Arduino AVR core passes to its compiler
        "-Wall",
        "-Wextra",
        "-Werror",
        "-x",
        "c++",
        *[f"-D{d}" for d in defines],
        f"-I{HARNESS.parent}",  # Arduino.h is the mock
        str(HARNESS),
        "-o",
        str(binary),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return binary


@pytest.fixture(scope="module")
def sketch(tmp_path_factory) -> Path:
    return build(tmp_path_factory, "default")


def run(binary: Path, script: list[str]) -> tuple[str, dict[int, tuple[int, int]], str]:
    """Returns the boot line, the effective motor duties at every change and everything sent."""
    out = subprocess.run(
        [str(binary)], input="\n".join(script) + "\n", capture_output=True, text=True, check=True
    ).stdout.splitlines()
    outputs, sent = {}, []
    for line in out[1:]:
        head, _, text = line.partition(" | ")
        ms, left, right = (int(v) for v in head.split())
        outputs[ms] = (left, right)
        sent.append(text)
    return out[0].partition(" | ")[2], outputs, "".join(sent)


def scenario() -> dict[int, bytes]:
    """Driving, a reversal, damaged lines, silence long enough for the watchdog, resuming, a stop."""
    events = {1: protocol.query(1)}
    seq = 2
    for t in range(50, 1000, 100):
        events[t] = protocol.drive(seq, 200, 120)
        seq += 1
    for t in range(1000, 1500, 100):
        events[t] = protocol.drive(seq, -180, 60)
        seq += 1
    events[1550] = b"$D,99,300,0*00\n"
    events[1560] = protocol.frame("D,7,300,0")
    events[1570] = protocol.frame("D,7,10,20,")
    events[1580] = protocol.frame("D, 7,10,20")
    events[1590] = b"\x00\xff garbage without an end"
    events[1600] = b"\n"
    events[1610] = b"$" + b"D" * 80 + b"\n"
    for t in range(2100, 2600, 100):
        events[t] = protocol.drive(seq, 255, -255)
        seq += 1
    events[2650] = protocol.stop(seq)
    return events


def as_script(events: dict[int, bytes], until: int, inputs: dict[int, tuple[int, int]] | None = None) -> list[str]:
    script, last = [], 0
    for t in sorted(set(events) | set(inputs or {})):
        if t - 1 > last:
            script.append(f"run {t - 1}")
        if inputs and t in inputs:
            script.append(f"input {inputs[t][0]} {inputs[t][1]}")
        script.append(f"raw {t} {events.get(t, b'').hex()}" if events.get(t) else f"at {t}")
        last = t
    script.append(f"run {until}")
    return script


def python_trace(
    events: dict[int, bytes], until: int, controller: VirtualController, inputs: dict[int, bool] | None = None
):
    sent = [controller.transmit().decode()]
    outputs = {}
    for t in range(1, until + 1):
        if inputs and t in inputs:
            controller.emergency_input(inputs[t])
        controller.step(t, events.get(t, b""))
        outputs[t] = controller.outputs
        sent.append(controller.transmit().decode())
    return sent[0], outputs, "".join(sent[1:])


def test_the_sketch_and_its_python_port_agree_at_every_millisecond(sketch):
    events = scenario()
    boot, changes, sent = run(sketch, as_script(events, 3200))
    py_boot, outputs, py_sent = python_trace(events, 3200, VirtualController())
    assert boot.replace(";", "\n") == py_boot and sent.replace(";", "\n") == py_sent  # byte for byte
    current = (0, 0)
    for t in range(1, 3201):
        current = changes.get(t, current)
        assert outputs[t] == current, f"at {t} ms"
    reports = [protocol.parse_report(line.encode()) for line in sent.split(";") if line]
    assert all(r is not None for r in reports) and isinstance(reports[0], protocol.Info)  # the answer to the query
    telemetry = [r for r in reports if isinstance(r, protocol.Telemetry)]
    assert telemetry[-1].rejected == 6 and {r.state for r in telemetry} == {"I", "D", "W"}


def test_what_the_bridge_sees_follows_the_commands(sketch):
    """Pin states, turned back into the duty the motors get: forward, backward, turn, stop."""
    script = []
    plan = {0: (255, 255), 1000: (-255, -255), 2200: (128, -128), 3200: (0, 0)}
    for t in range(0, 4000, 100):
        speeds = plan[max(k for k in plan if k <= t)]
        script.append(f"raw {t + 1} {protocol.drive(t // 100 + 1, *speeds).hex()}")
        script.append(f"run {t + 100}")
    _, changes, _ = run(sketch, script)

    def duty(ms: int) -> tuple[int, int]:
        return changes[max(k for k in changes if k <= ms)]

    assert duty(900) == (255, 255) and duty(2100) == (-255, -255) and duty(3100) == (128, -128) and duty(3900) == (0, 0)
    values = [changes[k][0] for k in sorted(changes)]
    assert max(abs(b - a) for a, b in itertools.pairwise(values)) <= 2  # what reaches the motor never jumps


def test_wiring_variants_compile_and_behave_the_same(tmp_path_factory, sketch):
    events = {t: protocol.drive(t // 100 + 1, 200, -90) for t in range(1, 1500, 100)}
    _, default, _ = run(sketch, as_script(events, 1500))
    on_enable = build(tmp_path_factory, "enable", "NWC_PWM_ON_ENABLE=1")
    _, enable, _ = run(on_enable, as_script(events, 1500))
    assert default == enable and default[max(default)] == (200, -90)
    inverted = build(tmp_path_factory, "inverted", "NWC_INVERT_LEFT=1", "NWC_MIN_DUTY=60")
    _, flipped, _ = run(inverted, as_script(events, 1500))
    left, right = flipped[max(flipped)]
    assert left == -(60 + 195 * 200 // 255) and right == -(
        60 + 195 * 90 // 255
    )  # reversed on the left, and nothing below the minimum duty


def test_emergency_stop_input(tmp_path_factory):
    with_input = build(tmp_path_factory, "estop", "NWC_ESTOP_PIN=7")
    events = {t: protocol.drive(t // 100 + 1, 200, 200) for t in range(1, 3000, 100)}
    events[2450] = protocol.release(99)
    inputs = {900: (7, 1), 1600: (7, 0)}  # the switch opens at 0.9 s and is closed again at 1.6 s
    boot, changes, sent = run(with_input, as_script(events, 3000, inputs))
    assert changes[900] == (0, 0)  # cut in the same pass of the loop, no ramp
    assert all(
        v == (0, 0) for t, v in changes.items() if 900 <= t < 2450
    )  # closing the switch alone does not restart anything
    assert changes[max(changes)] == (200, 200)  # after the release from the host it drives again
    states = [protocol.parse_report(line.encode()).state for line in sent.split(";") if line]
    assert "E" in states and states[-1] == "D"
    py_boot, _, py_sent = python_trace(
        events, 3000, VirtualController(DriveConfig(), emergency_input=True), {900: True, 1600: False}
    )
    assert sent.replace(";", "\n") == py_sent and boot.replace(";", "\n") == py_boot

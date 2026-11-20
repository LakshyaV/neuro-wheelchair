"""Closed loop: a simulated wearer, the real engine, the virtual motor controller and chair.

Time is counted in samples, so every scenario plays out identically on every run.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import pytest

from nwc.control.supervisor import ARMED, DISARMED, FAULT, HALTED
from nwc.dsp.frontend import FrontendConfig
from nwc.io.stream import StreamInfo
from nwc.link.motor import NullLink
from nwc.runtime.engine import BASELINE, READY, Engine, EngineConfig, Status
from nwc.sim.scenario import Act, simulate_drive

SWITCH = EngineConfig(scheme="switch", baseline_s=12.0)
DIRECT = EngineConfig(scheme="direct", baseline_s=15.0)


# The simulated wearer's commands vary in strength from one time to the next. Where a test is about
# what the system does with a command, and not about that variability, the strength is fixed.
FORWARD, LEFT = ("forward", 1.0), ("left", 1.0)


def press(at: float) -> list[Act]:
    return [Act(at, "clench", True), Act(at + 0.5, "clench", False)]


# ---------------------------------------------------------------------- the switch scheme needs no model
def test_one_switch_drives_the_chair_without_any_training():
    # armed at 18 s: settle until 19.5, then forward is on offer until 21.5, left until 23.5, ...
    plan = [Act(18.0, "arm"), *press(20.0), *press(25.0), *press(29.3), *press(33.0), *press(37.0)]
    log = simulate_drive(plan, until_s=42.0, seed=5, config=SWITCH)
    forward = log.first("forward", after=20.0)
    assert forward is not None and forward - 20.0 < 0.4  # the clench is noticed within 0.4 s
    stopped = log.first("stop", after=25.0)
    assert stopped - 25.0 < 0.4 and log.motion_at(24.9) == "forward"
    # stopped at ~25.2: settle until ~26.7, forward on offer until ~28.7, then left: the clench at 29.3 takes left
    left = log.first("left", after=29.0)
    assert left is not None and left - 29.3 < 0.4
    assert log.first("stop", after=left) - left == pytest.approx(SWITCH.switch.turn_s, abs=0.1)  # a turn ends by itself
    assert log.motion_at(34.0) == "forward" and log.motion_at(38.0) == "stop"
    chair = log.chair
    assert chair.distance > 3.0 and abs(chair.heading) > 0.3 and not chair.moving
    assert {t.state for t in log.transitions} == {DISARMED, ARMED}  # in this scheme a clench never halts


def test_nothing_moves_before_the_baseline_is_complete_and_the_system_is_armed():
    plan = [
        Act(3.0, "arm"),
        *press(8.0),
        *press(22.0),
    ]  # armed too early; clenches before anyone has armed successfully
    log = simulate_drive(plan, until_s=30.0, seed=5, config=SWITCH)
    assert all(t.motion == "stop" for t in log.transitions) and log.chair.distance == 0.0
    assert log.state_at(29.0) == DISARMED
    phases = [status.phase for _, status in log.statuses]
    assert phases[0] == BASELINE and phases[-1] == READY
    done = next(now for now, status in log.statuses if status.phase == READY)
    assert 12.0 + 2.0 < done < 12.0 + 8.0  # the baseline needs twelve seconds of clean frames
    progress = [status.baseline_progress for _, status in log.statuses]
    assert progress[0] == 0.0 and progress[-1] == 1.0 and all(b >= a for a, b in itertools.pairwise(progress))
    assert "baseline" in log.statuses[0][1].line() and "disarmed" in log.statuses[-1][1].line()


# ---------------------------------------------------------------------- direct driving with a trained model
@pytest.fixture(scope="module")
def drive(trained):
    """One long drive with the trained model: the wearer asks for forward, lets go, asks for left, then clenches."""
    plan = [
        Act(22.0, "arm"),
        Act(60.0, "intent", FORWARD), Act(68.0, "intent", "rest"),
        Act(76.0, "intent", LEFT), Act(82.0, "intent", "rest"),
        Act(90.0, "intent", FORWARD), *press(96.0), Act(97.0, "intent", "rest"),
        Act(104.0, "intent", FORWARD), Act(112.0, "arm"),
    ]  # fmt: skip
    from conftest import SIM

    return simulate_drive(plan, until_s=124.0, bundle=trained.bundle, sim=SIM, seed=21, session_seed=3, config=DIRECT)


def test_the_chair_stays_put_while_the_wearer_rests(drive):
    """Thirty-eight seconds armed with nobody asking for anything."""
    assert drive.state_at(30.0) == ARMED
    assert all(t.motion == "stop" for t in drive.transitions if t.at < 60.0)


def test_commands_start_the_chair_and_letting_go_stops_it(drive):
    forward = drive.first("forward", after=60.0)
    assert forward is not None and forward - 60.0 < 6.0
    assert drive.motion_at(67.5) == "forward"
    stopped = drive.first("stop", after=68.0)
    assert stopped - 68.0 < 4.0
    left = drive.first("left", after=76.0)
    assert left is not None and left - 76.0 < 6.0
    assert drive.first("stop", after=82.0) - 82.0 < 4.0
    assert drive.first("right") is None and drive.first("backward") is None  # nothing that was not asked for


def test_a_clench_halts_at_once_and_only_arming_ends_it(drive):
    assert drive.motion_at(95.9) == "forward"
    halted = next(t for t in drive.transitions if t.state == HALTED)
    assert halted.at - 96.0 < 0.4 and halted.motion == "stop" and "emergency" in halted.note
    # the wearer asks for forward again from 104 s: nothing happens while halted
    assert all(t.motion == "stop" for t in drive.transitions if 96.5 < t.at < 112.0)
    assert drive.state_at(111.0) == HALTED
    again = drive.first("forward", after=112.0)  # the operator arms at 112 s; the command is still being held
    assert again is not None and again - 112.0 < 6.0
    assert not drive.chair.moving  # and at the end of the drive the engine leaves the chair standing


def test_the_wheels_follow_the_decisions(drive):
    assert drive.wheels_moving_at(66.0) and not drive.wheels_moving_at(59.0) and not drive.wheels_moving_at(100.0)
    assert drive.chair.distance > 2.0


# ---------------------------------------------------------------------- faults
def test_a_lifted_electrode_stops_the_chair():
    plan = [Act(18.0, "arm"), *press(20.0), Act(22.0, "contact", ("AF7", False)), Act(40.0, "contact", ("AF7", True))]
    log = simulate_drive(
        plan, until_s=52.0, seed=5, config=replace(SWITCH, switch=replace(SWITCH.switch, max_run_s=60.0))
    )
    fault = next(t for t in log.transitions if t.state == FAULT)
    assert fault.at - 22.0 < 7.0 and fault.motion == "stop" and "AF7" in fault.note  # noticed within seconds
    assert log.motion_at(21.9) == "forward" and all(t.motion == "stop" for t in log.transitions if 30.0 < t.at < 40.0)
    assert not log.wheels_moving_at(35.0)
    assert (
        log.state_at(51.0) == ARMED and log.motion_at(51.0) == "stop"
    )  # contact is back: armed again, but the drive does not resume


def test_a_silent_headset_stops_the_chair():
    plan = [Act(18.0, "arm"), *press(20.0), Act(23.0, "stall", 6.0)]
    log = simulate_drive(
        plan, until_s=36.0, seed=5, config=replace(SWITCH, switch=replace(SWITCH.switch, max_run_s=60.0))
    )
    fault = next(t for t in log.transitions if t.state == FAULT)
    assert 23.0 < fault.at < 23.0 + 1.5 and "no data" in fault.note
    assert not log.wheels_moving_at(25.0)
    assert log.state_at(35.5) == ARMED and log.motion_at(35.5) == "stop"


def test_the_controller_stops_the_wheels_when_the_host_program_dies():
    log = simulate_drive(
        [Act(18.0, "arm"), *press(20.0)],
        until_s=24.0,
        seed=5,
        config=replace(SWITCH, switch=replace(SWITCH.switch, max_run_s=60.0)),
    )
    chair = log.chair
    # simulate_drive ends with a stop. Drive on for a second, then let the host fall silent.
    link, now = log.engine.link, 26.0
    for i in range(20):
        link.drive(0.6, 0.6, now + i * 0.05)
    assert chair.moving
    chair.controller.advance(chair.controller.now_ms + 1000)  # a second passes without a single command
    assert not chair.moving and chair.controller.drive.state == "W"


def test_lost_packets_do_not_stop_the_drive():
    plan = [Act(18.0, "arm"), *press(20.0), *press(26.0)]
    log = simulate_drive(plan, until_s=30.0, seed=5, config=SWITCH, loss=0.03, counted=True)
    stats = log.engine.assembler.stats
    assert stats.counted and stats.filled > 100 and 0.01 < stats.loss < 0.06
    assert log.first("forward", after=20.0) - 20.0 < 0.6 and log.first("stop", after=26.0) - 26.0 < 0.6
    assert all(t.state != FAULT for t in log.transitions)


# ---------------------------------------------------------------------- set-up
def test_engine_set_up_is_checked(trained):
    info = StreamInfo(FrontendConfig().channels, 256.0)
    with pytest.raises(ValueError, match="needs a trained model"):
        Engine(info, NullLink(), config=EngineConfig(scheme="direct"))
    with pytest.raises(ValueError, match="not one of the model's commands"):
        Engine(info, NullLink(), trained.bundle, EngineConfig(scheme="switch:dance"))
    with pytest.raises(ValueError, match="unknown scheme"):
        Engine(info, NullLink(), config=EngineConfig(scheme="telepathy"))
    with pytest.raises(ValueError, match="channels"):
        Engine(StreamInfo(("A", "B", "C", "D"), 256.0), NullLink())
    with pytest.raises(ValueError, match="does not match"):
        Engine(info, NullLink(), trained.bundle, frontend=FrontendConfig(hop_s=0.5))
    switch = Engine(info, NullLink(), trained.bundle, EngineConfig(scheme=f"switch:{trained.bundle.classes[1]}"))
    assert switch.scheme.name == "switch" and switch.scheme.config.switch == trained.bundle.classes[1]
    automatic = Engine(info, NullLink(), trained.bundle)
    assert automatic.scheme.name == trained.bundle.recommended_scheme.partition(":")[0]
    assert Engine(info, NullLink()).scheme.name == "switch"  # no model: the switch scheme
    assert not automatic.arm(0.0) and automatic.phase == BASELINE  # cannot be armed before the baseline
    assert Status().line().startswith("baseline")


def test_the_baseline_can_be_repeated(trained):
    plan = [Act(22.0, "arm"), Act(30.0, "note")]
    log = simulate_drive(plan, until_s=32.0, bundle=trained.bundle, seed=8, config=DIRECT)
    engine = log.engine
    assert engine.phase == READY and log.state_at(31.0) == ARMED
    engine.rebaseline(32.0)
    assert engine.phase == BASELINE and engine.supervisor.state == DISARMED and not engine.arm(32.1)


def test_the_driving_loop_on_the_wall_clock():
    """The loop as `nwc drive` runs it: a source paced in real time, wall-clock time, a link that answers."""
    import os
    import time

    from nwc.runtime.live import StatusLine, drive
    from nwc.sim.eeg import EEGSynth
    from nwc.sim.source import SimulatedSource

    source = SimulatedSource(EEGSynth(seed=1), realtime=True, seed=1)
    engine = Engine(source.info, NullLink(), config=EngineConfig(scheme="switch", baseline_s=0.5))
    seen = []
    started = time.monotonic()
    with open(os.devnull, "w") as screen:
        status = drive(
            engine,
            source,
            auto_arm=True,
            duration_s=4.5,
            show=StatusLine(screen),
            on_step=lambda now, s: seen.append(s.state),
        )
    elapsed = time.monotonic() - started
    assert 4.4 < elapsed < 10.0  # paced by the stream, not as fast as the computer can go
    received = engine.assembler.stats.received
    assert 0.85 * 256 * elapsed < received < 1.05 * 256 * elapsed
    assert status.phase == READY and ARMED in seen and status.link_ok and status.stream["delivered_hz"] > 200
    assert engine.link.last_command == (0, 0)  # the loop always ends with a stop

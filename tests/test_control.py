from __future__ import annotations

import numpy as np
import pytest

from nwc.control.arbiter import Arbiter, ArbiterConfig, replay
from nwc.control.scheme import DirectConfig, DirectScheme, Inputs, SwitchConfig, SwitchScheme
from nwc.control.supervisor import ARMED, DISARMED, FAULT, HALTED, Supervisor, SupervisorConfig

CLASSES = ("rest", "forward", "left")
RATE = 4.0


def log_p(*values: float) -> np.ndarray:
    return np.log(np.asarray(values, dtype=float))


SURE_FORWARD, SURE_LEFT, SURE_REST = log_p(0.1, 0.8, 0.1), log_p(0.1, 0.1, 0.8), log_p(0.9, 0.05, 0.05)
UNSURE = log_p(0.34, 0.33, 0.33)


def feed(arbiter: Arbiter, frames: list) -> list[str]:
    return [arbiter.push(frame) for frame in frames]


# ---------------------------------------------------------------------- arbiter
def test_a_held_command_switches_on_and_a_released_one_off():
    arbiter = Arbiter(CLASSES, ArbiterConfig(), RATE)
    out = feed(arbiter, [SURE_FORWARD] * 16 + [SURE_REST] * 16)
    on, off = out.index("forward"), 16 + out[16:].index("rest")
    assert 3 <= on <= 12 and out[on:16] == ["forward"] * (
        16 - on
    )  # needs a second or two of consistent evidence, then holds
    assert 16 < off <= 26 and set(out[off:]) == {"rest"}
    assert np.isclose(arbiter.posterior.sum(), 1.0)


def test_single_frames_do_nothing_and_unsure_frames_never_start_anything():
    arbiter = Arbiter(CLASSES, ArbiterConfig(), RATE)
    assert set(feed(arbiter, [SURE_REST] * 8 + [SURE_FORWARD] + [SURE_REST] * 8)) == {"rest"}  # one odd frame
    assert set(feed(arbiter, [UNSURE] * 200)) == {"rest"}  # an honest "I don't know", for as long as you like
    assert set(feed(arbiter, [SURE_FORWARD, SURE_LEFT] * 40)) == {"rest"}  # evidence that keeps changing its mind


def test_unusable_frames_are_bridged_briefly_and_never_start_anything():
    arbiter = Arbiter(CLASSES, ArbiterConfig(hold_s=2.5), RATE)
    assert set(feed(arbiter, [None] * 100)) == {"rest"}
    feed(arbiter, [SURE_FORWARD] * 16)
    assert arbiter.active == "forward"
    before = arbiter.posterior.copy()
    assert set(feed(arbiter, [None] * 8)) == {"forward"}  # a blink spoils two seconds of windows: the command holds
    assert np.array_equal(arbiter.posterior, before)  # and nothing was learned or forgotten meanwhile
    assert feed(arbiter, [SURE_FORWARD] * 4)[-1] == "forward"
    out = feed(arbiter, [None] * 40)  # the signal stays away
    ended = out.index("rest")
    assert 10 <= ended <= 16  # held for two and a half seconds, then over within a second and a half
    # evidence that was building towards a command is not completed by unusable frames
    building = Arbiter(CLASSES, ArbiterConfig(hold_s=2.5, dwell_s=1.0), RATE)
    feed(building, [SURE_FORWARD] * 3)
    assert set(feed(building, [None] * 60)) == {"rest"}


def test_commands_hand_over_only_through_rest():
    arbiter = Arbiter(CLASSES, ArbiterConfig(refractory_s=1.0), RATE)
    feed(arbiter, [SURE_FORWARD] * 16)
    out = feed(arbiter, [SURE_LEFT] * 40)
    assert "left" in out
    first_left = out.index("left")
    between = out[:first_left]
    assert "rest" in between and between.index("rest") < first_left  # forward -> rest -> left, never forward -> left
    rest_run = first_left - between.index("rest")
    assert rest_run >= 4  # at least the refractory second at four frames per second


def test_arbiter_configuration_and_replay():
    with pytest.raises(ValueError):
        Arbiter(("forward", "rest"))
    for bad in ({"on": 0.4, "off": 0.5}, {"integration_s": 0.0}, {"gain": -1.0}):
        with pytest.raises(ValueError):
            ArbiterConfig(**bad)
    frames = np.array([SURE_FORWARD] * 16 + [SURE_REST] * 16)
    usable = np.ones(32, dtype=bool)
    active = replay(CLASSES, frames, usable, ArbiterConfig(), RATE)
    assert set(active) == {0, 1} and active[15] == 1 and active[-1] == 0
    assert replay(CLASSES, frames, ~usable, ArbiterConfig(), RATE).max() == 0  # nothing usable, nothing happens


# ---------------------------------------------------------------------- schemes
def test_direct_scheme_drives_while_held_and_halts_on_a_clench():
    scheme = DirectScheme(DirectConfig({"forward": "forward", "left": "left"}))
    assert scheme.step(Inputs(0.0, "rest")).motion == "stop"
    assert scheme.step(Inputs(1.0, "forward")).motion == "forward" and scheme.step(Inputs(2.0, "left")).motion == "left"
    assert scheme.step(Inputs(3.0, "sing")).motion == "stop"  # a command without a binding does nothing
    halted = scheme.step(Inputs(4.0, "forward", pressed=True, press=True))
    assert halted.motion == "stop" and halted.halt
    with pytest.raises(ValueError):
        DirectConfig({"forward": "sideways"})


def run_switch(scheme: SwitchScheme, until: float, presses: list[float], step: float = 0.05, command=lambda t: "rest"):
    log, t, presses = [], 0.0, sorted(presses)
    while t < until:
        press = bool(presses) and presses[0] <= t
        if press:
            presses.pop(0)
        log.append((round(t, 2), scheme.step(Inputs(t, command(t), pressed=press, press=press))))
        t += step
    return log


def at(log, seconds: float):
    return next(request for t, request in log if t >= seconds)


def test_switch_scheme_offers_takes_and_stops():
    config = SwitchConfig(options=("forward", "left", "right"), scan_s=2.0, turn_s=1.2, max_run_s=8.0, settle_s=1.5)
    scheme = SwitchScheme(config)
    scheme.reset(0.0)
    # offers after the settle time: forward 1.5-3.5, left 3.5-5.5, right 5.5-7.5, forward again...
    log = run_switch(scheme, 30.0, presses=[4.0, 9.0, 12.0, 26.0])
    assert at(log, 1.0).note == "wait" and "forward" in at(log, 2.0).note and "left" in at(log, 3.8).note
    assert at(log, 4.1).motion == "left" and at(log, 5.0).motion == "left"  # the press at 4.0 took what was on offer
    assert at(log, 5.4).motion == "stop"  # a turn ends by itself after 1.2 s
    # stopped at 5.2: quiet until 6.7, then forward 6.7-8.7, left 8.7-10.7. The press at 9.0 takes left again.
    assert at(log, 9.1).motion == "left"
    # stopped at 10.2: quiet until 11.7, forward on offer: the press at 12.0 drives...
    assert at(log, 12.1).motion == "forward" and at(log, 19.0).motion == "forward"
    assert at(log, 20.2).motion == "stop"  # ...until the time limit, eight seconds later
    assert all(not request.halt for _, request in log)  # in this scheme the clench is the switch, not an emergency stop
    # a press while moving stops, and only stops
    log = run_switch(SwitchScheme(config), 12.0, presses=[2.0, 4.0, 4.5])
    assert (
        at(log, 2.1).motion == "forward" and at(log, 4.1).motion == "stop" and at(log, 4.6).motion == "stop"
    )  # the third press falls in the settle time
    with pytest.raises(ValueError):
        SwitchConfig(options=("stop",))


def test_switch_scheme_with_a_trained_command_as_the_switch():
    scheme = SwitchScheme(SwitchConfig(switch="forward", settle_s=1.0, scan_s=2.0))
    held = lambda t: "forward" if 2.0 <= t < 3.0 or 6.0 <= t < 9.0 else "rest"  # noqa: E731
    log = run_switch(scheme, 12.0, presses=[], command=held)
    assert at(log, 2.1).motion == "forward"  # the command coming on is a press
    assert (
        at(log, 5.9).motion == "forward" and at(log, 6.1).motion == "stop"
    )  # coming on again stops; holding it does not start anything new
    assert at(log, 8.5).motion == "stop"
    halted = SwitchScheme(SwitchConfig(switch="forward")).step(Inputs(5.0, "rest", pressed=True, press=True))
    assert halted.halt  # with a trained switch the clench is the emergency stop again


# ---------------------------------------------------------------------- supervisor
def step(supervisor, now, motion="stop", **kwargs):
    from nwc.control.scheme import Request

    defaults = {"data": True, "lost": [], "link_ok": True, "pressed": False}
    defaults.update(kwargs)
    halt = defaults.pop("halt", False)
    return supervisor.step(now, Request(motion, halt=halt), **defaults)


def test_nothing_moves_until_armed():
    supervisor = Supervisor()
    assert step(supervisor, 0.0, "forward").motion == "stop" and supervisor.state == DISARMED
    assert supervisor.arm(1.0) and step(supervisor, 1.1, "forward").motion == "forward"
    supervisor.disarm()
    assert step(supervisor, 1.2, "forward").motion == "stop"


@pytest.mark.parametrize(
    ("trouble", "reason"),
    [({"data": False}, "no data"), ({"lost": ["AF7"]}, "AF7"), ({"link_ok": False}, "motor controller")],
)
def test_faults_stop_the_chair_and_clear_by_themselves(trouble, reason):
    supervisor = Supervisor(SupervisorConfig(stale_s=1.0, recover_s=2.0))
    supervisor.arm(0.0)
    t = 0.0
    while t < 5.0:
        assert step(supervisor, t, "forward").motion == "forward"
        t += 0.1
    stopped_at = None
    while t < 8.0:  # the trouble lasts three seconds
        decision = step(supervisor, t, "forward", **trouble)
        if decision.motion == "stop" and stopped_at is None:
            stopped_at = t
            assert decision.state == FAULT and reason in decision.reason
        t += 0.1
    assert stopped_at is not None and stopped_at - 5.0 <= 1.11  # at once, or after the staleness limit for missing data
    assert not supervisor.arm(t)  # cannot be armed over a fault
    recovered_at = None
    while t < 12.0:
        decision = step(supervisor, t, "forward")
        if decision.motion == "forward" and recovered_at is None:
            recovered_at = t
        t += 0.1
    assert recovered_at is not None and 1.9 <= recovered_at - 8.0 <= 2.3  # healthy for two seconds first


def test_emergency_stop_latches_until_rearmed():
    supervisor = Supervisor(SupervisorConfig(rearm_hold_s=2.0))
    supervisor.arm(0.0)
    assert step(supervisor, 1.0, "forward").motion == "forward"
    assert step(supervisor, 1.1, "stop", halt=True, pressed=True).state == HALTED
    for t in np.arange(1.2, 30.0, 0.1):  # it stays halted for as long as nobody re-arms
        assert step(supervisor, float(t), "forward").motion == "stop"
    assert supervisor.state == HALTED and supervisor.arm(30.0) and step(supervisor, 30.1, "forward").motion == "forward"

    # the wearer can re-arm alone: let go, hold a clench for two seconds, let go
    step(supervisor, 31.0, "stop", halt=True, pressed=True)
    step(supervisor, 31.5, "forward", pressed=True)  # still the same clench: does not count
    step(supervisor, 32.0, "forward")
    step(supervisor, 33.0, "forward", pressed=True)
    step(supervisor, 34.0, "forward", pressed=True)
    assert supervisor.state == HALTED  # a short clench is not enough
    step(supervisor, 34.1, "forward")
    assert supervisor.state == HALTED
    step(supervisor, 35.0, "forward", pressed=True)
    step(supervisor, 37.1, "forward", pressed=True)
    assert step(supervisor, 37.2, "forward").state == ARMED

    never = Supervisor(SupervisorConfig(rearm_hold_s=0.0))  # the gesture can be switched off
    never.arm(0.0)
    step(never, 1.0, halt=True, pressed=True)
    step(never, 2.0)
    step(never, 3.0, pressed=True)
    step(never, 9.0, pressed=True)
    assert step(never, 9.1).state == HALTED


def test_uninterrupted_motion_is_limited():
    supervisor = Supervisor(SupervisorConfig(max_motion_s=10.0))
    supervisor.arm(0.0)
    motions = [step(supervisor, t / 10, "forward").motion for t in range(0, 150)]
    assert motions[99] == "forward" and set(motions[102:]) == {"stop"}  # ten seconds, then no more
    assert step(supervisor, 15.0, "left").motion == "stop"  # switching commands does not reset the limit
    assert (
        step(supervisor, 15.1, "stop").motion == "stop" and step(supervisor, 15.2, "forward").motion == "forward"
    )  # letting go does

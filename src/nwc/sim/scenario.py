"""Scripted drives in simulation: a virtual wearer, the real engine, a virtual chair.

A plan says what the virtual wearer does and when. The engine sees nothing but
the resulting EEG stream, exactly as it would from a headset, and its commands
go to the virtual motor controller that moves the chair. Time is counted in
samples, so a scenario plays out identically every time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..decode.bundle import Bundle
from ..runtime.engine import Engine, EngineConfig, Status
from .chair import Chair
from .eeg import EEGSynth, SimConfig
from .source import SimulatedSource


@dataclass(frozen=True)
class Act:
    """Something the virtual wearer (or the operator) does at a time in seconds."""

    at: float
    what: str  # "intent", "clench", "contact", "arm", "disarm", "stall", "note"
    value: object = None


@dataclass
class Transition:
    at: float
    state: str
    motion: str
    note: str


@dataclass
class DriveLog:
    transitions: list[Transition] = field(default_factory=list)
    statuses: list[tuple[float, Status]] = field(default_factory=list)
    chair: Chair | None = None
    engine: Engine | None = None

    def motion_at(self, seconds: float) -> str:
        motion = "stop"
        for now, status in self.statuses:
            if now > seconds:
                break
            motion = status.motion
        return motion

    def state_at(self, seconds: float) -> str:
        state = ""
        for now, status in self.statuses:
            if now > seconds:
                break
            state = status.state
        return state

    def first(self, motion: str, after: float = 0.0) -> float | None:
        """When ``motion`` first began after a given time."""
        return next((t.at for t in self.transitions if t.at >= after and t.motion == motion), None)

    def wheels_moving_at(self, seconds: float) -> bool:
        moving = False
        for now, status in self.statuses:
            if now > seconds:
                break
            moving = "wheels +0/+0" not in status.link and "no telemetry" not in status.link
        return moving


def simulate_drive(
    plan: list[Act],
    *,
    until_s: float,
    bundle: Bundle | None = None,
    sim: SimConfig | None = None,
    seed: int = 0,
    session_seed: int | None = None,
    config: EngineConfig | None = None,
    loss: float = 0.0,
    counted: bool = False,
) -> DriveLog:
    """Play ``plan`` against the engine and a virtual chair."""
    sim = sim or SimConfig()
    source = SimulatedSource(EEGSynth(sim, seed=seed, session_seed=session_seed), seed=seed, loss=loss, counted=counted)
    chair = Chair()
    engine = Engine(source.info, chair.link(), bundle, config)
    log = DriveLog(chair=chair, engine=engine)
    todo = sorted(plan, key=lambda act: act.at)
    fs = sim.fs
    stalled_until = 0.0
    previous = ("", "")
    while source.synth.position / fs < until_s:
        now = source.synth.position / fs
        while todo and todo[0].at <= now:
            act = todo.pop(0)
            if act.what in ("intent", "clench", "contact"):
                source.at(now, act.what, act.value)
            elif act.what == "arm":
                engine.arm(now)
            elif act.what == "disarm":
                engine.disarm(now)
            elif act.what == "stall":  # the headset goes silent for a while: samples are produced but never arrive
                stalled_until = now + float(act.value)  # type: ignore[arg-type]
        chunk = source.read()
        if now < stalled_until:
            chunk = None
        status = engine.feed(chunk, now)
        log.statuses.append((now, status))
        current = (status.state, status.motion)
        if current != previous:
            log.transitions.append(Transition(now, status.state, status.motion, status.reason or status.note))
            previous = current
    end = source.synth.position / fs
    engine.close(end)
    engine.link.poll(end + 1.0)  # let the controller's ramp run out, so the log shows where the chair came to rest
    return log

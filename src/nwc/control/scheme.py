"""Driving schemes: how decoded signals become motions.

Two signals are available: the *command* the arbiter currently holds (from the
trained classifier), and the *switch*, a jaw clench.

``direct``
    Each command drives while it is held: think "forward", the chair moves;
    let go, it stops. A clench is the emergency stop and latches. This scheme
    needs commands that evaluation has shown to be reliable for the wearer.

``switch``
    One switch does everything. While the chair stands, the options are offered
    in turn; a press takes the one on offer. While it moves, a press stops it.
    Turns end by themselves after a fixed time, driving ends on the next press or
    after ``max_run_s``. The switch is the clench by default and can be a trained
    command instead. This scheme works with no training at all.

A scheme only ever asks for a motion. Whether the chair is allowed to carry it
out is the supervisor's decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STOP = "stop"
MOTIONS = ("stop", "forward", "backward", "left", "right")


@dataclass
class Inputs:
    """What a scheme sees at one step."""

    now: float  # seconds on the stream's clock
    command: str = "rest"  # the arbiter's active command
    pressed: bool = False  # the clench is being held
    press: bool = False  # the clench has just begun


@dataclass
class Request:
    motion: str = STOP
    halt: bool = False  # the wearer asked for an emergency stop
    note: str = ""  # one line for the display


class Scheme:
    name = ""

    def reset(self, now: float = 0.0) -> None:
        raise NotImplementedError

    def step(self, inputs: Inputs) -> Request:
        raise NotImplementedError


@dataclass(frozen=True)
class DirectConfig:
    #: Which motion each command asks for. Commands not listed do nothing.
    bindings: dict[str, str] = field(
        default_factory=lambda: {"forward": "forward", "left": "left", "right": "right", "backward": "backward"}
    )

    def __post_init__(self) -> None:
        unknown = [m for m in self.bindings.values() if m not in MOTIONS]
        if unknown:
            raise ValueError(f"unknown motions {unknown}; choose from {MOTIONS}")


class DirectScheme(Scheme):
    name = "direct"

    def __init__(self, config: DirectConfig | None = None):
        self.config = config or DirectConfig()

    def reset(self, now: float = 0.0) -> None:
        pass

    def step(self, inputs: Inputs) -> Request:
        if inputs.press or inputs.pressed:
            return Request(STOP, halt=True, note="clench: emergency stop")
        motion = self.config.bindings.get(inputs.command, STOP)
        return Request(motion, note=inputs.command if motion != STOP else "")


@dataclass(frozen=True)
class SwitchConfig:
    options: tuple[str, ...] = ("forward", "left", "right")
    scan_s: float = 2.0  # how long each option is on offer
    turn_s: float = 1.2  # a turn ends by itself after this
    max_run_s: float = 8.0  # driving ends by itself after this
    settle_s: float = 1.5  # nothing is offered for this long after a motion ends
    switch: str = "clench"  # "clench" or the name of a trained command

    def __post_init__(self) -> None:
        unknown = [m for m in self.options if m not in MOTIONS or m == STOP]
        if unknown or not self.options:
            raise ValueError(f"options must be motions other than stop, got {self.options}")


class SwitchScheme(Scheme):
    name = "switch"

    def __init__(self, config: SwitchConfig | None = None):
        self.config = config or SwitchConfig()
        self.reset()

    def reset(self, now: float = 0.0) -> None:
        self.motion = STOP
        self.offer = 0
        self._motion_until = 0.0
        self._quiet_until = now + self.config.settle_s
        self._offer_since = self._quiet_until  # the first option gets its full time once offering starts
        self._was_on = False

    def _pressed(self, inputs: Inputs) -> bool:
        """A fresh press of the configured switch."""
        if self.config.switch == "clench":
            return inputs.press
        on = inputs.command == self.config.switch
        fresh, self._was_on = on and not self._was_on, on
        return fresh

    def step(self, inputs: Inputs) -> Request:
        cfg, now = self.config, inputs.now
        # With a trained command as the switch, the clench keeps its role as emergency stop.
        if cfg.switch != "clench" and (inputs.press or inputs.pressed):
            self._stop(now)
            return Request(STOP, halt=True, note="clench: emergency stop")
        pressed = self._pressed(inputs)
        if self.motion != STOP:
            if pressed or now >= self._motion_until:
                self._stop(now)
                return Request(STOP, note="stopped")
            return Request(self.motion, note=f"{self.motion}: press to stop")
        if now < self._quiet_until:
            return Request(STOP, note="wait")
        if now - self._offer_since >= cfg.scan_s:
            self.offer = (self.offer + 1) % len(cfg.options)
            self._offer_since = now
        option = cfg.options[self.offer]
        if pressed:
            self.motion = option
            self._motion_until = now + (cfg.max_run_s if option in ("forward", "backward") else cfg.turn_s)
            return Request(self.motion, note=f"{self.motion}: press to stop")
        return Request(STOP, note=f"offering {option}: press to take it")

    def _stop(self, now: float) -> None:
        self.motion = STOP
        self._quiet_until = now + self.config.settle_s
        self._offer_since = self._quiet_until
        self.offer = 0

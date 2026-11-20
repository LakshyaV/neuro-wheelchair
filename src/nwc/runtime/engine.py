"""The engine: everything between arriving samples and wheel commands, in one object.

::

    chunk -> assembler -> front-end -+-> fast frames -> clench detector --------+
                                     |                                          v
                                     +-> slow frames -> quality -> model -> arbiter -> scheme
                                                           |                           |
                                                           v                           v
                                                    contact monitor ----------> supervisor -> motor link

The engine has no thread and no clock of its own. The caller hands it whatever
arrived together with the current time, and gets back a :class:`Status`. Live
driving passes wall-clock time; tests and replays pass stream time, which makes
a simulated drive exactly repeatable.

Every session starts with a baseline: the wearer sits relaxed until enough
clean frames have been collected. The baseline becomes the reference for the
classifier and the resting level for the clench detector, and nothing can be
armed before it is complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..control.arbiter import REST, Arbiter, ArbiterConfig
from ..control.scheme import STOP, DirectConfig, DirectScheme, Inputs, Request, Scheme, SwitchConfig, SwitchScheme
from ..control.supervisor import ARMED, DISARMED, Supervisor, SupervisorConfig
from ..decode.bundle import Bundle
from ..decode.features import mean_reference
from ..dsp.frontend import Envelope, Frontend, FrontendConfig
from ..dsp.quality import ContactMonitor, QualityConfig, QualityMeter
from ..dsp.reflex import ClenchConfig, ClenchDetector, resting_level
from ..io.stream import Chunk, StreamAssembler, StreamInfo
from ..link.motor import MotorLink

BASELINE, READY = "baseline", "ready"


def _default_speeds() -> dict[str, tuple[float, float]]:
    # (left wheel, right wheel) as fractions of full scale
    return {"forward": (0.6, 0.6), "backward": (-0.4, -0.4), "left": (-0.35, 0.35), "right": (0.35, -0.35)}


@dataclass(frozen=True)
class EngineConfig:
    scheme: str = "auto"  # "auto", "direct", "switch" or "switch:<command>"
    baseline_s: float = 20.0
    command_interval_s: float = 0.1
    speeds: dict[str, tuple[float, float]] = field(default_factory=_default_speeds)
    direct: DirectConfig = field(default_factory=DirectConfig)
    switch: SwitchConfig = field(default_factory=SwitchConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    # Left unset, these come from the model bundle (or their defaults without one).
    arbiter: ArbiterConfig | None = None
    clench: ClenchConfig | None = None
    quality: QualityConfig | None = None


@dataclass
class Status:
    now: float = 0.0
    phase: str = BASELINE
    baseline_progress: float = 0.0
    state: str = DISARMED
    reason: str = ""
    command: str = REST
    posterior: dict[str, float] = field(default_factory=dict)
    pressed: bool = False
    clench_db: float = 0.0
    motion: str = STOP
    note: str = ""
    wheels: tuple[float, float] = (0.0, 0.0)
    link: str = ""
    link_ok: bool = False
    lost: list[str] = field(default_factory=list)
    usable: bool = False
    stream: dict = field(default_factory=dict)

    def line(self) -> str:
        """One line for a terminal."""
        if self.phase == BASELINE:
            return f"baseline {100 * self.baseline_progress:3.0f} %: sit relaxed, eyes open | link: {self.link}"
        posterior = " ".join(f"{name[:4]} {value:.2f}" for name, value in self.posterior.items())
        clench = "CLENCH" if self.pressed else f"{self.clench_db:+3.0f} dB"
        tail = self.reason or self.note
        return (
            f"{self.state:8s} {self.motion:8s} | {posterior or 'no model':34s} | jaw {clench:6s} | link: {self.link}"
            + (f" | {tail}" if tail else "")
        )


class Engine:
    def __init__(
        self,
        info: StreamInfo,
        link: MotorLink,
        bundle: Bundle | None = None,
        config: EngineConfig | None = None,
        frontend: FrontendConfig | None = None,
    ):
        self.config = config or EngineConfig()
        self.bundle = bundle
        self.link = link
        if bundle is not None:
            if frontend is not None and frontend.fingerprint != bundle.frontend.fingerprint:
                raise ValueError("the front-end given does not match the one the model was trained with")
            frontend = bundle.frontend
        self.frontend_config = frontend or FrontendConfig()
        self.frontend_config.matches(info)
        cfg = self.config
        self._quality_config = cfg.quality or (bundle.quality if bundle else QualityConfig())
        self._clench_config = cfg.clench or (bundle.clench if bundle else ClenchConfig())
        arbiter_config = cfg.arbiter or (bundle.arbiter if bundle else ArbiterConfig())

        self.assembler = StreamAssembler(info)
        self.frontend = Frontend(self.frontend_config)
        self.quality = QualityMeter(self.frontend_config, self._quality_config)
        self.contact = ContactMonitor(self.frontend_config, self._quality_config)
        self.arbiter = Arbiter(bundle.classes, arbiter_config, self.frontend_config.frame_rate) if bundle else None
        self.scheme = self._build_scheme()
        self.supervisor = Supervisor(cfg.supervisor)
        self.clench: ClenchDetector | None = None
        self.reference: np.ndarray | None = None

        self.phase = BASELINE
        self._baseline_cosp: list[np.ndarray] = []
        self._baseline_fast: list[Envelope] = []
        self._baseline_need = max(1, round(cfg.baseline_s * self.frontend_config.frame_rate))
        self._last_sent: float | None = None
        self._last_motion = STOP
        self._was_armed = False
        self.status = Status()

    # ------------------------------------------------------------------ set-up
    def _build_scheme(self) -> Scheme:
        cfg, bundle = self.config, self.bundle
        name = cfg.scheme
        automatic = name == "auto"
        if automatic:
            name = bundle.recommended_scheme if bundle else "switch"
        kind, _, switch = name.partition(":")
        if kind == "direct":
            if bundle is None:
                raise ValueError("the direct scheme needs a trained model: give a bundle or use the switch scheme")
            allowed = set(bundle.reliable) if automatic and bundle.reliable else set(bundle.classes)
            bindings = {command: motion for command, motion in cfg.direct.bindings.items() if command in allowed}
            if not bindings:
                raise ValueError(f"none of the model's commands {bundle.classes[1:]} is bound to a motion")
            return DirectScheme(DirectConfig(bindings))
        if kind == "switch":
            switch_config = cfg.switch
            if switch:
                if bundle is None or switch not in bundle.classes[1:]:
                    raise ValueError(f"the switch command {switch!r} is not one of the model's commands")
                switch_config = SwitchConfig(
                    cfg.switch.options,
                    cfg.switch.scan_s,
                    cfg.switch.turn_s,
                    cfg.switch.max_run_s,
                    cfg.switch.settle_s,
                    switch,
                )
            elif switch_config.switch != "clench" and (
                bundle is None or switch_config.switch not in bundle.classes[1:]
            ):
                raise ValueError(f"the switch command {switch_config.switch!r} is not one of the model's commands")
            return SwitchScheme(switch_config)
        raise ValueError(f"unknown scheme {cfg.scheme!r}: use auto, direct, switch or switch:<command>")

    # ------------------------------------------------------------------ operator
    def arm(self, now: float) -> bool:
        """Allow motion. Refused until the baseline is complete; releases a latched controller."""
        if self.phase != READY:
            return False
        if self.link.emergency:
            self.link.release(now)
        armed = self.supervisor.arm(now)
        if armed:
            self._reset_decoding(now)
        return armed

    def disarm(self, now: float) -> None:
        self.supervisor.disarm()
        self._send(STOP, now, force=True)

    def rebaseline(self, now: float) -> None:
        """Start the baseline again, e.g. after re-seating the headband. Disarms."""
        self.disarm(now)
        self.phase = BASELINE
        self._baseline_cosp.clear()
        self._baseline_fast.clear()
        self.contact.reset()

    def close(self, now: float) -> None:
        self._send(STOP, now, force=True)

    def _reset_decoding(self, now: float) -> None:
        if self.arbiter is not None:
            self.arbiter.reset()
        self.scheme.reset(now)

    # ------------------------------------------------------------------ baseline
    def _finish_baseline(self, now: float) -> None:
        self.reference = mean_reference(np.concatenate(self._baseline_cosp))
        fast = Envelope.concatenate(self._baseline_fast, self.frontend_config)
        self.clench = ClenchDetector(self.frontend_config.channels, resting_level(fast), self._clench_config)
        self._baseline_cosp.clear()
        self._baseline_fast.clear()
        self.phase = READY
        self._reset_decoding(now)

    # ------------------------------------------------------------------ one step
    def feed(self, chunk: Chunk | None, now: float) -> Status:
        """Take in what arrived (possibly nothing) at time ``now`` and act on it."""
        frames, envelope = self.frontend.push(self.assembler.push(chunk))
        quality = self.quality(frames)
        self.contact.push(quality, frames.valid)
        data = bool(frames.valid.any())
        press = False

        if self.phase == BASELINE:
            if len(frames):
                self._baseline_cosp.append(frames.cosp[quality.usable])
            if len(envelope):
                self._baseline_fast.append(envelope)
            collected = sum(len(part) for part in self._baseline_cosp)
            if collected >= self._baseline_need and any(part.valid.any() for part in self._baseline_fast):
                self._finish_baseline(now)
        else:
            press = any(event.pressed for event in self.clench.push(envelope))
            if self.arbiter is not None:
                for i in range(len(frames)):
                    usable = bool(quality.usable[i])
                    self.arbiter.push(
                        self.bundle.model.log_proba(frames.cosp[i : i + 1], self.reference)[0] if usable else None
                    )

        pressed = self.clench.pressed if self.clench is not None else False
        command = self.arbiter.active if self.arbiter is not None else REST
        request = (
            self.scheme.step(Inputs(now, command, pressed, press))
            if self.phase == READY
            else Request(STOP, note="baseline")
        )

        self.link.poll(now)
        link_ok = self.link.healthy(now) and not self.link.emergency
        decision = self.supervisor.step(
            now, request, data=data, lost=self.contact.lost_channels, link_ok=link_ok, pressed=pressed
        )
        if self.link.emergency and decision.state != ARMED and not decision.reason.startswith("emergency"):
            decision.reason = "the controller's emergency stop is active"
        armed = decision.state == ARMED
        if self._was_armed and not armed:
            self._reset_decoding(now)  # nothing that was in progress may resume by itself
        self._was_armed = armed
        self._send(decision.motion, now)

        collected = sum(len(part) for part in self._baseline_cosp)
        self.status = Status(
            now=now,
            phase=self.phase,
            baseline_progress=1.0 if self.phase == READY else min(collected / self._baseline_need, 1.0),
            state=decision.state,
            reason=decision.reason if decision.state != ARMED else "",
            command=command,
            posterior=dict(zip(self.arbiter.classes, self.arbiter.posterior.tolist()))
            if self.arbiter is not None
            else {},
            pressed=pressed,
            clench_db=self.clench.excess_db if self.clench is not None else 0.0,
            motion=decision.motion,
            note=decision.reason if decision.state == ARMED else "",
            wheels=self.config.speeds.get(decision.motion, (0.0, 0.0)),
            link=self.link.describe(),
            link_ok=link_ok,
            lost=self.contact.lost_channels,
            usable=bool(quality.usable[-1]) if len(quality) else self.status.usable,
            stream=self.assembler.stats.as_dict(),
        )
        return self.status

    def _send(self, motion: str, now: float, force: bool = False) -> None:
        """Send the wheel speeds for ``motion``: at a steady rate, and at once when the motion changes."""
        due = self._last_sent is None or now - self._last_sent >= self.config.command_interval_s
        if not (due or force or motion != self._last_motion):
            return
        left, right = self.config.speeds.get(motion, (0.0, 0.0))
        self.link.drive(left, right, now)
        self._last_sent, self._last_motion = now, motion

"""The safety supervisor: the only place that can let the chair move.

A scheme asks for a motion; the supervisor grants it only while every condition
for safe driving holds. It starts disarmed and falls back to *stop* on anything
it does not like:

``DISARMED``  nothing moves until someone arms the system
``ARMED``     requests are passed on, subject to the motion time limit
``FAULT``     data stopped arriving, an electrode lost contact or the motor
              controller went silent. Clears by itself once things have been
              healthy for ``recover_s``
``HALTED``    the wearer asked for an emergency stop. Stays until re-armed, by
              the operator or by holding a clench for ``rearm_hold_s``

Stopping never depends on this code alone: the motor controller has its own
watchdog and halts when commands stop arriving, whatever the reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scheme import STOP, Request

DISARMED, ARMED, FAULT, HALTED = "disarmed", "armed", "fault", "halted"


@dataclass(frozen=True)
class SupervisorConfig:
    stale_s: float = 1.0
    recover_s: float = 2.0
    max_motion_s: float = 10.0
    rearm_hold_s: float = 2.0
    require_link: bool = True


@dataclass
class Decision:
    motion: str
    state: str
    reason: str = ""


class Supervisor:
    def __init__(self, config: SupervisorConfig | None = None):
        self.config = config or SupervisorConfig()
        self.state = DISARMED
        self.reason = "not armed"
        self._last_data: float | None = None
        self._healthy_since: float | None = None
        self._moving_since: float | None = None
        self._limit = False
        self._hold_since: float | None = None
        self._released = False

    # ------------------------------------------------------------------ operator
    def arm(self, now: float) -> bool:
        """Arm from DISARMED or HALTED. Refused while a fault is present."""
        if self.state in (DISARMED, HALTED):
            self.state, self.reason = ARMED, ""
            self._healthy_since = now
            self._moving_since, self._limit = None, False
            return True
        return self.state == ARMED

    def disarm(self) -> None:
        self.state, self.reason = DISARMED, "not armed"

    # ------------------------------------------------------------------ one step
    def step(
        self,
        now: float,
        request: Request,
        *,
        data: bool,
        lost: list[str] | None = None,
        link_ok: bool = True,
        pressed: bool = False,
    ) -> Decision:
        """Decide what the chair may do now.

        ``data`` says whether a fully received frame arrived since the previous step;
        ``lost`` names electrodes without contact; ``pressed`` is the clench.
        """
        cfg = self.config
        if data or self._last_data is None:
            self._last_data = now  # the first step starts the clock
        problem = ""
        if now - self._last_data > cfg.stale_s:
            problem = "no data from the headset"
        elif lost:
            problem = f"electrode {', '.join(lost)} lost contact"
        elif cfg.require_link and not link_ok:
            problem = "no answer from the motor controller"

        if self.state == HALTED:
            self._rearm_gesture(now, pressed)
        elif self.state in (ARMED, FAULT):
            if problem:
                self.state, self.reason, self._healthy_since = FAULT, problem, None
            elif self.state == FAULT:
                self._healthy_since = now if self._healthy_since is None else self._healthy_since
                if now - self._healthy_since >= cfg.recover_s:
                    self.state, self.reason = ARMED, ""
                    self._moving_since, self._limit = None, False
            if self.state == ARMED and request.halt:
                self.state, self.reason = HALTED, "emergency stop"
                self._hold_since, self._released = None, False

        if self.state != ARMED:
            self._moving_since = None
            return Decision(STOP, self.state, self.reason)

        # Armed and healthy: pass the request on, within the time limit for uninterrupted motion.
        if request.motion == STOP:
            self._moving_since, self._limit = None, False
            return Decision(STOP, ARMED, request.note)
        if self._limit:
            return Decision(STOP, ARMED, "motion time limit: let go first")
        self._moving_since = now if self._moving_since is None else self._moving_since
        if now - self._moving_since > cfg.max_motion_s:
            self._limit = True
            return Decision(STOP, ARMED, "motion time limit: let go first")
        return Decision(request.motion, ARMED, request.note)

    def _rearm_gesture(self, now: float, pressed: bool) -> None:
        """After an emergency stop: let go, then hold a clench for ``rearm_hold_s`` and let go again."""
        cfg = self.config
        if cfg.rearm_hold_s <= 0:
            return
        if not self._released:
            self._released = not pressed  # the clench that caused the stop has to end first
            return
        if pressed:
            self._hold_since = now if self._hold_since is None else self._hold_since
            return
        if self._hold_since is not None and now - self._hold_since >= cfg.rearm_hold_s:
            self.arm(now)
        self._hold_since = None

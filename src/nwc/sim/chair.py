"""A virtual wheelchair: the controller's outputs integrated into a path.

Two driven wheels on a common axis, no slip, no inertia beyond the controller's
own ramps. It is enough to see where a sequence of commands takes the chair and
to check, in tests, that the chair stops when it must.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..link.controller import DriveConfig, VirtualController
from ..link.motor import VirtualLink


@dataclass(frozen=True)
class ChairConfig:
    max_speed: float = 0.8  # metres per second at full output
    track: float = 0.55  # metres between the drive wheels


class Chair:
    def __init__(self, config: ChairConfig | None = None, drive: DriveConfig | None = None):
        self.config = config or ChairConfig()
        self.controller = VirtualController(drive)
        self.x = self.y = self.heading = 0.0  # metres, metres, radians (0 along +x, positive to the left)
        self.distance = 0.0  # metres travelled, forwards and backwards
        self.path: list[tuple[float, float]] = [(0.0, 0.0)]
        self._last_ms = 0

    def link(self) -> VirtualLink:
        """A motor link whose controller moves this chair."""
        return VirtualLink(self.controller, on_step=self._follow)

    def _follow(self, controller: VirtualController) -> None:
        """Advance the pose to the controller's clock using its current outputs."""
        dt = (controller.now_ms - self._last_ms) / 1000.0
        self._last_ms = controller.now_ms
        if dt <= 0:
            return
        left, right = (v / 255.0 * self.config.max_speed for v in controller.outputs)
        speed = (left + right) / 2.0
        self.heading += (right - left) / self.config.track * dt
        self.x += speed * math.cos(self.heading) * dt
        self.y += speed * math.sin(self.heading) * dt
        self.distance += abs(speed) * dt
        if math.hypot(self.x - self.path[-1][0], self.y - self.path[-1][1]) > 0.02:
            self.path.append((self.x, self.y))

    @property
    def moving(self) -> bool:
        return any(self.controller.outputs)

    def describe(self) -> str:
        return f"x {self.x:+.2f} m, y {self.y:+.2f} m, heading {math.degrees(self.heading):+.0f} deg"

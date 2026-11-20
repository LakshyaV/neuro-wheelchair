"""Choosing the arbiter's thresholds from held-out calibration data.

Classification accuracy says how often a two-second window is labelled
correctly. What matters in a wheelchair is different: does the chair start when
asked, how long does it take, and how often does it start when nobody asked?

This module answers that by replay. Every calibration run is played back through
the arbiter, frame by frame, with probabilities from a model that was fitted
without that run, and the result is compared with what the wearer was being
asked to do at each moment:

* a command trial is a *hit* if its command switched on during the cue (or
  shortly after), and *wrong* if a different command switched on;
* a *false start* is a command switching on while rest was being asked for;
* *latency* is the time from the cue to the command switching on.

Thresholds are then picked from a grid. Settings whose false starts and wrong
commands stay within their limits compete on hit rate, and among those within one
standard error of the best the one that errs least wins. The commands that switched on in most of
their trials at that setting are the ones the report calls reliable, and the
driving scheme it recommends follows from how many there are. With none, that is
the switch scheme, which needs no decoded commands at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..decode.dataset import Recording
from ..dsp.frontend import FrontendConfig
from ..io.session import CUE
from .arbiter import REST, ArbiterConfig, replay


@dataclass
class ReplayScore:
    trials: int = 0
    hits: int = 0
    wrong: int = 0
    false_starts: int = 0
    rest_minutes: float = 0.0
    latencies: list[float] = field(default_factory=list)
    per_command: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.trials if self.trials else 0.0

    @property
    def wrong_rate(self) -> float:
        return self.wrong / self.trials if self.trials else 0.0

    @property
    def false_per_min(self) -> float:
        return self.false_starts / self.rest_minutes if self.rest_minutes > 0 else 0.0

    @property
    def latency_s(self) -> float:
        return float(np.median(self.latencies)) if self.latencies else float("nan")

    def summary(self) -> dict:
        return {
            "trials": self.trials,
            "hit_rate": round(self.hit_rate, 4),
            "wrong_rate": round(self.wrong_rate, 4),
            "false_starts_per_min": round(self.false_per_min, 4),
            "rest_minutes": round(self.rest_minutes, 3),
            "median_latency_s": None if not self.latencies else round(self.latency_s, 3),
            "per_command": {k: {a: round(b, 4) for a, b in v.items()} for k, v in self.per_command.items()},
        }


def score_replay(
    recordings: list[Recording],
    held_out: list[np.ndarray],
    classes: tuple[str, ...],
    config: ArbiterConfig,
    frontend: FrontendConfig,
    *,
    grace_s: float = 1.5,
) -> ReplayScore:
    """Replay every run through the arbiter and score it against the cue timeline."""
    score = ReplayScore()
    counts = {name: {"trials": 0, "hits": 0} for name in classes[1:]}
    grace = round(grace_s * frontend.fs)
    for rec, log_proba in zip(recordings, held_out):
        ends = rec.frames.index * frontend.hop + frontend.window
        for run in np.unique(rec.run[rec.run > 0]):
            rows = np.flatnonzero((rec.run == run) & np.isfinite(log_proba[:, 0]))
            if len(rows) == 0:
                continue
            active = replay(classes, log_proba[rows], rec.quality.usable[rows], config, frontend.frame_rate)
            end = ends[rows]
            asked = rec.cue[rows]
            resting = asked == REST
            started = np.flatnonzero((active != 0) & (np.concatenate(([0], active[:-1])) != active))
            for event in rec.events:
                if event.kind != CUE or event.run != run or event.label == REST or event.label not in classes:
                    continue
                k = classes.index(event.label)
                span = (end > event.start) & (end <= event.stop + grace)
                if not span.any():
                    continue
                on = span & (active == k)
                score.trials += 1
                counts[event.label]["trials"] += 1
                if on.any():
                    score.hits += 1
                    counts[event.label]["hits"] += 1
                    score.latencies.append(float(end[np.argmax(on)] - event.start) / frontend.fs)
                if (span & (active != 0) & (active != k)).any():
                    score.wrong += 1
                resting &= ~((end > event.stop) & (end <= event.stop + grace))  # a command may take a moment to let go
            score.false_starts += int(resting[started].sum())
            score.rest_minutes += float(resting.sum()) / frontend.frame_rate / 60.0
    score.per_command = {
        name: {"trials": c["trials"], "hit_rate": c["hits"] / c["trials"] if c["trials"] else 0.0}
        for name, c in counts.items()
    }
    return score


@dataclass
class Tuning:
    config: ArbiterConfig
    score: ReplayScore
    within_target: bool  # some setting kept false starts and wrong commands within their limits
    target_false_per_min: float
    reliable: tuple[str, ...]  # commands that switched on in most of their trials at the chosen setting
    table: list[dict] = field(default_factory=list)

    @property
    def recommended_scheme(self) -> str:
        """What the replay supports.

        ``direct`` needs at least two commands that can be relied on. A single
        reliable command can serve as the switch of the switch scheme
        (``switch:<command>``). With none, the switch is the jaw clench.
        """
        if not self.within_target or not self.reliable:
            return "switch"
        if len(self.reliable) == 1:
            return f"switch:{self.reliable[0]}"
        return "direct"

    def summary(self) -> dict:
        return {
            "arbiter": {
                k: getattr(self.config, k)
                for k in ("integration_s", "gain", "on", "off", "dwell_s", "release_s", "refractory_s")
            },
            "replay": self.score.summary(),
            "within_target": self.within_target,
            "target_false_starts_per_min": self.target_false_per_min,
            "reliable_commands": list(self.reliable),
            "recommended_scheme": self.recommended_scheme,
        }


GRID_ON = (0.6, 0.7, 0.8, 0.9, 0.95)
GRID_DWELL = (0.25, 0.5)
GRID_INTEGRATION = (1.0, 2.0, 3.0)
GRID_GAIN = (0.25, 0.5, 1.0)


def tune(
    recordings: list[Recording],
    held_out: list[np.ndarray],
    classes: tuple[str, ...],
    frontend: FrontendConfig,
    *,
    target_false_per_min: float = 0.5,
    max_wrong_rate: float = 0.15,
    min_hit_rate: float = 0.7,
    base: ArbiterConfig | None = None,
) -> Tuning:
    base = base or ArbiterConfig()
    rows = []
    for integration in GRID_INTEGRATION:
        for gain in GRID_GAIN:
            for on in GRID_ON:
                for dwell in GRID_DWELL:
                    config = replace(
                        base,
                        integration_s=integration,
                        gain=gain,
                        on=on,
                        off=round(max(on - 0.3, 0.3), 3),
                        dwell_s=dwell,
                    )
                    rows.append((config, score_replay(recordings, held_out, classes, config, frontend)))
    acceptable = [r for r in rows if r[1].false_per_min <= target_false_per_min and r[1].wrong_rate <= max_wrong_rate]
    if acceptable:
        # Hit rates within one standard error of the best are not really different. Among those,
        # take the setting that errs least, and of equals the more demanding one.
        best = max(r[1].hit_rate for r in acceptable)
        trials = max(acceptable[0][1].trials, 1)
        margin = float(np.sqrt(max(best * (1.0 - best), 1e-9) / trials))
        close = [r for r in acceptable if r[1].hit_rate >= best - margin]
        config, score = min(
            close, key=lambda r: (r[1].false_per_min + r[1].wrong_rate, -r[0].on, -r[0].dwell_s, -r[1].hit_rate)
        )
    else:
        config, score = min(rows, key=lambda r: (r[1].false_per_min + r[1].wrong_rate, -r[1].hit_rate))
    reliable = (
        tuple(name for name, c in score.per_command.items() if c["trials"] and c["hit_rate"] >= min_hit_rate)
        if acceptable
        else ()
    )
    table = [
        {
            "integration_s": c.integration_s,
            "gain": c.gain,
            "on": c.on,
            "dwell_s": c.dwell_s,
            "hit_rate": round(s.hit_rate, 3),
            "false_per_min": round(s.false_per_min, 3),
            "wrong_rate": round(s.wrong_rate, 3),
        }
        for c, s in rows
    ]
    return Tuning(config, score, bool(acceptable), target_false_per_min, reliable, table)

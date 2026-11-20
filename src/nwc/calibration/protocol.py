"""The calibration protocol: what is asked of the wearer, and in which order.

The schedule fixes the two properties that make a recording usable for training
and for an honest estimate of how well the result will work:

* Commands are interleaved. Every run contains every command the same number of
  times in random order, with rest in between. A classifier trained on one
  recording per command learns to tell recordings apart (electrode contact, time
  of day) at least as readily as mental states, and scores well for the wrong
  reason. With interleaving there is nothing but the command to tell trials apart.
* Runs are separate. Evaluation holds out whole runs, so overlapping windows of
  one trial never sit on both sides of a split.

Every session starts with a baseline: the wearer sits relaxed with eyes open.
The baseline is the reference that all later activity is expressed against, and
the resting level for the stop reflex. A short block of jaw clenches follows, to
set that reflex's threshold for this person.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REST = "rest"
CLENCH = "clench"


@dataclass(frozen=True)
class ProtocolConfig:
    commands: tuple[str, ...] = ("forward", "left", "right")
    baseline_s: float = 30.0
    runs: int = 3
    trials_per_command: int = 8
    cue_s: float = 5.0
    pause_s: tuple[float, float] = (2.5, 3.5)
    clench_trials: int = 6
    clench_s: float = 1.5
    seed: int = 0

    def __post_init__(self) -> None:
        if REST in self.commands or CLENCH in self.commands:
            raise ValueError(f"{REST!r} and {CLENCH!r} are reserved names")
        if len(set(self.commands)) != len(self.commands) or not self.commands:
            raise ValueError("commands must be distinct and non-empty")

    @property
    def labels(self) -> tuple[str, ...]:
        """Classes the decoder is trained on: rest first, then the commands."""
        return (REST, *self.commands)


#: What the wearer is asked to do for each cue. Any consistent strategy works; these are starting points.
HINTS = {
    REST: "relax, eyes open, let your mind wander",
    CLENCH: "clench your jaw firmly",
    "forward": "count down from 300 in sevens",
    "left": "silently recite the words of a song",
    "right": "walk through your home in your mind, room by room",
    "backward": "close your eyes and breathe slowly",
}


@dataclass(frozen=True)
class Step:
    kind: str  # "baseline" | "cue" | "pause" | "clench" | "break"
    label: str
    duration: float
    run: int = 0
    trial: int = -1

    @property
    def hint(self) -> str:
        return HINTS.get(self.label, "")


def _shuffled_trials(labels: tuple[str, ...], repeats: int, rng: np.random.Generator) -> list[str]:
    """Each label ``repeats`` times, in random order, never three alike in a row."""
    order = list(labels) * repeats
    for _ in range(200):
        rng.shuffle(order)
        if all(not (a == b == c) for a, b, c in zip(order, order[1:], order[2:])):
            break
    return order


def build_schedule(config: ProtocolConfig) -> list[Step]:
    rng = np.random.default_rng(config.seed)
    steps: list[Step] = [Step("baseline", REST, config.baseline_s)]
    for _ in range(config.clench_trials):
        steps.append(Step("pause", REST, float(rng.uniform(*config.pause_s))))
        steps.append(Step("clench", CLENCH, config.clench_s))
    trial = 0
    for run in range(1, config.runs + 1):
        if run > 1:
            steps.append(Step("break", REST, 0.0, run))
        for label in _shuffled_trials(config.labels, config.trials_per_command, rng):
            steps.append(Step("pause", REST, float(rng.uniform(*config.pause_s)), run))
            steps.append(Step("cue", label, config.cue_s, run, trial))
            trial += 1
    steps.append(Step("pause", REST, config.pause_s[0], config.runs))
    return steps


def duration(steps: list[Step]) -> float:
    return float(sum(step.duration for step in steps))


def describe(config: ProtocolConfig) -> str:
    steps = build_schedule(config)
    trials = sum(step.kind == "cue" for step in steps)
    return (
        f"{config.runs} runs of {config.trials_per_command} trials for each of {', '.join(config.labels)} "
        f"({trials} trials of {config.cue_s:g} s), {config.clench_trials} jaw clenches, "
        f"{config.baseline_s:g} s baseline: about {duration(steps) / 60:.0f} minutes plus breaks"
    )

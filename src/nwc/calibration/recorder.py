"""The guided calibration recording.

Walks the wearer through a schedule (:mod:`nwc.calibration.protocol`), writes
every sample to a session as it arrives, and marks each step by the position of
its first and last sample.

Steps are timed on the stream itself: a five-second cue lasts until five seconds
of samples have arrived. For a healthy stream that is five seconds on the wall,
and it makes a recording from a replayed or simulated source identical to a live
one. Wall-clock time is only used to notice that the stream has stopped.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TextIO

from ..dsp.frontend import Frames, Frontend, FrontendConfig
from ..dsp.quality import QualityConfig, QualityMeter
from ..io.session import BASELINE, CUE, PAUSE, RUN, SessionWriter
from ..io.sources import Source
from ..io.stream import StreamAssembler
from .protocol import CLENCH, REST, ProtocolConfig, Step, build_schedule


class RecordingError(RuntimeError):
    """The recording could not be completed."""


class ConsoleUI:
    """Cues on a terminal. Subclass to show them elsewhere."""

    def __init__(self, stream: TextIO | None = None, *, bell: bool = True, interactive: bool = True):
        self.stream = stream or sys.stdout
        self.bell = bell
        self.interactive = interactive

    def say(self, text: str) -> None:
        print(text, file=self.stream, flush=True)

    def cue(self, step: Step, position: str) -> None:
        if step.kind == "cue" and step.label != REST:
            self.say(
                f"{position}  >>> {step.label.upper()} <<<   {step.hint}  ({step.duration:g} s)"
                + ("\a" if self.bell else "")
            )
        elif step.kind == "cue":
            self.say(
                f"{position}      rest            {step.hint}  ({step.duration:g} s)" + ("\a" if self.bell else "")
            )
        elif step.kind == "clench":
            self.say(f"{position}  >>> CLENCH <<<   {step.hint}  ({step.duration:g} s)" + ("\a" if self.bell else ""))
        elif step.kind == "baseline":
            self.say(f"baseline: sit relaxed with your eyes open and look straight ahead ({step.duration:g} s)")

    def pause(self, step: Step) -> None:
        pass  # silence between trials is the cue to let go

    def wait(self, text: str) -> None:
        """Between runs: let the wearer rest until the operator continues."""
        if self.interactive and sys.stdin.isatty():
            input(text)
        else:
            self.say(text)


@dataclass
class RecordingSummary:
    samples: int
    seconds: float
    trials: int
    delivered_hz: float
    loss: float
    usable_share: float
    lost_channels: list[str]


class Recorder:
    def __init__(
        self,
        source: Source,
        writer: SessionWriter,
        protocol: ProtocolConfig | None = None,
        ui: ConsoleUI | None = None,
        *,
        frontend: FrontendConfig | None = None,
        quality: QualityConfig | None = None,
        stall_s: float = 5.0,
    ):
        self.source, self.writer = source, writer
        self.protocol = protocol or ProtocolConfig()
        self.ui = ui or ConsoleUI()
        self.stall_s = stall_s
        self._frontend_config = frontend or FrontendConfig()
        self._assembler = StreamAssembler(source.info)
        self._frontend = Frontend(self._frontend_config)
        self._quality = QualityMeter(self._frontend_config, quality)
        self._usable = 0
        self._frames = 0
        self._out_of_range = None

    # ------------------------------------------------------------------ stream
    def _pump(self) -> int:
        """Read once, store what arrived and keep the running quality figures. Returns samples received."""
        chunk = self.source.read()
        if chunk is None or len(chunk) == 0:
            return 0
        self.writer.write(chunk)
        frames, _ = self._frontend.push(self._assembler.push(chunk))
        self._account(frames)
        return len(chunk)

    def _account(self, frames: Frames) -> None:
        if len(frames) == 0:
            return
        quality = self._quality(frames)
        self._frames += len(frames)
        self._usable += int(quality.usable.sum())
        counts = quality.out_of_range.sum(axis=0)
        self._out_of_range = counts if self._out_of_range is None else self._out_of_range + counts

    def _collect(self, seconds: float) -> tuple[int, int]:
        """Record ``seconds`` of stream. Returns the first and one-past-last row."""
        start = self.writer.samples
        target = start + round(seconds * self.source.info.fs)
        last_arrival = time.monotonic()
        while self.writer.samples < target:
            if self._pump():
                last_arrival = time.monotonic()
            elif self.source.exhausted:
                raise RecordingError("the stream ended before the recording was complete")
            elif time.monotonic() - last_arrival > self.stall_s:
                raise RecordingError(f"no samples for {self.stall_s:g} s: check the headset and the stream")
        return start, self.writer.samples

    # ------------------------------------------------------------------ protocol
    def run(self) -> RecordingSummary:
        steps = build_schedule(self.protocol)
        total = sum(step.kind == "cue" for step in steps)
        self.ui.say(f"recording {self.writer.path}")
        run_first: dict[int, int] = {}
        run_last: dict[int, int] = {}
        done = 0
        for step in steps:
            if step.kind == "break":
                self.writer.flush()
                self.ui.wait(
                    f"run {step.run - 1} of {self.protocol.runs} done. Rest, then press Enter for the next run. "
                )
                continue
            position = (
                f"[run {step.run}/{self.protocol.runs}  trial {done + 1}/{total}]"
                if step.kind == "cue"
                else "[clench]"
                if step.kind == "clench"
                else ""
            )
            if step.kind == "pause":
                self.ui.pause(step)
            else:
                self.ui.cue(step, position)
            first, last = self._collect(step.duration)
            if step.run:
                run_first.setdefault(step.run, first)
                run_last[step.run] = last
            kind = {"baseline": BASELINE, "cue": CUE, "pause": PAUSE, "clench": CLENCH}[step.kind]
            self.writer.event(kind, first, last, step.label, step.run, step.trial)
            done += step.kind == "cue"
        for run, first in run_first.items():
            self.writer.event(RUN, first, run_last[run], "", run)
        stats = self._assembler.stats
        channels = self.source.info.channels
        lost = (
            []
            if self._out_of_range is None
            else [name for name, n in zip(channels, self._out_of_range) if n > 0.5 * self._frames]
        )
        summary = RecordingSummary(
            self.writer.samples,
            self.writer.samples / self.source.info.fs,
            done,
            stats.delivered_hz,
            stats.loss,
            self._usable / self._frames if self._frames else 0.0,
            lost,
        )
        self.ui.say(
            f"done: {summary.trials} trials, {summary.seconds / 60:.1f} min, {summary.delivered_hz:.1f} samples/s, "
            f"{100 * summary.usable_share:.0f} % of frames usable"
            + (f"; out of range most of the time: {', '.join(lost)}" if lost else "")
        )
        return summary

"""From recorded sessions to labelled frames.

A session is run through the same front-end that runs live, and every frame is
then placed on the session's timeline:

* its *label* is the cued command if the whole analysis window lies inside the
  cue, after a reaction time. These frames train and test the classifier;
* its *run* tells evaluation which frames belong together. Splits are made by
  run, never by frame, because consecutive frames share most of their samples;
* its *cue* is what the wearer was being asked to do when the window ended,
  including pauses. It is the timeline against which simulated driving is scored.

The session's baseline provides the reference for all of its frames and the
resting muscle level for the stop reflex.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..calibration.protocol import CLENCH, REST
from ..dsp.frontend import Envelope, Frames, FrontendConfig, process
from ..dsp.quality import FrameQuality, QualityConfig, QualityMeter
from ..dsp.reflex import resting_level
from ..io.session import BASELINE, CUE, PAUSE, RUN, Event, Session
from ..io.stream import StreamStats
from .features import mean_reference


@dataclass
class Recording:
    """One session after the front-end, with every frame placed on its timeline."""

    key: str
    frames: Frames
    quality: FrameQuality
    envelope: Envelope
    events: list[Event]
    stats: StreamStats
    reference: np.ndarray  # [F, P]
    resting_db: np.ndarray  # [C]
    baseline_frames: int
    label: np.ndarray  # [n] object: cue label when the frame is usable for training, else ""
    cue: np.ndarray  # [n] object: what was asked for when the window ended ("" outside the protocol)
    trial: np.ndarray  # [n] int, -1 outside trials
    run: np.ndarray  # [n] int, 0 outside runs
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def labels(self) -> list[str]:
        return [str(v) for v in dict.fromkeys(self.label[self.label != ""])]


def _inside(config: FrontendConfig, index: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Frames whose whole window lies within samples ``[start, stop)``."""
    first = index * config.hop
    return (first >= start) & (first + config.window <= stop)


def load_recording(
    session: Session,
    frontend: FrontendConfig | None = None,
    quality: QualityConfig | None = None,
    *,
    reaction_s: float = 1.0,
) -> Recording:
    frontend = frontend or FrontendConfig()
    frontend.matches(session.info)
    block, stats = session.block()
    frames, envelope = process(block, frontend)
    q = QualityMeter(frontend, quality)(frames)
    events = session.grid_events(block)
    n = len(frames)
    label = np.full(n, "", dtype=object)
    cue = np.full(n, "", dtype=object)
    trial = np.full(n, -1, dtype=np.int64)
    run = np.zeros(n, dtype=np.int64)
    ends = frames.index * frontend.hop + frontend.window  # one past the last sample of each window
    react = round(reaction_s * frontend.fs)

    for event in events:
        if event.kind == RUN:
            run[_inside(frontend, frames.index, event.start, event.stop)] = event.run
        elif event.kind == CUE:
            usable = _inside(frontend, frames.index, event.start + react, event.stop)
            label[usable] = event.label
            trial[usable] = event.trial
            cue[(ends > event.start) & (ends <= event.stop)] = event.label
        elif event.kind == PAUSE:
            cue[(ends > event.start) & (ends <= event.stop)] = REST
        elif event.kind == CLENCH:
            cue[(ends > event.start) & (ends <= event.stop)] = CLENCH

    baseline = [e for e in events if e.kind == BASELINE]
    if baseline:
        in_base = np.zeros(n, dtype=bool)
        fast = np.zeros(len(envelope), dtype=bool)
        for event in baseline:
            in_base |= _inside(frontend, frames.index, event.start, event.stop)
            first = envelope.index * frontend.emg_hop
            fast |= (first >= event.start) & (first + frontend.emg_window <= event.stop)
    else:
        # A recording without a marked baseline (an import): fall back to its rest trials, then to everything.
        in_base = label == REST
        if not in_base.any():
            in_base = np.ones(n, dtype=bool)
        fast = np.ones(len(envelope), dtype=bool)
    chosen = in_base & q.usable
    if not chosen.any():
        chosen = in_base & frames.valid
    if not chosen.any():
        raise ValueError(f"{session.key}: no usable frames to build the session reference from")
    reference = mean_reference(frames.cosp[chosen])
    resting = resting_level(Envelope(envelope.index[fast], envelope.power[fast], envelope.valid[fast]))
    return Recording(
        session.key,
        frames,
        q,
        envelope,
        events,
        stats,
        reference,
        resting,
        int(chosen.sum()),
        label,
        cue,
        trial,
        run,
        dict(session.meta),
    )


@dataclass
class Dataset:
    """Labelled, usable frames of one or more recordings, ready for a classifier."""

    classes: tuple[str, ...]
    cosp: np.ndarray  # [n, F, P]
    y: np.ndarray  # [n] class index
    trial: np.ndarray  # [n] trial id, unique across recordings
    group: np.ndarray  # [n] run id, unique across recordings
    session: np.ndarray  # [n] index into ``keys``
    references: np.ndarray  # [S, F, P]
    keys: list[str]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_trials(self) -> int:
        return len(np.unique(self.trial))

    @property
    def n_groups(self) -> int:
        return len(np.unique(self.group))

    def reference_of(self, rows: np.ndarray | slice = slice(None)) -> np.ndarray:
        return self.references[self.session[rows]]

    def counts(self) -> dict[str, int]:
        """Trials per class."""
        out = {}
        for k, name in enumerate(self.classes):
            out[name] = len(np.unique(self.trial[self.y == k]))
        return out


def build_dataset(
    recordings: list[Recording], classes: tuple[str, ...] | None = None, *, clean_only: bool = True
) -> Dataset:
    """Collect training frames. ``clean_only`` leaves out frames flagged as artefacts."""
    if not recordings:
        raise ValueError("no recordings")
    if classes is None:
        seen: dict[str, None] = {}
        for rec in recordings:
            seen.update(dict.fromkeys(rec.labels))
        classes = tuple(sorted(seen, key=lambda name: (name != REST, list(seen).index(name))))
    cosp, y, trial, group, session, references = [], [], [], [], [], []
    for s, rec in enumerate(recordings):
        keep = np.isin(rec.label, classes) & (rec.quality.usable if clean_only else rec.frames.valid) & (rec.run > 0)
        if not keep.any():
            continue
        cosp.append(rec.frames.cosp[keep])
        y.append(np.array([classes.index(v) for v in rec.label[keep]], dtype=np.int64))
        trial.append(rec.trial[keep] + 100_000 * s)
        group.append(rec.run[keep] + 1_000 * s)
        session.append(np.full(int(keep.sum()), s, dtype=np.int64))
    if not cosp:
        raise ValueError("the recordings contain no usable labelled frames")
    found = np.unique(np.concatenate(y))
    missing = [name for k, name in enumerate(classes) if k not in found]
    if missing:
        raise ValueError(f"no usable frames for {', '.join(missing)} in {', '.join(rec.key for rec in recordings)}")
    references = np.stack([rec.reference for rec in recordings])
    return Dataset(
        tuple(classes),
        np.concatenate(cosp),
        np.concatenate(y),
        np.concatenate(trial),
        np.concatenate(group),
        np.concatenate(session),
        references,
        [rec.key for rec in recordings],
    )

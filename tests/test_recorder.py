from __future__ import annotations

import io
import itertools

import numpy as np
import pytest

from nwc.calibration.protocol import ProtocolConfig, build_schedule
from nwc.calibration.recorder import ConsoleUI, Recorder, RecordingError
from nwc.decode.dataset import load_recording
from nwc.io.session import Session, SessionWriter
from nwc.io.sources import ReplaySource
from nwc.sim.eeg import EEGSynth
from nwc.sim.source import SimulatedSource

PROTOCOL = ProtocolConfig(runs=2, trials_per_command=2, baseline_s=6.0, clench_trials=1, cue_s=4.0, pause_s=(1.0, 1.5))


def record(tmp_path, source, protocol=PROTOCOL):
    screen = io.StringIO()
    with SessionWriter(tmp_path / "rec" / "a", source.info, subject="rec", session="a") as writer:
        summary = Recorder(source, writer, protocol, ConsoleUI(screen, bell=False, interactive=False)).run()
    return Session(tmp_path / "rec" / "a"), summary, screen.getvalue()


def test_recording_follows_the_schedule_sample_for_sample(tmp_path):
    session, summary, screen = record(tmp_path, SimulatedSource(EEGSynth(seed=1), seed=1))
    steps = [s for s in build_schedule(PROTOCOL) if s.kind != "break"]
    events = [e for e in session.events() if e.kind != "run"]
    assert [(e.kind, e.label, e.run, e.trial) for e in events] == [(s.kind, s.label, s.run, s.trial) for s in steps]
    fs = session.info.fs
    assert all(e.start == p.stop for p, e in itertools.pairwise(events))  # steps follow each other without a gap
    for event, step in zip(events, steps):
        assert (event.stop - event.start) / fs == pytest.approx(
            step.duration, abs=12 / fs
        )  # timed on the stream, to the packet
    runs = session.events("run")
    assert [r.run for r in runs] == [1, 2] and runs[0].stop <= runs[1].start
    assert (
        summary.trials == 16
        and summary.samples == session.meta["samples"]
        and summary.delivered_hz == pytest.approx(256.0, rel=0.02)
    )
    assert 0.5 < summary.usable_share <= 1.0 and summary.lost_channels == []
    assert (
        ">>> FORWARD <<<" in screen
        and "baseline" in screen
        and ">>> CLENCH <<<" in screen
        and "done: 16 trials" in screen
    )
    assert "Rest, then press Enter" in screen  # between the runs
    rec = load_recording(session)  # and what was recorded can be used
    assert set(rec.labels) == set(PROTOCOL.labels) and rec.baseline_frames > 0


def test_recording_reports_a_stream_that_ends_or_stalls(tmp_path):
    short = ReplaySource(np.arange(2560) / 256.0, np.zeros((2560, 4)), SimulatedSource(EEGSynth()).info)
    screen = io.StringIO()
    writer = SessionWriter(tmp_path / "short", short.info, subject="x", session="short")
    with writer, pytest.raises(RecordingError, match="ended"):
        Recorder(short, writer, PROTOCOL, ConsoleUI(screen, interactive=False)).run()
    kept = Session(tmp_path / "short")
    assert (
        kept.meta["complete"] and kept.meta["samples"] == 2560 and len(kept.events("baseline")) == 1
    )  # what arrived is on disk

    class Silent(SimulatedSource):
        def read(self, timeout: float = 0.05):
            return None

    silent = Silent(EEGSynth())
    writer = SessionWriter(tmp_path / "silent", silent.info, subject="x", session="silent")
    with writer, pytest.raises(RecordingError, match="no samples"):
        Recorder(silent, writer, PROTOCOL, ConsoleUI(screen, interactive=False), stall_s=0.2).run()


def test_a_lifted_electrode_is_reported_at_the_end(tmp_path):
    source = SimulatedSource(EEGSynth(seed=2), seed=2).at(1.0, "contact", ("TP10", False))
    _, summary, screen = record(tmp_path, source)
    assert summary.lost_channels == ["TP10"] and "TP10" in screen and summary.usable_share < 0.2

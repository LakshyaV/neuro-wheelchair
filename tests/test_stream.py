from __future__ import annotations

import numpy as np
import pytest

from nwc.io.session import CUE, RUN, Session, SessionWriter, find_sessions, iter_chunks
from nwc.io.stream import Chunk, StreamAssembler, StreamInfo, assemble, grid_of_rows

INFO = StreamInfo(("TP9", "AF7", "AF8", "TP10"), 256.0, source="test")


def ramp(n: int, start: int = 0) -> np.ndarray:
    """Samples whose value is their own device index, so that gaps are visible in the data."""
    index = np.arange(start, start + n, dtype=np.float64)
    return np.column_stack([index, index + 0.25, index + 0.5, index + 0.75])


def jittered_arrival(n: int, rng: np.random.Generator, packet: int = 12) -> np.ndarray:
    """Timestamps the way a stream bridge stamps them: back from each packet's jittered arrival."""
    t = np.empty(n)
    arrival = 100.0
    for start in range(0, n, packet):
        arrival = max(arrival + packet / 256.0 + rng.normal(0, 0.006), arrival + 1e-4)
        size = min(packet, n - start)
        t[start : start + size] = arrival - (size - 1 - np.arange(size)) / 256.0
    return t


def test_arrival_jitter_is_not_mistaken_for_loss():
    rng = np.random.default_rng(0)
    n = 256 * 20
    t = jittered_arrival(n, rng)
    steps = np.rint(np.diff(t) * 256).astype(int)
    assert (steps != 1).sum() > 100 and (np.diff(t) < 0).any()  # timestamps jump at packet boundaries, both ways
    block, stats = assemble(t, ramp(n), INFO)
    assert len(block) == n and not block.filled.any() and len(block.breaks) == 0
    assert np.array_equal(block.x, ramp(n))
    assert np.allclose(np.diff(block.t), 1 / 256.0)  # grid time is regular whatever the stamps did
    assert stats.delivered_hz == pytest.approx(256.0, rel=0.01) and stats.loss < 0.01 and not stats.counted


def test_counted_stream_locates_lost_packets_exactly():
    n = 256 * 4
    keep = np.ones(n, dtype=bool)
    keep[300:312] = False  # one packet lost: short enough to fill
    keep[600:760] = False  # more than a quarter of a second: a break
    seq = np.arange(n)[keep]
    block, stats = assemble(seq / 256.0, ramp(n)[keep], INFO, seq)
    assert stats.counted and stats.filled == 12 and stats.breaks == 1 and stats.lost_in_breaks == 160
    assert len(block) == keep.sum() + 12
    assert block.filled[300:312].all() and not block.filled[:300].any()
    assert np.allclose(block.x[300:312, 0], np.arange(300, 312))  # interpolation across the hole is exact for a ramp
    assert block.breaks.tolist() == [600]  # the sample at this grid position is the first after the hole
    assert block.x[600, 0] == 760
    assert stats.loss == pytest.approx(172 / n)
    rows = grid_of_rows(block)
    assert len(rows) == keep.sum() and rows[300] == 312  # the 301st sample to arrive sits after the filled packet


def test_assembly_does_not_depend_on_chunking():
    rng = np.random.default_rng(1)
    n = 3000
    keep = rng.random(n // 12).repeat(12) > 0.05
    keep[:12] = True
    seq, x = np.arange(n)[keep], ramp(n)[keep]
    whole, _ = assemble(seq / 256.0, x, INFO, seq)
    assembler = StreamAssembler(INFO)
    parts, cursor = [], 0
    for size in rng.integers(1, 90, 400):
        if cursor >= len(seq):
            break
        parts.append(
            assembler.push(
                Chunk(seq[cursor : cursor + size] / 256.0, x[cursor : cursor + size], seq[cursor : cursor + size])
            )
        )
        cursor += size
    assert np.array_equal(np.concatenate([p.x for p in parts]), whole.x)
    assert np.array_equal(np.concatenate([p.row for p in parts]), whole.row)
    assert [int(p.start + b) for p in parts for b in p.breaks] == whole.breaks.tolist()
    assert (
        assembler.push(None).start == assembler.count and len(assembler.push(Chunk(np.zeros(0), np.zeros((0, 4))))) == 0
    )


def test_a_stall_is_a_break_and_does_not_count_as_rate():
    t = np.concatenate([np.arange(512) / 256.0, 10.0 + np.arange(512) / 256.0])
    block, stats = assemble(t, ramp(1024), INFO)
    assert block.breaks.tolist() == [512] and stats.breaks == 1
    assert stats.stalled_s == pytest.approx(8.0, abs=0.01)
    assert stats.delivered_hz == pytest.approx(256.0, rel=0.01)  # measured while the stream was flowing
    with pytest.raises(ValueError):
        StreamAssembler(INFO).push(Chunk(np.zeros(3), np.zeros((3, 2))))


def test_session_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    n = 256 * 3
    t, x = jittered_arrival(n, rng), np.round(rng.normal(0, 20, (n, 4)), 3)
    with SessionWriter(tmp_path / "s01" / "a", INFO, subject="s01", session="a", meta={"note": "x"}) as writer:
        for chunk in iter_chunks(t, x):
            writer.write(chunk)
        writer.event(RUN, 0, n, "", 1)
        writer.event(CUE, 100, 400, "left", 1, 0, strength=2)
    session = Session(tmp_path / "s01" / "a")
    assert session.key == "s01/a" and session.meta["note"] == "x" and session.meta["complete"]
    read_t, read_x, seq = session.raw()
    assert seq is None and np.allclose(read_t, t, atol=1e-4) and np.allclose(read_x, x, atol=1e-3)
    assert (tmp_path / "s01" / "a" / "eeg.csv").read_text().splitlines()[0] == "timestamps,TP9,AF7,AF8,TP10"
    cue = session.events(CUE)[0]
    assert (cue.start, cue.stop, cue.label, cue.extra) == (100, 400, "left", {"strength": 2})
    assert session.labels == ["left"] and session.duration == pytest.approx(3.0)
    assert [s.key for s in find_sessions(tmp_path)] == ["s01/a"] and find_sessions(tmp_path / "s01" / "a")[
        0
    ].key == "s01/a"
    with pytest.raises(FileExistsError):
        SessionWriter(tmp_path / "s01" / "a", INFO, subject="s01", session="a")
    with pytest.raises(FileNotFoundError):
        Session(tmp_path)


def test_events_follow_their_samples_onto_the_grid(tmp_path):
    n = 2560
    keep = np.ones(n, dtype=bool)
    keep[1000:1024] = False  # two packets lost in the middle of the recording
    seq = np.arange(n)[keep]
    with SessionWriter(tmp_path / "s", INFO, subject="s", session="s") as writer:
        writer.write(Chunk(seq / 256.0, ramp(n)[keep], seq))
        writer.event(CUE, 1200, 1500, "left")  # rows of arrival: device samples 1224 to 1524
    session = Session(tmp_path / "s")
    assert session.raw()[2] is not None  # the counter was stored
    block, stats = session.block()
    assert stats.filled == 24 and len(block) == n
    event = session.grid_events(block, CUE)[0]
    assert (event.start, event.stop) == (1224, 1524)
    assert block.x[event.start, 0] == 1224

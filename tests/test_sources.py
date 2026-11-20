from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from nwc.io.sources import MINDMONITOR_ADDRESS, PETAL_ADDRESS, OSCSource, ReplaySource, SourceError, open_source
from nwc.io.stream import StreamAssembler, StreamInfo
from nwc.sim.eeg import EEGSynth
from nwc.sim.source import SimulatedSource, Transport

INFO = StreamInfo(("TP9", "AF7", "AF8", "TP10"), 256.0)


def drain(source, limit: int = 100_000) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    t, x, seq = [], [], []
    for _ in range(limit):
        if source.exhausted:
            break
        chunk = source.read()
        if chunk is None:
            continue
        t.append(chunk.t)
        x.append(chunk.x)
        if chunk.seq is not None:
            seq.append(chunk.seq)
    return np.concatenate(t), np.concatenate(x), np.concatenate(seq) if seq else None


def test_replay_delivers_packets_in_order(sim_session):
    source = ReplaySource.from_session(sim_session)
    first = source.read()
    assert len(first) == 12 and source.position == 12 and not source.exhausted
    t, x, _ = drain(source)
    raw_t, raw_x, _ = sim_session.raw()
    assert np.array_equal(np.concatenate((first.x, x)), raw_x) and len(t) == len(raw_t) - 12
    assert source.exhausted and source.read() is None
    assert isinstance(open_source(f"replay:{sim_session.path}"), ReplaySource)


def test_replay_in_real_time_is_paced():
    n = 128
    source = ReplaySource(np.arange(n) / 256.0, np.zeros((n, 4)), INFO, realtime=True)
    started = time.monotonic()
    t, _, _ = drain(source)
    assert len(t) == n and 0.35 < time.monotonic() - started < 1.5  # half a second of samples takes about half a second


def test_transport_reproduces_packet_timing_and_loss():
    samples = np.arange(12 * 400, dtype=float)[:, None].repeat(4, axis=1)
    clean = Transport(256.0, jitter_ms=8.0, seed=1).push(samples)
    stamps = np.concatenate([chunk.t for _, _, chunk in clean])
    steps = np.diff(stamps)
    inside = np.delete(steps, np.arange(11, len(steps), 12))
    assert np.allclose(inside, 1 / 256.0)  # regular within a packet
    between = steps[11::12]
    assert (
        between.std() > 0.004 and (np.rint(between * 256) != 1).mean() > 0.3
    )  # jumps between packets, as on a real headset
    assert all(a <= b for (a, _, _), (b, _, _) in itertools.pairwise(clean))  # packets arrive in order

    lossy = Transport(256.0, loss=0.2, counted=True, seed=1).push(samples)
    assert 0.7 < len(lossy) / 400 < 0.9
    block = StreamAssembler(INFO).push(lossy[0][2])
    assert block.x[0, 0] == 0.0 and lossy[0][2].seq[0] == 0  # the first packet always arrives
    firsts = [first for _, first, _ in lossy]
    assert all(chunk.seq[0] == first for (_, first, chunk) in lossy) and sorted(firsts) == firsts


def test_simulated_source_follows_its_script():
    source = SimulatedSource(EEGSynth(seed=2), duration_s=6.0, counted=True, seed=2)
    source.at(2.0, "clench", True).at(3.0, "clench", False).at(4.0, "contact", ("AF7", False))
    _, x, seq = drain(source)
    assert source.exhausted and len(x) == 6 * 256 and np.array_equal(seq, np.arange(len(x)))
    emg = np.abs(np.diff(x[:, 0]))  # sample-to-sample change at TP9: large while the jaw is clenched
    assert emg[2 * 256 + 64 : 3 * 256].mean() > 5 * emg[: 2 * 256].mean()
    assert x[4 * 256 + 64 :, 1].std() > 5 * x[: 2 * 256, 1].std()  # AF7 is noise once it has lifted off
    with pytest.raises(ValueError):
        SimulatedSource(EEGSynth(seed=2)).at(0.0, "dance")._produce(12)
    assert isinstance(open_source("sim:3", duration_s=1.0), SimulatedSource)
    with pytest.raises(SourceError):
        open_source("telepathy")
    with pytest.raises(SourceError):
        open_source("replay")


@pytest.mark.network
def test_osc_petal_messages_with_a_usable_counter():
    pytest.importorskip("pythonosc")
    from pythonosc.udp_client import SimpleUDPClient

    source = OSCSource(port=0)  # any free port
    source.open()
    try:
        client = SimpleUDPClient("127.0.0.1", source.port)
        sent = 0
        for sample in list(range(200)) + list(range(212, 400)):  # one packet of twelve never arrives
            stamp = 1_700_000_000.0 + sample / 256.0
            client.send_message(
                PETAL_ADDRESS, [sample, int(stamp), stamp % 1.0, 1000, 0.5, 1.0 * sample, 2.0, 3.0, 4.0]
            )
            sent += 1
            if sample % 40 == 0:
                time.sleep(0.002)
        client.send_message(PETAL_ADDRESS, ["broken"])
        chunks, deadline = [], time.monotonic() + 3.0
        while sum(map(len, chunks)) < sent and time.monotonic() < deadline:
            chunk = source.read(timeout=0.02)
            if chunk is not None:
                chunks.append(chunk)
    finally:
        source.close()
    x = np.concatenate([c.x for c in chunks])
    seq = np.concatenate([c.seq for c in chunks])
    assert len(x) == sent and np.array_equal(x[:, 0], seq.astype(float)) and source.malformed == 1
    assert np.concatenate([c.t for c in chunks])[0] == pytest.approx(1_700_000_000.0)
    assembler = StreamAssembler(source.info)
    for chunk in chunks:
        assembler.push(chunk)
    assert assembler.stats.counted and assembler.stats.filled == 12  # the hole is found from the sample ids


@pytest.mark.network
def test_osc_mind_monitor_messages_and_an_unusable_counter():
    pytest.importorskip("pythonosc")
    from pythonosc.udp_client import SimpleUDPClient

    source = OSCSource(port=0, address=MINDMONITOR_ADDRESS)
    source.open()
    try:
        client = SimpleUDPClient("127.0.0.1", source.port)
        for sample in range(160):
            client.send_message(
                MINDMONITOR_ADDRESS, [1.0 * sample, 2.0, 3.0, 4.0, 5.0]
            )  # a fifth value (AUX) is ignored
        chunks, deadline = [], time.monotonic() + 3.0
        while sum(map(len, chunks)) < 160 and time.monotonic() < deadline:
            chunk = source.read(timeout=0.02)
            if chunk is not None:
                chunks.append(chunk)
    finally:
        source.close()
    assert sum(map(len, chunks)) == 160 and all(c.seq is None and c.x.shape[1] == 4 for c in chunks)

    # Where to listen comes from the specification: this computer only unless a host is given.
    assert (open_source("osc").host, open_source("osc").port) == ("127.0.0.1", 14739)
    assert (open_source("osc:5000").host, open_source("osc:5000").port) == ("127.0.0.1", 5000)
    assert (open_source("osc:0.0.0.0:5000").host, open_source("osc:0.0.0.0:5000").port) == ("0.0.0.0", 5000)

    # A source left at its default address understands Mind Monitor all the same.
    either = OSCSource(port=0)
    either.open()
    try:
        client = SimpleUDPClient("127.0.0.1", either.port)
        for sample in range(40):
            client.send_message(MINDMONITOR_ADDRESS, [1.0 * sample, 2.0, 3.0, 4.0])
        got, deadline = 0, time.monotonic() + 3.0
        while got < 40 and time.monotonic() < deadline:
            chunk = either.read(timeout=0.02)
            got += 0 if chunk is None else len(chunk)
    finally:
        either.close()
    assert got == 40

    # A Petal-style stream whose sample id does not count samples (here: it counts packets) is not trusted.
    probe = OSCSource(port=0)
    for sample in range(OSCSource.PROBE):
        probe._on_message(PETAL_ADDRESS, sample // 12, 1_700_000_000, 0.0, 0, 0.0, 1.0, 2.0, 3.0, 4.0)
    assert probe.read().seq is None


@pytest.mark.network
def test_lsl_stream_is_found_and_read_by_channel_label():
    try:
        import pylsl
    except Exception as error:  # not installed, or installed without its native library
        pytest.skip(f"pylsl cannot be used here: {error}")
    from nwc.io.sources import LSLSource

    name = f"nwc-test-{time.monotonic_ns()}"
    info = pylsl.StreamInfo(name, "EEG", 5, 256, "float32", name)
    channels = info.desc().append_child("channels")
    for label in ("AF7", "TP9", "Right AUX", "TP10", "AF8"):  # not in the order the front-end wants
        channels.append_child("channel").append_child_value("label", label)
    outlet = pylsl.StreamOutlet(info, chunk_size=12)
    source = LSLSource(name=name, timeout=10.0)
    source.open()
    try:
        assert source.info.channels == ("TP9", "AF7", "AF8", "TP10") and source.info.fs == 256.0
        sent = np.arange(5 * 240, dtype=np.float32).reshape(240, 5)
        for start in range(0, 240, 12):
            outlet.push_chunk(sent[start : start + 12].tolist())
        chunks, deadline = [], time.monotonic() + 5.0
        while sum(map(len, chunks)) < 240 and time.monotonic() < deadline:
            chunk = source.read(timeout=0.05)
            if chunk is not None:
                chunks.append(chunk)
    finally:
        source.close()
    x = np.concatenate([c.x for c in chunks])
    assert len(x) == 240 and np.array_equal(x, sent[:, [1, 0, 4, 3]])  # columns picked by their labels
    assert all(c.seq is None for c in chunks)
    # LSL stamps a chunk backwards from the moment it was pushed, so chunks sent in a burst overlap in time.
    # That is the jitter the assembler ignores: every sample is kept, in order.
    assembler = StreamAssembler(source.info)
    assert sum(len(assembler.push(c)) for c in chunks) == 240 and assembler.stats.breaks == 0
    with pytest.raises(SourceError):
        LSLSource(name="no-such-stream", timeout=0.2).open()

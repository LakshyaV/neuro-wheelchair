from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from nwc.calibration.protocol import CLENCH, REST, ProtocolConfig, build_schedule, describe, duration
from nwc.dsp.frontend import FrontendConfig, process
from nwc.dsp.quality import QualityMeter
from nwc.io.stream import StreamInfo, assemble
from nwc.sim.eeg import SIGNATURES, EEGSynth, SimConfig
from nwc.sim.session import simulate_session

CFG = FrontendConfig()
INFO = StreamInfo(CFG.channels, CFG.fs)


def test_schedule_interleaves_commands_in_every_run():
    config = ProtocolConfig(commands=("forward", "left", "right"), runs=3, trials_per_command=8)
    steps = build_schedule(config)
    assert steps[0].kind == "baseline" and sum(s.kind == "clench" for s in steps) == config.clench_trials
    cues = [s for s in steps if s.kind == "cue"]
    assert len(cues) == 3 * 8 * 4 and [s.trial for s in cues] == list(range(len(cues)))
    for run in (1, 2, 3):
        labels = [s.label for s in cues if s.run == run]
        assert {label: labels.count(label) for label in config.labels} == dict.fromkeys(config.labels, 8)
        assert all(not (a == b == c) for a, b, c in zip(labels, labels[1:], labels[2:]))
    assert all(a.kind == "pause" for a, b in itertools.pairwise(steps) if b.kind == "cue")  # every cue follows a pause
    assert build_schedule(config) == steps and build_schedule(replace(config, seed=1)) != steps
    assert (
        sum(s.kind == "break" for s in steps) == 2
        and 13 < duration(steps) / 60 < 15
        and "96 trials" in describe(config)
    )
    assert steps[0].hint and cues[0].hint
    for bad in (("rest", "left"), ("left", "left"), (), (CLENCH,)):
        with pytest.raises(ValueError):
            ProtocolConfig(commands=bad)
    assert config.labels[0] == REST


def test_synthesiser_is_deterministic_and_chunk_invariant():
    whole = EEGSynth(seed=4).generate(6000)
    parts = EEGSynth(seed=4)
    pieces = np.concatenate([parts.generate(n) for n in (1, 11, 500, 7, 2300, 3181)])
    assert np.array_equal(whole, pieces) and parts.position == 6000
    assert not np.array_equal(whole, EEGSynth(seed=5).generate(6000))
    assert np.allclose(whole / 0.48828125, np.rint(whole / 0.48828125))  # quantised like the headset
    with pytest.raises(ValueError):
        EEGSynth().set_intent("levitate")


def test_levels_are_those_of_a_real_headset():
    """Median amplitude, low-frequency swing and muscle-band floor within the range measured on real Muse recordings."""
    quiet = SimConfig(gain_spread=0.0, session_spread_db=0.0)
    x = EEGSynth(quiet, seed=1).generate(256 * 120)
    block, _ = assemble(np.arange(len(x)) / 256.0, x, INFO)
    frames, fast = process(block, CFG)
    quality = QualityMeter(CFG)(frames)
    rms = np.median(quality.rms[frames.valid], axis=0)
    assert np.all((rms > 3.0) & (rms < 15.0)), rms  # real: 4 to 8 uV per channel, 6 to 19 across recordings
    swing = np.median(frames.swing[frames.valid], axis=0)
    assert np.all((swing > 12.0) & (swing < 60.0)), swing  # real: 20 to 37 uV
    floor = np.median(fast.level_db[fast.valid], axis=0)
    assert np.all((floor > 4.0) & (floor < 16.0)), floor  # real: 6 to 12 dB re 1 uV^2
    assert (
        0.02 < quality.artefact[frames.valid].mean() < 0.3
    )  # blinks and electrode shifts happen, but not all the time


def test_a_command_changes_the_rhythms_it_is_meant_to():
    config = SimConfig(effect=3.0, blink_rate_hz=0, twitch_rate_hz=0, shift_rate_hz=0, trial_spread=0.0)
    alpha = {}
    for label in ("rest", "forward", "backward"):
        synth = EEGSynth(config, seed=2)
        synth.set_intent(label)
        x = synth.generate(256 * 40)[256 * 4 :]
        spectrum = np.abs(np.fft.rfft(x[:, 0] - x[:, 0].mean())) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / 256.0)
        alpha[label] = spectrum[(freqs >= 8) & (freqs <= 12)].sum()
    assert (
        alpha["forward"] < 0.6 * alpha["rest"] < 0.6 * alpha["backward"]
    )  # the signatures say: down for forward, up for backward
    assert set(SIGNATURES) >= {"rest", "forward", "backward", "left", "right"}


def test_simulated_session_has_the_structure_of_a_recorded_one(sim_session):
    from conftest import PROTOCOL

    assert sim_session.meta["simulated"] and sim_session.meta["complete"] and sim_session.labels[0] in PROTOCOL.labels
    cues, runs = sim_session.events("cue"), sim_session.events("run")
    assert len(cues) == PROTOCOL.runs * PROTOCOL.trials_per_command * len(PROTOCOL.labels)
    assert (
        len(runs) == PROTOCOL.runs
        and len(sim_session.events("baseline")) == 1
        and len(sim_session.events("clench")) == PROTOCOL.clench_trials
    )
    fs = sim_session.info.fs
    assert all(abs((c.stop - c.start) / fs - PROTOCOL.cue_s) < 0.1 for c in cues)
    assert all(any(r.start <= c.start and c.stop <= r.stop and r.run == c.run for r in runs) for c in cues)
    block, stats = sim_session.block()
    assert stats.delivered_hz == pytest.approx(fs, rel=0.01) and stats.breaks == 0 and not stats.counted
    assert (
        np.rint(np.diff(sim_session.raw()[0]) * fs).astype(int).tolist().count(1) < len(block) - 1000
    )  # jittered stamps


def test_lost_packets_keep_events_on_their_samples(tmp_path):
    protocol = ProtocolConfig(runs=1, trials_per_command=2, baseline_s=8.0, clench_trials=1)
    clean = simulate_session(tmp_path / "a", protocol, seed=7)
    lossy = simulate_session(tmp_path / "b", protocol, seed=7, loss=0.05, counted=True)
    block, stats = lossy.block()
    assert stats.counted and 0.02 < stats.loss < 0.09 and stats.filled > 0
    assert abs(len(block) - len(clean.block()[0])) <= 12  # filled back to (almost) the full length
    for a, b in zip(clean.grid_events(clean.block()[0], "cue"), lossy.grid_events(block, "cue")):
        assert a.label == b.label and abs(a.start - b.start) <= 12 and abs(a.stop - b.stop) <= 12

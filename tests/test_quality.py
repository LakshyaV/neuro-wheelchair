from __future__ import annotations

import numpy as np
import pytest

from nwc.dsp.frontend import Envelope, FrontendConfig, process
from nwc.dsp.quality import ContactMonitor, QualityConfig, QualityMeter
from nwc.dsp.reflex import ClenchConfig, ClenchDetector, fit_threshold, resting_level
from nwc.io.stream import StreamInfo, assemble
from nwc.sim.eeg import EEGSynth, SimConfig

CFG = FrontendConfig()
INFO = StreamInfo(CFG.channels, CFG.fs)
CALM = SimConfig(blink_rate_hz=0, twitch_rate_hz=0, shift_rate_hz=0, gain_spread=0.0, session_spread_db=0.0)


def analyse(x: np.ndarray):
    block, _ = assemble(np.arange(len(x)) / CFG.fs, x, INFO)
    frames, fast = process(block, CFG)
    return frames, fast, QualityMeter(CFG)(frames)


def test_clean_signal_is_usable():
    frames, _, quality = analyse(EEGSynth(CALM, seed=1).generate(256 * 30))
    assert quality.usable[frames.valid].all() and not quality.out_of_range[frames.valid].any()
    assert np.array_equal(quality.usable, frames.valid & ~quality.artefact)


def test_a_blink_sized_swing_is_an_artefact_but_not_lost_contact():
    x = EEGSynth(CALM, seed=1).generate(256 * 40)
    t = np.arange(256) / 256.0
    x[256 * 20 : 256 * 21, 1] += 260.0 * np.exp(-0.5 * ((t - 0.5) / 0.08) ** 2)  # a large deflection on AF7
    frames, _, quality = analyse(x)
    first, last = frames.index * CFG.hop, frames.index * CFG.hop + CFG.window
    hit = (first < 256 * 21) & (last > 256 * 20) & frames.valid
    assert quality.artefact[hit].mean() > 0.7 and not quality.artefact[frames.valid & ~hit].any()
    monitor = ContactMonitor(CFG)
    monitor.push(quality, frames.valid)
    assert monitor.lost_channels == []  # two seconds of trouble are not a lost electrode


def test_a_lifted_electrode_is_lost_contact_until_it_is_back():
    synth = EEGSynth(CALM, seed=2)
    before = synth.generate(256 * 15)
    synth.set_contact("AF8", False)
    off = synth.generate(256 * 15)
    synth.set_contact("AF8", True)
    after = synth.generate(256 * 15)
    frames, _, quality = analyse(np.concatenate((before, off, after)))
    monitor = ContactMonitor(CFG)
    lost_at = []
    for i in range(len(frames)):
        lost = monitor.push(
            type(quality)(
                quality.rms[i : i + 1],
                quality.out_of_range[i : i + 1],
                quality.artefact[i : i + 1],
                quality.usable[i : i + 1],
            ),
            frames.valid[i : i + 1],
        )
        lost_at.append(bool(lost[2]))
    seconds = CFG.frame_time(frames.index)
    lost_at = np.array(lost_at)
    assert (
        not lost_at[seconds < 15].any()
        and lost_at[(seconds > 20) & (seconds < 30)].all()
        and not lost_at[seconds > 36].any()
    )
    first = seconds[np.argmax(lost_at)]
    assert 15.0 < first < 15.0 + QualityConfig().lost_after_s + CFG.window_s + 0.5  # noticed within a few seconds
    assert monitor.channels[2] == "AF8" and not lost_at[:, None].repeat(1, 1).all()
    monitor.reset()
    assert monitor.lost_channels == []


def test_a_flat_channel_is_out_of_range():
    x = EEGSynth(CALM, seed=3).generate(256 * 12)
    x[:, 0] = 123.0
    frames, _, quality = analyse(x)
    assert quality.out_of_range[frames.valid][:, 0].all() and not quality.out_of_range[frames.valid][:, 1:].any()


def envelope_of(levels_db: np.ndarray, valid: np.ndarray | None = None) -> Envelope:
    power = 10.0 ** (np.asarray(levels_db, dtype=float) / 10.0)
    return Envelope(
        np.arange(len(power)), power.astype(np.float32), np.ones(len(power), dtype=bool) if valid is None else valid
    )


def test_clench_needs_both_sides_and_some_persistence():
    rest = np.array([12.0, 6.0, 10.0, 10.0])
    quiet = np.tile(rest, (20, 1))
    both = quiet.copy()
    both[5:15, [0, 3]] += 25.0
    detector = ClenchDetector(CFG.channels, rest)
    events = detector.push(envelope_of(both))
    assert [(e.frame, e.pressed) for e in events] == [(7, True), (18, False)]  # on after three frames, off after four
    one_side = quiet.copy()
    one_side[5:15, 0] += 30.0  # chewing on one side, an electrode being touched
    assert ClenchDetector(CFG.channels, rest).push(envelope_of(one_side)) == []
    blip = quiet.copy()
    blip[5:7, [0, 3]] += 30.0  # too short
    assert ClenchDetector(CFG.channels, rest).push(envelope_of(blip)) == []
    # frames without data neither start nor end a clench
    valid = np.ones(20, dtype=bool)
    valid[6:9] = False
    held = ClenchDetector(CFG.channels, rest)
    assert [(e.frame, e.pressed) for e in held.push(envelope_of(both, valid))] == [(10, True), (18, False)]
    with pytest.raises(ValueError):
        ClenchDetector(("A", "B"), rest[:2])


def test_clench_detector_on_simulated_clenches_and_twitches():
    config = SimConfig(
        blink_rate_hz=0.25, twitch_rate_hz=0.1, shift_rate_hz=0.02
    )  # plenty of incidental muscle activity
    synth = EEGSynth(config, seed=5)
    parts, clenches = [synth.generate(256 * 20)], []
    for _ in range(8):
        clenches.append(synth.position)
        synth.set_clench(True)
        parts.append(synth.generate(256))
        synth.set_clench(False)
        parts.append(synth.generate(256 * 6))
    _, fast, _ = analyse(np.concatenate(parts))
    baseline = fast.index * CFG.emg_hop + CFG.emg_window <= 256 * 20
    detector = ClenchDetector(
        CFG.channels, resting_level(Envelope(fast.index[baseline], fast.power[baseline], fast.valid[baseline]))
    )
    presses = [e.frame * CFG.emg_hop for e in detector.push(fast) if e.pressed]
    delays = [min((p - c) / 256.0 for p in presses if p >= c) for c in clenches]
    assert len(presses) == len(clenches) and max(delays) < 0.4  # every clench, nothing else, within 0.4 s


def test_threshold_is_fitted_between_rest_and_calibration_clenches():
    rng = np.random.default_rng(0)
    rest, clench = rng.normal(0.0, 1.5, 5000), rng.normal(26.0, 3.0, 200)
    fitted, report = fit_threshold(rest, clench)
    assert report["rest_p99_9_db"] < fitted.on_db < report["clench_p20_db"] and fitted.off_db < fitted.on_db
    assert report["margin_db"] > 10 and fitted.channels == ClenchConfig().channels
    weak, weak_report = fit_threshold(rest, rng.normal(1.0, 1.5, 200))  # clenches no stronger than rest
    assert weak.on_db == ClenchConfig().on_db and weak_report["margin_db"] < 0
    with pytest.raises(ValueError):
        resting_level(Envelope.empty(CFG))


@pytest.mark.parametrize("mains_hz", [50.0, 60.0])
def test_the_stop_reflex_does_not_care_which_power_line_is_in_the_room(mains_hz):
    """Nothing is set for the country: a strong line at 50 or at 60 Hz leaves rest and clench where they were."""
    quiet = SimConfig(**{**CALM.__dict__, "mains_hz": 0.0})
    noisy = SimConfig(**{**CALM.__dict__, "mains_hz": mains_hz, "mains_uv": 40.0})  # far more than the EEG itself
    levels = {}
    for name, sim in (("quiet", quiet), ("noisy", noisy)):
        synth = EEGSynth(sim, seed=4)
        rest = synth.generate(256 * 20)
        synth.set_clench(1.0)
        clench = synth.generate(256 * 2)
        synth.set_clench(0.0)
        after = synth.generate(256 * 4)
        _, fast, _ = analyse(np.concatenate((rest, clench, after)))
        resting = fast.index * CFG.emg_hop + CFG.emg_window <= 256 * 20
        base = resting_level(Envelope(fast.index[resting], fast.power[resting], fast.valid[resting]))
        events = ClenchDetector(CFG.channels, base).push(fast)
        during = (fast.index * CFG.emg_hop >= 256 * 20.5) & (fast.index * CFG.emg_hop + CFG.emg_window <= 256 * 22)
        excess = 10 * np.log10(np.median(fast.power[during], axis=0)) - base
        levels[name] = (base[[0, 3]], excess[[0, 3]], [e.pressed for e in events])
    assert levels["noisy"][2] == [True, False] == levels["quiet"][2]  # one press, one release, either way
    assert np.abs(levels["noisy"][0] - levels["quiet"][0]).max() < 1.0  # the resting level moved by under 1 dB
    assert np.abs(levels["noisy"][1] - levels["quiet"][1]).max() < 1.0  # and so did the clench's margin
    assert levels["noisy"][1].min() > 15.0

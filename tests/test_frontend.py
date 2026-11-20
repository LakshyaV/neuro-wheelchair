from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import logm as scipy_logm

from nwc.dsp import spectral
from nwc.dsp.frontend import Envelope, Frames, Frontend, FrontendConfig, process
from nwc.io.stream import Chunk, StreamAssembler, StreamInfo, assemble

CFG = FrontendConfig()
INFO = StreamInfo(CFG.channels, CFG.fs)


def noise(seconds: float, seed: int = 0, scale: float = 10.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, scale, (round(seconds * CFG.fs), 4))


def frames_of(x: np.ndarray, config: FrontendConfig = CFG) -> tuple[Frames, Envelope]:
    block, _ = assemble(np.arange(len(x)) / config.fs, x, StreamInfo(config.channels, config.fs))
    return process(block, config)


def test_frames_do_not_depend_on_chunking():
    x = noise(12)
    whole, fast = frames_of(x)
    for sizes in (np.full(400, 12), np.random.default_rng(1).integers(1, 700, 400)):
        assembler, frontend = StreamAssembler(INFO), Frontend(CFG)
        parts, envelopes, cursor = [], [], 0
        for size in sizes:
            if cursor >= len(x):
                break
            f, e = frontend.push(
                assembler.push(Chunk(np.arange(cursor, min(cursor + size, len(x))) / CFG.fs, x[cursor : cursor + size]))
            )
            parts.append(f)
            envelopes.append(e)
            cursor += size
        chunked, chunked_fast = Frames.concatenate(parts, CFG), Envelope.concatenate(envelopes, CFG)
        assert np.array_equal(chunked.index, whole.index) and np.array_equal(chunked.valid, whole.valid)
        assert np.array_equal(chunked.cosp, whole.cosp) and np.array_equal(chunked.swing, whole.swing)  # bit for bit
        assert np.array_equal(chunked_fast.power, fast.power) and np.array_equal(chunked_fast.valid, fast.valid)


def test_frame_grid_and_settling():
    frames, fast = frames_of(noise(10))
    assert frames.cosp.shape == (len(frames), CFG.n_bins, 10) and frames.cosp.dtype == np.float32
    assert np.array_equal(frames.index, np.arange(len(frames))) and len(frames) == (2560 - CFG.window) // CFG.hop + 1
    assert CFG.frame_span(3) == (192, 704) and CFG.frame_time(3) == pytest.approx(704 / 256)
    # nothing is trusted until the filters have settled after the start of the stream
    first_valid = int(np.argmax(frames.valid))
    assert first_valid == int(np.ceil(CFG.settle_s * CFG.fs / CFG.hop)) and frames.valid[first_valid:].all()
    assert len(fast) == (2560 - CFG.emg_window) // CFG.emg_hop + 1 and not fast.valid[0] and fast.valid[-1]


def test_spectrum_is_calibrated_and_resolves_frequency():
    t = np.arange(round(8 * CFG.fs)) / CFG.fs
    x = np.zeros((len(t), 4))
    x[:, 0] = 20 * np.sqrt(2) * np.sin(2 * np.pi * 10.0 * t)  # 20 uV RMS at 10 Hz
    x[:, 1] = np.random.default_rng(0).normal(0, 7.0, len(t))  # 7 uV RMS, white
    x[:, 2] = x[:, 0]  # in phase with channel 0
    x[:, 3] = -x[:, 0]  # in antiphase
    frames, _ = frames_of(x)
    cosp = spectral.unpack(frames.cosp[frames.valid].astype(np.float64)).mean(axis=0)  # [F, C, C]
    power = np.einsum("fcc->fc", cosp)
    assert power[:, 0].sum() == pytest.approx(400.0, rel=0.03)  # summing bins gives the variance
    assert power[8:12, 0].sum() > 0.97 * power[:, 0].sum()  # ...concentrated at the tone
    assert power[1:, 1].sum() == pytest.approx(
        49.0, rel=0.1
    )  # white noise keeps its variance (minus the high-passed DC bin)
    assert cosp[8:12, 0, 2].sum() == pytest.approx(400.0, rel=0.03) and cosp[8:12, 0, 3].sum() == pytest.approx(
        -400.0, rel=0.03
    )
    weights = spectral.band_weights(((8.0, 12.0), (20.0, 30.0)), CFG.n_bins)
    covs = spectral.band_covariances(frames.cosp[frames.valid].astype(np.float64), weights)
    assert covs.shape[1:] == (2, 4, 4) and np.all(np.linalg.eigvalsh(covs) > 0)


def test_band_weights_and_mains_exclusion():
    weights = spectral.band_weights(((1.0, 4.0), (7.5, 9.0)), 128)
    assert weights[0, :6].tolist() == [0, 1, 1, 1, 0, 0] and weights[1, 7:10].tolist() == [0.5, 1.0, 0.0]
    # Both power-line families are left out everywhere: there is no setting to get wrong.
    assert spectral.mains_harmonics(25.0, 125.0) == (50.0, 60.0, 100.0, 120.0)
    exclude = spectral.mains_exclusion(128)
    assert exclude == ((48.0, 52.0), (58.0, 62.0), (98.0, 102.0), (118.0, 122.0))
    blocked = spectral.band_weights(((52.0, 70.0),), 128, exclude)[0]
    assert blocked[58:62].sum() == 0 and blocked[52:58].sum() == 6 and blocked[62:70].sum() == 8
    with pytest.raises(ValueError):
        spectral.band_weights(((59.0, 61.0),), 128, exclude)  # nothing left of this band
    assert (
        spectral.uniform_bands(1, 45, 2)[:2] == ((1.0, 3.0), (3.0, 5.0)) and len(spectral.uniform_bands(1, 45, 2)) == 22
    )
    with pytest.raises(ValueError):
        spectral.Multitaper(100, 256.0)  # shorter than a second: 1 Hz bins cannot be resolved


def test_matrix_functions_and_tangent_space():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 4, 4))
    spd = a @ a.transpose(0, 2, 1) + 0.5 * np.eye(4)
    assert np.allclose(spectral.logm(spd)[0], scipy_logm(spd[0]).real, atol=1e-9)
    assert np.allclose(spectral.expm(spectral.logm(spd)), spd)
    root = spectral.invsqrtm(spd)
    assert np.allclose(root @ spd @ root, np.eye(4), atol=1e-9)
    assert np.allclose(spectral.unpack(spectral.pack(spd)), spd) and spectral.n_pairs(4) == 10
    covs, reference = spd[:, None], spd[0][None]
    vectors = spectral.tangent_features(covs, reference)
    assert vectors.shape == (5, 10) and np.allclose(vectors[0], 0.0, atol=1e-9)  # the reference itself is the origin
    # the length of a tangent vector is the Riemannian distance to the reference
    assert np.allclose(np.linalg.norm(vectors, axis=1), spectral.riemann_distance(spd, spd[0]), atol=1e-9)
    assert spectral.power_features(covs, reference).shape == (5, 4)


def test_features_relative_to_the_reference_ignore_electrode_gain():
    """Re-seating the headband changes each channel's gain. Expressed against a baseline with the same gains, nothing changes."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 3, 4, 4))
    covs = a @ a.transpose(0, 1, 3, 2) + np.eye(4)
    reference = covs.mean(axis=0)
    gain = np.diag([0.6, 1.4, 1.0, 2.5])
    for features in (spectral.tangent_features, spectral.power_features):
        assert np.allclose(features(covs, reference), features(gain @ covs @ gain, gain @ reference @ gain), atol=1e-9)


def test_breaks_and_heavy_loss_invalidate_exactly_their_frames():
    n = round(24 * CFG.fs)
    keep = np.ones(n, dtype=bool)
    keep[2000:2012] = False  # one lost packet, filled: 2 % of a window, tolerated
    keep[3500:3800] = False  # a long hole: a break
    keep[5000:5060] = False  # five packets in a row, filled: 12 % of a window, too much
    seq = np.arange(n)[keep]
    block, _ = assemble(seq / CFG.fs, noise(24)[keep], INFO, seq)
    frames, fast = process(block, CFG)
    first, last = frames.index * CFG.hop, frames.index * CFG.hop + CFG.window
    settle = round(CFG.settle_s * CFG.fs)
    after_break = (first < 3500 + settle) & (last > 3500)  # grid position 3500 is the first sample after the hole
    lost = n - 300 - len(block)
    assert lost == 0  # the two short holes were filled back in, only the long one is missing
    # A window is dropped once more than a tenth of it is interpolated. Grid positions after the
    # break are shifted by its 300 samples, so the five filled packets sit at 4700 to 4760.
    interpolated = np.clip(np.minimum(last, 4760) - np.maximum(first, 4700), 0, None)
    heavy = interpolated > CFG.max_filled * CFG.window
    assert heavy.sum() >= 5 and (interpolated > 0).sum() > heavy.sum()  # windows that only graze the stretch survive
    start_up = first < settle
    assert np.array_equal(~frames.valid, after_break | heavy | start_up)
    single = (first < 2012) & (last > 2000)
    assert frames.valid[single & ~start_up].all()  # the frames over the single lost packet are kept
    # fast frames are three packets long: one with any interpolated sample is not used
    fast_first, fast_last = fast.index * CFG.emg_hop, fast.index * CFG.emg_hop + CFG.emg_window
    assert not fast.valid[(fast_first < 2012) & (fast_last > 2000)].any()
    assert (~fast.valid).sum() < 0.2 * len(fast)


def test_frames_survive_realistic_packet_loss():
    """At 5 % packet loss most windows hold two or three filled packets. They must stay usable."""
    rng = np.random.default_rng(0)
    n = round(60 * CFG.fs)
    keep = (rng.random(n // 12) > 0.05).repeat(12)
    keep[:12] = True
    seq = np.arange(n)[keep]
    frames, _ = process(assemble(seq / CFG.fs, noise(60)[keep], INFO, seq)[0], CFG)
    assert frames.valid[8:].mean() > 0.9
    clean, _ = frames_of(noise(60))
    both = frames.valid & clean.valid
    ratio = np.log(frames.cosp[both][:, 1:45, 0] + 1e-12) - np.log(clean.cosp[both][:, 1:45, 0] + 1e-12)
    assert abs(np.median(ratio)) < 0.1  # and filling the holes barely moves the spectrum


def test_configuration_fingerprint_and_checks():
    assert CFG.fingerprint == FrontendConfig().fingerprint and len(CFG.fingerprint) == 16
    assert (
        replace(CFG, hop_s=0.5).fingerprint != CFG.fingerprint
        and replace(CFG, highpass_hz=0.5).fingerprint != CFG.fingerprint
    )
    CFG.matches(INFO)
    with pytest.raises(ValueError):
        CFG.matches(StreamInfo(("A", "B", "C", "D"), 256.0))
    with pytest.raises(ValueError):
        CFG.matches(StreamInfo(CFG.channels, 220.0))
    with pytest.raises(ValueError):
        FrontendConfig(fs=160.0)  # the default muscle band does not fit below 80 Hz
    slow = FrontendConfig(fs=160.0, emg_band_hz=(30.0, 70.0))
    frames, _ = frames_of(np.random.default_rng(0).normal(0, 5, (1600, 4)), slow)
    assert frames.cosp.shape[1] == 80
    with pytest.raises(ValueError):
        Frontend(CFG).push(
            assemble(np.arange(10) / 256.0 + 5, np.zeros((10, 4)), INFO)[0].__class__(
                7, np.zeros(1), np.zeros((1, 4)), np.zeros(1, dtype=int), np.zeros(0, dtype=int)
            )
        )

from __future__ import annotations

import json

import numpy as np
import pytest

from nwc.calibration.protocol import ProtocolConfig
from nwc.decode.bundle import BundleError, load_bundle
from nwc.decode.dataset import build_dataset, load_recording
from nwc.decode.features import BAND_LAYOUTS, FeatureSpec, Featurizer, mean_reference
from nwc.decode.lda import fit_lda, ledoit_wolf_shrinkage
from nwc.decode.model import IntentModel, log_softmax
from nwc.decode.pipeline import train_bundle
from nwc.decode.train import (
    Candidate,
    Corpus,
    default_candidates,
    evaluate,
    fit_final,
    fit_temperature,
    trial_decisions,
)
from nwc.dsp.frontend import FrontendConfig
from nwc.io.importers import import_labelled_recordings, import_marker_recordings
from nwc.sim.eeg import EEGSynth, SimConfig
from nwc.sim.session import simulate_session


# ---------------------------------------------------------------------- classifier
def test_shrinkage_matches_scikit_learn():
    covariance = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(0)
    for n, p in ((600, 220), (150, 220), (40, 88), (900, 24)):
        x = rng.standard_normal((n, p)) @ rng.standard_normal((p, p)) * 0.3
        x -= x.mean(axis=0)
        assert ledoit_wolf_shrinkage(x) == pytest.approx(covariance.ledoit_wolf(x, assume_centered=True)[1], abs=1e-9)
    assert ledoit_wolf_shrinkage(np.zeros((1, 5))) == 1.0


def test_lda_matches_scikit_learn_without_shrinkage_and_separates_classes():
    discriminant = pytest.importorskip("sklearn.discriminant_analysis")
    rng = np.random.default_rng(1)
    k, n, p = 4, 800, 30
    y = np.repeat(np.arange(k), n // k)
    x = rng.standard_normal((n, p)) @ rng.standard_normal((p, p)) + rng.standard_normal((k, p))[y] * 1.5
    mean, scale, weights, bias = fit_lda(x, y, k, shrinkage=0.0)
    z = (x - mean) / scale
    ours = log_softmax(z @ weights.T + bias)
    reference = discriminant.LinearDiscriminantAnalysis(solver="lsqr", priors=np.full(k, 1 / k)).fit(z, y)
    theirs = reference.predict_log_proba(z)
    comparable = theirs > -500  # below that scikit-learn's probabilities have underflowed
    assert np.allclose(ours[comparable], theirs[comparable], rtol=1e-4, atol=1e-3) and comparable.mean() > 0.4
    mean, scale, weights, bias = fit_lda(x, y, k)
    assert (((x - mean) / scale @ weights.T + bias).argmax(axis=1) == y).mean() > 0.9

    # a class that never occurred in training can never be predicted
    seen = y < 3
    _, _, weights, bias = fit_lda(x[seen], y[seen], k)
    assert not (((x - mean) / scale @ weights.T + bias).argmax(axis=1) == 3).any()
    with pytest.raises(ValueError):
        fit_lda(x[y == 0], y[y == 0], k)
    assert fit_lda(x[:20], y[:20].clip(0, 0) + (np.arange(20) % 2), 2)[2].shape == (2, p)  # fewer samples than features


def test_temperature_calibrates_overconfident_scores():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 3000)
    honest = rng.normal(0, 1, (3000, 3))
    honest[np.arange(3000), y] += 1.5
    assert fit_temperature(honest * 8.0, y) == pytest.approx(8.0 * fit_temperature(honest, y), rel=0.05)
    assert fit_temperature(rng.normal(0, 5, (3000, 3)), y) > 20  # scores that say nothing are flattened


# ---------------------------------------------------------------------- features and model
def test_feature_specs_and_sizes():
    for layout, bands in BAND_LAYOUTS.items():
        for kind, per_band in (("power", 4), ("tangent", 10)):
            featurizer = Featurizer(FeatureSpec(kind, layout), 128, 4)
            assert featurizer.size == len(bands) * per_band == len(featurizer.names)
    custom = FeatureSpec("power", bands=((8.0, 13.0),))
    assert custom.resolved_bands == ((8.0, 13.0),) and "1 bands" in custom.name
    for bad in ({"kind": "wavelets"}, {"layout": "exotic"}):
        with pytest.raises(ValueError):
            FeatureSpec(**bad)
    with pytest.raises(ValueError):
        mean_reference(np.zeros((0, 128, 10)))
    with pytest.raises(ValueError):
        IntentModel(("rest", "go"), FeatureSpec(), np.zeros(3), np.ones(3), np.zeros((2, 3)), np.zeros(2))


def test_frames_are_placed_on_the_session_timeline(recording, sim_session):
    from conftest import PROTOCOL

    cfg, rec = FrontendConfig(), recording
    assert set(rec.labels) == set(PROTOCOL.labels)
    events = [e for e in rec.events if e.kind == "cue"]
    first, last = rec.frames.index * cfg.hop, rec.frames.index * cfg.hop + cfg.window
    for event in events[:20]:
        mine = rec.trial == event.trial
        assert mine.any() and (rec.label[mine] == event.label).all() and (rec.run[mine] == event.run).all()
        assert (first[mine] >= event.start + cfg.fs).all() and (
            last[mine] <= event.stop
        ).all()  # after the reaction time, inside the cue
    assert (
        set(np.unique(rec.run)) == {0, 1, 2, 3} and (rec.label[rec.run == 0] == "").all()
    )  # baseline and clench block are outside the runs
    assert set(np.unique(rec.cue)) >= {"", "rest", "clench", *PROTOCOL.commands}
    assert rec.baseline_frames > 40 and rec.reference.shape == (128, 10) and rec.resting_db.shape == (4,)

    data = build_dataset([rec])
    assert data.classes[0] == "rest" and set(data.classes) == set(PROTOCOL.labels)
    assert data.n_groups == 3 and data.n_trials <= len(events) and min(data.counts().values()) >= 12
    assert len(build_dataset([rec], clean_only=False)) > len(data)
    two = build_dataset([rec, rec])
    assert two.n_groups == 6 and two.n_trials == 2 * data.n_trials and two.references.shape[0] == 2
    assert build_dataset([rec], ("rest", "left")).classes == ("rest", "left")
    with pytest.raises(ValueError):
        build_dataset([])
    with pytest.raises(ValueError):
        build_dataset([rec], ("rest", "teleport"))
    assert load_recording(sim_session, reaction_s=2.0).label.tolist().count("rest") < rec.label.tolist().count("rest")


# ---------------------------------------------------------------------- evaluation
def test_evaluation_finds_a_signal_that_is_there(recording):
    corpus = Corpus([recording])
    result = evaluate(corpus, permutations=60)
    assert (
        result.accuracy > 0.75
        and result.chance == pytest.approx(1 / len(corpus.dataset.classes))
        and result.p_value < 0.05
    )
    assert result.n_runs == 3 and not result.weak_grouping and not result.confounded and not result.selected
    assert result.confusion.sum() == result.n_trials and set(result.recall) == set(corpus.dataset.classes)
    assert list(result.chosen) == ["tangent/fine"] and json.dumps(result.summary())
    _, truth, prediction = trial_decisions(log_softmax(result.scores), corpus.dataset.trial, corpus.dataset.y)
    assert (truth == prediction).mean() == pytest.approx(np.trace(result.confusion) / result.n_trials)

    nested = evaluate(corpus, select=True, permutations=0)
    assert (
        nested.selected
        and sum(nested.chosen.values()) == 3
        and nested.p_value is None
        and len(nested.ranking) == len(default_candidates())
    )
    assert nested.accuracy > 0.7


def test_evaluation_says_so_when_there_is_nothing_to_decode(tmp_path):
    protocol = ProtocolConfig(runs=3, trials_per_command=6, baseline_s=20.0, clench_trials=2)
    session = simulate_session(tmp_path / "null" / "s01", protocol, SimConfig(effect=0.0), seed=9)
    trained = train_bundle([session], permutations=100)
    result = trained.evaluation
    assert result.accuracy < 0.45 and result.p_value > 0.05
    assert trained.bundle.recommended_scheme == "switch" and trained.bundle.reliable == ()
    assert trained.tuning.score.false_per_min <= 0.5  # and whatever it does, it does not start by itself
    from nwc.report import training_report

    assert "no evidence that these commands can be told apart" in training_report(trained.bundle)


def test_one_recording_per_command_is_flagged_as_confounded(tmp_path):
    """The layout of the 2024 prototype's training data: one file per command, a label column."""
    rng = np.random.default_rng(0)
    paths = []
    for label, seed in ((0, 1), (1, 2), (3, 3)):
        # Nothing distinguishes the "commands" here except the recording: each has its own electrode gains.
        x = EEGSynth(SimConfig(effect=0.0), seed=seed).generate(256 * 60)
        stamps = 1000.0 * label + np.arange(len(x)) / 256.0
        path = tmp_path / f"brain_data{label}.csv"
        np.savetxt(path, np.column_stack((stamps, x, np.full(len(x), label))), delimiter=",", fmt="%.3f")
        paths.append(path)
    session = import_labelled_recordings(
        paths, tmp_path / "old" / "s", subject="old", session="s", names={"0": "rest", "1": "forward", "3": "left"}
    )
    assert session.labels == ["rest", "forward", "left"] and len(session.events("run")) == 3
    assert rng is not None
    trained = train_bundle([session], permutations=20)
    assert trained.evaluation.confounded and trained.evaluation.weak_grouping
    assert trained.evaluation.accuracy > 0.8  # it scores well, by telling the recordings apart
    from nwc.report import training_report

    assert "command and recording cannot be told apart" in training_report(trained.bundle)


def test_marker_recordings_import_as_runs(tmp_path):
    paths = []
    for run in range(2):
        x = EEGSynth(seed=run).generate(256 * 30)
        marker = np.zeros(len(x))
        marker[256 * 2 :: 256 * 4] = np.tile([1, 2], 4)[:7]  # a cue every four seconds, alternating
        stamps = np.arange(len(x)) / 256.0  # both files start at zero: they must still end up apart
        path = tmp_path / f"run{run}.csv"
        np.savetxt(
            path,
            np.column_stack((stamps, x, np.zeros(len(x)), marker)),
            delimiter=",",
            fmt="%.3f",
            header="timestamps,TP9,AF7,AF8,TP10,Right AUX,Marker0",
            comments="",
        )
        paths.append(path)
    session = import_marker_recordings(
        paths, tmp_path / "imp" / "s", markers={1: "fast", 2: "slow"}, cue_s=3.0, subject="imp", session="s"
    )
    cues = session.events("cue")
    assert len(cues) == 14 and {c.label for c in cues} == {"fast", "slow"} and {c.run for c in cues} == {1, 2}
    assert all(c.stop - c.start == 768 for c in cues)
    block, stats = session.block()
    assert stats.breaks == 1 and len(block) == 2 * 256 * 30  # one break, between the files
    rec = load_recording(session, reaction_s=0.5)
    assert rec.baseline_frames > 0 and set(rec.labels) == {
        "fast",
        "slow",
    }  # no baseline in an import: the session itself is the reference
    with pytest.raises(ValueError):
        import_marker_recordings(
            paths, tmp_path / "imp" / "t", markers={1: "a"}, cue_s=3.0, subject="imp", session="t", marker_column="Stim"
        )


# ---------------------------------------------------------------------- final model and bundle
def test_final_model_and_held_out_posteriors(recording):
    corpus = Corpus([recording])
    fitted = fit_final(corpus)
    model, data = fitted.model, corpus.dataset
    assert (
        model.classes == data.classes
        and fitted.candidate == Candidate(FeatureSpec("tangent", "fine"))
        and fitted.temperature > 1.0
    )
    proba = model.proba(data.cosp, recording.reference)
    assert (
        np.allclose(proba.sum(axis=1), 1.0) and (proba.argmax(axis=1) == data.y).mean() > 0.8
    )  # fitted on these very frames
    held = fitted.held_out[0]
    inside = recording.run > 0
    assert np.isfinite(held[inside]).all() and np.isnan(held[~inside]).all()
    # held-out posteriors are less sure than the final model's on its own training data
    rows = np.isin(recording.trial, data.trial) & (recording.label != "") & recording.quality.usable
    assert np.exp(held[rows]).max(axis=1).mean() <= proba.max(axis=1).mean() + 1e-6


def test_bundle_round_trip_and_self_checks(trained, bundle_dir, recording, tmp_path):
    bundle = load_bundle(bundle_dir)
    assert (
        bundle.classes == trained.bundle.classes
        and bundle.frontend == trained.bundle.frontend
        and bundle.arbiter == trained.bundle.arbiter
    )
    assert bundle.recommended_scheme == trained.bundle.recommended_scheme and bundle.reliable == trained.bundle.reliable
    assert bundle.clench == trained.bundle.clench and "held-out accuracy" in bundle.describe()
    data = trained.corpus.dataset
    assert np.allclose(
        bundle.model.log_proba(data.cosp[:50], recording.reference),
        trained.bundle.model.log_proba(data.cosp[:50], recording.reference),
    )
    report = json.loads((bundle_dir / "bundle.json").read_text())
    assert (
        report["frontend_fingerprint"] == bundle.frontend.fingerprint
        and report["report"]["evaluation"]["balanced_accuracy"] > 0.7
    )
    assert not list(bundle_dir.glob("*.pkl")) and not list(bundle_dir.glob("*.joblib"))  # arrays and JSON only

    import shutil

    # 1. the arithmetic changed: the stored self-test no longer reproduces
    damaged = tmp_path / "damaged"
    shutil.copytree(bundle_dir, damaged)
    with np.load(damaged / "model.npz") as arrays:
        content = {k: arrays[k] for k in arrays.files}
    content["weights"] = content["weights"] * 1.01
    np.savez(damaged / "model.npz", **content)
    with pytest.raises(BundleError, match="self-test"):
        load_bundle(damaged)
    # 2. the front-end was edited after training
    edited = tmp_path / "edited"
    shutil.copytree(bundle_dir, edited)
    manifest = json.loads((edited / "bundle.json").read_text())
    manifest["frontend"]["hop_s"] = 0.5
    (edited / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="front-end"):
        load_bundle(edited)
    # 3. not a bundle at all
    with pytest.raises(BundleError):
        load_bundle(tmp_path)


def test_a_model_from_one_day_holds_up_on_another(trained, second_session):
    """Each session is expressed against its own baseline, so a new seating of the headband does not start from zero."""
    model = trained.bundle.model
    rec = load_recording(second_session, trained.bundle.frontend, trained.bundle.quality)
    data = build_dataset([rec], trained.bundle.classes)

    def accuracy(cosp, reference):
        _, truth, prediction = trial_decisions(model.log_proba(cosp, reference), data.trial, data.y)
        return float((truth == prediction).mean())

    own = accuracy(data.cosp, rec.reference)
    assert own > 0.6  # four classes: chance is 0.25

    # Now the electrodes sit differently: every channel has another gain, as after re-seating the headband.
    from nwc.dsp import spectral

    gain = np.array([0.4, 2.0, 0.6, 1.8])
    rows, cols = spectral.pair_index(4)
    factor = gain[rows] * gain[cols]
    moved, moved_reference = data.cosp * factor, rec.reference * factor
    # Against a baseline taken with the electrodes where they are now, nothing has changed at all...
    assert np.allclose(
        model.log_proba(moved[:40], moved_reference), model.log_proba(data.cosp[:40], rec.reference), atol=1e-6
    )
    assert accuracy(moved, moved_reference) == own
    # ...against the baseline from before they moved, the model is lost.
    assert accuracy(moved, rec.reference) < own - 0.1

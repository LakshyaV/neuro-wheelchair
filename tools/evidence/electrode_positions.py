"""What electrode positions like a Muse's can and cannot pick up.

Data: the EEG Motor Movement/Imagery Database on PhysioNet (Schalk et al., 2004;
Goldberger et al., 2000): 64-channel research EEG with cued, randomised trials.
A Muse measures at AF7, AF8, TP9 and TP10 against Fpz. The database has AF7,
AF8, TP7, TP8 and Fpz, so a Muse-like montage can be emulated and compared with
four electrodes over the motor cortex, same reference, same pipeline.

    pip install mne
    python tools/evidence/electrode_positions.py                 # downloads 30 people on first use
    python tools/evidence/electrode_positions.py --subjects 10

Two questions are asked:

1. Imagined movement, the classic "think left, think right": can it be decoded
   at these positions? Each person's three runs are split by run, with the code
   that `nwc train` uses.
2. Eyes closed against eyes open: a model that has never seen the person, with
   and without expressing each person relative to their own reference.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from nwc.decode.dataset import load_recording
from nwc.decode.features import FeatureSpec, Featurizer
from nwc.decode.lda import fit_lda
from nwc.decode.train import Corpus, evaluate
from nwc.dsp.frontend import FrontendConfig, process
from nwc.dsp.quality import QualityConfig
from nwc.io.session import CUE, RUN, Session, SessionWriter
from nwc.io.stream import Chunk, StreamInfo, assemble

FS = 160.0
MONTAGES = {
    "Muse-like (AF7 AF8 TP7 TP8)": ("AF7", "AF8", "TP7", "TP8"),
    "motor cortex (C3 C4 Cz CPz)": ("C3", "C4", "CZ", "CPZ"),
}
TASKS = {
    "imagined left fist against right fist": ((4, 8, 12), {"T1": "left", "T2": "right"}),
    "imagined fists against feet": ((6, 10, 14), {"T1": "fists", "T2": "feet"}),
    "real left fist against right fist": ((3, 7, 11), {"T1": "left", "T2": "right"}),
}


#: The default limits are those of a Muse. This amplifier, referenced to Fpz, reads several times
#: larger (median 15 to 35 uV in 1-45 Hz), so only gross artefacts are excluded here.
QUALITY = QualityConfig(rms_min_uv=0.1, rms_max_uv=200.0, swing_max_uv=600.0)


def frontend(channels: tuple[str, ...]) -> FrontendConfig:
    return FrontendConfig(fs=FS, channels=channels, emg_band_hz=(30.0, 70.0))


def load_run(root: Path, subject: int, run: int, channels: tuple[str, ...]):
    """One run re-referenced to Fpz, in microvolts, with its annotations as (sample, code, samples)."""
    import mne
    from mne.datasets import eegbci

    mne.set_log_level("ERROR")
    path = eegbci.load_data(subject, [run], path=str(root), update_path=False, verbose="ERROR")[0]
    raw = mne.io.read_raw_edf(path, preload=True)
    names = [name.strip(".").upper() for name in raw.ch_names]
    data = raw.get_data() * 1e6
    reference = data[names.index("FPZ")]
    x = np.stack([data[names.index(c)] - reference for c in channels], axis=1)
    events = [
        (round(o * FS), d, round(n * FS))
        for o, d, n in zip(raw.annotations.onset, raw.annotations.description, raw.annotations.duration)
    ]
    return x, events


def as_session(
    directory: Path, root: Path, subject: int, runs: tuple[int, ...], labels: dict[str, str], channels: tuple[str, ...]
) -> Session:
    """The three runs of a task as one session, written the way `nwc record` writes."""
    info = StreamInfo(channels, FS, source="eegmmidb", device="BCI2000 64 channels")
    trial = 0
    with SessionWriter(directory, info, subject=f"S{subject:03d}", session=directory.name) as writer:
        for index, run in enumerate(runs, start=1):
            x, events = load_run(root, subject, run, channels)
            first = writer.samples
            writer.write(Chunk(index * 1000.0 + np.arange(len(x)) / FS, x))  # runs far apart in time: filters restart
            for onset, code, length in events:
                name = "rest" if code == "T0" else labels[code]
                if onset + length <= len(x):
                    writer.event(CUE, first + onset, first + onset + length, name, index, trial)
                    trial += 1
            writer.event(RUN, first, writer.samples, "", index)
    return Session(directory)


def imagery(root: Path, subjects: list[int]) -> None:
    from scipy import stats

    print("1. Per person, whole runs held out, decisions per trial (about 45 trials, chance 50 %)\n")
    for task, (runs, labels) in TASKS.items():
        print(f"  {task}")
        for name, channels in MONTAGES.items():
            scores, trials, skipped = [], [], []
            for subject in subjects:
                with tempfile.TemporaryDirectory() as tmp:
                    session = as_session(Path(tmp) / "task", root, subject, runs, labels, channels)
                    rec = load_recording(session, frontend(channels), QUALITY, reaction_s=0.5)
                try:
                    result = evaluate(Corpus([rec], tuple(labels.values())), permutations=0)
                except ValueError:  # a run without usable trials of both kinds
                    skipped.append(subject)
                    continue
                scores.append(result.accuracy)
                trials.append(result.n_trials)
            scores = np.array(scores)
            bound = stats.binom.ppf(0.95, int(np.mean(trials)), 0.5) / np.mean(
                trials
            )  # what a coin reaches one time in twenty
            above = int((scores > bound).sum())
            note = f"   (left out: {', '.join(f'S{n:03d}' for n in skipped)})" if skipped else ""
            print(
                f"    {name:30s} mean {100 * scores.mean():5.1f} %   median {100 * np.median(scores):5.1f} %   "
                f"people above {100 * bound:.0f} %: {above} of {len(scores)}{note}"
            )
        print()


def eyes(root: Path, subjects: list[int]) -> None:
    print("2. Eyes closed against eyes open, one minute each, tested on a person the model has never seen\n")
    for name, channels in MONTAGES.items():
        cfg = frontend(channels)
        people = []
        for subject in subjects:
            cosp, y = [], []
            for label, run in enumerate((1, 2)):  # run 1: eyes open, run 2: eyes closed
                x, _ = load_run(root, subject, run, channels)
                frames, _ = process(assemble(np.arange(len(x)) / FS, x, StreamInfo(channels, FS))[0], cfg)
                keep = frames.valid & (np.arange(len(frames)) % 8 == 0)  # non-overlapping two-second windows
                cosp.append(frames.cosp[keep])
                y.append(np.full(len(cosp[-1]), label))
            people.append((np.concatenate(cosp).astype(np.float64), np.concatenate(y)))
        for kind in ("power", "tangent"):
            featurizer = Featurizer(FeatureSpec(kind, "fine"), cfg.n_bins, cfg.n_channels)
            for own in (False, True):
                scores = []
                for held in range(len(people)):
                    pooled = np.concatenate([c for i, (c, _) in enumerate(people) if i != held]).mean(axis=0)
                    features = [featurizer(c, c.mean(axis=0) if own else pooled) for c, _ in people]
                    X = np.concatenate([f for i, f in enumerate(features) if i != held])
                    y = np.concatenate([labels for i, (_, labels) in enumerate(people) if i != held])
                    mean, scale, weights, bias = fit_lda(X, y, 2)
                    prediction = (((features[held] - mean) / scale) @ weights.T + bias).argmax(axis=1)
                    scores.append(float((prediction == people[held][1]).mean()))
                scores = np.array(scores)
                how = "relative to the person's own reference" if own else "relative to everyone else's reference"
                print(
                    f"    {name.split(' (')[0]:13s} {kind:8s} {how:40s} mean {100 * scores.mean():5.1f} %   median {100 * np.median(scores):5.1f} %   worst {100 * scores.min():5.1f} %"
                )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("evidence-data/eegmmidb"))
    parser.add_argument("--subjects", type=int, default=30)
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    subjects = list(range(1, args.subjects + 1))
    print(f"{len(subjects)} people, two-second windows, tangent-space features over 2 Hz bands, shrinkage LDA\n")
    imagery(args.data, subjects)
    eyes(args.data, subjects)


if __name__ == "__main__":
    main()

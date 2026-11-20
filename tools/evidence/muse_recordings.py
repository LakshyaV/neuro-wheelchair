"""Decoders compared on real Muse recordings with interleaved conditions.

Data: the visual SSVEP example recordings of EEG-ExPy (NeuroTechX), made with a
Muse 2016. Two flicker frequencies alternate at random every few seconds inside
each two-minute recording, so the condition is not tied to the recording and
whole recordings can be held out.

    python tools/evidence/muse_recordings.py --download        # fetch the recordings (needs gdown)
    python tools/evidence/muse_recordings.py                   # this pipeline and the 2024 features
    python tools/evidence/muse_recordings.py --eegnet          # plus a convolutional network (needs torch)

Everything except the two reference methods (the 2024 feature set with a linear
SVM, and EEGNet) is the code that `nwc train` runs.
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from nwc.decode.dataset import load_recording
from nwc.decode.train import Corpus, default_candidates, evaluate, rank_candidates
from nwc.io.importers import import_marker_recordings

DRIVE_ID = "1zj9Wx-YEMJo7GugUUu7Sshcybfsr-Fze"  # listed in eegnb/datasets/datasets.py of EEG-ExPy
MARKERS = {1: "30 Hz", 2: "20 Hz"}
SUBJECTS = ("subject0001", "subject0004")
FS = 256


def download(root: Path) -> None:
    import gdown

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "visual-SSVEP.zip"
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_ID}", str(archive), quiet=False)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(root)
    archive.unlink()


def recordings(root: Path, subject: str) -> list[Path]:
    """Recordings of one person in which all four electrodes had contact."""
    good = []
    for path in sorted(root.rglob(f"{subject}/**/*.csv")):
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        if data[:, 1:5].std(axis=0).max() <= 100.0:
            good.append(path)
    return good


def read(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header = path.read_text().split("\n", 1)[0].strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    marker = np.nan_to_num(data[:, header.index("Marker0")]).astype(int)
    onsets = np.flatnonzero(marker != 0)
    return data[:, 1:5], onsets, marker[onsets]


# ---------------------------------------------------------------------- the 2024 prototype's features
def prototype_features(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """classification.py's features with its bugs fixed and nothing else changed.

    A 55-65 Hz band-stop, one-second windows with half overlap, the mean FFT amplitude
    in delta, theta, alpha and beta per channel, log10. Returns features, labels, the
    recording and the trial of every window.
    """
    from scipy import signal

    notch = signal.butter(4, np.array([55, 65]) / (FS / 2), btype="bandstop")
    features, labels, groups, trials = [], [], [], []
    trial = -1
    for group, path in enumerate(paths):
        x, onsets, codes = read(path)
        x = signal.lfilter(*notch, x, axis=0)
        frequency = np.fft.rfftfreq(FS, 1 / FS)[: FS // 2]
        bands = [
            frequency < 4,
            (frequency >= 4) & (frequency <= 8),
            (frequency >= 8) & (frequency <= 12),
            (frequency >= 12) & (frequency < 30),
        ]
        for onset, code in zip(onsets, codes):
            trial += 1
            cue = x[onset : onset + 3 * FS]
            for start in range(0, len(cue) - FS + 1, FS // 2):
                window = cue[start : start + FS]
                window = (window - window.mean(axis=0)) * np.hamming(FS)[:, None]
                amplitude = 2 * np.abs(np.fft.rfft(window, axis=0)[: FS // 2]) / FS
                features.append(np.log10(np.concatenate([amplitude[band].mean(axis=0) for band in bands])))
                labels.append(code)
                groups.append(group)
                trials.append(trial)
    return np.array(features), np.array(labels), np.array(groups), np.array(trials)


def prototype_scores(paths: list[Path]) -> tuple[float, float, float]:
    """The 2024 method scored three ways.

    Accuracy on single windows with a random split (as the prototype did), the same
    with whole recordings held out, and, for a like-for-like comparison with the
    pipeline, balanced accuracy of one decision per trial with recordings held out
    (the trial's windows vote with their distance from the SVM's boundary).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y, groups, trials = prototype_features(paths)

    def model():
        return make_pipeline(StandardScaler(), SVC(kernel="linear", C=1))

    random_split = []
    for seed in range(20):
        a, b, ya, yb = train_test_split(X, y, test_size=0.3, random_state=seed)
        random_split.append(model().fit(a, ya).score(b, yb))
    correct = 0
    votes = np.zeros(len(y))
    classes = np.unique(y)
    for g in np.unique(groups):
        test = groups == g
        fitted = model().fit(X[~test], y[~test])
        correct += int((fitted.predict(X[test]) == y[test]).sum())
        votes[test] = fitted.decision_function(X[test])  # positive: the second class
    truth = np.array([y[trials == t][0] for t in np.unique(trials)])
    decided = np.array([classes[int(votes[trials == t].mean() > 0)] for t in np.unique(trials)])
    per_trial = float(np.mean([(decided[truth == c] == c).mean() for c in classes]))
    return float(np.mean(random_split)), correct / len(y), per_trial


# ---------------------------------------------------------------------- a compact convolutional network
def eegnet_crops(paths: list[Path]):
    """Two-second crops at 128 Hz (0.5, 0.75 and 1.0 s after stimulus onset): crops, labels, recording, trial."""
    from scipy import signal

    highpass = signal.butter(2, 1.0, "highpass", fs=FS, output="sos")
    lowpass = signal.butter(4, 45.0, "lowpass", fs=FS, output="sos")
    notch = signal.iirnotch(60.0, 30.0, fs=FS)
    crops, labels, groups, trials = [], [], [], []
    trial = 0
    for group, path in enumerate(paths):
        x, onsets, codes = read(path)
        x = signal.sosfilt(lowpass, signal.lfilter(*notch, signal.sosfilt(highpass, x, axis=0), axis=0), axis=0)[::2]
        for onset, code in zip(onsets // 2, codes):
            if onset + 3 * FS // 2 > len(x):
                continue
            for start in (onset + FS // 4, onset + 3 * FS // 8, onset + FS // 2):
                crops.append(x[start : start + FS].T)
                labels.append(code - 1)
                groups.append(group)
                trials.append(trial)
            trial += 1
    return np.array(crops, dtype=np.float32), np.array(labels), np.array(groups), np.array(trials)


def eegnet_fit_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int, epochs: int = 100
) -> np.ndarray:
    """EEGNet (Lawhern et al., 2018), trained from scratch; returns log-probabilities for the test crops."""
    import torch
    from torch import nn

    torch.set_num_threads(4)
    torch.backends.nnpack.set_flags(False)
    torch.manual_seed(seed)
    scale = train_x.std(axis=(0, 2), keepdims=True)
    a, ya, b = torch.tensor(train_x / scale), torch.tensor(train_y), torch.tensor(test_x / scale)
    model = nn.Sequential(
        nn.Unflatten(1, (1, 4)),
        nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(8),
        nn.Conv2d(8, 16, (4, 1), groups=8, bias=False), nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.5),
        nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False), nn.Conv2d(16, 16, 1, bias=False),
        nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.5), nn.Flatten(), nn.LazyLinear(2),
    )  # fmt: skip
    model(a[:2])
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(a))
        for i in range(0, len(order), 64):
            batch = order[i : i + 64]
            if len(batch) < 2:
                continue  # batch normalisation needs more than one crop
            optimiser.zero_grad()
            nn.functional.cross_entropy(model(a[batch]), ya[batch]).backward()
            optimiser.step()
        schedule.step()
    model.eval()
    with torch.no_grad():
        return torch.log_softmax(model(b), dim=1).numpy()


def trial_accuracy(log_proba: np.ndarray, y: np.ndarray, trials: np.ndarray) -> float:
    """Balanced accuracy of per-trial decisions (summed log-probabilities)."""
    truth, prediction = [], []
    for t in np.unique(trials):
        rows = trials == t
        truth.append(y[rows][0])
        prediction.append(int(log_proba[rows].sum(axis=0).argmax()))
    truth, prediction = np.array(truth), np.array(prediction)
    return float(np.mean([(prediction[truth == k] == k).mean() for k in np.unique(truth)]))


def eegnet_score(paths: list[Path], seeds: int = 3) -> list[float]:
    """Whole recordings held out in turn, decisions per trial."""
    X, y, groups, trials = eegnet_crops(paths)
    scores = []
    for seed in range(seeds):
        log_proba = np.zeros((len(y), 2))
        for g in np.unique(groups):
            test = groups == g
            log_proba[test] = eegnet_fit_predict(X[~test], y[~test], X[test], seed)
        scores.append(trial_accuracy(log_proba, y, trials))
    return scores


def calibration_sized(paths: list[Path], corpus: Corpus, eegnet: bool, max_splits: int = 10) -> None:
    """Train on one, two or three recordings (about 30 trials each) and test on all the others."""
    from itertools import combinations

    from nwc.decode.features import FeatureSpec
    from nwc.decode.lda import fit_lda
    from nwc.decode.model import log_softmax

    data = corpus.dataset
    run_ids = np.unique(data.group)
    if eegnet:
        X, y, groups, trials = eegnet_crops(paths)
    rng = np.random.default_rng(0)
    print("  trained on k recordings and tested on all the others (mean over up to ten splits):")
    for k in (1, 2, 3):
        splits = list(combinations(range(len(run_ids)), k))
        if len(splits) > max_splits:
            splits = [splits[i] for i in rng.choice(len(splits), max_splits, replace=False)]
        scores: dict[str, list[float]] = {"tangent": [], "power": [], "eegnet": []}
        counts = []
        for split in splits:
            train = np.isin(data.group, run_ids[list(split)])
            counts.append(len(np.unique(data.trial[train])))
            for kind in ("tangent", "power"):
                F = corpus.features(FeatureSpec(kind, "fine"))
                mean, scale, weights, bias = fit_lda(F[train], data.y[train], len(data.classes))
                log_proba = log_softmax(((F[~train] - mean) / scale) @ weights.T + bias)
                scores[kind].append(trial_accuracy(log_proba, data.y[~train], data.trial[~train]))
            if eegnet:
                held = ~np.isin(groups, list(split))
                scores["eegnet"].append(
                    trial_accuracy(eegnet_fit_predict(X[~held], y[~held], X[held], seed=0), y[held], trials[held])
                )
        line = f"    k = {k} (about {int(np.mean(counts))} trials): tangent {100 * np.mean(scores['tangent']):5.1f} %   power {100 * np.mean(scores['power']):5.1f} %"
        if eegnet:
            line += f"   EEGNet {100 * np.mean(scores['eegnet']):5.1f} %"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("evidence-data/muse-ssvep"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--eegnet", action="store_true")
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    if args.download:
        download(args.data)
    for subject in SUBJECTS:
        paths = recordings(args.data, subject)
        if not paths:
            raise SystemExit(f"no recordings of {subject} under {args.data}: run with --download")
        with tempfile.TemporaryDirectory() as tmp:
            session = import_marker_recordings(
                paths, Path(tmp) / subject / "ssvep", markers=MARKERS, cue_s=3.0, subject=subject, session="ssvep"
            )
            rec = load_recording(session, reaction_s=0.5)
        corpus = Corpus([rec])
        data = corpus.dataset
        majority = max(data.counts().values()) / data.n_trials
        print(
            f"\n{subject}: {len(paths)} recordings, {data.n_trials} trials, {100 * rec.quality.usable.mean():.0f} % of frames usable, larger class {100 * majority:.0f} %"
        )
        random_split, held_out, per_trial = prototype_scores(paths)
        print(f"  2024 features + linear SVM, single windows, random split          {100 * random_split:5.1f} %")
        print(f"  2024 features + linear SVM, single windows, recordings held out   {100 * held_out:5.1f} %")
        print(f"  2024 features + linear SVM, one decision per trial, held out      {100 * per_trial:5.1f} %")
        result = evaluate(corpus, permutations=args.permutations)
        print(
            f"  this pipeline (tangent space, 2 Hz bands, shrinkage LDA)          {100 * result.accuracy:5.1f} %   p = {result.p_value:.3f}"
        )
        nested = evaluate(corpus, select=True, permutations=args.permutations)
        print(
            f"  this pipeline with nested selection among four configurations     {100 * nested.accuracy:5.1f} %   p = {nested.p_value:.3f}   chosen {nested.chosen}"
        )
        ranking = ", ".join(f"{c.name} {100 * s:.1f}" for c, s in rank_candidates(corpus, default_candidates()))
        print(f"  best-looking configuration when picked on the test data itself    {ranking}")
        if args.eegnet:
            scores = eegnet_score(paths)
            print(
                f"  EEGNet on raw two-second crops, three seeds                        {100 * np.mean(scores):5.1f} %   ({', '.join(f'{100 * v:.1f}' for v in scores)})"
            )
        calibration_sized(paths, corpus, args.eegnet)


if __name__ == "__main__":
    main()

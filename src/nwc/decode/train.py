"""Fitting and evaluation of the intent classifier.

Three rules shape this module.

*Splits are by run.* Frames of a trial overlap by seven eighths, and trials of a
run share electrode contact and time of day. Holding out whole runs is the only
split whose score predicts the next run.

*Chance is measured, not assumed.* Decisions are scored per trial, and the whole
procedure is repeated with trial labels shuffled inside each run. The share of
shuffles that do as well is the p-value printed with every result.

*One configuration by default.* A single feature set (tangent space, 2 Hz bands)
is used unless selection is asked for. Picking the best of several configurations
on the data that scores them flatters the result: on a real recording that is at
chance, the best-looking of four configurations showed 61 %. When selection is
requested it is therefore nested, so every held-out run is predicted by a
configuration chosen without it.

The classifier is shrinkage LDA (:mod:`nwc.decode.lda`): closed-form, nothing to
tune, a millisecond per fit. That speed is what makes the shuffling test here and
the replay tuning of the controller affordable. ``docs/evidence.md`` compares it
with a convolutional network on real recordings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from .dataset import Recording, build_dataset
from .features import FeatureSpec, Featurizer
from .lda import fit_lda
from .model import IntentModel, log_softmax


@dataclass(frozen=True)
class Candidate:
    spec: FeatureSpec

    @property
    def name(self) -> str:
        return self.spec.name


def default_candidates() -> list[Candidate]:
    """The default configuration first, then the alternatives that selection may consider."""
    return [Candidate(FeatureSpec(kind, layout)) for layout in ("fine", "classic") for kind in ("tangent", "power")]


# ---------------------------------------------------------------------- linear fits
def _scores(params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], X: np.ndarray) -> np.ndarray:
    mean, scale, weights, bias = params
    return ((X - mean) / scale) @ weights.T + bias


def fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    """The single scale that makes held-out probabilities as honest as possible (minimum log-loss)."""
    rows = np.arange(len(y))

    def loss(log_t: float) -> float:
        return float(-log_softmax(scores / np.exp(log_t))[rows, y].mean())

    result = minimize_scalar(loss, bounds=(np.log(0.05), np.log(50.0)), method="bounded")
    return float(np.exp(result.x))


# ---------------------------------------------------------------------- scoring
def trial_decisions(
    log_proba: np.ndarray, trial: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One decision per trial from the summed frame log-probabilities: ``(trial ids, truth, prediction)``."""
    ids, inverse = np.unique(trial, return_inverse=True)
    summed = np.zeros((len(ids), log_proba.shape[1]))
    np.add.at(summed, inverse, log_proba)
    truth = np.zeros(len(ids), dtype=np.int64)
    truth[inverse] = y
    return ids, truth, summed.argmax(axis=1)


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray, n_classes: int) -> float:
    recalls = [float((prediction[truth == k] == k).mean()) for k in range(n_classes) if (truth == k).any()]
    return float(np.mean(recalls)) if recalls else 0.0


def confusion(truth: np.ndarray, prediction: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    return matrix


# ---------------------------------------------------------------------- corpus
class Corpus:
    """Recordings, the dataset built from them and cached feature matrices."""

    def __init__(self, recordings: list[Recording], classes: tuple[str, ...] | None = None, *, clean_only: bool = True):
        self.recordings = recordings
        self.dataset = build_dataset(recordings, classes, clean_only=clean_only)
        first = recordings[0].frames
        self.n_bins, self.n_pairs = first.cosp.shape[1], first.cosp.shape[2]
        self.n_channels = recordings[0].frames.swing.shape[1]
        self._train: dict[FeatureSpec, np.ndarray] = {}
        self._all: dict[tuple[FeatureSpec, int], np.ndarray] = {}
        self.weak_grouping = False
        self.confounded = False
        self.groups = self._groups()

    def _groups(self) -> np.ndarray:
        """What is held out together.

        Normally the runs. Two situations need a substitute, and both are flagged
        because the resulting score is weaker evidence:

        * a single run is cut into four consecutive blocks of trials;
        * runs that each hold one command only (one recording per command) cannot
          be held out at all, so every run is cut into four consecutive quarters
          and the same quarter of all runs is held out together. The classifier
          can then still succeed by recognising the recording, not the command.
        """
        data = self.dataset
        per_run = [len(np.unique(data.y[data.group == g])) for g in np.unique(data.group)]
        self.confounded = data.n_groups >= 2 and max(per_run) == 1
        if data.n_groups >= 2 and not self.confounded:
            return data.group
        self.weak_grouping = True
        out = np.zeros(len(data), dtype=np.int64)
        for g in np.unique(data.group):
            ids = np.unique(data.trial[data.group == g])
            for b, block in enumerate(np.array_split(ids, min(4, len(ids)))):
                out[np.isin(data.trial, block)] = b
        return out

    def featurizer(self, spec: FeatureSpec) -> Featurizer:
        return Featurizer(spec, self.n_bins, self.n_channels)

    def features(self, spec: FeatureSpec) -> np.ndarray:
        """Features of the dataset's frames, each relative to its own session's reference."""
        if spec not in self._train:
            data, featurizer = self.dataset, self.featurizer(spec)
            X = np.empty((len(data), featurizer.size))
            for s in np.unique(data.session):
                rows = data.session == s
                X[rows] = featurizer(data.cosp[rows], data.references[s])
            self._train[spec] = X
        return self._train[spec]

    def features_all(self, spec: FeatureSpec, s: int) -> np.ndarray:
        """Features of every frame of recording ``s`` (labelled or not), for replay."""
        if (spec, s) not in self._all:
            rec = self.recordings[s]
            self._all[(spec, s)] = self.featurizer(spec)(rec.frames.cosp, rec.reference)
        return self._all[(spec, s)]


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_classes: int) -> np.ndarray:
    """Leave-one-group-out scores ``[n, K]`` (before temperature)."""
    scores = np.zeros((len(y), n_classes))
    for g in np.unique(groups):
        test = groups == g
        scores[test] = _scores(fit_lda(X[~test], y[~test], n_classes), X[test])
    return scores


def _trial_score(scores: np.ndarray, data_trial: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    _, truth, prediction = trial_decisions(log_softmax(scores), data_trial, y)
    return balanced_accuracy(truth, prediction, n_classes)


def rank_candidates(
    corpus: Corpus, candidates: list[Candidate], rows: np.ndarray | None = None
) -> list[tuple[Candidate, float]]:
    """Cross-validated trial-level balanced accuracy of every candidate on ``rows``, best first."""
    data = corpus.dataset
    rows = np.ones(len(data), dtype=bool) if rows is None else rows
    k = len(data.classes)
    results = []
    for candidate in candidates:
        X = corpus.features(candidate.spec)[rows]
        scores = cross_validate(X, data.y[rows], corpus.groups[rows], k)
        results.append((candidate, _trial_score(scores, data.trial[rows], data.y[rows], k)))
    # A stable sort keeps the first-listed candidate on ties, so the order of candidates is a preference.
    return sorted(results, key=lambda item: -item[1])


@dataclass
class Evaluation:
    """What held-out data says about a corpus."""

    classes: tuple[str, ...]
    selected: bool  # configurations were chosen per fold (nested)
    weak_grouping: bool  # a single run had to be cut into blocks to hold something out
    confounded: bool  # every run holds a single command: command and recording cannot be told apart
    n_trials: int
    n_runs: int
    accuracy: float  # trial-level balanced accuracy on held-out runs
    frame_accuracy: float
    chance: float
    p_value: float | None
    permutations: int
    confusion: np.ndarray  # [K, K] trials: rows truth, columns prediction
    recall: dict[str, float]
    chosen: dict[str, int]  # configuration -> number of folds that used it
    ranking: list[
        tuple[str, float]
    ]  # with selection: plain cross-validation of every candidate (optimistic, for information)
    scores: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros(0)
    )  # held-out scores of the dataset's frames

    def summary(self) -> dict:
        return {
            "classes": list(self.classes),
            "trials": self.n_trials,
            "runs": self.n_runs,
            "balanced_accuracy": round(self.accuracy, 4),
            "frame_accuracy": round(self.frame_accuracy, 4),
            "chance": round(self.chance, 4),
            "p_value": None if self.p_value is None else round(self.p_value, 4),
            "permutations": self.permutations,
            "recall": {k: round(v, 4) for k, v in self.recall.items()},
            "confusion": self.confusion.tolist(),
            "selected": self.selected,
            "weak_grouping": self.weak_grouping,
            "confounded": self.confounded,
            "chosen": self.chosen,
            "ranking": [[name, round(score, 4)] for name, score in self.ranking],
        }


def _held_out_scores(
    corpus: Corpus, candidates: list[Candidate], y: np.ndarray, select: bool
) -> tuple[np.ndarray, dict[str, int]]:
    """Leave-one-run-out scores. With ``select`` each fold's configuration is chosen on the other runs."""
    data, groups = corpus.dataset, corpus.groups
    k = len(data.classes)
    scores = np.zeros((len(data), k))
    chosen: dict[str, int] = {}
    for g in np.unique(groups):
        test = groups == g
        best = candidates[0]
        if select:
            inner = []
            for candidate in candidates:
                X = corpus.features(candidate.spec)[~test]
                inner_scores = cross_validate(X, y[~test], groups[~test], k)
                inner.append(_trial_score(inner_scores, data.trial[~test], y[~test], k))
            best = candidates[int(np.argmax(inner))]  # first of equals: the order of candidates is a preference
        chosen[best.name] = chosen.get(best.name, 0) + 1
        X = corpus.features(best.spec)
        scores[test] = _scores(fit_lda(X[~test], y[~test], k), X[test])
    return scores, chosen


def evaluate(
    corpus: Corpus,
    candidates: list[Candidate] | None = None,
    *,
    select: bool = False,
    permutations: int = 200,
    budget_s: float = 60.0,
    seed: int = 0,
) -> Evaluation:
    """Leave-one-run-out evaluation with a label-shuffling test of the same procedure.

    ``select`` needs at least three runs (two to choose on, one to test on). The
    shuffling test repeats everything, selection included; when that is slow the
    number of shuffles is reduced to fit ``budget_s`` and reported.
    """
    import time

    data = corpus.dataset
    candidates = candidates or default_candidates()
    k = len(data.classes)
    select = select and len(candidates) > 1 and len(np.unique(corpus.groups)) >= 3
    started = time.perf_counter()
    scores, chosen = _held_out_scores(corpus, candidates, data.y, select)
    one_pass = time.perf_counter() - started
    log_proba = log_softmax(scores)
    _, truth, prediction = trial_decisions(log_proba, data.trial, data.y)
    accuracy = balanced_accuracy(truth, prediction, k)

    p_value = None
    if permutations > 0:
        permutations = int(min(permutations, max(20, budget_s / max(one_pass, 1e-4))))
        rng = np.random.default_rng(seed)
        ids, first = np.unique(data.trial, return_index=True)
        trial_label, trial_group = data.y[first], corpus.groups[first]
        position = np.searchsorted(ids, data.trial)
        at_least = 0
        for _ in range(permutations):
            shuffled = trial_label.copy()
            for g in np.unique(trial_group):  # whole trials change label, within their run
                sel = np.flatnonzero(trial_group == g)
                shuffled[sel] = rng.permutation(shuffled[sel])
            y_perm = shuffled[position]
            perm_scores, _ = _held_out_scores(corpus, candidates, y_perm, select)
            at_least += _trial_score(perm_scores, data.trial, y_perm, k) >= accuracy
        p_value = (at_least + 1) / (permutations + 1)

    matrix = confusion(truth, prediction, k)
    recall = {name: float(matrix[i, i] / max(matrix[i].sum(), 1)) for i, name in enumerate(data.classes)}
    ranking = [(c.name, s) for c, s in rank_candidates(corpus, candidates)] if select else []
    return Evaluation(
        data.classes, select, corpus.weak_grouping, corpus.confounded, len(truth), len(np.unique(corpus.groups)), accuracy,
        float((log_proba.argmax(axis=1) == data.y).mean()), 1.0 / k, p_value, permutations, matrix, recall, chosen, ranking, scores,
    )  # fmt: skip


@dataclass
class Fitted:
    """A model fitted on everything, with the held-out material needed to tune the controller."""

    model: IntentModel
    candidate: Candidate
    temperature: float
    held_out: list[
        np.ndarray
    ]  # per recording: log-probabilities [frames, K] from models that never saw that run (NaN elsewhere)


def fit_final(corpus: Corpus, candidates: list[Candidate] | None = None, *, select: bool = False) -> Fitted:
    """Fit the configuration on everything and calibrate its probabilities on held-out runs.

    With ``select`` the configuration is the one that cross-validates best on all runs.
    """
    data = corpus.dataset
    candidates = candidates or default_candidates()
    k = len(data.classes)
    best = rank_candidates(corpus, candidates)[0][0] if select and len(candidates) > 1 else candidates[0]
    X = corpus.features(best.spec)
    scores = cross_validate(X, data.y, corpus.groups, k)
    temperature = fit_temperature(scores, data.y)
    mean, scale, weights, bias = fit_lda(X, data.y, k)
    model = IntentModel(
        data.classes, best.spec, mean, scale, weights, bias, temperature, corpus.n_bins, corpus.n_channels
    )

    # Posteriors for every frame of every run, each from a model fitted without that run.
    held_out = [np.full((len(rec), k), np.nan) for rec in corpus.recordings]
    for g in np.unique(corpus.groups):
        test = corpus.groups == g
        params = fit_lda(X[~test], data.y[~test], k)
        if corpus.weak_grouping:
            continue  # blocks of one run are not separable on the frame timeline
        s = int(data.session[test][0])
        rec = corpus.recordings[s]
        run = int(data.group[test][0]) - 1_000 * s
        frames = rec.run == run
        held_out[s][frames] = log_softmax(_scores(params, corpus.features_all(best.spec, s)[frames]) / temperature)
    return Fitted(model, best, temperature, held_out)

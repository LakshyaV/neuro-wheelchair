"""From recordings to a bundle: evaluate, fit, tune, set the reflex threshold, save.

This is what ``nwc train`` runs. The steps are separate functions elsewhere;
here they are put in the one order that keeps the result honest: the evaluation
that is reported is done before the final fit, and the controller's thresholds
are tuned on probabilities from models that never saw the run they are replayed
on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..calibration.protocol import CLENCH, REST
from ..control.tune import Tuning, tune
from ..dsp.frontend import FrontendConfig
from ..dsp.quality import QualityConfig
from ..dsp.reflex import ClenchConfig, fit_threshold
from ..io.session import Session
from .bundle import Bundle, save_bundle
from .dataset import Recording, load_recording
from .train import Candidate, Corpus, Evaluation, default_candidates, evaluate, fit_final


def clench_threshold(
    recordings: list[Recording], frontend: FrontendConfig, config: ClenchConfig | None = None
) -> tuple[ClenchConfig, dict[str, float]]:
    """Fit the stop reflex to the calibration clenches, if the recordings contain any."""
    config = config or ClenchConfig()
    columns = [frontend.channels.index(c) for c in config.channels]
    rest, clench = [], []
    react = round(0.5 * frontend.fs)
    for rec in recordings:
        first = rec.envelope.index * frontend.emg_hop
        last = first + frontend.emg_window
        excess = (rec.envelope.level_db - rec.resting_db)[:, columns].min(axis=1)
        ok = rec.envelope.valid
        for event in rec.events:
            inside = ok & (first >= event.start + react) & (last <= event.stop)
            if event.kind == CLENCH:
                clench.append(excess[inside])
            elif event.kind in ("baseline", "pause") or (event.kind == "cue" and event.label == REST):
                rest.append(excess[inside])
    if not clench or not rest or sum(map(len, clench)) < 8:
        return config, {}
    return fit_threshold(np.concatenate(rest), np.concatenate(clench), config)


@dataclass
class Trained:
    bundle: Bundle
    corpus: Corpus
    evaluation: Evaluation
    tuning: Tuning
    selftest: tuple[np.ndarray, np.ndarray]

    def save(self, directory: str | Path) -> Path:
        return save_bundle(directory, self.bundle, *self.selftest)


def train_bundle(
    sessions: list[Session],
    *,
    frontend: FrontendConfig | None = None,
    quality: QualityConfig | None = None,
    classes: tuple[str, ...] | None = None,
    candidates: list[Candidate] | None = None,
    select: bool = False,
    permutations: int = 200,
    target_false_per_min: float = 0.5,
    reaction_s: float = 1.0,
    seed: int = 0,
) -> Trained:
    if not sessions:
        raise ValueError("no sessions to train on")
    frontend = frontend or FrontendConfig()
    quality = quality or QualityConfig()
    recordings = [load_recording(session, frontend, quality, reaction_s=reaction_s) for session in sessions]
    corpus = Corpus(recordings, classes)
    candidates = candidates or default_candidates()
    evaluation = evaluate(corpus, candidates, select=select, permutations=permutations, seed=seed)
    fitted = fit_final(corpus, candidates, select=select)
    tuning = tune(
        recordings, fitted.held_out, corpus.dataset.classes, frontend, target_false_per_min=target_false_per_min
    )
    clench, clench_report = clench_threshold(recordings, frontend)

    data = corpus.dataset
    report = {
        "sessions": [rec.key for rec in recordings],
        "trials": data.counts(),
        "evaluation": evaluation.summary(),
        "features": fitted.candidate.name,
        "temperature": round(fitted.temperature, 4),
        "tuning": tuning.summary(),
        "clench": {k: round(v, 3) for k, v in clench_report.items()},
        "stream": {rec.key: rec.stats.as_dict() for rec in recordings},
    }
    bundle = Bundle(
        fitted.model, frontend, quality, clench, tuning.config, tuning.recommended_scheme, tuning.reliable, report
    )
    # A few real frames and their reference go into the bundle as its self-test.
    rows = np.linspace(0, len(data) - 1, num=min(8, len(data)), dtype=int)
    selftest = (data.cosp[rows], data.references[data.session[rows][0]])
    return Trained(bundle, corpus, evaluation, tuning, selftest)

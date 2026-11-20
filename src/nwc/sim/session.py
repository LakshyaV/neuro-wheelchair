"""Simulated calibration sessions, written in the same format as real ones.

The virtual wearer follows the calibration schedule the way a person does: with
a reaction time after each cue, a variable strength from trial to trial, and the
occasional trial where nothing much happens. Everything downstream of the files
(dataset building, training, evaluation, replay) is the code that runs on real
recordings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..calibration.protocol import CLENCH, ProtocolConfig, Step, build_schedule
from ..io.session import BASELINE, CUE, PAUSE, RUN, Session, SessionWriter
from ..io.stream import StreamInfo
from .eeg import EEGSynth, SimConfig
from .source import Transport


def render(
    steps: list[Step],
    synth: EEGSynth,
    rng: np.random.Generator,
    *,
    reaction_s: tuple[float, float] = (0.3, 0.7),
    lapse: float = 0.05,
) -> tuple[np.ndarray, list[tuple[Step, int, int]]]:
    """Device samples for a schedule and the sample span of every step."""
    fs = synth.config.fs
    parts: list[np.ndarray] = []
    spans: list[tuple[Step, int, int]] = []
    for step in steps:
        n = round(step.duration * fs)
        start = synth.position
        if n == 0:
            spans.append((step, start, start))
            continue
        react = min(round(rng.uniform(*reaction_s) * fs), n)
        if step.kind == "cue" and step.label != "rest":
            parts.append(synth.generate(react))  # still at rest while the cue is being read
            if rng.random() >= lapse:  # now and then the wearer misses a cue altogether
                synth.set_intent(step.label)
            parts.append(synth.generate(n - react))
            synth.set_intent("rest")
        elif step.kind == "clench":
            parts.append(synth.generate(react))
            synth.set_clench(True)
            parts.append(synth.generate(n - react))
            synth.set_clench(False)
        else:
            parts.append(synth.generate(n))
        spans.append((step, start, synth.position))
    return np.concatenate(parts), spans


def simulate_session(
    directory: str | Path,
    protocol: ProtocolConfig | None = None,
    sim: SimConfig | None = None,
    *,
    seed: int = 0,
    subject: str = "sim",
    session: str = "s01",
    loss: float = 0.0,
    counted: bool = False,
    jitter_ms: float = 8.0,
) -> Session:
    protocol = protocol or ProtocolConfig()
    sim = sim or SimConfig()
    synth = EEGSynth(sim, seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5C4ED]))
    samples, spans = render(build_schedule(protocol), synth, rng)

    transport = Transport(sim.fs, jitter_ms=jitter_ms, loss=loss, counted=counted, seed=seed)
    info = StreamInfo(sim.channels, sim.fs, source="sim", device="simulated Muse")
    meta = {
        "simulated": True,
        "protocol": {"commands": list(protocol.commands), "runs": protocol.runs, "cue_s": protocol.cue_s},
        "sim": {"effect": sim.effect, "seed": seed},
    }
    with SessionWriter(directory, info, subject=subject, session=session, meta=meta) as writer:
        delivered = []  # device index of every sample that arrived, in order
        for _, first, chunk in transport.push(samples):
            writer.write(chunk)
            delivered.append(first + np.arange(len(chunk)))
        rows_of_device = np.concatenate(delivered) if delivered else np.zeros(0, dtype=np.int64)

        def row(device_index: int) -> int:
            return int(np.searchsorted(rows_of_device, device_index))

        run_start: dict[int, int] = {}
        run_stop: dict[int, int] = {}
        for step, a, b in spans:
            kind = {"baseline": BASELINE, "cue": CUE, "pause": PAUSE, "clench": CLENCH}.get(step.kind)
            if step.run:
                run_start.setdefault(step.run, a)
                run_stop[step.run] = b
            if kind is None or a == b:
                continue
            writer.event(kind, row(a), row(b), step.label, step.run, step.trial)
        for run, a in run_start.items():
            writer.event(RUN, row(a), row(run_stop[run]), "", run)
    return Session(directory)

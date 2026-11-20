"""Describe a session or a model bundle.

nwc info data/s01/20261120-0930
nwc info runs/s01/bundle
"""

from __future__ import annotations

import argparse
from pathlib import Path

NAME = "info"
HELP = "describe a session or a bundle"
ORDER = 50


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", type=Path)


def run(args: argparse.Namespace) -> int:
    from ..decode.bundle import MANIFEST as BUNDLE
    from ..decode.bundle import BundleError, load_bundle
    from ..io.session import MANIFEST as SESSION
    from ..io.session import find_sessions
    from ..report import training_report

    path: Path = args.path
    if (path / BUNDLE).exists():
        try:
            bundle = load_bundle(path)
        except BundleError as error:
            print(f"error: {error}")
            return 1
        print(f"bundle {path}: self-test passed, front-end {bundle.frontend.fingerprint}")
        print(training_report(bundle))
        return 0
    sessions = find_sessions(path) if path.exists() else []
    if not sessions:
        print(f"error: {path} holds neither a bundle ({BUNDLE}) nor a session ({SESSION})")
        return 1
    for session in sessions:
        block, stats = session.block()
        cues = session.events("cue")
        counts = {label: sum(e.label == label for e in cues) for label in session.labels}
        runs = len(session.events("run"))
        extra = " (simulated)" if session.meta.get("simulated") else ""
        print(f"{session.key}{extra}: {len(block) / session.info.fs / 60:.1f} min, {', '.join(session.info.channels)}")
        print(f"  stream: {stats.delivered_hz:.1f} samples/s, {100 * stats.loss:.2f} % lost, {stats.breaks} breaks")
        print(
            f"  protocol: {runs} runs, cues {counts}, baseline {'yes' if session.events('baseline') else 'no'}, clench trials {len(session.events('clench'))}"
        )
    return 0

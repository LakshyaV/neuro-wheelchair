"""Test a trained bundle on sessions it was not trained on.

    nwc evaluate --bundle runs/s01/bundle --data data/s01/20261121-1000

This is the question that matters after the first day: does yesterday's model
still work today? Each session is expressed relative to its own baseline, the
model is applied unchanged, and its commands are replayed through the controller
with the thresholds stored in the bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

NAME = "evaluate"
HELP = "test a bundle on other sessions"
ORDER = 35


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="a session or a directory of sessions")
    parser.add_argument("--reaction-s", type=float, default=1.0)


def run(args: argparse.Namespace) -> int:
    import numpy as np

    from ..control.tune import score_replay
    from ..decode.bundle import BundleError, load_bundle
    from ..decode.dataset import build_dataset, load_recording
    from ..decode.train import balanced_accuracy, confusion, trial_decisions
    from ..io.session import find_sessions
    from ..report import confusion_table, recording_line

    try:
        bundle = load_bundle(args.bundle)
    except BundleError as error:
        print(f"error: {error}")
        return 1
    sessions = find_sessions(args.data)
    if not sessions:
        print(f"error: no sessions under {args.data}")
        return 1
    trained_on = set(bundle.report.get("sessions", []))
    model, classes = bundle.model, bundle.classes
    print(f"bundle: {bundle.describe()}")
    for session in sessions:
        try:
            rec = load_recording(session, bundle.frontend, bundle.quality, reaction_s=args.reaction_s)
            data = build_dataset([rec], classes)
        except ValueError as error:
            print(f"{session.key}: {error}")
            continue
        note = "  (used in training: this is not a test)" if rec.key in trained_on else ""
        print(f"\n{recording_line(rec, bundle.frontend.frame_rate)}{note}")
        log_proba = model.log_proba(data.cosp, rec.reference)
        _, truth, prediction = trial_decisions(log_proba, data.trial, data.y)
        accuracy = balanced_accuracy(truth, prediction, len(classes))
        print(
            f"  decoding: {100 * accuracy:.0f} % of {len(truth)} trials, where guessing gets {100 / len(classes):.0f} %"
        )
        for row in confusion_table(list(classes), confusion(truth, prediction, len(classes)).tolist()):
            print("  " + row)
        everything = np.full((len(rec), len(classes)), np.nan)
        inside = rec.run > 0
        everything[inside] = model.log_proba(rec.frames.cosp[inside], rec.reference)
        score = score_replay([rec], [everything], classes, bundle.arbiter, bundle.frontend)
        if score.trials:
            latency = f", {score.latency_s:.1f} s after the cue (median)" if score.latencies else ""
            print(
                f"  driving: commands switched on in {100 * score.hit_rate:.0f} % of trials{latency}; a wrong command in {100 * score.wrong_rate:.0f} %; {score.false_per_min:.2f} false starts per minute of rest"
            )
            print(
                "  per command: " + ", ".join(f"{k} {100 * v['hit_rate']:.0f} %" for k, v in score.per_command.items())
            )
    return 0

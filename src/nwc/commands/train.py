"""Train an intent model from calibration sessions and report what it can do.

    nwc train --data data/s01 --out runs/s01
    nwc train --data data/s01/20261120-0930 --out runs/today --select

Everything below --data is used. The report is the point of this command: how
often held-out trials are decoded and how that compares with shuffled labels,
then how the commands behave when held-out runs are replayed through the
controller, and from that which driving scheme the data supports. The model and
the thresholds found are written to <out>/bundle for `nwc drive`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

NAME = "train"
HELP = "train and evaluate an intent model"
ORDER = 30


def configure(parser: argparse.ArgumentParser) -> None:
    from .. import settings

    parser.add_argument("--data", type=Path, required=True, help="a session or a directory of sessions")
    parser.add_argument("--out", type=Path, required=True, help="where the bundle and report go")
    parser.add_argument(
        "--commands", default=None, help="train on these commands only (comma-separated; rest is always included)"
    )
    parser.add_argument(
        "--select", action="store_true", help="choose the feature configuration by nested cross-validation"
    )
    parser.add_argument("--permutations", type=int, default=200, help="label shuffles for the p-value (0 to skip)")
    parser.add_argument(
        "--false-starts", type=float, default=0.5, help="tolerated false starts per minute of rest (default: 0.5)"
    )
    parser.add_argument("--reaction-s", type=float, default=1.0, help="time after a cue that is not used (default: 1)")
    parser.add_argument("--seed", type=int, default=0)
    settings.add_arguments(parser)


def run(args: argparse.Namespace) -> int:

    from .. import settings
    from ..config import save_json
    from ..decode.pipeline import train_bundle
    from ..io.session import find_sessions
    from ..report import training_report

    cfg = settings.load(args)
    sessions = find_sessions(args.data)
    if not sessions:
        print(f"error: no sessions under {args.data}")
        return 1
    frontend = cfg.frontend
    classes = ("rest", *[c for c in args.commands.split(",") if c != "rest"]) if args.commands else None
    try:
        trained = train_bundle(
            sessions,
            frontend=frontend,
            quality=cfg.quality,
            classes=classes,
            select=args.select,
            permutations=args.permutations,
            target_false_per_min=args.false_starts,
            reaction_s=args.reaction_s,
            seed=args.seed,
        )
    except ValueError as error:
        print(f"error: {error}")
        return 1
    directory = trained.save(args.out / "bundle")
    save_json(args.out / "report.json", trained.bundle.report)
    print(training_report(trained.bundle, trained.corpus.recordings))
    print(f"bundle      {directory}")
    print(f"next: nwc drive --bundle {directory}")
    return 0

"""Simulate calibration sessions, for trying the system without a headset.

    nwc sim --out data --subject demo
    nwc sim --out data --subject demo --sessions 2 --effect 4

The files are written exactly as `nwc record` writes them. `--effect` scales how
strongly each command changes the simulated rhythms: at 0 there is nothing to
decode and evaluation must say so, at 4 the commands are easy. It is a setting of
the simulator, not a prediction about a person.
"""

from __future__ import annotations

import argparse
from pathlib import Path

NAME = "sim"
HELP = "simulate calibration sessions"
ORDER = 25


def configure(parser: argparse.ArgumentParser) -> None:
    from .. import settings

    parser.add_argument("--out", type=Path, default=Path("data"), help="where sessions are kept (default: data)")
    parser.add_argument("--subject", default="sim")
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--effect", type=float, default=None, help="strength of the simulated commands (default: 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--commands", default=None, help="comma-separated command names (default: forward,left,right)")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None, help="trials per command and run")
    parser.add_argument("--loss", type=float, default=0.0, help="share of Bluetooth packets lost")
    parser.add_argument("--counted", action="store_true", help="include the device's sample counter in the stream")
    settings.add_arguments(parser)


def run(args: argparse.Namespace) -> int:
    from .. import settings
    from ..calibration.protocol import describe
    from ..sim.session import simulate_session

    protocol = {
        "commands": args.commands.split(",") if args.commands else None,
        "runs": args.runs,
        "trials_per_command": args.trials,
    }
    cfg = settings.load(args, protocol=protocol, sim={"effect": args.effect})
    print(describe(cfg.protocol))
    for index in range(args.sessions):
        name = f"s{index + 1:02d}"
        directory = args.out / args.subject / name
        if directory.exists():
            print(f"error: {directory} exists already")
            return 1
        session = simulate_session(
            directory,
            cfg.protocol,
            cfg.sim,
            seed=args.seed + index,
            subject=args.subject,
            session=name,
            loss=args.loss,
            counted=args.counted,
        )
        print(f"  {directory}  ({session.duration / 60:.1f} min, effect {cfg.sim.effect:g})")
    print(f"next: nwc train --data {args.out / args.subject} --out runs/{args.subject}")
    return 0

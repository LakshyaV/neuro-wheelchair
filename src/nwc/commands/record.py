"""Record a calibration session: a baseline, a few jaw clenches and cued trials of every command.

    nwc record --subject s01
    nwc record --subject s01 --commands forward,left,right,backward --runs 4
    nwc record --subject s01 --source lsl

The wearer is told what to do for each cue. Any strategy can stand for a command
as long as it is the same one every time; docs/workflow.md discusses choices.
The recording is written to data/<subject>/<session>/ as it arrives.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

NAME = "record"
HELP = "record a guided calibration session"
ORDER = 20


def configure(parser: argparse.ArgumentParser) -> None:
    from .. import settings

    parser.add_argument("--subject", required=True, help="who is wearing the headset")
    parser.add_argument("--session", default=None, help="name of this session (default: date and time)")
    parser.add_argument(
        "--source", default="osc", help="osc[:[host:]port], lsl[:name], replay:<dir> or sim[:seed] (default: osc)"
    )
    parser.add_argument("--address", default=None, help="OSC address, if the app uses one of its own")
    parser.add_argument("--out", type=Path, default=Path("data"), help="where sessions are kept (default: data)")
    parser.add_argument("--commands", default=None, help="comma-separated command names (default: forward,left,right)")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None, help="trials per command and run")
    parser.add_argument("--yes", action="store_true", help="do not wait for Enter between runs")
    settings.add_arguments(parser)


def run(args: argparse.Namespace) -> int:
    from .. import settings
    from ..calibration.protocol import describe
    from ..calibration.recorder import ConsoleUI, Recorder, RecordingError
    from ..io.session import SessionWriter
    from ..io.sources import SourceError, open_source

    protocol = {
        "commands": args.commands.split(",") if args.commands else None,
        "runs": args.runs,
        "trials_per_command": args.trials,
    }
    cfg = settings.load(args, protocol=protocol)
    front = cfg.frontend
    session = args.session or time.strftime("%Y%m%d-%H%M%S")
    directory = args.out / args.subject / session
    if directory.exists():
        print(f"error: {directory} exists already; choose another --session")
        return 1
    options = {"address": args.address} if args.address else {}
    live = not args.source.startswith(("replay", "sim"))
    try:
        source = open_source(args.source, channels=front.channels, fs=front.fs, realtime=False, **options)
        source.open()
    except SourceError as error:
        print(f"error: {error}")
        return 1
    print(describe(cfg.protocol))
    ui = ConsoleUI(interactive=live and not args.yes)
    meta = {
        "protocol": {"commands": list(cfg.protocol.commands), "runs": cfg.protocol.runs, "cue_s": cfg.protocol.cue_s}
    }
    try:
        with SessionWriter(directory, source.info, subject=args.subject, session=session, meta=meta) as writer:
            summary = Recorder(source, writer, cfg.protocol, ui, frontend=front, quality=cfg.quality).run()
    except RecordingError as error:
        print(f"error: {error}")
        print(f"what was recorded so far is in {directory}")
        return 1
    except KeyboardInterrupt:
        print(f"\ninterrupted: what was recorded so far is in {directory}")
        return 130
    finally:
        source.close()
    if summary.lost_channels:
        print(
            f"warning: {', '.join(summary.lost_channels)} out of range for most of the recording; check with `nwc monitor` and record again"
        )
    print(f"next: nwc train --data {args.out / args.subject} --out runs/{args.subject}")
    return 0

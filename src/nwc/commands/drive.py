"""Drive.

    nwc drive --link /dev/cu.usbmodem1101                       switch scheme, no training needed
    nwc drive --bundle runs/s01/bundle --link /dev/cu.usbmodem1101
    nwc drive --bundle runs/s01/bundle --link none              dry run: decode and decide, move nothing
    nwc drive --source sim --link sim                           everything simulated

After a baseline (the wearer sits relaxed, eyes open) the operator arms the
system with `a`. Space disarms, `b` repeats the baseline, `q` quits. Without a
bundle the switch scheme is used: directions are offered in turn, a jaw clench
takes the one on offer, and a clench stops the chair. With a bundle the scheme it
recommends is used unless --scheme says otherwise.

Keep the drive wheels off the ground until the setup has been checked, and keep
a hand on the power switch of the motors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

NAME = "drive"
HELP = "drive the chair from the headset"
ORDER = 40


def configure(parser: argparse.ArgumentParser) -> None:
    from .. import settings

    parser.add_argument("--bundle", type=Path, default=None, help="trained model (from `nwc train`)")
    parser.add_argument(
        "--source", default="osc", help="osc[:[host:]port], lsl[:name], replay:<dir> or sim[:seed] (default: osc)"
    )
    parser.add_argument("--address", default=None, help="OSC address, if the app uses one of its own")
    parser.add_argument(
        "--link", default="none", help="serial port of the motor controller, `sim` or `none` (default: none, a dry run)"
    )
    parser.add_argument(
        "--scheme", default=None, help="direct, switch or switch:<command> (default: what the bundle recommends)"
    )
    parser.add_argument("--seconds", type=float, default=None, help="stop after this long")
    parser.add_argument(
        "--arm", action="store_true", help="arm as soon as the baseline is complete (simulation and bench tests)"
    )
    settings.add_arguments(parser)


def run(args: argparse.Namespace) -> int:
    import time

    from .. import settings
    from ..decode.bundle import BundleError, load_bundle
    from ..io.sources import SourceError, open_source
    from ..link.motor import LinkError, open_link
    from ..runtime.engine import Engine
    from ..runtime.live import Keys, StatusLine, drive, stream_clock

    cfg = settings.load(args, engine={"scheme": args.scheme})
    bundle = None
    if args.bundle is not None:
        try:
            bundle = load_bundle(args.bundle)
        except BundleError as error:
            print(f"error: {error}")
            return 1
        print(f"model: {bundle.describe()}")
    front = bundle.frontend if bundle else cfg.frontend
    live = not args.source.startswith(("replay", "sim"))
    options = {"address": args.address} if args.address else {}
    try:
        source = open_source(
            args.source,
            channels=front.channels,
            fs=front.fs,
            realtime=live or args.link not in ("sim", "none"),
            duration_s=args.seconds,
            **options,
        )
        source.open()
        link = open_link(args.link)
        link.open()
    except (SourceError, LinkError) as error:
        print(f"error: {error}")
        return 1
    try:
        engine = Engine(source.info, link, bundle, cfg.engine, None if bundle else front)
    except ValueError as error:
        print(f"error: {error}")
        source.close()
        link.close()
        return 1
    simulated = not live and args.link in ("sim", "none")
    print(f"scheme: {engine.scheme.name} | link: {args.link} | keys: a arm, space disarm, b baseline, q quit")
    try:
        with Keys() as keys:
            status = drive(
                engine,
                source,
                clock=stream_clock(engine) if simulated else time.monotonic,
                keys=keys,
                show=StatusLine(),
                duration_s=args.seconds,
                auto_arm=args.arm,
            )
    except LinkError as error:
        print(f"\nerror: {error}")
        return 1
    finally:
        source.close()
        link.close()
    print(f"\nstopped. {status.line()}")
    return 0

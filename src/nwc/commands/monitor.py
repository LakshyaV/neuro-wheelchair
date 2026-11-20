"""Check the headset: is the stream complete, and does every electrode have contact?

    nwc monitor                          Petal Metrics over OSC, the default
    nwc monitor --source lsl             muselsl, BlueMuse or Petal Metrics over LSL
    nwc monitor --source replay:data/s01/day1

Run this before every recording and before driving. A good stream delivers close
to the nominal 256 samples per second with nothing lost, and every channel sits
at a few microvolts with eyes open and a relaxed face.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

NAME = "monitor"
HELP = "check stream completeness and electrode contact"
ORDER = 10


def configure(parser: argparse.ArgumentParser) -> None:
    from .. import settings

    parser.add_argument(
        "--source", default="osc", help="osc[:[host:]port], lsl[:name], replay:<dir> or sim[:seed] (default: osc)"
    )
    parser.add_argument("--address", default=None, help="OSC address, if the app uses one of its own")
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to watch (default: 30)")
    settings.add_arguments(parser)


def run(args: argparse.Namespace) -> int:
    import numpy as np

    from .. import settings
    from ..dsp.frontend import Frontend
    from ..dsp.quality import ContactMonitor, QualityMeter
    from ..io.sources import SourceError, open_source
    from ..io.stream import StreamAssembler

    cfg = settings.load(args)
    front = cfg.frontend
    options = {"address": args.address} if args.address else {}
    live = not args.source.startswith(("replay", "sim"))
    try:
        source = open_source(
            args.source, channels=front.channels, fs=front.fs, realtime=False, duration_s=args.seconds, **options
        )
        source.open()
    except SourceError as error:
        print(f"error: {error}")
        return 1
    assembler, frontend = StreamAssembler(source.info), Frontend(front)
    meter, contact = QualityMeter(front, cfg.quality), ContactMonitor(front, cfg.quality)
    print(f"source {source.info.source}: {', '.join(source.info.channels)} at {source.info.fs:g} samples/s nominal")
    print(f"{'time':>6}  {'rate':>7}  {'lost':>5}   " + "".join(f"{c:>9}{'':5}" for c in front.channels) + "  usable")
    started = time.monotonic()
    next_line, rms, last_arrival = 1.0, [], time.monotonic()
    # the share of usable frames is taken over the last five seconds, from the moment the filters have settled
    recent: deque[bool] = deque(maxlen=round(5.0 * front.fs / front.hop))
    usable = total = 0
    try:
        while not source.exhausted:
            chunk = source.read()
            if chunk is not None and len(chunk):
                last_arrival = time.monotonic()
            elif live and time.monotonic() - last_arrival > 5.0:
                print("no samples for 5 s: is the headset on and the stream running?")
                return 1
            frames, _ = frontend.push(assembler.push(chunk))
            if len(frames):
                quality = meter(frames)
                contact.push(quality, frames.valid)
                rms.append(quality.rms[frames.valid])
                settled = bool(recent) or bool(frames.valid.any())
                flags = quality.usable[int(np.argmax(frames.valid)) :] if settled and not recent else quality.usable
                if settled:
                    recent.extend(bool(flag) for flag in flags)
                    usable += int(flags.sum())
                    total += len(flags)
            seconds = assembler.count / front.fs
            if seconds >= next_line and rms and sum(map(len, rms)):
                level = np.median(np.concatenate(rms), axis=0)
                cells = "".join(f"{v:6.1f} uV{' LOST' if lost else '':5}" for v, lost in zip(level, contact.lost))
                stats = assembler.stats
                share = 100 * sum(recent) / max(len(recent), 1)
                print(f"{seconds:6.0f}  {stats.delivered_hz:7.1f}  {100 * stats.loss:4.1f}%   {cells}  {share:5.0f}%")
                rms, next_line = [], max(next_line, float(int(seconds))) + 1.0
            if live and time.monotonic() - started >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
    stats = assembler.stats
    print(
        f"stream: {stats.received} samples, {stats.delivered_hz:.1f} per second, {100 * stats.loss:.2f} % lost, {stats.breaks} breaks"
        + ("" if stats.counted else " (no device counter: loss is estimated from the rate)")
    )
    if total:
        share = 100 * usable / total
        print(
            f"frames: {share:.0f} % usable"
            + ("" if share >= 99.5 else " (the others held an artefact, such as a blink)")
        )
    lost = contact.lost_channels
    print(
        "contact: "
        + (
            f"lost on {', '.join(lost)}: wet the electrodes and re-seat the headband"
            if lost
            else "all channels in range"
        )
    )
    return 1 if lost or stats.received == 0 else 0

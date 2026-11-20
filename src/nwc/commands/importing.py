"""Import recordings made with other tools.

    nwc import markers --markers 1=left,2=right --cue-s 3 --subject s01 --session old run1.csv run2.csv
    nwc import labels  --names 0=rest,1=forward,3=left,4=right --subject s01 --session old brain_data*.csv

`markers`: muselsl or EEG-ExPy recordings with a Marker0 column; each marker
starts a cue of --cue-s seconds. `labels`: files with one label per sample in the
last column, which is how the 2024 prototype kept its training data. Each file
becomes one run of the session.
"""

from __future__ import annotations

import argparse
from pathlib import Path

NAME = "import"
HELP = "import recordings made with other tools"
ORDER = 28


def _mapping(text: str | None) -> dict[str, str]:
    out = {}
    for item in (text or "").split(","):
        if item.strip():
            key, _, value = item.partition("=")
            if not value:
                raise SystemExit(f"expected key=value, got {item!r}")
            out[key.strip()] = value.strip()
    return out


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("layout", choices=("markers", "labels"))
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--markers", default=None, help="markers: code=name,code=name")
    parser.add_argument("--cue-s", type=float, default=3.0, help="markers: length of each cue in seconds")
    parser.add_argument("--names", default=None, help="labels: value=name,value=name")
    parser.add_argument("--fs", type=float, default=256.0)


def run(args: argparse.Namespace) -> int:
    from ..io.importers import import_labelled_recordings, import_marker_recordings

    directory = args.out / args.subject / args.session
    if directory.exists():
        print(f"error: {directory} exists already")
        return 1
    missing = [str(p) for p in args.files if not p.exists()]
    if missing:
        print(f"error: not found: {', '.join(missing)}")
        return 1
    common = {"subject": args.subject, "session": args.session, "fs": args.fs}
    try:
        if args.layout == "markers":
            markers = {int(k): v for k, v in _mapping(args.markers).items()}
            if not markers:
                print("error: --markers is needed, e.g. --markers 1=left,2=right")
                return 1
            session = import_marker_recordings(args.files, directory, markers=markers, cue_s=args.cue_s, **common)
        else:
            session = import_labelled_recordings(args.files, directory, names=_mapping(args.names), **common)
    except ValueError as error:
        print(f"error: {error}")
        return 1
    cues = session.events("cue")
    counts = {label: sum(e.label == label for e in cues) for label in session.labels}
    print(f"{directory}: {len(args.files)} runs, {session.duration / 60:.1f} min, cues {counts}")
    print(f"next: nwc train --data {directory} --out runs/{args.subject}")
    return 0

"""What goes wrong when every mental state gets a recording of its own.

Data: the Muse recordings behind "A Study on Mental State Classification using
EEG-based Brain-Machine Interface" (Bird et al., 2018), published with the
authors' feature-generation code. Four people, three states (relaxed, neutral,
concentrating), two one-minute recordings per state, one state per file.

    python tools/evidence/one_recording_per_state.py --download
    python tools/evidence/one_recording_per_state.py

This is also how the 2024 prototype of this project collected its training
data: one file per command. The script shows when each file was recorded, how
the electrodes sat in it, and what a classifier makes of that.
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nwc.decode.dataset import load_recording
from nwc.decode.train import Corpus, evaluate
from nwc.dsp.frontend import FrontendConfig, process
from nwc.dsp.quality import QualityMeter
from nwc.io.importers import import_labelled_recordings
from nwc.io.stream import StreamInfo, assemble

SOURCE = "https://raw.githubusercontent.com/jordan-bird/eeg-feature-generation/master/dataset/original_data"
STATES = ("relaxed", "neutral", "concentrating")
PEOPLE = "abcd"


def download(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for person in PEOPLE:
        for state in STATES:
            for take in (1, 2):
                name = f"subject{person}-{state}-{take}.csv"
                if not (root / name).exists():
                    urllib.request.urlretrieve(f"{SOURCE}/{name}", root / name)
                    print(f"  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("evidence-data/mental-state"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download(args.data)
    cfg = FrontendConfig()
    info = StreamInfo(cfg.channels, cfg.fs)
    meter = QualityMeter(cfg)

    for person in PEOPLE:
        files = []
        print(f"\nperson {person}")
        print(
            f"  {'state':15s} {'recorded (UTC)':18s} {'length':>7s}   median amplitude 1-45 Hz: "
            + "  ".join(f"{c:>5s}" for c in cfg.channels)
        )
        for state in STATES:
            for take in (1, 2):
                path = args.data / f"subject{person}-{state}-{take}.csv"
                if not path.exists():
                    raise SystemExit(f"{path} is missing: run with --download")
                data = np.loadtxt(path, delimiter=",", skiprows=1)
                if len(data) < 20 * cfg.fs:
                    continue  # two of the files are a few seconds long
                frames, _ = process(assemble(data[:, 0], data[:, 1:5], info)[0], cfg)
                level = np.median(meter(frames).rms[frames.valid], axis=0)
                when = datetime.fromtimestamp(data[0, 0], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                print(
                    f"  {state:15s} {when:18s} {len(data) / cfg.fs:6.0f} s   {' ' * 26}"
                    + "  ".join(f"{v:5.1f}" for v in level)
                )
                files.append((state, data))

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, (state, data) in enumerate(files):
                path = Path(tmp) / f"{index}.csv"
                np.savetxt(
                    path,
                    np.column_stack((data[:, :5], np.full(len(data), STATES.index(state)))),
                    delimiter=",",
                    fmt="%.3f",
                )
                paths.append(path)
            names = {str(i): state for i, state in enumerate(STATES)}
            session = import_labelled_recordings(
                paths, Path(tmp) / "session", subject=person, session="all", names=names
            )
            recording = load_recording(session, cfg)
        # Everything is kept here, including the frames the quality check would refuse: the point is
        # what a classifier does when nothing stops it.
        corpus = Corpus([recording], STATES, clean_only=False)
        result = evaluate(corpus, permutations=0)
        refused = {state: 1.0 - float(recording.quality.usable[recording.label == state].mean()) for state in STATES}
        print(
            f"  classifier: {100 * result.accuracy:.0f} % of five-second pieces assigned to the right state (three states: guessing gets 33 %)"
        )
        print(f"  flagged by the evaluation as one state per recording: {result.confounded}")
        print(
            "  frames the quality check refuses: "
            + ", ".join(f"{state} {100 * share:.0f} %" for state, share in refused.items())
        )


if __name__ == "__main__":
    main()

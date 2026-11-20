"""What real Muse recordings look like at rest: the numbers behind the thresholds.

Runs the front-end, the quality check and the clench detector of this package
over every recording of the two public Muse data sets used elsewhere in this
directory (fetch them first with the other scripts' --download). Nobody was
asked to clench or to move in any of them.

    python tools/evidence/resting_statistics.py

It reports how packets are stamped, how large a channel in contact is and how
large one that has lifted off, how many analysis windows the quality check
refuses, how strong the power line is and what it would do to the stop reflex
if it were left in, and how often the reflex would have fired by itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import signal

from nwc.dsp.frontend import NOTCH_WIDTH_HZ, Envelope, FrontendConfig, process
from nwc.dsp.quality import QualityConfig, QualityMeter
from nwc.dsp.reflex import ClenchConfig, ClenchDetector, resting_level
from nwc.dsp.spectral import mains_harmonics
from nwc.io.stream import StreamInfo, assemble


def power_line_db(x: np.ndarray, fs: float, hz: float) -> float:
    """How far the line at ``hz`` stands above the spectrum around it, on the channel where it is strongest."""
    frequency, power = signal.welch(x - x.mean(axis=0), fs, nperseg=int(4 * fs), axis=0)
    near = np.abs(frequency - hz) <= 0.5
    around = (np.abs(frequency - hz) > 2.0) & (np.abs(frequency - hz) <= 5.0)
    return float((10 * np.log10(power[near].max(axis=0) / np.median(power[around], axis=0))).max())


def left_in_db(x: np.ndarray, cfg: FrontendConfig, hz: float) -> float:
    """Resting muscle-band level behind the ears with the line at ``hz`` left in, against the front-end's own."""
    band = signal.butter(4, cfg.emg_band_hz, "bandpass", fs=cfg.fs, output="sos")
    notches = [
        signal.tf2sos(*signal.iirnotch(f, f / NOTCH_WIDTH_HZ, fs=cfg.fs))
        for f in mains_harmonics(cfg.emg_band_hz[0] - 5.0, cfg.fs / 2 - 2.0)
    ]
    kept = [n for n, f in zip(notches, mains_harmonics(cfg.emg_band_hz[0] - 5.0, cfg.fs / 2 - 2.0)) if f % hz]
    ears = x[:, [0, 3]] - x[:, [0, 3]].mean(axis=0)
    both = signal.sosfilt(np.vstack([band, *notches]), ears, axis=0)[int(2 * cfg.fs) :]
    other_only = signal.sosfilt(np.vstack([band, *kept]), ears, axis=0)[int(2 * cfg.fs) :]
    return float((10 * np.log10(np.median(other_only**2, axis=0) / np.median(both**2, axis=0))).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data", type=Path, nargs="+", default=[Path("evidence-data/muse-ssvep"), Path("evidence-data/mental-state")]
    )
    args = parser.parse_args()
    cfg, limits = FrontendConfig(), QualityConfig()
    info, meter = StreamInfo(cfg.channels, cfg.fs), QualityMeter(cfg)
    paths = [p for root in args.data for p in sorted(root.rglob("*.csv"))]
    if not paths:
        raise SystemExit(
            f"no recordings under {', '.join(map(str, args.data))}: fetch them with the other scripts' --download"
        )

    steps, rates, levels, lifted, refused, minutes, fired = [], [], [], [], [], 0.0, {12.0: 0, 15.0: 0, 18.0: 0}
    lines, raised = {50.0: [], 60.0: []}, []
    for path in paths:
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        if len(data) < 30 * cfg.fs:
            continue
        t, x = data[:, 0], data[:, 1:5]
        line = {hz: power_line_db(x, cfg.fs, hz) for hz in lines}
        present = max(line, key=line.get)
        lines[present].append(line[present])
        raised.append(left_in_db(x, cfg, present))
        block, stats = assemble(t, x, info)
        steps.append(np.diff(t) * 1e3)
        rates.append(stats.delivered_hz)
        frames, fast = process(block, cfg)
        quality = meter(frames)
        level = np.median(quality.rms[frames.valid], axis=0)
        (lifted if level.max() > limits.rms_max_uv else levels).append(level)
        if level.max() > limits.rms_max_uv:
            continue  # an electrode without contact: not a resting recording of the head
        refused.append(1.0 - float(quality.usable[frames.valid].mean()))
        # the first twenty seconds stand in for the baseline, the rest is watched
        first = fast.index * cfg.emg_hop + cfg.emg_window <= 20 * cfg.fs
        rest = resting_level(Envelope(fast.index[first], fast.power[first], fast.valid[first]))
        watched = Envelope(fast.index[~first], fast.power[~first], fast.valid[~first])
        minutes += len(watched) * cfg.emg_hop / cfg.fs / 60.0
        for threshold in fired:
            detector = ClenchDetector(cfg.channels, rest, ClenchConfig(on_db=threshold, off_db=0.55 * threshold))
            fired[threshold] += sum(event.pressed for event in detector.push(watched))

    period = 1e3 / cfg.fs
    irregular = [d for d in steps if (np.rint(d[np.abs(d) < 250] / period) != 1).mean() > 0.005]
    stalled = sum((d >= 250).any() for d in steps)
    print(f"{len(rates)} recordings of at least 30 s")
    print("\npacket timing")
    if irregular:
        jumps = np.concatenate([d[np.abs(d) < 250] for d in irregular])
        share = float((np.rint(jumps / period) != 1).mean())
        print(
            f"  {len(irregular)} recordings carry arrival-time stamps: consecutive stamps differ by {jumps.min():.0f} to {jumps.max():.0f} ms "
            f"(one sample is {period:.1f} ms), and {100 * share * 12:.0f} % of packet boundaries are off by a sample or more"
        )
    print(
        f"  the other {len(steps) - len(irregular)} carry stamps regularised by the stream bridge; {stalled} contain pauses of a quarter second or more"
    )
    print(
        f"  samples delivered per second while the stream was flowing, counted: {np.min(rates):.1f} to {np.max(rates):.1f}"
    )
    levels, lifted = np.array(levels), np.array(lifted)
    print(
        f"\namplitude between 1 and 45 Hz, median over each recording (uV), {len(levels)} recordings with all electrodes in contact"
    )
    for c, name in enumerate(cfg.channels):
        print(
            f"  {name:5s} median {np.median(levels[:, c]):4.1f}   range {levels[:, c].min():4.1f} to {levels[:, c].max():4.1f}"
        )
    if len(lifted):
        print(
            f"  {len(lifted)} recordings with an electrode above the {limits.rms_max_uv:g} uV limit: its level was {lifted.max(axis=1).min():.0f} to {lifted.max(axis=1).max():.0f} uV, the others {np.median(np.sort(lifted, axis=1)[:, :3]):.1f} uV"
        )
    print(
        f"\nanalysis windows refused by the quality check: {100 * np.min(refused):.0f} to {100 * np.max(refused):.0f} % per recording, {100 * np.mean(refused):.0f} % on average"
    )
    print("\npower line, strongest channel, against the spectrum 2 to 5 Hz to either side")
    for hz, found in lines.items():
        if found:
            print(
                f"  {hz:g} Hz in {len(found)} recordings: {np.min(found):.0f} to {np.max(found):.0f} dB (median {np.median(found):.0f})"
            )
    print(
        f"  left in the muscle band, the line would raise the resting level behind the ears by {np.median(raised):.1f} dB (median), "
        f"up to {np.max(raised):.1f} dB"
    )
    print(f"\nstop reflex on {minutes:.0f} minutes in which nobody was asked to clench")
    for threshold, count in fired.items():
        per_minute = count / minutes
        print(
            f"  threshold {threshold:4.0f} dB above rest: fired {count} times, once in {1 / per_minute:.0f} minutes"
            if count
            else f"  threshold {threshold:4.0f} dB above rest: never fired"
        )


if __name__ == "__main__":
    main()

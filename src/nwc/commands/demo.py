"""See the whole system work without any hardware.

    nwc demo
    nwc demo --scheme switch
    nwc demo --effect 2

A calibration session is simulated and a model is trained on it. A virtual
wearer then drives a virtual chair along a short script while the real decoder,
supervisor and motor-controller logic do their work. What was asked for and what
the chair did are printed side by side.

--effect sets how clearly the simulated commands show in the EEG. It is a
setting of the simulator, not a prediction about a person: the report that
`nwc train` prints on your own calibration data is what tells you that.
"""

from __future__ import annotations

import argparse

NAME = "demo"
HELP = "simulated calibration, training and drive"
ORDER = 5


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scheme", choices=("direct", "switch"), default="direct")
    parser.add_argument(
        "--effect", type=float, default=6.0, help="strength of the simulated commands (default: 6, clearly separable)"
    )
    parser.add_argument("--seed", type=int, default=1)


def run(args: argparse.Namespace) -> int:
    import tempfile
    from dataclasses import replace
    from pathlib import Path

    from ..calibration.protocol import ProtocolConfig, describe
    from ..decode.pipeline import train_bundle
    from ..report import training_report
    from ..runtime.engine import EngineConfig
    from ..sim.eeg import SimConfig
    from ..sim.scenario import Act, simulate_drive
    from ..sim.session import simulate_session

    sim = SimConfig(effect=args.effect)
    bundle = None
    if args.scheme == "direct":
        protocol = ProtocolConfig()
        print(f"1. calibration (simulated wearer, effect {args.effect:g}): {describe(protocol)}")
        with tempfile.TemporaryDirectory() as tmp:
            session = simulate_session(Path(tmp) / "demo" / "s01", protocol, sim, seed=args.seed, subject="demo")
            trained = train_bundle([session], permutations=100)
        bundle = trained.bundle
        print("\n2. training")
        print(training_report(bundle, trained.corpus.recordings))
        if bundle.recommended_scheme != "direct":
            print(
                "\nThe calibration does not support direct driving. The drive below uses it anyway, to show what that looks like."
            )
        plan = [
            Act(24.0, "arm"),
            Act(30.0, "intent", "forward"), Act(38.0, "intent", "rest"),
            Act(44.0, "intent", "left"), Act(50.0, "intent", "rest"),
            Act(56.0, "intent", "forward"), Act(61.0, "clench", True), Act(61.6, "clench", False), Act(62.0, "intent", "rest"),
        ]  # fmt: skip
        asked = [
            "30 s  think forward",
            "38 s  let go",
            "44 s  think left",
            "50 s  let go",
            "56 s  think forward",
            "61 s  clench the jaw (emergency stop)",
        ]
        until = 68.0
    else:
        print("1. no calibration and no training: the switch scheme needs neither")
        plan = [
            Act(24.0, "arm"),
            Act(26.5, "clench", True), Act(27.0, "clench", False),  # forward is on offer first
            Act(31.0, "clench", True), Act(31.5, "clench", False),  # stop
            Act(35.5, "clench", True), Act(36.0, "clench", False),  # left is on offer by now; the turn ends by itself
            Act(39.5, "clench", True), Act(40.0, "clench", False),  # forward is on offer again
            Act(43.0, "clench", True), Act(43.5, "clench", False),  # stop
        ]  # fmt: skip
        asked = [
            "26.5 s  clench while forward is on offer",
            "31 s    clench: stop",
            "35.5 s  clench while left is on offer (a turn ends by itself)",
            "39.5 s  clench while forward is on offer",
            "43 s    clench: stop",
        ]
        until = 48.0

    config = replace(EngineConfig(), scheme=args.scheme, baseline_s=20.0)
    log = simulate_drive(
        plan, until_s=until, bundle=bundle, sim=sim, seed=args.seed + 1, session_seed=args.seed, config=config
    )
    print(
        f"\n{'3' if bundle else '2'}. drive ({args.scheme} scheme, virtual chair). The operator arms at 24 s, after the baseline."
    )
    print("   the wearer:")
    for line in asked:
        print(f"      {line}")
    print("   the chair:")
    for t in log.transitions:
        if t.state in ("armed", "halted", "fault"):
            print(f"      {t.at:5.1f} s  {t.state:7s} {t.motion:8s} {t.note}")
    chair = log.chair
    print(f"   it ended at {chair.describe()}, after {chair.distance:.1f} m, standing still: {not chair.moving}")
    return 0

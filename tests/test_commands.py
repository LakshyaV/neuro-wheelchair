"""The workflows of the README, through the command line."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nwc import __version__, settings
from nwc.cli import build_parser, main
from nwc.config import ConfigError, load_yaml, save_yaml
from nwc.sim.eeg import EEGSynth

FAST = ["--runs", "2", "--trials", "4", "-s", "protocol.baseline_s=20", "-s", "protocol.clench_trials=4"]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Simulated calibrations of one wearer on two days, and a model trained on the first."""
    root = tmp_path_factory.mktemp("work")
    data = root / "data"
    assert (
        main(["sim", "--out", str(data), "--subject", "demo", "--sessions", "2", "--effect", "6", "--seed", "3", *FAST])
        == 0
    )
    assert (
        main(
            [
                "train",
                "--data",
                str(data / "demo" / "s01"),
                "--out",
                str(root / "runs" / "demo"),
                "--permutations",
                "30",
            ]
        )
        == 0
    )
    return root


def test_help_lists_every_command(capsys):
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices
    assert list(commands) == [
        "demo",
        "monitor",
        "record",
        "sim",
        "import",
        "train",
        "evaluate",
        "drive",
        "link",
        "info",
    ]
    assert main([]) == 0 and "check stream completeness" in capsys.readouterr().out
    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0 and __version__ in capsys.readouterr().out
    for name in commands:
        with pytest.raises(SystemExit):
            main([name, "--help"])
        assert "usage: nwc" in capsys.readouterr().out


def test_simulate_train_and_inspect(workspace, capsys):
    capsys.readouterr()
    bundle = workspace / "runs" / "demo" / "bundle"
    assert (bundle / "bundle.json").exists() and (bundle / "model.npz").exists()
    report = json.loads((workspace / "runs" / "demo" / "report.json").read_text())
    assert report["evaluation"]["balanced_accuracy"] > 0.7 and report["sessions"] == ["demo/s01"]
    assert main(["info", str(bundle)]) == 0
    out = capsys.readouterr().out
    assert "self-test passed" in out and "decoding" in out and "driving" in out and "stop reflex threshold" in out
    assert main(["info", str(workspace / "data" / "demo")]) == 0
    out = capsys.readouterr().out
    assert "demo/s01 (simulated)" in out and "demo/s02" in out and "clench trials 4" in out
    assert main(["info", str(workspace)]) == 0  # finds the sessions below
    assert main(["info", str(workspace / "nothing")]) == 1
    assert main(["sim", "--out", str(workspace / "data"), "--subject", "demo"]) == 1  # refuses to overwrite a session
    assert main(["train", "--data", str(workspace / "runs"), "--out", str(workspace / "x")]) == 1  # no sessions there
    capsys.readouterr()


def test_evaluate_a_model_on_another_day(workspace, capsys):
    capsys.readouterr()
    bundle = str(workspace / "runs" / "demo" / "bundle")
    assert main(["evaluate", "--bundle", bundle, "--data", str(workspace / "data" / "demo")]) == 0
    out = capsys.readouterr().out
    first, second = out.split("demo/s02")
    assert "used in training: this is not a test" in first and "used in training" not in second
    assert "decoding:" in second and "driving:" in second and "per command:" in second
    assert main(["evaluate", "--bundle", str(workspace), "--data", str(workspace / "data")]) == 1


def test_drive_in_simulation(workspace, capsys):
    capsys.readouterr()
    bundle = str(workspace / "runs" / "demo" / "bundle")
    assert (
        main(
            [
                "drive",
                "--bundle",
                bundle,
                "--source",
                "sim:5",
                "--link",
                "sim",
                "--seconds",
                "30",
                "--arm",
                "-s",
                "engine.baseline_s=12",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "model: classes rest" in out and "armed" in out and "stopped." in out
    assert (
        main(["drive", "--source", "sim:5", "--link", "none", "--seconds", "24", "--arm", "-s", "engine.baseline_s=12"])
        == 0
    )
    out = capsys.readouterr().out
    assert "scheme: switch" in out and "armed" in out
    assert (
        main(["drive", "--source", "sim", "--scheme", "direct", "--seconds", "1"]) == 1
    )  # direct driving needs a model
    assert "needs a trained model" in capsys.readouterr().out
    assert main(["drive", "--bundle", str(workspace), "--source", "sim"]) == 1


def test_demo_runs_both_schemes(capsys):
    assert main(["demo", "--scheme", "switch"]) == 0
    out = capsys.readouterr().out
    assert "forward: press to stop" in out and "standing still: True" in out
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert (
        "1. calibration" in out
        and "2. training" in out
        and "3. drive" in out
        and "emergency stop" in out
        and "standing still: True" in out
    )


def test_monitor_and_controller_check(capsys):
    assert main(["monitor", "--source", "sim:1", "--seconds", "8"]) == 0
    out = capsys.readouterr().out
    assert "all channels in range" in out and "samples" in out
    assert main(["link", "--port", "sim"]) == 0
    out = capsys.readouterr().out
    assert "protocol 1" in out and "watchdog: reported after the commands stopped" in out


def test_record_from_a_replayed_session(workspace, capsys):
    capsys.readouterr()
    source = f"replay:{workspace / 'data' / 'demo' / 's01'}"
    out = workspace / "recorded"
    # a shorter protocol than the one the replayed session was made with, so the stream is long enough
    shorter = ["--runs", "2", "--trials", "3", "-s", "protocol.baseline_s=20", "-s", "protocol.clench_trials=4"]
    args = [
        "record",
        "--subject",
        "copy",
        "--session",
        "again",
        "--source",
        source,
        "--out",
        str(out),
        "--yes",
        *shorter,
    ]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "done: 24 trials" in text and "next: nwc train" in text
    assert (out / "copy" / "again" / "events.jsonl").exists()
    assert main(args) == 1 and "exists already" in capsys.readouterr().out  # never over an existing recording
    # the replayed stream is too short for the full default protocol: the command says so and keeps what arrived
    assert main([*args[:4], "longer", *args[5:9], "--yes"]) == 1
    assert "the stream ended before the recording was complete" in capsys.readouterr().out


def test_import_recordings_from_other_tools(tmp_path, capsys):
    paths = []
    for run in range(2):
        x = EEGSynth(seed=run).generate(256 * 40)
        marker = np.zeros(len(x))
        marker[256 * 2 :: 256 * 5] = np.tile([1, 2], 4)
        path = tmp_path / f"run{run}.csv"
        np.savetxt(
            path,
            np.column_stack((np.arange(len(x)) / 256.0, x, marker)),
            delimiter=",",
            fmt="%.3f",
            header="timestamps,TP9,AF7,AF8,TP10,Marker0",
            comments="",
        )
        paths.append(str(path))
    out = tmp_path / "data"
    assert (
        main(
            [
                "import",
                "markers",
                *paths,
                "--markers",
                "1=fast,2=slow",
                "--cue-s",
                "3",
                "--subject",
                "old",
                "--session",
                "ssvep",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert "cues {'fast': 8, 'slow': 8}" in capsys.readouterr().out
    assert (
        main(["import", "markers", *paths, "--subject", "old", "--session", "none", "--out", str(out)]) == 1
    )  # no --markers
    assert (
        main(
            [
                "import",
                "markers",
                str(tmp_path / "missing.csv"),
                "--markers",
                "1=a",
                "--subject",
                "old",
                "--session",
                "m",
                "--out",
                str(out),
            ]
        )
        == 1
    )
    labelled = tmp_path / "brain_data1.csv"
    x = EEGSynth(seed=5).generate(256 * 30)
    np.savetxt(
        labelled,
        np.column_stack((np.arange(len(x)) / 256.0, x, np.repeat([0, 1, 0], 256 * 10))),
        delimiter=",",
        fmt="%.3f",
    )
    assert (
        main(
            [
                "import",
                "labels",
                str(labelled),
                "--names",
                "0=rest,1=forward",
                "--subject",
                "old",
                "--session",
                "day1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert "cues {'rest': 4, 'forward': 2}" in capsys.readouterr().out


# ---------------------------------------------------------------------- settings
def test_settings_files_and_overrides(tmp_path):
    cfg = settings.load()
    assert (
        cfg.frontend.fs == 256.0
        and cfg.engine.scheme == "auto"
        and cfg.protocol.commands == ("forward", "left", "right")
    )
    save_yaml(
        tmp_path / "site.yaml",
        {"frontend": {"settle_s": 2.0}, "engine": {"baseline_s": 30.0, "speeds": {"forward": [0.4, 0.4]}}},
    )
    args = build_parser().parse_args(
        ["monitor", "-c", str(tmp_path / "site.yaml"), "-s", "quality.rms_max_uv=80", "-s", "engine.switch.scan_s=3"]
    )
    cfg = settings.load(args, protocol={"runs": 5, "commands": None})
    assert cfg.frontend.settle_s == 2.0 and cfg.engine.baseline_s == 30.0 and cfg.quality.rms_max_uv == 80.0
    assert cfg.engine.switch.scan_s == 3.0 and cfg.protocol.runs == 5 and cfg.engine.speeds["forward"] == (0.4, 0.4)
    assert cfg.engine.speeds["left"] == (-0.35, 0.35)  # what the file does not mention keeps its default
    for bad in (["-s", "frontend.hopp_s=1"], ["-s", "nonsense"], ["-s", "frontend.fs=fast"]):
        with pytest.raises(ConfigError):
            settings.load(build_parser().parse_args(["monitor", *bad]))
    assert load_yaml(tmp_path / "site.yaml")["frontend"] == {"settle_s": 2.0}

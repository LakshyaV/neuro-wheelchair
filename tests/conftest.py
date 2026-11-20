"""Shared fixtures: simulated sessions and a model trained on one of them.

Everything is built once per test run. The simulated wearer's commands are made
clearly separable here, because these tests are about the software: that a
signal which is there gets decoded, tuned and acted on. Tests that need a signal
which is *not* there simulate their own.
"""

from __future__ import annotations

import pytest

from nwc.calibration.protocol import ProtocolConfig
from nwc.sim.eeg import SimConfig

PROTOCOL = ProtocolConfig(runs=3, trials_per_command=6, baseline_s=24.0, clench_trials=5)
SIM = SimConfig(effect=6.0)


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def sim_session(data_root):
    """A simulated calibration session on disk."""
    from nwc.sim.session import simulate_session

    return simulate_session(data_root / "sim" / "s01", PROTOCOL, SIM, seed=3, subject="sim", session="s01")


@pytest.fixture(scope="session")
def second_session(data_root):
    """The same wearer on another day: different seating of the headband, new data."""
    from nwc.sim.session import simulate_session

    return simulate_session(data_root / "sim" / "s02", PROTOCOL, SIM, seed=4, subject="sim", session="s02")


@pytest.fixture(scope="session")
def recording(sim_session):
    from nwc.decode.dataset import load_recording

    return load_recording(sim_session)


@pytest.fixture(scope="session")
def trained(sim_session):
    from nwc.decode.pipeline import train_bundle

    return train_bundle([sim_session], permutations=40)


@pytest.fixture(scope="session")
def bundle_dir(trained, tmp_path_factory):
    return trained.save(tmp_path_factory.mktemp("runs") / "bundle")

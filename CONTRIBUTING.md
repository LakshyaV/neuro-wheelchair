# Contributing

## Setup

```bash
git clone https://github.com/LakshyaV/neuro-wheelchair.git
cd neuro-wheelchair
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
make check
```

`make check` runs ruff and the test suite. The suite needs no headset and no
Arduino: streams come from the simulator and from local loopback sockets, and
the motor-controller sketch is compiled for the host and run against a mock
board (a C++ compiler is needed for that part; without one those tests are
skipped).

## How changes land

1. Open an issue that says what should change and how it will be known to work.
2. Branch from `main`: `feat/…`, `fix/…` or `docs/…`.
3. Commit in the conventional style (`feat(decode): …`, `fix(firmware): …`).
4. Open a pull request that closes the issue and says how the change was verified.
5. CI must pass: lint, tests on Python 3.10 and 3.12, and the firmware build for the Arduino Uno.

## Ground rules

- **Anything that can make the chair move, or keep it from stopping, comes
  with a test** that closes the loop in simulation (`tests/test_engine.py`) or
  runs the sketch (`tests/test_firmware.py`).
- **A claim about decoding comes with held-out evidence.** Splits are by run,
  never by frame, and a result is reported with its shuffled-label p-value.
  Numbers quoted in the documentation must be reproducible with a script in
  `tools/evidence/`.
- **Thresholds come from data.** If a limit is changed, say which recordings
  it was measured on.
- **The front-end is a contract.** Changing `FrontendConfig` defaults or the
  arithmetic of `nwc.dsp` changes the fingerprint, and existing models must
  then be retrained. Call that out in the pull request.
- **The firmware and its Python port stay identical.** `firmware/…/drive.h`
  and `nwc.link.controller` are compared at every millisecond by a test;
  change both together.
- After changing a settings dataclass, run `make config` to regenerate
  `configs/default.yaml`.

## Layout

See the repository layout in the [README](README.md) and the walk through the
signal path in [docs/architecture.md](docs/architecture.md).

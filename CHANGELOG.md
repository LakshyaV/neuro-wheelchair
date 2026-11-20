# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-20

A new architecture, built on measurements made on public recordings
([docs/evidence.md](docs/evidence.md)). The 2024 scripts are replaced; they
remain available under the tag `v0.1.0`.

### Added

- **Package and command line** (`nwc`): `demo`, `monitor`, `record`, `sim`,
  `import`, `train`, `evaluate`, `drive`, `link`, `info`, with typed settings in
  `configs/default.yaml`.
- **Stream handling** (`nwc.io`): OSC (Petal Metrics, Mind Monitor), Lab
  Streaming Layer, replay and simulator sources; an assembler that places
  samples by count or by the device's counter and measures loss; a session
  format that opens in other Muse tools; importers for marker and labelled
  recordings.
- **Front-end** (`nwc.dsp`): one causal, chunk-invariant pass producing
  multitaper cross-spectral frames and muscle-band power, with a configuration
  fingerprint; signal-quality limits set from real recordings; a jaw-clench
  detector fitted to the wearer.
- **Calibration** (`nwc.calibration`): a protocol that interleaves commands
  within every run, and a guided recorder.
- **Decoder** (`nwc.decode`): features relative to each session's baseline,
  exactly independent of electrode gain; shrinkage LDA in numpy; evaluation by
  held-out runs with a shuffled-label test and optional nested selection;
  models stored as arrays and JSON with a self-test on load.
- **Controller** (`nwc.control`): an arbiter that accumulates evidence with
  hysteresis and bridges short unusable stretches; thresholds tuned by
  replaying held-out runs against a false-start target; a direct and a switch
  driving scheme; a safety supervisor.
- **Motor link and firmware** (`nwc.link`, `firmware/wheelchair_controller`):
  a checksummed serial protocol; ramps, a command watchdog and an
  emergency-stop latch on the Arduino; speed control on the existing wiring.
- **Simulator** (`nwc.sim`): synthetic Muse EEG matched to real recordings, a
  Bluetooth transport with jitter and loss, a virtual controller and chair,
  scripted closed-loop drives.
- **Evidence scripts** (`tools/evidence`) and documentation: architecture,
  evidence, workflow, hardware, protocol, safety.
- **Tests and CI**: the test suite, including the sketch compiled for the host
  and run; lint, tests on Python 3.10 and 3.12, and firmware builds for the
  Arduino Uno.

### Changed

- The serial protocol to the Arduino. The single-digit commands of the 2024
  sketch are replaced by checksummed frames, so the sketch and the host program
  have to be updated together.
- Training data is recorded with `nwc record`. Files recorded as one file per
  command can be imported with `nwc import labels`.

### Removed

- `classification.py`, `realTime.py`, `saveRecorded.py`, `Arduino.ino` and the
  copied muselsl modules (`muse.py`, `stream.py`, `record.py`, `backends.py`,
  `constants.py`).

## [0.1.0] - 2024-07-28

The original prototype: band-power features and a linear SVM, OSC input from
Petal Metrics, single-digit commands to an Arduino with an L298N.

[0.2.0]: https://github.com/LakshyaV/neuro-wheelchair/releases/tag/v0.2.0
[0.1.0]: https://github.com/LakshyaV/neuro-wheelchair/tree/v0.1.0

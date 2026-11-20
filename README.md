# Mind-Controlled Wheelchair

[![CI](https://github.com/LakshyaV/neuro-wheelchair/actions/workflows/ci.yml/badge.svg)](https://github.com/LakshyaV/neuro-wheelchair/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Drive a wheelchair from a Muse EEG headband. The headset's signal is decoded
into commands on a laptop, a safety supervisor decides whether the chair may
act on them, and an Arduino drives the motors and stops them by itself when
anything goes quiet.

<img src="https://github.com/user-attachments/assets/281f2171-3195-4668-bf5a-cb52502bc661" alt="Mind-Controlled Wheelchair" width="330">

> **Before a person sits in it:** this is a research prototype, not a medical
> device. Follow [docs/safety.md](docs/safety.md): first runs with the drive
> wheels off the ground, a hand on the motor power switch, and the checks listed
> there passed on your own hardware.

## How it works

```
Muse headband ──Bluetooth──▶ stream app (Petal Metrics, muselsl, Mind Monitor)
                                   │  OSC or LSL
                                   ▼
        assembler          samples onto a uniform grid; loss counted, not guessed
                                   ▼
        front-end          one causal pass: cross-spectral frames 4x per second,
                           muscle-band power 16x per second
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
        signal quality       intent model           clench detector
        (artefacts, lost     (features relative     (the stop reflex,
         electrodes)          to today's baseline,   within 0.4 s)
                              shrinkage LDA)
              │                    ▼                     │
              │              arbiter: evidence           │
              │              accumulates, hysteresis     │
              └──────────┬─────────┴──────────┬──────────┘
                         ▼                    ▼
                   safety supervisor ◀── driving scheme (direct or switch)
                         │
                         ▼  checksummed serial frames, 10 per second
                   Arduino: ramps, watchdog, emergency-stop latch ──▶ L298N ──▶ motors
```

Two ways to drive:

- **Switch scheme, no training.** Directions are offered in turn. A jaw clench
  takes the one on offer; a clench while moving stops. It works for anyone the
  moment the headband is on, because a clench is the clearest deliberate signal
  these electrodes pick up.
- **Direct scheme, trained commands.** A fifteen-minute calibration records the
  wearer's own mental commands. Training reports honestly which of them can be
  told apart, and those drive the chair while they are held. A clench is the
  emergency stop.

## Try it without any hardware

```bash
git clone https://github.com/LakshyaV/neuro-wheelchair.git
cd neuro-wheelchair
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

nwc demo                  # simulated calibration, training and a drive with trained commands
nwc demo --scheme switch  # a drive with the jaw-clench switch alone
```

A virtual wearer drives a virtual chair through the real decoder, supervisor
and motor-controller logic:

```text
3. drive (direct scheme, virtual chair). The operator arms at 24 s, after the baseline.
   the wearer:
      30 s  think forward
      38 s  let go
      44 s  think left
      50 s  let go
      56 s  think forward
      61 s  clench the jaw (emergency stop)
   the chair:
       24.0 s  armed   stop
       35.7 s  armed   forward  forward
       41.0 s  armed   stop
       47.7 s  armed   left     left
       51.7 s  armed   stop
       58.2 s  armed   forward  forward
       61.2 s  halted  stop     emergency stop
   it ended at x +1.53 m, y -1.02 m, heading +228 deg, after 3.8 m, standing still: True
```

The simulator's commands are as separable as its `--effect` setting makes
them. What a real wearer's commands are worth is what `nwc train` reports on
their own calibration.

## From headset to driving

```bash
nwc monitor                                  # 1. is the stream complete, does every electrode have contact?
nwc record --subject s01                     # 2. guided calibration: baseline, jaw clenches, cued commands
nwc train --data data/s01 --out runs/s01     # 3. train, and read what the report says
nwc drive --bundle runs/s01/bundle --link /dev/cu.usbmodem1101    # 4. drive
```

Step 3 prints what matters for the decision to drive:

```text
decoding
  82 % of trials on held-out runs, where guessing gets 25 % (p = 0.010, 100 shuffles)
  per command: rest 88 %, forward 95 %, left 74 %, right 71 %
driving     held-out runs replayed through the controller
  commands switched on in 74 % of trials, 3.1 s after the cue (median); a wrong command in 4 %; 0.00 false starts per minute of rest
  reliable commands: forward, left
scheme      direct: the reliable commands drive while held
stop reflex threshold 12.5 dB above rest (rest stayed below 6.5 dB, calibration clenches above 18.6 dB)
```

To drive without calibrating, leave out the bundle: `nwc drive --link <port>`
uses the switch scheme. The whole procedure, with the choice of mental
strategies and what to do on the next day, is in [docs/workflow.md](docs/workflow.md).

EEG reaches the program over OSC from Petal Metrics (the default), over OSC
from Mind Monitor (`--source osc:0.0.0.0:5000`) or over Lab Streaming Layer
from muselsl, BlueMuse or Petal Metrics (`--source lsl`).

## What the design rests on

Each choice below was measured on recordings before it was built, and
[docs/evidence.md](docs/evidence.md) has the scripts that reproduce the numbers.

| Question | Measurement | Consequence |
| --- | --- | --- |
| Do the 2024 features (four band powers, linear SVM) find a signal that is there? | Real Muse recordings with interleaved conditions, whole recordings held out: 58 % of trials, where guessing gets 50 %. The pipeline here: 91 to 93 % on the same recordings. | Multitaper cross-spectra, 2 Hz bands, tangent-space features, shrinkage LDA. |
| Does it invent a signal that is not there? | A second person in the same data set: 51 %, and shuffled labels do as well (p = 0.36). Picking the best of four configurations on the test data would have shown 61 %. | Whole runs held out, a shuffled-label test with every result, one fixed configuration by default. |
| Can "think left, think right" be read at a Muse's electrode positions? | 30 people, research-grade EEG: imagined left against right fist 54 % at Muse-like positions. | Commands are whatever the wearer's calibration shows to be separable, and the switch scheme needs none. |
| Does a model survive re-seating the headband? | A model that has never seen the person: 79 % without, 92 % with each person expressed against their own reference. | Every session starts with a baseline; features are taken relative to it and are exactly independent of electrode gain. |
| What happens with one recording per command, as the 2024 training data had? | A public Muse data set built that way: 91 to 100 % "accuracy", where one state's files stand out by a badly seated electrode and, for three of four people, by the day they were recorded. | Calibration interleaves commands within every run, and evaluation flags data that does not. |
| Are packet timestamps a clock? | In real recordings stamped by packet arrival, consecutive stamps differ by anything from -15 to +33 ms and three packet boundaries in ten are off by a sample or more, while the samples are regular. | Samples are placed by count, and by the device's counter where there is one. |

## Hardware

- Muse headband (2016, 2 or S) and a computer with Bluetooth
- Arduino Uno with an L298N motor driver and two DC motors
- The wiring of the original build is unchanged: IN1 to IN4 on pins 2 to 5

```bash
arduino-cli compile --fqbn arduino:avr:uno --upload -p /dev/cu.usbmodem1101 firmware/wheelchair_controller
nwc link --port /dev/cu.usbmodem1101          # identify, telemetry, watchdog: moves nothing
nwc link --port /dev/cu.usbmodem1101 --keys   # drive from the keyboard, wheels off the ground
```

The controller ramps the motors, stops them 400 ms after the last valid
command, and has an optional emergency-stop input that latches. Speed control
works on the existing wiring; see [docs/hardware.md](docs/hardware.md).

## Commands

| Command | Purpose |
| --- | --- |
| `nwc demo` | simulated calibration, training and drive |
| `nwc monitor` | stream completeness and electrode contact |
| `nwc record` | guided calibration session |
| `nwc sim` | simulated calibration sessions |
| `nwc import` | recordings made with other tools |
| `nwc train` | train an intent model and report what it can do |
| `nwc evaluate` | test a model on sessions it was not trained on |
| `nwc drive` | drive, with a model or with the switch alone |
| `nwc link` | test the motor controller without EEG |
| `nwc info` | describe a session or a model |

Every setting is in [configs/default.yaml](configs/default.yaml); pass your own
file with `-c` or single values with `-s engine.baseline_s=30`.

## Repository layout

```
src/nwc/
  io/           sources (OSC, LSL, replay), stream assembly, sessions on disk, importers
  dsp/          front-end, cross-spectra and covariance geometry, signal quality, clench detector
  calibration/  the calibration protocol and the guided recorder
  decode/       features, shrinkage LDA, evaluation, the model bundle
  control/      arbiter, driving schemes, safety supervisor, threshold tuning
  link/         serial protocol, motor link, the controller's logic in Python
  runtime/      the engine and the driving loop
  sim/          synthetic EEG, virtual controller and chair, scripted drives
firmware/wheelchair_controller/   Arduino sketch: protocol.h and drive.h (the logic), controller.h (pins, serial), config.h
tests/          the test suite, including the sketch compiled for the host and run
tools/evidence/ the measurements above, reproducible on public data
docs/           architecture, evidence, workflow, hardware, protocol, safety
```

## Documentation

- [docs/architecture.md](docs/architecture.md): every stage and why it is built this way
- [docs/evidence.md](docs/evidence.md): the measurements, and how to repeat them
- [docs/workflow.md](docs/workflow.md): from first stream check to driving, day after day
- [docs/hardware.md](docs/hardware.md): wiring, firmware options, bench tests
- [docs/protocol.md](docs/protocol.md): the serial protocol
- [docs/safety.md](docs/safety.md): what stops the chair, how fast, and the checks to pass first

The scripts of the 2024 prototype are kept under the tag
[`v0.1.0`](https://github.com/LakshyaV/neuro-wheelchair/tree/v0.1.0).

## Development

```bash
pip install -e ".[all]"
make check        # ruff and the test suite
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).

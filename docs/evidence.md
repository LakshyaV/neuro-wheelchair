# Evidence

The design decisions in [architecture.md](architecture.md) were made on
measurements. This page gives the measurements, the data they were made on and
the commands that repeat them. The scripts are in
[`tools/evidence/`](../tools/evidence) and run the code of this package, with
the two reference methods (the 2024 feature set and a convolutional network)
implemented beside it.

```bash
pip install -e ".[all]" -r tools/evidence/requirements.txt
```

All data sets are public and are downloaded by the scripts on request. None of
them is redistributed here.

| Data | What it is | Used for |
| --- | --- | --- |
| EEG-ExPy visual SSVEP examples (NeuroTechX) | Muse 2016, two people, two flicker frequencies alternating at random within each recording | decoders on real Muse data with a sound design |
| EEG Motor Movement/Imagery Database (Schalk et al. 2004, on PhysioNet) | 64-channel research EEG, 109 people, cued trials | what Muse-like electrode positions can pick up; transfer to unseen people |
| Mental-state recordings of Bird et al. (2018) | Muse 2016, four people, one state per file | what one recording per state does to an evaluation |

## 1. Decoders on real Muse recordings

```bash
python tools/evidence/muse_recordings.py --download
python tools/evidence/muse_recordings.py --eegnet      # the network needs torch
```

Two flicker frequencies (20 and 30 Hz) alternate at random every 3.6 seconds
inside each two-minute recording. The condition is therefore not tied to the
recording, and whole recordings can be held out, which is the split that
predicts a new recording.

The first two rows score the 2024 method the way the prototype did, on single
one-second windows; always answering the more frequent condition gets 55 %
there. All other rows are the balanced accuracy of one decision per trial with
whole recordings held out, where guessing gets 50 %.

| Method | Person 1 (190 trials) | Person 2 (140 trials) |
| --- | --- | --- |
| 2024 features (four band amplitudes) + linear SVM, single windows split at random | 59.0 % | 54.9 % |
| The same, whole recordings held out | 57.2 % | 50.2 % |
| The same, one decision per trial | 57.7 % | 49.4 % |
| This pipeline: tangent space, 2 Hz bands, shrinkage LDA | **91.3 %** (p = 0.005) | **50.9 %** (p = 0.36) |
| This pipeline with nested selection among four configurations | 93.1 % (p = 0.005) | 52.6 % (p = 0.21) |
| Best of the four configurations when picked on the test data | 93.1 % | 61.2 % |
| EEGNet on raw two-second windows, three seeds | 96.3 % (96.0 to 96.9) | 56.3 % (53.7 to 58.7) |

What this shows:

- **The 2024 feature set misses a signal that is there.** Four broad bands
  cannot tell 20 Hz from 30 Hz; both fall into "beta". A spectrum resolved in
  2 Hz steps can.
- **The pipeline does not invent a signal that is not there.** For the second
  person nothing decodes the condition, and the shuffled-label test says so.
  The 61.2 % in the sixth row is what choosing a configuration on the test data
  produces from recordings that are at chance. This is why selection is off by
  default and nested when on.
- **A random split of overlapping windows flatters only slightly here**,
  because conditions alternate within the recording. With one recording per
  condition the same split is badly misleading (section 3).

### The convolutional network

EEGNet (Lawhern et al. 2018), about 1,200 convolution weights, was trained from
scratch on the same windows. With five recordings to train on it is ahead of
the linear pipeline for the first person. The question for a calibration that
lasts a quarter of an hour is what happens with less:

| Trained on | Trials | Tangent + LDA | Band power + LDA | EEGNet |
| --- | --- | --- | --- | --- |
| 1 recording | about 31 | 73.3 % | 73.9 % | 66.3 % |
| 2 recordings | about 63 | 86.4 % | 86.7 % | 93.0 % |
| 3 recordings | about 95 | 89.1 % | 87.8 % | 93.8 % |
| 5 recordings | about 160 | 91.3 % | 93.1 % | 96.3 % |

(Person 1, tested on all recordings not trained on, mean over splits.)

On this data the network leads by four to seven points from about sixty
trials upward and trails with thirty. A plausible reason for the lead is the
kind of signal: a flicker evokes a narrow line that is locked to the stimulus,
and a temporal convolution can match its exact frequency, where 2 Hz bands
blur it. The commands of this project are states held without a stimulus, for
which no public Muse recordings with a sound design exist, so whether the lead
carries over cannot be answered here.

The linear model is the default because of what else it brings: a fit in a
millisecond, which makes the shuffled-label test and the replay tuning of the
controller possible; determinism; features taken relative to the session's
baseline with exact independence of electrode gain (section 2); and no
dependency beyond numpy at run time.

## 2. What these electrode positions can pick up

```bash
pip install mne
python tools/evidence/electrode_positions.py
```

A Muse measures at AF7, AF8, TP9 and TP10 against Fpz. The PhysioNet database
has AF7, AF8, TP7, TP8 and Fpz, so a Muse-like montage can be emulated on
research-grade recordings and compared with four electrodes over the motor
cortex (C3, C4, Cz, CPz), same reference, same pipeline. Thirty people.

**Imagined and real movement.** Per person, whole runs held out, about 45
trials each, guessing gets 50 %. The last column counts people whose score a
coin would reach less than one time in twenty.

| Task | Montage | Mean | Median | People above chance |
| --- | --- | --- | --- | --- |
| Imagined left fist against right fist | Muse-like | 53.9 % | 55.3 % | 5 of 30 |
| | motor cortex | 55.4 % | 54.0 % | 8 of 30 |
| Imagined fists against feet | Muse-like | 56.4 % | 58.0 % | 10 of 29 |
| | motor cortex | 69.0 % | 66.7 % | 18 of 29 |
| Real left fist against right fist | Muse-like | 54.9 % | 53.5 % | 7 of 29 |
| | motor cortex | 60.5 % | 57.7 % | 10 of 29 |

At Muse-like positions imagined movement is within a few points of guessing
for most people. Electrodes over the motor cortex do clearly better on the
task that separates well (fists against feet), with the same four-channel
pipeline. Left against right fist is hard for four channels wherever they sit.
The commands of a Muse-driven chair should therefore not be imagined
movements.

**Eyes closed against eyes open, on a person the model has never seen.** One
minute of each, two-second windows, all other people used for training.

| Montage | Features | Relative to everyone else's reference | Relative to the person's own |
| --- | --- | --- | --- |
| Muse-like | band power | 77.2 % | 89.0 % |
| Muse-like | tangent space | 78.7 % | **91.9 %** |
| motor cortex | band power | 74.0 % | 84.7 % |
| motor cortex | tangent space | 77.4 % | 85.9 % |

Two things follow. A change of state that these electrodes do pick up is
decoded at 92 % without a single labelled example from the person. And it
takes expressing each person against their own reference to get there, which
is worth thirteen points. That is the mechanism behind the baseline at the
start of every session.

Independence of electrode gain is exact, not approximate, and is checked by a
test (`test_a_model_from_one_day_holds_up_on_another`): every channel of a
session is multiplied by a different factor between 0.4 and 2, and the model's
output against a baseline taken with the same factors does not change.

## 3. One recording per state

```bash
python tools/evidence/one_recording_per_state.py --download
python tools/evidence/one_recording_per_state.py
```

A public Muse data set with three "mental states", each recorded in files of
its own. The 2024 prototype of this project collected its training data the
same way, one file per command.

| Person | "Accuracy" on five-second pieces (guessing: 33 %) | Recorded | A forehead electrode in the "concentrating" files | The same electrode in the other files |
| --- | --- | --- | --- | --- |
| a | 93 % | concentrating two days after the rest | 34 and 70 µV | 4 to 12 µV |
| b | 97 % | concentrating two days after the rest | 113 and 145 µV | 8 to 17 µV |
| c | 100 % | same day, the third state hours later | 30 and 34 µV | 5 to 7 µV |
| d | 91 % | concentrating two days after the rest | 86 µV | 4 to 18 µV |

A classifier separates the files almost perfectly. What it has learned about
states of mind cannot be told from such data, because every file of one state
differs from the others in when it was recorded and how the headband sat. The
quality check of this package refuses 62 to 100 % of the "concentrating"
frames as artefact or lost contact.

The evaluation flags data of this shape (`confounded`), and the report prints
a warning in place of a recommendation. The calibration protocol is built so
that the situation cannot arise: all commands in every run, in random order.

## 4. What real recordings look like at rest

```bash
python tools/evidence/resting_statistics.py
```

Forty-two recordings from the two Muse data sets, run through the front-end,
the quality check and the clench detector. Nobody was asked to clench or move.

| | |
| --- | --- |
| Packet timing | 6 recordings carry arrival-time stamps: consecutive stamps differ by -15 to +33 ms (one sample is 3.9 ms) and 30 % of packet boundaries are off by a sample or more. The other 36 carry regularised stamps. |
| Delivered rate, counted | 254.2 to 256.2 samples per second |
| Amplitude 1 to 45 Hz, electrodes in contact | median TP9 7.6, AF7 4.3, AF8 5.6, TP10 7.2 µV |
| An electrode above the 60 µV limit (4 recordings) | 70 to 145 µV, while the other channels read about 10 µV |
| Analysis windows refused by the quality check | 20 % on average, 0 to 78 % per recording |
| Power line, strongest channel, above the spectrum around it | 50 Hz in 22 recordings: 8 to 38 dB. 60 Hz in 20 recordings: 16 to 56 dB |
| The same line left in the muscle band | resting level behind the ears raised by 12.8 dB in the median, by up to 33.5 dB |
| Stop reflex in 33 minutes without a clench | fired four times at 12 dB, once at 15 dB (the default), never at 18 dB |

These are the numbers behind the assembler placing samples by count, the
amplitude limits of the quality check, the arbiter bridging short stretches
without usable frames, the default threshold of the stop reflex, and the
front-end removing the power line at both frequencies without being told which
one to expect.

## 5. The controller and the firmware

These are verified by tests, not by measurement on recordings.

- `tests/test_firmware.py` compiles the unmodified Arduino sketch for the host
  against a mock of the Arduino API and runs it: ramps, reversal through
  standstill, damaged lines, the watchdog, the emergency-stop latch, three
  wiring variants. A Python port of the controller's logic agrees with it at
  every millisecond of a scripted scenario and byte for byte on the serial
  line.
- `tests/test_engine.py` closes the loop in simulation: a virtual wearer, the
  real engine, the virtual controller and chair. The chair must stay put while
  the wearer rests, start and stop on command, halt within 0.4 s of a clench
  and stay halted, and stop when an electrode lifts off, when the headset goes
  silent and when the host program dies.
- `tests/test_decode.py` checks that a simulated wearer whose commands leave no
  trace in the EEG is reported as such: accuracy near chance, a large p-value,
  no reliable commands, and no false starts.

## What this evidence does not cover

The measurements above establish that the decoder finds what is there and
reports what is not, on real Muse recordings; what these electrode positions
can and cannot carry; and that the controller behaves as specified. They do
not say how well a particular person's mental commands will separate. That
depends on the person and the strategies chosen, and it is what the report of
`nwc train` on their own calibration is for.

Nor do they include a real jaw clench. The public recordings contain none on
cue, so on real data only the false-trigger side of the stop reflex was
measured (section 4). Detection was tested on simulated clenches, and for a
wearer it is measured from their calibration clenches: `nwc train` prints the
margin between their rest and their clench and warns when it is small.

## References

- Barachant A., Bonnet S., Congedo M., Jutten C. (2012). Multiclass brain-computer interface classification by Riemannian geometry. *IEEE Transactions on Biomedical Engineering* 59(4).
- Bird J. J., Manso L. J., Ribeiro E. P., Ekart A., Faria D. R. (2018). A study on mental state classification using EEG-based brain-machine interface. *International Conference on Intelligent Systems*.
- Blankertz B., Lemm S., Treder M., Haufe S., Müller K.-R. (2011). Single-trial analysis and classification of ERP components: a tutorial. *NeuroImage* 56(2).
- Friedrich E. V. C., Neuper C., Scherer R. (2013). Whatever works: a systematic user-centered training protocol to optimize brain-computer interfacing individually. *PLoS ONE* 8(9).
- Goldberger A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23).
- Lawhern V. J. et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering* 15(5).
- Ledoit O., Wolf M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2).
- Schalk G., McFarland D. J., Hinterberger T., Birbaumer N., Wolpaw J. R. (2004). BCI2000: a general-purpose brain-computer interface system. *IEEE Transactions on Biomedical Engineering* 51(6).
- Thomson D. J. (1982). Spectrum estimation and harmonic analysis. *Proceedings of the IEEE* 70(9).
- Zanini P., Congedo M., Jutten C., Said S., Berthoumieu Y. (2018). Transfer learning: a Riemannian geometry framework with applications to brain-computer interfaces. *IEEE Transactions on Biomedical Engineering* 65(5).
- EEG-ExPy (formerly eeg-notebooks), NeuroTechX: https://github.com/NeuroTechX/EEG-ExPy

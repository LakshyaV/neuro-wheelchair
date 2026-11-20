# Architecture

The system turns four channels of scalp voltage into wheel speeds. This
document walks the signal path once, stage by stage, and says for each stage
what it does and why it is built that way. Where a choice rests on a
measurement, the measurement is in [evidence.md](evidence.md).

```
source ─▶ assembler ─▶ front-end ─┬─▶ quality ──────────────┐
                                  ├─▶ model ─▶ arbiter ──┐  │
                                  └─▶ clench detector ─┐ │  │
                                                       ▼ ▼  ▼
                                          scheme ─▶ supervisor ─▶ motor link ─▶ controller
```

The guiding rule throughout: nothing may make the chair move that could not
also be shown, on held-out data or in a test, to be the wearer's doing.

## What the electrodes can see

A Muse has four dry electrodes, on the forehead (AF7, AF8) and behind the ears
(TP9, TP10), referenced to the middle of the forehead. That decides what can be
decoded at all, before any algorithm is chosen.

- **Not imagined movement.** The classic commands of brain-computer
  interfaces, "imagine moving your left hand, your right hand", live over the
  motor cortex at the top of the head. On 30 people of research-grade EEG,
  imagined left against right fist is decoded at 54 % from Muse-like positions,
  where 50 % is guessing.
- **Rhythms that change with what the wearer is doing.** Closing the eyes,
  doing arithmetic, recalling a tune, relaxing: these change the strength of
  rhythms across the scalp in ways these electrodes pick up. How well, and
  which, differs from person to person, which is why the commands are learned
  from the wearer and tested before they are used.
- **Muscle.** The jaw muscles run under the electrodes behind the ears, and
  clenching them produces broadband activity far larger than EEG at those
  sites. It is the clearest deliberate signal a Muse offers; the headset's own
  apps report a jaw clench as an event for that reason. How far a wearer's
  clench stands above their rest is measured from their calibration clenches
  and printed by `nwc train`.

The architecture follows from these three facts. The clearest signal is
reserved for stopping. Commands are learned per wearer and must prove
themselves on held-out data. And for a wearer whose commands do not separate,
there is a driving scheme that needs only the clench.

## Sources and the assembler

`nwc.io.sources`, `nwc.io.stream`

EEG arrives over OSC (Petal Metrics, Mind Monitor) or Lab Streaming Layer
(muselsl, BlueMuse, Petal Metrics), from a recording being replayed, or from
the simulator. All hand over the same thing: chunks of samples with a time and,
where the transport has one, the device's own sample counter.

Everything downstream assumes equally spaced samples, so the assembler puts
the stream on a grid. Two properties of Bluetooth headsets shaped it:

- **Timestamps are not a sample clock.** Some stream bridges stamp samples
  by the arrival of their packet. In real Muse recordings made that way,
  consecutive stamps differ by anything from -15 to +33 ms and three packet
  boundaries in ten are off by a sample or more, while the samples themselves
  are perfectly regular. Other bridges regularise the stamps, which hides lost
  packets altogether. Samples are therefore placed by count.
- **Loss is only visible in the device's counter.** Where the transport
  carries it, a hole is located exactly: up to a quarter of a second is filled
  by interpolation and flagged, a longer hole is a break and the filters
  restart after it. Without a counter only an unmistakable stall counts as a
  break.

Stream health is reported as samples delivered per second and share lost,
counted, not inferred from a typical interval.

## The front-end

`nwc.dsp.frontend`, `nwc.dsp.spectral`

One implementation serves recording analysis, training, evaluation and live
driving. It is causal and chunk-invariant: the same stream fed sample by
sample, in packets of twelve or in one piece yields bit-identical frames, which
a test checks. A model therefore cannot meet features in use that differ from
the ones it was trained on, and the configuration's fingerprint is stored with
every model to enforce it.

Two kinds of frame come out.

**Slow frames, four per second.** Each summarises the last two seconds as a
multitaper cross-spectrum in 1 Hz bins. That one array answers every later
question: the covariance of the signal in any band is a sum over bins, so band
layouts are chosen after the fact; the diagonal is band power and the
off-diagonal terms are how channels move together; amplitude for the quality
check comes from other bins. Multitaper estimation uses the whole window with
nearly uniform weight and suppresses leakage from the large slow components of
dry-electrode EEG.

A frame is valid if the filters had settled and at most a tenth of its samples
were interpolated. Insisting on complete windows would discard nine in ten of
them at 5 % packet loss, for no gain: filling the holes barely moves the
spectrum.

**Fast frames, sixteen per second.** Each is the power of the muscle band
over an eighth of a second, for the clench detector. A stop must not wait for a
two-second window to fill.

**The power line** is taken out everywhere, at 50 and at 60 Hz and their
harmonics, without a setting for the country. A line left in the muscle band
would raise its resting level, on real recordings by 13 dB in the median and
by up to 33 dB, and a clench would stand out that much less. Whether the stop
reflex works must not depend on a setting somebody has to remember.

## Signal quality

`nwc.dsp.quality`

The limits are in microvolts and come from real recordings. A channel in good
contact has a median of 4 to 8 µV between 1 and 45 Hz; a forehead electrode that
has lifted off reads 70 µV and more for as long as it is off. Single frames
above the limit are common in good recordings too. So two things are kept
apart:

- an **artefact** is a frame that is too large or swings too far at low
  frequency (a blink, a head movement). It carries no usable evidence;
- **lost contact** is a channel out of range for most of the last three
  seconds. That is a fault and stops the chair.

Power-line pickup is not used as a sign of contact. Between recordings that
were otherwise equally good the line stood anywhere from 16 to 56 dB above its
surroundings, depending on the room.

## The baseline, and features relative to it

`nwc.decode.features`

Every session, recorded or live, begins with the wearer sitting relaxed with
eyes open. The mean cross-spectrum of that stretch is the session's reference,
and every later frame is expressed against it: a band covariance `C` becomes
`W C Wᵀ`, where `W` first divides each channel by its baseline level and then
removes the correlation that remains. The feature vector is the matrix
logarithm of the result (tangent space), or only its diagonal (band power).

This does two things.

- It turns the question from "what does the signal look like" into "how has it
  changed since the wearer sat relaxed a few minutes ago", which is what a
  mental command is.
- It makes the features exactly independent of each electrode's gain, because
  gains cancel in the first step. Contact quality, and with it gain, is what
  differs most between two sittings with dry electrodes. A test multiplies
  every channel by a different factor and checks that the model's output does
  not change in the sixth decimal.

On 30 people, a model that had never seen the person told eyes closed from
eyes open in 79 % of windows without this step and in 92 % with it.

## The intent model

`nwc.decode.lda`, `nwc.decode.model`, `nwc.decode.train`

The classifier is linear discriminant analysis with Ledoit-Wolf shrinkage on
the tangent-space features over 2 Hz bands from 1 to 45 Hz. It has no
hyper-parameters, a closed-form fit that takes a millisecond, and is
implemented here in a few dozen lines of numpy, checked against scikit-learn.

A model is arrays: a standardisation, a weight vector and bias per class, a
temperature. It runs on numpy alone and is stored as arrays and JSON. Nothing
is pickled, so a model does not stop loading when a library changes version.

Why not a neural network? A compact convolutional network (EEGNet) trained on
raw windows was measured against this model on real Muse recordings, and on
that data it is ahead by four to seven points once it has sixty or more
training trials ([evidence.md](evidence.md#1-decoders-on-real-muse-recordings)).
The linear model is the default all the same, for reasons that weigh more in
this application than those points. It fits in a millisecond, which is what
makes the shuffled-label test and the replay tuning below possible at all. It
is deterministic. Its features carry the baseline normalisation that lets a
model outlive one sitting, with exact independence of electrode gain. It runs
on numpy. And it does better than the network when a calibration yields few
trials. The model is behind a small interface (class probabilities for a
frame), so another family can be added without touching what surrounds it.

### Evaluation

Three rules, each a response to a way of fooling oneself.

**Splits are by run.** Frames of a trial overlap by seven eighths, and trials
of a run share electrode contact and time of day. Only a held-out run predicts
the next run. The calibration protocol therefore interleaves all commands
within every run, in random order, so that nothing but the command tells
trials apart. Data recorded as one file per command, as the 2024 prototype's
was, can be imported, but evaluation flags it: on a public data set built that
way a classifier scores 91 to 100 % by telling the files apart.

**Chance is measured.** Decisions are scored per trial, and the whole
procedure is repeated with trial labels shuffled inside each run. The share of
shuffles that do as well is the p-value printed with every result. One of two
people in the real Muse recordings is at 51 % with p = 0.36, and the report
says so in words.

**One configuration unless asked.** Picking the best of several
configurations on the data that scores them flatters the result; on that same
person the best-looking of four showed 61 %. Selection is available
(`--select`) and is then nested, so each held-out run is predicted by a
configuration chosen without it.

## The arbiter

`nwc.control.arbiter`, `nwc.control.tune`

A classifier that is honest about a single two-second window is rarely sure of
anything; after calibration its probabilities are flat. A wheelchair has to be
sure. The arbiter accumulates log-probabilities with a leak, so a command held
for a few seconds builds to a confident posterior while an odd frame fades. A
command switches on after its posterior has stayed above one threshold for a
dwell time and off after it has stayed below a lower one. Commands hand over
only through rest.

Unusable frames are bridged, briefly. One blink spoils two seconds of analysis
windows, and on real recordings one window in five is spoiled by something, so
evidence is held unchanged for up to two and a half seconds without usable
frames. Beyond that it drains and the command ends. Unusable frames can never
start a command.

The thresholds are not chosen by hand. Training replays every held-out
calibration run through the arbiter, with probabilities from a model fitted
without that run, and scores what a rider would experience: how often a command
switched on when asked, after how long, how often a wrong one did, and how
often one started while rest was being asked for. The setting is picked for a
target false-start rate (0.5 per minute by default). Commands that switched on
in at least 70 % of their trials are called reliable, and the scheme the
report recommends follows from how many there are.

## Driving schemes

`nwc.control.scheme`

- **Direct.** Each reliable command drives while it is held. A clench is the
  emergency stop and latches.
- **Switch.** One switch does everything. While the chair stands, the options
  are offered in turn and a press takes the one on offer; while it moves, a
  press stops it. Turns end by themselves, driving ends at the next press or a
  time limit. The switch is the clench, or a single reliable command with the
  clench kept as emergency stop.

With two or more reliable commands the report recommends direct, with one it
recommends that command as the switch, with none the clench.

## The stop reflex

`nwc.dsp.reflex`

Power in the muscle band on both sides must exceed the resting level, measured
in the baseline, by a threshold for three fast frames; it releases when either
side falls below a lower one. The threshold is fitted to the wearer's
calibration clenches, halfway in decibels between their largest resting
excursions and a weak clench. Requiring both sides keeps chewing on one side
and a touched electrode from counting. On 33 minutes of real recordings in
which nobody was asked to clench, the detector fired once at its default
threshold, which costs a stop and nothing else.

## The supervisor

`nwc.control.supervisor`

The only place that can let the chair move. It starts disarmed and grants a
scheme's request only while every condition holds:

| State | Meaning | Leaves by |
| --- | --- | --- |
| disarmed | nothing moves | the operator arming, after the baseline |
| armed | requests pass, within a limit on uninterrupted motion | a fault or an emergency stop |
| fault | no data for a second, an electrode lost, or the controller silent | itself, after two healthy seconds |
| halted | the wearer asked for an emergency stop | the operator, or a two-second clench |

When the supervisor leaves the armed state, the arbiter and the scheme are
reset: nothing that was in progress resumes by itself.

## The motor link and the controller

`nwc.link`, `firmware/wheelchair_controller`

The host sends wheel speeds ten times a second as checksummed lines of text
([protocol.md](protocol.md)). The controller is where stopping is guaranteed,
because it is the one part that keeps running when the laptop does not:

- outputs ramp and never jump; a reversal goes through standstill;
- 400 ms without a valid drive command and the outputs go to zero;
- an optional input for a hardware switch cuts the outputs at once and latches
  until the switch is closed again and the host sends a release.

The logic lives in two headers with no Arduino dependencies (`protocol.h`,
`drive.h`); a third connects them to the pins and the serial port. The test
suite compiles the unmodified sketch for the host, at the language level of
the board's toolchain, against a mock of the Arduino API and runs it. A Python
port of the same logic, used by the simulator, is checked against it at every
millisecond of a scripted scenario.

## The simulator

`nwc.sim`

Synthetic Muse EEG with levels matched to real recordings, a transport with
jittered packet arrival and loss, a virtual controller and a chair. It exists
so that every path can be exercised without hardware, in tests and in
`nwc demo`. It says nothing about how separable a person's commands are; that
is a setting (`effect`), and at zero the evaluation has to report chance,
which a test checks.

## What changed from the 2024 prototype

| The prototype | What was found | Now |
| --- | --- | --- |
| One CSV per command, windows split at random | A classifier can score by recognising the file | Interleaved cued trials, whole runs held out, shuffled-label test |
| Four band powers and a linear SVM | 58 % on real recordings where 91 to 93 % is there | Multitaper cross-spectra, 2 Hz bands, tangent space, shrinkage LDA |
| A scaler fitted on each stretch of data | Features follow electrode contact | Features relative to the session's baseline, independent of gain |
| Filters restarted on every buffer | Training and driving see different signals | One causal, chunk-invariant front-end with a fingerprint |
| A pickled model | Breaks with library versions | Arrays and JSON, self-test on every load |
| Every window's label sent to the motors | The chair follows the classifier's noise | Evidence accumulation with hysteresis, tuned by replay |
| A digit over serial, motors latched | A crashed laptop leaves the chair driving | Checksummed frames, watchdog, ramps, emergency-stop latch |
| No way to stop but another classification | | A jaw-clench reflex, about a quarter of a second |
| Copied library files for Bluetooth | | Standard stream sources: OSC and LSL |

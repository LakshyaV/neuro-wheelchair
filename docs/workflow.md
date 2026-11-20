# Workflow

From the first stream check to driving, and what to do on the days after.
Every command has `--help`; every setting is in
[configs/default.yaml](../configs/default.yaml).

## 1. Get the headset streaming

The program reads EEG from a stream that another app provides.

| App | Start it with | Read it with |
| --- | --- | --- |
| Petal Metrics | streaming enabled, OSC, port 14739 | the default (`--source osc`) |
| Mind Monitor (on a phone) | OSC streaming to this computer's address, port 5000 | `--source osc:0.0.0.0:5000` |
| muselsl | `muselsl stream` | `--source lsl` |
| BlueMuse | start streaming | `--source lsl` |
| Petal Metrics over LSL | LSL enabled | `--source lsl:PetalStream_eeg` |

By default the OSC source listens on this computer only, which is where Petal
Metrics sends. A host in the specification (`0.0.0.0` for all interfaces)
opens it to a stream sent over the network.

Nothing has to be set for the country you are in: power-line pickup at 50 and
at 60 Hz is removed either way.

LSL input needs the liblsl library. Current `pylsl` packages for macOS,
Windows and recent Linux include it; where they do not, install it from the
[liblsl releases](https://github.com/sccn/liblsl/releases) or with
`conda install -c conda-forge liblsl`.

## 2. Check the signal

```bash
nwc monitor
```

```text
source osc://127.0.0.1:14739/PetalStream/eeg: TP9, AF7, AF8, TP10 at 256 samples/s nominal
  time     rate   lost         TP9           AF7           AF8          TP10       usable
     3    255.8   0.1%      7.1 uV        5.3 uV        5.4 uV        9.6 uV         100%
     4    256.1   0.0%      7.9 uV       13.6 uV       12.1 uV       10.3 uV         100%
     5    255.9   0.0%      8.2 uV       16.5 uV       15.7 uV        9.9 uV         100%
     6    255.7   0.1%      7.4 uV        6.5 uV        8.3 uV        9.0 uV         100%
```

- **rate** should sit at the nominal 256 samples per second and **lost** near
  zero. A few percent of lost packets are tolerated; a stream that stalls is not.
- Each channel should read a few microvolts with a relaxed face. A channel
  marked `LOST` reads tens of microvolts and more: wet the electrode, move
  hair out of the way, re-seat the band, and wait for it to settle.
- **usable** is the share of analysis windows of the last five seconds that
  are free of artefacts. Blinking and moving lower it; 80 % and more is a good
  recording condition.

Do this before every recording and every drive.

## 3. Decide what the commands will be

A command is something the wearer does with their mind, the same way every
time, for about five seconds. What matters is that the commands differ from
each other and from rest in a way these four electrodes can pick up.

- **Imagined movement is a poor choice with this headset.** It is read over
  the motor cortex, where a Muse has no electrode
  ([evidence.md](evidence.md#2-what-these-electrode-positions-can-pick-up)).
- **Mental tasks of different kinds work better as a set.** Friedrich, Neuper
  and Scherer (2013) screened mental subtraction, word association, auditory
  imagery, spatial navigation and mental rotation as commands and chose the
  best combination per person. The cues of `nwc record` suggest some of these;
  replace them with whatever the wearer can do consistently.
- **Fewer commands are more reliable.** The default is three (forward, left,
  right) plus rest. Start there.
- **Closing the eyes** changes the signal strongly and makes a clear command
  for something done while standing still. It is not a command to give while
  the chair moves.

The wearer should keep the face relaxed and blink normally. Tensing the jaw is
the stop signal and is calibrated separately.

## 4. Record a calibration

```bash
nwc record --subject s01
nwc record --subject s01 --commands forward,left,right,backward --runs 4
```

The session begins with a 30 s baseline (relaxed, eyes open), then six jaw
clenches on cue, then three runs. In each run every command and rest come up
eight times in random order, five seconds each, with a pause in between. It
takes about a quarter of an hour, with a break between runs.

Two properties of this protocol are what make the result trustworthy, and
recordings made another way lack them: commands are interleaved, so nothing
but the command distinguishes trials, and runs are separate, so whole runs can
be held out.

The recording is written to `data/<subject>/<session>/` as it arrives.

## 5. Train, and read the report

```bash
nwc train --data data/s01 --out runs/s01
```

```text
recordings
  s01/20261120-0930: 13.8 min, 256.0 samples/s, 91 % of frames usable
trials      rest 24, forward 20, left 23, right 24 in 3 runs
decoding
  82 % of trials on held-out runs, where guessing gets 25 % (p = 0.010, 100 shuffles)
  per command: rest 88 %, forward 95 %, left 74 %, right 71 %
  asked (rows) against decoded (columns):
                  rest  forward     left    right
    rest            21        0        0        3
    forward          1       19        0        0
    left             6        0       17        0
    right            7        0        0       17
driving     held-out runs replayed through the controller
  commands switched on in 74 % of trials, 3.1 s after the cue (median); a wrong command in 4 %; 0.00 false starts per minute of rest
  per command: forward 88 %, left 75 %, right 58 %
  reliable commands: forward, left
scheme      direct: the reliable commands drive while held
stop reflex threshold 12.5 dB above rest (rest stayed below 6.5 dB, calibration clenches above 18.6 dB)
```

How to read it:

- **decoding** is how often a held-out trial was assigned the right command,
  and the p-value says how often shuffled labels did as well. If the report
  warns that shuffled labels do about as well, the commands cannot be told
  apart from this data. Change the strategies or reduce their number, and
  record again. No setting of the software makes up for it.
- **the table** shows which commands are confused with which. A command that
  is mostly decoded as rest is too weak; two that are decoded as each other are
  too similar.
- **driving** is what a rider would experience: held-out runs are replayed
  through the controller. A false start is the chair moving when nobody asked.
- **reliable commands** switched on in at least 70 % of their trials. Only
  those are bound to motions in the direct scheme.
- **scheme** is what this data supports: `direct` with two or more reliable
  commands, `switch:<command>` with one, `switch` with none.
- **stop reflex** shows how far the wearer's calibration clenches stood above
  rest. A margin of under 6 dB draws a warning: clench harder, or check the
  fit behind the ears.

The model, its thresholds and this report are written to `runs/s01/bundle`.

## 6. Drive

First with nothing connected, to see the decisions:

```bash
nwc drive --bundle runs/s01/bundle --link none
```

Then with the controller, wheels off the ground ([safety.md](safety.md)):

```bash
nwc drive --bundle runs/s01/bundle --link /dev/cu.usbmodem1101
```

1. The wearer sits relaxed with eyes open until the baseline reaches 100 %.
2. The operator presses `a` to arm.
3. In the **direct** scheme a command drives while it is held and the chair
   stops when it is let go. Neither is instant: the median latency in the
   training report says how long a command takes to switch on, and letting go
   takes about as long. A jaw clench stops within 0.4 s and latches.
4. In the **switch** scheme the options are offered in turn, two seconds
   each. A clench takes the one on offer; a clench while moving stops. Turns
   end by themselves.

| Key | |
| --- | --- |
| `a` | arm (also re-arms after an emergency stop) |
| space | disarm: stop and stay stopped |
| `b` | take the baseline again, after re-seating the headband |
| `q` | quit |

After an emergency stop the wearer can re-arm alone: let go, hold a clench for
two seconds, let go.

To drive with no calibration at all, leave out the bundle:

```bash
nwc drive --link /dev/cu.usbmodem1101
```

## 7. The next day

The headband never sits exactly as it did. That is what the baseline at the
start of every drive is for, and often it is enough. To find out:

```bash
nwc record --subject s01          # a new calibration on the new day
nwc evaluate --bundle runs/s01/bundle --data data/s01/<today's session>
```

`evaluate` applies yesterday's model unchanged to today's session and replays
it through the controller. If the commands still switch on, keep driving with
it. If not, train on both days together, which usually makes a model that
carries over better:

```bash
nwc train --data data/s01 --out runs/s01-two-days
```

## Recordings made with other tools

```bash
nwc import markers run1.csv run2.csv --markers 1=left,2=right --cue-s 3 --subject s01 --session old
nwc import labels brain_data*.csv --names 0=rest,1=forward,3=left,4=right --subject s01 --session 2024
```

`markers` reads muselsl and EEG-ExPy recordings with a marker column.
`labels` reads files with one label per sample in the last column. Training
data with one file per command can be imported this way, and `nwc train` will
report on it, with the warning that command and recording cannot be told apart
in such data.

## If something is off

| What you see | What to do |
| --- | --- |
| `no samples for 5 s` | The stream app is not sending, or sends to another port or address. |
| A channel `LOST` | Wet and re-seat that electrode; behind the ears, move hair away. |
| Few frames usable | Jaw, forehead or neck tension, or a lot of movement. Relax the face; rest the head. |
| `no answer from the motor controller` | Wrong port, the sketch is not on the board, or the cable. `nwc link --port ...` tells which. |
| The chair stops often with `emergency stop` | Talking and chewing can look like a clench. Raise `engine.clench.on_db`, or record calibration clenches that are firmer. |
| Commands switch on late or not at all | Read the report again: are they reliable? Record more runs, or use the switch scheme. |
| Commands worked yesterday, not today | Press `b` for a fresh baseline once the band has settled; then see "The next day". |

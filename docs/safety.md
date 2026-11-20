# Safety

A wheelchair that moves on its own can hurt the person in it and people
around it. This page says what the system does to stop, how fast, what it
cannot do, and what to check on your own hardware before anyone sits in the
chair.

This is a research prototype. It is not a medical device and has not been
assessed as one.

## What stops the chair

Stopping never rests on one mechanism. From fastest to slowest:

| What happens | What stops the chair | How fast |
| --- | --- | --- |
| The hardware switch is opened (if fitted) | the controller cuts the outputs and latches | at once |
| The host program crashes, the laptop sleeps, the USB cable comes out | the controller's watchdog | 400 ms, then a 200 ms ramp down |
| The wearer clenches the jaw | the stop reflex; in the direct scheme it latches | within 0.4 s |
| The operator presses space | the supervisor disarms | at once |
| No data from the headset | the supervisor | 1 s |
| An electrode loses contact | the supervisor | about 3 s |
| The wearer lets go of a command | the arbiter | a few seconds |
| A command is held without pause | the limit on uninterrupted motion | 10 s (8 s for a drive in the switch scheme) |

The last two rows are the slow ones, and they are slow by nature: a state of
mind is read from two seconds of signal, and evidence has to build up and
drain. The first six are exercised by the closed-loop and firmware tests. **The jaw clench, the operator and the hardware switch are the ways to
stop quickly.** Everyone involved should know that before the first drive.

## What makes the chair move

Only this chain, with every link required:

1. The baseline of this session is complete.
2. The operator has armed the system.
3. Data is arriving, every electrode has contact, the controller is answering.
4. The scheme asks for a motion: a command whose evidence has stayed above its
   threshold for the dwell time, or a press of the switch while an option is on
   offer.
5. The controller has received a valid command within the last 400 ms.

Frames without usable signal cannot start a command. After any fault or
emergency stop, what was in progress does not resume by itself.

## What it cannot do

- **It does not see.** There is no obstacle detection. The person next to the
  chair is the collision avoidance.
- **Commands can be wrong.** The report of `nwc train` states how often a
  wrong command switched on and how often the chair started unasked, on
  held-out data. Those rates are low by design of the thresholds, and they are
  not zero.
- **Mental commands are slow to stop.** See above.
- **The clench has to be checked for each wearer.** On real recordings the
  default threshold was tested only for how rarely it fires by itself. That a
  wearer's clench crosses it is confirmed in the dry run below, and a
  calibration then fits the threshold to them and reports the margin.
- **Talking, chewing and swallowing** can look like a clench. In the direct
  scheme that costs a stop. In the switch scheme a press while standing takes
  the option on offer, so the wearer should not talk while options are offered.
- **A model can go stale** as the headband shifts or dries. The baseline at
  the start of each drive covers what it can; `b` takes a new one.

## Before a person sits in the chair

Do these on your own hardware, in this order.

1. **Controller alone.** `nwc link --port <port>` passes, including the
   watchdog. With `--keys`, wheels off the ground: directions are right, the
   wheels ramp, and pulling the USB cable stops them.
2. **Emergency-stop switch,** if fitted: opening it stops the wheels at once;
   closing it does not restart them.
3. **Dry run.** `nwc drive --link none` with the headset on: the baseline
   completes, arming works, a clench shows as `CLENCH` and stops.
4. **Wheels off the ground, headset on, controller connected.** Drive every
   command. Clench: the wheels stop. Take the headband off while driving: the
   wheels stop. Quit the program while driving: the wheels stop.
5. **On the ground, nobody in the chair,** in an open space, at reduced speeds
   (`engine.speeds`).
6. **With a rider,** only with a second person in reach of the motor power
   switch, at low speed, away from stairs, roads and water.

Keep the speeds low. The defaults are 60 % of full scale forwards and 35 % in
turns; what that means in metres per second depends on your motors.

## If something goes wrong

Cut the motor power. Then find out why from the status line, which names the
state and the reason, before trying again.

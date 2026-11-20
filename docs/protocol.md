# Serial protocol

Between the host and the motor controller, over USB serial at 115200 baud.
The reference implementations are `firmware/wheelchair_controller/protocol.h`
and `src/nwc/link/protocol.py`; a test feeds both the same bytes.

## Framing

Every message is one line of ASCII:

```
$<payload>*<HH>\n
```

`<payload>` is a comma-separated list whose first field is the message type.
`<HH>` is the CRC-8 of the payload (polynomial 0x07, initial value 0, no
reflection) as two hexadecimal digits. Carriage returns are ignored.

A line is dropped, and counted, if it is longer than 48 characters, does not
have this shape, fails its checksum, or has a field that is not a plain
integer in range. Fields are an optional minus sign and one to five digits;
spaces, plus signs and decimals are refused.

Text was chosen over a binary encoding because it can be read and typed in a
serial monitor, and at ten short lines a second there is nothing to save.

## Host to controller

| Message | Meaning |
| --- | --- |
| `D,<seq>,<left>,<right>` | Drive. Wheel speeds from -255 to 255, positive forwards. |
| `S,<seq>` | Stop: both targets to zero. |
| `R,<seq>` | Release a latched emergency stop. Refused while the input is still active. |
| `Q,<seq>` | Ask for the information message. |

`<seq>` runs from 0 to 65535 and wraps. The controller echoes the last one it
accepted, which shows the host that its commands arrive.

The host repeats `D` ten times a second, also when the speeds are zero. The
repetition is the heartbeat.

## Controller to host

| Message | Meaning |
| --- | --- |
| `T,<seq>,<state>,<left>,<right>,<age_ms>,<rejected>` | Telemetry, every 100 ms. |
| `I,<protocol>,<firmware>,<watchdog_ms>,<ramp_up_ms>,<ramp_down_ms>` | Information, at start-up and on `Q`. |

- `<state>` is `I` idle, `D` driving, `W` watchdog (no valid drive command for
  `watchdog_ms`), or `E` emergency stop latched.
- `<left>`, `<right>` are the outputs as they are now, after ramping.
- `<age_ms>` is the time since the last valid drive command, capped at 9999.
- `<rejected>` counts lines the controller dropped.

Example: the host asks for 60 % forwards four times and then goes silent.
The outputs ramp up, hold, and the watchdog brings them down.

```
    0 ms   controller   $I,1,2,400,600,200*93
   50 ms   host         $D,1,153,153*D7
  100 ms   controller   $T,1,D,21,21,50,0*14
  150 ms   host         $D,2,153,153*5C
  200 ms   controller   $T,2,D,63,63,50,0*4E
  250 ms   host         $D,3,153,153*25
  300 ms   controller   $T,3,D,105,105,50,0*F2
  350 ms   host         $D,4,153,153*4D
  400 ms   controller   $T,4,D,148,148,50,0*B1
  500 ms   controller   $T,4,D,153,153,150,0*A7
  600 ms   controller   $T,4,D,153,153,250,0*01
  700 ms   controller   $T,4,D,153,153,350,0*63
  800 ms   controller   $T,4,W,89,89,450,0*53
  900 ms   controller   $T,4,W,0,0,550,0*EA
 1000 ms   controller   $T,4,W,0,0,650,0*4C
 1100 ms   controller   $T,4,W,0,0,750,0*2E
```

## What the controller does with it

- **Ramps.** Outputs move towards their targets, from standstill to full
  speed in `ramp_up_ms` (600) and back in `ramp_down_ms` (200). A change of
  direction first comes to a stop.
- **Watchdog.** If no valid `D` arrives for `watchdog_ms` (400), the targets
  go to zero and the state to `W`. The next valid `D` resumes.
- **Emergency stop.** While the input is active the outputs are zero, without
  a ramp, and the state is `E`. It stays `E` after the input is released until
  an `R` arrives.

The host treats a controller as gone when no telemetry has arrived for half a
second, and the supervisor then stops asking for motion.

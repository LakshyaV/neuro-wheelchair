# Hardware

## Parts

- Muse headband (2016, Muse 2 or Muse S) and a computer with Bluetooth
- Arduino Uno (or another ATmega328P board) and a USB cable
- L298N dual H-bridge motor driver
- Two DC motors and a supply suited to them
- Optional and recommended: a normally-closed push switch for the emergency-stop input

## Wiring

The wiring of the original build is unchanged.

| L298N | Arduino | |
| --- | --- | --- |
| IN1 | 2 | left motor |
| IN2 | 3 | left motor (PWM) |
| IN3 | 4 | right motor |
| IN4 | 5 | right motor (PWM) |
| ENA, ENB | jumpers left on | |
| GND | GND | common ground with the motor supply |

The motor supply goes to the L298N, not through the Arduino. If a motor turns
the wrong way, either swap its two wires or set `NWC_INVERT_LEFT` /
`NWC_INVERT_RIGHT`.

### Speed control

Two ways, chosen at compile time in `firmware/wheelchair_controller/config.h`:

- **On the input pins (default, no rewiring).** With the jumpers on ENA and
  ENB, the second input of each motor is pulse-width modulated, so the bridge
  alternates between driving and braking. This needs IN2 and IN4 on PWM pins,
  which 3 and 5 are.
- **On the enable pins** (`NWC_PWM_ON_ENABLE=1`). Remove the jumpers and wire
  ENA to pin 9 and ENB to pin 10. The inputs then only set the direction.

Small motors do not turn below some duty. Set `NWC_MIN_DUTY` (0 to 255) to the
value at which yours start, and non-zero speeds are mapped onto the range
above it.

### Emergency-stop input

Wire a normally-closed switch between a free pin and GND and build with
`NWC_ESTOP_PIN=<pin>`. While the switch is closed the pin reads low. Opening
it, or a wire coming off, stops the motors at once and latches: they stay
stopped until the switch is closed again and the host sends a release (arming
in `nwc drive` does). Mount the switch where the person next to the chair can
reach it.

## Firmware

```bash
arduino-cli core install arduino:avr
arduino-cli compile --fqbn arduino:avr:uno --upload -p /dev/cu.usbmodem1101 firmware/wheelchair_controller
```

With options:

```bash
arduino-cli compile --fqbn arduino:avr:uno --upload -p /dev/cu.usbmodem1101 \
  --build-property "compiler.cpp.extra_flags=-DNWC_ESTOP_PIN=7 -DNWC_MIN_DUTY=60" \
  firmware/wheelchair_controller
```

The Arduino IDE works as well: open `wheelchair_controller.ino` and edit
`config.h`.

| Setting | Default | |
| --- | --- | --- |
| `NWC_LEFT_IN1`, `NWC_LEFT_IN2`, `NWC_RIGHT_IN1`, `NWC_RIGHT_IN2` | 2, 3, 4, 5 | L298N inputs |
| `NWC_PWM_ON_ENABLE` | 0 | 1: speed on ENA/ENB (`NWC_LEFT_EN` 9, `NWC_RIGHT_EN` 10) |
| `NWC_INVERT_LEFT`, `NWC_INVERT_RIGHT` | 0 | reverse a motor's sense |
| `NWC_MIN_DUTY` | 0 | duty below which the motors do not turn |
| `NWC_ESTOP_PIN` | -1 (none) | emergency-stop input |
| `NWC_WATCHDOG_MS` | 400 | stop after this long without a valid command |
| `NWC_RAMP_UP_MS`, `NWC_RAMP_DOWN_MS` | 600, 200 | standstill to full speed, and back |
| `NWC_BAUD` | 115200 | |

What the controller does with commands is described in
[protocol.md](protocol.md).

## Bench tests

With the motor supply off, or the wheels off the ground.

```bash
nwc link --port /dev/cu.usbmodem1101
```

```text
controller: protocol 1, firmware 2, watchdog 400 ms, ramps 600/200 ms
telemetry: idle, wheels +0/+0, 0 lines rejected, 0 damaged on the way back
watchdog: reported after the commands stopped
after resuming: idle, wheels +0/+0
```

This moves nothing. It confirms that the right sketch is on the board, that
commands arrive intact, and that the controller reports its watchdog when the
commands stop.

```bash
nwc link --port /dev/cu.usbmodem1101 --keys
```

`w` forward, `x` backward, `a` left, `d` right, space stop, `q` quit. A motion
lasts one and a half seconds unless the key is pressed again. Check that:

- both wheels turn forwards for `w` (otherwise invert one);
- `a` turns the chair to the left: left wheel backwards, right wheel forwards;
- the wheels start gently and stop quickly;
- pulling the USB cable while they turn stops them within about half a second;
- with an emergency-stop switch fitted, opening it stops them at once, and
  closing it again does not restart them.

Speeds for each motion, as fractions of full scale for the left and right
wheel, are in the `engine.speeds` section of the settings.

## Serial ports

- macOS: `/dev/cu.usbmodem…` or `/dev/cu.usbserial…`
- Linux: `/dev/ttyACM0` or `/dev/ttyUSB0`
- Windows: `COM3` and the like

An Arduino resets when its port is opened and takes about two seconds to
start. `nwc link` waits for it, and `nwc drive` shows `link: no telemetry`
until the controller answers; the chair cannot be driven before that.

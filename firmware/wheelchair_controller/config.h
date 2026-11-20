// Build-time settings of the motor controller. Change them here or pass
// -D<NAME>=<value> to the compiler.
#pragma once

#define NWC_FIRMWARE_VERSION 2

// ---------------------------------------------------------------- L298N wiring
// The defaults are the wiring of the original build: IN1..IN4 on pins 2..5,
// with the jumpers left on ENA and ENB.
#ifndef NWC_LEFT_IN1
#define NWC_LEFT_IN1 2
#endif
#ifndef NWC_LEFT_IN2
#define NWC_LEFT_IN2 3  // must be a PWM pin in the default speed mode
#endif
#ifndef NWC_RIGHT_IN1
#define NWC_RIGHT_IN1 4
#endif
#ifndef NWC_RIGHT_IN2
#define NWC_RIGHT_IN2 5  // must be a PWM pin in the default speed mode
#endif

// Speed control.
//   0  on the input pins (default). Works with the jumpers on ENA/ENB: the
//      second input of each motor is pulse-width modulated, so the bridge
//      alternates between driving and braking.
//   1  on the enable pins. Remove the jumpers and wire ENA/ENB to the two PWM
//      pins below; the inputs then only set the direction.
#ifndef NWC_PWM_ON_ENABLE
#define NWC_PWM_ON_ENABLE 0
#endif
#ifndef NWC_LEFT_EN
#define NWC_LEFT_EN 9
#endif
#ifndef NWC_RIGHT_EN
#define NWC_RIGHT_EN 10
#endif

// Swap the sense of a motor that turns the wrong way instead of rewiring it.
#ifndef NWC_INVERT_LEFT
#define NWC_INVERT_LEFT 0
#endif
#ifndef NWC_INVERT_RIGHT
#define NWC_INVERT_RIGHT 0
#endif

// Duty (0..255) below which the motors do not turn. Non-zero speeds are mapped
// onto [NWC_MIN_DUTY, 255].
#ifndef NWC_MIN_DUTY
#define NWC_MIN_DUTY 0
#endif

// ---------------------------------------------------------------- emergency stop
// A normally-closed switch between this pin and GND. Opening the switch, or a
// broken wire, stops the motors and latches until the host sends a release.
// -1 disables the input.
#ifndef NWC_ESTOP_PIN
#define NWC_ESTOP_PIN -1
#endif

// ---------------------------------------------------------------- timing
#ifndef NWC_BAUD
#define NWC_BAUD 115200
#endif
#ifndef NWC_WATCHDOG_MS
#define NWC_WATCHDOG_MS 400  // no valid drive command for this long: stop
#endif
#ifndef NWC_RAMP_UP_MS
#define NWC_RAMP_UP_MS 600  // standstill to full speed
#endif
#ifndef NWC_RAMP_DOWN_MS
#define NWC_RAMP_DOWN_MS 200  // full speed to standstill
#endif
#ifndef NWC_TELEMETRY_MS
#define NWC_TELEMETRY_MS 100
#endif

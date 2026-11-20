// The board side of the motor controller: pins, the serial port and the loop.
//
// protocol.h and drive.h hold the logic and know nothing about Arduino. This
// file connects them to an L298N H-bridge and to USB serial. Nothing here
// blocks: bytes are parsed as they arrive and the loop runs every few hundred
// microseconds.
//
// Everything lives in headers and the sketch file only forwards setup() and
// loop(), so the build does not depend on how the Arduino tools rewrite a
// sketch. Include this file from the sketch and from nowhere else.
#pragma once

#include <Arduino.h>

#include "config.h"
#include "drive.h"
#include "protocol.h"

namespace nwc {
namespace controller {

static Drive drive(DriveConfig(NWC_WATCHDOG_MS, NWC_RAMP_UP_MS, NWC_RAMP_DOWN_MS));
static LineReader reader;
static uint16_t lastSeq = 0;
static uint32_t lastTelemetryMs = 0;

inline void send(const char* payload) {
  char line[MAX_LINE + 8];
  const size_t length = frame(payload, line);
  Serial.write(reinterpret_cast<const uint8_t*>(line), length);
}

inline void sendInfo() {
  char payload[MAX_LINE];
  // Every argument is cast: on this board an int has 16 bits, and what printf reads must match what is passed.
  snprintf(payload, sizeof(payload), "I,%u,%u,%u,%u,%u", static_cast<unsigned>(PROTOCOL_VERSION), static_cast<unsigned>(NWC_FIRMWARE_VERSION),
           static_cast<unsigned>(NWC_WATCHDOG_MS), static_cast<unsigned>(NWC_RAMP_UP_MS), static_cast<unsigned>(NWC_RAMP_DOWN_MS));
  send(payload);
}

inline void sendTelemetry(uint32_t nowMs) {
  char payload[MAX_LINE];
  snprintf(payload, sizeof(payload), "T,%u,%c,%d,%d,%u,%u", static_cast<unsigned>(lastSeq), static_cast<char>(drive.state()), static_cast<int>(drive.left()),
           static_cast<int>(drive.right()), static_cast<unsigned>(drive.commandAge(nowMs)), static_cast<unsigned>(reader.rejected()));
  send(payload);
}

inline uint8_t duty(int16_t speed) {
  const int16_t magnitude = speed < 0 ? static_cast<int16_t>(-speed) : speed;
  if (magnitude == 0) return 0;
  return static_cast<uint8_t>(NWC_MIN_DUTY + (static_cast<int32_t>(255 - NWC_MIN_DUTY) * magnitude) / 255);
}

#if NWC_PWM_ON_ENABLE
// Direction on the inputs, speed on the enable pin.
inline void applyMotor(uint8_t in1, uint8_t in2, uint8_t enable, int16_t speed) {
  digitalWrite(in1, speed > 0 ? HIGH : LOW);
  digitalWrite(in2, speed < 0 ? HIGH : LOW);
  analogWrite(enable, duty(speed));
}
#else
// Speed on the second input: the bridge alternates between driving and braking.
// Forward keeps the first input high and pulls the second low for the drive
// part of each period; reverse keeps the first low and raises the second.
inline void applyMotor(uint8_t in1, uint8_t in2, uint8_t /*enable*/, int16_t speed) {
  if (speed > 0) {
    digitalWrite(in1, HIGH);
    analogWrite(in2, 255 - duty(speed));
  } else if (speed < 0) {
    digitalWrite(in1, LOW);
    analogWrite(in2, duty(speed));
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
  }
}
#endif

inline void applyOutputs() {
  const int16_t left = NWC_INVERT_LEFT ? static_cast<int16_t>(-drive.left()) : drive.left();
  const int16_t right = NWC_INVERT_RIGHT ? static_cast<int16_t>(-drive.right()) : drive.right();
  applyMotor(NWC_LEFT_IN1, NWC_LEFT_IN2, NWC_LEFT_EN, left);
  applyMotor(NWC_RIGHT_IN1, NWC_RIGHT_IN2, NWC_RIGHT_EN, right);
}

inline void handle(const Command& command, uint32_t nowMs) {
  lastSeq = command.seq;
  switch (command.type) {
    case 'D':
      drive.command(command.left, command.right, nowMs);
      break;
    case 'S':
      drive.stop();
      break;
    case 'R':
      drive.release();
      break;
    case 'Q':
      sendInfo();
      break;
  }
}

inline void setup() {
  pinMode(NWC_LEFT_IN1, OUTPUT);
  pinMode(NWC_LEFT_IN2, OUTPUT);
  pinMode(NWC_RIGHT_IN1, OUTPUT);
  pinMode(NWC_RIGHT_IN2, OUTPUT);
#if NWC_PWM_ON_ENABLE
  pinMode(NWC_LEFT_EN, OUTPUT);
  pinMode(NWC_RIGHT_EN, OUTPUT);
#endif
#if NWC_ESTOP_PIN >= 0
  pinMode(NWC_ESTOP_PIN, INPUT_PULLUP);
#endif
  applyOutputs();  // everything low before the first command
  Serial.begin(NWC_BAUD);
  sendInfo();
}

inline void loop() {
  const uint32_t nowMs = millis();
#if NWC_ESTOP_PIN >= 0
  // First of all: the switch. Open (or its wire broken) means stop, whatever the host says in this pass.
  drive.emergencyInput(digitalRead(NWC_ESTOP_PIN) == HIGH);
#endif
  while (Serial.available() > 0) {
    Command command;
    if (reader.feed(static_cast<char>(Serial.read()), command)) handle(command, nowMs);
  }
  drive.update(nowMs);
  applyOutputs();
  if (nowMs - lastTelemetryMs >= NWC_TELEMETRY_MS) {
    lastTelemetryMs = nowMs;
    sendTelemetry(nowMs);
  }
}

}  // namespace controller
}  // namespace nwc

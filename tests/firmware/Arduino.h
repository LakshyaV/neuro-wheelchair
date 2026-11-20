// A minimal stand-in for <Arduino.h>, enough to compile and run the sketch
// on the host: a settable clock, recorded pin writes and a serial port backed by
// two byte queues. Included by one translation unit only (harness.cpp), and kept
// to C++11 because that is what the board's toolchain compiles.
#pragma once

#include <stdint.h>
#include <stdio.h>

#include <deque>
#include <string>

#define HIGH 1
#define LOW 0
#define OUTPUT 1
#define INPUT_PULLUP 2

namespace mock {
static uint32_t now_ms = 0;
static int pin_mode[32] = {0};
static int pin_value[32] = {0};   // last digitalWrite level, or -1 after analogWrite
static int pin_pwm[32] = {0};     // last analogWrite duty, or -1 after digitalWrite
static int pin_input[32] = {0};   // level seen by digitalRead
static std::deque<uint8_t> rx;
static std::string tx;
}  // namespace mock

inline uint32_t millis() { return mock::now_ms; }
inline void pinMode(uint8_t pin, uint8_t mode) { mock::pin_mode[pin] = mode; }
inline void digitalWrite(uint8_t pin, uint8_t level) {
  mock::pin_value[pin] = level;
  mock::pin_pwm[pin] = -1;
}
inline void analogWrite(uint8_t pin, int duty) {
  mock::pin_pwm[pin] = duty;
  mock::pin_value[pin] = -1;
}
inline int digitalRead(uint8_t pin) { return mock::pin_input[pin]; }

struct MockSerial {
  void begin(unsigned long) {}
  int available() { return static_cast<int>(mock::rx.size()); }
  int read() {
    if (mock::rx.empty()) return -1;
    const uint8_t byte = mock::rx.front();
    mock::rx.pop_front();
    return byte;
  }
  size_t write(const uint8_t* data, size_t length) {
    mock::tx.append(reinterpret_cast<const char*>(data), length);
    return length;
  }
};
static MockSerial Serial;

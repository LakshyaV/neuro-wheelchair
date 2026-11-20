// Runs the unmodified sketch on the host against the mock Arduino API.
//
// A script on stdin drives it, one instruction per line:
//
//   at <ms> [text]     one pass of loop() at that time; `text` (if any) plus a
//                      newline has arrived on the serial port just before
//   raw <ms> <hex>     the same with arbitrary bytes
//   run <ms>           one pass of loop() per millisecond up to that time
//   input <pin> <0|1>  level on a digital input from now on
//
// After every pass of loop() that changed something, one line is printed:
//
//   <ms> <left effective> <right effective> | <serial output, newlines shown as ;>
//
// where "effective" is the signed duty the bridge sees, reconstructed from the
// pin states the same way the hardware would respond to them.
// Compiled with -I tests/firmware, where Arduino.h is the mock.
#include <Arduino.h>

#include "../../firmware/wheelchair_controller/wheelchair_controller.ino"

#include <iostream>
#include <sstream>

namespace {

// Signed duty of one motor from its pin states, for both wiring modes.
int effective(int in1, int in2, int enable) {
#if NWC_PWM_ON_ENABLE
  const int direction = mock::pin_value[in1] - mock::pin_value[in2];  // +1, 0 or -1
  return direction * (mock::pin_pwm[enable] < 0 ? 0 : mock::pin_pwm[enable]);
#else
  (void)enable;
  const int first = mock::pin_value[in1];
  const int second = mock::pin_pwm[in2] >= 0 ? mock::pin_pwm[in2] : mock::pin_value[in2] * 255;
  return first == HIGH ? 255 - second : -second;  // first high: drives while the second is low
#endif
}

int lastLeft = 0, lastRight = 0;

void pass(uint32_t ms) {
  mock::now_ms = ms;
  loop();
  const int left = effective(NWC_LEFT_IN1, NWC_LEFT_IN2, NWC_LEFT_EN);
  const int right = effective(NWC_RIGHT_IN1, NWC_RIGHT_IN2, NWC_RIGHT_EN);
  if (left != lastLeft || right != lastRight || !mock::tx.empty()) {
    std::string shown = mock::tx;
    for (char& c : shown) {
      if (c == '\n') c = ';';
    }
    printf("%u %d %d | %s\n", static_cast<unsigned>(ms), left, right, shown.c_str());
    mock::tx.clear();
    lastLeft = left;
    lastRight = right;
  }
}

}  // namespace

int main() {
  setup();
  {
    std::string shown = mock::tx;
    for (char& c : shown) {
      if (c == '\n') c = ';';
    }
    printf("boot | %s\n", shown.c_str());
    mock::tx.clear();
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string op;
    in >> op;
    if (op == "at") {
      uint32_t ms;
      in >> ms;
      std::string text;
      std::getline(in, text);
      if (!text.empty() && text[0] == ' ') text.erase(0, 1);
      if (!text.empty()) {
        for (char c : text) mock::rx.push_back(static_cast<uint8_t>(c));
        mock::rx.push_back('\n');
      }
      pass(ms);
    } else if (op == "raw") {
      uint32_t ms;
      std::string hex;
      in >> ms >> hex;
      for (size_t i = 0; i + 1 < hex.size(); i += 2) mock::rx.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
      pass(ms);
    } else if (op == "run") {
      uint32_t until;
      in >> until;
      for (uint32_t ms = mock::now_ms + 1; ms <= until; ms++) pass(ms);
    } else if (op == "input") {
      int pin, level;
      in >> pin >> level;
      mock::pin_input[pin] = level;
    }
  }
  return 0;
}
